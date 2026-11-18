"""FastAPI backend for the Tamil Nadu Agriculture Schemes RAG.

Pipeline 1 (ingest):  /api/ingest/*  → background job → load → split → embed → FAISS
Pipeline 2 (query):   /api/search, /api/ask → query plan → hybrid retrieval → rerank → cited LLM answer
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .ingest.crawler import discover, download_many, downloaded_urls, read_manifest
from .ingest.loaders import SUPPORTED_EXT
from .ingest.pipeline import ingest_files
from .jobs import Job, get_jobs
from . import telemetry
from .llm import openai_client
from .llm.generator import provider_status, stream_answer, suggest_followups
from .llm.usage import Usage
from .retrieval.hybrid import SearchRequest, search
from .retrieval.query_planner import plan_query
from .retrieval.reranker import CrossEncoderReranker, get_reranker, reranker_status, reranker_usable
from .store.embeddings import get_embedder
from .store.vector_store import ModelMismatchError, get_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tn-agri-rag")

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    st = get_store().stats()
    get_jobs()
    print(f"[uzhavan] FAISS index (local): {Path(st['store_dir']) / 'index.faiss'} — "
          f"{st['documents']} documents, {st['vectors']} vectors", flush=True)
    if settings.llm_enabled:
        print(f"[uzhavan] LLM: {settings.answer_model()} (helper: {settings.fast_model()}, fallback: "
              f"{settings.openai_fallback_model or 'off'}) · embeddings: {settings.embedding_label()}", flush=True)
    if settings.warmup_models:
        def warm():
            try:
                if settings.embedding_backend == "sentence-transformers":
                    get_embedder().embed_query("warm up")
                if isinstance(get_reranker(), CrossEncoderReranker) and reranker_usable():
                    get_reranker().score("warm up", ["warm up"])
                log.info("models warmed up")
            except Exception as e:
                log.warning("model warm-up failed: %s", e)
        threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="Uzhavan Assist — TN Agri Schemes RAG", version="2.0.0", lifespan=lifespan)


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Protects ingestion / delete / reset when ADMIN_TOKEN is set (recommended for public demos)."""
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(401, "admin token required for this action (X-Admin-Token header)")


ADMIN = [Depends(require_admin)]

_hits: dict[str, list[float]] = {}
_hits_lock = threading.Lock()


def rate_limit(request: Request) -> None:
    """Sliding one-minute window per client IP for the endpoints that spend OpenAI tokens."""
    limit = settings.rate_limit_per_min
    if not limit:
        return
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "unknown"))
    now = time.time()
    with _hits_lock:
        recent = [t for t in _hits.get(ip, []) if now - t < 60]
        if len(recent) >= limit:
            raise HTTPException(429, f"Too many requests — limit is {limit} per minute. Please wait a moment.")
        recent.append(now)
        _hits[ip] = recent
        if len(_hits) > 10000:
            _hits.clear()


LIMITED = [Depends(rate_limit)]
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
                   allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- models
class SearchBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    rerank: bool = True
    candidate_k: int = Field(default=settings.candidate_k, ge=5, le=500)
    rerank_candidates: int = Field(default=settings.rerank_candidates, ge=5, le=200)
    dense_weight: float = settings.dense_weight
    bm25_weight: float = settings.bm25_weight
    max_per_doc: int = Field(default=settings.max_chunks_per_doc, ge=1, le=50)
    rewrite: bool = True
    filters: dict | None = None


class AskBody(SearchBody):
    history: list[dict] = []


class UrlsBody(BaseModel):
    urls: list[str]
    category: str = "web"
    discover: bool = False  # treat URLs as seeds and crawl for documents
    max_files: int = 300


class ManifestBody(BaseModel):
    relevance: Literal["core", "all"] = "core"
    categories: list[str] = []
    ids: list[str] = []
    limit: int = 0
    include_discovered: bool = False  # also crawl seeds.txt for more documents
    max_discovered: int = 300


class FolderBody(BaseModel):
    path: str = "raw"
    category: str = "local"


class FeedbackBody(BaseModel):
    question: str
    answer: str = ""
    rating: int = Field(..., ge=-1, le=1)
    comment: str = ""
    sources: list[dict] = []


def _search_req(b: SearchBody, history: list[dict] | None = None) -> SearchRequest:
    return SearchRequest(query=b.query, mode=b.mode, top_k=b.top_k, rerank=b.rerank, rewrite=b.rewrite,
                         candidate_k=b.candidate_k, rerank_candidates=b.rerank_candidates,
                         dense_weight=b.dense_weight, bm25_weight=b.bm25_weight, max_per_doc=b.max_per_doc,
                         filters=b.filters, history=history)


# ----------------------------------------------------------------------------- health
@app.get("/api/health")
def health():
    emb = get_embedder()
    return {
        "status": "ok",
        "embedder": emb.status(),
        "reranker": reranker_status(),
        "reranker_usable": reranker_usable(),
        "usage": telemetry.usage_totals(),
        "llm": provider_status(),
        "models": _active_models(),
        "store": get_store().stats(),
        "active_job": (get_jobs().active().to_dict(include_logs=False) if get_jobs().active() else None),
        "settings": settings.public_dict(),
        "supported_extensions": sorted(SUPPORTED_EXT),
    }


@app.get("/api/stats")
def stats():
    return get_store().stats()


# ----------------------------------------------------------------------------- PIPELINE 1
_SAFE = re.compile(r"[^\w஀-௿.\- ()]+")


@app.post("/api/ingest/upload", dependencies=ADMIN)
async def ingest_upload(files: list[UploadFile] = File(...), category: str = Form("uploaded"),
                        chunk_size: int = Form(0), chunk_overlap: int = Form(0)):
    if not files:
        raise HTTPException(400, "no files")
    batch_dir = settings.uploads_dir / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths, meta, rejected = [], {}, []
    limit = settings.max_upload_mb * 1024 * 1024
    for uf in files:
        name = _SAFE.sub("_", Path(uf.filename or "file").name)[:180] or "file"
        dest = batch_dir / name
        i = 1
        while dest.exists():
            dest = batch_dir / f"{Path(name).stem}_{i}{Path(name).suffix}"
            i += 1
        size = 0
        with dest.open("wb") as out:
            while chunk := await uf.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    break
                out.write(chunk)
        if size > limit:
            dest.unlink(missing_ok=True)
            rejected.append({"name": name, "reason": f"larger than {settings.max_upload_mb} MB"})
            continue
        paths.append(dest)
        meta[str(dest)] = {"original_name": uf.filename, "title": Path(uf.filename or name).stem}
    if not paths:
        raise HTTPException(400, {"message": "all files rejected", "rejected": rejected})
    opts = {"category": category, "chunk_size": chunk_size or None, "chunk_overlap": chunk_overlap or None}

    def run(job: Job):
        job.log(f"received {len(paths)} uploaded file(s)")
        for r in rejected:
            job.log(f"rejected {r['name']}: {r['reason']}", "warn")
        ingest_files(job, paths, meta, opts)

    job = get_jobs().submit("upload", {"files": len(paths), "category": category}, run)
    return {"job_id": job.id, "accepted": len(paths), "rejected": rejected}


def _download_then_ingest(job: Job, items: list[dict], category: str) -> None:
    job.stage = "downloading"
    job.downloads_total = len(items)
    job.log(f"downloading {len(items)} source(s) …")

    def prog(res, i, n):
        if res.ok:
            job.downloads_done += 1
        else:
            job.downloads_failed += 1
            job.log(f"download failed: {res.url} ({res.error})", "warn")

    results = download_many(items, prog, should_stop=lambda: job.cancel_requested)
    paths = [Path(r.path) for r in results if r.ok and r.path]
    job.log(f"downloaded {len(paths)} / {len(items)} ({job.downloads_failed} failed)")
    if job.cancel_requested:
        return
    ingest_files(job, paths, {}, {"category": category})


@app.post("/api/ingest/urls", dependencies=ADMIN)
def ingest_urls(body: UrlsBody):
    urls = [u.strip() for u in body.urls if u.strip().startswith(("http://", "https://"))]
    if not urls:
        raise HTTPException(400, "no valid http(s) URLs")

    def run(job: Job):
        items = [{"url": u, "category": body.category} for u in urls]
        if body.discover:
            job.stage = "downloading"
            found = discover(urls, max_files=body.max_files, on_log=job.log)
            job.log(f"discovered {len(found)} document link(s)")
            items += [{**f, "category": body.category} for f in found]
        _download_then_ingest(job, items, body.category)

    job = get_jobs().submit("urls", {"urls": len(urls), "discover": body.discover}, run)
    return {"job_id": job.id}


@app.post("/api/ingest/manifest", dependencies=ADMIN)
def ingest_manifest(body: ManifestBody):
    rows = read_manifest()
    if body.relevance == "core":
        rows = [r for r in rows if r.get("relevance") == "core"]
    if body.categories:
        rows = [r for r in rows if r.get("category") in body.categories]
    if body.ids:
        wanted = set(body.ids)
        rows = [r for r in rows if r.get("id") in wanted]
    indexed = get_store().known_source_urls()
    rows = [r for r in rows if r["url"] not in indexed]
    if body.limit:
        rows = rows[: body.limit]
    if not rows and not body.include_discovered:
        raise HTTPException(400, "nothing to fetch — every selected source is already indexed")

    def run(job: Job):
        items = [{"url": r["url"], "title": r.get("title") or None, "category": r.get("category"),
                  "source_id": r.get("id"), "language": r.get("language")} for r in rows]
        if body.include_discovered:
            seeds_file = settings.sources_dir / "seeds.txt"
            seeds = [l.strip() for l in seeds_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
            job.stage = "downloading"
            found = discover(seeds, max_files=body.max_discovered, on_log=job.log)
            known = {i["url"] for i in items} | indexed
            extra = [f for f in found if f["url"] not in known]
            job.log(f"crawler discovered {len(extra)} additional document(s)")
            items += extra
        _download_then_ingest(job, items, "manifest")

    job = get_jobs().submit("manifest", body.model_dump(), run)
    return {"job_id": job.id, "selected": len(rows)}


@app.post("/api/ingest/folder", dependencies=ADMIN)
def ingest_folder(body: FolderBody):
    base = settings.data_dir.resolve()
    target = (base / body.path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(400, "folder must be inside the backend data directory")
    if not target.is_dir():
        raise HTTPException(404, f"folder not found: {target}")
    files = sorted(p for p in target.rglob("*") if p.is_file() and not p.name.startswith(("_", "."))
                   and p.suffix.lower() in SUPPORTED_EXT)
    if not files:
        raise HTTPException(400, "no supported files in folder")

    def run(job: Job):
        ingest_files(job, files, {}, {"category": body.category})

    job = get_jobs().submit("folder", {"path": str(target), "files": len(files)}, run)
    return {"job_id": job.id, "files": len(files)}


@app.get("/api/jobs")
def list_jobs():
    return get_jobs().list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = get_jobs().jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j.to_dict()


@app.post("/api/jobs/{job_id}/cancel", dependencies=ADMIN)
def cancel_job(job_id: str):
    if not get_jobs().cancel(job_id):
        raise HTTPException(400, "job not running")
    return {"ok": True}


# ----------------------------------------------------------------------------- library
@app.get("/api/documents")
def documents(q: str = "", file_type: str = "", category: str = "", language: str = "", limit: int = 50,
              offset: int = 0, sort: str = "added_at"):
    return get_store().list_documents(q, file_type, category, language, min(limit, 500), offset, sort)


@app.get("/api/documents/{doc_id}")
def document(doc_id: str, limit: int = 20, offset: int = 0, q: str = ""):
    s = get_store()
    d = s.get_document(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    return {"document": d, "chunks": s.document_chunks(doc_id, min(limit, 200), offset, q)}


@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: str):
    p = get_store().document_path(doc_id)
    if not p or not p.exists():
        raise HTTPException(404, "original file not available")
    return FileResponse(p, filename=p.name.split("__")[0] + p.suffix if "__" in p.name else p.name)


@app.delete("/api/documents/{doc_id}", dependencies=ADMIN)
def delete_document(doc_id: str):
    if get_jobs().active():
        raise HTTPException(409, "an ingestion job is running — try again when it finishes")
    if not get_store().delete_document(doc_id):
        raise HTTPException(404, "document not found")
    return {"ok": True}


@app.post("/api/index/reset", dependencies=ADMIN)
def reset_index(confirm: str = ""):
    if confirm != "RESET":
        raise HTTPException(400, "pass ?confirm=RESET")
    if get_jobs().active():
        raise HTTPException(409, "an ingestion job is running")
    get_store().reset()
    return {"ok": True}


# ----------------------------------------------------------------------------- PIPELINE 2
@app.post("/api/search", dependencies=LIMITED)
def api_search(body: SearchBody):
    usage = Usage()
    try:
        res = search(_search_req(body), usage)
    except ModelMismatchError as e:
        raise HTTPException(409, str(e))
    telemetry.record_usage("search", usage.steps)
    return res


@app.post("/api/search/compare", dependencies=LIMITED)
def api_compare(body: SearchBody):
    """Same query through BM25, dense, hybrid and hybrid+rerank side by side (one shared query plan)."""
    usage = Usage()
    out = {}
    try:
        plan = plan_query(body.query, None, usage) if body.rewrite else None
        for key, mode, rr in (("bm25", "bm25", False), ("dense", "dense", False), ("hybrid", "hybrid", False),
                              ("hybrid_rerank", "hybrid", True)):
            b = body.model_copy(update={"mode": mode, "rerank": rr})
            out[key] = search(_search_req(b), usage, plan=plan)
    except ModelMismatchError as e:
        raise HTTPException(409, str(e))
    telemetry.record_usage("search", usage.steps)
    out["usage"] = usage.summary()
    return out


NO_HITS = {
    "ta": "மன்னிக்கவும், பதிவேற்றப்பட்ட ஆவணங்களில் இதற்கான தகவல் கிடைக்கவில்லை. அருகிலுள்ள வேளாண்மை விரிவாக்க மையம் அல்லது உதவி வேளாண்மை அலுவலரை அணுகவும்.",
    "en": "I couldn't find this in the indexed documents. Try rephrasing, or check with your nearest Agricultural Extension Centre / Assistant Agricultural Officer.",
}


@app.post("/api/ask", dependencies=LIMITED)
def ask(body: AskBody):
    """NDJSON stream: plan → sources → token… → followups → done (timings, trace, tokens, cost)."""
    def emit(obj) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def gen():
        t0 = time.perf_counter()
        usage = Usage()
        try:
            res = search(_search_req(body, body.history), usage)
        except ModelMismatchError as e:
            yield emit({"type": "error", "message": str(e)})
            return
        except Exception as e:
            log.exception("retrieval failed")
            yield emit({"type": "error", "message": f"Retrieval failed: {e}"})
            return
        plan = res.get("plan") or {}
        yield emit({"type": "plan", "plan": plan, "variants": res.get("variants", [])})
        yield emit({"type": "sources", "results": res["results"], "reranked": res["reranked"],
                    "timings": res["timings"], "mode": res["mode"], "trace": res["trace"],
                    "candidates": res["candidates"]})
        lang = "ta" if plan.get("language") == "ta" or any("\u0b80" <= ch <= "\u0bff" for ch in body.query) else "en"
        trace = list(res["trace"])
        answer = ""
        if not res["results"]:
            answer = NO_HITS[lang]
            yield emit({"type": "token", "text": answer})
        elif not settings.llm_enabled:
            yield emit({"type": "notice", "text": "Retrieval-only mode (no LLM configured): showing the best "
                                                  "matching passages below."})
        else:
            t1 = time.perf_counter()
            first = None
            question = plan.get("standalone_question") or body.query
            try:
                for piece in stream_answer(question, res["results"], body.history, plan, usage):
                    if first is None:
                        first = time.perf_counter()
                    answer += piece
                    yield emit({"type": "token", "text": piece})
            except Exception as e:
                log.warning("LLM error: %s", e)
                yield emit({"type": "error", "message": f"LLM error ({settings.llm_provider}): {e}. "
                                                        "The retrieved passages are still shown."})
            trace.append({"step": "answer", "model": _answer_model(),
                          "ms": round((time.perf_counter() - t1) * 1000, 1),
                          "first_token_ms": round((first - t1) * 1000, 1) if first else None})
            if answer:
                t2 = time.perf_counter()
                fu = suggest_followups(body.query, answer, res["results"], plan, usage)
                if fu:
                    yield emit({"type": "followups", "questions": fu})
                    trace.append({"step": "followups", "ms": round((time.perf_counter() - t2) * 1000, 1)})
        telemetry.record_usage("ask", usage.steps)
        timings = {s["step"] + "_ms": s["ms"] for s in trace if s.get("ms") is not None}
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        yield emit({"type": "done", "timings": timings, "trace": trace, "usage": usage.summary(),
                    "model": _answer_model() if settings.llm_enabled else None})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _active_models() -> dict:
    from .llm import openai_client
    return openai_client.active_models()


def _answer_model() -> str:
    """Configured answer model, or the fallback actually in use if OpenAI retired it."""
    from .llm import openai_client
    return openai_client.resolve_model(settings.answer_model())


@app.post("/api/feedback")
def feedback(body: FeedbackBody):
    fid = telemetry.add_feedback(body.question, body.answer, body.rating, body.comment,
                                 [{"title": s.get("title"), "doc_id": s.get("doc_id"), "chunk_id": s.get("chunk_id")}
                                  for s in body.sources][:10], _answer_model())
    return {"id": fid}


@app.get("/api/usage")
def usage_totals():
    return {**telemetry.usage_totals(), "feedback": telemetry.feedback_summary()}


@app.post("/api/llm/ping", dependencies=ADMIN)
def llm_ping():
    """Checks the OpenAI key, embedding model and fast model with two tiny calls (≈ $0.0001)."""
    if not settings.helper_llm_enabled and settings.embedding_backend != "openai":
        raise HTTPException(400, "OpenAI is not configured (set OPENAI_API_KEY in backend/.env)")
    try:
        return openai_client.ping()
    except openai_client.LLMNotConfigured as e:
        raise HTTPException(400, str(e))


# ----------------------------------------------------------------------------- sources
@app.get("/api/sources")
def sources():
    rows = read_manifest()
    dl = downloaded_urls()
    indexed = get_store().known_source_urls()
    out = []
    for r in rows:
        rec = dl.get(r["url"])
        out.append({**r, "downloaded": bool(rec), "indexed": r["url"] in indexed,
                    "size": rec.get("size") if rec else None})
    return {"total": len(out), "items": out}


# ----------------------------------------------------------------------------- frontend
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        f = _dist / full_path
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
