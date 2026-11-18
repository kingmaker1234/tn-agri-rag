"""PIPELINE 1 — Upload → Load → Split → Embed → Index (FAISS file).

`ingest_files()` is the single entry point used by uploads, URL imports, the
manifest crawler and the CLI. It is idempotent: files whose SHA-256 is already
in the store are skipped, so re-running a crawl never duplicates vectors.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from ..config import settings
from ..jobs import Job
from ..llm.usage import Usage
from ..store.embeddings import embedder_identity, get_embedder
from .. import telemetry
from .doc_context import describe, header
from ..store.vector_store import get_store
from ..text_utils import detect_language
from .crawler import load_meta
from .loaders import SUPPORTED_EXT, LoaderError, load_file, sniff_extension
from .splitter import split_sections

MAX_ZIP_BYTES = 2 * 1024**3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _loc_label(meta: dict) -> str:
    for key, word in (("page", "p."), ("slide", "slide"), ("sheet", "sheet")):
        if key in meta:
            end = meta.get(f"{key}_end")
            return f"{word} {meta[key]}" + (f"–{end}" if end and end != meta[key] else "")
    return meta.get("heading", "")


def _expand_zip(path: Path, job: Job) -> list[Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="zip_", dir=settings.uploads_dir))
    total = 0
    files = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts or name.name.startswith("."):
                continue  # zip-slip / hidden files
            total += info.file_size
            if total > MAX_ZIP_BYTES:
                job.log(f"{path.name}: archive larger than 2 GB, remaining members skipped", "warn")
                break
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            files.append(target)
    job.log(f"{path.name}: expanded {len(files)} file(s) from archive")
    return files


def ingest_files(job: Job, paths: list[Path], file_meta: dict[str, dict] | None = None,
                 options: dict | None = None) -> None:
    options = options or {}
    file_meta = file_meta or {}
    crawl_meta = load_meta()
    chunk_size = int(options.get("chunk_size") or settings.chunk_size)
    chunk_overlap = int(options.get("chunk_overlap") or settings.chunk_overlap)
    default_category = options.get("category") or "uploaded"
    store = get_store()
    if settings.embedding_backend == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in backend/.env — add it and restart the server.")
    embedder = get_embedder()
    model_id = embedder_identity()
    store.check_model(model_id)

    # expand archives first so totals are right
    queue: list[Path] = []
    for p in paths:
        if p.suffix.lower() == ".zip":
            try:
                queue.extend(_expand_zip(p, job))
            except zipfile.BadZipFile:
                job.file_entry(p.name, status="failed", error="corrupt zip")
                job.files_failed += 1
        else:
            queue.append(p)
    job.files_total = len(queue)
    job.stage = "loading"
    job.log(f"pipeline start: {len(queue)} file(s), chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"embedder={model_id}")

    since_save = 0
    for path in queue:
        if job.cancel_requested:
            job.log("cancelled — stopping before next file", "warn")
            break
        entry = job.file_entry(path.name, size=path.stat().st_size if path.exists() else 0)
        job.current_file = path.name
        try:
            meta = {**crawl_meta.get(str(path.resolve()), {}), **file_meta.get(str(path), {})}
            sha = sha256_file(path)
            dup = store.doc_by_sha(sha)
            if dup:
                entry.update(status="skipped", error=f"duplicate of “{dup['title'] or dup['filename']}”",
                             doc_id=dup["doc_id"])
                job.files_skipped += 1
                job.files_loaded += 1
                job.files_split += 1
                continue
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXT:
                sniffed = sniff_extension(path)
                renamed = path.with_name(path.name + sniffed)
                path.rename(renamed)
                path, ext = renamed, sniffed
            entry["type"] = ext.lstrip(".")

            # ---- LOAD
            job.stage = "loading"
            loaded = load_file(path)
            job.files_loaded += 1
            title = (meta.get("title") or loaded.title or path.stem.split("__")[0]).strip()[:300]

            # ---- SPLIT
            job.stage = "splitting"
            chunks = split_sections(loaded.sections, chunk_size, chunk_overlap, settings.min_chunk_chars)
            job.files_split += 1
            if not chunks:
                entry.update(status="failed", error="no text extracted " + "; ".join(loaded.notes))
                job.files_failed += 1
                job.log(f"{path.name}: no text extracted ({', '.join(loaded.notes) or 'empty'})", "warn")
                continue
            job.chunks_total += len(chunks)

            # ---- CONTEXT (one fast-model call per document → summary, year, department, schemes)
            doc_usage = Usage()
            full_text = "\n".join(s.text for s in loaded.sections)
            ctx = describe(title, path.name, full_text, doc_usage)

            # ---- EMBED (contextual header: title, year, type, location and summary on every chunk)
            job.stage = "embedding"
            texts = [f"{header(title, ctx, _loc_label(c.meta))}\n{c.text}" for c in chunks]
            vecs = []
            step = 96 if settings.embedding_backend == "openai" else max(8, settings.embedding_batch_size * 4)
            for i in range(0, len(texts), step):
                if job.cancel_requested:
                    break
                vecs.append(embedder.embed_documents(texts[i:i + step], doc_usage))
                job.chunks_embedded += len(texts[i:i + step])
            if job.cancel_requested:
                entry.update(status="cancelled")
                break
            import numpy as np

            vectors = np.vstack(vecs)

            # ---- INDEX
            job.stage = "indexing"
            langs = [detect_language(c.text) for c in chunks]
            doc_lang = Counter(langs).most_common(1)[0][0]
            file_type = "html" if ext in (".htm", ".html", ".xhtml", ".asp", ".aspx", ".php", ".jsp") else ext.lstrip(".")
            doc_id = sha[:16]
            store.add_document(
                {
                    "doc_id": doc_id, "filename": meta.get("original_name") or path.name, "title": title,
                    "file_type": file_type, "category": meta.get("category") or default_category,
                    "language": doc_lang, "source_url": meta.get("url") or meta.get("final_url"),
                    "source_id": meta.get("source_id"), "stored_path": str(path.resolve()), "sha256": sha,
                    "size_bytes": path.stat().st_size, "n_pages": loaded.n_pages, "extraction": loaded.extraction,
                    "notes": json.dumps(loaded.notes, ensure_ascii=False), "job_id": job.id,
                    "summary": ctx.get("summary") or None, "year": ctx.get("year") or None,
                    "doc_type": ctx.get("doc_type") or None, "department": ctx.get("department") or None,
                    "schemes": json.dumps(ctx.get("schemes") or [], ensure_ascii=False),
                    "embed_tokens": sum(s["input_tokens"] for s in doc_usage.steps if s["step"].startswith("embed")),
                    "cost_usd": doc_usage.cost,
                },
                [{"index": c.index, "text": c.text, "meta": c.meta, "language": l} for c, l in zip(chunks, langs)],
                vectors, model_id)
            job.files_indexed += 1
            job.cost_usd += doc_usage.cost
            job.tokens += doc_usage.input_tokens + doc_usage.output_tokens
            telemetry.record_usage("ingest", doc_usage.steps)
            entry.update(status="indexed", chunks=len(chunks), doc_id=doc_id, language=doc_lang, title=title,
                         notes=loaded.notes, year=ctx.get("year"), summary=ctx.get("summary"))
            job.log(f"✓ {path.name}: {len(chunks)} chunks, {doc_lang}, {loaded.n_pages or '-'} pages"
                    + (f", {ctx['year']}" if ctx.get("year") else "")
                    + (f" — notes: {'; '.join(loaded.notes)}" if loaded.notes else ""))
            since_save += 1
            if since_save >= 25:  # persist progress periodically
                store.save()
                since_save = 0
        except LoaderError as e:
            entry.update(status="failed", error=str(e))
            job.files_failed += 1
            job.log(f"✗ {path.name}: {e}", "warn")
        except Exception as e:
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            job.files_failed += 1
            job.log(f"✗ {path.name}: {type(e).__name__}: {e}", "error")
            if type(e).__name__ == "ModelMismatchError":
                raise

    # ---- FINALIZE: write FAISS file + rebuild BM25 over the whole corpus
    job.stage = "finalizing"
    job.current_file = None
    job.log("writing FAISS index + rebuilding BM25 …")
    store.save()
    store.rebuild_bm25()
    s = store.stats()
    job.log(f"index now holds {s['documents']} documents / {s['vectors']} vectors "
            f"({s['index_bytes'] / 1e6:.1f} MB index.faiss); this run cost ≈ ${job.cost_usd:.4f}")
