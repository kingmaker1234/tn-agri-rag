"""PIPELINE 2 — query → plan (rewrite) → (dense ∥ BM25 for every variant) → RRF fusion → rerank → top-k.

Each returned hit carries a full score breakdown (best dense rank/score, best BM25 rank/score,
fused RRF score, rerank score) so the UI can show *why* a chunk was retrieved, and the response
carries a step-by-step trace with timings, models, tokens and cost.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..config import settings
from ..llm.usage import Usage
from ..store.embeddings import embedder_identity, get_embedder
from ..store.vector_store import get_store
from .query_planner import fallback_plan, plan_query, variants
from .reranker import get_reranker, reranker_usable

log = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    query: str
    mode: str = field(default_factory=lambda: settings.retrieval_mode)  # hybrid | dense | bm25
    top_k: int = field(default_factory=lambda: settings.top_k)
    rerank: bool = True
    rewrite: bool = True
    candidate_k: int = field(default_factory=lambda: settings.candidate_k)
    rerank_candidates: int = field(default_factory=lambda: settings.rerank_candidates)
    dense_weight: float = field(default_factory=lambda: settings.dense_weight)
    bm25_weight: float = field(default_factory=lambda: settings.bm25_weight)
    max_per_doc: int = field(default_factory=lambda: settings.max_chunks_per_doc)
    filters: dict | None = None
    min_rerank_score: float = 0.0
    history: list[dict] | None = None


def rrf_fuse(lists: list[tuple[list[tuple[int, float]], float]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for hits, w in lists:
        for rank, (cid, _) in enumerate(hits, 1):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)
    return fused


def _best(rank_maps: list[dict[int, tuple[int, float]]], cid: int) -> tuple[int, float] | None:
    best = None
    for m in rank_maps:
        if cid in m and (best is None or m[cid][0] < best[0]):
            best = m[cid]
    return best


def search(req: SearchRequest, usage: Usage | None = None, plan: dict | None = None) -> dict:
    usage = usage if usage is not None else Usage()
    t0 = time.perf_counter()
    store = get_store()
    trace: list[dict] = []
    q = req.query.strip()
    if not q:
        return {"query": q, "results": [], "timings": {}, "mode": req.mode, "reranked": False, "trace": []}

    # 1. understand the query
    if plan is None:
        t = time.perf_counter()
        plan = plan_query(q, req.history, usage) if req.rewrite else fallback_plan(q)
        trace.append({"step": "plan", "ms": round((time.perf_counter() - t) * 1000, 1), "source": plan.get("source"),
                      "language": plan.get("language")})
    qvars = variants(q, plan) if req.rewrite else [("original", q)]
    keywords = " ".join(plan.get("keywords") or [])
    allowed = store.allowed_ids(req.filters)

    # 2. retrieve every variant with both retrievers
    dense_lists: list[list[tuple[int, float]]] = []
    bm25_lists: list[list[tuple[int, float]]] = []
    if req.mode in ("hybrid", "dense"):
        store.check_model(embedder_identity())
        t = time.perf_counter()
        qvecs = get_embedder().embed_queries([v for _, v in qvars], usage)
        t_emb = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        dense_lists = [store.dense_search(vec, req.candidate_k, allowed) for vec in qvecs]
        trace.append({"step": "dense", "ms": round(t_emb + (time.perf_counter() - t) * 1000, 1),
                      "queries": len(qvars), "hits": sum(len(x) for x in dense_lists)})
    if req.mode in ("hybrid", "bm25"):
        t = time.perf_counter()
        bm25_lists = [store.bm25_search(v, req.candidate_k, allowed) for _, v in qvars]
        if keywords:
            bm25_lists.append(store.bm25_search(keywords, req.candidate_k, allowed))
        trace.append({"step": "bm25", "ms": round((time.perf_counter() - t) * 1000, 1),
                      "queries": len(bm25_lists), "hits": sum(len(x) for x in bm25_lists)})

    dense_maps = [{cid: (i + 1, s) for i, (cid, s) in enumerate(lst)} for lst in dense_lists]
    bm25_maps = [{cid: (i + 1, s) for i, (cid, s) in enumerate(lst)} for lst in bm25_lists]

    # 3. fuse (original query lists weigh a little more than rewrites)
    def w(i: int, base: float) -> float:
        return base * (1.0 if i == 0 else 0.85)

    t = time.perf_counter()
    lists = [(lst, w(i, req.dense_weight)) for i, lst in enumerate(dense_lists)] + \
            [(lst, w(i, req.bm25_weight)) for i, lst in enumerate(bm25_lists)]
    if req.mode == "hybrid" or len(lists) > 1:
        fused = rrf_fuse(lists, settings.rrf_k)
    else:
        fused = {cid: s for cid, s in (lists[0][0] if lists else [])}
    order = sorted(fused, key=lambda c: -fused[c])
    fused_rank = {cid: i + 1 for i, cid in enumerate(order)}
    trace.append({"step": "fuse", "ms": round((time.perf_counter() - t) * 1000, 1), "candidates": len(order)})

    # 4. rerank the top candidates
    reranked = False
    rr_scores: dict[int, float] = {}
    pool = order[: max(req.rerank_candidates, req.top_k)] if req.rerank else order[:200]
    chunks = store.get_chunks(pool[:200])
    if req.rerank and pool and reranker_usable():
        rr = get_reranker()
        t = time.perf_counter()
        try:
            ids = [c for c in pool if c in chunks]
            texts = [f"{chunks[c]['title'] or chunks[c]['filename']}"
                     + (f" ({chunks[c]['year']})" if chunks[c].get("year") else "") + f"\n{chunks[c]['text']}"
                     for c in ids]
            rq = plan.get("standalone_question") or q
            if plan.get("english_query") and plan.get("language") == "tanglish":
                rq = f"{rq}\n(English: {plan['english_query']})"
            for cid, s in zip(ids, rr.score(rq, texts, usage)):
                rr_scores[cid] = s
            pool = sorted(ids, key=lambda c: (-rr_scores[c], fused_rank[c]))
            reranked = True
            trace.append({"step": "rerank", "ms": round((time.perf_counter() - t) * 1000, 1), "kind": rr.kind,
                          "model": rr.name, "candidates": len(ids)})
        except Exception as e:  # reranker unavailable → keep fused order
            log.warning("rerank skipped: %s", e)
            trace.append({"step": "rerank", "skipped": str(e)[:200]})

    # 5. diversity cap per document, then top_k
    per_doc: dict[str, int] = {}
    results = []
    for cid in pool:
        c = chunks.get(cid)
        if not c:
            continue
        if reranked and rr_scores.get(cid, 1) < req.min_rerank_score:
            continue
        if per_doc.get(c["doc_id"], 0) >= max(1, req.max_per_doc):
            continue
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
        d = _best(dense_maps, cid)
        b = _best(bm25_maps, cid)
        hl_query = " ".join([q, plan.get("english_query") or "", plan.get("tamil_query") or "", keywords])
        results.append({
            "chunk_id": cid, "doc_id": c["doc_id"], "chunk_index": c["chunk_index"], "n_chunks": c["n_chunks"],
            "title": c["title"] or c["filename"], "filename": c["filename"], "file_type": c["file_type"],
            "category": c["category"], "source_url": c["source_url"], "language": c["language"],
            "year": c.get("year"), "doc_type": c.get("doc_type"), "location": c["meta"], "text": c["text"],
            "matched_terms": store.bm25.matched_terms(hl_query, c["text"]),
            "scores": {"dense": d[1] if d else None, "dense_rank": d[0] if d else None,
                       "bm25": b[1] if b else None, "bm25_rank": b[0] if b else None,
                       "fused": fused.get(cid), "fused_rank": fused_rank.get(cid), "rerank": rr_scores.get(cid)},
        })
        if len(results) >= req.top_k:
            break
    for i, r in enumerate(results, 1):
        r["rank"] = i
    total = (time.perf_counter() - t0) * 1000
    timings = {s["step"] + "_ms": s["ms"] for s in trace if "ms" in s}
    timings["total_ms"] = round(total, 1)
    return {
        "query": q, "mode": req.mode, "reranked": reranked, "results": results, "plan": plan,
        "variants": [{"label": lbl, "text": txt} for lbl, txt in qvars],
        "candidates": {"dense": len({c for lst in dense_lists for c, _ in lst}),
                       "bm25": len({c for lst in bm25_lists for c, _ in lst}), "fused": len(fused)},
        "timings": timings, "trace": trace, "usage": usage.summary(),
    }
