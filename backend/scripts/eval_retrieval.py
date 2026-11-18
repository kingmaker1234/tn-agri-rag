"""Compare BM25 vs dense vs hybrid vs hybrid+rerank on your own index.

Each eval item is a question plus keywords that a correct passage must contain.
Metrics: Hit@k (any correct passage in top-k) and MRR@k.

  python scripts/eval_retrieval.py                  # uses scripts/eval_questions.json
  python scripts/eval_retrieval.py --k 5 --file my_questions.json
"""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401

from app.retrieval.hybrid import SearchRequest, search

CONFIGS = [("BM25", "bm25", False), ("Dense", "dense", False), ("Hybrid (RRF)", "hybrid", False),
           ("Hybrid + Rerank", "hybrid", True)]


def is_hit(text: str, must: list[str]) -> bool:
    low = text.lower()
    return all(m.lower() in low for m in must)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).with_name("eval_questions.json")))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    print(f"{len(items)} questions, k={args.k}\n")
    rows = []
    for name, mode, rr in CONFIGS:
        hits, mrr, ms = 0, 0.0, 0.0
        reranked_any = False
        for it in items:
            res = search(SearchRequest(query=it["q"], mode=mode, rerank=rr, top_k=args.k, max_per_doc=args.k))
            reranked_any |= res["reranked"]
            ms += res["timings"].get("total_ms", 0)
            for i, r in enumerate(res["results"], 1):
                if is_hit(r["title"] + "\n" + r["text"], it["must"]):
                    hits += 1
                    mrr += 1 / i
                    break
        label = name + ("" if not rr or reranked_any else " (reranker unavailable → same as hybrid)")
        rows.append((label, hits / len(items), mrr / len(items), ms / len(items)))
    w = max(len(r[0]) for r in rows)
    print(f"{'config':<{w}}  Hit@{args.k}   MRR@{args.k}   avg ms")
    for label, h, m, t in rows:
        print(f"{label:<{w}}  {h:6.2%}  {m:6.3f}  {t:7.1f}")


if __name__ == "__main__":
    main()
