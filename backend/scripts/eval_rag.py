"""End-to-end RAG evaluation with an LLM judge — run it with your OpenAI key and put the table in the README.

For every question in eval_questions.json it runs the full Pipeline 2 (plan → hybrid → rerank → answer)
and measures:
  Retrieval hit@k   a passage containing the expected fact is in the top-k
  Answer exact      the answer contains the expected fact (e.g. "75,000")
  Faithfulness      judge: share of the answer's claims supported by the retrieved passages (0–1)
  Cited             judge: every factual sentence carries a [n] citation
  Language match    judge: answer is in the question's language (Tamil / English / Tanglish)
Results are written to docs/EVAL.md (+ docs/eval_results.json), broken down by question language.

  python scripts/eval_rag.py                    # ≈ $0.10–0.30 with default models
  python scripts/eval_rag.py --limit 5 --k 6    # quick smoke run
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from _common import ROOT

from app.config import settings
from app.llm import openai_client
from app.llm.generator import stream_answer
from app.llm.usage import Usage
from app.retrieval.hybrid import SearchRequest, search

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "all_claims_cited": {"type": "boolean"},
        "language_match": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["faithfulness", "all_claims_cited", "language_match", "notes"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = """You grade a RAG answer strictly. Given the numbered context passages, the question and the answer:
- faithfulness: fraction (0..1) of the answer's factual claims that are directly supported by the passages.
- all_claims_cited: true if every factual sentence carries a [n] citation pointing at a supporting passage.
- language_match: true if the answer uses the question's language (Tamil script → Tamil, English → English,
  Tanglish → Tanglish or simple English mixed with Tanglish).
- notes: one short sentence on the main problem, or "ok". Reply with JSON only."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).with_name("eval_questions.json")))
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not settings.llm_enabled:
        raise SystemExit("Set OPENAI_API_KEY in backend/.env first.")
    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    usage = Usage()
    rows = []
    for n, it in enumerate(items, 1):
        t0 = time.perf_counter()
        res = search(SearchRequest(query=it["q"], top_k=args.k), usage)
        hits = res["results"]
        hit = any(all(m.lower() in (h["title"] + "\n" + h["text"]).lower() for m in it["must"]) for h in hits)
        answer = "".join(stream_answer(it["q"], hits, None, res.get("plan"), usage)) if hits else ""
        secs = time.perf_counter() - t0
        exact = all(m.lower() in answer.lower() for m in it["must"])
        ctx = "\n\n".join(f"[{i}] {h['text'][:1200]}" for i, h in enumerate(hits, 1))
        try:
            judge = openai_client.complete_json(settings.answer_model(), JUDGE_SYSTEM,
                                                f"Passages:\n{ctx}\n\nQuestion: {it['q']}\n\nAnswer:\n{answer}",
                                                JUDGE_SCHEMA, "rag_judge", usage, "judge", max_output_tokens=3000,
                                                effort=settings.openai_reasoning_effort)
        except Exception as e:
            judge = {"notes": f"judge error {e}"}
        judge = {"faithfulness": float(judge.get("faithfulness", 0.0)), "all_claims_cited": bool(judge.get("all_claims_cited")),
                 "language_match": bool(judge.get("language_match")), "notes": str(judge.get("notes", ""))}
        rows.append({**it, "hit": hit, "exact": exact, "seconds": round(secs, 2), **judge, "answer": answer})
        print(f"[{n:>2}/{len(items)}] {'✓' if hit else '✗'} retrieval  {'✓' if exact else '✗'} answer  "
              f"faith {judge['faithfulness']:.2f}  {secs:4.1f}s  {it['q'][:60]}")

    def agg(rs):
        return {"n": len(rs), "hit": sum(r["hit"] for r in rs) / len(rs), "exact": sum(r["exact"] for r in rs) / len(rs),
                "faith": statistics.mean(r["faithfulness"] for r in rs),
                "cited": sum(r["all_claims_cited"] for r in rs) / len(rs),
                "lang": sum(r["language_match"] for r in rs) / len(rs),
                "sec": statistics.median(r["seconds"] for r in rs)}

    groups = defaultdict(list)
    for r in rows:
        groups[r.get("lang", "en")].append(r)
    table = [("All", agg(rows))] + [(LANG, agg(rs)) for LANG, rs in sorted(groups.items())]
    md = ["# Evaluation", "", f"Models: answers `{settings.answer_model()}`, helper `{settings.fast_model()}`, "
          f"embeddings `{settings.embedding_label()}`, reranker `{settings.reranker_backend}`. "
          f"Top-k = {args.k}. Run: {time.strftime('%Y-%m-%d')}.", "",
          "| Questions | n | Retrieval hit@k | Answer exact | Faithfulness | Fully cited | Language match | Median latency |",
          "|---|---|---|---|---|---|---|---|"]
    for name, a in table:
        md.append(f"| {name} | {a['n']} | {a['hit']:.0%} | {a['exact']:.0%} | {a['faith']:.2f} | {a['cited']:.0%} | "
                  f"{a['lang']:.0%} | {a['sec']:.1f} s |")
    md += ["", f"Total evaluation cost ≈ ${usage.cost:.3f} ({usage.input_tokens:,} input + {usage.output_tokens:,} "
               "output tokens, including the judge).", "", "## Misses", ""]
    for r in rows:
        if not (r["hit"] and r["exact"]) or r["faithfulness"] < 0.8:
            md.append(f"- **{r['q']}** — retrieval {'ok' if r['hit'] else 'miss'}, answer {'ok' if r['exact'] else 'miss'}, "
                      f"faithfulness {r['faithfulness']:.2f}: {r['notes']}")
    docs = ROOT.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "EVAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (docs / "eval_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(md[4:4 + len(table) + 2]))
    print(f"\nwritten: {docs / 'EVAL.md'}")


if __name__ == "__main__":
    main()
