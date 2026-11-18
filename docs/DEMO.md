# 3-minute demo script

Use this for the live demo or the submission video. Before recording: index the full corpus
(`./run.sh download`, or `run.bat download` on Windows), run `python scripts/eval_rag.py` in `backend/`, and keep
the app open at http://localhost:8000.

| Time | Show | Say |
|---|---|---|
| 0:00 | Ask page, empty state | "Tamil Nadu farm-scheme rules are spread over hundreds of Tamil and English government PDFs. Farmers ask in Tamil or Tanglish." |
| 0:15 | Ask **"drip irrigation ku evlo subsidy kidaikum?"** | Point at **Searched as**: the Tanglish question was rewritten into English and Tamil queries. The answer streams with citations. |
| 0:40 | Click a citation `[1]` | The evidence panel jumps to the exact passage, page and year, with a link to the original government PDF. |
| 0:55 | Open **How this answer was built** | Each pipeline step with latency, model, tokens and **cost per answer**. |
| 1:10 | Click a follow-up chip, then ask in Tamil: **"உழவர் சந்தை எப்போது தொடங்கப்பட்டது?"** | Follow-ups keep context; Tamil in, Tamil out. |
| 1:30 | **Explore → Compare 4 ways** with `G.O. Ms. No. 65 green gram` | Point at where each retriever ranks the passage with the exact G.O. number, and how hybrid + rerank combines them. "This is why we use hybrid retrieval with reranking." |
| 1:55 | **Add documents**: drop a PDF + an Excel file | Live stages: load → split → summarise + embed → FAISS → save, with per-file status and indexing cost. |
| 2:20 | **Library**: open a policy note | The LLM-extracted year, department and schemes, the stored chunks, and the Tamil text repaired from a broken PDF. |
| 2:40 | **System** → Test OpenAI connection, spend by model, **Saved on this computer** path; then `docs/EVAL.md` | "The FAISS index is a local file that survives restarts. Evaluated on 24 English, Tamil and Tanglish questions: <paste your hit@k / faithfulness>." |
| 2:55 | Toggle dark theme, close | "Runs locally with one command on gpt-4o-mini, about a third of a cent per answer; 23 tests in CI on Linux and Windows; MIT licensed." |

## Judging checklist

- [ ] README hero screenshot shows a **real** answer from your OpenAI key (replace `docs/screenshots/*`).
- [ ] `docs/EVAL.md` filled by `scripts/eval_rag.py` and its table pasted into the README.
- [ ] CI badge URL updated with your GitHub user/repo, and the first Actions run is green.
- [ ] `LICENSE` has your name.
- [ ] If judges need a link, run it on any machine with `HOST=0.0.0.0 ./run.sh` and set `ADMIN_TOKEN` in `backend/.env`, so judges can ask but not delete.
- [ ] Judging after 14 Jan 2027? OpenAI shuts `gpt-4o-mini` down that day. The app falls back to `gpt-6-luna` on its own, but test it once or set `OPENAI_CHAT_MODEL` yourself.
- [ ] `git log` tells a story: small, descriptive commits (not one "final" commit).
