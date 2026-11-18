"""Both pipelines through the official `openai` SDK against an OpenAI-compatible mock server.

Covers: OpenAI embeddings (dimensions), document context, query planning (Tanglish), LLM reranking,
streamed answers with citations, follow-ups, token/cost accounting, the Responses API and the Chat
Completions style, gpt-4o-mini defaults (temperature, no reasoning), automatic handling of parameters
that reasoning models reject, and the fallback model when the configured model is retired.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests import mock_openai
from tests.conftest import reconfigure
from tests.make_fixtures import make_all

FIX = make_all(Path(tempfile.mkdtemp(prefix="tnrag_fix_oa_")))
client = TestClient(app)
STATE: dict = {}


def openai_settings(base: str, style: str, **kw) -> dict:
    """The shipped defaults: gpt-4o-mini for answers and helper calls, text-embedding-3-large @1536."""
    return {**dict(embedding_backend="openai", openai_api_key="test-key", openai_base_url=base,
                   openai_api_style=style, openai_embedding_model="text-embedding-3-large",
                   openai_embedding_dimensions=1536, llm_provider="openai", openai_chat_model="gpt-4o-mini",
                   openai_fast_model="gpt-4o-mini", openai_fallback_model="gpt-6-luna", llm_max_output_tokens=4000,
                   reranker_enabled=True, reranker_backend="llm"), **kw}


@pytest.fixture(scope="module", autouse=True)
def openai_mode():
    base, rec, srv = mock_openai.start()
    STATE.update(base=base, rec=rec)
    reconfigure("openai", **openai_settings(base, "responses"))
    yield
    srv.shutdown()


def wait(job_id: str, timeout: float = 60) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def ask(query: str, history=None) -> list[dict]:
    with client.stream("POST", "/api/ask", json={"query": query, "history": history or []}) as r:
        return [json.loads(l) for l in r.iter_lines() if l]


def test_pipeline1_with_openai_embeddings_and_doc_context():
    files = [("files", (p.name, p.read_bytes())) for k, p in FIX.items() if k in ("pdf", "xlsx", "html", "md", "docx")]
    job = wait(client.post("/api/ingest/upload", files=files).json()["job_id"])
    assert job["status"] == "done" and job["files_indexed"] == 5, job
    assert job["cost_usd"] > 0 and job["tokens"] > 0
    stats = client.get("/api/stats").json()
    assert stats["dim"] == 1536 and stats["embedding_model"] == "openai:text-embedding-3-large@1536"
    docs = client.get("/api/documents?limit=50").json()["items"]
    assert all(d["summary"].startswith("Mock summary") for d in docs)
    emb = STATE["rec"].bodies("/embeddings")
    assert emb and all(b["dimensions"] == 1536 for b in emb)
    ctx = [b for b in STATE["rec"].bodies("/responses") if (b.get("text") or {}).get("format", {}).get("name") == "doc_context"]
    assert len(ctx) == 5 and all(b["store"] is False for b in ctx)


def test_pipeline2_search_with_plan_and_llm_rerank():
    res = client.post("/api/search", json={"query": "drip irrigation ku evlo subsidy kidaikum"}).json()
    assert res["plan"]["language"] == "tanglish" and res["plan"]["source"] == "llm"
    assert len(res["variants"]) >= 2 and res["reranked"] is True
    assert any("100%" in r["text"] for r in res["results"][:3])
    assert res["results"][0]["scores"]["rerank"] is not None
    assert res["usage"]["cost_usd"] > 0
    steps = {s["step"] for s in res["trace"]}
    assert {"plan", "dense", "bm25", "fuse", "rerank"} <= steps


def test_ask_streams_cited_answer_followups_and_cost():
    ev = ask("What subsidy do small farmers get for drip irrigation?")
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "plan" and kinds[1] == "sources" and "token" in kinds and kinds[-1] == "done"
    answer = "".join(e["text"] for e in ev if e["type"] == "token")
    assert "[1]" in answer
    assert next(e for e in ev if e["type"] == "followups")["questions"]
    done = ev[-1]
    steps = {s["step"] for s in done["usage"]["steps"]}
    assert {"plan_query", "rerank", "answer", "followups"} <= steps
    assert done["usage"]["cost_usd"] > 0 and done["model"] == "gpt-4o-mini"
    assert {s["model"] for s in done["usage"]["steps"]} == {"gpt-4o-mini", "text-embedding-3-large"}
    # gpt-4o-mini via the Responses API: temperature, no reasoning parameter, never rejected
    body = STATE["rec"].bodies("/responses")
    ans = [b for b in body if b.get("stream")]
    assert ans and all("reasoning" not in b and "max_tokens" not in b for b in ans)
    assert ans[-1]["temperature"] == 0.2 and ans[-1]["max_output_tokens"] == 4000
    helpers = [b for b in body if not b.get("stream")]
    assert helpers and all(b["temperature"] == 0 and "reasoning" not in b for b in helpers)


def test_followup_question_uses_history():
    ev = ask("documents enna venum?", history=[{"role": "user", "content": "PM-KUSUM solar pump subsidy?"},
                                                {"role": "assistant", "content": "70% subsidy [1]."}])
    plan_req = [b for b in STATE["rec"].bodies("/responses")
                if (b.get("text") or {}).get("format", {}).get("name") == "query_plan"][-1]
    assert "PM-KUSUM" in json.dumps(plan_req["input"])
    assert ev[-1]["type"] == "done"


def test_chat_completions_style_with_gpt_4o_mini():
    reconfigure("openai_chat_mini", **openai_settings(STATE["base"], "chat"))
    files = [("files", (FIX["xlsx"].name, FIX["xlsx"].read_bytes()))]
    assert wait(client.post("/api/ingest/upload", files=files).json()["job_id"])["files_indexed"] == 1
    n = len(STATE["rec"].bodies("/chat/completions"))
    ev = ask("micro irrigation subsidy for other farmers")
    assert "[1]" in "".join(e["text"] for e in ev if e["type"] == "token")
    chats = STATE["rec"].bodies("/chat/completions")[n:]
    assert chats and all(b["model"] == "gpt-4o-mini" and "reasoning_effort" not in b for b in chats)
    assert all("max_tokens" in b and "temperature" in b for b in chats)  # accepted first time, no retries


def test_chat_completions_style_adapts_rejected_params():
    reconfigure("openai_chat", **openai_settings(STATE["base"], "chat", **{
        "openai_chat_model": "gpt-6.1-sol", "openai_fast_model": "gpt-6-luna"}))
    files = [("files", (FIX["xlsx"].name, FIX["xlsx"].read_bytes()))]
    assert wait(client.post("/api/ingest/upload", files=files).json()["job_id"])["files_indexed"] == 1
    ev = ask("micro irrigation subsidy for other farmers")
    assert "[1]" in "".join(e["text"] for e in ev if e["type"] == "token")
    chats = STATE["rec"].bodies("/chat/completions")
    # first call carried max_tokens + temperature and was rejected; retries remapped / dropped them
    assert any("max_tokens" in b for b in chats)
    assert any("max_completion_tokens" in b and "temperature" not in b for b in chats)


def test_ping_and_bad_key():
    reconfigure("openai_ping", **openai_settings(STATE["base"], "responses"))
    out = client.post("/api/llm/ping").json()
    assert out["embeddings"].startswith("ok") and out["fast_model"] == "ok (gpt-4o-mini)"
    assert out["answer"] == "gpt-4o-mini" and out["fallback_active"] is False
    reconfigure("openai_badkey", **{**openai_settings(STATE["base"], "responses"), "openai_api_key": "wrong"})
    out = client.post("/api/llm/ping").json()
    assert "error" in out["embeddings"] and "error" in out["fast_model"]


def test_unsupported_reasoning_effort_is_dropped():
    reconfigure("openai_effort", **openai_settings(STATE["base"], "responses", openai_chat_model="gpt-6.1-sol",
                                                   openai_reasoning_effort="minimal"))
    files = [("files", (FIX["md"].name, FIX["md"].read_bytes()))]
    assert wait(client.post("/api/ingest/upload", files=files).json()["job_id"])["files_indexed"] == 1
    ev = ask("நுண்ணீர் பாசனம் மானியம்")
    assert "".join(e["text"] for e in ev if e["type"] == "token").strip()
    sol = [b for b in STATE["rec"].bodies("/responses") if b["model"] == "gpt-6.1-sol"]
    assert sol[-2]["reasoning"] == {"effort": "minimal"} and "reasoning" not in sol[-1]


def test_retired_model_falls_back_automatically():
    """If OpenAI retires gpt-4o-mini (announced for 14 Jan 2027), the app keeps working on the fallback."""
    reconfigure("openai_retired", **openai_settings(STATE["base"], "responses", openai_chat_model="gpt-retired-mini",
                                                    openai_fast_model="gpt-retired-mini"))
    files = [("files", (FIX["md"].name, FIX["md"].read_bytes()))]
    assert wait(client.post("/api/ingest/upload", files=files).json()["job_id"])["files_indexed"] == 1
    ev = ask("drip irrigation subsidy")
    assert "[1]" in "".join(e["text"] for e in ev if e["type"] == "token")
    assert ev[-1]["model"] == "gpt-6-luna"
    models = client.get("/api/health").json()["models"]
    assert models["fallback_active"] is True and models["answer"] == "gpt-6-luna"
    luna = [b for b in STATE["rec"].bodies("/responses") if b["model"] == "gpt-6-luna"]
    assert luna and all("temperature" not in b for b in luna)  # reasoning-model params for the fallback


def test_retired_model_without_fallback_reports_error():
    reconfigure("openai_retired_off", **openai_settings(STATE["base"], "responses", openai_chat_model="gpt-retired-mini",
                                                        openai_fast_model="gpt-retired-mini", openai_fallback_model=""))
    files = [("files", (FIX["md"].name, FIX["md"].read_bytes()))]
    job = wait(client.post("/api/ingest/upload", files=files).json()["job_id"])
    assert job["files_indexed"] == 1  # document context is optional; embeddings still work
    ev = ask("drip irrigation subsidy")
    err = [e for e in ev if e["type"] == "error"]
    assert err and "gpt-retired-mini" in err[0]["message"]
    assert any(e["type"] == "sources" for e in ev)  # retrieval still shown
