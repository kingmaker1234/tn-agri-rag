"""End-to-end tests for both pipelines (offline: hash embeddings, no reranker, no LLM).

    cd backend && pytest -q
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingest.loaders import load_file
from app.ingest.splitter import split_sections
from app.main import app
from tests.conftest import reconfigure
from tests.make_fixtures import make_all

FIX = make_all(Path(tempfile.mkdtemp(prefix="tnrag_fix_")))
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def offline_mode():
    reconfigure("offline")
    yield


def wait(job_id: str, timeout: float = 60) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_loaders_every_format():
    for kind, path in FIX.items():
        if kind in ("zip", "noext"):
            continue
        doc = load_file(path)
        text = "\n".join(s.text for s in doc.sections)
        assert len(text) > 20, f"{kind}: no text"
    pdf = load_file(FIX["pdf"])
    assert [s.meta["page"] for s in pdf.sections] == [1, 2]
    xl = load_file(FIX["xlsx"]).sections[0].text
    assert "Scheme: Micro irrigation | Category: Small/Marginal | Subsidy %: 100%" in xl
    html = load_file(FIX["html"]).sections[0].text
    assert "2,338" in html and "var x" not in html and "Home | About" not in html


def test_splitter_tracks_pages():
    doc = load_file(FIX["pdf"])
    chunks = split_sections(doc.sections, 300, 50)
    assert len(chunks) > 3
    assert chunks[0].meta["page"] == 1 and chunks[-1].meta.get("page_end", chunks[-1].meta["page"]) == 2
    for c in chunks:
        assert len(c.text) <= 300


def test_pipeline1_upload_and_index():
    files = [("files", (p.name, p.read_bytes())) for p in FIX.values()]
    r = client.post("/api/ingest/upload", files=files, data={"category": "test"})
    assert r.status_code == 200, r.text
    job = wait(r.json()["job_id"])
    assert job["status"] == "done", job
    names = {f["name"]: f for f in job["files"]}
    failed = {n: f["error"] for n, f in names.items() if f["status"] == "failed"}
    assert not failed, failed
    assert not any("evil" in n for n in names)  # zip-slip member ignored
    assert job["files_indexed"] >= 13
    s = client.get("/api/stats").json()
    assert s["vectors"] == s["chunks"] > 0
    for t in ("pdf", "docx", "pptx", "xlsx", "csv", "html", "txt", "md", "json", "xml", "doc"):
        assert t in s["by_type"], t
    from app.config import settings
    assert (settings.data_dir / "vectorstore" / "index.faiss").exists()

    # idempotent: same files again -> all skipped as duplicates
    r = client.post("/api/ingest/upload", files=files, data={"category": "test"})
    job2 = wait(r.json()["job_id"])
    assert job2["files_indexed"] == 0 and job2["files_skipped"] >= 12


def test_pipeline2_search_modes():
    for mode in ("hybrid", "dense", "bm25"):
        r = client.post("/api/search", json={"query": "drip irrigation subsidy small marginal farmers", "mode": mode})
        assert r.status_code == 200
        res = r.json()["results"]
        assert res, mode
        assert any("100%" in x["text"] for x in res[:3]), (mode, [x["title"] for x in res])
    r = client.post("/api/search", json={"query": "உழவர் சந்தை மதுரை", "mode": "hybrid"}).json()
    assert any("1999" in x["text"] for x in r["results"][:3])
    # score breakdown present in hybrid mode
    top = client.post("/api/search", json={"query": "KAVIADP 2,338 gram panchayats", "mode": "hybrid"}).json()["results"][0]
    assert top["scores"]["fused"] is not None and "2,338" in top["text"]
    # filter by file type
    only_xlsx = client.post("/api/search", json={"query": "subsidy", "filters": {"file_types": ["xlsx"]}}).json()
    assert only_xlsx["results"] and all(x["file_type"] == "xlsx" for x in only_xlsx["results"])


def test_compare_and_ask_stream():
    cmp = client.post("/api/search/compare", json={"query": "solar pump subsidy 70%"}).json()
    assert {"bm25", "dense", "hybrid", "hybrid_rerank"} <= set(cmp)
    with client.stream("POST", "/api/ask", json={"query": "What is the PM-KUSUM subsidy?"}) as r:
        events = [json.loads(l) for l in r.iter_lines() if l]
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["plan", "sources"] and kinds[-1] == "done"
    sources = next(e for e in events if e["type"] == "sources")
    assert any("70%" in x["text"] for x in sources["results"])
    assert events[0]["plan"]["source"] == "fallback"  # no LLM in offline mode


def test_library_and_delete():
    docs = client.get("/api/documents?limit=100").json()
    assert docs["total"] >= 13
    d = docs["items"][0]
    detail = client.get(f"/api/documents/{d['doc_id']}").json()
    assert detail["chunks"]["total"] == d["n_chunks"]
    assert client.get(f"/api/documents/{d['doc_id']}/file").status_code == 200
    before = client.get("/api/stats").json()["vectors"]
    assert client.delete(f"/api/documents/{d['doc_id']}").status_code == 200
    after = client.get("/api/stats").json()["vectors"]
    assert after == before - d["n_chunks"]


def test_admin_token_protects_mutations():
    from app.config import settings
    settings.admin_token = "s3cret"
    try:
        assert client.post("/api/index/reset?confirm=RESET").status_code == 401
        assert client.post("/api/search", json={"query": "subsidy"}).status_code == 200  # reads stay open
        assert client.post("/api/index/reset?confirm=WRONG", headers={"X-Admin-Token": "s3cret"}).status_code == 400
    finally:
        settings.admin_token = ""


def test_feedback_and_usage():
    assert client.post("/api/feedback", json={"question": "q", "answer": "a", "rating": 1}).status_code == 200
    u = client.get("/api/usage").json()
    assert u["feedback"]["up"] == 1


def test_rate_limit_on_paid_endpoints():
    from app import main
    from app.config import settings
    settings.rate_limit_per_min, main._hits = 3, {}
    try:
        codes = [client.post("/api/search", json={"query": "subsidy"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert client.get("/api/stats").status_code == 200  # free endpoints unaffected
    finally:
        settings.rate_limit_per_min, main._hits = 0, {}
