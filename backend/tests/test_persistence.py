"""The FAISS index is a local file: it must survive restarts, unusual folder names and crashes."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import BACKEND_DIR, settings
from app.main import app
from app.store import vector_store
from app.store.vector_store import VectorStore
from tests.conftest import ROOT, reconfigure
from tests.make_fixtures import make_all

FIX = make_all(Path(tempfile.mkdtemp(prefix="tnrag_fix_persist_")))
client = TestClient(app)
# spaces + Tamil letters in the path: faiss.write_index(str) fails on such paths on Windows
STORE = ROOT / "my index வேளாண்"


@pytest.fixture(scope="module", autouse=True)
def local_store():
    reconfigure("persist", vector_store_dir=str(STORE))
    yield
    settings.vector_store_dir = ""


def wait(job_id: str, timeout: float = 60) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def upload(*keys: str) -> dict:
    files = [("files", (FIX[k].name, FIX[k].read_bytes())) for k in keys]
    return wait(client.post("/api/ingest/upload", files=files).json()["job_id"])


def restart() -> VectorStore:
    """Simulate stopping and starting the server: drop the in-memory store, load from disk."""
    vector_store._store = None
    return vector_store.get_store()


def test_index_is_written_to_the_configured_local_folder():
    assert upload("pdf", "xlsx", "md")["files_indexed"] == 3
    assert (STORE / "index.faiss").stat().st_size > 0
    assert (STORE / "meta.sqlite").exists() and (STORE / "store.json").exists()
    assert not list(STORE.glob("*.tmp"))
    stats = client.get("/api/stats").json()
    assert stats["store_dir"] == str(STORE) and stats["vectors"] == stats["chunks"] > 0
    assert client.get("/api/health").json()["settings"]["vector_store_dir"] == str(STORE)


def test_index_reloads_after_restart_with_same_results():
    before = client.post("/api/search", json={"query": "drip irrigation subsidy", "rewrite": False}).json()
    n = vector_store.get_store().index.ntotal
    store = restart()
    assert store.index.ntotal == n and store.repaired == {"orphan_vectors": 0, "documents_dropped": 0}
    after = client.post("/api/search", json={"query": "drip irrigation subsidy", "rewrite": False}).json()
    assert [r["chunk_id"] for r in after["results"]] == [r["chunk_id"] for r in before["results"]]


def test_crash_before_save_is_repaired_and_reindexed():
    store = vector_store.get_store()
    docs_before = store.stats()["documents"]
    # a document whose text reached SQLite but whose vectors never reached index.faiss (killed mid-ingest)
    store.add_document({"doc_id": "crash1", "filename": "lost.txt", "title": "lost", "file_type": "txt",
                        "sha256": "deadbeef"}, [{"index": 0, "text": "never saved", "language": "en"}],
                       np.ones((1, store.index.d), np.float32) / np.sqrt(store.index.d), store.info["embedding_model"])
    store = restart()  # no save() happened
    assert store.repaired["documents_dropped"] == 1
    assert store.stats()["documents"] == docs_before and store.doc_by_sha("deadbeef") is None
    assert store.index.ntotal == store.stats()["chunks"]


def test_relative_vector_store_dir_resolves_from_backend():
    settings.vector_store_dir = "data/vectorstore"
    assert settings.store_dir == (BACKEND_DIR / "data" / "vectorstore").resolve()
    settings.vector_store_dir = str(STORE)
