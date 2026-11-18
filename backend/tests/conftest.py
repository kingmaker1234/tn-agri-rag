"""Test configuration: every test module gets its own data dir and settings, offline by default."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="tnrag_tests_")).resolve()  # long names on Windows
os.environ.update({"DATA_DIR": str(ROOT / "default"), "EMBEDDING_BACKEND": "hash", "RERANKER_ENABLED": "false",
                   "LLM_PROVIDER": "none", "WARMUP_MODELS": "false", "OPENAI_API_KEY": "", "ADMIN_TOKEN": "", "RATE_LIMIT_PER_MIN": "0"})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reconfigure(name: str, **overrides) -> Path:
    """Point the app at a fresh data dir with the given settings and drop every cached singleton."""
    from app import jobs, telemetry
    from app.config import settings
    from app.llm import openai_client
    from app.retrieval import query_planner, reranker
    from app.store import embeddings, vector_store

    data = ROOT / name
    defaults = dict(data_dir=data, vector_store_dir="", embedding_backend="hash", reranker_enabled=False, reranker_backend="llm",
                    llm_provider="none", openai_api_key="", openai_api_style="auto", admin_token="", rate_limit_per_min=0,
                    query_rewrite=True, doc_context=True, followups=True)
    for k, v in {**defaults, **overrides}.items():
        setattr(settings, k, v)
    settings.ensure_dirs()
    vector_store._store = None
    embeddings._embedder = None
    reranker._rr = None
    query_planner._cache.clear()
    openai_client.reset_client()
    telemetry._db = None
    jobs._manager = None
    return data
