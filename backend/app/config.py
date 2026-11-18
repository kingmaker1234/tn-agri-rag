"""Central configuration. Every value can be overridden with an environment
variable (or a .env file in backend/)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- paths -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BACKEND_DIR / "data"))).resolve())
    # where the FAISS index (index.faiss) + chunk metadata (meta.sqlite) are saved on this computer
    vector_store_dir: str = field(default_factory=lambda: _env("VECTOR_STORE_DIR", ""))

    # --- embeddings ----------------------------------------------------------
    # "openai" (default, no GPU / no model download), "sentence-transformers" (local, needs
    # requirements-local.txt) or "hash" (offline test mode, not semantic)
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "openai"))
    openai_embedding_model: str = field(default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    # text-embedding-3 models support shortened (Matryoshka) vectors; 1536 keeps ~all quality at half the size
    openai_embedding_dimensions: int = field(default_factory=lambda: _env_int("OPENAI_EMBEDDING_DIMENSIONS", 1536))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-m3"))  # local backend
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 16))
    embedding_max_seq_length: int = field(default_factory=lambda: _env_int("EMBEDDING_MAX_SEQ_LENGTH", 512))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))  # auto | cpu | cuda | mps

    # --- reranker ------------------------------------------------------------
    # "llm" (OpenAI fast model scores the candidates), "cross-encoder" (local bge-reranker), "none"
    reranker_backend: str = field(default_factory=lambda: _env("RERANKER_BACKEND", "llm"))
    reranker_enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", True))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    reranker_max_length: int = field(default_factory=lambda: _env_int("RERANKER_MAX_LENGTH", 512))

    # --- chunking --------------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 40))

    # --- retrieval defaults ------------------------------------------------------
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))  # hybrid | dense | bm25
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 50))  # per retriever
    rerank_candidates: int = field(default_factory=lambda: _env_int("RERANK_CANDIDATES", 30))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 1.0))
    bm25_weight: float = field(default_factory=lambda: _env_float("BM25_WEIGHT", 1.0))
    max_chunks_per_doc: int = field(default_factory=lambda: _env_int("MAX_CHUNKS_PER_DOC", 3))

    # --- LLM -----------------------------------------------------------------
    # openai (default; Responses API) | openai-compatible (Chat Completions: Groq, Gemini, vLLM,
    # LM Studio …) | ollama (local, via its OpenAI-compatible endpoint) | anthropic | none
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_style: str = field(default_factory=lambda: _env("OPENAI_API_STYLE", "auto"))  # auto | responses | chat
    openai_organization: str = field(default_factory=lambda: _env("OPENAI_ORG_ID", ""))
    # answer model and fast model (query planning, reranking, document context, follow-ups)
    openai_chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_fast_model: str = field(default_factory=lambda: _env("OPENAI_FAST_MODEL", "gpt-4o-mini"))
    # used automatically if OpenAI reports the configured model as retired / unavailable
    # (gpt-4o-mini is deprecated; OpenAI's announced shutdown is 14 Jan 2027). Empty = off.
    openai_fallback_model: str = field(default_factory=lambda: _env("OPENAI_FALLBACK_MODEL", "gpt-6-luna"))
    # only sent to reasoning models (o-series, gpt-5/6 …); gpt-4o-mini uses temperature instead
    openai_reasoning_effort: str = field(default_factory=lambda: _env("OPENAI_REASONING_EFFORT", "low"))
    openai_fast_reasoning_effort: str = field(default_factory=lambda: _env("OPENAI_FAST_REASONING_EFFORT", "low"))
    openai_timeout: float = field(default_factory=lambda: _env_float("OPENAI_TIMEOUT", 120))
    openai_max_retries: int = field(default_factory=lambda: _env_int("OPENAI_MAX_RETRIES", 4))
    # gpt-4o-mini allows up to 16,384; reasoning models (fallback) count hidden reasoning tokens here too
    llm_max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 4000))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))  # non-reasoning models
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"))

    # --- advanced RAG switches (all use the fast model; turned off automatically without an LLM)
    query_rewrite: bool = field(default_factory=lambda: _env_bool("QUERY_REWRITE", True))  # multi-query + Tanglish
    doc_context: bool = field(default_factory=lambda: _env_bool("DOC_CONTEXT", True))  # contextual chunk headers
    followups: bool = field(default_factory=lambda: _env_bool("FOLLOWUPS", True))
    # optional JSON overriding the price table, e.g. {"gpt-4o-mini": [0.15, 0.6]}  (USD per 1M in/out tokens)
    pricing_json: str = field(default_factory=lambda: _env("PRICING_JSON", ""))

    # --- ingestion extras --------------------------------------------------------
    ocr_enabled: str = field(default_factory=lambda: _env("OCR_ENABLED", "auto"))  # auto | true | false
    ocr_languages: str = field(default_factory=lambda: _env("OCR_LANGUAGES", "tam+eng"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 200))
    max_table_rows: int = field(default_factory=lambda: _env_int("MAX_TABLE_ROWS", 20000))

    # --- crawler -------------------------------------------------------------
    http_timeout: float = field(default_factory=lambda: _env_float("HTTP_TIMEOUT", 60))
    crawl_concurrency: int = field(default_factory=lambda: _env_int("CRAWL_CONCURRENCY", 4))
    # Several *.tn.gov.in sites serve incomplete TLS chains. Leave this False unless you
    # understand the risk; when True, a download that fails TLS verification is retried
    # once without verification and flagged in the metadata.
    allow_insecure_ssl_fallback: bool = field(default_factory=lambda: _env_bool("ALLOW_INSECURE_SSL_FALLBACK", False))

    # --- server --------------------------------------------------------------
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))
    # When set, ingestion / delete / reset endpoints require header  X-Admin-Token: <value>
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
    # Per-IP limit for /api/ask and /api/search (protects your OpenAI credit on a public demo). 0 = off.
    rate_limit_per_min: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MIN", 30))
    warmup_models: bool = field(default_factory=lambda: _env_bool("WARMUP_MODELS", True))

    # derived paths
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def store_dir(self) -> Path:
        if self.vector_store_dir:
            p = Path(self.vector_store_dir).expanduser()
            return (p if p.is_absolute() else BACKEND_DIR / p).resolve()
        return self.data_dir / "vectorstore"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.raw_dir, self.uploads_dir, self.store_dir, self.sources_dir):
            p.mkdir(parents=True, exist_ok=True)

    # derived -------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider in ("openai", "openai-compatible", "ollama", "anthropic") and (
            self.llm_provider in ("ollama", "openai-compatible") or bool(self.openai_api_key or self.anthropic_api_key))

    @property
    def helper_llm_enabled(self) -> bool:
        """Query planning / LLM rerank / doc context need an OpenAI-compatible endpoint."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return self.llm_provider in ("openai-compatible", "ollama")

    def answer_model(self) -> str:
        return {"ollama": self.ollama_model, "anthropic": self.anthropic_model}.get(self.llm_provider,
                                                                                   self.openai_chat_model)

    def fast_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.openai_fast_model

    def embedding_label(self) -> str:
        if self.embedding_backend == "openai":
            return f"{self.openai_embedding_model} ({self.openai_embedding_dimensions}d)"
        if self.embedding_backend == "hash":
            return "hash-768 (offline test mode)"
        return self.embedding_model

    def public_dict(self) -> dict:
        """Settings safe to show in the UI (no secrets)."""
        return {
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_label(),
            "reranker_backend": self.reranker_backend if self.reranker_enabled else "none",
            "reranker_model": self.fast_model() if self.reranker_backend == "llm" else self.reranker_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "rerank_candidates": self.rerank_candidates,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "max_chunks_per_doc": self.max_chunks_per_doc,
            "llm_provider": self.llm_provider,
            "llm_model": self.answer_model(),
            "fast_model": self.fast_model(),
            "fallback_model": self.openai_fallback_model or None,
            "reasoning_effort": self.openai_reasoning_effort,
            "vector_store_dir": str(self.store_dir),
            "query_rewrite": self.query_rewrite and self.helper_llm_enabled,
            "doc_context": self.doc_context and self.helper_llm_enabled,
            "followups": self.followups and self.helper_llm_enabled,
            "ocr_enabled": self.ocr_enabled,
            "admin_protected": bool(self.admin_token),
        }


settings = Settings()
