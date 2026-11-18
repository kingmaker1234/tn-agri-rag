"""Embedding backends.

* OpenAIEmbedder – default. text-embedding-3-large (multilingual), batched over the API.
* SentenceTransformerEmbedder – real multilingual models (default BAAI/bge-m3,
  which handles Tamil, English and mixed text in one vector space).
* HashEmbedder – deterministic, dependency-free fallback used for offline tests
  and CI (EMBEDDING_BACKEND=hash). Not semantic, but lets every pipeline run.
"""
from __future__ import annotations

import logging
import threading
import zlib

import numpy as np

from ..config import settings
from ..text_utils import tokenize

log = logging.getLogger(__name__)


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Embedder:
    name: str = "base"
    dim: int = 0

    def embed_documents(self, texts: list[str], usage=None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed_queries(self, texts: list[str], usage=None) -> np.ndarray:
        return np.vstack([self.embed_query(t) for t in texts])

    def status(self) -> dict:
        return {"name": self.name, "dim": self.dim, "ready": True}


class HashEmbedder(Embedder):
    """Signed feature hashing of word tokens + character 3/4-grams (L2 normalised)."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = list(toks)
        low = text.lower()
        for n in (3, 4):
            feats.extend("c" + low[i:i + n] for i in range(0, max(len(low) - n + 1, 0)) if not low[i:i + n].isspace())
        for f in feats:
            h = zlib.crc32(f.encode("utf-8"))
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str], usage=None) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()
        self.device = resolve_device(settings.device)
        self.error: str | None = None
        low = model_name.lower()
        # model-specific instruction prefixes
        if "e5" in low:
            self.q_prefix, self.d_prefix = "query: ", "passage: "
        elif "bge" in low and ("-en" in low or "-zh" in low) and "m3" not in low:
            self.q_prefix, self.d_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self.q_prefix, self.d_prefix = "", ""
        self.dim = 0

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading embedding model %s on %s", self.name, self.device)
                    try:
                        m = SentenceTransformer(self.name, device=self.device)
                    except Exception as e:
                        self.error = str(e)
                        raise
                    if settings.embedding_max_seq_length:
                        m.max_seq_length = min(m.max_seq_length or 10**6, settings.embedding_max_seq_length)
                    getter = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
                    self.dim = int(getter())
                    self._model = m
        return self._model

    def embed_documents(self, texts: list[str], usage=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), np.float32)
        vecs = self.model.encode(
            [self.d_prefix + t for t in texts],
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        v = self.model.encode([self.q_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float32)

    def status(self) -> dict:
        return {"name": self.name, "dim": self.dim, "device": self.device, "ready": self._model is not None,
                "error": self.error}


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3 (multilingual, Tamil OK). Batched + parallel, normalised,
    query vectors cached. No GPU, no model download."""

    MAX_BATCH = 96            # inputs per request
    MAX_BATCH_CHARS = 120_000  # keeps each request far below the per-request token cap
    MAX_INPUT_CHARS = 6_000    # one input must stay under 8,191 tokens (Tamil ≈ 1 token/char worst case)

    def __init__(self):
        self.name = settings.openai_embedding_model
        self.dim = settings.openai_embedding_dimensions or (3072 if "large" in self.name else 1536)
        self.error: str | None = None
        self._qcache: dict[str, np.ndarray] = {}
        self.ready = False

    def _batches(self, texts: list[str]) -> list[list[str]]:
        out, cur, size = [], [], 0
        for t in texts:
            t = (t or " ")[: self.MAX_INPUT_CHARS]
            if cur and (len(cur) >= self.MAX_BATCH or size + len(t) > self.MAX_BATCH_CHARS):
                out.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += len(t)
        if cur:
            out.append(cur)
        return out

    def _embed(self, texts: list[str], usage=None, step: str = "embed") -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor

        from ..llm import openai_client

        batches = self._batches(texts)
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(batches) or 1)) as ex:
                parts = list(ex.map(lambda b: openai_client.embed(b, usage, step), batches))
        except Exception as e:
            self.error = str(e)
            raise
        self.error, self.ready = None, True
        vecs = np.asarray([v for part in parts for v in part], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: list[str], usage=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self._embed(texts, usage, "embed_documents")

    def embed_queries(self, texts: list[str], usage=None) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._qcache]
        if missing:
            for t, v in zip(missing, self._embed(missing, usage, "embed_query")):
                if len(self._qcache) > 2000:
                    self._qcache.clear()
                self._qcache[t] = v
        return np.vstack([self._qcache[t] for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def status(self) -> dict:
        err = self.error or (None if settings.openai_api_key else "OPENAI_API_KEY is not set")
        return {"name": self.name, "dim": self.dim, "device": "OpenAI API", "ready": not err, "error": err}


_embedder: Embedder | None = None
_elock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _elock:
            if _embedder is None:
                if settings.embedding_backend == "hash":
                    _embedder = HashEmbedder()
                elif settings.embedding_backend == "openai":
                    _embedder = OpenAIEmbedder()
                else:
                    _embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return _embedder


def embedder_identity() -> str:
    """String stored with the index so a model switch is detected."""
    if settings.embedding_backend == "hash":
        return "hash-768"
    if settings.embedding_backend == "openai":
        return f"openai:{settings.openai_embedding_model}@{OpenAIEmbedder().dim}"
    return settings.embedding_model
