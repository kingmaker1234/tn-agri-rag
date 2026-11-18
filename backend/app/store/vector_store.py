"""FAISS vector index + SQLite metadata store, saved as plain files on this computer.

Files written to VECTOR_STORE_DIR (default backend/data/vectorstore/):
    index.faiss   – IndexIDMap2(IndexFlatIP) over L2-normalised vectors (cosine)
    meta.sqlite   – documents + chunks (text, page/sheet/slide, language …)
    store.json    – embedding model name, dimension, timestamps

FAISS ids == SQLite chunk ids, so a search hit maps straight to its text.

Durability: index.faiss is written to a temp file and atomically renamed (a crash never leaves a
half-written index), through Python file I/O so paths with spaces or Tamil characters work on
Windows too. On start-up the two files are reconciled: documents whose vectors never reached disk
(e.g. the process was killed mid-ingest) are dropped from the metadata so the next run re-indexes them.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from .bm25 import BM25Index

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT, title TEXT, file_type TEXT, category TEXT, language TEXT,
    source_url TEXT, source_id TEXT, stored_path TEXT, sha256 TEXT UNIQUE,
    size_bytes INTEGER, n_pages INTEGER, n_chunks INTEGER, extraction TEXT,
    notes TEXT, added_at TEXT, job_id TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER, text TEXT, meta TEXT, language TEXT, n_chars INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(file_type);
"""
# columns added in v2 (document context); migrated in place for older indexes
EXTRA_DOC_COLUMNS = {"summary": "TEXT", "year": "TEXT", "doc_type": "TEXT", "department": "TEXT",
                     "schemes": "TEXT", "embed_tokens": "INTEGER", "cost_usd": "REAL"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ModelMismatchError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, store_dir: Path):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.info_path = self.dir / "store.json"
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "meta.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        have = {r[1] for r in self.db.execute("PRAGMA table_info(documents)")}
        for col, typ in EXTRA_DOC_COLUMNS.items():
            if col not in have:
                self.db.execute(f"ALTER TABLE documents ADD COLUMN {col} {typ}")
        self.db.commit()
        self.info: dict = json.loads(self.info_path.read_text(encoding="utf-8")) if self.info_path.exists() else {}
        self.index: faiss.Index | None = self._read_index()
        self.repaired = self.reconcile()
        self.bm25 = BM25Index()
        self.rebuild_bm25()
        self._dirty = False
        log.info("vector store: %s (%d vectors, %d chunks)", self.dir,
                 self.index.ntotal if self.index is not None else 0,
                 self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # ------------------------------------------------------------------ persistence
    def _read_index(self) -> faiss.Index | None:
        if not self.index_path.exists():
            return None
        data = np.frombuffer(self.index_path.read_bytes(), dtype=np.uint8)
        return faiss.deserialize_index(data)

    def reconcile(self) -> dict:
        """Make meta.sqlite and index.faiss agree after an unclean shutdown."""
        rows = self.db.execute("SELECT id, doc_id FROM chunks").fetchall()
        chunk_ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        index_ids = (faiss.vector_to_array(self.index.id_map).astype(np.int64) if self.index is not None
                     else np.empty(0, np.int64))
        missing = np.setdiff1d(chunk_ids, index_ids)  # text saved, vector never written
        orphans = np.setdiff1d(index_ids, chunk_ids)  # vector without text
        out = {"orphan_vectors": int(orphans.size), "documents_dropped": 0}
        if orphans.size:
            self.index.remove_ids(orphans)
        if missing.size:
            miss = set(missing.tolist())
            bad_docs = sorted({d for i, d in rows if i in miss})
            bad = set(bad_docs)
            bad_ids = np.fromiter((i for i, d in rows if d in bad), dtype=np.int64)
            if self.index is not None and bad_ids.size:
                self.index.remove_ids(bad_ids)
            for d in bad_docs:
                self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (d,))
                self.db.execute("DELETE FROM documents WHERE doc_id = ?", (d,))
            self.db.commit()
            out["documents_dropped"] = len(bad_docs)
        if orphans.size or missing.size:
            log.warning("vector store repaired after an unclean shutdown: %s (re-run ingestion to re-index "
                        "dropped documents)", out)
            self.save()
        return out

    # ------------------------------------------------------------------ index
    def check_model(self, model_id: str, dim: int | None = None) -> None:
        stored = self.info.get("embedding_model")
        if stored and stored != model_id and self.index is not None and self.index.ntotal > 0:
            raise ModelMismatchError(
                f"Index was built with '{stored}' but the server is configured with '{model_id}'. "
                "Re-index (System → Reset index) or switch EMBEDDING_MODEL back.")
        if dim and self.info.get("dim") and self.info["dim"] != dim and self.index is not None and self.index.ntotal > 0:
            raise ModelMismatchError(f"Vector dimension changed ({self.info['dim']} → {dim}). Reset the index.")

    def _ensure_index(self, dim: int, model_id: str) -> None:
        if self.index is None or self.index.ntotal == 0 and self.index.d != dim:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            self.info.update({"embedding_model": model_id, "dim": dim, "created_at": self.info.get("created_at") or _now()})

    # -------------------------------------------------------------- documents
    def doc_by_sha(self, sha: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM documents WHERE sha256 = ?", (sha,)).fetchone()

    def add_document(self, doc: dict, chunks: list[dict], vectors: np.ndarray, model_id: str) -> int:
        """doc: documents columns; chunks: [{text, meta, language}]; vectors: (n, dim)."""
        assert len(chunks) == len(vectors)
        with self.lock:
            self.check_model(model_id, vectors.shape[1] if len(vectors) else None)
            if len(vectors):
                self._ensure_index(vectors.shape[1], model_id)
            cols = ["doc_id", "filename", "title", "file_type", "category", "language", "source_url", "source_id",
                    "stored_path", "sha256", "size_bytes", "n_pages", "n_chunks", "extraction", "notes", "added_at",
                    "job_id", *EXTRA_DOC_COLUMNS]
            doc = {**doc, "n_chunks": len(chunks), "added_at": doc.get("added_at") or _now()}
            self.db.execute(f"INSERT INTO documents ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            [doc.get(c) for c in cols])
            ids = []
            for c in chunks:
                cur = self.db.execute(
                    "INSERT INTO chunks (doc_id, chunk_index, text, meta, language, n_chars) VALUES (?,?,?,?,?,?)",
                    (doc["doc_id"], c["index"], c["text"], json.dumps(c.get("meta") or {}, ensure_ascii=False),
                     c.get("language"), len(c["text"])))
                ids.append(cur.lastrowid)
            self.db.commit()
            if len(vectors):
                self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, np.int64))
            self._dirty = True
            return len(ids)

    def delete_document(self, doc_id: str) -> bool:
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            row = self.db.execute("SELECT stored_path FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not row:
                return False
            if ids and self.index is not None:
                self.index.remove_ids(np.asarray(ids, np.int64))
            self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self.db.commit()
            self._dirty = True
            self.save()
            self.rebuild_bm25()
            return True

    def reset(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM documents")
            self.db.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
            self.db.commit()
            self.index = None
            self.info = {}
            for p in (self.index_path, self.info_path):
                if p.exists():
                    p.unlink()
            self.bm25 = BM25Index()

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(6):  # Windows: antivirus / indexer may hold the old file for a moment
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.2 * (attempt + 1))

    def save(self) -> None:
        """Write index.faiss + store.json to disk (atomic; safe to call often)."""
        with self.lock:
            if self.index is not None:
                self._atomic_write(self.index_path, faiss.serialize_index(self.index).tobytes())
            self.info["updated_at"] = _now()
            self.info["n_vectors"] = int(self.index.ntotal) if self.index is not None else 0
            self._atomic_write(self.info_path, json.dumps(self.info, indent=2).encode("utf-8"))
            self._dirty = False

    def rebuild_bm25(self) -> None:
        with self.lock:
            rows = self.db.execute(
                "SELECT c.id, COALESCE(d.title, d.filename) AS title, COALESCE(d.year, '') AS year, "
                "COALESCE(d.summary, '') AS summary, c.text FROM chunks c "
                "JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.id").fetchall()
            bm = BM25Index()
            # contextual BM25: title + year + document summary travel with every chunk
            bm.build([r["id"] for r in rows], [f"{r['title']} {r['year']}\n{r['summary']}\n{r['text']}" for r in rows])
            self.bm25 = bm

    # ---------------------------------------------------------------- search
    def allowed_ids(self, filters: dict | None) -> np.ndarray | None:
        if not filters:
            return None
        clauses, args = [], []
        for col, key in (("d.file_type", "file_types"), ("d.category", "categories"), ("c.language", "languages"),
                         ("d.doc_id", "doc_ids")):
            vals = [v for v in (filters.get(key) or []) if v]
            if vals:
                clauses.append(f"{col} IN ({','.join('?' * len(vals))})")
                args.extend(vals)
        if not clauses:
            return None
        rows = self.db.execute(
            f"SELECT c.id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE {' AND '.join(clauses)}", args)
        return np.asarray([r[0] for r in rows], dtype=np.int64)

    def dense_search(self, qvec: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            if allowed is not None and allowed.size == 0:
                return []
            q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
            k = min(k, self.index.ntotal)
            if allowed is not None:
                try:
                    params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
                    D, I = self.index.search(q, k, params=params)
                except Exception:  # older faiss: post-filter
                    D, I = self.index.search(q, min(self.index.ntotal, k * 20))
                    keep = np.isin(I[0], allowed)
                    D, I = D[:, keep][:, :k], I[:, keep][:, :k]
            else:
                D, I = self.index.search(q, k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    def bm25_search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        return self.bm25.search(query, k, allowed)

    # ------------------------------------------------------------- read APIs
    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        q = ("SELECT c.id, c.doc_id, c.chunk_index, c.text, c.meta, c.language, d.title, d.filename, d.file_type, "
             "d.category, d.source_url, d.n_chunks, d.year, d.doc_type, d.summary FROM chunks c "
             "JOIN documents d ON d.doc_id = c.doc_id "
             f"WHERE c.id IN ({','.join('?' * len(ids))})")
        out = {}
        for r in self.db.execute(q, ids):
            d = dict(r)
            d["meta"] = json.loads(d["meta"] or "{}")
            out[d["id"]] = d
        return out

    def neighbor_text(self, doc_id: str, chunk_index: int) -> tuple[str, str]:
        rows = self.db.execute(
            "SELECT chunk_index, text FROM chunks WHERE doc_id = ? AND chunk_index IN (?, ?)",
            (doc_id, chunk_index - 1, chunk_index + 1)).fetchall()
        prev = next((r["text"] for r in rows if r["chunk_index"] == chunk_index - 1), "")
        nxt = next((r["text"] for r in rows if r["chunk_index"] == chunk_index + 1), "")
        return prev, nxt

    def list_documents(self, q: str = "", file_type: str = "", category: str = "", language: str = "",
                       limit: int = 50, offset: int = 0, sort: str = "added_at") -> dict:
        clauses, args = [], []
        if q:
            clauses.append("(title LIKE ? OR filename LIKE ? OR source_url LIKE ? OR summary LIKE ? OR schemes LIKE ?)")
            args += [f"%{q}%"] * 5
        for col, val in (("file_type", file_type), ("category", category), ("language", language)):
            if val:
                clauses.append(f"{col} = ?")
                args.append(val)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = {"added_at": "added_at DESC", "title": "COALESCE(title, filename) COLLATE NOCASE",
                 "year": "COALESCE(year, '') DESC",
                 "chunks": "n_chunks DESC", "size": "size_bytes DESC"}.get(sort, "added_at DESC")
        total = self.db.execute(f"SELECT COUNT(*) FROM documents {where}", args).fetchone()[0]
        rows = self.db.execute(f"SELECT * FROM documents {where} ORDER BY {order} LIMIT ? OFFSET ?",
                               args + [limit, offset]).fetchall()
        return {"total": total, "items": [self._doc_out(r) for r in rows]}

    @staticmethod
    def _doc_out(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["notes"] = json.loads(d["notes"]) if d.get("notes") else []
        d["schemes"] = json.loads(d["schemes"]) if d.get("schemes") else []
        d.pop("stored_path", None)
        return d

    def get_document(self, doc_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._doc_out(r) if r else None

    def document_path(self, doc_id: str) -> Path | None:
        r = self.db.execute("SELECT stored_path FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return Path(r[0]) if r and r[0] else None

    def document_chunks(self, doc_id: str, limit: int = 20, offset: int = 0, q: str = "") -> dict:
        args: list = [doc_id]
        extra = ""
        if q:
            extra = " AND text LIKE ?"
            args.append(f"%{q}%")
        total = self.db.execute(f"SELECT COUNT(*) FROM chunks WHERE doc_id = ?{extra}", args).fetchone()[0]
        rows = self.db.execute(
            f"SELECT id, chunk_index, text, meta, language, n_chars FROM chunks WHERE doc_id = ?{extra} "
            "ORDER BY chunk_index LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"] or "{}")
            items.append(d)
        return {"total": total, "items": items}

    def stats(self) -> dict:
        db = self.db

        def grouped(sql):
            return {(r[0] or "unknown"): r[1] for r in db.execute(sql)}

        n_docs = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        size = self.index_path.stat().st_size if self.index_path.exists() else 0
        return {
            "store_dir": str(self.dir),
            "documents": n_docs,
            "chunks": n_chunks,
            "vectors": int(self.index.ntotal) if self.index is not None else 0,
            "bm25_terms": len(self.bm25.vocab),
            "index_bytes": size,
            "embedding_model": self.info.get("embedding_model"),
            "dim": self.info.get("dim"),
            "updated_at": self.info.get("updated_at"),
            "by_type": grouped("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type ORDER BY 2 DESC"),
            "by_category": grouped("SELECT category, COUNT(*) FROM documents GROUP BY category ORDER BY 2 DESC"),
            "by_language": grouped("SELECT language, COUNT(*) FROM chunks GROUP BY language ORDER BY 2 DESC"),
            "total_bytes": db.execute("SELECT COALESCE(SUM(size_bytes),0) FROM documents").fetchone()[0],
            "by_year": grouped("SELECT year, COUNT(*) FROM documents WHERE year IS NOT NULL AND year != '' "
                               "GROUP BY year ORDER BY year DESC LIMIT 12"),
            "ingest_cost_usd": round(db.execute("SELECT COALESCE(SUM(cost_usd),0) FROM documents").fetchone()[0], 4),
        }

    def known_source_urls(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT source_url FROM documents WHERE source_url IS NOT NULL")}


_store: VectorStore | None = None
_slock = threading.Lock()


def get_store() -> VectorStore:
    global _store
    if _store is None:
        with _slock:
            if _store is None:
                from ..config import settings

                settings.ensure_dirs()
                _store = VectorStore(settings.store_dir)
    return _store
