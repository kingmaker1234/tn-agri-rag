"""Sparse BM25 (Okapi) over chunk texts, built with scipy so scoring the whole
corpus for one query is a single sparse mat-vec (milliseconds for 100k chunks)."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from ..text_utils import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = np.zeros(0, dtype=np.int64)  # row -> chunk id
        self.vocab: dict[str, int] = {}
        self.W: sparse.csc_matrix | None = None
        self.idf: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def build(self, ids: list[int], texts: list[str]) -> None:
        vocab: dict[str, int] = {}
        indptr, indices, data, dls = [0], [], [], []
        for t in texts:
            c = Counter(tokenize(t))
            for tok, n in c.items():
                j = vocab.setdefault(tok, len(vocab))
                indices.append(j)
                data.append(n)
            indptr.append(len(indices))
            dls.append(sum(c.values()))
        n_docs = len(texts)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.vocab = vocab
        if n_docs == 0 or not vocab:
            self.W, self.idf = None, None
            return
        tf = sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices), np.asarray(indptr)),
                               shape=(n_docs, len(vocab)))
        df = np.bincount(tf.indices, minlength=len(vocab)).astype(np.float32)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(dls, np.float32)
        avgdl = float(dl.mean()) or 1.0
        norm = self.k1 * (1 - self.b + self.b * dl / avgdl)
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def search(self, query: str, k: int = 50, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        if self.W is None:
            return []
        q = Counter(t for t in tokenize(query) if t in self.vocab)
        if not q:
            return []
        cols = [self.vocab[t] for t in q]
        weights = np.asarray([q[t] for t in q], np.float32)
        scores = np.asarray(self.W[:, cols] @ weights).ravel()
        if allowed is not None:
            mask = np.isin(self.ids, allowed)
            scores = np.where(mask, scores, 0.0)
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return []
        top = nz[np.argsort(-scores[nz])[:k]]
        return [(int(self.ids[i]), float(scores[i])) for i in top]

    def matched_terms(self, query: str, text: str) -> list[str]:
        """Surface strings to highlight: exact token matches, plus the 5-letter stem
        for Tamil words that only matched in an inflected form."""
        doc = set(tokenize(text))
        out = set()
        for t in tokenize(query):
            if t.startswith("ta~"):
                if t in doc:
                    out.add(t[3:])
            elif t in doc:
                out.add(t)
        # drop stems already covered by a full-word match
        return sorted(o for o in out if not any(o != p and p.startswith(o) for p in out))
