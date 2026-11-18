"""Pipeline 1 · stage 2 — SPLIT.

A recursive, separator-aware character splitter that works on *offsets*, so every
chunk knows which page / sheet / slide range it came from. Sections of a document
are concatenated first, which avoids the tiny-chunk problem with short pages.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from .loaders import Section

# strongest boundary first. Tamil prose uses '.', and '।' appears in some docs.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "। ", "; ", ": ", ", ", " "]


@dataclass
class Chunk:
    text: str
    index: int
    meta: dict


def _split_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    n = len(text)
    if n <= size:
        return [(0, n)]
    spans = []
    start = 0
    min_cut = int(size * 0.55)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = -1
            for sep in SEPARATORS:
                pos = window.rfind(sep, min_cut)
                if pos != -1:
                    cut = start + pos + len(sep)
                    break
            if cut != -1:
                end = cut
        spans.append((start, end))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        # move the overlap start forward to a word boundary
        ws = text.find(" ", nxt, end)
        if ws != -1 and ws + 1 < end:
            nxt = ws + 1
        start = nxt
    return spans


def _loc_label(meta: dict, key: str, a, b) -> dict:
    if a is None:
        return {}
    if a == b:
        return {key: a}
    return {key: a, f"{key}_end": b}


def split_sections(sections: list[Section], chunk_size: int, chunk_overlap: int, min_chars: int = 40) -> list[Chunk]:
    if not sections:
        return []
    # concatenate, remembering where each section starts
    starts, metas, parts = [], [], []
    pos = 0
    for s in sections:
        starts.append(pos)
        metas.append(s.meta)
        parts.append(s.text)
        pos += len(s.text) + 2
    full = "\n\n".join(parts)

    chunks: list[Chunk] = []
    for a, b in _split_spans(full, chunk_size, chunk_overlap):
        text = full[a:b].strip()
        if len(text) < min_chars:
            continue
        i0 = bisect.bisect_right(starts, a) - 1
        i1 = bisect.bisect_right(starts, max(a, b - 1)) - 1
        m0, m1 = metas[max(i0, 0)], metas[max(i1, 0)]
        meta: dict = {}
        for key in ("page", "slide", "sheet"):
            if key in m0 or key in m1:
                meta.update(_loc_label(meta, key, m0.get(key, m1.get(key)), m1.get(key, m0.get(key))))
        if m0.get("heading"):
            meta["heading"] = m0["heading"]
        chunks.append(Chunk(text=text, index=len(chunks), meta=meta))
    return chunks
