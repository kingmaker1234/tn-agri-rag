"""Query understanding (fast model, structured output).

Farmers ask in Tamil, English or Tanglish ("drip irrigation ku evlo subsidy kidaikum?").
Tanglish is poorly served by both BM25 and embeddings, and follow-ups ("documents enna venum?")
lose their subject. One cheap call turns the question into:

  standalone_question  – follow-up resolved against the conversation
  english_query        – clean English search query
  tamil_query          – the same in Tamil script (matches Tamil documents)
  keywords             – exact identifiers worth keyword-matching (PMFBY, G.O. Ms. No. 65, 2025-26)
  language             – what the user wrote in, so the answer can reply in kind

Retrieval then runs every variant through BM25 and FAISS and fuses all lists (RAG-Fusion).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from ..config import settings
from ..llm import openai_client
from ..llm.usage import Usage
from ..text_utils import detect_language

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["ta", "en", "tanglish", "mixed"]},
        "standalone_question": {"type": "string"},
        "english_query": {"type": "string"},
        "tamil_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "years": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["language", "standalone_question", "english_query", "tamil_query", "keywords", "years"],
    "additionalProperties": False,
}

SYSTEM = """You turn questions about Tamil Nadu agriculture and farmer-welfare schemes into search queries \
for a document index (government policy notes, budget speeches, citizen charters, G.O.s, guidelines, news).
Do NOT answer the question. Reply with JSON only.

- language: "ta" (Tamil script), "en", "tanglish" (Tamil written in Latin letters, e.g. "evlo subsidy kidaikum" = \
how much subsidy is available, "yaaru apply pannalam" = who can apply), or "mixed".
- standalone_question: the question made self-contained using the conversation (resolve "it", "that scheme", \
"adhu", follow-ups). Keep the user's language.
- english_query: a precise English search query. Expand known acronyms once \
(PMFBY = Pradhan Mantri Fasal Bima Yojana, KAVIADP = Kalaignarin All Village Integrated Agriculture Development \
Programme, SMAM = Sub-Mission on Agricultural Mechanisation, PM-KUSUM = solar pumps, MIDH/NHM = horticulture mission).
- tamil_query: the same query in Tamil script using the words government documents use \
(e.g. மானியம், நுண்ணீர் பாசனம், பயிர் காப்பீடு, உழவர் சந்தை).
- keywords: up to 6 exact strings worth keyword search: scheme names, acronyms, G.O. numbers, crop names, \
amounts, district names. No generic words.
- years: fiscal years or years mentioned or implied ("latest", "this year" → the newest you know is 2026-27); else []."""


_cache: dict[str, dict] = {}


def fallback_plan(question: str) -> dict:
    lang = detect_language(question)
    return {"language": lang if lang in ("ta", "en", "mixed") else "en", "standalone_question": question,
            "english_query": question if lang != "ta" else "", "tamil_query": question if lang == "ta" else "",
            "keywords": [], "years": [], "source": "fallback"}


def plan_query(question: str, history: list[dict] | None, usage: Usage) -> dict:
    if not (settings.query_rewrite and settings.helper_llm_enabled):
        return fallback_plan(question)
    hist = [{"role": h["role"], "content": h["content"][:600]} for h in (history or [])[-4:] if h.get("content")]
    key = hashlib.sha1(json.dumps([question, hist], ensure_ascii=False).encode()).hexdigest()
    if key in _cache:
        return {**_cache[key], "source": "cache"}
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in hist)
    user = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Question: {question}"
    t0 = time.perf_counter()
    try:
        plan = openai_client.complete_json(settings.fast_model(), SYSTEM, user, SCHEMA, "query_plan", usage,
                                           "plan_query", max_output_tokens=2500,
                                           effort=settings.openai_fast_reasoning_effort)
        plan["keywords"] = [k for k in plan.get("keywords", []) if k and len(k) < 80][:6]
        plan["source"] = "llm"
        plan["ms"] = round((time.perf_counter() - t0) * 1000)
        if len(_cache) > 1000:
            _cache.clear()
        _cache[key] = plan
        return plan
    except Exception as e:  # never block retrieval on the planner
        log.warning("query planning failed: %s", e)
        p = fallback_plan(question)
        p["error"] = str(e)[:200]
        return p


def variants(question: str, plan: dict) -> list[tuple[str, str]]:
    """(label, text) pairs to retrieve with, de-duplicated, original first."""
    out: list[tuple[str, str]] = [("original", question)]
    for label in ("standalone_question", "english_query", "tamil_query"):
        t = (plan.get(label) or "").strip()
        if t and all(t.lower() != o[1].lower() for o in out):
            out.append((label.replace("_question", "").replace("_query", ""), t))
    return out
