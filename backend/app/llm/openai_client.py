"""One OpenAI client for everything: answers, fast helper calls and embeddings.

* api.openai.com → **Responses API** (OpenAI's recommended interface; `store=False`).
* any other base URL (Groq, Gemini, Ollama /v1, vLLM, LM Studio) → **Chat Completions**.

Default models are `gpt-4o-mini` (answers + helper calls) and `text-embedding-3-large`.

Parameters per model family:
* classic chat models (gpt-4o-mini, gpt-4.1 …) get `temperature` and no `reasoning`;
* reasoning models (o-series, gpt-5/6 …) get `reasoning.effort` and no `temperature`.
If a guess is wrong, nothing breaks: when the API answers 400 "unsupported parameter/value", that
parameter is dropped (or remapped) and the call is retried; the decision is cached per model.

Model retirement: gpt-4o-mini is deprecated by OpenAI (shutdown announced for 14 Jan 2027). If the
API reports that a model no longer exists, calls switch once to OPENAI_FALLBACK_MODEL (default
gpt-6-luna) and stay there; set OPENAI_FALLBACK_MODEL= (empty) to disable.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..config import settings
from .usage import Usage

log = logging.getLogger(__name__)

_client = None
_lock = threading.Lock()
_unsupported: dict[str, set[str]] = {}  # model -> params the API rejected
_redirect: dict[str, str] = {}  # retired model -> fallback model actually used
_REASONING_RE = re.compile(r"^(o\d|gpt-5|gpt-6|gpt-7|gpt-oss)", re.I)


class LLMNotConfigured(RuntimeError):
    pass


def base_url() -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_base_url.rstrip("/") + "/v1"
    return settings.openai_base_url


def api_style() -> str:
    if settings.openai_api_style in ("responses", "chat"):
        return settings.openai_api_style
    return "responses" if "api.openai.com" in base_url() and settings.llm_provider == "openai" else "chat"


def get_client():
    """Shared, thread-safe OpenAI SDK client (built-in retries with exponential backoff on 429/5xx)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                from openai import OpenAI

                key = settings.openai_api_key or ("ollama" if settings.llm_provider == "ollama" else "")
                if not key:
                    raise LLMNotConfigured("OPENAI_API_KEY is not set (backend/.env)")
                _client = OpenAI(api_key=key, base_url=base_url(), timeout=settings.openai_timeout,
                                 max_retries=settings.openai_max_retries,
                                 organization=settings.openai_organization or None)
    return _client


def reset_client() -> None:
    global _client
    _client = None
    _unsupported.clear()
    _redirect.clear()


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_RE.match(model.rsplit("/", 1)[-1]))


def resolve_model(model: str) -> str:
    """The model calls really go to (differs from the configured one only after a fallback)."""
    return _redirect.get(model, model)


def active_models() -> dict:
    return {"answer": resolve_model(settings.answer_model()), "fast": resolve_model(settings.fast_model()),
            "fallback_model": settings.openai_fallback_model or None,
            "fallback_active": bool(_redirect)}


def _model_gone(err: Exception) -> bool:
    """True when OpenAI says the model is retired / unknown / not available to this key."""
    status = getattr(err, "status_code", None)
    body = getattr(err, "body", None)
    code = ""
    if isinstance(body, dict):
        e = body.get("error") if isinstance(body.get("error"), dict) else body
        code = str(e.get("code") or "")
    if code in ("model_not_found", "model_deprecated", "model_decommissioned", "model_retired"):
        return True
    msg = str(err).lower()
    return status in (400, 403, 404) and "model" in msg and any(
        w in msg for w in ("does not exist", "not found", "deprecated", "decommissioned", "shut down",
                           "no longer available", "retired", "do not have access"))


def _with_fallback(model: str, attempt: Callable[[str], Any]) -> tuple[str, Any]:
    m = resolve_model(model)
    try:
        return m, attempt(m)
    except Exception as e:
        fb = (settings.openai_fallback_model or "").strip()
        if fb and m != fb and settings.llm_provider == "openai" and _model_gone(e):
            log.warning("model %s is not available (%s) — switching to fallback model %s", m, e, fb)
            _redirect[model] = fb
            return fb, attempt(fb)
        raise


def _responses_params(model: str, system: str, input_: list[dict], max_output_tokens: int, effort: str | None,
                      temperature: float | None) -> dict[str, Any]:
    params: dict[str, Any] = dict(model=model, instructions=system, input=input_, store=False,
                                  max_output_tokens=max_output_tokens)
    if is_reasoning_model(model):
        if effort:
            params["reasoning"] = {"effort": effort}
    elif temperature is not None:
        params["temperature"] = temperature
    return params


def _chat_params(model: str, messages: list[dict], max_output_tokens: int, effort: str | None,
                 temperature: float) -> dict[str, Any]:
    params: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_output_tokens,
                                  temperature=temperature)
    if effort and settings.llm_provider == "openai" and is_reasoning_model(model):
        params["reasoning_effort"] = effort
    return params


# ------------------------------------------------------------------ adaptive parameters
_PARAM_RE = re.compile(r"(?:parameter|param)[:\s]+'?\"?([a-z_.]+)", re.I)


def _rejected_param(err: Exception) -> str | None:
    """Return the parameter an OpenAI 400 error complains about, if any."""
    msg = str(getattr(err, "message", "") or err)
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        errobj = body.get("error") if isinstance(body.get("error"), dict) else body
        if errobj.get("param"):
            return str(errobj["param"])
    low = msg.lower()
    for name in ("max_completion_tokens", "max_tokens", "max_output_tokens", "temperature", "reasoning_effort",
                 "reasoning.effort", "reasoning", "response_format", "text.format", "text", "store", "top_p"):
        if name in low and ("unsupported" in low or "not supported" in low or "invalid" in low
                            or "unrecognized" in low or "unknown" in low):
            return name
    m = _PARAM_RE.search(msg)
    return m.group(1) if m else None


def _drop(params: dict, model: str, name: str) -> bool:
    """Remove/remap a rejected parameter. Returns True if params changed."""
    name = name.split(".")[0] if name.startswith("text.") else name
    if name == "max_tokens" and "max_tokens" in params:  # chat: legacy → new name
        params["max_completion_tokens"] = params.pop("max_tokens")
    elif name == "max_completion_tokens" and "max_completion_tokens" in params:  # old servers
        params["max_tokens"] = params.pop("max_completion_tokens")
    elif name in ("reasoning.effort", "reasoning") and "reasoning" in params:
        params.pop("reasoning")
    elif name in params:
        params.pop(name)
    else:
        return False
    _unsupported.setdefault(model, set()).add(name)
    return True


def _call(fn: Callable[..., Any], model: str, params: dict) -> Any:
    from openai import BadRequestError

    for key in list(_unsupported.get(model, ())):
        _drop(params, model, key)
    for _ in range(4):
        try:
            return fn(**params)
        except BadRequestError as e:
            name = _rejected_param(e)
            if not name or not _drop(params, model, name):
                raise
            log.info("model %s rejected '%s' — retrying without it", model, name)
    return fn(**params)


# ------------------------------------------------------------------ text generation
@dataclass
class StreamResult:
    usage: Usage = field(default_factory=Usage)
    finish: str | None = None
    model: str | None = None


def stream_text(model: str, system: str, messages: list[dict], max_output_tokens: int, effort: str | None,
                result: StreamResult, step: str = "answer") -> Iterator[str]:
    """Yield answer text deltas. Token usage and the model really used are written into `result`."""
    client = get_client()
    if api_style() == "responses":
        def start(m: str):
            params = _responses_params(m, system, messages, max_output_tokens, effort, settings.llm_temperature)
            params["stream"] = True
            return _call(client.responses.create, m, params)

        used, stream = _with_fallback(model, start)
        result.model = used
        for ev in stream:
            t = getattr(ev, "type", "")
            if t == "response.output_text.delta":
                yield ev.delta
            elif t in ("response.completed", "response.incomplete"):
                resp = ev.response
                result.usage.add(step, used, *_resp_tokens(resp))
                result.finish = getattr(getattr(resp, "incomplete_details", None), "reason", None) or "stop"
            elif t in ("error", "response.failed"):
                raise RuntimeError(getattr(ev, "message", None) or str(getattr(ev, "response", ev)))
    else:
        def start(m: str):
            params = _chat_params(m, [{"role": "system", "content": system}, *messages], max_output_tokens, effort,
                                  settings.llm_temperature)
            params.update(stream=True, stream_options={"include_usage": True})
            return _call(client.chat.completions.create, m, params)

        used, stream = _with_fallback(model, start)
        result.model = used
        for chunk in stream:
            if getattr(chunk, "usage", None):
                u = chunk.usage
                result.usage.add(step, used, u.prompt_tokens or 0, u.completion_tokens or 0)
            for ch in chunk.choices or []:
                piece = getattr(ch.delta, "content", None)
                if piece:
                    yield piece
                if ch.finish_reason:
                    result.finish = ch.finish_reason


def _resp_tokens(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0
    return int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)


def complete_json(model: str, system: str, user: str, schema: dict, name: str, usage: Usage, step: str,
                  max_output_tokens: int = 3000, effort: str | None = None) -> dict:
    """Structured output (strict JSON schema). Falls back to JSON mode / plain parsing if needed."""
    client = get_client()
    t0 = time.perf_counter()
    if api_style() == "responses":
        def run(m: str):
            params = _responses_params(m, system, [{"role": "user", "content": user}], max_output_tokens, effort, 0)
            params["text"] = {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}
            return _call(client.responses.create, m, params)

        used, resp = _with_fallback(model, run)
        usage.add(step, used, *_resp_tokens(resp), ms=(time.perf_counter() - t0) * 1000)
        text = resp.output_text or ""
    else:
        def run(m: str):
            params = _chat_params(m, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                  max_output_tokens, effort, 0)
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": name, "schema": schema, "strict": True}}
            try:
                return _call(client.chat.completions.create, m, params)
            except Exception as e:  # provider without json_schema support → JSON mode
                if _model_gone(e):
                    raise
                log.info("json_schema not supported (%s); using json_object", e)
                params["response_format"] = {"type": "json_object"}
                params["messages"][0]["content"] += "\nReply with JSON only, matching: " + json.dumps(schema)
                return _call(client.chat.completions.create, m, params)

        used, resp = _with_fallback(model, run)
        u = resp.usage
        usage.add(step, used, (u.prompt_tokens if u else 0) or 0, (u.completion_tokens if u else 0) or 0,
                  ms=(time.perf_counter() - t0) * 1000)
        text = resp.choices[0].message.content or ""
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ------------------------------------------------------------------ embeddings
def embed(texts: list[str], usage: Usage | None = None, step: str = "embed") -> list[list[float]]:
    client = get_client()
    model = settings.openai_embedding_model
    params: dict[str, Any] = {"model": model, "input": texts, "encoding_format": "float"}
    if settings.openai_embedding_dimensions and model.startswith("text-embedding-3"):
        params["dimensions"] = settings.openai_embedding_dimensions
    t0 = time.perf_counter()
    resp = _call(client.embeddings.create, model, params)
    if usage is not None:
        usage.add(step, model, resp.usage.prompt_tokens if resp.usage else 0, 0, ms=(time.perf_counter() - t0) * 1000)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def ping() -> dict:
    """Cheap connectivity check used by the System page."""
    t0 = time.perf_counter()
    usage = Usage()
    out = {"base_url": base_url(), "api_style": api_style()}
    try:
        vec = embed(["ping"], usage) if settings.embedding_backend == "openai" else None
        out["embeddings"] = f"ok ({len(vec[0])} dims)" if vec else "not used"
    except Exception as e:
        out["embeddings"] = f"error: {e}"
    try:
        r = complete_json(settings.fast_model(), "Reply with JSON.", 'Return {"ok": true}.',
                          {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                           "additionalProperties": False}, "ping", usage, "ping", max_output_tokens=400,
                          effort=settings.openai_fast_reasoning_effort)
        used = resolve_model(settings.fast_model())
        out["fast_model"] = (f"ok ({used})" if r.get("ok") else f"unexpected reply {r}")
    except Exception as e:
        out["fast_model"] = f"error: {e}"
    out.update(active_models())
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    out["cost_usd"] = round(usage.cost, 6)
    return out
