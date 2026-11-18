"""A small OpenAI-compatible server for offline tests and CI.

Implements the wire formats the official `openai` SDK parses:
  POST /v1/embeddings           (dimensions honoured, usage returned)
  POST /v1/responses            (JSON-schema output + SSE streaming with response.output_text.delta)
  POST /v1/chat/completions     (json_schema response_format + SSE streaming)
It mimics OpenAI's per-family rules with the same 400/404 error shapes:
  * classic models (gpt-4o-mini, gpt-4.1 …) reject `reasoning` / `reasoning_effort`;
  * reasoning models (o-series, gpt-5/6 …) reject `temperature` and legacy `max_tokens`;
  * models in RETIRED answer 404 `model_not_found` (exercises the fallback model).
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from app.store.embeddings import HashEmbedder

_HASH = HashEmbedder(3072)
RETIRED = {"gpt-retired-mini"}
_REASONING = re.compile(r"^(o\d|gpt-5|gpt-6|gpt-7|gpt-oss)")


def _is_reasoning(model: str) -> bool:
    return bool(_REASONING.match(model))


def _gone(model: str) -> dict:
    return {"error": {"message": f"The model `{model}` does not exist or you do not have access to it.",
                      "type": "invalid_request_error", "param": None, "code": "model_not_found"}}
TANGLISH = {"evlo": "how much", "kidaikum": "is available", "yaaru": "who", "apply pannalam": "can apply",
            "ku": "for", "enna": "what", "venum": "is needed"}


class Recorder:
    def __init__(self):
        self.requests: list[tuple[str, dict]] = []

    def bodies(self, path: str) -> list[dict]:
        return [b for p, b in self.requests if p.endswith(path)]


def _err(param: str, msg: str) -> dict:
    return {"error": {"message": msg, "type": "invalid_request_error", "param": param, "code": "unsupported_parameter"}}


def _usage_resp(i: int, o: int) -> dict:
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o,
            "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}


def _structured(name: str, prompt: str) -> dict:
    if name == "query_plan":
        q = prompt.rsplit("Question:", 1)[-1].strip()
        tamil = any("஀" <= c <= "௿" for c in q)
        eng = q
        for k, v in TANGLISH.items():
            eng = re.sub(rf"\b{k}\b", v, eng)
        tanglish = eng != q
        return {"language": "ta" if tamil else ("tanglish" if tanglish else "en"), "standalone_question": q,
                "english_query": eng if not tamil else "micro irrigation subsidy",
                "tamil_query": "நுண்ணீர் பாசனம் மானியம்" if "irrigation" in eng.lower() or tamil else "",
                "keywords": [w for w in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", q)][:6], "years": []}
    if name == "rerank_scores":
        q = prompt.split("Passages:", 1)[0].lower()
        qwords = {w for w in re.findall(r"[a-z0-9%]+", q) if len(w) > 2}
        out = []
        for pid, body in re.findall(r"<passage id=(\d+)>\n(.*?)\n</passage>", prompt, re.S):
            words = set(re.findall(r"[a-z0-9%]+", body.lower()))
            out.append({"id": int(pid), "score": min(10, len(qwords & words) * 2)})
        return {"scores": out}
    if name == "doc_context":
        title = re.search(r"Title: (.*)", prompt).group(1)
        year = re.search(r"(20\d\d)\s*[-–]\s*(\d\d)", prompt)
        return {"summary": f"Mock summary of {title}.", "doc_type": "other",
                "year": f"{year.group(1)}-{year.group(2)}" if year else "", "department": "Agriculture–Farmers Welfare",
                "schemes": ["Micro Irrigation"] if "irrigation" in prompt.lower() else []}
    if name == "followups":
        return {"questions": ["What documents are needed?", "Where do I apply?", "Is there a deadline?"]}
    if name == "ping":
        return {"ok": True}
    return {}


def _answer(prompt: str) -> str:
    m = re.search(r'<passage id="1"[^>]*>\n(.*?)\n</passage>', prompt, re.S)
    first = (m.group(1) if m else "no context").replace("\n", " ")[:160]
    return f"Based on the documents, {first} [1]."


def make_handler(rec: Recorder):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse(self, events: list[tuple[str | None, dict | str]]):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ev, data in events:
                payload = data if isinstance(data, str) else json.dumps(data)
                self.wfile.write(((f"event: {ev}\n" if ev else "") + f"data: {payload}\n\n").encode())
                self.wfile.flush()

        def do_GET(self):
            self._json(200, {"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model"},
                                                       {"id": "gpt-6-luna", "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            rec.requests.append((self.path, body))
            if self.headers.get("Authorization") != "Bearer test-key":
                return self._json(401, {"error": {"message": "Incorrect API key", "type": "invalid_request_error"}})
            if self.path.endswith("/embeddings"):
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                dims = body.get("dimensions", 3072)
                data = []
                for i, t in enumerate(inputs):
                    v = _HASH._vec(t)[:dims]
                    v = v / (np.linalg.norm(v) or 1)
                    data.append({"object": "embedding", "index": i, "embedding": v.tolist()})
                toks = sum(len(t) // 4 + 1 for t in inputs)
                return self._json(200, {"object": "list", "data": data, "model": body["model"],
                                        "usage": {"prompt_tokens": toks, "total_tokens": toks}})
            if body.get("model") in RETIRED and not self.path.endswith("/embeddings"):
                return self._json(404, _gone(body["model"]))
            if self.path.endswith("/responses"):
                if not _is_reasoning(body["model"]) and "reasoning" in body:
                    return self._json(400, _err("reasoning.effort", "Unsupported parameter: 'reasoning.effort' is "
                                                                    "not supported with this model."))
                if _is_reasoning(body["model"]) and "temperature" in body:
                    return self._json(400, _err("temperature", "Unsupported parameter: 'temperature' is not "
                                                               "supported with this model."))
                if body["model"] == "gpt-6.1-sol" and (body.get("reasoning") or {}).get("effort") in ("none", "minimal"):
                    return self._json(400, _err("reasoning.effort", "Unsupported value: 'reasoning.effort' does not "
                                                                    "support 'minimal' with this model."))
                prompt = (body.get("instructions") or "") + "\n" + "\n".join(
                    m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                    for m in body.get("input", []))
                fmt = (body.get("text") or {}).get("format") or {}
                text = json.dumps(_structured(fmt.get("name", ""), prompt), ensure_ascii=False) \
                    if fmt.get("type") == "json_schema" else _answer(prompt)
                resp = {"id": "resp_mock", "object": "response", "created_at": int(time.time()),
                        "model": body["model"], "status": "completed", "parallel_tool_calls": True,
                        "tool_choice": "auto", "tools": [], "output": [
                            {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                             "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                        "usage": _usage_resp(len(prompt) // 4, len(text) // 4)}
                if not body.get("stream"):
                    return self._json(200, resp)
                events = [("response.created", {"type": "response.created", "sequence_number": 0,
                                                "response": {**resp, "status": "in_progress", "output": []}})]
                for i, w in enumerate(re.findall(r"\S+\s*", text), 1):
                    events.append(("response.output_text.delta", {
                        "type": "response.output_text.delta", "sequence_number": i, "item_id": "msg_1",
                        "output_index": 0, "content_index": 0, "delta": w, "logprobs": []}))
                events.append(("response.completed", {"type": "response.completed", "sequence_number": 999,
                                                      "response": resp}))
                return self._sse(events)
            if self.path.endswith("/chat/completions"):
                if not _is_reasoning(body["model"]) and "reasoning_effort" in body:
                    return self._json(400, _err("reasoning_effort", "Unrecognized request argument supplied: "
                                                                    "reasoning_effort"))
                if _is_reasoning(body["model"]) and "max_tokens" in body:
                    return self._json(400, _err("max_tokens", "Unsupported parameter: 'max_tokens' is not supported "
                                                              "with this model. Use 'max_completion_tokens' instead."))
                if _is_reasoning(body["model"]) and "temperature" in body:
                    return self._json(400, _err("temperature", "Unsupported value: 'temperature' does not support "
                                                               "0.2 with this model."))
                prompt = "\n".join(m["content"] for m in body["messages"])
                rf = body.get("response_format") or {}
                text = json.dumps(_structured(rf.get("json_schema", {}).get("name", ""), prompt), ensure_ascii=False) \
                    if rf.get("type") == "json_schema" else _answer(prompt)
                usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4,
                         "total_tokens": len(prompt) // 4 + len(text) // 4}
                base = {"id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
                        "model": body["model"]}
                if not body.get("stream"):
                    return self._json(200, {**base, "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": text}}], "usage": usage})
                ev = [(None, {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {"content": w}, "finish_reason": None}]}) for w in re.findall(r"\S+\s*", text)]
                ev.append((None, {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}]}))
                ev.append((None, {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage}))
                ev.append((None, "[DONE]"))
                return self._sse(ev)
            self._json(404, {"error": {"message": "not found"}})

    return H


def start() -> tuple[str, Recorder, ThreadingHTTPServer]:
    rec = Recorder()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(rec))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1", rec, srv
