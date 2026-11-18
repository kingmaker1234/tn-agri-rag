"""Persistent usage totals and answer feedback (data/telemetry.sqlite)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from .config import settings

_db: sqlite3.Connection | None = None
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    global _db
    if _db is None:
        settings.ensure_dirs()
        _db = sqlite3.connect(settings.data_dir / "telemetry.sqlite", check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.executescript("""
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, step TEXT, model TEXT,
            input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL);
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, question TEXT, answer TEXT, rating INTEGER,
            comment TEXT, sources TEXT, model TEXT);
        """)
    return _db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_usage(kind: str, steps: list[dict]) -> None:
    if not steps:
        return
    with _lock:
        db = _conn()
        db.executemany("INSERT INTO usage (ts, kind, step, model, input_tokens, output_tokens, cost_usd) "
                       "VALUES (?,?,?,?,?,?,?)",
                       [(_now(), kind, s["step"], s["model"], s["input_tokens"], s["output_tokens"], s["cost_usd"])
                        for s in steps])
        db.commit()


def usage_totals() -> dict:
    db = _conn()
    tot = db.execute("SELECT COALESCE(SUM(cost_usd),0), COALESCE(SUM(input_tokens),0), "
                     "COALESCE(SUM(output_tokens),0) FROM usage").fetchone()
    by_kind = {r[0]: {"cost_usd": round(r[1], 4), "calls": r[2]} for r in db.execute(
        "SELECT kind, SUM(cost_usd), COUNT(DISTINCT ts) FROM usage GROUP BY kind")}
    by_model = {r[0]: {"cost_usd": round(r[1], 4), "input_tokens": r[2], "output_tokens": r[3]} for r in db.execute(
        "SELECT model, SUM(cost_usd), SUM(input_tokens), SUM(output_tokens) FROM usage GROUP BY model ORDER BY 2 DESC")}
    questions = db.execute("SELECT COUNT(DISTINCT ts) FROM usage WHERE kind='ask'").fetchone()[0]
    return {"cost_usd": round(tot[0], 4), "input_tokens": tot[1], "output_tokens": tot[2], "by_kind": by_kind,
            "by_model": by_model, "questions": questions,
            "avg_cost_per_question": round(by_kind.get("ask", {}).get("cost_usd", 0) / questions, 5) if questions else 0}


def add_feedback(question: str, answer: str, rating: int, comment: str, sources: list, model: str) -> int:
    with _lock:
        db = _conn()
        cur = db.execute("INSERT INTO feedback (ts, question, answer, rating, comment, sources, model) "
                         "VALUES (?,?,?,?,?,?,?)",
                         (_now(), question, answer, rating, comment, json.dumps(sources, ensure_ascii=False), model))
        db.commit()
        return cur.lastrowid


def feedback_summary(limit: int = 20) -> dict:
    db = _conn()
    up = db.execute("SELECT COUNT(*) FROM feedback WHERE rating > 0").fetchone()[0]
    down = db.execute("SELECT COUNT(*) FROM feedback WHERE rating < 0").fetchone()[0]
    recent = [dict(r) for r in db.execute(
        "SELECT ts, question, rating, comment FROM feedback ORDER BY id DESC LIMIT ?", (limit,))]
    return {"up": up, "down": down, "recent": recent}
