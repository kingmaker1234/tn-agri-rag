"""Background job manager. Ingestion jobs run one at a time on a worker thread
(FAISS writes are single-writer); the UI polls /api/jobs/{id} for progress."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .config import settings

log = logging.getLogger(__name__)

STAGES = ["queued", "downloading", "loading", "splitting", "embedding", "indexing", "finalizing", "done"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    kind: str
    params: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    # counters (each stage has its own bar in the UI)
    downloads_total: int = 0
    downloads_done: int = 0
    downloads_failed: int = 0
    files_total: int = 0
    files_loaded: int = 0
    files_split: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_total: int = 0
    chunks_embedded: int = 0
    current_file: str | None = None
    cost_usd: float = 0.0
    tokens: int = 0
    files: list[dict] = field(default_factory=list)
    logs: deque = field(default_factory=lambda: deque(maxlen=400))
    cancel_requested: bool = False
    _t0: float = 0.0

    def log(self, msg: str, level: str = "info") -> None:
        self.logs.append({"t": _now(), "level": level, "msg": msg})
        getattr(log, "warning" if level == "warn" else level if level in ("info", "error") else "info")(
            "[job %s] %s", self.id, msg)

    def file_entry(self, name: str, **kw) -> dict:
        e = {"name": name, "status": "pending", "chunks": 0, "error": None, **kw}
        self.files.append(e)
        return e

    def to_dict(self, include_logs: bool = True) -> dict:
        elapsed = (time.time() - self._t0) if self._t0 and self.status == "running" else None
        d = {k: getattr(self, k) for k in (
            "id", "kind", "params", "status", "stage", "created_at", "started_at", "finished_at", "error",
            "downloads_total", "downloads_done", "downloads_failed", "files_total", "files_loaded", "files_split",
            "files_indexed", "files_skipped", "files_failed", "chunks_total", "chunks_embedded", "current_file",
            "cost_usd", "tokens")}
        d["elapsed_s"] = round(elapsed, 1) if elapsed else None
        d["files"] = self.files[-500:]
        if include_logs:
            d["logs"] = list(self.logs)
        return d


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        self._runners: dict[str, Callable[[Job], None]] = {}
        self._worker = threading.Thread(target=self._loop, daemon=True, name="ingest-worker")
        self._worker.start()
        self._history_file = settings.data_dir / "jobs_history.json"
        self._load_history()

    def _load_history(self) -> None:
        try:
            for d in json.loads(self._history_file.read_text())[-30:]:
                j = Job(kind=d["kind"], params=d.get("params", {}))
                for k, v in d.items():
                    if k in ("logs", "elapsed_s"):
                        continue
                    if hasattr(j, k):
                        setattr(j, k, v)
                for entry in d.get("logs", [])[-50:]:
                    j.logs.append(entry)
                self.jobs[j.id] = j
        except Exception:
            pass

    def _save_history(self) -> None:
        try:
            done = [j.to_dict() for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")][-30:]
            for d in done:
                d["logs"] = d["logs"][-50:]
                d["files"] = d["files"][-200:]
            self._history_file.write_text(json.dumps(done, ensure_ascii=False))
        except Exception as e:  # pragma: no cover
            log.warning("could not save job history: %s", e)

    def submit(self, kind: str, params: dict, runner: Callable[[Job], None]) -> Job:
        job = Job(kind=kind, params=params)
        self.jobs[job.id] = job
        self._runners[job.id] = runner
        job.log(f"queued ({kind})")
        self.q.put(job.id)
        return job

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status not in ("queued", "running"):
            return False
        j.cancel_requested = True
        j.log("cancel requested", "warn")
        return True

    def list(self) -> list[dict]:
        return [j.to_dict(include_logs=False) for j in sorted(self.jobs.values(), key=lambda j: j.created_at,
                                                              reverse=True)]

    def active(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            job = self.jobs.get(job_id)
            runner = self._runners.pop(job_id, None)
            if not job or not runner:
                continue
            if job.cancel_requested:
                job.status, job.stage, job.finished_at = "cancelled", "done", _now()
                continue
            job.status, job.started_at, job._t0 = "running", _now(), time.time()
            try:
                runner(job)
                job.status = "cancelled" if job.cancel_requested else "done"
                job.stage = "done"
                job.log(f"finished in {time.time() - job._t0:.1f}s — indexed {job.files_indexed} file(s), "
                        f"{job.chunks_embedded} chunk(s); skipped {job.files_skipped}, failed {job.files_failed}")
            except Exception as e:
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                job.log(job.error, "error")
                log.error(traceback.format_exc())
            finally:
                job.finished_at = _now()
                job.current_file = None
                self._save_history()


_manager: JobManager | None = None


def get_jobs() -> JobManager:
    global _manager
    if _manager is None:
        settings.ensure_dirs()
        _manager = JobManager()
    return _manager
