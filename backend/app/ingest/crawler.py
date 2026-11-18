"""Downloader + polite crawler for the source manifest.

* `download_many(urls)`  – fetch a list of URLs into data/raw/<host>/...
* `discover(seeds)`      – walk Apache "Index of /" listings (recursively) and
                           HTML hub pages, returning document URLs that match
                           the agriculture keyword filter.

Every download is recorded in data/raw/_meta.jsonl so the ingestion pipeline
can attach source URL, title and category to each document.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from ..config import settings

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (X11; Linux x86_64) TN-Agri-RAG/1.0 (+research crawler; respects robots via low rate)"
DOC_EXT = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".odt", ".ods", ".json", ".txt", ".rtf")
KEEP = re.compile(
    r"agri|horti|scheme|farm|crop|seed|subsid|budget|policy|pn_|_cc_|citizen|charter|uzhav|irrigat|kisan|fasal|pmfby|"
    r"nadp|rkvy|millet|paddy|pulse|organic|market|sandhai|tawdeva|tanhoda|tanhope|engg|mechani|soil|water|dap|guideline|"
    r"g\.?o|mission|kusum|solar|fpo|coconut|sugar|cotton|banana|vegetable|oilseed|nfsm|midh|nhm|pmksy|iamp|iamwarm|"
    r"tnau|insurance|loan|credit|cooperat|plan|report|survey|statistic|advisory|kharif|rabi|samba|kuruvai|வேளாண்|திட்ட",
    re.I)
SKIP = re.compile(r"birth|death|ration|community|cert-income|hrd_|nutritive|ngo_|icon|logo|button|thumbs|\.gif$|\.png$|"
                  r"\.jpg$|\.swf$|training_?calend|vacanc|tender", re.I)


@dataclass
class FetchResult:
    url: str
    ok: bool
    path: str | None = None
    status: int | None = None
    error: str | None = None
    content_type: str | None = None
    size: int = 0
    sha256: str | None = None
    insecure: bool = False
    skipped: bool = False


_meta_lock = threading.Lock()


def _meta_file() -> Path:
    return settings.raw_dir / "_meta.jsonl"


def load_meta() -> dict[str, dict]:
    """path -> metadata (latest record wins)."""
    out: dict[str, dict] = {}
    f = _meta_file()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("path"):
                    out[str(Path(rec["path"]).resolve())] = rec
            except json.JSONDecodeError:
                continue
    return out


def downloaded_urls() -> dict[str, dict]:
    return {rec["url"]: rec for rec in load_meta().values() if rec.get("ok")}


def _append_meta(rec: dict) -> None:
    with _meta_lock:
        _meta_file().parent.mkdir(parents=True, exist_ok=True)
        with _meta_file().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _safe_name(url: str, content_type: str | None) -> Path:
    parts = urlsplit(url)
    host = re.sub(r"[^a-z0-9.\-]", "_", parts.netloc.lower())
    name = unquote(parts.path.rstrip("/").rsplit("/", 1)[-1]) or "index"
    if parts.query:
        name += "_" + hashlib.md5(parts.query.encode()).hexdigest()[:8]
    name = re.sub(r"[^\w஀-௿.\-]+", "_", name).strip("._")[:150] or "file"
    stem, dot, ext = name.rpartition(".")
    has_ext = dot and len(ext) <= 5 and ext.isalnum()
    if not has_ext:
        guess = mimetypes.guess_extension((content_type or "").split(";")[0].strip()) or ""
        if guess in (".htm", ".shtml", ""):
            guess = ".html" if "html" in (content_type or "") or not guess else guess
        name = name + guess
    # path hash keeps identical file names from different folders apart
    h = hashlib.md5(parts.path.encode()).hexdigest()[:6]
    stem, _, ext = name.rpartition(".")
    return settings.raw_dir / host / f"{stem}__{h}.{ext}"


def _client(verify: bool = True) -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=settings.http_timeout, verify=verify,
                        headers={"User-Agent": UA, "Accept": "*/*"})


def fetch(url: str, meta: dict | None = None, retries: int = 2, skip_existing: bool = True,
          known: dict[str, dict] | None = None) -> FetchResult:
    meta = meta or {}
    if skip_existing:
        prev = (known if known is not None else downloaded_urls()).get(url)
        if prev and Path(prev["path"]).exists():
            return FetchResult(url, True, prev["path"], prev.get("status"), skipped=True, size=prev.get("size", 0),
                               sha256=prev.get("sha256"), content_type=prev.get("content_type"))
    last_err = None
    insecure = False
    for attempt in range(retries + 1):
        try:
            with _client(verify=not insecure) as c:
                r = c.get(url)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}"
                if r.status_code in (404, 403, 410):
                    break
                time.sleep(1.5 * (attempt + 1))
                continue
            data = r.content
            ctype = r.headers.get("content-type", "")
            path = _safe_name(str(r.url), ctype)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            rec = {"url": url, "final_url": str(r.url), "path": str(path.resolve()), "ok": True, "status": r.status_code,
                   "content_type": ctype, "size": len(data), "sha256": sha, "insecure_tls": insecure,
                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **meta}
            _append_meta(rec)
            return FetchResult(url, True, str(path), r.status_code, None, ctype, len(data), sha, insecure)
        except httpx.ConnectError as e:
            last_err = str(e)
            if "CERTIFICATE_VERIFY_FAILED" in last_err and settings.allow_insecure_ssl_fallback and not insecure:
                log.warning("TLS verify failed for %s — retrying without verification (ALLOW_INSECURE_SSL_FALLBACK)", url)
                insecure = True
                continue
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # timeouts, protocol errors …
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    _append_meta({"url": url, "ok": False, "error": last_err, **meta,
                  "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return FetchResult(url, False, error=last_err)


def download_many(items: Iterable[dict], on_progress: Callable[[FetchResult, int, int], None] | None = None,
                  should_stop: Callable[[], bool] | None = None) -> list[FetchResult]:
    """items: [{url, title?, category?, source_id?, language?}]"""
    items = list(items)
    results: list[FetchResult] = []
    host_last: dict[str, float] = {}
    host_lock = threading.Lock()
    known = downloaded_urls()

    def polite(item):
        host = urlsplit(item["url"]).netloc
        with host_lock:
            wait = host_last.get(host, 0) + 0.4 - time.time()
            host_last[host] = time.time() + max(wait, 0)
        if wait > 0:
            time.sleep(wait)
        meta = {k: item.get(k) for k in ("title", "category", "source_id", "language", "relevance") if item.get(k)}
        return fetch(item["url"], meta, known=known)

    with ThreadPoolExecutor(max_workers=max(1, settings.crawl_concurrency)) as ex:
        futs = {}
        for it in items:
            if should_stop and should_stop():
                break
            futs[ex.submit(polite, it)] = it
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            if on_progress:
                on_progress(res, i, len(futs))
    return results


# ------------------------------------------------------------------ discovery
def _is_dir_index(html: str) -> bool:
    return "<title>Index of /" in html[:2000]


def discover(seeds: list[str], max_depth: int = 2, max_files: int = 1000, use_filter: bool = True,
             on_log: Callable[[str], None] | None = None) -> list[dict]:
    from bs4 import BeautifulSoup

    seen_pages, found, out = set(), set(), []
    queue = [(s, 0) for s in seeds]
    with _client() as c:
        while queue and len(out) < max_files:
            url, depth = queue.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            try:
                r = c.get(url)
                if r.status_code >= 400 or "html" not in r.headers.get("content-type", ""):
                    continue
            except Exception as e:
                on_log and on_log(f"discover: {url} failed ({type(e).__name__})")
                continue
            soup = BeautifulSoup(r.text, "lxml")
            dir_index = _is_dir_index(r.text)
            base_host = urlsplit(url).netloc
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("?", "#", "mailto:", "javascript:")):
                    continue
                full = urljoin(url, href)
                parts = urlsplit(full)
                if parts.netloc != base_host:
                    continue
                low = parts.path.lower()
                text = a.get_text(" ", strip=True)
                if low.endswith(DOC_EXT):
                    name = unquote(parts.path.rsplit("/", 1)[-1])
                    if SKIP.search(name) or (use_filter and not dir_index and not KEEP.search(name + " " + text)):
                        continue
                    if use_filter and dir_index and not KEEP.search(name) and len(name) < 6:
                        continue
                    if full not in found:
                        found.add(full)
                        out.append({"url": full, "title": text or name.rsplit(".", 1)[0], "category": "discovered",
                                    "origin": url})
                elif dir_index and href.endswith("/") and not href.startswith("/") and ".." not in href \
                        and depth < max_depth:
                    queue.append((full, depth + 1))
            on_log and on_log(f"discover: {url} → {len(out)} documents so far")
    return out[:max_files]


# ------------------------------------------------------------------ manifest
def read_manifest(path: Path | None = None) -> list[dict]:
    path = path or settings.sources_dir / "manifest.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
