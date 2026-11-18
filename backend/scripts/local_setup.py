"""Helpers for run.sh / run.bat (standard library only, works on Windows, macOS and Linux).

  python scripts/local_setup.py deps-ok     exit 0 if .venv packages match requirements.txt
  python scripts/local_setup.py deps-mark   record that requirements.txt was installed
  python scripts/local_setup.py key         make sure backend/.env exists and holds an OpenAI key
"""
from __future__ import annotations

import getpass
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
MARK = BACKEND / ".venv" / ".deps"
ENV = BACKEND / ".env"
KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


def _req_hash() -> str:
    return hashlib.sha256((BACKEND / "requirements.txt").read_bytes()).hexdigest()


def deps_ok() -> int:
    return 0 if MARK.exists() and MARK.read_text().strip() == _req_hash() else 1


def deps_mark() -> int:
    MARK.write_text(_req_hash())
    return 0


def key() -> int:
    if not ENV.exists():
        shutil.copy(BACKEND / ".env.example", ENV)
    text = ENV.read_text(encoding="utf-8")
    m = re.search(r"(?m)^OPENAI_API_KEY=(.*)$", text)
    if m and KEY_RE.fullmatch(m.group(1).strip()):
        return 0
    if os.environ.get("OPENAI_API_KEY", "").strip():  # exported in the shell: .env not needed
        return 0
    if not sys.stdin.isatty():
        print("! OPENAI_API_KEY is not set in backend/.env")
        return 0
    value = getpass.getpass("Paste your OpenAI API key (saved only in backend/.env): ").strip()
    if not value:
        print("! No key entered. Answers and embeddings need one: edit backend/.env later.")
        return 0
    if not value.startswith("sk-"):
        print("! That doesn't look like an OpenAI key (they start with sk-). Saved anyway.")
    line = "OPENAI_API_KEY=" + value
    text = (re.sub(r"(?m)^OPENAI_API_KEY=.*$", lambda _: line, text) if m
            else text.rstrip("\n") + "\n" + line + "\n")
    ENV.write_text(text, encoding="utf-8")
    print("Key saved to backend/.env (git ignores this file).")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"deps-ok": deps_ok, "deps-mark": deps_mark, "key": key}.get(cmd, lambda: 2)())
