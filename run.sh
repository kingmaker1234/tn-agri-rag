#!/usr/bin/env bash
# Local start, no Docker (Linux / macOS)
#
#   ./run.sh            set up once, index the starter pack, start  →  http://localhost:8000
#   ./run.sh download   also download the 569 researched sources (+ crawl) and index them (300+ files)
#
# Everything stays on this computer: the FAISS index is saved in backend/data/vectorstore/
# (change with VECTOR_STORE_DIR in backend/.env). Stop with Ctrl+C; the index stays on disk.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# ---- 1. Python 3.10–3.13 ---------------------------------------------------------
PYTHON=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
    PYTHON="$c"; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "! Python 3.10–3.13 is needed (faiss-cpu wheels). Install it and run again."
  exit 1
fi

# ---- 2. virtual env + packages (reinstalled only when requirements.txt changes) ----
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
PY=.venv/bin/python
if ! "$PY" scripts/local_setup.py deps-ok; then
  echo "▸ installing Python packages (first run takes about a minute)…"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  "$PY" scripts/local_setup.py deps-mark
fi

# ---- 3. OpenAI key (stored only in backend/.env, which git ignores) ----------------
"$PY" scripts/local_setup.py key

# ---- 4. UI (built once into frontend/dist and served by the backend) --------------
if [ ! -f "$ROOT/frontend/dist/index.html" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "▸ building the UI (one time)…"
    (cd "$ROOT/frontend" && npm ci --no-audit --no-fund && npm run build)
  else
    echo "! Node.js 20+ not found, so the UI can't be built. The API still runs (http://localhost:$PORT/docs)."
  fi
fi

# ---- 5. vector DB: starter pack, or the full 300+ file corpus ----------------------
if [ "${1:-}" = "download" ]; then
  echo "▸ downloading the researched sources and indexing them (takes a while; safe to re-run)…"
  "$PY" scripts/bootstrap_corpus.py --all --discover
fi
"$PY" scripts/build_index.py --dir data/seed --if-empty || echo "! starter pack indexing failed — check OPENAI_API_KEY in backend/.env"

echo "▸ open http://localhost:$PORT    (Ctrl+C to stop; the FAISS index stays saved on disk)"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
