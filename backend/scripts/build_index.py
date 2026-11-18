"""Run PIPELINE 1 from the command line (no server needed).

  python scripts/build_index.py                 # index data/raw (crawled files) + data/seed
  python scripts/build_index.py --dir /path/to/folder --category local
  python scripts/build_index.py --reset         # wipe the FAISS index first
"""
import argparse
from pathlib import Path

from _common import run_with_progress

from app.config import settings
from app.ingest.loaders import SUPPORTED_EXT
from app.ingest.pipeline import ingest_files
from app.jobs import Job
from app.store.vector_store import get_store


def collect(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and not p.name.startswith(("_", ".")) and p.suffix.lower() in SUPPORTED_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", help="folder(s) to ingest (default: data/seed and data/raw)")
    ap.add_argument("--category", default=None)
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--chunk-overlap", type=int, default=0)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--if-empty", action="store_true", help="do nothing if the index already has documents")
    args = ap.parse_args()

    settings.ensure_dirs()
    store = get_store()
    if args.if_empty and store.stats()["documents"]:
        print(f"index already has {store.stats()['documents']} documents → {settings.store_dir / 'index.faiss'}")
        return
    if args.reset:
        store.reset()
        print("index reset")
    folders = [Path(d) for d in args.dir] if args.dir else [settings.seed_dir, settings.raw_dir]
    files: list[Path] = []
    for f in folders:
        if f.exists():
            files += collect(f)
    print(f"{len(files)} file(s) to process from {', '.join(map(str, folders))}")
    job = Job(kind="cli", params={"dirs": [str(f) for f in folders]})
    opts = {"category": args.category or "seed", "chunk_size": args.chunk_size or None,
            "chunk_overlap": args.chunk_overlap or None}
    meta = {str(p): {"category": "seed"} for p in files if settings.seed_dir in p.parents}
    run_with_progress(job, lambda: ingest_files(job, files, meta, opts))
    s = store.stats()
    print(f"\nFAISS index: {s['documents']} documents, {s['vectors']} vectors, {s['index_bytes']/1e6:.1f} MB "
          f"→ {settings.store_dir / 'index.faiss'}")
    print("by type:", s["by_type"])
    print("by language (chunks):", s["by_language"])


if __name__ == "__main__":
    main()
