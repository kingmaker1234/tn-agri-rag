"""One command to build the full knowledge base: download 300+ files and index them.

  python scripts/bootstrap_corpus.py                 # core manifest + seed pack
  python scripts/bootstrap_corpus.py --all --discover  # everything (≈ 569 manifest + crawled extras)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include 'supporting' sources too")
    ap.add_argument("--discover", action="store_true", help="crawl seeds.txt for extra documents")
    args = ap.parse_args()
    crawl = [sys.executable, str(HERE / "crawl.py"), "--relevance", "all" if args.all else "core"]
    if args.discover:
        crawl.append("--discover")
    print("STEP 1/2 — downloading sources …")
    subprocess.run(crawl, check=True)
    print("\nSTEP 2/2 — load → split → embed → FAISS …")
    subprocess.run([sys.executable, str(HERE / "build_index.py")], check=True)


if __name__ == "__main__":
    main()
