"""Download the research manifest (and optionally crawl seed pages) into data/raw/.

Examples
  python scripts/crawl.py                         # all 'core' manifest sources
  python scripts/crawl.py --relevance all         # core + supporting (569 URLs)
  python scripts/crawl.py --discover --max-discovered 400   # + crawl seeds.txt directory listings
  python scripts/crawl.py --categories policy_note budget citizen_charter
"""
import argparse

from _common import ROOT  # noqa: F401  (sets sys.path)

from app.config import settings
from app.ingest.crawler import discover, download_many, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relevance", choices=["core", "all"], default="core")
    ap.add_argument("--categories", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--discover", action="store_true", help="also crawl seeds.txt for more documents")
    ap.add_argument("--max-discovered", type=int, default=400)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    settings.ensure_dirs()
    rows = read_manifest()
    if args.relevance == "core":
        rows = [r for r in rows if r["relevance"] == "core"]
    if args.categories:
        rows = [r for r in rows if r["category"] in args.categories]
    if args.limit:
        rows = rows[: args.limit]
    items = [{"url": r["url"], "title": r["title"] or None, "category": r["category"], "source_id": r["id"],
              "language": r["language"], "relevance": r["relevance"]} for r in rows]
    print(f"manifest: {len(items)} source(s) selected")

    if args.discover:
        seeds = [l.strip() for l in (settings.sources_dir / "seeds.txt").read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        found = discover(seeds, max_depth=args.depth, max_files=args.max_discovered, on_log=print)
        known = {i["url"] for i in items}
        extra = [f for f in found if f["url"] not in known]
        print(f"crawler discovered {len(extra)} extra document(s)")
        items += extra

    ok = fail = skipped = 0

    def prog(res, i, n):
        nonlocal ok, fail, skipped
        if res.ok:
            ok += 1
            skipped += res.skipped
        else:
            fail += 1
        mark = "↷" if res.skipped else ("✓" if res.ok else "✗")
        print(f"[{i:>4}/{n}] {mark} {res.url[:110]}" + (f"  ({res.error})" if res.error else ""))

    download_many(items, prog)
    print(f"\ndone: {ok} available ({skipped} already downloaded), {fail} failed → {settings.raw_dir}")
    if ok < 300:
        print("note: fewer than 300 files available — rerun with --relevance all --discover to reach 300+.")


if __name__ == "__main__":
    main()
