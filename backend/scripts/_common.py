import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def progress_printer(job, stop: threading.Event):
    """Prints a one-line live status for a Job while the pipeline runs."""
    last_log = 0
    while not stop.is_set():
        logs = list(job.logs)
        for entry in logs[last_log:]:
            print(f"  · {entry['msg']}")
        last_log = len(logs)
        line = (f"\r[{job.stage:<11}] dl {job.downloads_done}/{job.downloads_total} "
                f"| files {job.files_indexed + job.files_skipped + job.files_failed}/{job.files_total} "
                f"(✓{job.files_indexed} ↷{job.files_skipped} ✗{job.files_failed}) "
                f"| chunks embedded {job.chunks_embedded}   ")
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(1.0)
    for entry in list(job.logs)[last_log:]:
        print(f"\n  · {entry['msg']}", end="")
    print()


def run_with_progress(job, fn):
    stop = threading.Event()
    t = threading.Thread(target=progress_printer, args=(job, stop), daemon=True)
    t.start()
    try:
        fn()
    finally:
        stop.set()
        t.join()
