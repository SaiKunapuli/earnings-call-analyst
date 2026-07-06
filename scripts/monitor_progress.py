"""Live progress bar for the full 03_sentiment FinBERT run.

FinBERT checkpoints every 100 transcripts to the `finbert_scores` table, which
is an external progress signal (the notebook's own prints are buffered by
nbclient and never reach the terminal). Run this in a SECOND terminal while
`run_pipeline.py --from 03` works in the first:

    .venv\\Scripts\\python.exe scripts\\monitor_progress.py

Shows a live bar with %, count, scoring rate, and ETA. Ctrl+C stops the MONITOR
only — the pipeline keeps running. `--once` prints a single snapshot and exits.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB = Path(__file__).resolve().parent.parent / "data" / "market.db"
MIN_PUB_DATE = "2016-07-01"   # must match MAX/MIN filter in 03_sentiment
BAR_W = 40


def _counts():
    """(total transcripts to score, transcripts scored so far)."""
    c = sqlite3.connect(str(DB), timeout=5)
    try:
        total = c.execute(
            "SELECT COUNT(*) FROM transcripts_text WHERE pub_date >= ?",
            (MIN_PUB_DATE,)).fetchone()[0]
        try:
            done = c.execute("SELECT COUNT(*) FROM finbert_scores").fetchone()[0]
        except sqlite3.OperationalError:
            done = 0   # table not created yet
        return total, done
    finally:
        c.close()


def _fmt(secs):
    if secs is None or secs != secs or secs < 0:
        return "?"
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _render(total, done, rate_per_min, elapsed):
    done = min(done, total) if total else done
    pct = min(100.0, done / total * 100) if total else 0.0
    filled = min(BAR_W, int(BAR_W * done / total)) if total else 0
    bar = "#" * filled + "-" * (BAR_W - filled)
    eta = ((total - done) / rate_per_min * 60) if rate_per_min > 0 else None
    return (f"FinBERT  [{bar}] {pct:5.1f}%  {done:,}/{total:,}  |  "
            f"{rate_per_min:5.1f} tx/min  |  ETA {_fmt(eta)}  |  elapsed {_fmt(elapsed)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0, help="poll seconds")
    ap.add_argument("--once", action="store_true", help="one snapshot then exit")
    args = ap.parse_args()

    total, start_done = _counts()
    if args.once:
        print(_render(total, start_done, 0.0, 0.0))
        return 0

    print(f"Monitoring FinBERT (poll {args.interval:.0f}s). "
          f"Ctrl+C stops the monitor only — the pipeline keeps running.\n")
    t0 = time.time()
    hist = deque(maxlen=20)          # rolling (time, done) for a smoothed rate
    hist.append((t0, start_done))
    try:
        while True:
            total, done = _counts()
            now = time.time()
            hist.append((now, done))
            t_old, d_old = hist[0]
            dt = now - t_old
            rate = ((done - d_old) / dt * 60) if dt > 0 else 0.0

            if total and done >= total:
                print("\n" + _render(total, done, rate, now - t0))
                print("\nFinBERT COMPLETE — 04_modeling + 05_backtest finish in minutes.")
                return 0
            if rate <= 0 and done <= start_done:
                line = (f"Waiting for FinBERT to start (Stage A VADER/LM, or model "
                        f"loading)...  {done:,}/{total:,}   elapsed {_fmt(now - t0)}")
            else:
                line = _render(total, done, rate, now - t0)
            print("\r" + line + "    ", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped (pipeline unaffected).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
