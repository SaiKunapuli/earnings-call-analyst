"""Incrementally ingest new earnings-call transcripts via defeatbeta-api.

The HF corpus ends 2025-05-15; this pulls newer quarters (free, no key) so the
model can be tested truly out-of-sample on calls it never saw. Incremental and
resume-safe: for each ticker only fetches quarters whose report_date is newer
than that ticker's latest stored pub_date. Ctrl+C safe (one transaction per
ticker; re-run continues).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/update_transcripts.py
    ... --tickers AAPL MSFT        # restrict
    ... --limit 20                 # first N tickers (smoke)

NOTE: installing defeatbeta-api upgrades numpy/pandas past the pins; restore
with `pip install numpy==1.26.4 pandas==2.2.0` afterwards (library still works).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")   # defeatbeta prints an emoji banner
except Exception:
    pass

import pandas as pd

from src.config import DB_PATH, get_full_universe
from src.transcripts_io import (make_row, normalize_turns, open_db, split_turns,
                                write_batch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="+", help="restrict to these tickers")
    ap.add_argument("--limit", type=int, help="only the first N tickers")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between tickers")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    from defeatbeta_api.data.ticker import Ticker   # import here (slow, emoji banner)

    conn = open_db(args.db)
    known = pd.read_sql(
        "SELECT ticker, quarter, year, pub_date FROM transcripts_text", conn)
    existing = set(known[["ticker", "quarter", "year"]]
                   .itertuples(index=False, name=None))
    latest_pub = known.groupby("ticker")["pub_date"].max().to_dict()
    print(f"corpus: {len(known):,} transcripts | latest overall "
          f"{known['pub_date'].max()}")

    universe = args.tickers or get_full_universe(args.db)
    if args.limit:
        universe = universe[:args.limit]
    print(f"scanning {len(universe)} tickers for newer quarters...\n")

    n_new = n_skip = n_fail = 0
    t0 = time.time()
    for i, symbol in enumerate(universe, 1):
        try:
            tr = Ticker(symbol).earning_call_transcripts()
            listing = tr.get_transcripts_list()
        except Exception as e:
            n_fail += 1
            continue
        cutoff = latest_pub.get(symbol, "")
        batch = []
        for r in listing.itertuples():
            try:
                year, q = int(r.fiscal_year), int(r.fiscal_quarter)
            except Exception:
                continue
            pub_date = str(r.report_date)[:10]
            key = (symbol, f"q{q}", year)
            if not (1 <= q <= 4) or key in existing or pub_date <= cutoff:
                n_skip += 1
                continue
            try:
                tdf = tr.get_transcript(year, q)
            except Exception:
                n_fail += 1
                continue
            turns = normalize_turns(
                [{"speaker": t.speaker, "text": t.content} for t in tdf.itertuples()])
            if not turns:
                n_skip += 1
                continue
            sections = split_turns(turns)
            if not sections.get("full_text"):
                n_skip += 1
                continue
            batch.append(make_row(symbol, f"q{q}", year, pub_date,
                                  "defeatbeta", sections))
            existing.add(key)
        if batch:
            write_batch(conn, batch)
            n_new += len(batch)
            print(f"  [{i}/{len(universe)}] {symbol}: +{len(batch)} new "
                  f"({', '.join(f'{r[2]} {r[1]}' for r in batch[:4])})")
        if i % 50 == 0:
            print(f"  ...{i}/{len(universe)} scanned | new={n_new} skip={n_skip} "
                  f"fail={n_fail} | {(time.time()-t0)/60:.1f} min")
        if args.delay:
            time.sleep(args.delay)

    total = conn.execute("SELECT COUNT(*) FROM transcripts_text").fetchone()[0]
    newmax = conn.execute("SELECT MAX(pub_date) FROM transcripts_text").fetchone()[0]
    conn.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min — new={n_new} skip={n_skip} "
          f"fail={n_fail}; transcripts_text now {total:,} (latest {newmax}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
