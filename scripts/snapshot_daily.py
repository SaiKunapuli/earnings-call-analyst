"""Nightly snapshot recorder — builds a proprietary time series from free data.

Free sources (yfinance) expose only the CURRENT analyst estimates, price
targets, and short interest. Historical revisions — the single best-documented
PEAD companion signal ("Layer 4" in docs/trading_bot_layers.md) — are
paywalled. Recording a daily snapshot makes that history ours: in 12 months
this becomes an estimate-revisions dataset that cannot be downloaded
retroactively. Every day not recording is history lost forever.

Writes to data/snapshots.db (separate from market.db so pipeline runs and
snapshots never contend for locks):

  analyst_snapshot   one row per (snap_date, ticker): price targets,
                     recommendation mean, forward EPS/PE, short interest
                     (shares, prior month, ratio, % of float)
  estimate_snapshot  one row per (snap_date, ticker, period in 0q/+1q/0y/+1y):
                     EPS estimate trend (current vs 7/30/60/90 days ago) and
                     up/down revision counts

Usage:
    .venv/Scripts/python.exe scripts/snapshot_daily.py              # full universe
    .venv/Scripts/python.exe scripts/snapshot_daily.py --limit 5    # test
    .venv/Scripts/python.exe scripts/snapshot_daily.py --tickers MSFT NVDA

Safe to re-run within the same day: already-snapshotted tickers are skipped,
so a crashed/interrupted run resumes where it left off.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import DB_PATH, TICKERS, get_full_universe

SNAP_DB = PROJECT_ROOT / "data" / "snapshots.db"
# Committed snapshot of the full universe so GitHub Actions (which has no
# market.db) can snapshot all ~690 names, not just the 57 curated ones.
# Refreshed automatically on every LOCAL run (where market.db is present).
UNIVERSE_FILE = PROJECT_ROOT / "data" / "universe.txt"


def resolve_universe(db_path, tickers_arg) -> list[str]:
    """Full tradeable universe for the snapshot.

    Precedence: explicit --tickers > market.db (local) > committed
    universe.txt (CI) > curated TICKERS. When market.db is present the file is
    rewritten so the committed list stays current for CI runs.
    """
    if tickers_arg:
        return [t.upper() for t in tickers_arg]
    full = get_full_universe(db_path)          # 690 locally; falls back to TICKERS in CI
    if len(full) > len(TICKERS) + 5:           # market.db present with HF universe
        try:
            UNIVERSE_FILE.write_text("\n".join(full) + "\n", encoding="utf-8")
        except OSError:
            pass
        return full
    if UNIVERSE_FILE.exists():                  # CI: read the committed list
        names = [ln.strip() for ln in
                 UNIVERSE_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    return sorted(TICKERS)

INFO_FIELDS = {                     # info key -> column
    "targetMeanPrice": "target_mean", "targetHighPrice": "target_high",
    "targetLowPrice": "target_low", "recommendationMean": "rec_mean",
    "numberOfAnalystOpinions": "n_analysts", "forwardEps": "forward_eps",
    "forwardPE": "forward_pe", "sharesShort": "shares_short",
    "sharesShortPriorMonth": "shares_short_prior", "shortRatio": "short_ratio",
    "shortPercentOfFloat": "short_pct_float",
    "dateShortInterest": "date_short_interest",
    "sharesOutstanding": "shares_outstanding",
}


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS analyst_snapshot (
        snap_date TEXT NOT NULL, ticker TEXT NOT NULL,
        {', '.join(f'{c} REAL' for c in INFO_FIELDS.values())},
        PRIMARY KEY (snap_date, ticker))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS estimate_snapshot (
        snap_date TEXT NOT NULL, ticker TEXT NOT NULL, period TEXT NOT NULL,
        eps_current REAL, eps_7d_ago REAL, eps_30d_ago REAL,
        eps_60d_ago REAL, eps_90d_ago REAL,
        rev_up_7d REAL, rev_up_30d REAL, rev_down_30d REAL, rev_down_7d REAL,
        est_eps_avg REAL, est_eps_n REAL, est_rev_avg REAL,
        PRIMARY KEY (snap_date, ticker, period))""")
    conn.commit()


def snapped_today(conn: sqlite3.Connection, snap_date: str) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT ticker FROM analyst_snapshot WHERE snap_date = ?", (snap_date,))}


def _num(x):
    try:
        v = float(x)
        return v if v == v else None    # NaN -> None
    except (TypeError, ValueError):
        return None


def snapshot_ticker(tk, snap_date: str, ticker: str):
    """Returns (analyst_row, [estimate_rows]) or raises."""
    info = tk.info or {}
    arow = [snap_date, ticker] + [_num(info.get(k)) for k in INFO_FIELDS]

    erows = []
    try:
        trend = tk.eps_trend
        revs = tk.eps_revisions
        est_e = tk.earnings_estimate
        est_r = tk.revenue_estimate
        for period in ("0q", "+1q", "0y", "+1y"):
            def g(df, col):
                try:
                    return _num(df.loc[period, col])
                except Exception:
                    return None
            erows.append([
                snap_date, ticker, period,
                g(trend, "current"), g(trend, "7daysAgo"), g(trend, "30daysAgo"),
                g(trend, "60daysAgo"), g(trend, "90daysAgo"),
                g(revs, "upLast7days"), g(revs, "upLast30days"),
                g(revs, "downLast30days"), g(revs, "downLast7Days"),
                g(est_e, "avg"), g(est_e, "numberOfAnalysts"), g(est_r, "avg"),
            ])
    except Exception:
        pass    # estimates missing for some names; the analyst row still counts
    return arow, erows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="+", help="restrict to these tickers")
    ap.add_argument("--limit", type=int, help="snapshot at most N tickers")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds between tickers (be polite to Yahoo)")
    ap.add_argument("--db", default=str(SNAP_DB))
    args = ap.parse_args()

    import yfinance as yf

    # Ensure data/ directory exists (needed in CI where data/ is gitignored)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    snap_date = dt.date.today().isoformat()
    universe = resolve_universe(DB_PATH, args.tickers)
    conn = sqlite3.connect(args.db)
    ensure_tables(conn)
    done = snapped_today(conn, snap_date)
    todo = [t for t in universe if t not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{snap_date}: universe {len(universe)} | already snapped today "
          f"{len(done)} | to do {len(todo)}")

    n_ok = n_fail = 0
    t0 = time.time()
    a_cols = ", ".join(["snap_date", "ticker"] + list(INFO_FIELDS.values()))
    a_ph = ", ".join("?" * (2 + len(INFO_FIELDS)))
    for i, ticker in enumerate(todo, 1):
        try:
            arow, erows = snapshot_ticker(yf.Ticker(ticker), snap_date, ticker)
            conn.execute(f"INSERT OR REPLACE INTO analyst_snapshot ({a_cols}) "
                         f"VALUES ({a_ph})", arow)
            if erows:
                conn.executemany(
                    "INSERT OR REPLACE INTO estimate_snapshot VALUES "
                    "(" + ", ".join("?" * 15) + ")", erows)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"\nFAIL {ticker}: {type(e).__name__}: {str(e)[:120]}")
        if i % 10 == 0:
            conn.commit()
        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1) * 60
            print(f"\r{i}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                  f"{rate:.0f}/min  ETA {(len(todo)-i)/max(rate,1):.0f}m   ",
                  end="", flush=True)
        time.sleep(args.delay)
    conn.commit()

    n_days = conn.execute(
        "SELECT COUNT(DISTINCT snap_date) FROM analyst_snapshot").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM analyst_snapshot").fetchone()[0]
    conn.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min — ok={n_ok} fail={n_fail}; "
          f"{total:,} analyst rows across {n_days} snapshot day(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
