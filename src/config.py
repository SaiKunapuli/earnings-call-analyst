"""Shared project configuration.

Single source of truth for the ticker universe, benchmarks, date ranges and
database path.  Notebooks 01/02 previously each hard-coded their own copies
of these lists — import from here instead so they can never drift apart.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root; notebooks add ".." to sys.path first)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "market.db"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

START_DATE = "2016-07-01"
END_DATE = "2026-06-25"

COVID_START = "2020-02-20"
COVID_END = "2021-06-30"

# ---------------------------------------------------------------------------
# Core ticker universe (curated 57 names across 11 sectors).
# The HF transcript ingest (02b) can extend this dynamically — see
# get_full_universe() below.
# ---------------------------------------------------------------------------

TICKERS = [
    # Technology
    "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD",
    "INTC", "CRM", "ORCL", "ADBE", "CSCO", "IBM", "NOW", "NFLX",
    "DOCU", "TWLO", "PINS", "SNAP", "NET", "DDOG", "SQ", "ROKU",
    "CRWD", "ZM", "TEAM",
    # Financials
    "JPM", "BAC", "GS", "COF",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT",
    # Consumer Cyclical
    "HD", "SBUX", "NKE", "UBER", "DASH",
    # Consumer Defensive
    "WMT", "COST", "PG", "KO",
    # Industrials
    "CAT", "BA", "GE", "DE", "DAL",
    # Energy
    "XOM", "CVX", "DVN",
    # Materials
    "FCX", "NEM",
    # Real Estate
    "PLD", "SPG",
    # Utilities
    "NEE", "DUK",
    # Communication Services / other
    "DIS", "VZ",
]

BENCHMARKS = ["SPY", "XLK", "IWM"]


def get_hf_universe(db_path=None) -> list[str]:
    """Return distinct tickers present in the ``transcripts_text`` table.

    Empty list if the table does not exist yet (02b not run).
    Cheap: single indexed DISTINCT query, no blob columns touched.
    """
    import sqlite3

    path = str(db_path or DB_PATH)
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM transcripts_text"
            ).fetchall()
        return sorted(r[0] for r in rows)
    except sqlite3.OperationalError:  # table missing
        return []


def get_full_universe(db_path=None) -> list[str]:
    """Curated tickers + every ticker with an ingested HF transcript."""
    return sorted(set(TICKERS) | set(get_hf_universe(db_path)))
