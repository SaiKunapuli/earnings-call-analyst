"""Multiprocess VADER / Loughran-McDonald / readability featurization.

The single-threaded loop in 03_sentiment.ipynb takes on the order of
half a day for the 33k-transcript HF universe — the work is pure-Python
regex/dict scoring, so it parallelizes almost linearly across cores.

Design (Windows ``spawn``-safe):
- Workers are module-level functions (picklable) initialized once per
  process: own SQLite connection (WAL readers don't block each other),
  own VADER analyzer and LM dictionary.
- Each task is one (ticker, quarter, year) key; the worker fetches the
  compressed sections, cleans, scores, and returns a flat feature dict —
  transcript text never crosses process boundaries.
- ``executor.map`` with a chunksize keeps IPC overhead low and preserves
  input order, so results align 1:1 with the input key list.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

# Per-process state, populated by _init_worker (never shared/pickled).
_STATE: dict = {}


def _init_worker(db_path: str) -> None:
    import sqlite3

    import nltk
    import pysentiment2 as ps2
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    for pkg in ("vader_lexicon", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)

    _STATE["conn"] = sqlite3.connect(db_path)
    _STATE["vader"] = SentimentIntensityAnalyzer()
    _STATE["lm"] = ps2.LM()


def _score_one(key: tuple) -> dict:
    """Fetch, clean, and score one transcript; returns a flat feature row."""
    from src.sentiment import clean_transcript, compute_all_sentiment_features
    from src.transcripts_io import fetch_sections

    ticker, quarter, year = key
    sections = fetch_sections(_STATE["conn"], ticker, quarter, year)
    row = {"ticker": ticker, "quarter": quarter, "year": year,
           "has_qa": int("qa" in sections)}
    for name, text in sections.items():
        cleaned = clean_transcript(text)
        row.update(compute_all_sentiment_features(
            cleaned, _STATE["vader"], _STATE["lm"], prefix=name))
    return row


def compute_sentiment_features_parallel(
    db_path,
    keys: list[tuple],
    max_workers: int | None = None,
    chunksize: int = 16,
    log_every: int = 1000,
) -> pd.DataFrame:
    """Score *keys* (list of ``(ticker, quarter, year)``) across CPU cores.

    Returns one row per key, in input order.  ``max_workers`` defaults to
    ``os.cpu_count() - 1`` (leave a core for the kernel/UI).
    """
    if not keys:
        return pd.DataFrame()
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    rows: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(db_path),),
    ) as ex:
        for i, row in enumerate(ex.map(_score_one, keys, chunksize=chunksize)):
            rows.append(row)
            if log_every and (i + 1) % log_every == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(keys) - i - 1)
                print(f"  [{i + 1:,}/{len(keys):,}] {el / 60:.1f} min elapsed, "
                      f"~{eta / 60:.0f} min left ({(i + 1) / el:.1f} rows/s, "
                      f"{max_workers} workers)")
    return pd.DataFrame(rows)
