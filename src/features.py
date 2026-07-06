"""Leak-free panel feature engineering for the modeling notebook.

Every function here is *point-in-time correct*: a feature value for an
event only uses information available strictly before (or at) that
event's date.  This is the difference between a backtest and a mirage —
the original ``compute_ticker_z_scores`` in :mod:`src.sentiment`
normalizes with each ticker's FULL-sample mean/std, which lets every
prediction peek at future earnings calls.  Use these instead for
anything that feeds a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sentiment import SENTIMENT_Z_COLS

_EPS = 1e-12


def _expanding_past(grouped, fn: str) -> pd.Series:
    """Expanding statistic over strictly PRIOR observations (shifted 1)."""
    return grouped.transform(lambda s: getattr(s.expanding(), fn)().shift(1))


def compute_ticker_z_scores_expanding(
    df: pd.DataFrame,
    prefix: str = "full",
    min_obs: int = 3,
    date_col: str = "matched_earnings_date",
) -> pd.DataFrame:
    """Within-ticker z-scores using only each event's PAST observations.

    Drop-in replacement for ``sentiment.compute_ticker_z_scores`` (same
    ``{prefix}_{col}_z`` output columns) minus the look-ahead bias:
    the mean/std for event *t* come from that ticker's events strictly
    before *t*.  Events with fewer than *min_obs* prior observations get
    NaN.
    """
    df = df.copy()
    src_cols = [f"{prefix}_{c}" for c in SENTIMENT_Z_COLS
                if f"{prefix}_{c}" in df.columns]
    if not src_cols:
        return df

    tmp = df[["ticker", date_col] + src_cols].sort_values(date_col)
    for src in src_cols:
        g = tmp.groupby("ticker")[src]
        mu = _expanding_past(g, "mean")
        sigma = _expanding_past(g, "std")
        n = _expanding_past(g, "count")
        z = (tmp[src] - mu) / sigma
        z[(n < min_obs) | (sigma < _EPS)] = np.nan
        df[f"{src}_z"] = z  # index-aligned back to original row order
    return df


def add_qoq_deltas(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "matched_earnings_date",
    suffix: str = "_qoq",
) -> pd.DataFrame:
    """Change vs. the same ticker's PREVIOUS call for each column in *cols*.

    Changes in tone/hedging quarter-over-quarter are typically more
    informative than levels (a CEO who is always sunny drops to "cautious"
    — that's the event).  Leak-free: only the prior call is referenced.
    Missing columns are skipped; the first call per ticker gets NaN.
    """
    df = df.copy()
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    tmp = df[["ticker", date_col] + present].sort_values(date_col)
    g = tmp.groupby("ticker")
    for c in present:
        df[f"{c}{suffix}"] = tmp[c] - g[c].shift(1)
    return df


def add_past_target_stats(
    df: pd.DataFrame,
    target: str = "abnormal_1d",
    date_col: str = "matched_earnings_date",
    min_obs: int = 2,
) -> pd.DataFrame:
    """Expanding mean/std of the ticker's PAST target values.

    A PEAD-style prior: some names habitually pop or fade after earnings.
    Both stats exclude the current event (shifted expanding window) and
    are NaN until *min_obs* prior events exist.
    """
    df = df.copy()
    tmp = df[["ticker", date_col, target]].sort_values(date_col)
    g = tmp.groupby("ticker")[target]
    n = _expanding_past(g, "count")
    mean_ = _expanding_past(g, "mean")
    std_ = _expanding_past(g, "std")
    df[f"past_{target}_mean"] = mean_.where(n >= min_obs)
    df[f"past_{target}_std"] = std_.where(n >= min_obs)
    return df


# ---------------------------------------------------------------------------
# FinBERT section combining (compute optimization, not a leak fix)
# ---------------------------------------------------------------------------

def combine_finbert_sections(
    prep: dict | None, qa: dict | None,
    prep_words: int, qa_words: int,
) -> dict:
    """Derive full-transcript FinBERT scores from the two section scores.

    ``full`` is the concatenation of prepared remarks and Q&A, and FinBERT
    chunk probabilities are averaged per section — so the full-document
    average is just the word-count-weighted average of the section
    averages.  Scoring only the sections and combining saves ~40% of GPU
    time versus re-scoring the concatenated text.

    Sections that are missing/empty (None, zero words, or NaN scores)
    contribute zero weight; if neither section is usable all scores are
    NaN, mirroring ``_finbert_score`` on empty text.
    """
    parts = []
    for scores, words in ((prep, prep_words), (qa, qa_words)):
        if not scores or not words or words <= 0:
            continue
        if any(np.isnan(scores.get(f"finbert_{k}", np.nan))
               for k in ("positive", "negative", "neutral")):
            continue
        parts.append((scores, float(words)))

    if not parts:
        return {
            "finbert_positive": np.nan, "finbert_negative": np.nan,
            "finbert_neutral": np.nan, "finbert_net": np.nan,
            "finbert_label": "neutral", "finbert_chunks": 0,
        }

    total = sum(w for _, w in parts)
    probs = {
        k: sum(s[f"finbert_{k}"] * w for s, w in parts) / total
        for k in ("positive", "negative", "neutral")
    }
    return {
        "finbert_positive": probs["positive"],
        "finbert_negative": probs["negative"],
        "finbert_neutral": probs["neutral"],
        "finbert_net": probs["positive"] - probs["negative"],
        "finbert_label": max(probs, key=probs.get),
        "finbert_chunks": int(sum(s.get("finbert_chunks", 0) for s, _ in parts)),
    }
