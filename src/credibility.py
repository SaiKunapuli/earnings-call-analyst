"""Management-credibility signal — do a team's words match subsequent reality?

Turns weak call language into a slow-moving quality factor: instead of "does
this call sound positive?" (static sentiment, which the ablation showed is
~useless alone), it asks "when THIS management team sounds positive, does it
come true?" — a per-team track record built over years.

Mechanism (all leak-free / point-in-time):
  1. optimism_score  — collapse the LLM's forward claims (guidance direction,
                       demand/margin outlook) into one "how bullish about the
                       future" number in ~[-1, +1].
  2. grade           — did that optimism match the NEXT quarter's realized EPS
                       surprise? Sign agreement in {+1, 0, -1}. Ground truth is
                       market fundamentals (the surprise), NOT the LLM and NOT
                       the stock return — so it's non-circular with the PEAD
                       target and carries no LLM look-ahead.
  3. credibility     — per ticker, the expanding PRIOR mean of past agreements
                       (shifted; the current call is excluded). A team that has
                       been right accumulates high credibility.
  4. signal          — cred_weighted_optimism = credibility x current optimism.
                       Trust the optimism of teams that have earned it; discount
                       a chronic over-promiser. This is the tradeable feature.

Consumes the LLM pilot output (`llm_qa_scores`) + the `earnings` table. Depends
on pilot data existing, so it only produces signal once the pilot has run.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH

# Forward-looking claim fields and how they map to a [-1, +1] optimism scale.
# guidance_direction is already -1/0/+1; the outlooks are -2..+2 -> /2.
_CLAIM_WEIGHTS = {
    "guidance_direction": 1.0,      # already -1..+1
    "demand_outlook": 0.5,          # -2..+2 -> -1..+1
    "margin_outlook": 0.5,
}


def optimism_score(scores: dict) -> float:
    """Collapse forward claims into one ~[-1, +1] 'forward optimism' number.
    Averages whichever of the claim fields are present (guidance may be null)."""
    vals = []
    for field, scale in _CLAIM_WEIGHTS.items():
        v = scores.get(field)
        if v is None:
            continue
        vals.append(scale * float(v))
    return float(np.mean(vals)) if vals else 0.0


def grade_agreement(optimism: float, next_surprise: float | None,
                    dead: float = 0.0) -> float:
    """+1 if optimism and the next-quarter EPS surprise share a sign, -1 if they
    disagree, 0 if either is missing or (near-)zero. ``dead`` = magnitude below
    which a value is treated as no-signal."""
    if next_surprise is None or not np.isfinite(next_surprise):
        return 0.0
    if abs(optimism) <= dead or abs(next_surprise) <= dead:
        return 0.0
    return 1.0 if np.sign(optimism) == np.sign(next_surprise) else -1.0


def build_credibility(calls: pd.DataFrame) -> pd.DataFrame:
    """Add ``agreement``, ``credibility`` and ``cred_weighted_optimism`` columns.

    ``calls`` needs ``[ticker, date, optimism, next_surprise]``. Credibility is
    the expanding mean of a ticker's PRIOR agreements (shifted one call, so the
    current call never sees its own outcome) — NaN until a track record exists;
    the weighted signal treats no-track-record as neutral (0)."""
    df = calls.sort_values(["ticker", "date"]).copy()
    df["agreement"] = [grade_agreement(o, s)
                       for o, s in zip(df["optimism"], df["next_surprise"])]
    df["credibility"] = (df.groupby("ticker")["agreement"]
                         .transform(lambda s: s.expanding().mean().shift(1)))
    df["cred_weighted_optimism"] = df["credibility"].fillna(0.0) * df["optimism"]
    return df


def load_credibility_features(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Wire it to the DB: join llm_qa_scores -> call date (transcripts_text) ->
    next-quarter EPS surprise (earnings), then build the credibility feature.

    Returns ``[ticker, quarter, year, date, optimism, credibility,
    cred_weighted_optimism]`` — empty if the pilot hasn't scored anything yet.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        llm = pd.read_sql("SELECT ticker, quarter, year, scores FROM llm_qa_scores", conn)
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError):
        conn.close()
        return pd.DataFrame(columns=["ticker", "quarter", "year", "date",
                                     "optimism", "credibility", "cred_weighted_optimism"])
    if llm.empty:
        conn.close()
        return pd.DataFrame(columns=["ticker", "quarter", "year", "date",
                                     "optimism", "credibility", "cred_weighted_optimism"])

    # call date = transcript pub_date
    tx = pd.read_sql("SELECT ticker, quarter, year, pub_date FROM transcripts_text",
                     conn, parse_dates=["pub_date"])
    earn = pd.read_sql("SELECT ticker, earnings_date, eps_surprise_pct FROM earnings",
                       conn, parse_dates=["earnings_date"])
    conn.close()

    llm["optimism"] = llm["scores"].map(lambda s: optimism_score(json.loads(s)))
    df = llm.merge(tx, on=["ticker", "quarter", "year"], how="left").rename(
        columns={"pub_date": "date"}).dropna(subset=["date"])

    # next-quarter surprise = first earnings event strictly after the call date
    earn = earn.dropna(subset=["eps_surprise_pct"]).sort_values("earnings_date")
    df = df.sort_values("date")
    nxt = pd.merge_asof(
        df[["ticker", "date"]], earn.rename(columns={"earnings_date": "date"}),
        on="date", by="ticker", direction="forward",
        allow_exact_matches=False)
    df["next_surprise"] = nxt["eps_surprise_pct"].values

    out = build_credibility(df)
    return out[["ticker", "quarter", "year", "date", "optimism",
                "credibility", "cred_weighted_optimism"]].reset_index(drop=True)
