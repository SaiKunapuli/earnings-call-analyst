"""Layered ensemble — merges signals into one forecast per stock per day.

Implements Layer 9 of ``docs/trading_bot_layers.md``:
- Cross-sectional z-score per signal per date
- ECA event score decays linearly to 0 over 30 trading days
- Regime multiplier gates every signal daily
- Equal-weight blend (v1; IC-proportional weights once all signals validated)

Usage (daily production)::

    scores = compute_ensemble_scores(as_of_date)
    # DataFrame with columns: ticker, ensemble_z, momentum_z, eca_z, regime

Or run backfill to evaluate the combined signal historically.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH, BENCHMARKS
from src.momentum import build_momentum_panel, add_composite_score, MOMENTUM_COMPOSITE_SIGNS
from src.regime import compute_regime

# ECA score decays to zero over this many trading days from the event date.
# Matches the PEAD holding window (abnormal_30d).
ECA_DECAY_DAYS = 30

# v1: equal weight per active signal.  A signal is "active" on a given
# day for a given ticker if it produces a non-NaN z-score.
SIGNAL_WEIGHTS = {
    "momentum": 0.50,   # fires daily on every name
    "eca": 0.50,        # fires only around earnings (event + 30d decay)
}


def _load_prices_wide(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Wide (date x ticker) close prices.

    IMPORTANT: keeps SPY — ``build_momentum_panel`` needs it for beta
    computation and the beta-adjusted target.  The panel's own ticker
    filtering drops benchmarks from the output rows.
    """
    conn = sqlite3.connect(str(db_path))
    px = pd.read_sql(
        "SELECT date, ticker, close FROM prices ORDER BY date",
        conn, parse_dates=["date"],
    )
    conn.close()
    return px.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    ).sort_index()


def _load_volume_wide(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Wide (date x ticker) volume."""
    conn = sqlite3.connect(str(db_path))
    px = pd.read_sql(
        "SELECT date, ticker, volume FROM prices ORDER BY date",
        conn, parse_dates=["date"],
    )
    conn.close()
    return px.pivot_table(
        index="date", columns="ticker", values="volume", aggfunc="last"
    ).sort_index()


def momentum_daily_scores(
    wide: pd.DataFrame | None = None,
    volume_wide: pd.DataFrame | None = None,
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Daily cross-sectional momentum composite scores (0-1 rank).

    Computes the monthly momentum panel once, then forward-fills the
    composite rank to all trading days between rebalance dates.

    Returns a DataFrame with columns ``[date, ticker, momentum_z]`` where
    ``momentum_z`` is the cross-sectional z-score of the composite rank
    (winner = positive z, loser = negative z).
    """
    if wide is None:
        wide = _load_prices_wide(db_path)
    if volume_wide is None:
        volume_wide = _load_volume_wide(db_path)

    # Build the monthly panel with composite score
    panel = build_momentum_panel(wide, volume_wide=volume_wide, benchmarks=BENCHMARKS)
    panel = add_composite_score(panel, signs=MOMENTUM_COMPOSITE_SIGNS)

    # For each monthly rebalance date, compute cross-sectional z-score of composite
    panel["momentum_z"] = panel.groupby("date")["composite"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std() > 0 else 0
    )

    # Forward-fill monthly scores to daily
    daily_scores = panel[["date", "ticker", "momentum_z"]].copy()
    daily_scores["date"] = pd.to_datetime(daily_scores["date"])

    # Create a daily grid of all (date, ticker) pairs present in the price data
    all_dates = wide.index
    all_tickers = wide.columns.tolist()
    daily_grid = pd.MultiIndex.from_product(
        [all_dates, all_tickers], names=["date", "ticker"]
    ).to_frame(index=False)

    # Merge monthly scores onto the daily grid, forward-fill
    daily_grid = daily_grid.merge(
        daily_scores, on=["date", "ticker"], how="left"
    )
    daily_grid["momentum_z"] = daily_grid.groupby("ticker")["momentum_z"].ffill()

    # Drop days before the first monthly rebalance
    first_score_date = daily_scores["date"].min()
    daily_grid = daily_grid[daily_grid["date"] >= first_score_date].copy()

    # Cap forward-fill to the ticker's last known price date.
    # Without this, ffill carries delisted tickers' last momentum score
    # forward indefinitely into dates where they're no longer tradeable.
    last_price_dates = wide.notna()[::-1].idxmax()  # last valid price per ticker
    for ticker in daily_grid["ticker"].unique():
        if ticker in last_price_dates.index:
            last_d = pd.Timestamp(last_price_dates[ticker])
            if pd.isna(last_d):
                continue
            mask = (daily_grid["ticker"] == ticker) & (daily_grid["date"] > last_d)
            daily_grid.loc[mask, "momentum_z"] = np.nan

    return daily_grid.sort_values(["date", "ticker"]).reset_index(drop=True)


def eca_daily_scores(
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Daily ECA scores with linear decay over 30 trading days.

    Pulls predictions from ``model_predictions``, computes cross-sectional
    z-scores per event date, then decays each score linearly to zero over
    ``ECA_DECAY_DAYS`` trading days from the event date.

    Returns a DataFrame with columns ``[date, ticker, eca_z]``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        preds = pd.read_sql(
            "SELECT * FROM model_predictions", conn,
            parse_dates=["matched_earnings_date"],
        )
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError):
        conn.close()
        return pd.DataFrame(columns=["date", "ticker", "eca_z"])
    conn.close()

    if preds.empty:
        return pd.DataFrame(columns=["date", "ticker", "eca_z"])

    # Find the prediction column (handles _va variants)
    pred_cols = sorted(
        [c for c in preds.columns if c.startswith("predicted_")],
        key=lambda c: c.endswith("_va"),
    )
    if not pred_cols:
        return pd.DataFrame(columns=["date", "ticker", "eca_z"])

    pred_col = pred_cols[0]  # prefer raw-space prediction

    # Cross-sectional z-score per event date
    event = preds[["ticker", "matched_earnings_date", pred_col]].dropna().copy()
    z_col = "eca_z"
    event[z_col] = event.groupby("matched_earnings_date")[pred_col].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std() > 0 else 0
    )

    # Get all trading days from the prices table
    conn = sqlite3.connect(str(db_path))
    trading_days = pd.read_sql(
        "SELECT DISTINCT date FROM prices ORDER BY date",
        conn, parse_dates=["date"],
    )
    conn.close()
    trading_days = trading_days["date"].sort_values().values

    # For each event, spread the score over the next ECA_DECAY_DAYS trading days
    rows = []
    for _, ev in event.iterrows():
        event_date = pd.Timestamp(ev["matched_earnings_date"])
        score = ev[z_col]
        ticker = ev["ticker"]

        # Find all trading days in the decay window
        mask = (trading_days >= np.datetime64(event_date)) & (
            trading_days <= np.datetime64(event_date + pd.Timedelta(days=60))
        )
        future_days = trading_days[mask]
        if len(future_days) == 0:
            continue
        # Take up to ECA_DECAY_DAYS trading days
        future_days = future_days[:ECA_DECAY_DAYS]
        for i, d in enumerate(future_days):
            # Linear decay: score * (1 - i / ECA_DECAY_DAYS)
            decay_factor = 1.0 - i / ECA_DECAY_DAYS
            rows.append({
                "date": pd.Timestamp(d),
                "ticker": ticker,
                "eca_z": score * decay_factor,
            })

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "eca_z"])

    result = pd.DataFrame(rows)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def compute_ensemble_scores(
    as_of_date: str | pd.Timestamp | None = None,
    db_path: str | Path = DB_PATH,
    gate: bool = False,
) -> pd.DataFrame:
    """Compute ensemble scores for all tickers on a given date.

    If ``as_of_date`` is ``None``, computes the full history.

    ``gate``: multiply the blend by the daily regime multiplier before the
    final cross-sectional z-score. DEFAULT FALSE — the 2026-07 backtest
    (scratch: ensemble_backtest) found the gate LOWERS Sharpe on the strong
    momentum+ECA book (1.75 -> 1.66) with no drawdown benefit in a calm
    window; it only helps a *losing* signal by dodging crashes. Treat the
    gate as optional drawdown insurance, not default alpha. The ``regime``
    column is always returned for reference/sizing regardless of this flag.

    Returns a DataFrame with columns:
    - date, ticker
    - momentum_z, eca_z  (cross-sectional z-scores per signal)
    - regime              (daily multiplier: 1.0, 0.6, or 0.25)
    - ensemble_z          (equal-weight blend [x regime if gate], cross-sectional)
    """
    # Load regime (daily)
    regime = compute_regime(db_path)[["multiplier"]].copy()
    regime.index.name = "date"
    regime = regime.reset_index()

    # Load momentum scores (daily)
    momentum = momentum_daily_scores(db_path=db_path)

    # Load ECA scores (daily, with decay)
    eca = eca_daily_scores(db_path=db_path)

    # Merge all onto trading day grid.
    # Aggregate ECA first: when multiple events for the same ticker overlap
    # in their decay windows, take the strongest (most recent) signal.
    if not eca.empty:
        eca = eca.groupby(["date", "ticker"], as_index=False)["eca_z"].max()
    scores = momentum.merge(eca, on=["date", "ticker"], how="outer")
    scores = scores.merge(regime, on="date", how="left")

    # Fill missing signals and regime
    scores["momentum_z"] = scores["momentum_z"].fillna(0)
    scores["eca_z"] = scores["eca_z"].fillna(0)
    scores["multiplier"] = scores["multiplier"].fillna(1.0)

    # Equal-weight blend (only count active signals per ticker-day)
    w_mom = SIGNAL_WEIGHTS["momentum"]
    w_eca = SIGNAL_WEIGHTS["eca"]
    scores["raw_blend"] = scores["momentum_z"] * w_mom + scores["eca_z"] * w_eca

    # Apply the regime gate only when requested (see docstring — off by default)
    scores["final_blend"] = (scores["raw_blend"] * scores["multiplier"]
                             if gate else scores["raw_blend"])

    # Cross-sectional z-score of the blend within each date
    scores["ensemble_z"] = scores.groupby("date")["final_blend"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std() > 0 else 0
    )

    # Filter to a specific date if requested
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        scores = scores[scores["date"] == as_of]

    result = scores[["date", "ticker", "momentum_z", "eca_z", "multiplier", "ensemble_z"]].copy()
    result = result.rename(columns={"multiplier": "regime"})
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def ensemble_summary(scores: pd.DataFrame) -> dict:
    """Quick stats on the ensemble scores."""
    n_dates = scores["date"].nunique()
    n_tickers = scores["ticker"].nunique()
    has_momentum = (scores["momentum_z"].abs() > 1e-8).mean()
    has_eca = (scores["eca_z"].abs() > 1e-8).mean()
    avg_regime = scores["regime"].mean()
    return {
        "n_dates": n_dates,
        "n_tickers": n_tickers,
        "pct_momentum_active": has_momentum * 100,
        "pct_eca_active": has_eca * 100,
        "avg_regime_multiplier": avg_regime,
        "date_range": (scores["date"].min(), scores["date"].max()),
    }
