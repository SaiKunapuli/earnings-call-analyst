"""Macro regime gate — a rules-based exposure multiplier (Layer 7).

Decides *when* to trade, not *what*.  Every signal in the ensemble gets
multiplied by the daily gate score, so the system sizes down
automatically in bad regimes and goes flat in crashes.

Design constraints (from ``docs/trading_bot_layers.md``):
- **Rules-based** — thresholds, not ML.  Few knobs are an overfit magnet.
- **All inputs are free** — SPY, VIX, HYG, LQD, universe breadth.
- **Risk-off does NOT kill ECA/PEAD** — earnings drift still works in
  stress (sometimes better), just at reduced size.  Momentum gets turned
  OFF in risk-off (crashes are reversal events).
- **Thresholds are standard market conventions**, not optimized on the
  ECA eval window.

Output:
    A daily ``regime`` Series with values in {1.0, 0.6, 0.25}:
    - 1.0  risk-on  — full exposure
    - 0.6  caution  — reduced (elevated VIX, SPY near 200dma)
    - 0.25 risk-off — minimal (VIX spike, credit stress, broad breakdown)
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH, BENCHMARKS


# ---------------------------------------------------------------------------
# Signal thresholds (market convention, NOT optimized on eval data)
# ---------------------------------------------------------------------------

VIX_CAUTION = 25.0       # VIX > 25 -> caution
VIX_RISK_OFF = 35.0      # VIX > 35 -> risk-off

SPY_DMA_CAUTION = 0.0    # SPY below 200dma -> caution (any amount below)
SPY_DMA_RISK_OFF = -0.10  # SPY >10% below 200dma -> risk-off

CREDIT_WINDOW = 20       # HYG-vs-LQD return over this many trading days
CREDIT_CAUTION = -0.02   # HYG underperforming LQD by 2%+ -> caution (one notch)
CREDIT_RISK_OFF = -0.05  # HYG underperforming LQD by 5%+ -> risk-off

BREADTH_WINDOW = 200     # 200-day moving average for breadth
BREADTH_CAUTION = 0.50   # <50% above 200dma -> caution
BREADTH_RISK_OFF = 0.30  # <30% above 200dma -> risk-off

CREDIT_TICKERS = ("HYG", "LQD")  # high-yield vs investment-grade corporate bond ETFs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_spy(db_path: str | Path = DB_PATH) -> pd.Series:
    """Daily SPY closes from the prices table."""
    conn = sqlite3.connect(str(db_path))
    px = pd.read_sql(
        "SELECT date, close FROM prices WHERE ticker = 'SPY' ORDER BY date",
        conn, parse_dates=["date"], index_col="date",
    )
    conn.close()
    return px["close"].rename("spy").dropna()


def _load_vix(db_path: str | Path = DB_PATH) -> pd.Series:
    """Daily VIX closes from the vix table."""
    conn = sqlite3.connect(str(db_path))
    try:
        # notebook 01 persists the column as `vix_close`, not `close`.
        vix = pd.read_sql(
            "SELECT date, vix_close AS close FROM vix ORDER BY date",
            conn, parse_dates=["date"], index_col="date",
        )
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        # LOUD degradation: a silent fallback here once ran the gate on 3 of 4
        # inputs for days with nobody noticing (journal §4.14).
        warnings.warn(f"regime gate: VIX input unavailable ({e}) — "
                      "gate will run WITHOUT the VIX signal", stacklevel=2)
        vix = pd.DataFrame(columns=["close"])
    conn.close()
    return vix["close"].rename("vix").dropna() if not vix.empty else pd.Series(dtype=float)


def _load_credit(
    start: str = "2016-01-01",
) -> pd.DataFrame:
    """HYG and LQD closes from Yahoo Finance (free, few tickers).

    Returns an empty DataFrame on failure — the credit signal will be
    neutral.  No caching (acceptable for notebook use; wrap with a
    persistent cache before using in a daily production loop).
    """
    import yfinance as yf
    try:
        raw = yf.download(list(CREDIT_TICKERS), start=start, auto_adjust=True,
                          progress=False)
        closes = raw.get("Close", raw)
        if isinstance(closes, pd.DataFrame) and len(closes.columns) >= 2:
            return closes
        warnings.warn("regime gate: HYG/LQD download returned no usable columns "
                      "— gate will run WITHOUT the credit signal", stacklevel=2)
    except Exception as e:
        warnings.warn(f"regime gate: HYG/LQD download failed ({e}) — "
                      "gate will run WITHOUT the credit signal", stacklevel=2)
    return pd.DataFrame()  # empty — credit signal will be neutral


def _load_universe_prices(
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Wide (date x ticker) close prices for all non-benchmark tickers."""
    conn = sqlite3.connect(str(db_path))
    px = pd.read_sql(
        "SELECT date, ticker, close FROM prices ORDER BY date",
        conn, parse_dates=["date"],
    )
    conn.close()
    wide = px.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    ).sort_index()
    # Drop benchmark columns
    for b in BENCHMARKS:
        if b in wide.columns:
            wide = wide.drop(columns=b)
    return wide


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _signal_vix(vix: pd.Series) -> pd.Series:
    """0 (low), -0.5 (elevated), -1 (stress).

    Low VIX is neutral, not bullish — it never adds to the composite."""
    s = pd.Series(0, index=vix.index, dtype=float)
    s[vix > VIX_CAUTION] = -0.5
    s[vix > VIX_RISK_OFF] = -1.0
    return s


def _signal_spy_dma(spy: pd.Series) -> pd.Series:
    """+1 (above), 0 (near/below), -1 (well below)."""
    dma = spy.rolling(200, min_periods=200).mean()
    pct_from_dma = spy / dma - 1.0
    s = pd.Series(1, index=spy.index, dtype=float)
    s[pct_from_dma < SPY_DMA_CAUTION] = 0.0
    s[pct_from_dma < SPY_DMA_RISK_OFF] = -1.0
    return s


def _signal_credit(credit: pd.DataFrame) -> pd.Series:
    """HYG vs LQD relative return: +1 (risk-on credit), -1 (stress)."""
    if credit.empty or "HYG" not in credit.columns or "LQD" not in credit.columns:
        return pd.Series(dtype=float)  # empty — neutral
    hyg_lqd = credit["HYG"] / credit["LQD"]
    rel_ret = hyg_lqd.pct_change(CREDIT_WINDOW)
    s = pd.Series(0, index=rel_ret.index, dtype=float)
    s[rel_ret < CREDIT_CAUTION] = -0.5
    s[rel_ret < CREDIT_RISK_OFF] = -1.0
    return s


def _signal_breadth(wide: pd.DataFrame) -> pd.Series:
    """% of universe above 200dma: +1 (broad), 0 (mixed), -1 (narrow)."""
    dma = wide.rolling(BREADTH_WINDOW, min_periods=200).mean()
    above = (wide > dma).sum(axis=1)
    count = wide.notna().sum(axis=1)
    pct_above = above / count.replace(0, np.nan)
    s = pd.Series(1, index=pct_above.index, dtype=float)
    s[pct_above < BREADTH_CAUTION] = 0.0
    s[pct_above < BREADTH_RISK_OFF] = -1.0
    return s


# ---------------------------------------------------------------------------
# Gate — combines signals into a daily multiplier
# ---------------------------------------------------------------------------

def compute_regime(
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Compute the daily regime multiplier.

    Returns a DataFrame with columns:
      - multiplier  : {1.0, 0.6, 0.25}
      - vix_signal, spy_signal, credit_signal, breadth_signal : component scores
      - vix_close   : raw VIX for reference
    """
    spy = _load_spy(db_path)
    vix = _load_vix(db_path)
    credit = _load_credit()
    wide = _load_universe_prices(db_path)

    # Align all signals to SPY dates
    common_idx = spy.index

    vix_signal = _signal_vix(vix).reindex(common_idx).fillna(0)
    spy_signal = _signal_spy_dma(spy).reindex(common_idx).fillna(0)
    breadth_signal = _signal_breadth(wide).reindex(common_idx).fillna(0)

    if not credit.empty:
        credit_signal = _signal_credit(credit).reindex(common_idx).fillna(0)
    else:
        credit_signal = pd.Series(0, index=common_idx)

    # Composite: most restrictive signal wins — if ANY signal says risk-off,
    # the gate goes risk-off.  No composite softening of a VIX spike.
    #
    # Default: risk-on (1.0)
    multiplier = pd.Series(1.0, index=common_idx)
    # Risk-off: any signal at -1.0 (extreme)
    risk_off_mask = (
        (vix_signal <= -1.0) | (spy_signal <= -1.0)
        | (breadth_signal <= -1.0) | (credit_signal <= -1.0)
    )
    # Caution: any signal negative but not risk-off
    caution_mask = (
        (~risk_off_mask)
        & (
            (vix_signal < 0) | (spy_signal <= 0)
            | (breadth_signal <= 0) | (credit_signal < 0)
        )
    )
    multiplier[risk_off_mask] = 0.25
    multiplier[caution_mask] = 0.60
    # Informational composite (sum of all signals) for diagnostics
    composite = vix_signal + spy_signal + breadth_signal + credit_signal

    result = pd.DataFrame({
        "multiplier": multiplier,
        "vix_signal": vix_signal,
        "spy_signal": spy_signal,
        "credit_signal": credit_signal,
        "breadth_signal": breadth_signal,
        "composite": composite,
    }, index=common_idx)

    if not vix.empty:
        result["vix_close"] = vix.reindex(common_idx)

    return result.sort_index()


def regime_summary(regime: pd.DataFrame) -> dict:
    """Quick stats: how often each regime occurs."""
    total = len(regime)
    if total == 0:
        return {"risk_on_pct": 0, "caution_pct": 0, "risk_off_pct": 0}
    return {
        "risk_on_pct": (regime["multiplier"] == 1.0).mean() * 100,
        "caution_pct": (regime["multiplier"] == 0.60).mean() * 100,
        "risk_off_pct": (regime["multiplier"] == 0.25).mean() * 100,
        "dates": (regime.index[0], regime.index[-1]),
    }
