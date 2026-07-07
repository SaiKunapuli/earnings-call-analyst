"""Momentum & reversal features for the Layer 2 signal.

Produces a monthly cross-sectional panel from daily price data under the
same leak-free discipline as ``src/features.py``: every feature value for
a rebalance date uses only information available on/before that date.

Key architectural decisions (from the thinker analysis):
- Stay **monthly** for research integrity — daily scoring of overlapping
  returns inflates t-stats via serial correlation.
- In production (Layer 11), the monthly-trained model's features & scores
  can be computed daily from the same formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sentiment import SECTOR_MAP


# ---------------------------------------------------------------------------
# Core feature computation from a wide price matrix
# ---------------------------------------------------------------------------

def compute_momentum_features(
    wide: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute raw momentum/reversal feature matrices from wide prices.

    ``wide`` is a (date x ticker) DataFrame of adjusted closes, sorted by
    date.  Returns a dict of same-shape DataFrames keyed by feature name.

    All rolling windows are strictly backward-looking (``.shift(1)`` where
    needed) so there is zero look-ahead.
    """
    px = wide.sort_index()
    rets = px.pct_change(fill_method=None)
    spy_ret = rets.get("SPY", pd.Series(0, index=rets.index))

    # Momentum horizons (skip the reversal month)
    r21 = px / px.shift(21) - 1
    r126 = px / px.shift(126) - 1
    r252 = px / px.shift(252) - 1
    mom_12_1 = (1 + r252) / (1 + r21) - 1
    mom_6_1 = (1 + r126) / (1 + r21) - 1
    rev_5d = px / px.shift(5) - 1       # short-term reversal, shorter than 21d
    rev_21d = r21

    # 52-week high distance & trailing vol
    dist_52wk = px / px.rolling(252, min_periods=200).max() - 1
    vol_60d = rets.rolling(60, min_periods=40).std()
    vol_20d = rets.rolling(20, min_periods=15).std()

    # Rolling beta (252-day, min 126 obs)
    m = 252
    cov = (
        rets.mul(spy_ret, axis=0).rolling(m, min_periods=126).mean()
        - rets.rolling(m, min_periods=126).mean()
        .mul(spy_ret.rolling(m, min_periods=126).mean(), axis=0)
    )
    beta_252 = cov.div(spy_ret.rolling(m, min_periods=126).var(), axis=0)

    # Idiosyncratic momentum (market-stripped)
    idio_mom = mom_12_1.sub(beta_252.mul(mom_12_1.get("SPY", pd.Series(0, index=px.index)), axis=0))

    # Liquidity / attention features
    # dollar volume from close * volume — compute from wide + volume wide
    # (volume is optional; if unavailable these features are NaN)

    # Vol-adjusted momentum (HIGH-IMPACT: divides raw mom by trailing vol
    # so the model can't just pick high-vol lottery tickets).
    vol_60d_safe = vol_60d.where(vol_60d > 1e-8)
    mom_12_1_va = mom_12_1 / vol_60d_safe
    mom_6_1_va = mom_6_1 / vol_60d_safe
    idio_mom_va = idio_mom / vol_60d_safe
    rev_5d_va = rev_5d / vol_20d.where(vol_20d > 1e-8)

    features = {
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "rev_5d": rev_5d,
        "rev_21d": rev_21d,
        "dist_52wk": dist_52wk,
        "vol_60d": vol_60d,
        "vol_20d": vol_20d,
        "beta_252": beta_252,
        "idio_mom": idio_mom,
        # Vol-adjusted variants (the key improvement)
        "mom_12_1_va": mom_12_1_va,
        "mom_6_1_va": mom_6_1_va,
        "idio_mom_va": idio_mom_va,
        "rev_5d_va": rev_5d_va,
    }
    return features


# ---------------------------------------------------------------------------
# Panel builder — slices features at monthly rebalance dates and applies
# sector-neutral ranks + beta-adjusted target
# ---------------------------------------------------------------------------

def build_momentum_panel(
    wide: pd.DataFrame,
    volume_wide: pd.DataFrame | None = None,
    benchmarks: tuple[str, ...] = ("SPY", "XLK", "IWM"),
    hold_days: int = 21,
) -> pd.DataFrame:
    """Build a monthly (date x ticker) panel for momentum modelling.

    Parameters
    ----------
    wide : (date x ticker) close prices.  Must be sorted by date.
    volume_wide : optional (date x ticker) share volume for liquidity feats.
    benchmarks : tickers to exclude from the panel (SPY, XLK, IWM).
    hold_days : forward window for the target return.

    Returns
    -------
    panel : DataFrame with columns [date, ticker, sector] + feature cols +
            raw_target + target.
    """
    # 1. Compute raw features
    feats = compute_momentum_features(wide)

    # 2. Liquidity features (if volume data available)
    if volume_wide is not None:
        vol = volume_wide.sort_index()
        dv = wide * vol  # dollar volume
        adv21 = dv.rolling(21, min_periods=15).mean()
        adv126 = dv.rolling(126, min_periods=60).mean()
        feats["turn_trend"] = adv21 / adv126.where(adv126 > 0) - 1
        feats["log_adv"] = np.log(adv21.where(adv21 > 0))

    # 3. Generate monthly rebalance dates (month-end, with warm-up)
    spy = wide.get("SPY", wide.iloc[:, 0])
    month_last = wide.index.to_series().groupby(
        wide.index.to_period("M")
    ).max()
    reb_dates = [
        d for d in month_last
        if d >= wide.index[260] and d <= wide.index[-hold_days - 1]
    ]

    # 4. Build the target (BETA-ADJUSTED: key improvement #3)
    #    Old: ret - spy_ret
    #    New: ret - beta * spy_ret  (idiosyncratic return)
    spy_ret = wide["SPY"].pct_change(fill_method=None) if "SPY" in wide.columns else pd.Series(0, index=wide.index)
    raw_target = wide.shift(-hold_days) / wide - 1
    beta = feats["beta_252"]
    # Beta-adjusted target: strips the market component the model could
    # mechanically exploit by tilting to high-beta names in a bull market.
    target = raw_target.sub(beta.mul(
        wide["SPY"].shift(-hold_days) / wide["SPY"] - 1, axis=0
    ), axis=0) if "SPY" in wide.columns else raw_target

    # 5. Slice features at rebalance dates
    feature_names = list(feats)
    parts = []
    for d in reb_dates:
        row = pd.DataFrame(
            {name: feats[name].loc[d] for name in feature_names}
        )
        row["raw_target"] = raw_target.loc[d]   # raw SPY-adjusted for reference
        row["target"] = target.loc[d]            # beta-adjusted (the one we model)
        row["date"] = d
        parts.append(row.rename_axis("ticker").reset_index())

    panel = pd.concat(parts, ignore_index=True)

    # 6. Drop benchmarks & rows missing key data
    panel = panel[~panel["ticker"].isin(benchmarks)]
    panel = panel.dropna(subset=["target", "mom_12_1"])

    # 7. Add sector labels
    panel["sector"] = panel["ticker"].map(SECTOR_MAP).fillna("Other")

    # 8. Sector-neutral ranks (HIGH-IMPACT: key improvement #2)
    #    Ranking within sectors strips macro sector bets from momentum,
    #    isolating true idiosyncratic stock-level momentum.
    for col in feature_names:
        if col in panel.columns:
            panel[f"{col}_sn"] = panel.groupby(["date", "sector"])[col].rank(
                pct=True, method="average"
            )

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# Composite benchmark (the "dumb" version every ML model must beat)
# ---------------------------------------------------------------------------

MOMENTUM_COMPOSITE_SIGNS: dict[str, int] = {
    # Level features (pre-sector-neutral)
    "mom_12_1": +1,
    "mom_6_1": +1,
    "idio_mom": +1,
    "dist_52wk": +1,
    "rev_5d": -1,
    "rev_21d": -1,
    "vol_60d": -1,
    # Vol-adjusted features
    "mom_12_1_va": +1,
    "mom_6_1_va": +1,
    "idio_mom_va": +1,
    "rev_5d_va": -1,
    # Sector-neutral variants (same signs)
    "mom_12_1_sn": +1,
    "mom_6_1_sn": +1,
    "idio_mom_sn": +1,
    "dist_52wk_sn": +1,
    "rev_5d_sn": -1,
    "rev_21d_sn": -1,
    "vol_60d_sn": -1,
    "mom_12_1_va_sn": +1,
    "mom_6_1_va_sn": +1,
    "idio_mom_va_sn": +1,
    "rev_5d_va_sn": -1,
}


def add_composite_score(
    panel: pd.DataFrame,
    signs: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add a rank-average composite column to the panel.

    Each feature is percentile-ranked cross-sectionally per date, then the
    weighted average (with literature-derived signs) becomes ``composite``.
    """
    if signs is None:
        signs = MOMENTUM_COMPOSITE_SIGNS
    available = [c for c in signs if c in panel.columns]
    if not available:
        panel["composite"] = np.nan
        return panel
    rk = panel.groupby("date")[available].rank(pct=True)
    panel["composite"] = sum(
        signs[c] * rk[c] for c in available
    ) / len(available)
    return panel
