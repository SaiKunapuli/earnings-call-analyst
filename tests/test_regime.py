"""Unit tests for the regime gate's pure signal functions (no DB / no network)."""
import numpy as np
import pandas as pd

from src import regime as R


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


# ---- VIX signal: low neutral, 25-35 caution, >35 stress --------------------
def test_signal_vix_thresholds():
    vix = pd.Series([15, 25, 26, 35, 40], index=_idx(5))
    s = R._signal_vix(vix)
    assert list(s) == [0.0, 0.0, -0.5, -0.5, -1.0]   # 25 not >25; 35 not >35


# ---- SPY vs 200dma: above +1, near/below 0, >10% below -1 ------------------
def test_signal_spy_dma():
    # 210 days: first 199 flat at 100 (dma warms up), then a level that sits
    # a known % from the trailing 200d mean.
    base = [100.0] * 200
    spy = pd.Series(base + [130, 100, 85], index=_idx(203))
    s = R._signal_spy_dma(spy)
    # last three: well above dma -> +1 ; at/just below -> 0 ; >10% below -> -1
    assert s.iloc[-3] == 1.0
    assert s.iloc[-2] == 0.0            # equal to a ~100 dma => not above
    assert s.iloc[-1] == -1.0           # 85 vs ~100 dma is >10% below


# ---- Credit: HYG/LQD relative return, negative = stress --------------------
def test_signal_credit_stress_and_neutral():
    n = 40
    lqd = pd.Series(100.0, index=_idx(n))
    hyg = pd.Series(100.0, index=_idx(n))
    hyg.iloc[-1] = 92.0                 # HYG drops 8% vs LQD over the window
    credit = pd.DataFrame({"HYG": hyg, "LQD": lqd})
    s = R._signal_credit(credit)
    assert s.iloc[-1] == -1.0           # <-5% -> risk-off
    assert s.iloc[10] == 0.0            # flat earlier -> neutral


def test_signal_credit_empty_is_neutral():
    assert R._signal_credit(pd.DataFrame()).empty


# ---- Breadth: % above 200dma; broad +1, mixed 0, narrow -1 -----------------
def test_signal_breadth():
    n = 210
    # 10 tickers all rising -> nearly all above their 200dma -> broad (+1)
    rising = {f"T{i}": pd.Series(np.linspace(50, 150, n), index=_idx(n)) for i in range(10)}
    wide_up = pd.DataFrame(rising)
    assert R._signal_breadth(wide_up).iloc[-1] == 1.0
    # all falling -> almost none above -> narrow (-1)
    falling = {f"T{i}": pd.Series(np.linspace(150, 50, n), index=_idx(n)) for i in range(10)}
    assert R._signal_breadth(pd.DataFrame(falling)).iloc[-1] == -1.0


# ---- thresholds are ordered the sane way (guards against edits) ------------
def test_threshold_ordering():
    assert R.VIX_CAUTION < R.VIX_RISK_OFF
    assert R.SPY_DMA_RISK_OFF < R.SPY_DMA_CAUTION
    assert R.CREDIT_RISK_OFF < R.CREDIT_CAUTION < 0
    assert R.BREADTH_RISK_OFF < R.BREADTH_CAUTION


def test_regime_summary_empty():
    out = R.regime_summary(pd.DataFrame({"multiplier": []}))
    assert out["risk_on_pct"] == 0
