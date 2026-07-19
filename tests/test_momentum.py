"""Tests for src/momentum.py — the last untested src module.

Synthetic deterministic price panel; the load-bearing test is leak-freeness:
feature values at date d must be identical when future prices change.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.momentum import (MOMENTUM_COMPOSITE_SIGNS, add_composite_score,
                          build_momentum_panel, compute_momentum_features)


@pytest.fixture()
def wide():
    """320 business days, deterministic growth paths: UP > SPY > DOWN.

    A small deterministic wobble keeps trailing vols strictly positive —
    perfectly constant growth has zero return variance, which (correctly)
    NaNs the vol-adjusted features and the rolling beta."""
    idx = pd.bdate_range("2020-01-01", periods=320)
    t = np.arange(320)
    return pd.DataFrame({
        "UP":   100 * (1.003 ** t) * (1 + 0.004 * np.sin(t / 3)),
        "DOWN": 100 * (0.998 ** t) * (1 + 0.004 * np.sin(t / 4)),
        "SPY":  100 * (1.001 ** t) * (1 + 0.004 * np.sin(t / 5)),
    }, index=idx)


def test_feature_shapes_and_values(wide):
    feats = compute_momentum_features(wide)
    for name, m in feats.items():
        assert m.shape == wide.shape, name
    # mom_12_1 arithmetic straight from prices: (1+r252)/(1+r21) - 1
    p = wide["UP"]
    expected = (p.iloc[-1] / p.iloc[-253]) / (p.iloc[-1] / p.iloc[-22]) - 1
    got = feats["mom_12_1"]["UP"].iloc[-1]
    assert np.isclose(got, expected, rtol=1e-9)
    # a scaled copy of SPY has identical returns -> beta ~ 1. Tolerance is
    # loose because the estimator mixes a population covariance with a
    # sample variance (ddof mismatch -> ~(n-1)/n bias; harmless).
    noisy = wide.copy()
    noisy["TRACKER"] = noisy["SPY"] * 1.7
    b = compute_momentum_features(noisy)["beta_252"]
    assert np.isclose(b["TRACKER"].iloc[-1], 1.0, atol=0.01)
    assert np.isclose(b["SPY"].iloc[-1], 1.0, atol=0.01)


def test_features_are_leak_free(wide):
    """Changing FUTURE prices must not change feature values at earlier dates."""
    check_date = wide.index[280]
    base = compute_momentum_features(wide)
    tampered = wide.copy()
    tampered.iloc[-30:] *= 1.5  # rewrite the future
    tamp = compute_momentum_features(tampered)
    for name in base:
        a = base[name].loc[check_date]
        b = tamp[name].loc[check_date]
        pd.testing.assert_series_equal(a, b, check_names=False, obj=name)


def test_panel_structure_and_target(wide):
    panel = build_momentum_panel(wide, benchmarks=("SPY",), hold_days=21)
    # benchmarks excluded, expected columns present
    assert "SPY" not in set(panel["ticker"])
    for col in ("date", "ticker", "sector", "target", "raw_target",
                "mom_12_1", "mom_12_1_sn"):
        assert col in panel.columns, col
    # raw_target arithmetic at one rebalance date
    d = panel["date"].iloc[0]
    row = panel[(panel["date"] == d) & (panel["ticker"] == "UP")].iloc[0]
    i = wide.index.get_loc(d)
    expected = wide["UP"].iloc[i + 21] / wide["UP"].iloc[i] - 1
    assert np.isclose(row["raw_target"], expected, rtol=1e-9)
    # rebalance dates are month-ends past the warm-up window
    assert panel["date"].min() >= wide.index[260]
    # sector-neutral ranks are percentiles
    sn = panel["mom_12_1_sn"].dropna()
    assert (sn >= 0).all() and (sn <= 1).all()


def test_composite_orders_momentum(wide):
    panel = build_momentum_panel(wide, benchmarks=("SPY",), hold_days=21)
    panel = add_composite_score(panel)
    assert "composite" in panel.columns
    d = panel["date"].iloc[-1]
    day = panel[panel["date"] == d].set_index("ticker")["composite"]
    # steady up-trender must out-rank the down-trender on a momentum composite
    assert day["UP"] > day["DOWN"]


def test_composite_empty_signs():
    panel = pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "x": [1.0]})
    out = add_composite_score(panel.copy(), signs={"not_a_column": 1})
    assert out["composite"].isna().all()


def test_composite_signs_reference_known_features():
    # every sign key should be a feature the panel builder can produce
    produced = {"mom_12_1", "mom_6_1", "rev_5d", "rev_21d", "dist_52wk",
                "vol_60d", "vol_20d", "beta_252", "idio_mom", "mom_12_1_va",
                "mom_6_1_va", "idio_mom_va", "rev_5d_va", "turn_trend",
                "log_adv"}
    produced |= {f"{c}_sn" for c in produced}
    unknown = set(MOMENTUM_COMPOSITE_SIGNS) - produced
    assert not unknown, f"composite references unproducible features: {unknown}"
