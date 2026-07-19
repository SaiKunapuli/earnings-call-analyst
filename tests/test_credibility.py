"""Unit tests for the management-credibility signal (pure logic, no DB)."""
import numpy as np
import pandas as pd

from src import credibility as C


# ---- optimism_score -------------------------------------------------------
def test_optimism_score_combines_forward_claims():
    bull = {"guidance_direction": 1, "demand_outlook": 2, "margin_outlook": 2}
    bear = {"guidance_direction": -1, "demand_outlook": -2, "margin_outlook": -2}
    assert C.optimism_score(bull) > 0.9
    assert C.optimism_score(bear) < -0.9
    assert C.optimism_score({"guidance_direction": 0, "demand_outlook": 0,
                             "margin_outlook": 0}) == 0.0


def test_optimism_score_handles_null_guidance():
    # guidance not discussed -> averaged over the remaining fields only
    s = {"guidance_direction": None, "demand_outlook": 2, "margin_outlook": 2}
    assert C.optimism_score(s) == 1.0
    assert C.optimism_score({}) == 0.0


# ---- grade_agreement ------------------------------------------------------
def test_grade_agreement_signs():
    assert C.grade_agreement(0.8, 5.0) == 1.0      # bullish + beat -> right
    assert C.grade_agreement(0.8, -5.0) == -1.0    # bullish + miss -> wrong
    assert C.grade_agreement(-0.8, -5.0) == 1.0    # bearish + miss -> right
    assert C.grade_agreement(0.8, None) == 0.0     # no outcome
    assert C.grade_agreement(0.0, 5.0) == 0.0      # no claim
    assert C.grade_agreement(0.8, float("nan")) == 0.0


# ---- build_credibility: accumulation + leak-free --------------------------
def _calls(ticker, opt_surprise_pairs, start="2020-01-01"):
    dates = pd.date_range(start, periods=len(opt_surprise_pairs), freq="90D")
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "optimism": [o for o, _ in opt_surprise_pairs],
        "next_surprise": [s for _, s in opt_surprise_pairs],
    })


def test_credibility_accumulates_for_reliable_team():
    # a team that is always right: optimism sign always matches next surprise
    calls = _calls("GOOD", [(0.8, 5), (0.8, 4), (-0.8, -3), (0.8, 6), (0.8, 2)])
    out = C.build_credibility(calls)
    assert (out["agreement"] == 1.0).all()
    # credibility is prior-only: first call NaN, then rises to +1
    assert np.isnan(out["credibility"].iloc[0])
    assert out["credibility"].iloc[-1] == 1.0


def test_credibility_penalizes_over_promiser():
    calls = _calls("BAD", [(0.8, -5), (0.8, -4), (0.9, -3), (0.8, -6)])
    out = C.build_credibility(calls)
    assert (out["agreement"] == -1.0).all()
    assert out["credibility"].iloc[-1] == -1.0
    # a chronic over-promiser sounding bullish -> NEGATIVE weighted signal
    assert out["cred_weighted_optimism"].iloc[-1] < 0


def test_credibility_is_leak_free():
    # the current call's own outcome must not affect its credibility
    calls = _calls("X", [(0.8, 5), (0.8, 5), (-0.8, 5)])  # 3rd call is "wrong"
    out = C.build_credibility(calls)
    # credibility on the 3rd call reflects only calls 1-2 (both right) -> +1,
    # regardless of the 3rd call's own (disagreeing) outcome
    assert out["credibility"].iloc[2] == 1.0


def test_credibility_first_call_neutral_weighted():
    calls = _calls("Y", [(0.8, 5)])
    out = C.build_credibility(calls)
    assert np.isnan(out["credibility"].iloc[0])           # no track record
    assert out["cred_weighted_optimism"].iloc[0] == 0.0   # -> neutral signal
