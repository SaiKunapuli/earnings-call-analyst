"""Unit tests for src/features.py — leak-free panel feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_past_target_stats,
    add_qoq_deltas,
    combine_finbert_sections,
    compute_ticker_z_scores_expanding,
)


# ---------------------------------------------------------------------------
# Expanding z-scores
# ---------------------------------------------------------------------------

def _panel(values, ticker="MSFT", start="2020-01-01"):
    """One-ticker panel with a quarterly date column and a z-eligible column."""
    n = len(values)
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "matched_earnings_date": pd.date_range(start, periods=n, freq="QS"),
        "full_vader_mean": values,
    })


class TestExpandingZScores:
    def test_uses_only_prior_observations(self):
        df = _panel([1.0, 2.0, 3.0, 4.0])
        out = compute_ticker_z_scores_expanding(df, prefix="full", min_obs=3)
        z = out["full_vader_mean_z"]
        # First 3 events: fewer than min_obs prior observations
        assert z.iloc[:3].isna().all()
        # Event 4: priors are [1,2,3] -> mean 2, std 1 -> z = (4-2)/1 = 2
        assert z.iloc[3] == pytest.approx(2.0)

    def test_no_lookahead(self):
        base = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
        bumped = _panel([1.0, 2.0, 3.0, 4.0, 500.0])  # only the FUTURE differs
        z_base = compute_ticker_z_scores_expanding(base, min_obs=3)
        z_bump = compute_ticker_z_scores_expanding(bumped, min_obs=3)
        # z at event 4 must be identical — event 5 hasn't happened yet
        assert (z_base["full_vader_mean_z"].iloc[3]
                == pytest.approx(z_bump["full_vader_mean_z"].iloc[3]))

    def test_constant_history_is_nan(self):
        df = _panel([2.0, 2.0, 2.0, 2.0, 2.0])
        out = compute_ticker_z_scores_expanding(df, min_obs=3)
        assert out["full_vader_mean_z"].isna().all()  # sigma == 0

    def test_row_order_preserved(self):
        df = _panel([1.0, 2.0, 3.0, 4.0]).iloc[[2, 0, 3, 1]]  # shuffled
        out = compute_ticker_z_scores_expanding(df, min_obs=3)
        assert list(out.index) == [2, 0, 3, 1]
        # The chronologically-4th event still gets z=2 wherever it sits
        assert out.loc[3, "full_vader_mean_z"] == pytest.approx(2.0)

    def test_tickers_are_independent(self):
        a = _panel([1.0, 2.0, 3.0, 4.0], ticker="MSFT")
        b = _panel([100.0, 100.0, 100.0, 100.0], ticker="AAPL")
        out = compute_ticker_z_scores_expanding(
            pd.concat([a, b], ignore_index=True), min_obs=3)
        msft = out[out["ticker"] == "MSFT"]["full_vader_mean_z"]
        assert msft.iloc[3] == pytest.approx(2.0)  # unpolluted by AAPL

    def test_missing_columns_are_skipped(self):
        df = pd.DataFrame({
            "ticker": ["MSFT"],
            "matched_earnings_date": pd.to_datetime(["2020-01-01"]),
        })
        out = compute_ticker_z_scores_expanding(df)
        assert not any(c.endswith("_z") for c in out.columns)


# ---------------------------------------------------------------------------
# Quarter-over-quarter deltas
# ---------------------------------------------------------------------------

class TestQoqDeltas:
    def test_delta_vs_previous_call(self):
        df = _panel([0.10, 0.30, 0.20])
        out = add_qoq_deltas(df, ["full_vader_mean"])
        d = out["full_vader_mean_qoq"]
        assert np.isnan(d.iloc[0])
        assert d.iloc[1] == pytest.approx(0.20)
        assert d.iloc[2] == pytest.approx(-0.10)

    def test_per_ticker_isolation(self):
        a = _panel([0.1, 0.2], ticker="MSFT")
        b = _panel([0.9, 0.5], ticker="AAPL")
        out = add_qoq_deltas(pd.concat([a, b], ignore_index=True),
                             ["full_vader_mean"])
        aapl = out[out["ticker"] == "AAPL"]["full_vader_mean_qoq"]
        assert np.isnan(aapl.iloc[0])           # AAPL's first call, not MSFT's next
        assert aapl.iloc[1] == pytest.approx(-0.4)

    def test_missing_columns_skipped(self):
        df = _panel([0.1, 0.2])
        out = add_qoq_deltas(df, ["not_a_column"])
        assert "not_a_column_qoq" not in out.columns


# ---------------------------------------------------------------------------
# Past target stats (PEAD prior)
# ---------------------------------------------------------------------------

class TestPastTargetStats:
    def test_expanding_mean_excludes_current(self):
        df = _panel([0, 0, 0, 0]).assign(abnormal_1d=[0.01, 0.03, -0.02, 0.10])
        out = add_past_target_stats(df, target="abnormal_1d", min_obs=2)
        m = out["past_abnormal_1d_mean"]
        assert m.iloc[:2].isna().all()                     # < min_obs priors
        assert m.iloc[2] == pytest.approx(0.02)            # mean(0.01, 0.03)
        assert m.iloc[3] == pytest.approx(np.mean([0.01, 0.03, -0.02]))

    def test_std_column_present(self):
        df = _panel([0, 0, 0]).assign(abnormal_1d=[0.01, 0.03, 0.05])
        out = add_past_target_stats(df, target="abnormal_1d", min_obs=2)
        assert out["past_abnormal_1d_std"].iloc[2] == pytest.approx(
            np.std([0.01, 0.03], ddof=1))


# ---------------------------------------------------------------------------
# FinBERT section combining
# ---------------------------------------------------------------------------

def _scores(pos, neg, neu, chunks=4):
    return {"finbert_positive": pos, "finbert_negative": neg,
            "finbert_neutral": neu, "finbert_net": pos - neg,
            "finbert_label": "positive", "finbert_chunks": chunks}


class TestCombineFinbertSections:
    def test_word_weighted_average(self):
        prep = _scores(0.8, 0.1, 0.1, chunks=6)
        qa = _scores(0.2, 0.5, 0.3, chunks=2)
        out = combine_finbert_sections(prep, qa, prep_words=3000, qa_words=1000)
        assert out["finbert_positive"] == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)
        assert out["finbert_negative"] == pytest.approx(0.75 * 0.1 + 0.25 * 0.5)
        assert out["finbert_net"] == pytest.approx(
            out["finbert_positive"] - out["finbert_negative"])
        assert out["finbert_chunks"] == 8

    def test_single_section_passthrough(self):
        prep = _scores(0.6, 0.2, 0.2)
        out = combine_finbert_sections(prep, None, prep_words=2000, qa_words=0)
        assert out["finbert_positive"] == pytest.approx(0.6)
        assert out["finbert_label"] == "positive"

    def test_nan_section_ignored(self):
        prep = _scores(0.6, 0.2, 0.2)
        qa = {"finbert_positive": np.nan, "finbert_negative": np.nan,
              "finbert_neutral": np.nan, "finbert_net": np.nan,
              "finbert_label": "neutral", "finbert_chunks": 0}
        out = combine_finbert_sections(prep, qa, prep_words=2000, qa_words=500)
        assert out["finbert_positive"] == pytest.approx(0.6)

    def test_no_usable_sections(self):
        out = combine_finbert_sections(None, None, 0, 0)
        assert np.isnan(out["finbert_positive"])
        assert out["finbert_label"] == "neutral"
        assert out["finbert_chunks"] == 0

    def test_label_is_argmax(self):
        prep = _scores(0.1, 0.7, 0.2)
        out = combine_finbert_sections(prep, None, 1000, 0)
        assert out["finbert_label"] == "negative"
