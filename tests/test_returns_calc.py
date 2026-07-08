"""Unit tests for src/returns_calc.py — vectorized return engine."""

import numpy as np
import pandas as pd
import pytest

from src.returns_calc import PriceMatrix, asof_values, build_event_returns


# ---------------------------------------------------------------------------
# Fixtures: 12 business days, deterministic prices
# ---------------------------------------------------------------------------

@pytest.fixture
def dates() -> pd.DatetimeIndex:
    # 2024-01-01 is a Monday: Jan 1-5, 8-12, 15-16
    return pd.bdate_range("2024-01-01", periods=12)


@pytest.fixture
def pm(dates) -> PriceMatrix:
    a = 100.0 + 2.0 * np.arange(12)      # A: 100, 102, ..., 122
    spy = 50.0 + 0.5 * np.arange(12)     # SPY: 50, 50.5, ..., 55.5
    b = 200.0 + np.zeros(12)
    b[1] = np.nan                        # B has a missing close on day 1
    wide = pd.DataFrame({"A": a, "B": b, "SPY": spy}, index=dates)
    return PriceMatrix(wide)


class TestForwardReturns:
    def test_entry_on_trading_day(self, pm):
        # Anchor Jan 1 (trading day): entry 100, 1d exit 102
        r = pm.forward_returns(["A"], ["2024-01-01"], 1)
        assert r[0] == pytest.approx(0.02)

    def test_weekend_anchor_rolls_forward(self, pm):
        # Jan 6 is a Saturday -> entry Mon Jan 8 (110), exit Jan 9 (112)
        r = pm.forward_returns(["A"], ["2024-01-06"], 1)
        assert r[0] == pytest.approx(112 / 110 - 1)

    def test_multi_day_window(self, pm):
        # entry 100, exit 5 trading days later = 110
        r = pm.forward_returns(["A"], ["2024-01-01"], 5)
        assert r[0] == pytest.approx(0.10)

    def test_unknown_ticker_is_nan(self, pm):
        assert np.isnan(pm.forward_returns(["ZZZ"], ["2024-01-01"], 1)[0])

    def test_exit_beyond_history_is_nan(self, pm):
        assert np.isnan(pm.forward_returns(["A"], ["2024-01-16"], 1)[0])

    def test_anchor_beyond_history_is_nan(self, pm):
        assert np.isnan(pm.forward_returns(["A"], ["2025-06-01"], 1)[0])

    def test_missing_price_is_nan(self, pm):
        # B's day-1 close is NaN (exit day for a Jan 1 anchor)
        assert np.isnan(pm.forward_returns(["B"], ["2024-01-01"], 1)[0])

    def test_vectorized_batch(self, pm):
        r = pm.forward_returns(["A", "SPY", "ZZZ"], ["2024-01-01"] * 3, 1)
        assert r[0] == pytest.approx(0.02)
        assert r[1] == pytest.approx(50.5 / 50 - 1)
        assert np.isnan(r[2])

    def test_from_long_matches_wide(self, pm, dates):
        long = pd.DataFrame({
            "date": list(dates) * 2,
            "ticker": ["A"] * 12 + ["SPY"] * 12,
            "close": list(100.0 + 2.0 * np.arange(12))
                     + list(50.0 + 0.5 * np.arange(12)),
        })
        pm2 = PriceMatrix.from_long(long)
        r1 = pm.forward_returns(["A"], ["2024-01-03"], 2)
        r2 = pm2.forward_returns(["A"], ["2024-01-03"], 2)
        assert r1[0] == pytest.approx(r2[0])


class TestTrailingReturns:
    def test_momentum(self, pm):
        # Entry Jan 8 (110), 5 trading days back = Jan 1 (100)
        r = pm.trailing_returns(["A"], ["2024-01-08"], 5)
        assert r[0] == pytest.approx(0.10)

    def test_lookback_before_history_is_nan(self, pm):
        assert np.isnan(pm.trailing_returns(["A"], ["2024-01-03"], 5)[0])


class TestAsofValues:
    def test_exact_and_between(self):
        vd = pd.to_datetime(["2024-01-01", "2024-01-03"])
        out = asof_values(vd, [10.0, 30.0],
                          ["2024-01-01", "2024-01-02", "2024-01-05"])
        assert out.tolist() == [10.0, 10.0, 30.0]

    def test_before_first_is_nan(self):
        vd = pd.to_datetime(["2024-01-03"])
        assert np.isnan(asof_values(vd, [30.0], ["2024-01-01"])[0])


class TestBuildEventReturns:
    def test_abnormal_is_excess_over_benchmark(self, pm):
        events = pd.DataFrame({"ticker": ["A"],
                               "earnings_date": ["2024-01-01"]})
        out = build_event_returns(pm, events, windows=(1,))
        expected = 0.02 - (50.5 / 50 - 1)
        assert out["return_1d"].iloc[0] == pytest.approx(0.02)
        assert out["abnormal_1d"].iloc[0] == pytest.approx(expected)

    def test_vix_and_covid_columns(self, pm, dates):
        vix = pd.Series([15.0] * 12, index=dates)
        events = pd.DataFrame({"ticker": ["A", "A"],
                               "earnings_date": ["2024-01-02", "2024-01-09"]})
        out = build_event_returns(pm, events, windows=(1,), vix=vix,
                                  covid_range=("2024-01-05", "2024-01-10"))
        assert out["vix_close"].tolist() == [15.0, 15.0]
        assert out["is_covid"].tolist() == [False, True]

    def test_custom_date_col(self, pm):
        events = pd.DataFrame({"ticker": ["A"], "pub_date": ["2024-01-01"]})
        out = build_event_returns(pm, events, windows=(1,),
                                  date_col="pub_date")
        assert out["return_1d"].iloc[0] == pytest.approx(0.02)
