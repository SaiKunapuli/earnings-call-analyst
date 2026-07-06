"""Vectorized forward/trailing return computation over a price matrix.

Single implementation shared by:
- 01_pull_prices.ipynb  — building the ``returns`` table for earnings events
- 03_sentiment.ipynb    — pub_date-anchored fallback returns for transcripts
  whose ticker has prices but no yfinance earnings date within 30 days
- 04_modeling.ipynb     — pre-earnings momentum features

All lookups are numpy ``searchsorted`` + fancy indexing: computing 6 return
windows for 30k events takes well under a second, versus minutes with
per-row ``.loc``/``get_loc`` scans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PriceMatrix:
    """Wide (date x ticker) close-price matrix with vectorized return lookups.

    Entry convention: the *entry* price for an anchor date is the close of
    the first trading day ON OR AFTER that date; an ``n_days`` forward
    return exits ``n_days`` trading days later.  Trailing (momentum)
    returns look ``n_days`` trading days back from the entry day.
    """

    def __init__(self, prices_wide: pd.DataFrame):
        prices_wide = prices_wide.sort_index()
        self.dates = prices_wide.index.values.astype("datetime64[ns]")
        self.px = prices_wide.to_numpy(dtype=float)
        self.columns = list(prices_wide.columns)
        self._col = {t: i for i, t in enumerate(self.columns)}

    @classmethod
    def from_long(cls, prices_long: pd.DataFrame, date_col: str = "date",
                  ticker_col: str = "ticker", price_col: str = "close") -> "PriceMatrix":
        wide = prices_long.pivot(index=date_col, columns=ticker_col,
                                 values=price_col)
        return cls(wide)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _positions(self, tickers, dates) -> tuple[np.ndarray, np.ndarray]:
        """(column index, entry-day row index) per event; col=-1 if unknown."""
        col = np.array([self._col.get(t, -1) for t in tickers], dtype=np.int64)
        d = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[ns]")
        sp = self.dates.searchsorted(d)  # first trading day on/after date
        return col, sp

    def _price_at(self, rows: np.ndarray, col: np.ndarray,
                  valid: np.ndarray) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        idx = np.where(valid)[0]
        out[idx] = self.px[rows[idx], col[idx]]
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_returns(self, tickers, dates, n_days: int) -> np.ndarray:
        """Vectorized ``n_days`` forward return per (ticker, anchor date).

        NaN where the ticker is unknown, the entry/exit day falls outside
        the price history, or either price is missing.
        """
        col, sp = self._positions(tickers, dates)
        ep = sp + n_days
        valid = (col >= 0) & (sp < len(self.dates)) & (ep < len(self.dates))
        ps = self._price_at(sp, col, valid)
        pe = self._price_at(ep, col, valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            return pe / ps - 1.0

    def trailing_returns(self, tickers, dates, n_days: int) -> np.ndarray:
        """Vectorized ``n_days`` trailing (momentum) return ending at entry day."""
        col, sp = self._positions(tickers, dates)
        lb = sp - n_days
        valid = (col >= 0) & (sp < len(self.dates)) & (lb >= 0)
        ps = self._price_at(np.clip(lb, 0, None), col, valid)
        pe = self._price_at(np.clip(sp, 0, len(self.dates) - 1), col, valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            return pe / ps - 1.0


def asof_values(value_dates, values, query_dates) -> np.ndarray:
    """Last known value on/before each query date (e.g. VIX close).

    NaN where the query date precedes the first observation.
    """
    vd = pd.to_datetime(pd.Series(value_dates)).values.astype("datetime64[ns]")
    vv = np.asarray(values, dtype=float)
    order = np.argsort(vd)
    vd, vv = vd[order], vv[order]
    qd = pd.to_datetime(pd.Series(query_dates)).values.astype("datetime64[ns]")
    pos = vd.searchsorted(qd, side="right") - 1
    out = np.full(len(qd), np.nan)
    ok = pos >= 0
    out[ok] = vv[pos[ok]]
    return out


RETURN_WINDOWS = (1, 30, 90)


def build_event_returns(pm: PriceMatrix, events: pd.DataFrame,
                        benchmark: str = "SPY",
                        windows: tuple[int, ...] = RETURN_WINDOWS,
                        date_col: str = "earnings_date",
                        vix: pd.Series | None = None,
                        covid_range: tuple[str, str] | None = None) -> pd.DataFrame:
    """Raw + benchmark-adjusted forward returns for a frame of events.

    ``events`` needs ``ticker`` and *date_col* columns.  Output columns
    mirror the ``returns`` table: ``return_{n}d``, ``abnormal_{n}d`` per
    window, plus ``vix_close`` / ``is_covid`` when the inputs are given.
    """
    out = events[["ticker", date_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    tickers = out["ticker"].to_numpy()
    dates = out[date_col].to_numpy()
    bench = np.full(len(out), benchmark, dtype=object)

    for n in windows:
        r = pm.forward_returns(tickers, dates, n)
        b = pm.forward_returns(bench, dates, n)
        out[f"return_{n}d"] = r
        out[f"abnormal_{n}d"] = r - b

    if vix is not None:
        out["vix_close"] = asof_values(vix.index, vix.to_numpy(), dates)
    if covid_range is not None:
        start, end = pd.Timestamp(covid_range[0]), pd.Timestamp(covid_range[1])
        out["is_covid"] = out[date_col].between(start, end)
    return out
