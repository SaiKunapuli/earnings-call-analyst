"""Unit tests for position sizing (Layer 10). Pure numpy/pandas, no DB."""
import numpy as np
import pandas as pd

from src import sizing as Z


def _series(d):
    return pd.Series(d, dtype=float)


# ---- trailing_vol is leak-free (uses strictly-prior returns) ---------------
def test_trailing_vol_is_shifted():
    idx = pd.date_range("2020-01-01", periods=40)
    wide = pd.DataFrame({"A": np.linspace(100, 140, 40)}, index=idx)
    v = Z.trailing_vol(wide, window=10, min_periods=5)["A"]
    # the vol on a given row must not depend on that row's own return (shifted)
    assert bool(v.isna().iloc[0])             # first row has no prior window
    assert bool(v.notna().iloc[-1])


# ---- size_one_date: gross, caps, net neutrality ----------------------------
def test_size_one_date_respects_gross_and_name_cap():
    score = _series({t: v for t, v in zip("ABCDEFGH", [3, 2, 1, 0.5, -0.5, -1, -2, -3])})
    vol = _series({t: 0.02 for t in "ABCDEFGH"})
    w = Z.size_one_date(score, vol, gross=2.0, max_name=0.30, max_sector=9.9, net_cap=9.9)
    assert abs(w.abs().sum() - 2.0) < 0.15     # ~gross (name cap can shrink it a bit)
    assert w.abs().max() <= 0.30 + 1e-9        # per-name cap holds
    assert np.sign(w["A"]) == 1 and np.sign(w["H"]) == -1   # winners long, losers short


def test_size_one_date_net_cap():
    score = _series({t: v for t, v in zip("ABCDE", [5, 4, 3, 2, 1])})  # all positive
    vol = _series({t: 0.02 for t in "ABCDE"})
    w = Z.size_one_date(score, vol, gross=2.0, max_name=1.0, max_sector=9.9, net_cap=0.20)
    assert abs(w.sum()) <= 0.20 + 1e-6         # net exposure pulled within cap


def test_size_one_date_degenerate_returns_zeros():
    score = _series({"A": 1.0, "B": 2.0})      # <4 names
    vol = _series({"A": 0.02, "B": 0.02})
    assert (Z.size_one_date(score, vol) == 0).all()


# ---- size_deciles: vol-balance within legs, name cap, leg gross ------------
def test_size_deciles_selects_tails_and_balances():
    n = 100
    score = _series({f"T{i}": float(i) for i in range(n)})     # T99 best, T0 worst
    vol = _series({f"T{i}": 0.02 for i in range(n)})
    w = Z.size_deciles(score, vol, top=0.10, gross=2.0, max_name=0.5)
    long = w[w > 0]
    short = w[w < 0]
    assert 9 <= len(long) <= 11 and 9 <= len(short) <= 11      # ~top/bottom decile
    assert set(long.index) >= {"T99", "T95"}                   # highest scores long
    assert abs(long.sum() - 1.0) < 1e-6                        # each leg == gross/2
    assert abs(short.sum() + 1.0) < 1e-6


def test_size_deciles_lower_vol_gets_more_weight():
    score = _series({f"T{i}": float(i) for i in range(20)})
    vol = _series({f"T{i}": 0.02 for i in range(20)})
    vol["T19"] = 0.04                          # highest scorer is twice as volatile
    w = Z.size_deciles(score, vol, top=0.10, gross=2.0, max_name=0.9)
    # among the 2 long names, the LOWER-vol one carries more weight
    longs = w[w > 0].sort_values()
    assert longs.index[0] == "T19"             # high-vol name gets the smaller weight


def test_size_deciles_name_cap_binds():
    score = _series({f"T{i}": float(i) for i in range(40)})
    vol = _series({f"T{i}": 0.02 for i in range(40)})
    vol["T39"] = 0.001                         # ultra-low vol would blow up the weight
    w = Z.size_deciles(score, vol, top=0.10, gross=2.0, max_name=0.15)
    assert w.abs().max() <= 0.15 + 1e-9


def test_size_deciles_sector_neutral_balances_legs():
    # Two sectors: TECH scores uniformly high, ENERGY uniformly low.
    # Raw ranking -> long leg all TECH, short leg all ENERGY (a sector bet).
    # Sector-neutral -> each sector contributes to BOTH legs.
    tickers = [f"TE{i}" for i in range(20)] + [f"EN{i}" for i in range(20)]
    score = _series({t: (10 + i if t.startswith("TE") else 1 + i)
                     for i, t in enumerate(tickers)})
    vol = _series({t: 0.02 for t in tickers})
    sector = pd.Series({t: ("Tech" if t.startswith("TE") else "Energy") for t in tickers})

    raw = Z.size_deciles(score, vol, sector=sector, sector_neutral=False,
                         top=0.20, gross=2.0, max_name=0.9)
    neu = Z.size_deciles(score, vol, sector=sector, sector_neutral=True,
                         top=0.20, gross=2.0, max_name=0.9)

    def sector_net(w):
        held = w[w != 0]
        return held.groupby(sector.reindex(held.index)).sum()

    # raw: one sector dominates a leg -> large |net| per sector
    assert sector_net(raw).abs().max() > 0.5
    # neutral: both legs drawn from both sectors -> net per sector near zero
    assert sector_net(neu).abs().max() < 0.2


def test_book_stats_effective_positions():
    w = _series({"A": 0.5, "B": -0.5, "C": 0.0})
    st = Z.book_stats(w)
    assert st["n"] == 2
    assert abs(st["gross"] - 1.0) < 1e-9
    assert abs(st["eff_n"] - 2.0) < 1e-9       # two equal names -> eff_n 2
