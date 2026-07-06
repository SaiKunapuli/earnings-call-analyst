"""Quick verdict on the trained model — run AFTER 04_modeling finishes.

Reads the walk-forward eval-region predictions the pipeline wrote to
`model_predictions` and prints the numbers that actually decide whether
there is signal:

  - Information Coefficient (monthly Spearman rank corr) + t-stat
  - Decile spread (top-minus-bottom predicted-decile actual return)
  - Sign accuracy and R^2
  - Classification AUC
  - Top features by mean |SHAP|

The full charts live in 04_modeling.ipynb / 05_backtest.ipynb — this is
the fast "did it work?" summary you can run from the terminal:

    .venv/Scripts/python.exe scripts/analyze_results.py
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Windows consoles default to cp1252, which can't encode arrows etc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> int:
    conn = sqlite3.connect(str(DB_PATH))

    # Coverage (from sentiment_features) — confirms the data actually expanded
    try:
        sf = pd.read_sql(
            "SELECT ticker, matched_earnings_date FROM sentiment_features",
            conn, parse_dates=["matched_earnings_date"])
        print(f"sentiment_features: {len(sf):,} rows, {sf['ticker'].nunique()} tickers, "
              f"{sf['matched_earnings_date'].min().date()} - {sf['matched_earnings_date'].max().date()}")
    except Exception as e:
        print(f"sentiment_features: not available ({e})")

    preds = pd.read_sql("SELECT * FROM model_predictions", conn,
                        parse_dates=["matched_earnings_date"])
    conn.close()

    # Prefer the raw-space (recombined) prediction for ranking when the
    # model trained on a transformed target — it's the tradeable signal.
    _pcols = sorted([c for c in preds.columns if c.startswith("predicted_")],
                    key=lambda c: c.endswith("_va"))
    pcol = _pcols[0] if _pcols else None
    if not pcol:
        print("No predicted_ column — has 04_modeling run yet?")
        return 1
    target = pcol.replace("predicted_", "")
    acol = f"actual_{target}"
    # The model may predict a transformed target (e.g. *_va = vol-adjusted
    # sigma units). Regression stats use the model's own target; IC and the
    # decile spread use RAW returns — what a book earns, and comparable
    # across target definitions.
    raw_target = re.sub(r"_va$", "", target)
    rcol = f"actual_{raw_target}" if f"actual_{raw_target}" in preds.columns else acol
    if acol not in preds.columns:
        print(f"No {acol} column — has 04_modeling run yet?")
        return 1

    print(f"\nmodel_predictions: {len(preds):,} eval rows | target = {target} | "
          f"{preds['matched_earnings_date'].min().date()} - {preds['matched_earnings_date'].max().date()}")
    print("(If target/row-count look like the OLD run, 04_modeling hasn't finished — re-run this after it does.)")
    if rcol != acol:
        print(f"(Regression vs {acol}; IC/decile spread vs raw {rcol}.)")

    _cols = list(dict.fromkeys(["matched_earnings_date", acol, pcol, rcol]))
    d = preds[_cols].dropna()
    a, p, r = d[acol].to_numpy(), d[pcol].to_numpy(), d[rcol].to_numpy()

    # ---- Regression quality ----
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    base = float(np.sqrt(np.mean((a - a.mean()) ** 2)))
    r2 = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
    sign = float(np.mean(np.sign(p) == np.sign(a)))

    # ---- Information Coefficient (monthly Spearman, vs RAW returns) ----
    d = d.assign(m=d["matched_earnings_date"].dt.to_period("M"))
    ics = (d.groupby("m")[[pcol, rcol]]
           .apply(lambda g: spearmanr(g[pcol], g[rcol])[0] if len(g) >= 5 else np.nan)
           .dropna())
    mean_ic = ics.mean() if len(ics) else np.nan
    t_ic = (mean_ic / ics.std() * np.sqrt(len(ics))) if len(ics) > 1 and ics.std() > 0 else np.nan

    # ---- Decile spread (vs RAW returns) ----
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    dmean = pd.DataFrame({"dec": dec, "a": r}).groupby("dec")["a"].mean()
    spread_bps = (dmean.iloc[-1] - dmean.iloc[0]) * 1e4

    print("\n=== Regression (eval region) ===")
    print(f"  RMSE            {_fmt(rmse,6)}   (naive mean {_fmt(base,6)})")
    print(f"  R^2             {_fmt(r2)}")
    print(f"  Sign accuracy   {_fmt(sign*100,1)}%")
    print("\n=== Cross-sectional signal (what a long/short book monetizes) ===")
    print(f"  Mean monthly IC {_fmt(mean_ic)}   t-stat {_fmt(t_ic,2)}   "
          f"({(ics > 0).mean()*100:.0f}% of {len(ics)} months positive)")
    print(f"  Decile spread   {spread_bps:+.0f} bps  (top predicted decile minus bottom)")

    # ---- Classification ----
    if {"clf_actual", "clf_predicted_prob"}.issubset(preds.columns):
        from sklearn.metrics import accuracy_score, roc_auc_score
        cc = preds[["clf_actual", "clf_predicted_prob"]].dropna()
        if cc["clf_actual"].nunique() == 2:
            auc = roc_auc_score(cc["clf_actual"], cc["clf_predicted_prob"])
            acc = accuracy_score(cc["clf_actual"], (cc["clf_predicted_prob"] >= 0.5).astype(int))
            n_dz = len(cc)
            print(f"\n=== Classification (up/down, {n_dz:,} clear-move events) ===")
            print(f"  AUC {_fmt(auc)}   Accuracy {_fmt(acc*100,1)}%")

    # ---- Top features by SHAP ----
    shap_cols = [c for c in preds.columns if c.startswith("shap_")]
    if shap_cols:
        imp = preds[shap_cols].abs().mean().sort_values(ascending=False).head(8)
        print("\n=== Top features (mean |SHAP|) ===")
        for c, v in imp.items():
            print(f"  {c[5:]:32s} {v:.6f}")

    # ---- One-line verdict ----
    print("\n" + "=" * 60)
    if not np.isnan(t_ic) and abs(t_ic) >= 2 and mean_ic > 0:
        print(f"VERDICT: usable cross-sectional signal (IC {mean_ic:+.3f}, t {t_ic:+.2f}).")
        print("Look at 05_backtest.ipynb for whether it survives costs.")
    elif not np.isnan(mean_ic) and mean_ic > 0:
        print(f"VERDICT: weak/positive but not significant (IC {mean_ic:+.3f}, t {t_ic:+.2f}).")
        print("Direction is right; needs more signal or a cleaner target.")
    else:
        print(f"VERDICT: no cross-sectional edge on {target} (IC {mean_ic:+.3f}).")
        print("Expected if the language just doesn't move this horizon — compare vs the abnormal_1d baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
