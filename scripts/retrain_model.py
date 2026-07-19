"""Pre-registered retrain of the abnormal_30d model.

Why this script exists instead of re-running notebook 04: 04 trains on
everything in sentiment_features TODAY and overwrites the canonical model +
model_predictions — running it now would train through 2026 and destroy the
frozen-model OOS experiment. This script reproduces 04's final-model recipe
exactly (params from the frozen booster, raw labels, 300 rounds, ticker/sector
categorical) but with an explicit training cutoff.

Modes
-----
default        RESEARCH retrain: train on rows <= 2025-05-15 — the frozen
               model's cutoff, but now INCLUDING the ~2.5k backfilled
               defeatbeta rows it never saw. Saves to the canonical
               models/lgbm_abnormal_30d.pkl (which run_oos_test.py loads).
               Follow with ONE run of scripts/run_oos_test.py and treat that
               as the final word from the 2025-26 window: every re-use of an
               OOS window spends it — do not iterate against it.
--with-llm     also add the LLM Q&A features (attach_llm_features from
               scripts/llm_ab_test.py — identical construction to the A/B).
               Credibility features are EXCLUDED: in the full-corpus A/B,
               arm C (+cred) underperformed arm B (+llm alone).
--production   DEPLOY retrain: no cutoff, train on all rows. Saves to
               models/lgbm_abnormal_30d_production_<today>.pkl and never
               touches the canonical path. This model has NO valid OOS
               number until genuinely new quarters arrive and are tested.
--dry-run      print the plan (rows, dates, features) and exit.

HISTORY (2026-07-09): the original "frozen" pkl turned out to be CONTAMINATED —
notebook 04 had silently re-run on the expanded panel (training through
2026-05) and overwritten it, so the celebrated OOS IC +0.33 was the model
re-predicting its own training data (uniform pooled IC ~0.31 pre- and
post-cutoff; model_predictions extended to 2026-05). That file is kept ONLY
as a recipe source (its Optuna params + 23-feature list were derived from the
pre-2023-08 tune split, which is safe to reuse) under the truthful name
lgbm_abnormal_30d_CONTAMINATED_trained_thru_202605.pkl — NEVER predict with
it. The honest frozen-cutoff artifact is lgbm_abnormal_30d_frozen20250515_llm_v2.pkl
(built by this script; T-anchored OOS: monthly IC +0.0695 t=+1.73, spread
+424bps; T+1 TRADEABLE: IC -0.065 — see PROJECT_JOURNAL 2026-07-09).

    .venv/Scripts/python.exe scripts/retrain_model.py [--with-llm] [--production] [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.run_oos_test import CUTOFF, MODEL_PATH, TARGET, build_features
from scripts.llm_ab_test import attach_llm_features

DB = ROOT / "data" / "market.db"
# Recipe source ONLY (params + base feature list, both from the pre-2023-08
# tune split). Its WEIGHTS are contaminated (trained through 2026-05) — never
# predict with it.
RECIPE_SOURCE = ROOT / "models" / "lgbm_abnormal_30d_CONTAMINATED_trained_thru_202605.pkl"
PARAM_KEEP = {"objective", "metric", "boosting_type", "num_leaves", "learning_rate",
              "feature_fraction", "bagging_fraction", "bagging_freq",
              "min_data_in_leaf", "min_sum_hessian_in_leaf", "lambda_l1", "lambda_l2"}
CRED_FEATS = ("credibility", "cred_weighted_optimism")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-llm", action="store_true",
                    help="include the LLM Q&A features (excl. credibility)")
    ap.add_argument("--production", action="store_true",
                    help="train on ALL data; save to a dated production file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without training")
    args = ap.parse_args()

    if not RECIPE_SOURCE.exists():
        print(f"ERROR: recipe source missing: {RECIPE_SOURCE.name}")
        return 1
    frozen = joblib.load(RECIPE_SOURCE)
    base_feats = frozen.feature_name()
    raw = dict(frozen.params) if frozen.params else {}
    params = {k: v for k, v in raw.items() if k in PARAM_KEEP}
    params.setdefault("objective", "regression")
    params.update({"verbose": -1, "random_state": 42, "num_threads": 8})

    conn = sqlite3.connect(str(DB))
    df = pd.read_sql("SELECT * FROM sentiment_features", conn,
                     parse_dates=["matched_earnings_date"])
    prices = pd.read_sql("SELECT date, ticker, close FROM prices", conn,
                         parse_dates=["date"])
    earn = pd.read_sql("SELECT * FROM earnings", conn,
                       parse_dates=["earnings_date"])
    conn.close()

    df = build_features(df, prices, earn)
    feats = list(base_feats)
    if args.with_llm:
        df, llm_feats = attach_llm_features(df, DB)
        llm_feats = [f for f in llm_feats if f not in CRED_FEATS]
        feats += llm_feats

    if args.production:
        train = df.dropna(subset=[TARGET])
        out = ROOT / "models" / f"lgbm_abnormal_30d_production_{date.today():%Y%m%d}.pkl"
        mode = "PRODUCTION — all data; no OOS number exists for this model yet"
    else:
        train = df[df["matched_earnings_date"] <= CUTOFF].dropna(subset=[TARGET])
        out = MODEL_PATH
        mode = (f"RESEARCH — rows <= {CUTOFF.date()} (frozen cutoff + backfilled "
                "ghosts); re-testable ONCE on the 2025-26 OOS window")

    n_llm = len(feats) - len(base_feats)
    print(f"mode:     {mode}")
    print(f"features: {len(feats)} = {len(base_feats)} base"
          + (f" + {n_llm} llm" if n_llm else ""))
    print(f"rows:     {len(train):,} "
          f"({train['matched_earnings_date'].min().date()} .. "
          f"{train['matched_earnings_date'].max().date()})")
    print(f"output:   {out.relative_to(ROOT)}")
    if args.dry_run:
        print("dry run — nothing trained.")
        return 0

    X = train[feats].copy()
    for c in ("ticker", "sector"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    cat = [c for c in ("ticker", "sector") if c in X.columns]
    # raw labels + 300 rounds: matches 04's final fit exactly
    booster = lgb.train(params,
                        lgb.Dataset(X, label=train[TARGET].values,
                                    categorical_feature=cat),
                        num_boost_round=300)
    joblib.dump(booster, out)
    print(f"\nsaved {out.name}")

    gains = pd.Series(booster.feature_importance("gain"), index=feats)
    gains = (gains / gains.sum()).sort_values(ascending=False)
    print("top-10 gain share:")
    print(gains.head(10).round(3).to_string())
    if not args.production:
        print("\nnext (once, pre-registered): "
              ".venv/Scripts/python.exe scripts/run_oos_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
