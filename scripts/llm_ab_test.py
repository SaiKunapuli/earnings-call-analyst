"""A/B test: do LLM-extracted Q&A features (and the credibility signal built on
them) add predictive power over the base model's features?

Read-only: does NOT retrain or overwrite the frozen model, model_predictions,
or any table. Runs a self-contained walk-forward on the pilot sub-universe
(tickers with >=60% LLM coverage), three arms with IDENTICAL params/folds:

  A. base       — the frozen model's 23 features
  B. +llm       — A + the 8 LLM ordinal features + their QoQ deltas + optimism
  C. +llm+cred  — B + credibility + cred_weighted_optimism (src/credibility.py)

Gate (docs/llm_qa_plan.md): arm B/C beats A by >= +0.010 mean monthly IC
(paired) or LLM features enter the top-8 gains. Otherwise the text features
don't earn their seat.

    .venv/Scripts/python.exe scripts/llm_ab_test.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.run_oos_test import build_features, MODEL_PATH, TARGET
from src.credibility import load_credibility_features
from src.features import add_qoq_deltas

DB = ROOT / "data" / "market.db"
MIN_COVERAGE = 0.60
MIN_TICKERS = 20
TUNE_FRAC = 0.70
N_FOLDS = 3
LLM_FIELDS = ["guidance_direction", "guidance_confidence", "demand_outlook",
              "margin_outlook", "n_questions_dodged", "tone_numbers_gap",
              "unexpected_negative", "analyst_pushback"]
LLM_QOQ_SRC = ["llm_guidance_direction", "llm_demand_outlook", "llm_margin_outlook",
               "llm_tone_numbers_gap", "llm_analyst_pushback"]


def attach_llm_features(df: pd.DataFrame, db_path=DB) -> tuple[pd.DataFrame, list[str]]:
    """Merge llm_qa_scores onto *df* and derive the full LLM feature set:
    the 8 ordinal fields, the optimism composite, QoQ deltas of the ordinals,
    and the credibility features (src/credibility.py — leak-free).

    Shared by llm_ab_test.py and retrain_model.py so the A/B and any retrain
    build these columns IDENTICALLY. Returns (df, list of feature names
    actually present). NaN where a call wasn't scored — LightGBM handles it.
    """
    conn = sqlite3.connect(str(db_path))
    llm = pd.read_sql("SELECT ticker, quarter, year, scores FROM llm_qa_scores", conn)
    conn.close()
    if llm.empty:
        return df, []
    sc = pd.json_normalize(llm["scores"].map(json.loads))
    sc.columns = [f"llm_{c}" for c in sc.columns]
    llm = pd.concat([llm[["ticker", "quarter", "year"]], sc], axis=1)
    df = df.merge(llm, on=["ticker", "quarter", "year"], how="left")

    w = {"llm_guidance_direction": 1.0, "llm_demand_outlook": 0.5,
         "llm_margin_outlook": 0.5}
    vals = pd.concat([(df[c] * s) for c, s in w.items() if c in df.columns], axis=1)
    df["llm_optimism"] = vals.mean(axis=1, skipna=True)
    df = add_qoq_deltas(df, [c for c in LLM_QOQ_SRC if c in df.columns])

    cred = load_credibility_features(db_path)
    if len(cred):
        df = df.merge(cred[["ticker", "quarter", "year", "credibility",
                            "cred_weighted_optimism"]],
                      on=["ticker", "quarter", "year"], how="left")

    feats = ([f"llm_{f}" for f in LLM_FIELDS] + ["llm_optimism"]
             + [f"{c}_qoq" for c in LLM_QOQ_SRC]
             + ["credibility", "cred_weighted_optimism"])
    return df, [f for f in feats if f in df.columns]


def main() -> int:
    conn = sqlite3.connect(str(DB))
    df = pd.read_sql("SELECT * FROM sentiment_features", conn,
                     parse_dates=["matched_earnings_date"])
    prices = pd.read_sql("SELECT date, ticker, close FROM prices", conn,
                         parse_dates=["date"])
    earn = pd.read_sql("SELECT * FROM earnings", conn, parse_dates=["earnings_date"])
    conn.close()

    # ---- LLM + credibility features (shared helper) ----
    df, all_llm_feats = attach_llm_features(df, DB)
    if not all_llm_feats:
        print("llm_qa_scores is empty — run the extraction first.")
        return 1
    print(f"scored coverage: {df['llm_guidance_confidence'].notna().sum():,} of "
          f"{len(df):,} rows")

    # coverage filter: complete-history tickers only (the pilot design)
    cov = df.assign(has=df["llm_guidance_confidence"].notna()) \
            .groupby("ticker")["has"].mean()
    covered = cov[cov >= MIN_COVERAGE].index.tolist()
    if len(covered) < MIN_TICKERS:
        print(f"Only {len(covered)} tickers with >={MIN_COVERAGE:.0%} coverage "
              f"(need {MIN_TICKERS}). Run the pilot first.")
        return 1
    df = df[df["ticker"].isin(covered)].copy()
    print(f"sub-universe: {len(covered)} tickers, {len(df):,} rows "
          f"({df['matched_earnings_date'].min().date()} .. "
          f"{df['matched_earnings_date'].max().date()})")

    # ---- base features (same pipeline the frozen model used) ----
    df = build_features(df, prices, earn)

    # ---- arms ----
    frozen = joblib.load(MODEL_PATH)
    base_feats = frozen.feature_name()
    cred_only = ("credibility", "cred_weighted_optimism")
    llm_feats = [f for f in all_llm_feats if f not in cred_only]
    cred_feats = [f for f in all_llm_feats if f in cred_only]
    arms = {
        "A base (frozen 23)": base_feats,
        "B +llm": base_feats + llm_feats,
        "C +llm+cred": base_feats + llm_feats + cred_feats,
    }

    # identical params across arms (from the frozen booster, defaults as backstop)
    raw = dict(frozen.params) if frozen.params else {}
    keep = {"objective", "metric", "boosting_type", "num_leaves", "learning_rate",
            "feature_fraction", "bagging_fraction", "bagging_freq",
            "min_data_in_leaf", "min_sum_hessian_in_leaf", "lambda_l1", "lambda_l2"}
    params = {k: v for k, v in raw.items() if k in keep}
    params.setdefault("objective", "regression")
    params.update({"verbose": -1, "random_state": 42, "num_threads": 8})

    df = df.dropna(subset=[TARGET]).sort_values("matched_earnings_date") \
           .reset_index(drop=True)
    y = df[TARGET].values
    dates = df["matched_earnings_date"]
    cut = int(len(df) * TUNE_FRAC)
    bounds = np.linspace(cut, len(df), N_FOLDS + 1).astype(int)
    print(f"walk-forward: train<{dates.iloc[cut].date()} then {N_FOLDS} expanding "
          f"folds over {len(df)-cut:,} eval rows "
          f"({dates.iloc[cut].date()} .. {dates.iloc[-1].date()})\n")

    results, preds_by_arm = {}, {}
    for arm, feats in arms.items():
        X = df[feats].copy()
        for c in ("ticker", "sector"):
            if c in X.columns:
                X[c] = X[c].astype("category")
        pred = np.full(len(df), np.nan)
        for f in range(N_FOLDS):
            tr_end, te_end = bounds[f], bounds[f + 1]
            y_tr = y[:tr_end]
            lo, hi = np.nanquantile(y_tr, [0.01, 0.99])
            m = lgb.train(params, lgb.Dataset(X.iloc[:tr_end],
                                              label=np.clip(y_tr, lo, hi)),
                          num_boost_round=300)
            pred[tr_end:te_end] = m.predict(X.iloc[tr_end:te_end])
        preds_by_arm[arm] = pred

        ev = pd.DataFrame({"d": dates, "p": pred, "a": y}).dropna()
        ev["m"] = ev["d"].dt.to_period("M")
        ics = ev.groupby("m").apply(
            lambda g: spearmanr(g["p"], g["a"])[0] if len(g) >= 5 else np.nan,
            include_groups=False).dropna()
        t = ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 1 else np.nan
        results[arm] = ics
        print(f"{arm:22s} monthly IC {ics.mean():+.4f} (t={t:+.2f}, "
              f"{len(ics)} months, {(ics > 0).mean():.0%} positive)")

        if arm != "A base (frozen 23)":
            gains = pd.Series(m.feature_importance("gain"), index=feats) \
                      .sort_values(ascending=False)
            in_top8 = [f for f in gains.head(8).index
                       if f in llm_feats + cred_feats]
            print(f"{'':22s} top-8 gain features incl. LLM/cred: "
                  f"{in_top8 if in_top8 else 'none'}")

    print("\n=== paired verdict (same months) ===")
    a = results["A base (frozen 23)"]
    # gemini-2.5's training data ends ~Jan 2025: months after that cannot be
    # explained by the LLM "remembering" what happened to the stock.
    kc = pd.Period("2025-02", freq="M")
    for arm in ("B +llm", "C +llm+cred"):
        both = pd.concat([a.rename("a"), results[arm].rename("b")], axis=1).dropna()
        d = both["b"] - both["a"]
        t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan
        gate = "PASSES" if d.mean() >= 0.010 else "fails"
        print(f"{arm:14s} minus base: {d.mean():+.4f} IC (paired t={t:+.2f}) "
              f"-> gate (+0.010) {gate}")
        pre, post = d[d.index < kc], d[d.index >= kc]
        if len(pre) > 2 and len(post) > 2:
            tp = (post.mean() / post.std() * np.sqrt(len(post))
                  if post.std() > 0 else np.nan)
            print(f"{'':14s} memorization check: delta pre-{kc} {pre.mean():+.4f} "
                  f"({len(pre)} mo) | post {post.mean():+.4f} "
                  f"(t={tp:+.2f}, {len(post)} mo beyond LLM knowledge cutoff)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
