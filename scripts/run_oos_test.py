"""Out-of-sample confirmation test for the frozen ECA model.

The eval region in notebook 04 was consulted repeatedly during model design, so
it is a *validation* set, not a clean out-of-sample one. This script runs the
genuinely-honest test: apply the FROZEN model (models/lgbm_abnormal_30d.pkl,
trained on data through 2025-05-15) to earnings calls published AFTER that date
— calls the model has never seen and that never influenced any design choice.

It reproduces notebook 04's leak-free feature pipeline (so features match the
training columns exactly — verified against model.feature_name()), predicts with
the frozen model (NO retraining, NO retuning), joins the realized 30-day drift,
and reports IC / decile spread on the OOS calls versus the in-sample numbers.

PREREQUISITE: notebook 03 must have scored the post-cutoff transcripts first
(run `.\\run.ps1 --only 03`). If sentiment_features still ends at the cutoff,
this script says so and exits.

    .venv/Scripts/python.exe scripts/run_oos_test.py
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.features import (add_past_target_stats, add_qoq_deltas,
                          compute_ticker_z_scores_expanding)
from src.returns_calc import PriceMatrix
from src.sentiment import SECTOR_MAP

DB = ROOT / "data" / "market.db"
MODEL_PATH = ROOT / "models" / "lgbm_abnormal_30d.pkl"
CUTOFF = pd.Timestamp("2025-05-15")     # last training earnings date
TARGET = "abnormal_30d"
# In-sample reference (notebook 04, level model on the reused eval tail):
IS_IC, IS_T, IS_SPREAD = 0.052, 1.88, 217


def build_features(df: pd.DataFrame, prices_long: pd.DataFrame,
                   earnings_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce 04's mod_load + mod_prep feature engineering (leak-free).

    Only the blocks that produce the 23 model features are kept (the vol-
    adjusted-target and LLM blocks are omitted — none of their columns are
    in feature_name()). Built on the FULL panel so each OOS row's expanding
    features use that ticker's genuine prior history.
    """
    # (A) sector
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Other")

    # (B) pre-earnings momentum
    pm = PriceMatrix.from_long(prices_long)
    tk, ed = df["ticker"].to_numpy(), df["matched_earnings_date"].to_numpy()
    df["pre_momentum_5d"] = pm.trailing_returns(tk, ed, 5)
    df["pre_momentum_20d"] = pm.trailing_returns(tk, ed, 20)

    # (C) EPS surprise (nearest within 30d)
    earnings_df = earnings_df.rename(columns={"surprise(%)": "eps_surprise_pct"})
    earnings_df["earnings_date"] = (pd.to_datetime(earnings_df["earnings_date"], utc=True)
                                    .dt.tz_localize(None).dt.normalize())
    lookup = (earnings_df[["ticker", "earnings_date", "eps_surprise_pct"]]
              .dropna(subset=["eps_surprise_pct"]).sort_values("earnings_date"))
    df = df.sort_values("matched_earnings_date")
    df = pd.merge_asof(df, lookup, left_on="matched_earnings_date",
                       right_on="earnings_date", by="ticker", direction="nearest",
                       tolerance=pd.Timedelta(days=30)).drop(columns=["earnings_date"])
    df = df.reset_index(drop=True)

    # (D) expanding within-ticker z-scores (leak-free)
    df = compute_ticker_z_scores_expanding(df, prefix="full", min_obs=3)
    if "qa_vader_mean" in df.columns:
        df = compute_ticker_z_scores_expanding(df, prefix="qa", min_obs=3)

    # (D2) QoQ deltas + PEAD prior (leak-free)
    qoq_cols = [c for c in [
        "full_finbert_net", "full_lm_net", "full_vader_mean", "full_vader_pct_neg",
        "qa_finbert_net", "qa_vader_mean", "hedge_per_1k", "answer_question_ratio",
        "full_flesch_kincaid_grade", "full_n_words",
    ] if c in df.columns]
    df = add_qoq_deltas(df, qoq_cols)
    df = add_past_target_stats(df, target=TARGET, min_obs=2)

    # (mod_prep) Q&A-minus-prepared-remarks deltas
    delta_met = ["vader_compound", "vader_mean", "vader_std", "vader_pct_neg",
                 "vader_pct_pos", "finbert_net", "lm_net", "flesch_reading_ease",
                 "unique_word_ratio"]
    for m in delta_met:
        fc, qc = f"full_{m}", f"qa_{m}"
        if fc in df.columns and qc in df.columns:
            df[f"qa_delta_{m}"] = df[qc] - df[fc]

    # (E) sentiment x EPS-surprise interactions
    surp = df["eps_surprise_pct"].fillna(0)
    for s in ["full_finbert_net", "full_lm_net", "full_vader_mean", "qa_finbert_net"]:
        if s in df.columns:
            df[f"ix_{s}_x_surprise"] = df[s].fillna(0) * surp
        if f"{s}_z" in df.columns:
            df[f"ix_{s}_z_x_surprise"] = df[f"{s}_z"].fillna(0) * surp
    return df


def report(oos: pd.DataFrame, pred_col: str, actual_col: str) -> None:
    d = oos[[pred_col, actual_col, "matched_earnings_date"]].dropna()
    p, a = d[pred_col].to_numpy(), d[actual_col].to_numpy()
    pooled_ic = spearmanr(p, a)[0]
    d = d.assign(m=d["matched_earnings_date"].dt.to_period("M"))
    ics = (d.groupby("m").apply(
        lambda g: spearmanr(g[pred_col], g[actual_col])[0] if len(g) >= 5 else np.nan,
        include_groups=False)
        .dropna())
    mean_ic = ics.mean()
    t_ic = mean_ic / ics.std() * np.sqrt(len(ics)) if len(ics) > 1 and ics.std() > 0 else float("nan")
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    dm = pd.DataFrame({"d": dec, "a": a}).groupby("d")["a"].mean()
    spread = (dm.iloc[-1] - dm.iloc[0]) * 1e4
    sign = float(np.mean(np.sign(p) == np.sign(a))) * 100

    print("\n" + "=" * 64)
    print(f"OUT-OF-SAMPLE RESULT  ({len(d):,} calls, "
          f"{d['matched_earnings_date'].min().date()} .. {d['matched_earnings_date'].max().date()}, "
          f"{len(ics)} months)")
    print("=" * 64)
    print(f"  Monthly IC       {mean_ic:+.4f}   t = {t_ic:+.2f}   "
          f"({(ics > 0).mean()*100:.0f}% of months positive)")
    print(f"  Pooled IC        {pooled_ic:+.4f}")
    print(f"  Decile spread    {spread:+.0f} bps  (D10 - D1)")
    print(f"  Sign accuracy    {sign:.1f}%")
    print(f"\n  In-sample ref:   IC {IS_IC:+.3f} (t {IS_T:+.2f}), spread {IS_SPREAD:+d} bps")
    print("-" * 64)
    if not np.isnan(t_ic) and t_ic >= 1.5 and mean_ic > 0:
        print("  VERDICT: edge HOLDS out-of-sample — the in-sample signal was real,")
        print("           not an artifact of reusing the eval tail.")
    elif mean_ic > 0:
        print("  VERDICT: WEAKER but positive out-of-sample. Direction survives;")
        print("           some in-sample strength was likely eval-tail-inflated.")
    else:
        print("  VERDICT: edge does NOT survive out-of-sample — the in-sample")
        print("           result was an artifact. This is the honest finding.")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Frozen model not found: {MODEL_PATH}. Run notebook 04 first.")
        return 1

    conn = sqlite3.connect(str(DB))
    df = pd.read_sql("SELECT * FROM sentiment_features", conn,
                     parse_dates=["matched_earnings_date"])
    prices_long = pd.read_sql("SELECT date, ticker, close FROM prices", conn,
                              parse_dates=["date"])
    earnings_df = pd.read_sql("SELECT * FROM earnings", conn, parse_dates=["earnings_date"])
    conn.close()

    n_oos_raw = int((df["matched_earnings_date"] > CUTOFF).sum())
    print(f"sentiment_features: {len(df):,} rows | "
          f"post-cutoff (>{CUTOFF.date()}): {n_oos_raw:,}")
    if n_oos_raw < 50:
        print("\nNOT READY: the post-cutoff transcripts are not scored yet.\n"
              "Run notebook 03 first:  .\\run.ps1 --only 03\n"
              "(it resumes from the checkpoint and scores only the new calls).")
        return 1

    df = build_features(df, prices_long, earnings_df)

    model = joblib.load(MODEL_PATH)
    feats = model.feature_name()
    if any(f.startswith(("llm_", "cred")) for f in feats):
        # model was retrained --with-llm: attach the LLM Q&A features the same
        # way the A/B built them (lazy import — llm_ab_test imports this module)
        from scripts.llm_ab_test import attach_llm_features
        df, _ = attach_llm_features(df, DB)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"ERROR: {len(missing)} model features not built: {missing[:8]}")
        return 1

    X = df[feats].copy()
    for c in ("ticker", "sector"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    df["pred"] = model.predict(X)

    oos = df[(df["matched_earnings_date"] > CUTOFF) & df[TARGET].notna()].copy()
    if len(oos) < 50:
        print(f"\nOnly {len(oos)} post-cutoff calls have a realized {TARGET} outcome "
              "yet (need ~30 trading days of prices after each call). Try again "
              "once more forward price data is available.")
        return 1

    # persist for inspection — WITH provenance, so the table can never lie
    # about which artifact produced it (PROJECT_JOURNAL §4.11)
    out = oos[["ticker", "quarter", "year", "matched_earnings_date", "pred", TARGET]].copy()
    out["model_file"] = MODEL_PATH.name
    out["model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16]
    out["run_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = sqlite3.connect(str(DB))
    out.to_sql("oos_predictions", conn, if_exists="replace", index=False)
    conn.close()

    report(oos, "pred", TARGET)
    print(f"\nSaved {len(oos):,} rows to oos_predictions table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
