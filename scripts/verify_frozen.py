"""Verify the model artifacts are what they claim to be.

Born from the §4.11 contamination (PROJECT_JOURNAL): the "frozen" model had
been silently overwritten by a pipeline re-run, and every leak diagnostic
interrogated the data while nobody checked the artifact. This script makes
that check routine:

  1. HASH CHECK (fast, default): every artifact listed in
     models/MODEL_MANIFEST.json must exist and match its recorded SHA256.
  2. FINGERPRINT CHECK (--fingerprint, ~2-4 min): predicts the full panel
     with the canonical model and compares pooled IC on pre-cutoff
     (training-era) vs post-cutoff rows. A genuinely frozen-cutoff model
     fits its training era much better than unseen data; NEAR-UNIFORM fit
     (post >= ~80% of pre) is the contamination signature that exposed §4.11.

  --update rewrites the manifest from the files currently on disk (do this
  deliberately, only when a new artifact is intentionally created).

    .venv/Scripts/python.exe scripts/verify_frozen.py
    .venv/Scripts/python.exe scripts/verify_frozen.py --fingerprint
    .venv/Scripts/python.exe scripts/verify_frozen.py --update
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODELS = ROOT / "models"
MANIFEST = MODELS / "MODEL_MANIFEST.json"
UNIFORM_RATIO = 0.80   # post-cutoff IC >= 80% of pre-cutoff IC -> suspicious

DESCRIPTIONS = {
    "lgbm_abnormal_30d.pkl":
        "CANONICAL. Clean frozen-cutoff retrain (<=2025-05-15, 23 base + 14 "
        "llm features). Owns the honest OOS numbers (journal §4.13).",
    "lgbm_abnormal_30d_frozen20250515_llm_v2.pkl":
        "Immutable archive of the canonical clean retrain (identical bytes "
        "at creation).",
    "lgbm_abnormal_30d_CONTAMINATED_trained_thru_202605.pkl":
        "RECIPE SOURCE ONLY — trained through 2026-05 (journal §4.11). "
        "Params/feature list safe (from pre-2023-08 tune split); weights are "
        "NOT. Never predict with it.",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_update() -> int:
    entries = {}
    for p in sorted(MODELS.glob("*.pkl")):
        entries[p.name] = {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "description": DESCRIPTIONS.get(p.name, "(undocumented artifact)"),
        }
    MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"manifest written: {MANIFEST.relative_to(ROOT)} ({len(entries)} artifacts)")
    for name in entries:
        print(f"  {name}  {entries[name]['sha256'][:16]}")
    return 0


def cmd_check() -> int:
    if not MANIFEST.exists():
        print("No manifest yet — run with --update to create it.")
        return 1
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for name, meta in entries.items():
        p = MODELS / name
        if not p.exists():
            print(f"MISSING   {name}")
            bad += 1
            continue
        h = sha256(p)
        if h != meta["sha256"]:
            print(f"MODIFIED  {name}")
            print(f"          manifest {meta['sha256'][:16]}.. != disk {h[:16]}..")
            bad += 1
        else:
            print(f"ok        {name}  {h[:16]}")
    # artifacts on disk that the manifest doesn't know about
    for p in sorted(MODELS.glob("*.pkl")):
        if p.name not in entries:
            print(f"UNTRACKED {p.name}  (run --update deliberately if intentional)")
            bad += 1
    if bad:
        print(f"\n{bad} problem(s). If a change was intentional, re-run with "
              "--update AND note it in docs/PROJECT_JOURNAL.md.")
    else:
        print("\nAll artifacts match the manifest.")
    return 1 if bad else 0


def cmd_fingerprint() -> int:
    """Uniform pre/post-cutoff fit = the §4.11 contamination signature."""
    import joblib
    import pandas as pd
    import sqlite3
    from scipy.stats import spearmanr

    from scripts.run_oos_test import CUTOFF, MODEL_PATH, TARGET, build_features

    conn = sqlite3.connect(str(ROOT / "data" / "market.db"))
    df = pd.read_sql("SELECT * FROM sentiment_features", conn,
                     parse_dates=["matched_earnings_date"])
    prices = pd.read_sql("SELECT date, ticker, close FROM prices", conn,
                         parse_dates=["date"])
    earn = pd.read_sql("SELECT * FROM earnings", conn, parse_dates=["earnings_date"])
    conn.close()

    df = build_features(df, prices, earn)
    model = joblib.load(MODEL_PATH)
    feats = model.feature_name()
    if any(f.startswith(("llm_", "cred")) for f in feats):
        from scripts.llm_ab_test import attach_llm_features
        df, _ = attach_llm_features(df, ROOT / "data" / "market.db")
    X = df[feats].copy()
    for c in ("ticker", "sector"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    df["pred"] = model.predict(X)

    d = df.dropna(subset=["pred", TARGET])
    pre = d[d["matched_earnings_date"] <= CUTOFF]
    post = d[d["matched_earnings_date"] > CUTOFF]
    ic_pre = spearmanr(pre["pred"], pre[TARGET])[0]
    ic_post = spearmanr(post["pred"], post[TARGET])[0] if len(post) > 50 else float("nan")
    print(f"pooled IC pre-cutoff (training era): {ic_pre:+.4f}  (n={len(pre):,})")
    print(f"pooled IC post-cutoff (unseen):      {ic_post:+.4f}  (n={len(post):,})")
    if pd.notna(ic_post) and ic_pre > 0 and ic_post >= UNIFORM_RATIO * ic_pre:
        print(f"\nSUSPICIOUS: near-uniform fit (post >= {UNIFORM_RATIO:.0%} of pre) — "
              "the §4.11 contamination signature. Check whether the canonical "
              "model was retrained past the cutoff.")
        return 1
    print("\nAsymmetric fit — consistent with a genuinely frozen-cutoff model.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true",
                    help="rewrite the manifest from files on disk (deliberate)")
    ap.add_argument("--fingerprint", action="store_true",
                    help="also run the pre/post-cutoff uniform-fit check (~2-4 min)")
    args = ap.parse_args()
    if args.update:
        return cmd_update()
    rc = cmd_check()
    if args.fingerprint:
        rc = max(rc, cmd_fingerprint())
    return rc


if __name__ == "__main__":
    sys.exit(main())
