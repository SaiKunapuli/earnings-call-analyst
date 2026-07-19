"""Name-masked LLM robustness check (PROJECT_JOURNAL §5.7 — designed there,
run 2026-07-09).

Question: are the LLM Q&A scores transcript-grounded, or is the model partly
"remembering" the company (training-data hindsight)? Test: re-score a sample
of ALREADY-SCORED calls with the company's identity masked (company name and
ticker replaced by neutral tokens) and measure agreement with the original
scores.

Design details:
- Sample is stratified around gemini-2.5's ~Jan-2025 knowledge cutoff:
  N/2 calls published before 2025-02-01 (memorization POSSIBLE) and N/2 after
  (memorization IMPOSSIBLE). If identity-memory drives scores, masked-vs-
  original divergence should be LARGER in the pre-cutoff half; the post half
  acts as the noise floor (prompt-sensitivity + sampling jitter).
- Masked scores land in a SEPARATE table (`llm_qa_scores_masked`) — the real
  feature table is never touched. Resume-by-key like the main runner.
- Company names come from yfinance shortName (cached in-run); masking removes
  the name, its suffix-stripped variant, its first distinctive token, and the
  exact-uppercase ticker (len>=2 — avoids destroying prose for tickers like
  "A"). Replacement token: "the Company".

    .venv/Scripts/python.exe scripts/llm_mask_check.py --dry-run
    .venv/Scripts/python.exe scripts/llm_mask_check.py --paid -n 200
    .venv/Scripts/python.exe scripts/llm_mask_check.py --report-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import DB_PATH
from src.llm_features import (FEATURE_SPEC, LLMExtractionError, get_provider,
                              score_qa)
from src.sentiment import clean_transcript
from src.transcripts_io import fetch_sections
from scripts.run_llm_extraction import RateLimiter, MIN_QA_WORDS

KNOWLEDGE_CUTOFF = "2025-02-01"     # gemini-2.5 training data ends ~Jan 2025
MASK_TOKEN = "the Company"
NAME_SUFFIXES = re.compile(
    r",?\s+(inc\.?|incorporated|corp\.?|corporation|company|co\.?|ltd\.?|plc|"
    r"group|holdings?|technologies|international|limited)\s*$", re.IGNORECASE)


def company_names(tickers: list[str]) -> dict[str, str]:
    """ticker -> shortName via yfinance; empty string on failure (ticker-only
    masking still applies)."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            out[t] = (yf.Ticker(t).info or {}).get("shortName") or ""
        except Exception:
            out[t] = ""
    return out


def mask_text(text: str, ticker: str, name: str) -> str:
    variants: list[str] = []
    if name:
        variants.append(name)
        stripped = NAME_SUFFIXES.sub("", name).strip()
        if stripped and stripped.lower() != name.lower():
            variants.append(stripped)
        first = stripped.split()[0] if stripped.split() else ""
        # first token only if distinctive (avoid masking generic words)
        if len(first) >= 4 and first.lower() not in {"first", "american", "united",
                                                     "general", "national", "global"}:
            variants.append(first)
    for v in sorted(set(variants), key=len, reverse=True):
        text = re.sub(re.escape(v), MASK_TOKEN, text, flags=re.IGNORECASE)
    if len(ticker) >= 2:
        text = re.sub(rf"\b{re.escape(ticker.upper())}\b", MASK_TOKEN, text)
    return text


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS llm_qa_scores_masked (
        ticker  TEXT NOT NULL,
        quarter TEXT NOT NULL,
        year    INTEGER NOT NULL,
        scores  TEXT NOT NULL,
        model   TEXT NOT NULL,
        pub_date TEXT,
        scored_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (ticker, quarter, year))""")
    conn.commit()


def pick_sample(conn: sqlite3.Connection, n: int, seed: int = 42) -> pd.DataFrame:
    """N/2 scored calls before the knowledge cutoff, N/2 after."""
    df = pd.read_sql(
        "SELECT s.ticker, s.quarter, s.year, t.pub_date "
        "FROM llm_qa_scores s JOIN transcripts_text t "
        "ON s.ticker=t.ticker AND s.quarter=t.quarter AND s.year=t.year "
        "WHERE t.has_qa = 1", conn, parse_dates=["pub_date"])
    rng = random.Random(seed)
    pre = df[df["pub_date"] < KNOWLEDGE_CUTOFF]
    post = df[df["pub_date"] >= KNOWLEDGE_CUTOFF]
    take = lambda part, k: part.iloc[sorted(rng.sample(range(len(part)),
                                                       min(k, len(part))))]
    return pd.concat([take(pre, n // 2), take(post, n - n // 2)], ignore_index=True)


def report(conn: sqlite3.Connection) -> int:
    orig = pd.read_sql("SELECT ticker, quarter, year, scores FROM llm_qa_scores", conn)
    mask = pd.read_sql(
        "SELECT ticker, quarter, year, scores AS scores_m, pub_date "
        "FROM llm_qa_scores_masked", conn, parse_dates=["pub_date"])
    if mask.empty:
        print("no masked scores yet — run the scoring pass first.")
        return 1
    m = mask.merge(orig, on=["ticker", "quarter", "year"], how="inner")
    o = pd.json_normalize(m["scores"].map(json.loads))
    k = pd.json_normalize(m["scores_m"].map(json.loads))
    pre_mask = m["pub_date"] < KNOWLEDGE_CUTOFF

    print(f"\n=== name-masked vs original agreement ({len(m)} calls: "
          f"{int(pre_mask.sum())} pre-cutoff / {int((~pre_mask).sum())} post) ===")
    print(f"{'field':22s} {'spearman':>9s} {'exact':>7s} {'|diff|':>7s} "
          f"{'|diff| pre':>11s} {'|diff| post':>12s}")
    rows = []
    for f in FEATURE_SPEC:
        a, b = o[f].astype(float), k[f].astype(float)
        ok = a.notna() & b.notna()
        if ok.sum() < 20:
            continue
        rho = spearmanr(a[ok], b[ok])[0]
        exact = (a[ok] == b[ok]).mean()
        diff = (a - b).abs()
        rows.append((f, rho, exact, diff[ok].mean(),
                     diff[ok & pre_mask].mean(), diff[ok & ~pre_mask].mean()))
        print(f"{f:22s} {rho:>+9.3f} {exact:>6.0%} {diff[ok].mean():>7.3f} "
              f"{diff[ok & pre_mask].mean():>11.3f} {diff[ok & ~pre_mask].mean():>12.3f}")

    # the verdict metric: is pre-cutoff divergence larger than post?
    d_pre = np.mean([r[4] for r in rows])
    d_post = np.mean([r[5] for r in rows])
    # nanmean: a field constant across the sample has undefined spearman
    rho_all = np.nanmean([r[1] for r in rows])
    print(f"\nmean |diff| pre-cutoff (memory possible):   {d_pre:.3f}")
    print(f"mean |diff| post-cutoff (noise floor):      {d_post:.3f}")
    print(f"mean spearman across fields:                {rho_all:+.3f}")
    if d_post > 0 and d_pre > 1.5 * d_post:
        print("\nVERDICT: masking changes PRE-cutoff scores substantially more than "
              "post — consistent with identity-memory leaking into scores. "
              "Treat pre-cutoff LLM features with the §4.12 deflation (or worse).")
    else:
        print("\nVERDICT: pre-cutoff divergence is comparable to the post-cutoff "
              "noise floor — no evidence that company identity drives the scores. "
              "(Complements the §4.12 temporal memorization deflation.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=200, help="sample size (default 200)")
    ap.add_argument("--paid", action="store_true", help="paid-tier concurrency")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="skip scoring; just compare what's already in the table")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    ensure_table(conn)
    if args.report_only:
        rc = report(conn)
        conn.close()
        return rc

    sample = pick_sample(conn, args.n)
    done = {(t, q, int(y)) for t, q, y in conn.execute(
        "SELECT ticker, quarter, year FROM llm_qa_scores_masked")}
    todo = [r for r in sample.itertuples(index=False)
            if (r.ticker, r.quarter, int(r.year)) not in done]
    est = len(todo) * (7000 * 0.10 + 150 * 0.40) / 1e6
    print(f"sample: {len(sample)} calls | already masked-scored: "
          f"{len(sample) - len(todo)} | to do: {len(todo)} | est. ~${est:.2f}")
    if args.dry_run:
        conn.close()
        print("dry run — nothing scored.")
        return 0

    if todo:
        tickers = sorted({r.ticker for r in todo})
        print(f"fetching company names for {len(tickers)} tickers...")
        names = company_names(tickers)
        n_named = sum(1 for v in names.values() if v)
        print(f"names resolved: {n_named}/{len(tickers)} "
              "(unresolved fall back to ticker-only masking)")

        provider = get_provider()
        limiter = RateLimiter(1500 if args.paid else 12)
        _orig = provider.complete
        provider.complete = lambda p: (limiter.wait() or _orig(p))
        workers = 24 if args.paid else 4
        print(f"provider: {provider.name} ({provider.model}) | workers {workers}")

        n_ok = n_skip = n_fail = 0
        buf: list[tuple] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for r in todo:
                qa = clean_transcript(
                    fetch_sections(conn, r.ticker, r.quarter, int(r.year)).get("qa", ""))
                if len(qa.split()) < MIN_QA_WORDS:
                    n_skip += 1
                    continue
                masked = mask_text(qa, r.ticker, names.get(r.ticker, ""))
                fut = pool.submit(score_qa, provider, masked)
                futures[fut] = r
            for fut in as_completed(futures):
                r = futures[fut]
                try:
                    scores = fut.result()
                    buf.append((r.ticker, r.quarter, int(r.year),
                                json.dumps(scores), provider.model,
                                str(pd.Timestamp(r.pub_date).date())))
                    n_ok += 1
                except LLMExtractionError as e:
                    n_fail += 1
                    print(f"\nFAIL {r.ticker} {r.quarter} {r.year}: {e}")
                if len(buf) >= 50:
                    conn.executemany(
                        "INSERT OR REPLACE INTO llm_qa_scores_masked "
                        "(ticker, quarter, year, scores, model, pub_date) "
                        "VALUES (?,?,?,?,?,?)", buf)
                    conn.commit()
                    buf = []
                print(f"\r{n_ok + n_skip + n_fail}/{len(todo)} ok={n_ok} "
                      f"skip={n_skip} fail={n_fail}   ", end="", flush=True)
        if buf:
            conn.executemany(
                "INSERT OR REPLACE INTO llm_qa_scores_masked "
                "(ticker, quarter, year, scores, model, pub_date) "
                "VALUES (?,?,?,?,?,?)", buf)
            conn.commit()
        print(f"\nscored in {(time.time() - t0) / 60:.1f} min")

    rc = report(conn)
    conn.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
