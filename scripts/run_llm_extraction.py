"""LLM Q&A feature extraction runner — checkpointed + resumable.

Clones the FinBERT Stage-B pattern from notebook 03: scores are flushed to
the `llm_qa_scores` table every FLUSH_EVERY transcripts, keyed by
(ticker, quarter, year); a re-run skips everything already scored, so the
job is pausable/restartable at any point (Ctrl+C is safe).

Usage (from the repo root, with the project venv):

    # what would run, no API calls
    .venv/Scripts/python.exe scripts/run_llm_extraction.py --dry-run

    # smoke: 20 transcripts on the auto-detected provider
    .venv/Scripts/python.exe scripts/run_llm_extraction.py --limit 20

    # the pilot from docs/llm_qa_plan.md: ~65 complete tickers
    .venv/Scripts/python.exe scripts/run_llm_extraction.py --pilot 65

    # billed Gemini, paid-tier concurrency (fast); --dry-run first to see cost
    .venv/Scripts/python.exe scripts/run_llm_extraction.py --paid --pilot 65
    .venv/Scripts/python.exe scripts/run_llm_extraction.py --paid   # full corpus

Provider selection (see src/llm_features.py): --provider gemini|ollama|anthropic,
or auto-detect from .env keys; falls back to local Ollama. `--dry-run` prices
the run (needs no key). `--paid` uses paid-tier rpm/worker defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import DB_PATH
from src.llm_features import LLMExtractionError, get_provider, load_env, score_qa
from src.sentiment import SECTOR_MAP, clean_transcript
from src.transcripts_io import fetch_sections

MIN_PUB_DATE = "2016-07-01"     # must match notebook 03's corpus filter
MIN_QA_WORDS = 200              # skip stub Q&A sections

# Requests/minute ceiling per provider. Free Gemini caps hard (~15 RPM on
# 2.5-flash-lite, ~a few hundred/day); paid Tier-1 flash-lite allows thousands.
FREE_RPM = {"gemini": 12}
PAID_RPM = {"gemini": 1500}     # anything not listed -> 0 (unlimited / tier-bound)

# Pre-flight cost estimate ($ per 1M tokens, PAID tiers). Recalled figures —
# verify against the provider's current pricing page before a large run.
PRICING = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash":      (0.30, 2.50),
    "claude-haiku-4-5":      (1.00, 5.00),
    "claude-sonnet-5":       (3.00, 15.00),
    "claude-opus-4-8":       (5.00, 25.00),
}
AVG_INPUT_TOKENS = 7000    # ~5,000-word Q&A cap + prompt/schema (measured shape)
AVG_OUTPUT_TOKENS = 150

# Default model per provider (mirrors src.llm_features) — used to price the run
# in --dry-run WITHOUT constructing a provider (which would need a key).
DEFAULT_MODEL = {"gemini": "gemini-2.5-flash-lite",
                 "anthropic": "claude-haiku-4-5",
                 "ollama": "qwen2.5:14b-instruct"}


def estimate_cost(model: str, n: int) -> float | None:
    price = PRICING.get(model)
    if not price:
        return None
    inp, out = price
    return n * (AVG_INPUT_TOKENS * inp + AVG_OUTPUT_TOKENS * out) / 1e6


def detect_provider() -> str:
    """Same precedence as src.llm_features.get_provider, without a network call."""
    load_env()
    name = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if name:
        return name
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "ollama"


class RateLimiter:
    """Global requests-per-minute gate shared by all worker threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.time()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS llm_qa_scores (
        ticker  TEXT NOT NULL,
        quarter TEXT NOT NULL,
        year    INTEGER NOT NULL,
        scores  TEXT NOT NULL,
        model   TEXT NOT NULL,
        scored_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (ticker, quarter, year))""")
    conn.commit()


def done_keys(conn: sqlite3.Connection) -> set[tuple]:
    try:
        return {(t, q, int(y)) for t, q, y in conn.execute(
            "SELECT ticker, quarter, year FROM llm_qa_scores")}
    except sqlite3.OperationalError:
        return set()


def candidate_keys(conn: sqlite3.Connection) -> list[tuple]:
    return [(t, q, int(y)) for t, q, y in conn.execute(
        "SELECT ticker, quarter, year FROM transcripts_text "
        "WHERE pub_date >= ? AND has_qa = 1 "
        "ORDER BY ticker, year, quarter", (MIN_PUB_DATE,))]


def pick_pilot_tickers(keys: list[tuple], n_tickers: int, seed: int = 42) -> list[str]:
    """~n complete tickers, stratified by sector (docs/llm_qa_plan.md §4).

    Sampling whole tickers (not random transcripts) keeps each name's panel
    complete so QoQ deltas and expanding stats work in notebook 04.
    """
    import random
    rng = random.Random(seed)
    tickers = sorted({k[0] for k in keys})
    by_sector: dict[str, list[str]] = {}
    for t in tickers:
        by_sector.setdefault(SECTOR_MAP.get(t, "Other"), []).append(t)
    frac = n_tickers / max(len(tickers), 1)
    chosen: list[str] = []
    for sec in sorted(by_sector):
        pool = by_sector[sec]
        k = max(1, round(len(pool) * frac))
        chosen.extend(rng.sample(pool, min(k, len(pool))))
    return sorted(chosen[:n_tickers]) if len(chosen) > n_tickers else sorted(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", choices=["gemini", "ollama", "anthropic"],
                    help="default: auto-detect from .env keys, else local Ollama")
    ap.add_argument("--model", help="override the provider's default model")
    ap.add_argument("--limit", type=int, help="score at most N transcripts")
    ap.add_argument("--tickers", nargs="+", help="restrict to these tickers")
    ap.add_argument("--pilot", type=int, metavar="N",
                    help="restrict to ~N complete tickers, sector-stratified (seed 42)")
    ap.add_argument("--workers", type=int,
                    help="parallel API calls (default: 4 free / 24 paid / 1 ollama)")
    ap.add_argument("--rpm", type=float,
                    help="max requests/minute across all workers "
                         "(default: 12 gemini-free / 1500 gemini-paid / unlimited)")
    ap.add_argument("--paid", action="store_true",
                    help="paid-tier concurrency defaults (higher rpm + workers)")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would run; no API calls, no writes")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_table(conn)
    keys = candidate_keys(conn)
    if args.pilot:
        pilot = set(pick_pilot_tickers(keys, args.pilot))
        keys = [k for k in keys if k[0] in pilot]
        print(f"pilot: {len(pilot)} tickers -> {len(keys):,} transcripts")
    if args.tickers:
        allow = {t.upper() for t in args.tickers}
        keys = [k for k in keys if k[0] in allow]
    done = done_keys(conn)
    todo = [k for k in keys if k not in done]
    if args.limit:
        todo = todo[:args.limit]

    # --- Cost pre-flight (works before any key is set, e.g. under --dry-run) ---
    prov_name = args.provider or detect_provider()
    model_name = args.model or DEFAULT_MODEL.get(prov_name, "")
    cost = estimate_cost(model_name, len(todo))
    cost_str = (f"~${cost:,.2f}" if cost is not None
                else "n/a (free/local or unpriced model)")
    print(f"corpus: {len(keys):,} Q&A transcripts | already scored: "
          f"{len(done & set(keys)):,} | to do: {len(todo):,}")
    print(f"est. cost: {cost_str}  ({prov_name}/{model_name or '?'}, "
          f"~{AVG_INPUT_TOKENS:,} in + {AVG_OUTPUT_TOKENS} out tok/tx)")
    if args.dry_run or not todo:
        conn.close()
        print("dry run — nothing scored." if args.dry_run else "nothing to do.")
        return 0

    # Resolve concurrency: explicit flags win; else free vs paid defaults.
    workers = args.workers if args.workers is not None else (
        1 if prov_name == "ollama" else 24 if args.paid else 4)
    rpm_table = PAID_RPM if args.paid else FREE_RPM
    rpm = args.rpm if args.rpm is not None else rpm_table.get(prov_name, 0)

    provider = get_provider(args.provider, args.model)
    limiter = RateLimiter(rpm)
    _orig_complete = provider.complete
    def _throttled(prompt):          # every API call (incl. retries) waits its turn
        limiter.wait()
        return _orig_complete(prompt)
    provider.complete = _throttled
    print(f"provider: {provider.name} ({provider.model}) | workers: {workers} "
          f"| flush every {args.flush_every} | rpm: {rpm or 'unlimited'} "
          f"| tier: {'paid' if args.paid else 'free'}")

    # Main thread owns SQLite (reads + writes); workers only make API calls.
    def fetch_qa(key):
        secs = fetch_sections(conn, *key)
        return clean_transcript(secs.get("qa", ""))

    n_ok = n_skip = n_fail = 0
    consec_fail = 0
    buffer: list[tuple] = []
    t0 = time.time()

    def flush():
        nonlocal buffer
        if buffer:
            conn.executemany(
                "INSERT OR REPLACE INTO llm_qa_scores (ticker, quarter, year, scores, model) "
                "VALUES (?, ?, ?, ?, ?)", buffer)
            conn.commit()
            buffer = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # submit in chunks so texts (fetched in main thread) don't pile up in RAM
        CHUNK = max(workers * 8, 32)
        for c0 in range(0, len(todo), CHUNK):
            chunk = todo[c0:c0 + CHUNK]
            futures = {}
            for key in chunk:
                qa = fetch_qa(key)
                if len(qa.split()) < MIN_QA_WORDS:
                    n_skip += 1
                    continue
                futures[pool.submit(score_qa, provider, qa)] = key
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    scores = fut.result()
                    buffer.append((key[0], key[1], key[2],
                                   json.dumps(scores), provider.model))
                    n_ok += 1
                    consec_fail = 0
                except LLMExtractionError as e:
                    n_fail += 1
                    consec_fail += 1
                    print(f"\nFAIL {key}: {e}")
                    if consec_fail >= 5:
                        flush()
                        conn.close()
                        print("5 consecutive failures — aborting (checkpoint saved; "
                              "re-run resumes where it left off).")
                        return 2
                if len(buffer) >= args.flush_every:
                    flush()
                donecount = n_ok + n_skip + n_fail
                rate = n_ok / max(time.time() - t0, 1) * 60
                eta_min = (len(todo) - donecount) / max(rate, 0.01)
                print(f"\r{donecount:,}/{len(todo):,}  ok={n_ok} skip={n_skip} "
                      f"fail={n_fail}  {rate:.1f} tx/min  ETA {eta_min:.0f}m   ",
                      end="", flush=True)

    flush()
    total = conn.execute("SELECT COUNT(*) FROM llm_qa_scores").fetchone()[0]
    conn.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min — ok={n_ok} skip={n_skip} "
          f"fail={n_fail}; llm_qa_scores now holds {total:,} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
