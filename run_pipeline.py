#!/usr/bin/env python
"""Execute the ECA notebook pipeline in dependency order, in place.

Default order:  01 -> 03 -> 04 -> 05   (the modeling pipeline)

02b (HuggingFace ingest of ~33k transcripts) is a one-time data-loading
step — the transcripts are already in the DB — so it is SKIPPED by default.
Pass --ingest to run it first (re-downloads ~1.8 GB; INSERT-OR-REPLACE, so
it's idempotent).

Each notebook runs with the project's venv as the kernel and its own
directory as the working directory, so relative paths like
'../data/market.db' resolve. The executed notebook (outputs included) is
written back in place, which is what you want for a portfolio. Execution
stops at the first failing cell and the partial outputs are still saved so
you can see where it broke.

Run it with the venv Python so the kernel matches:

    .venv/Scripts/python.exe run_pipeline.py                 # 01,03,04,05
    .venv/Scripts/python.exe run_pipeline.py --ingest        # 02b first
    .venv/Scripts/python.exe run_pipeline.py --only 04 05    # just those
    .venv/Scripts/python.exe run_pipeline.py --from 03       # 03 to the end
    .venv/Scripts/python.exe run_pipeline.py --dry-run       # show the plan
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except ImportError:
    sys.exit("Missing nbclient/nbformat. Install with:\n"
             "    python -m pip install nbclient nbformat ipykernel")

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
KERNEL = "eca-venv"

# (stem, per-cell timeout in seconds; None = no limit)
INGEST = ("02b_ingest_hf_transcripts", None)
PIPELINE = [
    ("01_pull_prices", 3600),   # yfinance: ~685 tickers + earnings (network-bound)
    ("03_sentiment",   None),   # parallel VADER/LM + FinBERT on GPU (hours; no cap)
    ("04_modeling",    3600),   # Optuna tuning + walk-forward + SHAP
    ("05_backtest",    1200),
]


def ensure_kernel() -> None:
    """Register the current interpreter as the `eca-venv` Jupyter kernel."""
    try:
        import ipykernel  # noqa: F401
    except ImportError:
        print("Installing ipykernel into the venv ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipykernel"],
                       check=True)
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--user",
         "--name", KERNEL, "--display-name", "ECA venv"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_notebook(stem: str, timeout) -> float:
    """Execute one notebook in place with the venv kernel; return elapsed sec."""
    path = NB_DIR / f"{stem}.ipynb"
    if not path.exists():
        raise FileNotFoundError(path)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name=KERNEL,
        resources={"metadata": {"path": str(NB_DIR)}},
        allow_errors=False,
    )
    t0 = time.time()
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)   # persist outputs even on failure
    return time.time() - t0


def corpus_rows() -> int | None:
    """Row count of transcripts_text, or None if the DB/table is absent.

    03_sentiment reads its input from this table (populated by 02b). Guard
    against running the pipeline against an empty corpus.
    """
    import sqlite3
    db = ROOT / "data" / "market.db"
    if not db.exists():
        return None
    try:
        c = sqlite3.connect(str(db))
        n = c.execute("SELECT COUNT(*) FROM transcripts_text").fetchone()[0]
        c.close()
        return n
    except sqlite3.OperationalError:
        return None


def _match(prefixes, stems):
    out = []
    for p in prefixes:
        out.extend(s for s in stems
                   if s.startswith(p) or s.split("_")[0] == p)
    return out


def build_plan(args):
    plan = ([INGEST] + PIPELINE) if args.ingest else list(PIPELINE)
    stems = [s for s, _ in plan]
    if args.only:
        want = set(_match(args.only, stems))
        plan = [(s, t) for s, t in plan if s in want]
    elif args.from_:
        hits = _match([args.from_], stems)
        if hits:
            plan = plan[stems.index(hits[0]):]
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the ECA notebook pipeline.")
    ap.add_argument("--ingest", action="store_true",
                    help="include the 02b HuggingFace ingest first")
    ap.add_argument("--only", nargs="+", metavar="NB",
                    help="run only these notebooks (e.g. 04 05)")
    ap.add_argument("--from", dest="from_", metavar="NB",
                    help="run from this notebook to the end")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    args = ap.parse_args()

    plan = build_plan(args)
    if not plan:
        print("Nothing to run — check the --only / --from names.")
        return 1

    print("ECA pipeline plan:")
    for s, t in plan:
        print(f"  - {s:34s} per-cell timeout: {'none' if t is None else f'{t}s'}")
    if args.dry_run:
        return 0

    # Guard: 03 reads transcripts_text; refuse to start if it's empty/missing.
    if any(s == "03_sentiment" for s, _ in plan):
        n = corpus_rows()
        if not n:
            print("\nERROR: transcripts_text is empty or missing — 03_sentiment has "
                  "no corpus to score.\nRun the ingest first:  "
                  ".venv\\Scripts\\python.exe run_pipeline.py --ingest --only 02b")
            return 1
        print(f"\nCorpus check: transcripts_text has {n:,} transcripts.")

    ensure_kernel()
    done = []
    t_all = time.time()
    for stem, timeout in plan:
        print(f"\n{'='*70}\n>> {stem}\n{'='*70}", flush=True)
        try:
            elapsed = run_notebook(stem, timeout)
        except CellExecutionError as e:
            print(f"\nFAILED: {stem} — a cell raised. Pipeline stopped.")
            print(f"  Partial outputs (incl. the error) saved to notebooks/{stem}.ipynb")
            print(f"  {str(e).splitlines()[-1]}")
            return 2
        except Exception as e:  # noqa: BLE001
            print(f"\nFAILED: {stem} — {type(e).__name__}: {e}")
            return 2
        print(f"OK: {stem} in {elapsed/60:.1f} min")
        done.append((stem, elapsed))

    print(f"\n{'='*70}\nPipeline complete in {(time.time()-t_all)/60:.1f} min")
    for s, e in done:
        print(f"  {s:34s} {e/60:6.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
