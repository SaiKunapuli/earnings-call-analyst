# Earnings Call Analyst

A quantitative pipeline that ingests 39k+ real earnings-call transcripts
(HuggingFace + defeatbeta-api, speaker-segmented; SEC EDGAR 8-K scraping kept
as a legacy fallback), computes NLP sentiment scores (VADER, Loughran-McDonald,
FinBERT) and LLM-extracted Q&A features per transcript section, and models the
relationship between earnings-call language and subsequent **post-earnings
drift** using LightGBM + SHAP, finishing with a cost-aware long/short event
backtest and a pre-registered frozen-cutoff out-of-sample test.

## The honest headline (read this before the metrics)

This is a research/portfolio project, and its most valuable result is a
**negative one, found by the project's own evaluation discipline**:

- The signal is *real*: in-sample rank IC +0.066 (t=2.34) on a 31-month
  untouched eval region, confirmed by a frozen-cutoff out-of-sample test on
  2,374 genuinely unseen calls (monthly IC +0.0695 — the in-sample level
  generalizes with no inflation).
- The signal is *untradeable as constructed*: the entire edge lives inside the
  overnight **announcement gap** (calls happen after the close — you cannot own
  that return). Re-anchoring entry to the next tradable close (T+1) flips the
  model's IC to **−0.065 (t=−2.26)**, and a *free* ranking by raw EPS surprise
  beats it (+0.089, t=+3.56).
- LLM-extracted Q&A features (gemini-2.5-flash-lite, 8 ordinal fields) add a
  real lift in a paired A/B (+0.038 IC, t=+3.80), which **deflates to +0.018**
  when measured only in months past the LLM's knowledge cutoff — an
  LLM-memorization control most published backtests skip.
- An earlier, spectacular OOS result (IC 0.33) was traced to a silently
  overwritten "frozen" model file and voided. The forensic method (uniform
  pre/post-cutoff fit as the contamination fingerprint) is now automated in
  `scripts/verify_frozen.py`, and every model artifact is hash-pinned in
  `models/MODEL_MANIFEST.json`.

The full experiment log — every number, dead end, and lesson — lives in
`docs/PROJECT_JOURNAL.md`; a working-paper write-up is in progress from it.

## Pipeline Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  01_pull_prices     │────▶│  02_transcripts     │────▶│  03_sentiment     │────▶│  04_modeling      │────▶│  05_backtest      │
│                     │     │                     │     │                   │     │                   │     │                   │
│  • yfinance prices  │     │  • SEC EDGAR 8-K    │     │  • VADER sentiment│     │  • LightGBM model │     │  • Long/short     │
│  • Earnings dates   │     │    exhibit scraping │     │  • LM dictionary  │     │  • Optuna tuning  │     │    strategy       │
│  • VIX pull         │     │  • Fiscal calendar  │     │  • FinBERT (GPU)  │     │  • Time-series CV │     │  • Sharpe / DD    │
│  • SPY-adj returns  │     │    mapping          │     │  • Readability    │     │  • SHAP analysis  │     │  • Sensitivity    │
│                     │     │                     │     │  • Join w/returns │     │  • Sector models  │     │                   │
└─────────┬───────────┘     └─────────┬───────────┘     └────────┬──────────┘     └─────────┬────────┘     └──────────────────┘
          │                           │                          │                          │
          ▼                           ▼                          ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    data/market.db (SQLite)                                         │
│  tables: prices | earnings | returns | transcripts | sentiment_features | model_predictions        │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> Transcripts come primarily from **02b** (HuggingFace, ~685 companies); **02** is the
> legacy SEC EDGAR scraper shown in the diagram. `run_pipeline.py` executes
> `01 → 03 → 04 → 05` (02b is a one-time ingest — run it with `--ingest`). Tables not
> drawn above: `transcripts_text` (compressed transcript store) and `vix`.

## Notebooks

| # | Notebook | Purpose | Key Output |
|---|----------|---------|------------|
| 01 | `01_pull_prices.ipynb` | Pull prices for the full universe (58 core + every ticker with an HF transcript, ~685) + SPY/XLK/IWM, fetch earnings dates, compute SPY-adjusted abnormal returns, pull VIX | `prices`, `earnings`, `returns`, `vix` tables in SQLite |
| 02 | `02_transcripts.ipynb` | (Legacy) Scrape SEC EDGAR 8-K exhibits (press releases) using `edgartools` | `transcripts` table + `.txt` files in `data/transcripts/` |
| 02b | `02b_ingest_hf_transcripts.ipynb` | **Primary transcript source**: ingest 33k+ real earnings-call transcripts (685 companies, 2005–2025, speaker-segmented) from HuggingFace `kurry/sp500_earnings_transcripts`; split prepared remarks vs Q&A by speaker turns; compute evasiveness stats; store zlib-compressed | `transcripts_text` table in SQLite |
| 03 | `03_sentiment.ipynb` | Compute VADER, Loughran-McDonald, FinBERT sentiment + readability metrics per transcript section (full / prepared remarks / Q&A); join with returns | `sentiment_features` table in SQLite |
| 04 | `04_modeling.ipynb` | Train Optuna-tuned LightGBM regression + classification predicting **30-day post-earnings drift** (`abnormal_30d`); leak-free expanding features; honest tune/eval split; SHAP; sector sub-models | `model_predictions` table + `models/lgbm_abnormal_30d.pkl` |
| 05 | `05_backtest.ipynb` | Backtest a long/short strategy on model predictions (top/bottom 20% by predicted return); **transaction costs**, holding-period-aware Sharpe, drawdown, cutoff sensitivity | Performance metrics + equity curves |

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the pipeline

One command runs the notebooks in dependency order (`01 → 03 → 04 → 05`),
executing each in place so its outputs (tables, charts) are saved into the
notebook:

```bash
python run_pipeline.py            # full pipeline
python run_pipeline.py --dry-run  # preview the plan, run nothing
python run_pipeline.py --from 03  # skip the slow price pull (data already loaded)
python run_pipeline.py --only 04 05
```

On Windows, `.\run.ps1 <args>` wraps the venv Python so you don't type the
full path. Notebook `02b` (the one-time HuggingFace transcript ingest) is
skipped by default — pass `--ingest` to include it. First run must include
`01` to populate prices for the full universe; after that, `--from 03` is the
fast path for model iteration.

After a run, get a quick verdict on whether there's signal:

```bash
python scripts/analyze_results.py   # IC + t-stat, decile spread, sign accuracy, AUC
```

During a long 03 run, a second terminal can watch FinBERT progress live
(the notebook's own prints are buffered until the stage finishes):

```bash
python scripts/monitor_progress.py  # progress bar + rate + ETA; Ctrl+C safe
```

> **Note on evaluation:** tuning and feature selection only ever see the first
> 70% of events by date; reported metrics come from the last 30% via expanding
> walk-forward. Beyond that, a frozen-cutoff protocol governs true OOS claims:
> the cutoff model is hash-pinned (`models/MODEL_MANIFEST.json`), retrains go
> through `scripts/retrain_model.py` (never by re-running 04 casually — it has
> a freeze guard), and `scripts/run_oos_test.py` stamps its output with the
> model hash. `scripts/verify_frozen.py` audits all of it. The 2025-05→2026-05
> OOS window has been used once and is declared spent.

## Data Flow

### 01 → SQLite tables: `prices`, `earnings`, `returns`

- Downloads daily OHLCV for the full universe (58 core names + every ticker with an ingested HF transcript, ~685) + SPY + XLK + IWM (2016-07 to present) via `yfinance`
- Fetches historical earnings dates and EPS surprises
- Computes **SPY-adjusted abnormal returns** (raw return minus market benchmark) at 1d/30d/90d horizons
- Pulls VIX close as a market-regime feature
- Flags COVID-era observations (2020-02-20 to 2021-06-30)

### 02 → SQLite table: `transcripts` + text files

- Uses SEC EDGAR to find 8-K filings near each earnings date
- Extracts exhibit text via BeautifulSoup
- Maps announcement dates to fiscal quarters using per-ticker fiscal-calendar rules
- Stores cleaned transcripts as `.txt` files in `data/transcripts/{TICKER}/`

### 03 → SQLite table: `sentiment_features`

- **VADER**: Chunk-level compound sentiment + distribution stats (mean/std/percentiles)
- **Loughran-McDonald**: Financial-domain dictionary counts (positive, negative, uncertainty, litigious, constraining, modal)
- **FinBERT**: Transformer-based financial sentiment via `ProsusAI/finbert` (accelerated on AMD GPU via DirectML)
- **Readability**: Flesch-Kincaid, Gunning Fog, SMOG, Dale-Chall, automated readability, unique word ratio, avg sentence length
- Features computed **per section**: `full_*`, `prepared_remarks_*`, `qa_*` (Q&A is where unscripted sentiment leaks)
- Joins to returns data by matching transcript pub_date to nearest earnings_date (≤30 day diff)

### 04 → SQLite table: `model_predictions` + `models/`

- **Target**: `abnormal_30d` (SPY-adjusted return over the 30 trading days *after* earnings — the post-earnings-announcement-drift horizon, where signal diffuses slowly, rather than the near-instantly arbitraged 1-day pop). Set `TARGET` in the imports cell to `abnormal_1d` for the hard baseline.
- **Features**: Mutual-information-selected sentiment, readability, momentum, EPS-surprise, and market-regime features + ticker/sector (categorical). All panel features are **point-in-time correct** (`src/features.py`): expanding within-ticker z-scores, quarter-over-quarter deltas, and PEAD priors use only strictly-prior observations — no look-ahead.
- **Honest evaluation**: Optuna tunes on the first 70% of events (by date); performance is reported on the untouched last 30% via expanding walk-forward. Feature selection and tuning never see the eval region.
- **Model**: Optuna-tuned LightGBM regressor and classifier, plus per-sector sub-models under the same protocol.
- **Explainability**: SHAP summary, dependence, and bar plots.
- **Output**: Pickled model + predictions with per-feature SHAP values. Run `scripts/analyze_results.py` afterwards for a one-line verdict (information coefficient + t-stat, decile spread, AUC).

### 05 → Backtest

- Ranks stocks by predicted abnormal return per earnings date
- Long top 20% / short bottom 20%, equal-weight legs, holding period = target horizon
- **Transaction costs** (round-trip spread/slippage + short-leg borrow), reported gross vs. net
- Sharpe annualized by √(252 / holding days), max drawdown, win rate, leg decomposition, cutoff sensitivity
- Benchmarks: equal-weight portfolio, SPY over the same window, long-only variant

## Tech Stack

- **Data**: `yfinance`, `edgartools`, `beautifulsoup4`
- **NLP**: `nltk` (VADER), `pysentiment2` (LM dictionary), `transformers` (FinBERT)
- **GPU**: `torch-directml` for AMD GPU acceleration
- **Modeling**: `lightgbm`, `scikit-learn`, `shap`, `optuna`
- **Viz**: `matplotlib`, `seaborn`, `plotly`
- **Storage**: SQLite (`sqlite3`), `joblib` for model serialization
- **Testing**: `pytest` (unit tests for `src/` in `tests/`)

## Project Structure

```
ECA/
├── notebooks/
│   ├── 01_pull_prices.ipynb
│   ├── 02_transcripts.ipynb          (legacy EDGAR scraper)
│   ├── 02b_ingest_hf_transcripts.ipynb
│   ├── 03_sentiment.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_backtest.ipynb
├── src/
│   ├── config.py                 (shared tickers/benchmarks/dates/paths)
│   ├── sentiment.py              (cleaning, chunking, VADER/LM/readability)
│   ├── transcripts_io.py         (speaker sectioning, evasiveness stats, compressed storage)
│   ├── returns_calc.py           (vectorized forward/trailing return engine)
│   ├── features.py               (leak-free expanding z-scores, QoQ deltas, PEAD priors, FinBERT combine)
│   ├── features_parallel.py      (multiprocess VADER/LM/readability for 39k transcripts)
│   ├── join.py                   (transcript-to-returns matching)
│   ├── llm_features.py           (LLM Q&A extraction: prompt, validation, providers)
│   ├── momentum.py               (Layer 2: price-momentum signal panel)
│   ├── regime.py                 (Layer 7: rules-based risk-on/off exposure gate)
│   ├── ensemble.py               (Layer 9: momentum+ECA score combination w/ event decay)
│   ├── sizing.py                 (Layer 10: vol-balanced decile book construction)
│   └── credibility.py            (research: management words-vs-outcomes track record)
├── scripts/
│   ├── analyze_results.py        (post-run model verdict: IC, decile spread, AUC)
│   ├── monitor_progress.py       (live FinBERT progress bar for long 03 runs)
│   ├── run_llm_extraction.py     (LLM Q&A feature extraction; checkpointed + resumable)
│   ├── llm_ab_test.py            (read-only A/B: do LLM features add IC? + memorization split)
│   ├── llm_mask_check.py         (name-masked re-scoring: is the LLM reading or remembering?)
│   ├── retrain_model.py          (cutoff-safe retrain — the ONLY sanctioned way to retrain)
│   ├── run_oos_test.py           (frozen-cutoff OOS test; output stamped with model hash)
│   ├── verify_frozen.py          (artifact hash audit + contamination fingerprint check)
│   ├── update_transcripts.py     (incremental defeatbeta transcript ingest)
│   └── snapshot_daily.py         (daily analyst-estimate/short-interest recorder)
├── run_pipeline.py               (runs 01→03→04→05 in order via the venv kernel)
├── run.ps1                       (Windows wrapper for run_pipeline.py)
├── tests/                        (pytest unit tests, 160+)
├── docs/                         (gitignored; project journal, paper outline, LLM plan)
├── graphs/
│   └── legacy/                   (gitignored; pre-rebuild charts — current charts live in the notebooks)
├── data/                         (gitignored)
│   ├── market.db                 (SQLite database)
│   └── transcripts/              (legacy EDGAR .txt files; primary store is the transcripts_text DB table)
├── models/                       (tracked + hash-pinned in MODEL_MANIFEST.json)
│   ├── lgbm_abnormal_30d.pkl     (canonical frozen-cutoff model, 23 base + 14 LLM features)
│   └── MODEL_MANIFEST.json       (SHA256 per artifact; audited by verify_frozen.py)
├── requirements.txt
├── .env.example                  (template — copy to .env for LLM extraction keys)
├── .env                          (gitignored; your API keys)
├── .gitignore
└── README.md
```

## Ticker Universe

**Core universe**: 58 curated large-cap US companies across 11 sectors (below).
**Extended universe**: after running `02b_ingest_hf_transcripts.ipynb`, notebook 01
automatically extends to every ticker with an ingested transcript (~685 names) via
`src.config.get_full_universe()`.

| Sector | Tickers |
|--------|---------|
| Technology | MSFT, GOOGL, META, AMZN, NVDA, AMD, INTC, CRM, ORCL, ADBE, CSCO, IBM, NOW, NFLX, DOCU, TWLO, PINS, SNAP, NET, DDOG, SQ, ROKU, CRWD, ZM, TEAM |
| Financials | JPM, BAC, GS, COF |
| Healthcare | JNJ, UNH, PFE, ABT |
| Consumer Cyclical | HD, SBUX, NKE, UBER, DASH, DIS |
| Consumer Defensive | WMT, COST, PG, KO |
| Industrials | CAT, BA, GE, DE, DAL |
| Energy | XOM, CVX, DVN |
| Materials | FCX, NEM |
| Real Estate | PLD, SPG |
| Utilities | NEE, DUK |
| Communication Services | VZ |

Non-calendar fiscal years (MSFT, NVDA, CRM, ORCL, ADBE, CSCO, CRWD, ZM, TEAM,
WMT, HD, COST, SBUX, DIS, NKE, PG, DE) matter only for the **legacy**
`02_transcripts.ipynb` scraper, which maps announcement dates to fiscal quarters.
The primary `02b` HuggingFace transcripts already carry fiscal year/quarter labels,
so no mapping is needed.

**Benchmarks**: SPY (S&P 500 — used for abnormal return adjustment), XLK (Technology Select Sector), IWM (Russell 2000)
