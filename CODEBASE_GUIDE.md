% ECA — Earnings Call Analyst: A Codebase Guide
% Plain-language walkthrough for a new reader
% Generated 2026-07-08

---

# 1. The 10,000-Foot View

## What this project actually does

**ECA (Earnings Call Analyst)** is a *quantitative research pipeline*. In one sentence:

> It reads what company executives *said* on their earnings calls, turns that language into numbers, and asks: **does the way management talks predict how the stock moves over the next month?**

The specific bet it chases is a real, documented stock-market pattern called **PEAD — post-earnings-announcement drift**: after a company reports earnings, its stock tends to keep drifting in the same direction for weeks, slower than an efficient market "should" allow. ECA tries to predict that 30-day drift from earnings-call text plus a handful of numeric signals, then simulates a **long/short trading strategy** (buy the names it thinks will drift up, short the ones it thinks will drift down) to see whether the prediction is worth anything after trading costs.

It is explicitly a **research and portfolio project**, not a product — nobody is trading real money with it. The value is in doing the science *honestly*: no data leakage, genuine out-of-sample testing, and cost-aware backtests that don't lie to you.

## The problem it solves (and why it's hard)

Predicting stock returns is close to impossible — markets are efficient enough that most "signals" are noise or already priced in. The project's whole discipline is built around *not fooling yourself*:

- **Data leakage** — accidentally letting the model peek at future information — makes a useless model look brilliant. Huge amounts of the code exist purely to prevent this (the word "leak-free" appears everywhere).
- **Overfitting** — tuning until the backtest looks great on data you've already seen. The project uses an "honest" tune/evaluate split to fight this.
- **Fake profits** — a strategy that looks profitable until you subtract realistic trading costs. The backtest always reports results *net of costs*.

## Tech stack

Everything is **Python**. There is no web server, no front-end, no database server — it's a data-science pipeline. Key libraries, grouped by what they do:

| Job | Libraries |
|---|---|
| Data wrangling | `pandas`, `numpy` (the backbone of everything) |
| Market data | `yfinance` (Yahoo Finance prices), `edgartools` (SEC filings, legacy), `datasets`/`huggingface_hub` (the transcript corpus) |
| NLP / text scoring | `nltk` (VADER sentiment), `pysentiment2` (Loughran-McDonald finance dictionary), `transformers` + `torch` + `torch-directml` (FinBERT, a finance BERT model run on an AMD GPU), `textstat` (readability) |
| Machine learning | `lightgbm` (the gradient-boosted-tree model), `optuna` (hyperparameter tuning), `scikit-learn` (utilities, metrics), `shap` (explaining model predictions) |
| Storage | `sqlite3` (built into Python) — a single file, `data/market.db`, is the whole "database" |
| Orchestration | `jupyter` / `nbclient` / `nbformat` (the pipeline is a set of notebooks run programmatically) |
| Testing | `pytest` |

**One term to define now: a "notebook."** A Jupyter notebook (`.ipynb` file) is a document that mixes code cells, their output (tables, charts), and notes. Normally you run them by hand in a browser. This project is unusual: it runs the notebooks *automatically, in order*, like a script — more on that below.

## How you'd run it, and where execution starts

The **entry point is `run_pipeline.py`** at the project root. You run it with the project's virtual-environment Python:

```bash
.venv/Scripts/python.exe run_pipeline.py            # run the whole modeling pipeline
.venv/Scripts/python.exe run_pipeline.py --dry-run  # just print the plan
.venv/Scripts/python.exe run_pipeline.py --from 03  # skip the slow price download
```

(`run.ps1` is a one-line Windows convenience wrapper that types the long venv path for you.)

`run_pipeline.py` doesn't contain the analysis itself. Its job is to **execute the notebooks in dependency order** — `01 → 03 → 04 → 05` — each one reading tables the previous one wrote. Think of it as a conductor: the notebooks are the musicians, and the conductor just makes sure they play in the right sequence. (Notebook `02b`, the one-time transcript download, is skipped by default because the data is already stored.)

There are also **standalone scripts** in `scripts/` (run directly, not through the pipeline) for side jobs: scoring transcripts with a large language model, recording daily analyst-estimate snapshots, updating the transcript corpus, and printing a quick verdict on the latest model.

## Overall architecture in a few sentences

The system is a **linear data pipeline built around one SQLite file**. Each stage reads tables from `data/market.db`, does one job (pull prices, score sentiment, train a model, backtest), and writes its results back as new tables. The heavy, reusable logic lives in a `src/` Python package that the notebooks import; the notebooks themselves are mostly orchestration and charts. A separate, newer layer of `src/` modules (momentum, regime, ensemble, sizing, credibility) explores turning the single earnings-call signal into a multi-signal "trading-bot" research stack.

```
                            data/market.db  (one SQLite file = the shared "warehouse")
                                    ▲   ▲   ▲   ▲
        writes prices/returns ──────┘   │   │   └────── writes model_predictions
                                        │   │
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │    01    │──▶│   02b    │──▶│    03    │──▶│    04    │──▶│    05    │
   │  prices  │   │ ingest   │   │sentiment │   │ modeling │   │ backtest │
   │ +returns │   │transcripts│  │ (NLP)    │   │(LightGBM)│   │(long/short)│
   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
        │              │              │               │              │
        └──────────────┴──────────────┴───────────────┴──────────────┘
                     all import shared logic from  src/*.py
                (returns_calc, sentiment, features, transcripts_io, join, ...)

   run_pipeline.py  = the conductor that runs 01→03→04→05 in order
   scripts/*.py     = side tools (LLM scoring, daily snapshots, corpus updates)
   src/{momentum,regime,ensemble,sizing,credibility}.py = newer multi-signal research
```

The mental model to hold: **notebooks are pipeline *stages*; `src/` is the *library* they share; `market.db` is the *conveyor belt* passing data between stages.**

---

# 2. The Map — Folder Structure & Data Flow

## What lives where

```
ECA/
├── run_pipeline.py     ← ENTRY POINT: runs the notebooks in order
├── run.ps1             ← one-line Windows wrapper for the above
├── requirements.txt    ← exact library versions (the "ingredients list")
├── README.md           ← human overview of the pipeline
│
├── notebooks/          ← THE PIPELINE STAGES (01–07). Where analysis happens.
├── src/                ← THE SHARED LIBRARY. Reusable, tested logic.
├── scripts/            ← STANDALONE TOOLS run outside the pipeline.
├── tests/              ← pytest unit tests for everything in src/.
│
├── data/               ← DATA (git-ignored, not shipped). market.db lives here.
├── models/             ← saved trained models (*.pkl).
├── graphs/             ← exported charts (legacy ones archived in graphs/legacy/).
├── docs/               ← deep-dive docs: PROJECT_JOURNAL, trading_bot_layers, etc.
└── .github/workflows/  ← daily_snapshot.yml: a scheduled cloud job (GitHub Actions)
```

### The folders that actually matter

**`src/` — the brain.** This is where you spend most of your reading time. It's a normal Python *package* (a folder with an `__init__.py` that lets you write `from src.sentiment import ...`). Every non-trivial algorithm lives here so it can be **unit-tested without spinning up a notebook**. That separation — "logic in `src/`, orchestration in notebooks" — is the single most important design decision in the repo. Files: `config`, `returns_calc`, `sentiment`, `transcripts_io`, `features`, `features_parallel`, `join` (the original pipeline), plus `momentum`, `regime`, `ensemble`, `sizing`, `credibility`, `llm_features` (the newer research layer).

**`notebooks/` — the pipeline.** Numbered `01`–`07` so the order is obvious. `01`–`05` are the core pipeline; `06`–`07` are newer signal research. Each is a thin orchestration layer: load tables → call `src/` functions → write tables → draw charts.

**`scripts/` — the side tools.** Things you run occasionally and directly, not part of the main flow: `run_llm_extraction.py` (score calls with an LLM), `snapshot_daily.py` (record analyst estimates each day), `update_transcripts.py` (pull newer transcripts), `analyze_results.py` (quick model verdict), `monitor_progress.py` (a live progress bar for the slow GPU stage).

**`tests/` — the safety net.** One test file per `src/` module. These are how you confirm the leak-free math actually is leak-free. 159 tests, all passing.

### The folders you can mostly ignore at first

- **`data/`** — generated data, not code. Git-ignores everything except two tiny committed files (`snapshots.db`, `universe.txt`) so a cloud job can use them. `market.db` (~1.3 GB) is rebuilt by the pipeline, never committed. (You may also see a stray `NO_SUCH.db` — a throwaway test artifact.)
- **`models/`, `graphs/`** — outputs, not inputs.
- **`docs/`** — excellent background reading (especially `PROJECT_JOURNAL.md`, which narrates every experiment and result), but not needed to understand the code.
- **`.github/workflows/`** — one YAML file describing a scheduled task that runs in GitHub's cloud. *GitHub Actions* is a service that runs a script for you on a timer; here it records analyst estimates every weekday so a proprietary dataset builds up over time.

## How data flows through the system

Follow one earnings call from raw text to a trade. The **currency passed between every stage is a SQLite table** — no stage talks to another directly; they communicate only through `data/market.db`.

```
1. yfinance  ─────────▶  prices, earnings, vix tables         (notebook 01)
   Yahoo Finance          + returns table (30-day post-earnings drift, etc.)

2. HuggingFace ────────▶  transcripts_text table              (notebook 02b)
   33k call transcripts   (zlib-COMPRESSED text, split into prepared remarks / Q&A)

3. transcripts_text ───▶  sentiment_features table            (notebook 03)
   + returns              (VADER + Loughran-McDonald + FinBERT + readability
                           scores per section, JOINED to the stock's drift)

4. sentiment_features ─▶  model_predictions table             (notebook 04)
                          + models/lgbm_abnormal_30d.pkl
   (LightGBM learns language→drift; predicts on unseen data; SHAP explains it)

5. model_predictions ──▶  performance metrics + equity curves (notebook 05)
   + prices              (rank stocks by prediction, go long top / short bottom,
                          subtract trading costs, measure Sharpe ratio & drawdown)
```

A concrete trace: Apple reports Q3 earnings. **(01)** yfinance records Apple's price path and computes its *abnormal 30-day return* (Apple's return minus the S&P 500's, so you're measuring stock-specific drift, not the market). **(02b)** The call transcript is stored, split into the scripted "prepared remarks" and the unscripted "Q&A." **(03)** Each section is scored — how positive is the tone (VADER), how much finance-negative language (Loughran-McDonald), what does a finance-trained neural net think (FinBERT), how complex is the prose (readability) — and these scores are matched to Apple's actual 30-day drift. **(04)** A LightGBM model learns, across thousands of such calls, which language patterns precede drift, and predicts drift for calls it was never trained on. **(05)** On each earnings date the model ranks all reporting companies; the strategy buys the top-predicted 20% and shorts the bottom 20%, holds 30 days, and reports the profit *after* subtracting spread and borrow costs.

**Two data-flow ideas worth naming:**

- **SQLite as a "data warehouse."** Instead of passing giant DataFrames between scripts (fragile, memory-hungry), each stage persists its output to a table. This makes the pipeline *restartable* — if `04` crashes you don't re-run the multi-hour `03`. It's the same reason factories use conveyor belts and buffers instead of handing parts worker-to-worker.
- **Point-in-time / leak-free flow.** Data always flows *forward in time*. A feature for an earnings call on date *t* may only use information from before *t*. Stage `03`/`04` go to great lengths (expanding windows, shifts) to enforce this, because the whole point is to simulate what you could actually have known on the day.

---

# 3. File by File (the files that matter)

I'll go in reading order: first the **shared library (`src/`)** grouped by role, then the **notebooks**, then the **scripts**. Trivial files (`__init__.py`, caches) are skipped. For each file: purpose, who it depends on / who depends on it, its key functions in plain English, and any new concept it introduces.

## 3A. The original pipeline library (`src/`)

### `src/config.py` — the settings file
**Purpose:** One place for constants everyone shares: where the database is, the date range, the ticker "universe" (which companies to study), benchmark tickers (SPY, XLK, IWM).
**Depended on by:** almost everything.
**Key pieces:**

- `DB_PATH`, `START_DATE`, `END_DATE`, `TICKERS` (58 curated large caps), `BENCHMARKS` — plain constants.
- `get_hf_universe(db_path)` → reads the distinct tickers actually present in the transcript table. **In:** db path. **Out:** sorted list of tickers. Returns `[]` if the table doesn't exist yet.
- `get_full_universe(db_path)` → the curated 58 **plus** every ticker that has a transcript (~690). **In:** db path. **Out:** the real working universe.

*Pattern to notice:* the "single source of truth." Notebooks used to each hard-code their own ticker lists; centralizing them here means they can never drift out of sync.

### `src/returns_calc.py` — the fast return calculator
**Purpose:** Compute forward and trailing stock returns *fast*, over tens of thousands of events at once. Shared by notebooks 01, 03, 04.
**Depends on:** numpy, pandas. **Depended on by:** 01/03/04, and `sizing.py` via similar patterns.
**The core idea — a "wide matrix + binary search":** Prices are reshaped into a big grid (`PriceMatrix`) with dates down the rows and tickers across the columns. To find "Apple's price 30 trading days after March 1st," instead of scanning a list row by row (slow), it uses numpy's `searchsorted` — a *binary search* — to jump straight to the right row, then simple array indexing. The docstring notes this turns *minutes into under a second* for 30k events.
**Key pieces:**

- `class PriceMatrix` — wraps the price grid.
  - `.from_long(prices_long)` → build one from a normal tall table. *(A `@classmethod` is an alternate constructor — a second way to make the object, here "from long-format data.")*
  - `.forward_returns(tickers, dates, n_days)` → **In:** arrays of tickers + anchor dates + a horizon. **Out:** the n-day-forward % return for each, all at once (NaN where data is missing). Entry price = first trading day on/after the date; exit = n trading days later.
  - `.trailing_returns(...)` → same but looking *backward* (used for pre-earnings momentum).
- `asof_values(value_dates, values, query_dates)` → "the last known value on or before each query date" (used for VIX-on-the-earnings-date). This "as-of join" is a classic time-series pattern: you want the most recent value that existed *at that moment*, never a future one.
- `build_event_returns(pm, events, ...)` → the orchestrator: given a table of (ticker, earnings_date), returns raw and **benchmark-adjusted** returns for 1/30/90-day windows, plus VIX and a COVID flag. "Abnormal" return = the stock's return minus SPY's over the same window — i.e., stock-specific drift with the market removed.

### `src/sentiment.py` — the text-scoring engine
**Purpose:** Turn transcript text into sentiment/linguistic numbers. The biggest `src/` file (~700 lines).
**Depends on:** nltk (VADER), pysentiment2 (LM), textstat (readability). **Depended on by:** notebook 03, `features_parallel.py`.
**Key pieces (each takes text, returns numbers):**

- `clean_transcript(raw_text)` → strips boilerplate (operator lines, legal disclaimers, headers) with surgical regex so the scorers see real content.
- `chunk_text(text, max_words=256)` → splits long text into ~256-word chunks. **Why:** neural models like FinBERT have a fixed input size; you feed them chunks and average.
- `compute_paragraph_vader` / `compute_vader_sentiment` → VADER is a rule-based sentiment scorer (positive/negative/neutral). Returns distribution stats (mean, std, percentiles) across chunks, not just one number.
- `compute_lm_sentiment` → counts words from the **Loughran-McDonald dictionary**, a lexicon built specifically for *financial* text (in finance, "liability" or "litigation" is negative in ways general sentiment tools miss).
- `compute_linguistic_features` / `compute_readability` → word counts, unique-word ratios, and readability scores (Flesch-Kincaid, Gunning Fog, etc.) — the *complexity* of the language, which can signal evasiveness.
- `compute_all_sentiment_features(text, vader, lm, prefix="")` → the orchestrator that runs all of the above and returns one flat dict. The `prefix` argument (e.g. `"qa"`) namespaces the keys so you can score each section separately (`qa_vader_mean`, `prepared_remarks_vader_mean`).
- `SECTOR_MAP` — a dict mapping each ticker to its sector (used to control for sector effects later).
- `SENTIMENT_Z_COLS` — the list of columns that get normalized into z-scores downstream.
- `compute_ticker_z_scores(...)` → **⚠️ the deliberately-kept "bad" version.** It normalizes using the *full-sample* mean/std, which peeks at the future. It's kept only as a documented contrast to the leak-free version in `features.py`. *This is a teaching artifact — a named example of the exact bug the project is built to avoid.*

### `src/transcripts_io.py` — transcript storage & structure
**Purpose:** Store/retrieve transcripts efficiently and split them into meaningful sections.
**Depends on:** zlib (compression), sqlite3. **Depended on by:** 02b, 03, `features_parallel.py`, `update_transcripts.py`.
**Key pieces:**

- `compress_text` / `decompress_text` → zlib compression. Transcripts are long; storing them compressed in SQLite saves a lot of space. (This is why the table is called `transcripts_text` and holds *blobs*, not readable strings.)
- `normalize_turns(structured_content)` → clean up the raw list of `{speaker, text}` turns.
- `find_qa_boundary(turns)` → **the interesting one.** Finds where "prepared remarks" (the scripted opening) end and "Q&A" (analysts asking unscripted questions) begins, by detecting the operator's hand-off line. **Why it matters:** the project's thesis is that *unscripted Q&A leaks more signal* than the rehearsed opening — so splitting them lets the model weigh them differently.
- `split_turns(turns)` → returns a dict with `full_text`, `prepared_remarks`, `qa`.
- `make_row(...)` / `write_batch(conn, rows)` / `open_db(...)` → build and insert compressed transcript rows into SQLite, one transaction per batch.
- `fetch_sections(conn, ticker, quarter, year)` → the read side: pull one transcript back out, decompressed, as `{full, prepared_remarks, qa}`.

### `src/features.py` — leak-free feature engineering
**Purpose:** Build model features that are *point-in-time correct*. This file is the project's conscience.
**Depends on:** `sentiment.SENTIMENT_Z_COLS`. **Depended on by:** notebook 04.
**The concept — "expanding window, shifted by one":** To make a feature that only uses the past, you compute a running statistic over all prior observations and then `.shift(1)` so the current row is excluded. Analogy: computing your "batting average coming into today's game" — it must use every game *before* today, never today's at-bats.
**Key pieces:**

- `compute_ticker_z_scores_expanding(df, prefix, min_obs=3)` → within-company z-scores using only that company's *past* calls. The leak-free replacement for the bad version in `sentiment.py`.
- `add_qoq_deltas(df, cols)` → change vs the *previous* call ("a CEO who's always sunny turns cautious — *that's* the event"). Quarter-over-quarter deltas often beat raw levels.
- `add_past_target_stats(df, target)` → each company's historical average drift (some names habitually pop or fade after earnings — a "PEAD prior").
- `combine_finbert_sections(...)` → a compute *optimization*: derive the whole-transcript FinBERT score as a word-count-weighted average of the two section scores, saving ~40% of GPU time versus re-scoring the concatenation.

### `src/features_parallel.py` — multi-core text scoring
**Purpose:** Scoring 33k transcripts single-threaded takes ~half a day; this spreads it across CPU cores.
**Depends on:** `concurrent.futures.ProcessPoolExecutor`, `sentiment`, `transcripts_io`. **Depended on by:** notebook 03.
**New concept — process pool + "spawn-safe" design:** Python can't truly run threads in parallel for CPU work (the GIL), so this uses *processes* instead. On Windows new processes are created by "spawn" — they re-import the module fresh — so the worker functions must be top-level and picklable. Each worker sets up its own SQLite connection + VADER + LM dictionary *once* (`_init_worker`), then scores keys. **Transcript text never crosses the process boundary** — workers fetch it themselves from SQLite — which keeps the inter-process chatter tiny.
**Key pieces:** `_init_worker(db_path)` (per-process setup), `_score_one(key)` (score one transcript), `compute_sentiment_features_parallel(db_path, keys, ...)` (the public entry that maps keys across workers, in input order, with a progress/ETA log).

### `src/join.py` — matching transcripts to returns
**Purpose:** Connect each transcript to the stock-return event it belongs to, by nearest date.
**Depends on:** pandas. **Depended on by:** notebook 03.
**Why it's non-trivial:** a transcript's *publication date* and the *earnings date* recorded by yfinance don't always line up exactly. `match_transcript_to_returns` finds the closest earnings date within a 30-day window; `join_sentiment_to_returns` runs that over the whole table and reports how many couldn't be matched. Extracted into its own file *specifically so the matching logic can be unit-tested* without a live database — a recurring theme.

## 3B. The newer multi-signal research layer (`src/`)

These modules explore turning the one earnings-call signal into a small **"trading-bot" stack of independent signals** — the vision is documented in `docs/trading_bot_layers.md`. Each is leak-free and independently tested.

### `src/momentum.py` — a second, price-based signal ("Layer 2")
**Purpose:** Predict drift from *price patterns* alone (classic momentum & reversal), as an independent signal to combine with the text one.
**Key pieces:**

- `compute_momentum_features(wide)` → from a wide price grid, computes momentum (12-month return skipping the last month), short-term reversal, distance from the 52-week high, volatility, market **beta**, and *vol-adjusted* variants (dividing by volatility so the model can't just chase lottery-ticket stocks). Everything uses `.shift()` to stay backward-looking.
- `build_momentum_panel(wide, volume_wide, ...)` → slices those features at monthly "rebalance" dates, attaches a **beta-adjusted target** (the return *minus beta × market return*, stripping out the mechanical market exposure), and adds **sector-neutral ranks** (ranking each feature *within* its sector to remove sector bets). **Out:** a tidy (date, ticker, features, target) table.
- `MOMENTUM_COMPOSITE_SIGNS` + `add_composite_score(panel)` → a deliberately *dumb* benchmark: rank each feature, flip signs per the finance literature, average them. Any fancy ML model must beat this simple composite to justify itself.

### `src/regime.py` — the market "gate" ("Layer 7")
**Purpose:** Decide *when* to trade at all. Outputs a daily multiplier (1.0 = full risk, 0.6 = caution, 0.25 = risk-off) from four free market indicators: VIX (fear index), S&P vs its 200-day average, credit-market stress (HYG vs LQD bond ETFs), and market breadth (% of stocks above their own 200-day average).
**Design choice worth noting:** it's **rules-based, not machine-learned** — the thresholds are standard market conventions, deliberately *not* tuned on this project's data, because "few knobs" resists overfitting. `compute_regime(db_path)` returns a daily table of the multiplier plus each component signal.

### `src/ensemble.py` — combining signals ("Layer 9")
**Purpose:** Merge the momentum signal and the earnings-call (ECA) signal into one score per stock per day.
**Key ideas:** each signal becomes a cross-sectional z-score; the ECA event score **decays linearly to zero over 30 trading days** (the drift fades); an optional regime gate can scale everything. `compute_ensemble_scores(...)` returns per-(date, ticker) `momentum_z`, `eca_z`, `regime`, and the blended `ensemble_z`.

### `src/sizing.py` — turning scores into positions ("Layer 10")
**Purpose:** Decide *how many dollars* to put on each name — the step that converts a ranking into an actual portfolio.
**Key pieces:**

- `trailing_vol(prices_wide)` → each stock's recent volatility, shifted to stay leak-free.
- `size_one_date(score, vol, sector, ...)` → full-breadth weighting: weight ∝ score ÷ volatility (so every position carries similar *risk*), then cap per-name, per-sector, and net exposure.
- `size_deciles(score, vol, sector, sector_neutral=...)` → the version that actually won in testing: pick the top/bottom slice, then *risk-balance within* those legs. A backtest note in the docstring records it beat equal-weighting (Sharpe 1.62 → 1.69, drawdown −12.7% → −10.5%).
- `book_stats(weights)` → concentration diagnostics (effective number of positions, gross/net exposure).

### `src/credibility.py` — "do executives keep their word?" (research signal)
**Purpose:** Turn weak text into a *quality* factor by tracking whether each management team's forward statements historically came true.
**Mechanism (all leak-free):** `optimism_score` collapses the LLM-extracted forward claims into one number; `grade_agreement` checks whether that optimism matched the *next quarter's actual EPS surprise* (ground truth from market fundamentals, so it's not circular with the target); `build_credibility` accumulates a per-company track record using only *prior* calls; the output feature `cred_weighted_optimism` = credibility × current optimism ("trust the teams that have earned it"). `load_credibility_features(db_path)` wires it to the database. Depends on LLM-scored data existing (see `llm_features.py`).

### `src/llm_features.py` — scoring calls with a large language model
**Purpose:** Use an LLM to read the Q&A and extract *semantic* features that word-counting misses (did guidance go up or down? did management dodge questions?).
**New concept — the "provider" abstraction:** rather than hard-code one AI vendor, it defines a base `LLMProvider` class and three interchangeable subclasses — `GeminiProvider` (Google), `OllamaProvider` (a model running locally on your own machine), `AnthropicProvider` (Claude) — each implementing one `complete(prompt)` method. `get_provider()` picks one based on which API key is configured. This is the **Strategy pattern**: define one interface, swap the implementation freely. It also includes strict JSON parsing/validation (`parse_llm_json`, `validate_scores`) and retry/back-off logic so a flaky free API doesn't crash the run.

## 3C. The notebooks (pipeline stages)

Each notebook is thin: load tables → call `src/` → write tables → chart. In order:

- **`01_pull_prices`** — downloads prices for ~690 tickers + benchmarks via yfinance, fetches earnings dates, computes SPY-adjusted abnormal returns (via `returns_calc`), pulls VIX. Writes `prices`, `earnings`, `returns`, `vix`.
- **`02b_ingest_hf_transcripts`** — one-time load of 33k transcripts from HuggingFace, split into prepared-remarks/Q&A, stored compressed. Writes `transcripts_text`. (Skipped by the pipeline once done. `02_transcripts` is a legacy SEC-scraper fallback, largely unused.)
- **`03_sentiment`** — the heavy one. *Stage A:* VADER/LM/readability across all sections, parallelized over CPU cores (`features_parallel`). *Stage B:* FinBERT on the Q&A, on the GPU. Both are **checkpointed** (results saved incrementally to `vaderlm_scores` / `finbert_scores`) so a crash resumes instead of restarting — a lesson learned after a power outage killed a 13-hour run. Joins to returns (`join`). Writes `sentiment_features`.
- **`04_modeling`** — builds leak-free features (`features`), selects the best ones, tunes a LightGBM model with **Optuna** on the first 70% of time (the "tune" region), then evaluates on the untouched last 30% via **walk-forward** validation (repeatedly train-on-past / test-on-next-block, marching forward in time). Explains predictions with **SHAP** (which features drove each prediction). Writes `model_predictions` and saves the model `.pkl`.
- **`05_backtest`** — ranks stocks by prediction each earnings date, goes long the top / short the bottom, holds 30 days, subtracts costs, and reports Sharpe ratio, drawdown, and sensitivity to the cutoff. Charts equity curves.
- **`06_momentum`, `07_regime`** — research notebooks that exercise the newer `momentum`/`regime` modules and validate them.

## 3D. The scripts (side tools)

- **`analyze_results.py`** — reads `model_predictions` and prints a fast verdict: information coefficient (IC) + t-stat, decile spread, accuracy, top SHAP features. Your "did it work?" one-liner.
- **`run_llm_extraction.py`** — batch-scores transcripts through `llm_features` into an `llm_qa_scores` table. Checkpointed/resumable, with a cost pre-flight estimate and paid-tier concurrency settings.
- **`snapshot_daily.py`** — records today's analyst estimates + short interest per ticker into `snapshots.db`. Run daily (locally or by the GitHub Action) to accumulate a *revision history* that free data sources don't provide retroactively.
- **`update_transcripts.py`** — pulls newer earnings-call transcripts (post-2025) via `defeatbeta-api`, incrementally and resume-safely. This is how genuinely out-of-sample test data gets added.
- **`monitor_progress.py`** — a live progress bar you run in a second terminal to watch the slow FinBERT stage.

---

# 4. The Glue — How the Pieces Talk

## The one thing everything revolves around: `data/market.db`

If you remember only one thing: **the SQLite database is the integration point.** No notebook calls another notebook. No stage imports another stage's variables. They communicate *entirely* by reading and writing tables in `market.db`. This is a deliberate architecture — sometimes called a **"medallion" or staged data pipeline** — and it buys three things: (1) restartability (each stage's output is durable), (2) inspectability (you can open any table and look), and (3) loose coupling (you can rerun `04` a hundred times without touching `03`).

The tables, and who writes/reads them:

| Table | Written by | Read by | Holds |
|---|---|---|---|
| `prices` | 01 | 03, 04, 05, momentum, regime, sizing | daily OHLCV per ticker |
| `earnings` | 01 | 04, credibility | earnings dates + EPS surprises |
| `returns` | 01 | 03 | post-earnings abnormal returns (the *target*) |
| `vix` | 01 | 03, 04, regime | daily VIX (market fear) |
| `transcripts_text` | 02b | 03, credibility, update_transcripts | compressed call transcripts |
| `sentiment_features` | 03 | 04 | all NLP scores, joined to returns |
| `model_predictions` | 04 | 05, analyze_results, ensemble | predicted vs actual drift + SHAP |
| `vaderlm_scores`, `finbert_scores` | 03 (checkpoints) | 03 (resume) | partial NLP results, for crash recovery |
| `llm_qa_scores` | run_llm_extraction | credibility, 04 | LLM-extracted semantic features |
| `snapshots.db → analyst/estimate` | snapshot_daily | (future signal) | daily analyst estimates over time |

## How the layers connect

The **notebooks are orchestration; `src/` is the library.** A notebook cell looks like: `df = pd.read_sql("SELECT * FROM transcripts_text", conn)` → `features = compute_sentiment_features_parallel(db, keys)` → `df.to_sql("sentiment_features", conn)`. All the *thinking* is in the `src/` function; the notebook just wires inputs to outputs and draws a chart. This is why the tests can cover the hard logic without ever launching a notebook — the logic isn't *in* the notebooks.

The **newer research modules stack on each other** in a clean dependency chain (this mirrors the "layers" vision in `docs/trading_bot_layers.md`):

```
   momentum.py ─┐
                ├──▶ ensemble.py ──▶ sizing.py ──▶ (a tradeable book)
   (ECA model) ─┘         ▲
                          │
   regime.py ─────────────┘ (optional gate)

   llm_features.py ──▶ (llm_qa_scores) ──▶ credibility.py ──▶ (a quality signal)
```

`ensemble` imports `momentum` and `regime`; `sizing` consumes `ensemble`'s scores; `credibility` consumes `llm_features`' output. Each layer is independently testable and answers one question: *what to predict* (momentum/ECA/credibility), *when to trade* (regime), *how to combine* (ensemble), *how much to bet* (sizing).

## The key data structures everything passes around

Three shapes recur constantly; recognizing them makes the code readable:

1. **The "long" event table** — one row per (ticker, date) with columns for features and outcomes. This is the normal `pandas` tidy format; `sentiment_features`, `model_predictions`, and the momentum panel are all this shape. Machine-learning code wants this.
2. **The "wide" price matrix** — dates as rows, tickers as columns, prices in the cells (`PriceMatrix`, and the `wide` DataFrames in momentum/sizing). Fast vectorized math wants this. Code constantly `pivot`s between long and wide.
3. **The flat feature dict** — one Python dict of `{feature_name: number}` for a single transcript, produced by `compute_all_sentiment_features`. Many of these get stacked into a DataFrame.

Plus one important convention: the **cross-sectional z-score / rank**. Repeatedly, a raw score is converted to "how this stock ranks *versus other stocks on the same date*." Trading is relative — you don't need to predict Apple's return, only whether Apple will beat Microsoft this month — so nearly every signal is ranked or z-scored *within each date* before use.

## What each external library actually does for us

| Library | What it does here, concretely |
|---|---|
| **pandas / numpy** | Every table and every array. numpy's `searchsorted` powers the fast return lookups. |
| **yfinance** | Free Yahoo Finance client — downloads historical prices, earnings dates, VIX. Our window into the market. |
| **huggingface_hub / datasets** | Download the 33k-transcript corpus (a public dataset hosted on HuggingFace). |
| **nltk** | Provides VADER, a rule-based sentiment scorer, plus tokenizers. |
| **pysentiment2** | The Loughran-McDonald *financial* sentiment dictionary — finance-aware word counting. |
| **transformers + torch + torch-directml** | Run **FinBERT**, a BERT neural net fine-tuned on financial text. `torch-directml` is the shim that lets PyTorch use an **AMD** GPU on Windows (normally PyTorch assumes NVIDIA/CUDA). |
| **textstat** | Readability formulas (Flesch-Kincaid, Gunning Fog, ...). |
| **lightgbm** | The prediction model: a gradient-boosted decision-tree ensemble. Fast, strong on tabular data, the workhorse of quant ML. |
| **optuna** | Automatically searches for good model hyperparameters (an intelligent replacement for hand-tuning / grid search). |
| **scikit-learn** | Metrics, scalers, time-series cross-validation splits. |
| **shap** | Explains *why* the model predicted what it did — attributes each prediction to its input features. |
| **sqlite3** | The built-in database. One file, no server. |
| **nbclient / nbformat** | Let `run_pipeline.py` execute notebooks programmatically and save their outputs. |
| **pytest** | Runs the unit-test suite. |
| **optional: defeatbeta-api** | Free source for *newer* transcripts (post-2025), used by `update_transcripts.py` to build out-of-sample test data. Warning in `requirements.txt`: installing it disturbs the numpy/pandas versions. |

---

# 5. Cheat Sheet — The One-Page Mental Model

## The whole system in five sentences
1. It predicts a stock's **30-day post-earnings drift** from the **language of the earnings call** plus numeric signals.
2. The pipeline is **five notebook stages** (`01 prices → 03 sentiment → 04 model → 05 backtest`, with `02b` a one-time transcript load) run in order by **`run_pipeline.py`**.
3. Stages never talk directly — they **pass data through tables in one SQLite file, `data/market.db`**.
4. All real logic lives in the **`src/` library** (so it's unit-tested); notebooks are thin orchestration.
5. The obsession throughout is **not fooling yourself**: leak-free (point-in-time) features, an honest tune/test split, walk-forward evaluation, and cost-aware backtests.

## Key files, ranked by importance to understanding
| Rank | File | Why it matters |
|---|---|---|
| 1 | `run_pipeline.py` | The entry point and the map of stage order. |
| 2 | `src/config.py` | Constants + the ticker universe everything shares. |
| 3 | `src/returns_calc.py` | How returns/targets are computed (the wide-matrix trick). |
| 4 | `src/sentiment.py` | How text becomes numbers (VADER/LM/readability). |
| 5 | `src/features.py` | The leak-free feature discipline — the project's conscience. |
| 6 | `notebooks/04_modeling` | Where the model is trained and honestly evaluated. |
| 7 | `notebooks/05_backtest` | Where predictions become a cost-aware strategy. |
| 8 | `src/{momentum,regime,ensemble,sizing,credibility}` | The newer multi-signal research stack. |

## Key terms (define-once glossary)
- **PEAD (post-earnings-announcement drift):** stocks keep drifting the direction of an earnings surprise for weeks — the pattern being predicted.
- **Abnormal return:** a stock's return minus the market's (SPY) — stock-specific movement.
- **Leak-free / point-in-time:** a feature only uses information available *before* its event. Enforced with "expanding window + `.shift(1)`."
- **Walk-forward validation:** repeatedly train on the past, test on the next block, march forward — how time-series models are honestly tested (you can't shuffle time).
- **Cross-sectional z-score / rank:** a stock's score *relative to other stocks on the same date* — because trading is relative.
- **Information Coefficient (IC):** the correlation between predictions and actual returns; the standard "is this signal real?" metric.
- **Sharpe ratio:** return per unit of risk — the headline score for a strategy.
- **Long/short book:** buy the best-ranked names, short the worst; profit from the *spread*, market-neutral.
- **FinBERT:** a BERT neural net fine-tuned on financial text.
- **VADER / Loughran-McDonald:** rule-based sentiment scorers (general / finance-specific).
- **LightGBM:** gradient-boosted decision trees — the prediction model.
- **Optuna:** automated hyperparameter search.
- **SHAP:** per-prediction feature attribution ("why did it predict that?").
- **Strategy pattern (in `llm_features.py`):** one interface (`LLMProvider`), swappable implementations (Gemini/Ollama/Anthropic).
- **Checkpointing (in notebook 03):** save partial results as you go so a crash resumes, not restarts.

## How to trace any feature end-to-end
Pick a question and follow the data:

- **"Where does the model's *target* come from?"** → `01` computes abnormal returns via `src/returns_calc.build_event_returns` → stored in the `returns` table → joined into `sentiment_features` in `03` (via `src/join`) → read as `y` in `04`.
- **"How is a transcript turned into features?"** → stored compressed by `02b` (`src/transcripts_io`) → fetched + cleaned + scored in `03` (`src/features_parallel` → `src/sentiment`) → leak-free-transformed in `04` (`src/features`).
- **"How does a prediction become a trade?"** → `04` writes `model_predictions` → `05` ranks by prediction, longs top / shorts bottom, subtracts costs, reports Sharpe.
- **"How would I add a new signal?"** → write a leak-free feature-producing module in `src/` (model it on `momentum.py`), add a test in `tests/`, blend it in `ensemble.py`, size it in `sizing.py`.

## How to run things (quick reference)
```bash
# full modeling pipeline (01 → 03 → 04 → 05)
.venv/Scripts/python.exe run_pipeline.py
# preview the plan only
.venv/Scripts/python.exe run_pipeline.py --dry-run
# re-model + backtest only (skip slow data stages)
.venv/Scripts/python.exe run_pipeline.py --only 04 05
# quick model verdict
.venv/Scripts/python.exe scripts/analyze_results.py
# run the tests
.venv/Scripts/python.exe -m pytest tests -q
```

## If you're unsure what something does
Three reliable moves: (1) read its **docstring** — this codebase documents heavily, including *why*; (2) read its **test** in `tests/` — a test shows the function's inputs and expected outputs concretely; (3) grep for where it's **called** (`grep -rn "function_name" notebooks src scripts`) to see it used in context. And `docs/PROJECT_JOURNAL.md` narrates the actual experiments and results if you want the research story behind the code.

