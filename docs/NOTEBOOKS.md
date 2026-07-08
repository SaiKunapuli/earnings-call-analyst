# ECA Project — Notebook Documentation

## Overview

> **Canonical reference is `README.md`.** This file has extra cell-level notes;
> where they disagree, trust the README. Some sections below describe an earlier
> 14-ticker / XLK-adjusted version of the pipeline and are kept only for detail.

This project performs **Earnings Call Analysis (ECA)** across a ~685-company
universe (curated 57 core names + every ticker with an ingested HuggingFace
transcript). The pipeline:

1. Pulls daily prices, earnings history, and VIX from Yahoo Finance (notebook 01)
2. Ingests 33k+ real earnings-call transcripts from HuggingFace, speaker-segmented
   (notebook 02b; legacy SEC EDGAR scraper is 02)
3. Runs multi-tier sentiment analysis (VADER, Loughran-McDonald, FinBERT) per
   transcript section — full / prepared remarks / Q&A (notebook 03)
4. Trains an Optuna-tuned LightGBM model on **SPY-adjusted 30-day post-earnings
   drift** with leak-free features and an honest tune/eval split (notebook 04)
5. Backtests a cost-aware long/short strategy on the predictions (notebook 05)

All data is persisted in a SQLite database (`data/market.db`).

---

## Data Flow

```
01_pull_prices.ipynb         02_transcripts.ipynb          03_sentiment.ipynb
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ yfinance API     │         │ SEC EDGAR API    │         │ Transcript .txt   │
│       │          │         │       │          │         │ files on disk     │
│       ▼          │         │       ▼          │         │       │           │
│ prices table     │         │ transcripts      │         │       ▼           │
│ earnings table   │         │ metadata table   │         │ Sentiment scores  │
│ returns table    │         │                  │         │ (VADER, LM,       │
│ (with VIX +      │         │                  │         │  FinBERT)         │
│  XLK-adjusted    │         │                  │         │       │           │
│  abnormal ret)   │──┐      │                  │──┐      │       ▼           │
└─────────────────┘  │      └──────────────────┘  │      │ sentiment_features│
                     └────────────┬───────────────┘      │ table             │
                                  │                      └──────────────────┘
                                  ▼
                         market.db (SQLite)
```

---

## Notebook 01: `01_pull_prices.ipynb`

**Purpose:** Pull daily OHLCV data for 14 tech stocks + 2 benchmarks from Yahoo Finance, collect earnings dates, compute forward returns (XLK-adjusted), and persist everything to SQLite.

### Constants & Configuration (Cells 1–2)

| Variable | Description |
|---|---|
| `TICKERS` | 14 tech stocks: MSFT, GOOGL, META, AMZN, NVDA, AMD, INTC, CRM, ORCL, ADBE, CSCO, IBM, NOW, NFLX |
| `BENCHMARKS` | SPY (S&P 500 ETF) and XLK (Technology Sector ETF) |
| `START_DATE` / `END_DATE` | 2019-01-01 to 2026-06-25 |

### Data Pipeline

1. **Price download (Cell 3):** `yfinance.download()` pulls all 16 symbols with `auto_adjust=True`. Returns a multi-level DataFrame (Price × Ticker) with ~1,879 trading days × 80 columns (5 price fields × 16 tickers).

2. **Reshape to long format (Cell 7):** `data.stack(level='Ticker')` then `reset_index()` converts the wide multi-level DataFrame into a tidy long-format table with columns: `date`, `ticker`, `close`, `high`, `low`, `open`, `volume`.

3. **Save to SQLite (Cell 8):** Long data written to `prices` table in `../data/market.db`.

4. **Earnings dates (Cells 9–10):** Uses `yfinance.Ticker.get_earnings_dates(limit=60)` for each ticker, collecting EPS estimates, reported EPS, and surprise %. Cleaned and saved to `earnings` table.

5. **VIX pull (Cell 13):** Pulls CBOE VIX index (^VIX) over the same date range. Stored separately from stock prices since VIX is a volatility index, not a stock.

### Notable Functions

#### `get_return(ticker, dd, n_days)` (Cell 14)
```
def get_return(ticker, dd, n_days):
```
Computes the **forward n-day return** for `ticker` starting at the first trading day on or after `dd`. Returns `None` if the window runs off the data. Uses `prices_wide` (pivoted by ticker) for fast lookups.

**Logic:**
- Find the first trading day ≥ `dd` → `start_idx`
- Offset forward by `n_days` in the index → `end_idx`
- Return `end_price / start_price - 1`

### Key Computations (Cell 14)

- **`return_1d`, `return_30d`, `return_90d`:** Forward returns at 1, 30, and 90 trading-day horizons
- **`abnormal_1d`, `abnormal_30d`, `abnormal_90d`:** Returns **minus XLK** (tech sector ETF) — strips out common sector beta
- **`vix_close`:** VIX close on or before the earnings date (captures pre-earnings risk regime)
- **`is_covid`:** Boolean for earnings dates within 2020-02-20 to 2021-06-30 (COVID volatility window)

### Output Tables

| Table | Rows | Description |
|---|---|---|
| `prices` | 30,064 | Long-format daily OHLCV for 16 symbols |
| `earnings` | 434 | EPS estimates & reported values per ticker |
| `returns` | 420 | Forward returns + abnormal returns + VIX + COVID flag per earnings event |

### Sanity Checks (Cells 16–17)
- Summary statistics of abnormal returns at each horizon
- VIX distribution on earnings dates
- COVID-window vs non-COVID cohort comparison
- Per-ticker mean abnormal 1d returns

---

## Notebook 02: `02_transcripts.ipynb`

**Purpose:** Scrape earnings call transcripts from SEC EDGAR 8-K filings using the `edgar` Python library, clean the HTML, and store metadata in SQLite.

### Configuration (Cell 2)

| Variable | Value | Description |
|---|---|---|
| `TICKERS` | 14 tickers | Same universe as notebook 01 |
| `TRANSCRIPTS_DIR` | `../data/transcripts` | Local storage for `.txt` files |
| `DB_PATH` | `../data/market.db` | Shared SQLite database |
| `SEARCH_DAYS_BEFORE` | 1 | Days before earnings date to search 8-K |
| `SEARCH_DAYS_AFTER` | 3 | Days after earnings date to search 8-K |
| `DELAY_SECONDS` | 0.5 | Polite delay between SEC requests |

### SEC EDGAR Identity
Uses `edgar.set_identity()` with a user agent string identifying the tool as educational use.

### Notable Functions

#### `_get_8k_filings(ticker)` (Cell 3)
```
def _get_8k_filings(ticker):
```
Fetches all 8-K filings for a ticker using `edgar.Company.get_filings(form='8-K')`. Results are cached in `_FILINGS_CACHE` dict to avoid redundant SEC API calls.

#### `find_8k_exhibit(ticker, earnings_date)` (Cell 3)
```
def find_8k_exhibit(ticker, earnings_date):
```
Given a ticker and earnings announcement date, searches the 8-K filings window (±1 day before, +3 days after) and returns the URL of Exhibit 99 (the earnings release / transcript). Falls back to the first exhibit if no Exhibit 99 is found.

#### `_extract_text_from_html(html)` (Cell 3)
```
def _extract_text_from_html(html):
```
Uses BeautifulSoup to extract readable text from SEC filing HTML. Tries SEC-specific CSS selectors (`div.body`, `div[class*="content"]`) first, falls back to full body text extraction. Removes scripts, styles, and navigation elements.

#### `scrape_transcript(ticker, quarter, year, earnings_date, overwrite=False)` (Cell 3)
```
def scrape_transcript(ticker, quarter, year, earnings_date, overwrite=False):
```
Orchestrates the full transcript fetch:
1. Checks if transcript already exists on disk (skips if so, unless `overwrite=True`)
2. Finds the 8-K exhibit via `find_8k_exhibit()`
3. Downloads the exhibit HTML from SEC
4. Extracts clean text via `_extract_text_from_html()`
5. Writes to `data/transcripts/{TICKER}/{quarter}{year}.txt`
6. Returns a metadata dict (status, word_count, url, etc.)

### Fiscal Calendar Mapping (Cell 8)

Not all 14 companies follow a standard calendar-year fiscal year. The `FISCAL_RULES` dict maps each ticker to its fiscal quarter convention:

| Ticker | FY End | Notes |
|---|---|---|
| MSFT | Jun 30 | Q1 = Oct–Dec (offset +1) |
| NVDA | Last Sun of Jan | Q1–Q3 offset +1 |
| CRM | Jan 31 | Same as NVDA |
| ORCL | May 31 | Complex mapping with split Q2 |
| ADBE | Last Fri of Nov | Q4 split Dec/Jan |
| CSCO | Last Sat of Jul | Q1 Oct–Dec offset +1 |
| GOOGL, META, AMZN, IBM, NOW, NFLX, AMD, INTC | Dec 31 | Calendar year (Q4 = Jan–Mar, offset -1) |

#### `get_fiscal_quarter_year(ticker, announcement_date)` (Cell 8)
```
def get_fiscal_quarter_year(ticker, announcement_date):
```
Maps an earnings announcement date to a (fiscal_quarter_label, fiscal_year) tuple using the `FISCAL_RULES` mapping.

### Scale Fetch (Cell 8)

Automatically generates fetch targets from the `earnings` table by matching each `(ticker, earnings_date)` to a fiscal quarter. Runs the full fetch loop over ~420 targets with a 0.5s polite delay (~10 minutes runtime).

### Output

| Table | Rows | Description |
|---|---|---|
| `transcripts` | 359 | Metadata: ticker, quarter, year, file_path, word_count, status, etc. |

Transcript text files stored at: `data/transcripts/{TICKER}/{quarter}{year}.txt`

---

## Notebook 03: `03_sentiment.ipynb`

**Purpose:** Run three tiers of sentiment analysis on earnings call transcripts, compute readability/linguistic features, join with returns data, and persist everything for downstream modeling.

### Imports & Setup (Cell 1)

Libraries used:
- **NLP:** `nltk` (VADER, sentence tokenizer), `pysentiment2` (Loughran-McDonald financial dictionary), `textstat` (readability)
- **Transformers:** `transformers` (FinBERT via `ProsusAI/finbert`)
- **Viz:** `matplotlib`, `seaborn`, `plotly`
- **Data:** `pandas`, `numpy`, `sqlite3`

### Notable Functions

#### `clean_transcript(raw_text)` (Cell 3)
```
def clean_transcript(raw_text):
```
Strips Motley Fool boilerplate headers/footers (copyright notices, "Should you invest" CTAs, etc.), removes URL-only lines, and normalizes whitespace.

#### `chunk_text(text, max_words=256)` (Cell 3)
```
def chunk_text(text, max_words=256):
```
Splits long transcript text into chunks of ~256 words at sentence boundaries using `nltk.sent_tokenize()`. Required because FinBERT has a 512-token context window.

### Sentiment Tiers

| Tier | Method | Library | Output Features |
|---|---|---|---|
| 1 | VADER | `nltk.sentiment.vader` | `vader_compound`, `vader_pos`, `vader_neg`, `vader_neu` |
| 2 | Loughran-McDonald | `pysentiment2` | `lm_positive`, `lm_negative`, `lm_uncertainty`, `lm_litigious`, `lm_constraining`, `lm_strong_modal`, `lm_weak_modal`, `lm_net` (derived) |
| 3 | FinBERT | `transformers` (ProsusAI/finbert) | `finbert_positive`, `finbert_negative`, `finbert_neutral`, `finbert_net`, `finbert_label` |

### Readability Features (Cell 6)

| Feature | Description |
|---|---|
| `flesch_reading_ease` | Higher = easier to read (0–100 scale) |
| `flesch_kincaid_grade` | US grade level equivalent |
| `gunning_fog` | Years of education needed to understand |
| `smog_index` | Simple Measure of Gobbledygook |
| `automated_readability_index` | Characters/word + words/sentence based |
| `dale_chall_score` | Based on familiar word list |
| `unique_word_ratio` | Unique words / total words |
| `avg_sentence_length` | Average words per sentence |

### Return Joining (Cell 7)

Matches each transcript to the closest earnings event in the `returns` table by:
1. Finding the `pub_date` (transcript publication date) for each (ticker, quarter, year)
2. Computing day-difference to each earnings_date in returns
3. Selecting the closest match within a 30-day window
4. Carrying forward: `return_1d`, `return_30d`, `return_90d`, `abnormal_1d`, `abnormal_30d`, `abnormal_90d`, `vix_close`, `is_covid`

### Output

| Table | Description |
|---|---|
| `sentiment_features` | All sentiment + readability + returns features per transcript |

### Visualizations (Cell 9)

1. **Correlation matrix** heatmap: sentiment scores × abnormal returns × VIX × readability
2. **Scatter plot:** VADER compound vs abnormal 1-day return (color-coded by COVID window)
3. **Horizontal bar charts:** mean sentiment and mean abnormal return by ticker
4. **Scatter plot:** sentiment vs readability (Flesch Reading Ease)

---

## Database Schema

All tables live in `data/market.db` (SQLite):

```sql
-- Daily price data (long format)
prices(date DATE, ticker TEXT, close REAL, high REAL, low REAL, open REAL, volume INTEGER)

-- Earnings announcement data
earnings(earnings_date DATE, eps_estimate REAL, reported_eps REAL, eps_surprise_pct REAL, ticker TEXT)

-- Forward returns + abnormal returns per earnings event
returns(ticker TEXT, earnings_date DATE, return_1d REAL, return_30d REAL, return_90d REAL,
        abnormal_1d REAL, abnormal_30d REAL, abnormal_90d REAL, vix_close REAL, is_covid INTEGER)

-- Transcript metadata
transcripts(ticker TEXT, quarter TEXT, year INTEGER, pub_date DATE, url TEXT,
            file_path TEXT, word_count INTEGER, status INTEGER, attempt INTEGER, scrape_time TEXT)

-- Sentiment features merged with returns
sentiment_features(ticker TEXT, quarter TEXT, year INTEGER,
                   vader_compound REAL, vader_pos REAL, vader_neg REAL, vader_neu REAL,
                   lm_positive INTEGER, lm_negative INTEGER, lm_uncertainty INTEGER, ...,
                   finbert_positive REAL, finbert_negative REAL, finbert_neutral REAL, ...,
                   flesch_reading_ease REAL, flesch_kincaid_grade REAL, ...,
                   return_1d REAL, return_30d REAL, return_90d REAL,
                   abnormal_1d REAL, abnormal_30d REAL, abnormal_90d REAL,
                   vix_close REAL, is_covid INTEGER)
```

---

## Suggested Next Notebooks

- **`04_modeling.ipynb`** — LightGBM regression predicting `abnormal_30d` from sentiment + VIX + readability features, with SHAP explainability
- **`05_portfolio.ipynb`** — Backtest a long/short strategy based on sentiment surprise (actual sentiment vs expected by VIX regime)

---

## Environment

- Python 3.11.9
- Virtual environment: `.venv/`
- Key packages: `yfinance`, `pandas`, `numpy`, `sqlite3`, `edgar`, `requests`, `beautifulsoup4`, `nltk`, `pysentiment2`, `textstat`, `transformers`, `torch`, `matplotlib`, `seaborn`, `plotly`
