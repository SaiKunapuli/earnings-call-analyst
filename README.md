# Earnings Call Analyst

A quantitative pipeline that scrapes SEC EDGAR 8-K earnings call transcripts,
computes NLP sentiment scores (VADER, Loughran-McDonald, FinBERT),
and models the relationship between earnings-call language and
subsequent abnormal stock returns using LightGBM + SHAP.

## Pipeline Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  01_pull_prices     │────▶│  02_transcripts     │────▶│  03_sentiment     │────▶│  04_modeling      │
│                     │     │                     │     │                   │     │                   │
│  • yfinance prices  │     │  • SEC EDGAR 8-K    │     │  • VADER sentiment│     │  • LightGBM model │
│  • Earnings dates   │     │    exhibit scraping │     │  • LM dictionary  │     │  • Time-series CV │
│  • VIX pull         │     │  • Fiscal calendar  │     │  • FinBERT (GPU)  │     │  • SHAP analysis  │
│  • XLK-adj returns  │     │    mapping          │     │  • Readability    │     │  • Predictions    │
│                     │     │                     │     │  • Join w/returns │     │                   │
└─────────┬───────────┘     └─────────┬───────────┘     └────────┬──────────┘     └──────────────────┘
          │                           │                          │
          ▼                           ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              data/market.db (SQLite)                            │
│  tables: prices | earnings | returns | transcripts | sentiment_features         │
│          model_predictions                                                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Notebooks

| # | Notebook | Purpose | Key Output |
|---|----------|---------|------------|
| 01 | `01_pull_prices.ipynb` | Pull prices for 14 tech tickers + SPY/XLK, fetch earnings dates, compute XLK-adjusted abnormal returns, pull VIX | `prices`, `earnings`, `returns` tables in SQLite |
| 02 | `02_transcripts.ipynb` | Scrape earnings call transcripts from SEC EDGAR 8-K filings using the `edgartools` library | `transcripts` table + `.txt` files in `data/transcripts/` |
| 03 | `03_sentiment.ipynb` | Compute VADER, Loughran-McDonald, FinBERT sentiment + readability metrics; join with returns | `sentiment_features` table in SQLite |
| 04 | `04_modeling.ipynb` | Train LightGBM model predicting abnormal 30d returns; SHAP explainability; persist model | `model_predictions` table + `models/lgbm_abnormal_30d.pkl` |

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Flow

### 01 → SQLite tables: `prices`, `earnings`, `returns`

- Downloads daily OHLCV for 14 tech stocks + SPY + XLK (2019–present) via `yfinance`
- Fetches historical earnings dates and EPS surprises
- Computes **XLK-adjusted abnormal returns** (raw return minus tech-sector benchmark) at 1d/30d/90d horizons
- Pulls VIX close as a market-regime feature
- Flags COVID-era observations (2020-02-20 to 2021-06-30)

### 02 → SQLite table: `transcripts` + text files

- Uses SEC EDGAR to find 8-K filings near each earnings date
- Extracts exhibit text via BeautifulSoup
- Maps announcement dates to fiscal quarters using per-ticker fiscal-calendar rules
- Stores cleaned transcripts as `.txt` files in `data/transcripts/{TICKER}/`

### 03 → SQLite table: `sentiment_features`

- **VADER**: Compound sentiment + pos/neg/neu breakdown
- **Loughran-McDonald**: Financial-domain dictionary counts (positive, negative, uncertainty, litigious, constraining, modal)
- **FinBERT**: Transformer-based financial sentiment via `ProsusAI/finbert` (accelerated on AMD GPU via DirectML)
- **Readability**: Flesch-Kincaid, Gunning Fog, SMOG, Dale-Chall, automated readability, unique word ratio, avg sentence length
- Joins to returns data by matching transcript pub_date to nearest earnings_date (≤30 day diff)

### 04 → SQLite table: `model_predictions` + `models/`

- **Target**: `abnormal_30d` (XLK-adjusted 30-trading-day return)
- **Features**: All sentiment, readability, and market-regime features + ticker (categorical)
- **Model**: LightGBM regressor with time-series cross-validation (5-fold `TimeSeriesSplit`)
- **Explainability**: SHAP summary, dependence, and bar plots
- **Output**: Pickled model + predictions with per-feature SHAP values

## Tech Stack

- **Data**: `yfinance`, `edgartools`, `beautifulsoup4`
- **NLP**: `nltk` (VADER), `pysentiment2` (LM dictionary), `transformers` (FinBERT)
- **GPU**: `torch-directml` for AMD GPU acceleration
- **Modeling**: `lightgbm`, `scikit-learn`, `shap`
- **Viz**: `matplotlib`, `seaborn`, `plotly`
- **Storage**: SQLite (`sqlite3`), `joblib` for model serialization

## Project Structure

```
ECA/
├── notebooks/
│   ├── 01_pull_prices.ipynb
│   ├── 02_transcripts.ipynb
│   ├── 03_sentiment.ipynb
│   └── 04_modeling.ipynb
├── src/
│   └── .gitkeep
├── data/                         (gitignored)
│   ├── market.db                 (SQLite database)
│   └── transcripts/              (scraped .txt files)
│       ├── MSFT/
│       ├── GOOGL/
│       └── ...
├── models/                       (gitignored)
│   └── lgbm_abnormal_30d.pkl
├── docs/                         (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Ticker Universe

14 large-cap US tech companies with sector benchmark adjustment:

| Ticker | Company | Fiscal Year End |
|--------|---------|----------------|
| MSFT | Microsoft | Jun 30 |
| GOOGL | Alphabet | Dec 31 |
| META | Meta | Dec 31 |
| AMZN | Amazon | Dec 31 |
| NVDA | NVIDIA | Last Sun of Jan |
| AMD | AMD | Dec 31 |
| INTC | Intel | Dec 31 |
| CRM | Salesforce | Jan 31 |
| ORCL | Oracle | May 31 |
| ADBE | Adobe | Last Fri of Nov |
| CSCO | Cisco | Last Sat of Jul |
| IBM | IBM | Dec 31 |
| NOW | ServiceNow | Dec 31 |
| NFLX | Netflix | Dec 31 |

**Benchmarks**: SPY (S&P 500), XLK (Technology Select Sector — used for abnormal return adjustment)
