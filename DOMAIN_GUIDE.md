# ECA — Domain Guide: The Finance & Quant Concepts

> A companion to the [CODEBASE_GUIDE](CODEBASE_GUIDE.md). That guide covers
> *where* the code lives and how the pipeline runs. This guide covers the
> **finance and quantitative concepts** the code implements, and *why* each
> piece matters to someone trading or investing on earnings.

**Audience:** Technical, no finance background. Every term is defined.

---

## Table of Contents

1. [The Earnings Call as a Market Event](#1-the-earnings-call-as-a-market-event)
2. [Financial Statement Literacy](#2-financial-statement-literacy)
3. [The Expectations Machinery](#3-the-expectations-machinery)
4. [The Quant / NLP Layer](#4-the-quant--nlp-layer)
5. [Market Reaction & Tradability](#5-market-reaction--tradability)
6. [Evaluation & Honesty](#6-evaluation--honesty)
7. [The Full Picture: How It All Connects](#7-the-full-picture-how-it-all-connects)
8. [Code Entry Points (Quick Reference)](#8-code-entry-points-quick-reference)

---

## 1. The Earnings Call as a Market Event

### What it is

Every three months, every publicly traded US company reports its quarterly
results. This is mandatory — the SEC requires it. The company publishes a
press release with the headline numbers (revenue, profit, etc.) and then
holds a **conference call**, usually within an hour, where executives discuss
the results and take questions from Wall Street analysts.

### The two-part structure (and why it matters)

The call has two distinct halves:

**Prepared remarks (scripted):** The CEO and CFO read a written statement.
It's been lawyered, polished, and rehearsed. Every word is chosen. The
prepared remarks are designed to *frame* the numbers, not reveal new ones.

**Q&A (unscripted):** Sell-side analysts — people whose job is to cover that
specific company for a bank — ask questions. Management has to answer live.
This is where **information leakage** happens: a CEO can't perfectly control
their tone for an hour straight. They might dodge a question, sound
defensive, accidentally reveal optimism or pessimism, or get challenged by an
analyst who's done their homework.

> **Sell-side analyst:** someone who works at a bank (Goldman Sachs, Morgan
> Stanley, etc.), studies a handful of companies full-time, builds financial
> models predicting their future earnings, and publishes "Buy/Hold/Sell"
> ratings. They're on this call to update their models.

The project's thesis — encoded in `src/transcripts_io.py`'s
`find_qa_boundary()` and `split_turns()` — is that **the Q&A leaks more
predictive signal than the prepared remarks**. The scripted opening is spin;
the unscripted answers are closer to truth. This is why the code splits each
transcript into sections and scores them separately.

```python
# From src/transcripts_io.py
def find_qa_boundary(turns):  # detects "Questions and Answers:" or "we'll now take questions"
def split_turns(turns):       # returns {prepared_text, qa_text, full_text} + evasiveness stats
```

The evasiveness stats are particularly clever: `hedge_per_1k` counts hedging
phrases per 1,000 words of executive answers ("I think," "we'll have to see,"
"it depends," "hard to say"). A CEO who hedges a lot in Q&A may know
something the numbers don't show yet.

### The reporting calendar

Most US companies follow the calendar year, reporting:

| Quarter | Period | Typical call window |
|---------|--------|---------------------|
| Q1 | Jan–Mar | Mid-April to early May |
| Q2 | Apr–Jun | Mid-July to early August |
| Q3 | Jul–Sep | Mid-October to early November |
| Q4 | Oct–Dec | Late January to late February |

Some companies (MSFT, NVDA, WMT, and others) have non-calendar **fiscal
years** — e.g., Microsoft's fiscal year ends June 30, so its "Q1" is
Jul–Sep. The repo handles this when matching transcripts to earnings dates.

Earnings season — the ~6-week window when most companies report — is when
most of the project's events cluster. The rest of the quarter is quiet.

### Why this moves prices

Public companies are valued based on their *future* earnings, not their past
ones. When new information shifts expectations about that future, the stock
price adjusts. The press release gives the hard numbers; the call gives
*context* — and sometimes context that contradicts the numbers. If revenue
beat estimates but the CEO sounds terrified about next quarter, the stock
might actually drop. This is the gap this project exploits: **what management
says can predict returns even after the numbers are known.**

---

## 2. Financial Statement Literacy

> You don't need to be an accountant. Here are the only concepts that show up
> in the code.

### Revenue, Earnings, EPS

| Term | Meaning | Also called |
|------|---------|-------------|
| **Revenue** | How much money the company brought in from selling stuff | Sales, top line |
| **Earnings** | What's left after all costs, taxes, etc. | Net income, profit, bottom line |
| **EPS** | Earnings ÷ number of shares outstanding | Earnings per share |

**Concrete example:** Apple reports $124.3B in revenue with $36.3B in net
income. With ~15.2B shares outstanding, that's $36.3B ÷ 15.2B = **$2.39 EPS**.

EPS is per-share so you can compare companies of different sizes. A $1B
company and a $3T company both report EPS. The headlines say "Apple beat EPS
estimates by $0.12" — that $0.12 is the per-share surprise.

### Guidance

In the call, management often gives **guidance** — their own forecast for
next quarter's revenue or EPS. This is optional but most companies do it.
Guidance is more forward-looking than the reported numbers and often moves the
stock more than the earnings beat/miss itself.

The `src/llm_features.py` module is designed to extract this:

| LLM Feature | Scale | Meaning |
|-------------|-------|---------|
| `guidance_direction` | −1 to +1 | Lowered / maintained / raised |
| `guidance_confidence` | 0 to 2 | Hedged / neutral / firmly confident |
| `demand_outlook` | −2 to +2 | Forward-looking demand (orders, pipeline, customers) |
| `margin_outlook` | −2 to +2 | Forward-looking cost, pricing, margin commentary |

### GAAP vs. Non-GAAP

- **GAAP** (Generally Accepted Accounting Principles): The official,
  standardized rules. Every public company *must* report GAAP numbers. These
  are audited.
- **Non-GAAP:** "Adjusted" numbers that strip out items the company says
  aren't representative. Stock-based compensation, restructuring costs,
  acquisition charges, etc.

Management *always* prefers non-GAAP because the numbers look better.
Sometimes that's legitimate (a one-time factory fire write-down genuinely
doesn't represent ongoing operations). Sometimes it's creative accounting
(systematically excluding real costs like stock compensation to inflate
profits).

The project's `src/credibility.py` is a direct response to this ambiguity.
Instead of trusting what management says, it asks: **"When THIS management
team was optimistic in the past, did the actual numbers (EPS surprise — the
real, market-verified outcome) back them up?"** It builds a per-company track
record.

### What "quality of earnings" means

If a company beats EPS but did it by cutting R&D, selling assets, or changing
accounting assumptions rather than actually selling more stuff, the "quality"
of that beat is low — it's not sustainable. This repo doesn't directly
measure this, but the **credibility module** approximates it: a management
team whose optimistic calls are consistently followed by negative EPS
surprises is low-quality.

---

## 3. The Expectations Machinery

### The core idea

> **The stock doesn't react to the raw number; it reacts to the surprise.**

If Apple reports $2.10 EPS, is that good or bad?

| Scenario | Consensus estimate | Surprise | Market reaction |
|----------|-------------------|----------|-----------------|
| A | $2.00 | +$0.10 (+5%) | Positive |
| B | $2.50 | −$0.40 (−16%) | Negative |

Same reported number. Radically different outcomes. The **surprise** — actual
minus expected — is what moves the stock.

### Who sets expectations?

**Sell-side analysts** publish estimates for each company they cover. The
average of all estimates is the **consensus** — that's what the news reports
as "expected." There are also **whisper numbers** (unofficial, word-of-mouth
expectations, not in this repo).

**The options market** prices in an "implied move" — how much the options
market expects the stock to move on earnings day. Options traders pay
attention to the implied move, not the consensus EPS.

### Standardizing the surprise (SUE)

Raw surprise (actual − expected) isn't comparable across companies:

> A $0.10 beat for a company earning $0.50/share is huge (20%).
> The same $0.10 beat for a company earning $10/share is trivial (1%).

Quants standardize it:

**SUE (Standardized Unexpected Earnings):**

$$\text{SUE} = \frac{\text{Actual EPS} - \text{Expected EPS}}{\sigma(\text{past surprises})}$$

where $\sigma$ is the standard deviation of that company's historical
surprises. A SUE of +2.0 means "this is a 2-standard-deviation beat — only
happens ~2.5% of the time."

This repo's `src/returns_calc.py` has an `eps_surprise_pct` column (the
surprise as a percentage), and `src/features.py` builds features from it,
including interactions with sentiment:

```
ix_full_finbert_net_x_surprise
```

Translation: "Does positive sentiment, amplified by a positive EPS surprise,
predict even stronger drift?" (It's the interaction effect — the product of
two signals — not just adding them.)

### Estimate revisions: the signal you can't download

The `scripts/snapshot_daily.py` script records analyst estimates *every day*
into `snapshots.db`. Why?

Free sources (Yahoo Finance, etc.) only expose *today's* estimates. If you
want to know what estimates were 7, 30, 60, or 90 days ago — to measure
estimate *revisions*, which are themselves a powerful trading signal — you
have to record them yourself.

```python
# From snapshot_daily.py — what gets recorded per ticker, per day
"eps_current REAL, eps_7d_ago REAL, eps_30d_ago REAL, eps_60d_ago REAL, eps_90d_ago REAL"
```

A stock whose estimates have been steadily revised *upward* for weeks is a
different animal from one that just beat a stale, unchanged estimate. Every
day you don't run the snapshot, that history is lost forever.

---

## 4. The Quant / NLP Layer

> How words become numbers. Four approaches, stacked from shallow to deep.

### Layer 1: VADER — the quick dictionary check

**What it is:** A rule-based sentiment tool. It has a dictionary of ~7,500
words, each scored for positivity/negativity. It reads text, looks up words,
applies some grammar rules (negation flips: "not good" ≠ "good"), and outputs
a **compound score** from −1 (very negative) to +1 (very positive).

**Why it's weak for finance:** VADER was built for tweets and movie reviews.
It doesn't know that "liability" is bad in finance or that "challenging
quarter" is management-speak for "we missed badly."

**How this repo uses it:** `src/sentiment.py` → `compute_paragraph_vader()`.
But it doesn't just take one score for the whole transcript — that would
produce a meaningless ~1.0 for a 50,000-character document. Instead, it
chunks the text into sentence groups of 5, scores each chunk, and returns the
*distribution*:

| Feature | What it captures |
|---------|-----------------|
| `vader_mean` | Average tone across all chunks |
| `vader_std` | Consistency — low std = uniformly positive (or negative) |
| `vader_min` | The worst moment in the call |
| `vader_p10`, `vader_p90` | The 10th and 90th percentile chunks |
| `vader_pct_neg` | What fraction of chunks were negative |

A CEO who's positive for 90% of the call but sharply negative in a few
specific moments — that pattern might matter, and the distribution stats
capture it.

### Layer 2: Loughran-McDonald — the finance dictionary

**What it is:** A word list built by two accounting professors (Tim
Loughran and Bill McDonald) who manually classified 80,000+ words from
10-K filings into categories relevant to financial text.

| Category | Example words | Why it matters |
|----------|--------------|----------------|
| **Negative** | bankruptcy, defect, misconduct | Finance-specific negatives |
| **Positive** | excellent, profitable, strengthen | Finance-specific positives |
| **Uncertainty** | may, might, could, approximately | Hedging language |
| **Litigious** | lawsuit, plaintiff, testimony | Legal risk signals |
| **Constraining** | must, required, compelled | Obligations/restrictions |
| **Strong Modal** | always, definitely, never | Overconfidence? |
| **Weak Modal** | possibly, perhaps, sometimes | Evasiveness? |

**Why it matters:** In general English "liability" is neutral; in LM it's
negative. "Depreciation" isn't negative in LM — it's just an accounting
entry. This is the project recognizing that generic sentiment tools miss
finance-specific meaning.

```python
# From src/sentiment.py
def compute_lm_sentiment(text, lm):
    # Returns: lm_positive, lm_negative, lm_uncertainty, lm_litigious,
    #          lm_constraining, lm_strong_modal, lm_weak_modal, lm_net
```

A transcript with high `lm_uncertainty` and high `lm_litigious` is one where
management keeps using legalistic, hedged language — classic evasiveness
signals.

### Layer 3: FinBERT — the finance-trained neural net

**What it is:** BERT is a transformer model (the same architecture behind
ChatGPT, but much smaller — ~110M parameters vs. hundreds of billions).
FinBERT is BERT fine-tuned on financial text: SEC filings, earnings reports,
analyst reports. It doesn't just count words; it understands context.

**Example:** "Revenue declined but margins expanded" — VADER might see
"declined" and score negative; FinBERT understands the "but" and the
positive second clause.

**The constraint:** FinBERT can only process ~512 tokens at a time (roughly
256 words). For a 50,000-word transcript, the code must:

1. Split into ~256-word chunks (`chunk_text()` in `sentiment.py`)
2. Run FinBERT on each chunk on the GPU
3. Average the probabilities across chunks

This is the slowest part of the pipeline — hours on a GPU. The
`combine_finbert_sections()` function in `src/features.py` is a clever
optimization: instead of scoring the full transcript separately, it computes
the full-transcript FinBERT score as a word-count-weighted average of the
prepared-remarks and Q&A scores, saving ~40% of GPU time.

**Running on an AMD GPU:** Most ML assumes NVIDIA/CUDA. This project uses
`torch-directml` to run on an AMD GPU on Windows — a practical constraint.

### Layer 4: LLM Extraction — the semantic layer

**What it is:** Word-counting and even FinBERT can't detect certain things.
Did management *raise* or *lower* guidance? Did they dodge a question? Was
the tone rosier than what the numbers would justify? These require
*reasoning*. `src/llm_features.py` sends the Q&A section to a language model
(Google Gemini free tier, Anthropic Claude, or a local Ollama model) with a
structured prompt:

```python
# From src/llm_features.py
FEATURE_SPEC = {
    "guidance_direction":  (-1, 1, True),   # lowered/maintained/raised; null if not discussed
    "guidance_confidence": (0, 2, True),     # hedged/neutral/firmly confident
    "demand_outlook":      (-2, 2, False),   # forward demand commentary
    "margin_outlook":      (-2, 2, False),   # cost/pricing/margin outlook
    "n_questions_dodged":  (0, 30, False),   # deflected/non-responsive answers
    "tone_numbers_gap":    (-2, 2, False),   # tone rosier/gloomier than figures
    "unexpected_negative": (0, 1, False),    # negative surfaced in Q&A only
    "analyst_pushback":    (0, 2, False),    # skepticism / repeated challenges
}
```

The **Strategy pattern** (`LLMProvider` base class → `GeminiProvider`,
`OllamaProvider`, `AnthropicProvider`) lets you swap the AI backend without
changing the extraction logic. Free Gemini tier is the default; Claude for
higher quality; Ollama for fully local/free.

### What makes a feature predictive vs. noise?

The project's answer, revealed in the ablation studies, is brutal: **raw
sentiment alone is nearly useless.** Every CEO sounds positive on every call.
What actually matters:

| Transformation | Code location | What it captures |
|---------------|--------------|------------------|
| **QoQ deltas** | `src/features.py` → `add_qoq_deltas()` | Did the tone *shift* vs. last quarter? |
| **Cross-sectional z-scores** | Per-date ranking | Was this call more positive than *other* companies this quarter? |
| **Within-ticker z-scores** | `compute_ticker_z_scores_expanding()` | Was this call unusual for *this company's history*? |
| **Q&A minus prepared** | `qa_delta_*` features | The gap between scripted and unscripted tone |

**Concrete example:** A CEO who's always at VADER 0.7 suddenly drops to
0.3 — *that's* the signal. The code captures this as a quarter-over-quarter
delta:

```python
# From src/features.py
df[f"{c}_qoq"] = tmp[c] - g[c].shift(1)  # change vs. the previous call
```

### The "deliberately bad" version (a teaching artifact)

`src/sentiment.py` contains `compute_ticker_z_scores()` — a function that
normalizes sentiment scores using each ticker's *full-sample* mean and
standard deviation. This peeks at the future (it includes later calls in the
mean/std). It's kept in the code as a named, documented example of the
exact bug the project is built to avoid. The leak-free replacement is
`compute_ticker_z_scores_expanding()` in `src/features.py`.

---

## 5. Market Reaction & Tradability

### PEAD — the pattern this whole thing bets on

**Post-Earnings-Announcement Drift:** After a company reports earnings, its
stock tends to keep drifting in the same direction as the surprise for weeks.

| Scenario | Day 1 reaction | Next 30 trading days |
|----------|---------------|---------------------|
| Big EPS beat | +3% gap up | Tends to drift another +1-2% |
| Big EPS miss | −4% gap down | Tends to drift another −1-2% |

**Why does this exist?** Three main theories:

1. **Underreaction:** Investors don't fully absorb the news immediately;
   they adjust slowly over weeks.
2. **Attention constraints:** Some investors don't notice earnings day-of;
   they discover the news later.
3. **Short-sale constraints:** If a stock is overpriced after bad news, it's
   hard/expensive to short it, so the correction is slow.

The academic literature has documented PEAD since 1968 (Ball & Brown). It's
one of the most robust anomalies in finance — it has survived 50+ years,
countless papers, and should have been arbitraged away by now. The fact that
it *hasn't* makes it interesting.

### Why 30 days?

The repo's primary target is `abnormal_30d` — the stock's return minus SPY's
return over the 30 trading days (~6 calendar weeks) post-earnings.

| Window | What it captures | Why (not) use it |
|--------|-----------------|-------------------|
| `abnormal_1d` | The instant gap at open next day | Mostly arbitraged away; tiny edge |
| `abnormal_30d` | The PEAD window | Sweet spot — signal exists, not fully decayed |
| `abnormal_90d` | Longer drift | Noisier — more non-earnings events intervene |

### Abnormal vs. raw returns

The target isn't "did Apple go up?" It's "did Apple go up *more than the
market*?" The **abnormal return** strips out the market's movement:

$$\text{abnormal} = \text{stock return} - \text{SPY return}$$

If the whole market rallied 5% and Apple went up 5%, that's zero abnormal
return — Apple just rode the wave. Only the *excess* return is attributable
to Apple's own news.

```python
# From src/returns_calc.py — the vectorized computation
r = pm.forward_returns(tickers, dates, 30)    # stock's 30d forward return
b = pm.forward_returns(benchmark, dates, 30)   # SPY's 30d forward return
out["abnormal_30d"] = r - b                     # the excess
```

### Beta-adjusted returns (the momentum module's refinement)

The simple abnormal return (`stock − SPY`) still has a problem: in a bull
market, high-beta stocks naturally outperform SPY.

**Beta ($\beta$):** How much a stock moves when the market moves 1%.

| Beta | Meaning |
|------|---------|
| 1.0 | Moves exactly with the market |
| 1.5 | Moves 1.5% for every 1% market move (more volatile) |
| 0.5 | Moves 0.5% for every 1% market move (more stable) |

A model could "predict" drift simply by buying high-beta stocks in a bull
market. That's not skill — it's leverage. The momentum module's target strips
this out:

```python
# From src/momentum.py
target = raw_target - beta * (SPY_fwd_return)
```

This is the **idiosyncratic return** — the return you *can't* explain by the
stock's beta to the market. It's a much harder, much more honest target.

### Implied volatility & IV crush

> Not directly in the code, but essential context for why earnings events
> are interesting.

Before earnings, options are expensive because everyone expects a big move.
The **implied volatility** (the market's expectation of future volatility,
derived from options prices) spikes. After the announcement, uncertainty
resolves — and options prices crash. This is **IV crush**.

The repo doesn't trade options, but the concept explains the opportunity:
earnings create a massive release of uncertainty → certainty. The market
*misprices* the speed and magnitude of that resolution. PEAD is one
manifestation.

### How a prediction becomes a position

The backtest (notebook `05`) works like this:

1. On each earnings date, rank all reporting companies by the model's
   predicted `abnormal_30d`
2. Go **long** the top 20% (buy them, betting they'll drift up)
3. Go **short** the bottom 20% (borrow and sell them, betting they'll drift
   down)
4. Hold for 30 trading days
5. Subtract costs, report the net return

**Long/short** means you're market-neutral: if the whole market crashes, you
lose on your longs but *make* on your shorts. You're isolating the
stock-picking skill, not betting on the market's direction.

**Long-only** means you only buy; you're always exposed to the market. The
backtest reports both variants.

### The costs (why they matter)

| Cost | What it is | Rough magnitude |
|------|-----------|-----------------|
| **Spread** | You buy at the ask (higher), sell at the bid (lower) | ~0.05–0.10% per trade |
| **Slippage** | Your order moves the price against you | ~0.05–0.20% per trade |
| **Short borrow** | You pay to borrow shares to short | 0.25–5% annualized |
| **Capacity** | Can you actually trade this at scale? | Unlimited for large caps |

The repo is honest: every result is reported **net of costs**, and the cost
assumptions are deliberately conservative. A strategy that makes 10% gross
but costs 12% to trade is a money-loser — the backtest makes this visible.

### Position sizing (`src/sizing.py`)

The `src/sizing.py` module transforms a ranking into actual dollar weights.
The naive approach — equal-weight the top and bottom deciles — lets one
volatile name dominate risk. The smarter approach:

| Step | What it does | Why |
|------|-------------|-----|
| **Vol-scale** | weight ∝ score ÷ trailing volatility | Equal risk contribution per name |
| **Demean** | subtract cross-sectional mean | Roughly dollar-neutral (longs ≈ shorts) |
| **Cap per name** | max 3% per stock | No single stock blows up the book |
| **Cap per sector** | max 25% net per sector | No sector-concentration bets |
| **Cap net exposure** | limit net long/short imbalance | Stay market-neutral |

The backtest docstring in `size_deciles()` reports: Sharpe 1.62 → 1.69, max
drawdown −12.7% → −10.5% just from better sizing. **Sizing is where backtest
Sharpe is kept or lost.**

---

## 6. Evaluation & Honesty

> This is where the project distinguishes itself. Most quant projects produce
> beautiful backtests that are worthless because they cheated without
> realizing it. This one builds its entire architecture around *not*
> cheating.

### The three deadly sins of quant research

#### Sin 1: Lookahead bias (data leakage)

**What it is:** Accidentally letting the model see future information.

**Example:** Computing a z-score for an event using the *full sample* mean and
standard deviation — that includes future events the model couldn't have
known about.

**The repo's fix:** `src/features.py` → `compute_ticker_z_scores_expanding()`:

```python
# The "expanding window, shifted by one" — the project's core discipline
def _expanding_past(grouped, fn):
    return grouped.transform(lambda s: getattr(s.expanding(), fn)().shift(1))
```

**Analogy:** Computing your "batting average coming into today's game." You
use every game *before* today, never today's at-bats. The `.shift(1)` is what
excludes the current event.

`src/sentiment.py` keeps the **deliberately bad version**
(`compute_ticker_z_scores()`) as a documented teaching artifact — a named
example of the exact bug the project exists to avoid.

#### Sin 2: Overfitting (tuning to noise)

**What it is:** Running 1,000 variations of your model, picking the one that
looks best on the backtest, and calling that your strategy. You've fitted
random noise.

**The repo's fix:** An honest tune/eval split:

- **Tune region** (first 70% of events by date): Optuna searches for good
  hyperparameters here. Feature selection happens here.
- **Eval region** (last 30%): The model is tested here **once**, and that
  result is reported. Tuning *never* sees this data.

**Walk-forward validation:** Instead of random train/test splits (which would
shuffle time — nonsense for a time-series problem), the model is repeatedly
trained on all past data and tested on the next block, marching forward in
time:

```
Train on 2016-2019 → Test on 2020 → Train on 2016-2020 → Test on 2021 → ...
```

This simulates what you could actually have done in real time.

#### Sin 3: Ignoring costs

A strategy that makes 10% gross but costs 12% to trade is a money-loser. This
repo reports everything **net of costs**. The backtest subtracts estimated
spread, slippage, and borrow costs. The equity curves show both gross and net
lines — if they diverge, costs matter.

### The truly-honest test: `scripts/run_oos_test.py`

Even the eval region has a problem: it was "consulted repeatedly during model
design." So there's a further test:

1. Take the **frozen model** (trained on data through May 15, 2025)
2. Apply it to earnings calls **after** that date
3. Compare the out-of-sample metrics to the in-sample reference

```python
# From run_oos_test.py
CUTOFF = pd.Timestamp("2025-05-15")     # last training earnings date
IS_IC, IS_T, IS_SPREAD = 0.052, 1.88, 217  # in-sample reference
```

If the OOS IC is similar to the in-sample IC → the signal is real.
If it vanishes → the in-sample result was an artifact. **This is the honest
finding.**

### The key metrics

| Metric | What it measures | Good value | Red flag |
|--------|-----------------|------------|----------|
| **IC** (Information Coefficient) | Rank correlation: predicted vs. actual returns | ~0.03–0.08 | < 0.01 or negative |
| **IC t-stat** | How statistically reliable the IC is | > 2.0 (conventionally significant) | < 1.5 |
| **Sharpe ratio** | Annualized return ÷ annualized volatility | > 1.0 (good), > 2.0 (exceptional) | > 3.0 (suspicious — check for cheating) |
| **Max drawdown** | Worst peak-to-trough decline | −10% to −20% for equity-like strategies | > −30% |
| **Decile spread** | Return difference: top 10% minus bottom 10% | Should be positive and monotonic | Flat or inverted |
| **Win rate** | % of months/trades with positive return | > 55% | < 50% |

### What the repo actually found

From the code, the in-sample reference:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| IC | 0.052 | Weak but positive signal |
| IC t-stat | 1.88 | Borderline significant (conventional threshold: 2.0) |
| Decile spread | +217 bps | Top decile outperforms bottom by ~2.2% per event |

The `run_oos_test.py` script exists precisely to determine whether even that
weak signal survives out-of-sample. This is honest quant research: the signal
is marginal, the costs are real, and the project doesn't oversell.

---

## 7. The Full Picture: How It All Connects

### The analyst's thought process, mapped to the pipeline

```
EARNINGS CALL HAPPENS
        │
        ▼
[01_pull_prices]  What did the stock do?  →  SPY-adjusted 30-day drift
        │
        ▼
[02b_ingest]  Split the call into prepared remarks vs Q&A
        │                                    │
        ▼                                    ▼
[03_sentiment]  What did they SAY?           [llm_features]  What did they MEAN?
  VADER → tone per chunk                      guidance raised or lowered?
  LM → finance-specific word counts           questions dodged?
  FinBERT → contextual sentiment              analyst pushback?
  Readability → evasiveness                   tone vs. numbers gap?
        │                                    │
        ▼                                    ▼
[04_modeling]  Which of ~100 features actually predict drift?
  Leak-free z-scores (only past history)
  Quarter-over-quarter changes (did tone shift?)
  EPS surprise × sentiment interactions
  LightGBM trained on 70% of time, tested on untouched 30%
        │
        ▼
[05_backtest]  If we traded on this, would we make money?
  Long top 20%, short bottom 20%
  Hold 30 days, subtract spread + borrow costs
  Report Sharpe, drawdown, decile spread
        │
        ▼
[run_oos_test]  But wait — did we overfit the eval region?
  Apply frozen model to genuinely new calls
  If IC survives → the signal is probably real
  If IC vanishes → we fooled ourselves (and learned something)
```

### The newer multi-signal research stack

```
   momentum.py ─┐
                ├──▶ ensemble.py ──▶ sizing.py ──▶ (a tradeable book)
   (ECA model) ─┘         ▲
                          │
   regime.py ─────────────┘ (optional gate: should we trade at all?)

   llm_features.py ──▶ credibility.py ──▶ (a quality-of-management signal)
```

| Module | Question it answers |
|--------|-------------------|
| `momentum.py` | What do price patterns predict? (Layer 2) |
| `regime.py` | Should we trade right now, or sit out? (Layer 7) |
| `ensemble.py` | How do we combine multiple signals? (Layer 9) |
| `sizing.py` | How many dollars go on each name? (Layer 10) |
| `credibility.py` | Does this management team keep its word? |
| `llm_features.py` | What did the Q&A *really* say? |

### The three data shapes that recur

| Shape | Format | Where used | Purpose |
|-------|--------|-----------|---------|
| **Long table** | One row per (ticker, date) | `sentiment_features`, `model_predictions` | ML input (tidy data) |
| **Wide matrix** | Dates × tickers | `PriceMatrix`, momentum panels | Fast vectorized math |
| **Flat feature dict** | `{name: number}` | Per-transcript sentiment output | Stack into DataFrames |

### Cross-sectional ranking: the unifying convention

Nearly every signal is converted to "how this stock ranks *versus other
stocks on the same date*." Trading is relative — you don't need to predict
Apple's return; you only need to predict whether Apple will beat Microsoft
this month. So scores are ranked or z-scored **within each date** before use.

---

## 8. Code Entry Points (Quick Reference)

### By topic area

| Topic | Files to read |
|-------|--------------|
| Earnings call structure | `src/transcripts_io.py`, notebook `02b` |
| Financial statements & EPS | `src/returns_calc.py`, `scripts/snapshot_daily.py`, `src/credibility.py` |
| Expectations & surprises | `scripts/snapshot_daily.py`, `src/returns_calc.py`, `src/features.py` |
| NLP / sentiment scoring | `src/sentiment.py`, `src/features_parallel.py`, notebook `03` |
| LLM semantic extraction | `src/llm_features.py`, `scripts/run_llm_extraction.py` |
| PEAD & market reaction | `src/returns_calc.py`, `src/momentum.py`, notebook `05` |
| Position sizing | `src/sizing.py` |
| Model training & evaluation | `src/features.py`, notebook `04` |
| Honesty / out-of-sample test | `scripts/run_oos_test.py`, `scripts/analyze_results.py` |
| Regime / when to trade | `src/regime.py` |
| Signal combination | `src/ensemble.py` |
| Management credibility | `src/credibility.py` |

### Key files ranked by importance to understanding the domain

| Rank | File | What it teaches |
|------|------|----------------|
| 1 | `src/returns_calc.py` | How returns are computed, what "abnormal" means, the VIX |
| 2 | `src/sentiment.py` | How text becomes numbers (VADER, LM, readability) |
| 3 | `src/features.py` | The leak-free discipline — the project's core idea |
| 4 | `notebooks/04_modeling.ipynb` | How the model is trained and honestly evaluated |
| 5 | `notebooks/05_backtest.ipynb` | How predictions become a cost-aware strategy |
| 6 | `src/momentum.py` | Beta-adjusted returns, momentum features, sector-neutral ranks |
| 7 | `src/sizing.py` | How rankings become dollar weights |
| 8 | `src/llm_features.py` | Semantic extraction of guidance, dodging, pushback |
| 9 | `src/credibility.py` | Tracking whether management's words match reality |
| 10 | `scripts/run_oos_test.py` | The acid test: does the signal survive on unseen data? |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Abnormal return** | Stock return minus the market (SPY) return — stock-specific movement |
| **Ask / Bid** | The price you can buy at (ask, higher) vs. sell at (bid, lower). The gap is the **spread** |
| **Beta ($\beta$)** | How much a stock moves when the market moves 1%. $\beta$ = 1.5 means 1.5% per 1% market move |
| **Consensus** | The average of all analyst estimates for a metric (EPS, revenue, etc.) |
| **Cross-sectional z-score** | A stock's score relative to other stocks *on the same date*. Trading is relative |
| **Decile spread** | The return difference between the top 10% and bottom 10% of predictions |
| **EPS (Earnings Per Share)** | Net income ÷ shares outstanding |
| **FinBERT** | A BERT neural net fine-tuned on financial text (SEC filings, earnings reports) |
| **GAAP** | Generally Accepted Accounting Principles — the official, audited rules |
| **Guidance** | Management's own forecast for future quarters, given during the earnings call |
| **Hedge ratio** | Hedging phrases per 1,000 words — an evasiveness signal |
| **IC (Information Coefficient)** | Rank correlation between predictions and actual returns — "is this signal real?" |
| **Idiosyncratic return** | The return not explained by market beta — stock-specific, not market-driven |
| **Implied volatility (IV)** | The market's expectation of future volatility, derived from options prices |
| **IV crush** | The sharp drop in options prices after earnings when uncertainty resolves |
| **Leak-free / point-in-time** | A feature computed using only information available *before* the event |
| **LightGBM** | Gradient-boosted decision trees — the workhorse prediction model |
| **Long** | Buying a stock, betting it goes up |
| **Long/short** | Buying the best, shorting the worst — market-neutral |
| **Loughran-McDonald (LM)** | A finance-specific word dictionary (positive, negative, uncertainty, litigious, etc.) |
| **Non-GAAP** | "Adjusted" earnings that exclude items management considers non-representative |
| **Optuna** | Automated hyperparameter search — finds good model settings without hand-tuning |
| **PEAD (Post-Earnings-Announcement Drift)** | Stocks keep drifting in the surprise direction for weeks after earnings |
| **QoQ (Quarter-over-Quarter)** | Change vs. the same company's previous quarter |
| **Regime** | The market's current state (risk-on, caution, risk-off) |
| **Sell-side analyst** | Works at a bank, covers specific companies, publishes estimates and ratings |
| **SHAP** | Explains *why* the model predicted what it did — attributes each prediction to input features |
| **Sharpe ratio** | Annualized return ÷ annualized volatility — the headline strategy score |
| **Short** | Borrowing shares and selling them, betting the price will drop (you buy back later) |
| **Slippage** | The difference between the expected trade price and the actual execution price |
| **Spread** | The gap between the bid (sell) and ask (buy) price — a transaction cost |
| **SUE (Standardized Unexpected Earnings)** | (Actual − Expected) ÷ std(past surprises) — how unusual this beat/miss is |
| **VADER** | A rule-based sentiment tool (Valence Aware Dictionary and sEntiment Reasoner) |
| **VIX** | The "fear index" — measures expected market volatility from S&P 500 options prices |
| **Walk-forward validation** | Repeatedly train on the past, test on the next block — honest time-series testing |
| **Whisper number** | Unofficial, word-of-mouth earnings expectation (not published) |

---

*Generated from the ECA codebase. Companion to [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md).*
