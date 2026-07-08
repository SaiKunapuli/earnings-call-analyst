# ECA Project Journal — Earnings Call Analyst

> **HOW TO USE THIS DOCUMENT (instructions for an AI reading it):** This is
> the complete, self-contained memory of the ECA project as of 2026-07-07.
> Reading it top to bottom gives you: the goals (§1), the current state and
> headline results (§2), the architecture (§3), every experiment with exact
> numbers (§4), all known caveats and holes — read these before praising any
> result (§5), lessons learned (§6), the roadmap and open decisions (§7),
> and a glossary (§8). All performance numbers come from an untouched
> out-of-sample eval region unless stated otherwise. When asked to write
> about, extend, or critique this project, treat §5 as load-bearing: every
> claim in §2/§4 is bounded by it.

---

## 1. GOALS

- **Project (near-term):** a learning/portfolio project proving the author
  can build an honest quant research pipeline: earnings-call transcripts →
  NLP features → LightGBM cross-sectional model → cost-aware long/short
  backtest, with leak-free features and genuinely out-of-sample evaluation.
- **Ambition (long-term):** ECA is "Layer 1" (the first signal) of a future
  multi-signal trading bot. The layered architecture is specced in
  `docs/trading_bot_layers.md`: 2–4 weak uncorrelated signals (transcripts,
  momentum/reversal, fundamentals, positioning) + a macro/regime gate + a
  decision stack (ensemble → sizing → paper-trading live loop).
- **Research question answered so far:** *does what management says on the
  earnings call predict the stock's 30-day drift?* Answer: the surrounding
  numbers (surprise, per-name drift persistence, vol regime) do — the
  word-count language features mostly do not (see §4.3). Semantic LLM
  extraction is the live hypothesis for making the text itself pay (§4.6).

## 2. CURRENT STATE (2026-07-07)

**Repo**: committed and pushed to GitHub (`SaiKunapuli/earnings-call-analyst`).
`docs/` and `tests/` remain gitignored per user preference.

**Headline result (the FINAL ECA model, currently saved in `model_predictions`
and `models/`):** target = raw `abnormal_30d` (PEAD drift), 17,663 feature
rows / 582 tickers / 2016–2025; eval = untouched last 30% (5,130 events,
Nov 2022 – May 2025, 31 months):

| Metric | Value |
|---|---|
| Rank-model monthly IC | **+0.066, t = 2.34** (77% of months positive) — statistically significant |
| Level-model IC | +0.052, t = 1.88 |
| Decile spread (D10−D1) | +217 bps per event |
| Long/short Sharpe, net of costs | **0.53** (0.58 gross; 10 bps round-trip + 50 bps/yr borrow) |
| Short leg | 65.2% hit rate, +2.2%/event (the edge lives here) |
| Long leg | 50.2% hit rate (coin flip on sign; wins by relative magnitude) |
| Direction classifier | DEAD — eval AUC 0.45–0.47 (below coin flip) despite tune AUC ~0.55 |

**Everything works end-to-end:** `run_pipeline.py` runs notebooks
01→03→04→05; 03 is checkpointed/resumable (survives power loss / Ctrl+C);
132 pytest tests pass; the LLM extraction skeleton is built, smoke-tested
(18/20 transcripts scored successfully on Gemini free tier), and ready
for a paid pilot when the user decides.

**Momentum signal (Layer 2) exists as `06_momentum.ipynb` with a reusable
`src/momentum.py` module.** Old baseline: IC +0.026 (t=1.88) — below the
t≥2 gate. Three improvements applied (vol-adjusted features, sector-neutral
ranks, beta-adjusted target per thinker analysis) but not yet re-run for
the new results.

**Daily snapshot automation live:** GitHub Actions workflow runs weekdays at
noon UTC, recording analyst estimates + short interest via
`scripts/snapshot_daily.py` and committing `data/snapshots.db` back to the
repo. Universe: 57 curated tickers (market.db not in repo, so the HF-extended
universe is unavailable in CI).

**Repo layout (post-reorg):** `notebooks/` (01, 02 legacy, 02b, 03, 04, 05,
06) · `src/` (config, sentiment, transcripts_io, returns_calc, features,
features_parallel, join, llm_features, momentum) · `scripts/`
(analyze_results, monitor_progress, run_llm_extraction, snapshot_daily —
tracked in git) · `.github/workflows/` (daily_snapshot.yml) · `tests/`
(132 tests; gitignored) · `docs/` (this journal, llm_qa_plan.md,
trading_bot_layers.md; gitignored) · `graphs/legacy/` (pre-rebuild charts;
gitignored) · `data/market.db` (SQLite; gitignored) · `data/snapshots.db`
(SQLite; committed for CI persistence) · `.env` (API keys; gitignored).

## 3. ARCHITECTURE

### Pipeline stages
| stage | notebook | function |
|---|---|---|
| 01 | 01_pull_prices | yfinance prices (698 tickers), SPY-adjusted abnormal returns (1d/30d/90d), VIX, EPS surprises → SQLite |
| 02b | 02b_ingest_hf_transcripts | one-time ingest of 33,361 transcripts from a HuggingFace dataset (zlib-compressed, sectioned into prepared remarks / Q&A); incremental updater via free defeatbeta-api |
| 03 | 03_sentiment | Stage A: VADER + Loughran-McDonald + readability + evasiveness on all sections, parallel on 24 cores → `vaderlm_scores` checkpoint. Stage B: FinBERT on Q&A only (AMD GPU, torch-directml) → `finbert_scores` checkpoint (flush every 100, resume-by-key). Joins pub-date-anchored forward returns → `sentiment_features` |
| 04 | 04_modeling | feature engineering (expanding z-scores, QoQ deltas, PEAD priors, SUE, surprise streak, interactions) → MI selection → Optuna LightGBM → honest walk-forward eval → SHAP → `model_predictions` |
| 05 | 05_backtest | decile long/short, transaction costs, holding-period-aware Sharpe, SPY/EW benchmarks, leg decomposition, cutoff sensitivity |

### The evaluation protocol (the project's backbone — reuse for every future signal)
- **Tune region:** first 70% of events by date. Optuna (30 trials) and MI
  feature selection see ONLY this.
- **Eval region:** last 30% (5,130 events, 31 months), predicted by
  expanding walk-forward (3 folds, each trained only on strictly earlier
  events). All reported numbers come from here.
- **Gate metrics:** monthly Spearman IC + t-stat (t ≥ 2 = usable), decile
  spread in raw bps, net-of-cost Sharpe. RMSE/R²/sign-accuracy are reported
  but near-meaningless on noise-dominated returns.
- **Leak-free feature rules:** expanding within-ticker z-scores use only
  strictly-prior observations; QoQ deltas and past-target stats are shifted;
  fundamentals/eventdata lag to public availability; and (learned the hard
  way, §6.2) any series used to demean/scale a target is banned from the
  feature set.

### Data facts
33,361 transcripts total; 18,392 in the modeled corpus (pub_date ≥
2016-07-01; avg 8,617 words; 99% have Q&A; 18,117 with Q&A usable for LLM
extraction). `sentiment_features`: 17,663 rows × 582 tickers. EPS surprise
matched: 17,510 rows. Prices: 698 tickers, 1.75M rows.

## 4. EXPERIMENT LOG (chronological; all numbers = untouched eval region)

### 4.0 Legacy model (pre-2026-07) — a null
1-day target, ~1,200 transcripts, leaky full-sample z-scores, tuning on the
reporting folds. After protocol fixes: IC −0.02, AUC 0.516, decile spread
−84 bps, sign 49.6%, R² −0.28. Conclusion: the 1-day post-earnings pop is
arbitraged; nothing to predict at that horizon.

### 4.1 Infrastructure rebuild (2026-07-04)
15× data via HF ingest, leak-free feature library, honest tune/eval
protocol, target pivot to `abnormal_30d` (PEAD), pipeline runner,
two-stage checkpointing in 03 (added after a POWER OUTAGE killed a 13.7h
un-checkpointed FinBERT run), Q&A-only FinBERT (~halves GPU time; prepared
remarks are scripted). 500-transcript smoke test passed end-to-end (25 min).

### 4.2 Full-run BASELINE (2026-07-05, raw abnormal_30d)
Level IC +0.041 (t=1.39); rank IC +0.055 (t=2.07) — first significant
signal; spread +302 bps; net Sharpe 0.49; short leg 61.5% hit. Top SHAP:
ticker, vix_close, qa_finbert_net_qoq, ix_full_lm_net_x_surprise. Concern:
`ticker` dominated SHAP → is this language or "knowing which names drift"?

### 4.3 Target ablation: demeaned + vol-adjusted (`abnormal_30d_va`)
Target = (abnormal_30d − ticker's strictly-prior mean drift) / (trailing
60d vol × √30, floor 3%).

- **Run 1 (BUGGED, instructive):** the demeaning series
  `past_abnormal_30d_mean` was left in the features → it topped SHAP at 10×
  everything else and the spread went NEGATIVE (−110 bps). The model was
  predicting the arithmetic −mean/scale component of the transformed target
  — a mechanical echo with zero raw-return alpha.
- **Run 2 (corrected — mechanical features excluded; predictions recombined
  to raw space):** pure language residual IC **−0.013**, spread **−139 bps**
  → NO standalone language signal. Recombined (residual + persistence
  prior): IC +0.021, Sharpe 0.42 — worse than baseline.
- **Conclusion (the project's key scientific finding):** the baseline's
  edge is mostly per-name PEAD persistence + VIX regime + surprise
  interactions, NOT transcript text. Word-count sentiment adds little.

### 4.4 FINAL model (raw target + SUE / surprise_streak / pre_vol_60d)
The §2 headline numbers: rank IC +0.066 (t=2.34), spread +217 bps, net
Sharpe 0.53, short leg 65.2%. NOTE: near-identical model variants swing
Sharpe 0.49↔0.53 and spread 217↔302 bps — treat single-run deltas as noise.

### 4.5 Direction classifier — honestly dead
Dead-zone design (train/eval only on |move| > 0.5σ, 10,248 events): tune
AUC ~0.55 but eval AUC 0.45–0.47 in both attempts. Direction of individual
names does not generalize even on large moves. The monetizable skill is
cross-sectional RANKING, expressed mostly through the short tail.

### 4.7 Momentum baseline (pre-2026-07-07) — notebook built, borderline
`06_momentum.ipynb` built with 9 features (mom_12_1, mom_6_1, rev_21d,
dist_52wk, vol_60d, beta_252, idio_mom, turn_trend, log_adv) on 106 monthly
rebalance dates (2017–2026, 62,756 rows, 614 tickers). Target: 21d forward
SPY-adjusted return. Same honest protocol as ECA (70/30 tune/eval split,
walk-forward).
- **Composite (dumb rank-average):** IC +0.0013 (t=0.03), decile spread
  −150 bps, gross Sharpe −0.76 — effectively zero.
- **LightGBM:** IC +0.0258 (t=1.88, 59% positive), decile spread +206 bps,
  gross Sharpe +1.83, beta-hedged net Sharpe +0.96.
- **Model vs composite:** paired IC diff +0.0245 (t=0.56) — not statistically
  better. The apparent Sharpe is mostly beta-tilting in a bull market.
- **Verdict:** doesn't pass the t≥2 gate. Needs improvement.

### 4.8 Momentum improvements (2026-07-07) — refactored, not yet re-run
Created `src/momentum.py` reusable module following `src/features.py`
conventions. Three key improvements from a thinker-with-files-gemini analysis:
1. **Vol-adjusted features** — `mom_12_1_va = mom_12_1 / vol_60d`, etc.
   Stops the model from picking high-vol lottery tickets.
2. **Sector-neutral ranks** — within-sector percentile ranks strip macro
   sector bets, isolating stock-level momentum.
3. **Beta-adjusted target** — `target = ret − beta × SPY_ret` instead of
   `ret − SPY_ret`. Forces the model to learn true idiosyncratic alpha,
   not "buy beta > 1.2 in a bull market."
Feature count grew from 9 to ~41 (13 base + 4 vol-adj + 13 sector-neutral
+ 4 vol-adj-sn + liquidity). Notebook updated to use `build_momentum_panel()`
and `add_composite_score()`. **Results pending — re-run needed.**

### 4.9 LLM engine comparison (2026-07-07) — Ollama ruled out
Head-to-head on 5 Q&A transcripts: Gemini (2.5-flash-lite) vs Ollama
(qwen2.5:14b and qwen2.5:7b).
- **Gemini:** 0.5s per call, works perfectly.
- **Ollama 7B:** 114s for a simple "hello" prompt. Unusable for 1,941
  transcripts.
- **Ollama 14B:** timed out (120s+). Too heavy for this hardware.
- **Decision:** Ollama is dead. Use Google billing on gemini-2.5-flash-lite
  (~$2) when the LLM pilot runs. Anthropic Haiku (~$15–20) is the fallback
  quality option.

### 4.10 Daily snapshots automated (2026-07-07)
`scripts/snapshot_daily.py` was already built. Set up GitHub Actions cron
(weekdays noon UTC) that runs the script, commits `data/snapshots.db` back
to the repo for persistence. Fixed `.gitignore` (`data/*` + exception for
snapshots.db) and workflow (`git add -f` to bypass gitignore). Universe:
57 curated tickers in CI (market.db not in repo). First run succeeded
(58/58 OK, 0.7 min).

### 4.6 LLM Q&A extraction — skeleton built + smoke-validated (not yet piloted)
Plan: `docs/llm_qa_plan.md`. Rationale sharpened by 4.3: if counting words
carries no residual signal, extracting MEANING (guidance direction/
confidence, demand/margin outlook, dodged questions, tone-vs-numbers gap,
unexpected negatives, analyst pushback — 8 ordinal features) is the
remaining language avenue.

- Built: `src/llm_features.py` (strict-JSON prompt, parse/validate/clamp,
  retry, providers: Gemini / local Ollama / Anthropic, auto-detect from
  .env; 429/5xx backoff; keys never echoed in errors) +
  `scripts/run_llm_extraction.py` (checkpoint table `llm_qa_scores`,
  resume-by-key, flush-every-50, --dry-run/--limit/--pilot N/--rpm).
  17 offline tests.
- **Smoke test (18/20 scored, all Agilent 2016–2021):** scores track known
  history — flagged guidance cuts exactly in Agilent's real stumble quarters
  (q2 FY2019 China slowdown: gd=−1, demand=−1, unexpected_negative=1;
  q2 FY2020 COVID: gd=−1) and beat-and-raise quarters as gd=+1/confidence 2.
  Nulls used correctly. Caveat: could partly be the LLM *remembering*
  Agilent — see §5.7.
- **Blocked on engine choice (§7):** the free Gemini key allows only ~20
  requests/DAY (Google slashed free tiers; 2.0-gen models have limit 0).
  Pilot = 1,941 transcripts (~65 sector-stratified complete tickers).

## 5. CAVEATS AND HOLES — read before believing anything above

1. **Eval-tail contamination (the biggest one).** The "untouched" eval
   region was consulted ~3× in one day while iterating on targets/features.
   Formally, tuning never saw it; informally, design decisions did. The
   final model's improvement over baseline is a *hypothesis*, not a
   confirmed result. Real confirmation requires data after 2025-05 (accrues
   with time) or a strict no-more-looks freeze.
2. **Single regime.** All eval numbers come from Nov 2022 – May 2025 — one
   bull market. Behavior in a drawdown/rate-shock regime is unknown. The
   Sharpe-1.43 SPY benchmark in the same window shows how favorable it was.
3. **Run-to-run variance.** Near-identical models swing Sharpe 0.49↔0.53,
   spread 217↔302 bps. Any single-run delta smaller than that is noise.
4. **The `ticker` feature still tops SHAP** even in the final model. The
   ablation says that's largely real per-name drift persistence (which the
   model is allowed to use), but it means the "NLP model" branding
   overstates the role of language.
5. **Backtest artifacts.** The `total_return`/`CAGR` columns in 05 compound
   463 *overlapping* 30-day events as if sequential — SPY shows an absurd
   24,878% under the same arithmetic. Only Sharpe/IC/spread are valid.
   The overlapping-hold Sharpe itself is a signal-quality proxy, not a live
   track record (no capital constraints, no fill modeling, close-price
   executions).
6. **Survivorship/coverage bias.** The HF transcript corpus covers current
   large/mid caps; delisted companies are underrepresented, which flatters
   short-side backtests especially.
7. **LLM look-ahead risk.** An LLM scoring 2016–2025 transcripts may "know"
   what happened to these companies from training data (the Agilent smoke
   accuracy is partly suspect for exactly this reason). Mitigations: prompt
   forbids outside knowledge; features are in-transcript facts, not
   predictions; planned robustness check = re-score ~200 transcripts with
   company names masked and compare. Must be disclosed in any write-up.
8. **Costs are assumptions.** 10 bps round-trip + 50 bps/yr borrow is
   plausible for liquid large caps but untested against real fills; borrow
   availability for the short leg (where ALL the edge is) is unverified.
9. **Direction is unpredictable — don't let anyone re-add it.** Two honest
   attempts failed out-of-sample (§4.5). Any future claim of direction
   accuracy should be presumed leaky until proven.
10. **Uncommitted work + hidden folders.** Everything from this sprint is
    uncommitted (user's call). `tests/` and `docs/` are gitignored per an
    earlier user request — for a portfolio repo, un-hiding tests (a selling
    point) is worth reconsidering.
11. **Sparse events.** ~4 events/ticker/year; the signal says nothing
    between earnings seasons and positions clump within them. This is why
    the trading-bot plan adds daily-firing signals (momentum) before any
    ensemble.
12. **Crowded trade.** Everyone parses earnings calls since LLMs got cheap.
    Assume published-anomaly decay; the measured edge may erode.

## 6. LESSONS LEARNED (write-up material)

1. **Leakage hides everywhere** — full-sample z-scores, tuning on reporting
   folds, and the subtlest: giving the model the series its target was
   demeaned with (§4.3 run 1). Each inflated results silently.
2. **Whatever you transform a target with must be excluded from features.**
   The mechanical-echo bug is a textbook case: top-SHAP feature, negative
   real-world performance.
3. **Pick metrics that match the trade.** RMSE/R² barely move on return
   noise; IC + t-stat and decile spread are what a book monetizes; AUC over
   all events is meaningless when 40% are ±0.2σ wiggles.
4. **Ablate before you brag.** Two 4-minute re-runs revealed the alpha
   lives in numbers-about-the-event, not words — reframing the whole
   project honestly.
5. **Negative language > positive language.** Short leg 65% vs long leg
   coin-flip: scripted optimism is uninformative; trouble leaks through.
6. **Horizon choice is the difference between null and signal.** Same
   pipeline: 1-day target = nothing (arbitraged), 30-day drift = t > 2.
7. **Checkpoint everything long-running.** A power outage destroyed 13.7h
   of GPU work; the rebuilt two-stage checkpointing has already paid for
   itself repeatedly (Ctrl+C safe, resume-by-key).
8. **Operational war stories:** defeatbeta-api install silently upgraded
   numpy/pandas past pins (restore 1.26.4/2.2.0); Windows cp1252 console
   breaks on Unicode (reconfigure stdout to utf-8); nbclient buffers
   notebook prints (hence the external progress monitor); free API tiers
   changed under us mid-project (Gemini 2.0 models now have ZERO free
   quota; 2.5-flash-lite ~20 requests/day on this key).
9. **Test local models before committing to them.** Ollama qwen2.5:7b took
   114s for a 1-word prompt. A 1,941-transcript pilot would take ~61 hours.
   Always run a simple head-to-head comparison before choosing an engine.
10. **GitHub Actions artifact retention is 90 days — not enough for
   cumulative data.** The snapshot workflow initially used artifacts, which
   would silently lose old data. Fixed by committing the DB back to the
   repo instead. For long-term accumulation, always persist to git.
11. **`.gitignore` directory rules block exceptions.** `data/` ignored the
   entire directory, so `!data/snapshots.db` couldn't un-ignore it. Fix:
   use `data/*` (ignore contents, not the directory itself) + the exception.
12. **Momentum Sharpe in a bull market is mostly beta.** The beta-hedged
   Sharpe (0.96) vs SPY-adjusted Sharpe (1.83) exposed that most of the
   apparent edge was mechanical beta tilt. Always beta-hedge momentum
   backtests; the honest number is the lower one.

## 7. ROADMAP AND OPEN DECISIONS

**Immediate (LLM pilot — skeleton done, engine DECIDED):**
| option | cost | pilot wall time | note |
|---|---|---|---|
| Google billing on gemini-2.5-flash-lite | ~$2 total | ~3 h | **chosen**: cheapest real option |
| ~~Local Ollama (llama3.1:8b)~~ | — | — | **ruled out 2026-07-07**: qwen2.5 models too slow (114s/call) on this hardware |
| Anthropic Haiku | ~$15–20 | ~2 h | fallback quality option |

Pilot protocol: `--pilot 65` (≈1,941 transcripts, complete tickers,
sector-stratified) → join `llm_qa_scores` into 04 → A/B with vs without
`llm_*` features on the same folds. **Gate: +0.010 mean IC or top-8 SHAP
entry with sensible signs; otherwise stop at ~$2–20 spent.** Then the
name-masking robustness check (§5.7).

**After the pilot (trading-bot layers, specced in trading_bot_layers.md):**
~~momentum/reversal signal (a weekend; free data; becomes the benchmark)~~
→ DONE (notebook exists, module built; needs re-run with improvements) →
regime gate (days; VIX/term-structure/credit/breadth → exposure multiplier;
rules-based) → fundamentals + positioning signals (1–2 weeks each) →
ensemble (z-scores, ECA event-decay over 30 days, IC-proportional weights)
→ vol-scaled sizing with caps and trade bands → nightly paper-trading loop
with per-signal IC probation. Membership rule: every new signal must show
t ≥ 2 held-out IC under the shared harness before joining.

**Hygiene:** git commit DONE (2026-07-07 — `d5e4059`). Fresh out-of-sample
confirmation window (post-2025-05) should be declared and left untouched.
`docs/` and `tests/` remain gitignored per user preference.

## 8. GLOSSARY

- **PEAD** — post-earnings-announcement drift: stocks keep moving in the
  direction of earnings news for weeks. The anomaly this project trades
  (`abnormal_30d` = 30-trading-day SPY-adjusted return after the call).
- **IC** — information coefficient: Spearman rank correlation between
  predictions and subsequent returns, computed monthly; reported as mean
  with a t-stat across months. 0.02–0.05 with t > 2 is practically usable.
- **Decile spread** — mean realized return of the top prediction decile
  minus the bottom decile, per event, in basis points.
- **SUE** — standardized unexpected earnings: EPS surprise ÷ the ticker's
  own historical surprise dispersion (strictly-prior).
- **Dead zone** — direction classification restricted to |move| > 0.5σ so
  unlabelable wiggles don't poison training or the metric.
- **Walk-forward** — each eval block predicted by a model trained only on
  strictly earlier events, expanding window.
- **Tune/eval split** — hyperparameters + feature selection see only the
  first 70% by date; all reported numbers come from the last 30%.
- **`abnormal_30d_va`** — the ablation target: (raw drift − ticker's prior
  mean drift) / (trailing vol × √30). Switchable via `TARGET` in notebook
  04's imports cell.
