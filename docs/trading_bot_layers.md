# Trading Bot Architecture — Layered Thinking

> ECA (earnings-call transcripts) as Layer 1 of an eventual trading system.
> What the layer buys, what it can't do, and what the other layers are.
> Written 2026-07-05.

## How to read the "layers" (they are three groups, not ten steps)

The numbered layers below are NOT a stack of ten required floors. The system
has three jobs, and each layer belongs to one group:

```
GROUP A: SIGNALS (a menu — pick 2-4)          "what looks cheap/expensive?"
   L1 transcripts · L2 price/technical · L3 fundamentals ·
   L4 analyst revisions · L5 alt-text · L6 options/positioning · L8 flows

GROUP B: THE GATE (one)                        "should we be trading at all right now?"
   L7 macro/regime (VIX, rates, credit spreads)

GROUP C: THE DECISION STACK (exactly one of each, always required)
   L9  ensemble  — merge all signals into ONE forecast per stock
   L10 sizing    — forecasts -> positions within risk limits (where Sharpe
                   is kept or lost)
   L11 live loop — paper trading, decay detection, kill switch
```

Why several weak signals instead of one perfected signal: every individual
signal is weak (ECA's IC ~0.06 is NORMAL), and performance scales roughly as
skill x sqrt(independent bets) — the fundamental law of active management.
Uncorrelated signals firing at different times (ECA speaks ~4 days a year
per stock; momentum speaks daily) add edge while their noise partially
cancels. Three IC-0.05 signals that don't overlap behave like one much
stronger signal — which is why funds run dozens of weak signals rather than
hunting a single strong one that would be arbitraged away.

A minimum viable bot = 2-3 signals from Group A + the gate + all of Group C.

---

## Layer 1: Earnings-call transcripts (what ECA is today)

### What this layer provides

1. **A catalyst-anchored signal.** Every prediction is tied to a scheduled
   event with a known timestamp — clean entry timing, a defined 30-day
   holding window (PEAD), and no ambiguity about when information arrived.
   Most signals don't have this luxury.
2. **The PEAD anomaly itself.** The measured edge (rank IC +0.066, t = 2.34;
   net Sharpe ~0.5) rides a documented, slow-diffusing market inefficiency —
   drift persists for weeks, so a retail-speed system can actually trade it.
3. **Forward-looking information numbers don't carry yet.** Guidance,
   demand/margin commentary, management confidence — the earnings *number* is
   backward-looking; the call is where the future gets discussed. (The LLM
   extraction layer is the attempt to capture this properly.)
4. **Behavioral tells, especially negative ones.** The short leg hits 65% —
   trouble leaks through language (hedging, dodged questions, evasiveness)
   even when the numbers look fine. Scripted optimism is uninformative;
   unscripted stress is.
5. **Cross-sectional ranking machinery.** The layer forces you to build the
   right skeleton — point-in-time features, walk-forward eval, IC/t-stat,
   cost-aware backtest — which every later layer plugs into unchanged.
6. **Companion numeric signals for free.** EPS surprise/SUE, surprise
   streaks, per-name drift persistence, pre-event momentum, VIX regime —
   already computed, already validated.
7. **Reusable NLP infrastructure.** Cleaning, sectioning, chunked scoring,
   checkpointed batch inference — the same machinery scores news, 10-K/Q
   filings, or press releases with minor changes.

### Drawbacks and hard limits

1. **The language alpha is thin.** The ablation was blunt: strip per-name
   drift habits and volatility, and word-count sentiment predicts ~nothing
   (residual IC −0.01). The proven edge is mostly *numbers about the event*
   (persistence, surprise, regime), not the words. Semantic LLM features are
   the remaining hope for text — unproven until the pilot runs.
2. **Sparse and lumpy.** ~4 events per ticker per year. Between earnings
   seasons the layer says nothing; during them, positions overlap and clump.
   A bot needs signals that fire on the other ~230 trading days too.
3. **Crowded trade.** Since LLMs became cheap (~2023), everyone parses calls.
   Public-anomaly edges decay; PEAD itself is weaker than in the 1990s
   literature. Assume the easy part is arbitraged and only the subtle
   residuals remain.
4. **Universe constraints.** English-language calls, mostly US large/mid
   caps, dataset coverage gaps, no delisted-company transcripts (survivorship
   bias flatters backtests).
5. **Weak standalone economics.** Sharpe ~0.5 before real-world frictions
   (borrow availability, fill quality, capacity) is a *component*, not a
   strategy. It needs an ensemble around it.
6. **One-regime evidence.** Everything is measured on Nov-2022→May-2025 — a
   bull market. Unknown behavior in a drawdown or rate shock.
7. **Evaluation debt.** The eval tail was consulted repeatedly during design;
   true confirmation needs fresh data. And LLM-scored history carries
   look-ahead risk (the model may "know" these companies' futures).

---

## The other layers

| # | Layer | What it adds | Data cost | Fires how often | Synergy with Layer 1 |
|---|-------|--------------|-----------|-----------------|----------------------|
| 2 | **Price/technical** | Momentum (12-1), short-term reversal, vol regimes, 52-week highs, liquidity | Free (already have prices) | Daily, every name | Momentum + PEAD interact (drift is stronger with the trend); fills the between-earnings gap |
| 3 | **Fundamentals/factors** | Value (E/P, B/P), quality (ROIC, accruals), growth — the classic factor prior | Free-ish (yfinance/defeatbeta financials) | Quarterly | The "default book" the bot holds when no event signal is live |
| 4 | **Analyst estimates** | Post-call estimate revisions, dispersion, recommendation changes | The hard part — free sources are thin | Weekly | **The single best PEAD companion**: analysts cutting numbers after a call is the drift's engine |
| 5 | **Alternative text** | News flow, 8-K/10-K language deltas (YoY change in risk factors), press releases | Free-to-cheap; reuses ECA's NLP stack | Daily | Extends the transcript machinery to 250 days/year |
| 6 | **Options/positioning** | Implied vs realized vol, skew, short interest, days-to-cover, 13F changes | Short interest free (FINRA); options history costs | Daily/biweekly | Short interest × negative-call signal is a natural short-leg filter |
| 7 | **Macro/regime** | VIX term structure, rates, credit spreads, breadth → regime classifier | Free | Daily | VIX is already a top-3 feature; formalizing regime gating decides *when* Layer 1 trades at all |
| 8 | **Flows/mechanics** | Index rebalances, buyback blackout windows, ETF flows | Mixed | Episodic | Explains "why did my clean signal fail this month" variance |
| 9 | **Ensemble/meta-model** | Stacks all layer outputs; time-varying weights; uncertainty estimates; purged CV | Engineering, not data | — | Where the layers become one forecast instead of a pile of scores |
| 10 | **Portfolio construction & execution** | Vol targeting, position limits, netting overlapping signals, cost-aware scheduling | Engineering | — | **Where Sharpe is kept or lost.** A 0.5-Sharpe signal with good sizing beats a 0.7 with bad sizing |
| 11 | **Live loop & monitoring** | Paper trading, IC-decay tracking, drift detection, kill switches | Engineering | Continuous | The only honest test left after in-sample iteration |

### Sensible build order

1. **Layer 2 (technical)** — free, immediate breadth, and it's the control
   variable every other layer must beat.
2. **Layer 7 (regime gate)** — small, high leverage: VIX already matters in
   ECA; make it a first-class switch instead of a feature.
3. **Layer 4 (estimate revisions)** — highest expected marginal value for the
   PEAD trade specifically; the blocker is a free data source, worth research.
4. **Layer 9 + 10 (ensemble + sizing)** — once ≥2 signal layers exist.
5. Layers 5/6/8 opportunistically, gated by the same A/B discipline as the
   LLM pilot: a new layer earns its place only by moving IC on held-out data.

### The principle that carries over from ECA

Every layer gets the same treatment the transcripts got: point-in-time
features, tune/eval separation, IC + t-stat as the gate, cost-aware backtest,
and an ablation that asks *where the alpha actually comes from* before any
claim is believed. ECA's most valuable output isn't the 0.5 Sharpe — it's
this protocol.

---

## Concrete v1 spec (chosen 2026-07-05: three more signals + gate + stack)

All chosen for zero/near-zero data cost using infrastructure already built.

### Signal 2 — Momentum & Reversal (data: existing `prices` table)
Per stock per day: 12-1 momentum (t-252 -> t-21 return), 6-1 momentum,
short-term reversal (5d/21d, negative predictor), distance from 52-week
high, trailing 60d vol (low-vol tilt), idiosyncratic momentum (minus
beta x SPY). Build order: (1) dumb rank-average composite as the benchmark,
(2) LightGBM on a monthly cross-section, target = next-21d abnormal return,
same walk-forward harness. Expect IC 0.02-0.05 but fires daily on every
name — the breadth engine. Failure mode: momentum crashes on rebounds
(handled by the gate, which turns momentum OFF in risk-off).

### Signal 3 — Fundamentals (data: defeatbeta/yfinance statements, free)
Per stock per quarter, lagged to filing date: FCF yield, EBIT/EV, gross
profitability (GP/assets), accruals (negative), asset growth (negative),
net share issuance, margin trend. Slow layer: quarterly refresh, months-long
holds, ~zero turnover cost. Most uncorrelated to the other signals (reads
accounting, not prices or words). This is the default book.

### Signal 4 — Positioning (data: FINRA short interest + SEC Form 4, free)
(a) Short interest: days-to-cover, and delta-SI (rising shorts + negative
ECA call = best shorts; extremely crowded shorts get a weight cap for
squeeze risk). (b) Insider Form 4 via edgartools: dollar-weighted net
officer buying over 90d, cluster-buy flag (3+ insiders). Event-driven like
ECA — same machinery.

### The Gate — src/regime.py, RULES-BASED, few knobs (overfit magnet!)
Daily inputs (all free): SPY vs 200dma; VIX level + VIX vs VIX3M
(term-structure inversion = acute stress); HYG-vs-LQD relative return
(credit); universe breadth (% above 200dma from own prices). Output: gross
multiplier — risk-on 1.0 / caution ~0.6 / risk-off ~0.25 with momentum OFF
but ECA/PEAD still on at reduced size. Do not iterate thresholds against
the same eval window.

### The Stack
- Ensemble v1: cross-sectional z-score per signal per date; ECA event score
  decays linearly to 0 over 30 trading days so event + daily signals merge
  into one daily book. Equal weights first -> IC-proportional (tune-region
  ICs only) -> eventually LightGBM meta-model with purged CV.
- Sizing v1: weight ∝ combined score / stock vol; normalize to gross target;
  caps ~3%/name, ~25%/sector, net within ±20%; trade band (rebalance only on
  |Δweight| > 0.5%) to control turnover; existing cost model applies.
- Live loop v1: nightly Task Scheduler job -> update data -> scores ->
  target book CSV, paper-filled at next open; weekly report of PnL + rolling
  6-month IC per signal; probation rule (6 months of negative rolling IC =
  benched). No broker connection until months of clean paper trading.

### Membership rule + effort estimates
A signal joins the ensemble only after showing t >= 2 held-out IC under the
shared harness (each new signal is a mini-ECA). Rough effort: momentum
composite = a weekend; gate = days; fundamentals = 1-2 weeks; positioning =
1-2 weeks; ensemble + sizing v1 = ~1 week once two signals exist.

---

## Part 3: After the layers — operating workflow & research frontiers

Once v1 exists, the work changes from BUILDING to OPERATING A RESEARCH
FACTORY. Two tracks, with the harness as the only door between them:

```
RESEARCH TRACK                        PRODUCTION TRACK
hypothesis -> leak-free features      nightly: data update -> QC checks ->
-> shared harness (walk-forward)      signals -> gate -> ensemble -> sizing ->
-> t >= 2 held-out IC? -- yes ----->  target book -> paper fills -> logs
        | no                          weekly: PnL attribution BY SIGNAL
        v                             quarterly: scheduled retrain, probation
   killed (logged, with numbers)      review, post-mortem of worst month
```

### Operating cadence (what makes it honest)
| when | do | forbidden |
|---|---|---|
| nightly (automated) | update data, QC alarms (stale prices, missing tickers, split anomalies), compute signals, emit paper book | manual tweaks |
| weekly | read the report: per-signal PnL attribution, rolling 6-month IC each, drawdown | reacting to one bad week |
| quarterly | scheduled retrain (expanding window, frozen protocol); signal probation (6 months negative rolling IC = benched); post-mortem | off-schedule re-tuning |
| once, immediately | declare an EVAL FREEZE: all data after 2025-05 is confirmation-only, never used for design | peeking |

Data QC is a first-class citizen: corporate actions, ticker changes,
delistings, stale quotes. Most retail bots die of bad data before bad alpha.

### Hardening checklist before real money
1. Regime replay: run the combined book through 2020-03 and 2022
   synthetically; verify the gate cuts exposure when it should have.
2. Cost sensitivity: strategy at 10/25/50 bps round-trip; if the edge dies
   at 25 bps it is not deployable.
3. Correlation audit: rolling correlation matrix of signal RETURNS; a new
   signal 0.8-correlated to momentum added nothing.
4. Sizing ramp: live (Alpaca/IBKR) at 1-5% of intended capital; scale only
   while live-vs-paper tracking error stays small. Hard-coded guardrails:
   max daily loss, per-name caps, kill switch.

### Research frontiers after v1 (ranked)
0. **START NOW (time-sensitive, free): proprietary data snapshots.** A tiny
   nightly job recording yfinance analyst estimates + FINRA short interest
   (+ options IV if reachable) into SQLite. Free sources expose only
   CURRENT values — in 12 months this becomes an estimate-revisions history
   (the paywalled Layer 4 data) that cannot be downloaded retroactively.
   Every day not recording is history lost forever.
1. Regime-conditional ensemble: the meta-model learns WHEN each signal
   works (momentum off post-vol-spike; PEAD stronger in high VIX) — usually
   worth more than a fifth signal.
2. Uncertainty-aware sizing: quantile LightGBM prediction intervals ->
   size by confidence, not just rank.
3. LLM-native deepening: per-call embeddings stored beside engineered
   features; historical-analog retrieval (20 most similar past calls ->
   what happened next); per-exchange Q&A analysis (question sentiment vs
   answer quality) instead of whole-transcript averages.
4. Execution alpha inside the drift window: day-2 entry vs day-0? profit-
   target exits vs fixed 30-day hold? Cheap experiments on existing data.
5. Universe expansion: small/mid caps = more inefficiency, worse costs and
   borrow — the cost model decides, not enthusiasm.
6. Auto-research agent: because the harness is standardized (features in ->
   walk-forward -> IC out), an LLM agent can propose features, run the
   harness, and report against the t>=2 gate — a feature factory where bad
   ideas die cheaply. The protocol is the guardrail that keeps a fast
   generator honest.

### The meta-principle
After v1 the scarce resource is not ideas or code — it is UNCONTAMINATED
EVALUATION DATA. Every process above either accumulates more of it
(snapshots, the freeze, months of paper trading) or spends it slower
(gates, scheduled retrains, probation instead of reactive tweaks).
