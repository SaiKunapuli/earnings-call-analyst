# LLM Q&A Feature Extraction — Implementation Plan

Goal: extract **semantic, directional** features from earnings-call Q&A that
word-counting (VADER/LM/FinBERT) cannot see. FinBERT knows the call *sounded
negative*; an LLM can tell you *management lowered guidance and dodged three
questions about margins*. Direction lives in content, not tone.

Status: **planned — start after the vol-adjusted target (1) and SUE features
(2) are validated.** Nothing here runs yet.

---

## 1. Features to extract (one JSON per transcript)

Deliberately few, mostly ordinal/categorical — robust to prompt noise, easy
to validate, and directly usable by LightGBM:

| field | type | meaning |
|---|---|---|
| `guidance_direction` | -1 / 0 / +1 / null | lowered / maintained / raised; null = no guidance discussed |
| `guidance_confidence` | 0–2 | how firmly management stood behind its outlook |
| `demand_outlook` | -2…+2 | forward demand commentary (orders, pipeline, bookings) |
| `margin_outlook` | -2…+2 | forward cost/pricing/margin commentary |
| `n_questions_dodged` | int | analyst questions deflected or answered non-responsively |
| `tone_numbers_gap` | -2…+2 | tone rosier (+) or gloomier (−) than the numbers discussed |
| `unexpected_negative` | 0/1 | a negative surprise surfaced in Q&A not present in prepared remarks |
| `analyst_pushback` | 0–2 | analysts openly skeptical / repeatedly challenging |

Derived in 04 afterwards: QoQ deltas (`llm_guidance_direction_qoq`, …) and
interactions with `sue` (guidance cut *despite* a beat is the classic PEAD short).

## 2. Architecture — clone the FinBERT stage pattern

The checkpointing/resume design from notebook 03 Stage B is already proven;
reuse it wholesale:

- `src/llm_features.py` — prompt template, JSON-schema validation, one
  `score_transcript(qa_text) -> dict` function. Temperature 0, strict JSON
  output, one retry on invalid JSON, then mark failed.
- `scripts/run_llm_extraction.py` — standalone runner (NOT in the notebook:
  API calls are slow/parallel and shouldn't block the pipeline). Reads keys
  from `transcripts_text`, skips ones already in the checkpoint table,
  scores with N parallel workers, flushes every 50.
- Checkpoint table `llm_qa_scores(ticker, quarter, year, scores TEXT,
  model TEXT, PRIMARY KEY(ticker,quarter,year))` — same resume semantics as
  `finbert_scores`. Fully pausable/restartable from day one.
- Notebook 03 join cell gains an optional merge of `llm_qa_scores` into
  `sentiment_features` (columns prefixed `llm_`); 04's candidate logic picks
  them up as a new group with a coverage gate scoped to the pilot universe.
- API key in `.env` (already gitignored). Never in code or notebooks.

## 3. Engine choice

| option | cost (2k pilot / 18k full) | wall time (pilot) | notes |
|---|---|---|---|
| **Claude Haiku (API) — recommended for pilot** | ~$15–20 / ~$140 (≈half with Batch API) | ~1–2 h at 10 parallel | best quality-per-dollar, strict JSON reliable |
| Local 8B (Ollama / llama.cpp on the AMD GPU) | free | ~12–24 h | fine for scale-up if pilot proves value; validate agreement vs API on ~100 transcripts first |

Q&A sections average ~4–5k words ≈ 6–7k tokens in; ~150 tokens out.

## 4. Pilot design (the part that makes the result trustworthy)

- **Sample complete tickers, not random transcripts**: all calls for a
  random ~65 tickers stratified by sector ≈ 2,000 transcripts. This keeps
  each ticker's panel complete so QoQ deltas and expanding stats work, and
  the honest tune/eval split stays intact within the sub-universe.
- Run 04's full protocol on that sub-universe twice: **with** and
  **without** `llm_*` features (same folds, same Optuna budget).
- **Success gate**: mean monthly IC improves by ≥ +0.010 on the eval tail,
  or `llm_*` features enter the top-8 SHAP with economically sensible signs.
  Pass → scale to all 18,392 (Batch API overnight). Fail → stop at ~$20 spent.

## 5. Leakage caveat (must go in any writeup)

The LLM's training data may include knowledge of what happened to these
companies *after* historical calls. Mitigations: the prompt forbids using
anything beyond the transcript text; features are relative/ordinal rather
than predictions ("did guidance go up?" not "will the stock go up?").
Optional robustness check: re-run ~200 transcripts with company names masked
to "the Company" and confirm feature agreement. Residual look-ahead risk is
a known limitation of every LLM-on-historical-text study — disclose it.

## 6. Build order (~half a day of work when started)

1. `src/llm_features.py` + prompt, unit tests with 2–3 canned Q&A snippets
2. `scripts/run_llm_extraction.py` with checkpoint/resume + `--limit`/`--tickers`
3. Smoke: 20 transcripts, eyeball the JSON against the actual Q&A text
4. Pilot: ~65 tickers → 04 A/B → gate decision
5. Scale-up + join into the main pipeline
