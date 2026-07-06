"""LLM Q&A feature extraction — core logic (see docs/llm_qa_plan.md).

Extracts semantic, directional features from earnings-call Q&A sections that
word-counting (VADER/LM/FinBERT) cannot see: did management raise or lower
guidance, dodge questions, sound rosier than the numbers?

Provider-agnostic: the runner picks whichever backend is configured —
    ANTHROPIC_API_KEY          -> Anthropic (Claude Haiku; pilot quality)
    GEMINI_API_KEY/GOOGLE_API_KEY -> Google Gemini (generous FREE tier:
                                      https://aistudio.google.com/apikey)
    neither                    -> local Ollama at localhost:11434 (free,
                                      needs `ollama pull llama3.1:8b`)

Keys belong in `.env` at the repo root (gitignored) — never in code.

No network calls happen at import time; everything is testable offline via a
fake provider (see tests/test_llm_features.py).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Feature specification — few, ordinal, robust to prompt noise
# ---------------------------------------------------------------------------
# name -> (lo, hi, nullable). All integers; None allowed only where nullable.
FEATURE_SPEC: dict[str, tuple[int, int, bool]] = {
    "guidance_direction":  (-1, 1, True),   # lowered/maintained/raised; null = not discussed
    "guidance_confidence": (0, 2, True),    # how firmly management stood behind outlook
    "demand_outlook":      (-2, 2, False),  # forward demand commentary
    "margin_outlook":      (-2, 2, False),  # forward cost/pricing/margin commentary
    "n_questions_dodged":  (0, 30, False),  # deflected / non-responsive answers
    "tone_numbers_gap":    (-2, 2, False),  # tone rosier (+) or gloomier (-) than numbers
    "unexpected_negative": (0, 1, False),   # negative surfaced in Q&A only
    "analyst_pushback":    (0, 2, False),   # open skepticism / repeated challenges
}

MAX_QA_WORDS = 5000          # truncate very long Q&A sections (keeps context sane)

SYSTEM_PROMPT = """You are a buy-side analyst scoring the Q&A section of an earnings call transcript.

Rules:
- Use ONLY information contained in the transcript text. Do not use any outside knowledge about the company, its industry, or what happened after this call.
- Respond with a single JSON object and NOTHING else — no prose, no markdown fences.
- Every field must be present. Integers only. Use null ONLY where allowed.

Fields:
- "guidance_direction": -1 if management lowered/withdrew guidance, 0 if maintained/reiterated, 1 if raised. null if guidance was not discussed.
- "guidance_confidence": 0 = hedged/uncertain about outlook, 1 = neutral, 2 = firmly confident. null if no outlook discussed.
- "demand_outlook": forward-looking demand commentary (orders, pipeline, bookings, customers): -2 very negative .. 0 neutral/mixed .. +2 very positive.
- "margin_outlook": forward-looking cost/pricing/margin commentary: -2 .. +2.
- "n_questions_dodged": count of analyst questions that were deflected, declined, or answered non-responsively.
- "tone_numbers_gap": -2 if tone is much gloomier than the figures discussed, 0 if aligned, +2 if tone is much rosier than the figures.
- "unexpected_negative": 1 if a negative fact surfaced during Q&A that was not part of the prepared narrative, else 0.
- "analyst_pushback": 0 = routine questions, 1 = some skepticism, 2 = open challenge / repeated pushback."""


def truncate_words(text: str, max_words: int = MAX_QA_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [TRUNCATED]"


def build_prompt(qa_text: str, max_words: int = MAX_QA_WORDS) -> str:
    return (f"{SYSTEM_PROMPT}\n\n"
            f"--- TRANSCRIPT Q&A SECTION ---\n{truncate_words(qa_text, max_words)}\n"
            f"--- END TRANSCRIPT ---\n\nJSON:")


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------
class LLMExtractionError(Exception):
    pass


def parse_llm_json(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Tolerates markdown fences and stray prose around the object, since not
    every backend honors 'JSON only' perfectly.
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise LLMExtractionError(f"no JSON object in response: {text[:200]!r}")
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMExtractionError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMExtractionError("response JSON is not an object")
    return obj


def validate_scores(obj: dict) -> dict:
    """Check all FEATURE_SPEC fields exist, are ints (or allowed nulls),
    and clamp to their ranges. Returns a clean dict with exactly the spec keys."""
    out = {}
    for name, (lo, hi, nullable) in FEATURE_SPEC.items():
        if name not in obj:
            raise LLMExtractionError(f"missing field: {name}")
        v = obj[name]
        if v is None:
            if not nullable:
                raise LLMExtractionError(f"{name} may not be null")
            out[name] = None
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise LLMExtractionError(f"{name} not numeric: {v!r}")
        out[name] = int(max(lo, min(hi, round(float(v)))))
    return out


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def load_env(path: Path | None = None) -> None:
    """Minimal .env loader (no python-dotenv dependency). Never overrides
    variables already set in the environment."""
    p = path or PROJECT_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120,
               max_retries: int = 5) -> dict:
    """POST JSON with automatic backoff on 429/5xx (free tiers rate-limit hard).

    Honors Retry-After when present; otherwise exponential backoff capped at
    60 s. Errors never echo the URL query string (it can carry the API key).
    """
    safe_url = url.split("?")[0]
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
            if e.code in (429, 500, 502, 503, 529) and attempt < max_retries:
                retry_after = (e.headers or {}).get("Retry-After", "")
                delay = (float(retry_after) if retry_after.replace(".", "", 1).isdigit()
                         else min(4.0 * 2 ** attempt, 60.0))
                time.sleep(delay)
                continue
            raise LLMExtractionError(f"HTTP {e.code} from {safe_url}: {body}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries:
                time.sleep(min(4.0 * 2 ** attempt, 60.0))
                continue
            raise LLMExtractionError(f"network error calling {safe_url}: {e}") from e
    raise LLMExtractionError(f"exhausted retries calling {safe_url}")


class LLMProvider:
    """One method to implement: complete(prompt) -> response text."""
    name = "base"
    model = ""

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini REST API. Free-tier key from https://aistudio.google.com.

    Default model: gemini-2.5-flash-lite — as of 2026-07 the 2.0-generation
    models have NO free-tier quota (limit 0 -> instant 429); the 2.5 Flash
    models do, and flash-lite has the highest free RPM/RPD.
    """
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMExtractionError("GEMINI_API_KEY / GOOGLE_API_KEY not set")

    def complete(self, prompt: str) -> str:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 400,
                                 "responseMimeType": "application/json"},
        }
        data = _post_json(url, payload, headers={})
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMExtractionError(f"unexpected Gemini response: {data}") from e


class OllamaProvider(LLMProvider):
    """Local Ollama server — fully free, no key. `ollama pull llama3.1:8b`."""
    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "format": "json", "options": {"temperature": 0.0}}
        data = _post_json(f"{self.base_url}/api/generate", payload,
                          headers={}, timeout=600)
        return data.get("response", "")


class AnthropicProvider(LLMProvider):
    """Anthropic API (Claude Haiku) — recommended for the paid pilot."""
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMExtractionError("ANTHROPIC_API_KEY not set")

    def complete(self, prompt: str) -> str:
        payload = {"model": self.model, "max_tokens": 400, "temperature": 0.0,
                   "messages": [{"role": "user", "content": prompt}]}
        data = _post_json("https://api.anthropic.com/v1/messages", payload,
                          headers={"x-api-key": self.api_key,
                                   "anthropic-version": "2023-06-01"})
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMExtractionError(f"unexpected Anthropic response: {data}") from e


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Explicit name, or auto-detect from configured keys (then local Ollama)."""
    load_env()
    kwargs = {"model": model} if model else {}
    if name:
        return {"gemini": GeminiProvider, "ollama": OllamaProvider,
                "anthropic": AnthropicProvider}[name](**kwargs)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(**kwargs)
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiProvider(**kwargs)
    return OllamaProvider(**kwargs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_qa(provider: LLMProvider, qa_text: str,
             max_words: int = MAX_QA_WORDS) -> dict:
    """Score one Q&A section. One retry with a JSON nudge, then raises."""
    prompt = build_prompt(qa_text, max_words)
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            raw = provider.complete(
                prompt if attempt == 0
                else prompt + "\n\nReturn ONLY the raw JSON object, nothing else.")
            return validate_scores(parse_llm_json(raw))
        except LLMExtractionError as e:
            last_err = e
    raise LLMExtractionError(f"failed after retry: {last_err}")
