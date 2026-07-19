"""Offline tests for src/llm_features.py — no network, no keys."""
import json

import pytest

from src.llm_features import (FEATURE_SPEC, LLMExtractionError, LLMProvider,
                              build_prompt, parse_llm_json, score_qa,
                              truncate_words, validate_scores)

GOOD = {
    "guidance_direction": -1, "guidance_confidence": 0,
    "demand_outlook": -2, "margin_outlook": -1,
    "n_questions_dodged": 3, "tone_numbers_gap": 1,
    "unexpected_negative": 1, "analyst_pushback": 2,
}


# ---------------------------------------------------------------------------
# parse_llm_json
# ---------------------------------------------------------------------------
def test_parse_plain_json():
    assert parse_llm_json(json.dumps(GOOD)) == GOOD


def test_parse_fenced_json():
    assert parse_llm_json(f"```json\n{json.dumps(GOOD)}\n```") == GOOD


def test_parse_json_with_surrounding_prose():
    text = f"Here is my assessment:\n{json.dumps(GOOD)}\nHope that helps!"
    assert parse_llm_json(text) == GOOD


def test_parse_no_json_raises():
    with pytest.raises(LLMExtractionError):
        parse_llm_json("I cannot score this transcript.")


def test_parse_invalid_json_raises():
    with pytest.raises(LLMExtractionError):
        parse_llm_json("{guidance_direction: -1,}")


# ---------------------------------------------------------------------------
# validate_scores
# ---------------------------------------------------------------------------
def test_validate_passes_good():
    assert validate_scores(GOOD) == GOOD


def test_validate_clamps_out_of_range():
    d = dict(GOOD, demand_outlook=7, n_questions_dodged=999)
    out = validate_scores(d)
    assert out["demand_outlook"] == 2
    assert out["n_questions_dodged"] == 30


def test_validate_rounds_floats():
    out = validate_scores(dict(GOOD, margin_outlook=-0.6))
    assert out["margin_outlook"] == -1


def test_validate_allows_null_guidance_only():
    out = validate_scores(dict(GOOD, guidance_direction=None, guidance_confidence=None))
    assert out["guidance_direction"] is None
    with pytest.raises(LLMExtractionError):
        validate_scores(dict(GOOD, demand_outlook=None))


def test_validate_missing_field_raises():
    d = dict(GOOD)
    del d["analyst_pushback"]
    with pytest.raises(LLMExtractionError):
        validate_scores(d)


def test_validate_rejects_non_numeric():
    with pytest.raises(LLMExtractionError):
        validate_scores(dict(GOOD, tone_numbers_gap="positive"))


def test_validate_output_has_exactly_spec_keys():
    out = validate_scores(dict(GOOD, extra_field=123))
    assert set(out) == set(FEATURE_SPEC)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def test_build_prompt_contains_schema_and_text():
    p = build_prompt("Operator: first question please.")
    for field in FEATURE_SPEC:
        assert field in p
    assert "first question please" in p


def test_truncate_words():
    text = " ".join(["word"] * 6000)
    out = truncate_words(text, max_words=100)
    assert len(out.split()) == 101          # 100 words + [TRUNCATED]
    assert out.endswith("[TRUNCATED]")
    assert truncate_words("short text") == "short text"


# ---------------------------------------------------------------------------
# score_qa end-to-end with a fake provider
# ---------------------------------------------------------------------------
class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.responses.pop(0)


def test_score_qa_happy_path():
    fp = FakeProvider([json.dumps(GOOD)])
    assert score_qa(fp, "some qa text") == GOOD
    assert fp.calls == 1


def test_score_qa_retries_once_then_succeeds():
    fp = FakeProvider(["not json at all", json.dumps(GOOD)])
    assert score_qa(fp, "some qa text") == GOOD
    assert fp.calls == 2


def test_score_qa_fails_after_retry():
    fp = FakeProvider(["nope", "still nope"])
    with pytest.raises(LLMExtractionError):
        score_qa(fp, "some qa text")
    assert fp.calls == 2
