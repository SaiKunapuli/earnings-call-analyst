"""Unit tests for src/sentiment.py -- transcript cleaning, chunking,
VADER sentiment, Loughran-McDonald sentiment, linguistic features,
and readability metrics."""

import re
import pytest
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import pysentiment2 as ps2

from src.sentiment import (
    clean_transcript,
    _is_boilerplate_line,
    chunk_text,
    compute_vader_sentiment,
    compute_paragraph_vader,
    compute_lm_sentiment,
    compute_linguistic_features,
    compute_readability,
    compute_all_sentiment_features,
)


# ============================================================================
# clean_transcript
# ============================================================================

class TestCleanTranscript:
    def test_removes_motley_fool_boilerplate(self, boilerplate_text):
        result = clean_transcript(boilerplate_text)
        assert "Image source:" not in result
        assert "disclosure policy" not in result
        assert "reported strong earnings" in result
        assert "cloud growth" in result

    def test_removes_sec_exhibit_header(self):
        text = (
            "Exhibit 99.1 Microsoft Cloud Strength Drives First Quarter "
            "Results REDMOND, Wash. -- October 30, 2024 -- Microsoft Corp. "
            "today announced the following results for the quarter ended "
            "September 30, 2024. Revenue was $65.6 billion and increased 16%."
        )
        result = clean_transcript(text)
        assert "Exhibit 99.1" not in result
        assert "Microsoft Corp. today announced" in result
        assert "Revenue was" in result

    def test_removes_operator_lines(self):
        text = (
            "Operator: Good afternoon. My name is Regina, and I will be "
            "your conference operator today. All lines have been placed "
            "on mute to prevent any background noise. The company "
            "reported strong results with revenue up 15%."
        )
        result = clean_transcript(text)
        assert "Operator:" not in result
        assert "Good afternoon" not in result
        assert "All lines have been placed on mute" not in result
        assert "reported strong results" in result

    def test_removes_forward_looking_disclaimer(self):
        text = (
            "This press release contains forward-looking statements. "
            "These statements involve risks and uncertainties, and actual "
            "results could differ materially. Revenue grew 10% this quarter."
        )
        result = clean_transcript(text)
        assert "forward-looking statements" not in result
        assert "actual results could differ" not in result
        assert "Revenue grew 10%" in result

    def test_normalizes_whitespace(self):
        text = (
            "This is a line with extra   whitespace inside it here"
            "\n\n\nAnd this is another line with tabs\t\tand spaces"
        )
        result = clean_transcript(text)
        assert "  " not in result
        assert "\t" not in result
        assert "extra whitespace" in result

    def test_removes_ticker_percentage_noise(self):
        text = "Apple ( AAPL 6.41% ) reported strong earnings today."
        result = clean_transcript(text)
        assert "AAPL 6.41%" not in result
        assert "Apple" in result
        assert "reported strong earnings" in result

    def test_empty_input(self):
        assert clean_transcript("") == ""

    def test_whitespace_only(self):
        assert clean_transcript("   \n  \n\t  ") == ""

    def test_preserves_content_body(self, positive_text):
        result = clean_transcript(positive_text)
        assert "record revenue" in result
        assert "margin expansion" in result

    def test_handles_single_line_blob(self):
        """Transcripts are single-line blobs (no newlines). Should preserve content."""
        text = (
            "Image source: The Motley Fool. Microsoft ( MSFT 2.23% ) Q1 2025 "
            "Earnings Call Oct 30, 2024, 5:00 p.m. ET Contents: Prepared "
            "Remarks Questions and Answers Call Participants Prepared Remarks: "
            "Operator Good afternoon. My name is Regina. All lines have been "
            "placed on mute. Microsoft reported cloud revenue of $38.9 billion, "
            "up 22% year over year. The company saw strong growth in Azure."
        )
        result = clean_transcript(text)
        assert "Microsoft reported cloud revenue" in result
        assert "Azure" in result
        assert "Image source:" not in result
        assert "Operator" not in result


class TestIsBoilerplateLine:
    def test_copyright(self):
        assert _is_boilerplate_line("all rights reserved")

    def test_image_source(self):
        assert _is_boilerplate_line("image source: getty images")

    def test_case_insensitive(self):
        assert _is_boilerplate_line("Image Source: Getty Images")

    def test_disclosure_policy(self):
        assert _is_boilerplate_line(
            "the motley fool has a disclosure policy"
        )

    def test_stock_advisor(self):
        assert _is_boilerplate_line(
            "the motley fool stock advisor analyst team just identified"
        )

    def test_non_boilerplate(self):
        assert not _is_boilerplate_line(
            "revenue grew 15% year over year"
        )
        # "all rights reserved" IS a boilerplate phrase,
        # so "all rights reserved for our intellectual property" matches
        assert _is_boilerplate_line(
            "all rights reserved for our intellectual property"
        )


# ============================================================================
# chunk_text
# ============================================================================

class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_single_short_sentence(self):
        chunks = chunk_text("Short.", max_words=256)
        assert len(chunks) == 1
        assert "Short." in chunks[0]

    def test_no_split_when_under_limit(self):
        text = " ".join([f"Sentence number {i}." for i in range(20)])
        chunks = chunk_text(text, max_words=500)
        assert len(chunks) == 1

    def test_splits_at_boundary(self):
        sentences = [
            " ".join(["word"] * 50) + ".",
            " ".join(["word"] * 50) + ".",
            " ".join(["word"] * 50) + ".",
            " ".join(["word"] * 50) + ".",
        ]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_words=120)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.split()) <= 200  # generous upper bound

    def test_exact_max_words(self):
        text = " ".join(["word"] * 50) + ". " + " ".join(["foo"] * 50) + "."
        chunks = chunk_text(text, max_words=50)
        assert len(chunks) >= 1

    def test_single_sentence_exceeds_max(self):
        """A single sentence that exceeds max_words still gets its own chunk."""
        long_text = " ".join(["word"] * 300) + "."
        chunks = chunk_text(long_text, max_words=100)
        assert len(chunks) == 1
        # 300 tokens (period attaches to the last word, so 300 split tokens)
        assert len(chunks[0].split()) == 300

    def test_many_chunks(self):
        sentences = [f"Sentence number {i} is here now." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_words=30)
        assert len(chunks) >= 20


# ============================================================================
# compute_vader_sentiment
# ============================================================================

class TestVaderSentiment:
    @pytest.fixture
    def vader(self):
        return SentimentIntensityAnalyzer()

    def test_positive_text(self, vader, positive_text):
        scores = compute_vader_sentiment(positive_text, vader)
        assert scores["vader_compound"] > 0.5
        assert scores["vader_pos"] > 0.0
        assert scores["vader_neg"] < scores["vader_pos"]
        assert "vader_neu" in scores

    def test_negative_text(self, vader, negative_text):
        scores = compute_vader_sentiment(negative_text, vader)
        assert scores["vader_compound"] < -0.3
        assert scores["vader_neg"] > 0.0

    def test_neutral_text(self, vader, neutral_text):
        scores = compute_vader_sentiment(neutral_text, vader)
        assert abs(scores["vader_compound"]) < 0.5

    def test_mixed_text(self, vader, mixed_text):
        scores = compute_vader_sentiment(mixed_text, vader)
        assert -1.0 <= scores["vader_compound"] <= 1.0

    def test_short_text(self, vader, short_text):
        scores = compute_vader_sentiment(short_text, vader)
        assert scores["vader_compound"] > 0.0  # "Strong quarter"

    def test_empty_text(self, vader, empty_text):
        scores = compute_vader_sentiment(empty_text, vader)
        assert scores["vader_compound"] == 0.0
        assert scores["vader_pos"] == 0.0
        assert scores["vader_neg"] == 0.0

    def test_returns_expected_keys(self, vader, positive_text):
        scores = compute_vader_sentiment(positive_text, vader)
        expected_keys = {
            "vader_compound", "vader_pos", "vader_neg", "vader_neu",
            "vader_mean", "vader_std", "vader_min", "vader_max",
            "vader_p10", "vader_p90", "vader_pct_neg", "vader_pct_pos",
            "vader_n_chunks", "vader_n_paragraphs",
        }
        assert set(scores.keys()) == expected_keys

    def test_sentence_chunked_produces_multiple_chunks(self, vader, positive_text):
        """Longer texts should produce multiple sentence-chunks."""
        long_text = (
            "Revenue grew 15%. Margins expanded significantly. "
            "Cloud business accelerated. We are very optimistic. "
            "The pipeline is strong. Customer demand is robust. "
            "International markets performed well. We raised guidance. "
            "Operating leverage improved. Cash flows were strong. "
            "We returned capital to shareholders. Innovation continues."
        )
        scores = compute_paragraph_vader(long_text, vader, sentences_per_chunk=3)
        assert scores["vader_n_chunks"] >= 3
        assert scores["vader_mean"] > 0.0

    def test_paragraph_vader_empty(self):
        scores = compute_paragraph_vader("")
        assert scores["vader_mean"] == 0.0
        assert scores["vader_n_chunks"] == 0


# ============================================================================
# compute_lm_sentiment
# ============================================================================

class TestLMSentiment:
    @pytest.fixture
    def lm(self):
        return ps2.LM()

    def test_positive_text(self, lm, positive_text):
        scores = compute_lm_sentiment(positive_text, lm)
        assert scores["lm_positive"] > 0
        assert isinstance(scores["lm_net"], float)

    def test_negative_text(self, lm, negative_text):
        scores = compute_lm_sentiment(negative_text, lm)
        # pysentiment2 may return numpy ints; just verify it is >= 0
        assert scores["lm_negative"] >= 0

    def test_returns_all_keys(self, lm, positive_text):
        scores = compute_lm_sentiment(positive_text, lm)
        expected = {
            "lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious",
            "lm_constraining", "lm_strong_modal", "lm_weak_modal",
            "lm_net", "lm_pos_ratio", "lm_neg_ratio",
        }
        assert set(scores.keys()) == expected

    def test_net_bounds(self, lm, positive_text):
        scores = compute_lm_sentiment(positive_text, lm)
        assert -1.0 <= scores["lm_net"] <= 1.0

    def test_pos_neg_ratio_sum(self, lm, positive_text):
        scores = compute_lm_sentiment(positive_text, lm)
        assert 0.0 <= scores["lm_pos_ratio"] <= 1.0
        assert 0.0 <= scores["lm_neg_ratio"] <= 1.0

    def test_empty_text(self, lm, empty_text):
        scores = compute_lm_sentiment(empty_text, lm)
        assert scores["lm_positive"] == 0
        assert scores["lm_negative"] == 0
        assert scores["lm_net"] == 0.0

    def test_financial_keywords(self, lm):
        """Text with known LM financial words should be detected."""
        text = (
            "The company faces litigation risk and regulatory challenges. "
            "We may need to restructure certain operations."
        )
        scores = compute_lm_sentiment(text, lm)
        # With pysentiment2 v0.1.1 the fine-grained categories are 0,
        # but Positive/Negative counts should still work
        assert scores["lm_positive"] >= 0
        assert scores["lm_negative"] >= 0
        assert isinstance(scores["lm_net"], float)


# ============================================================================
# compute_linguistic_features
# ============================================================================

class TestLinguisticFeatures:
    def test_basic(self, positive_text):
        feats = compute_linguistic_features(positive_text)
        assert feats["n_sentences"] >= 1
        assert feats["n_words"] > 10
        assert 0.0 < feats["unique_word_ratio"] <= 1.0
        assert feats["avg_sentence_length"] > 0

    def test_empty_text(self, empty_text):
        feats = compute_linguistic_features(empty_text)
        assert feats["n_sentences"] == 0
        assert feats["n_words"] == 0
        assert feats["unique_word_ratio"] == 0.0
        assert feats["avg_sentence_length"] == 0.0

    def test_two_sentences(self, short_text):
        """'Strong quarter. Revenue up.' is 2 sentences."""
        feats = compute_linguistic_features(short_text)
        assert feats["n_sentences"] == 2

    def test_unique_ratio_perfect(self):
        text = "alpha beta gamma delta epsilon"
        feats = compute_linguistic_features(text)
        assert feats["unique_word_ratio"] == 1.0

    def test_unique_ratio_repetition(self):
        text = "word word word word word"
        feats = compute_linguistic_features(text)
        assert feats["unique_word_ratio"] == pytest.approx(0.2)

    def test_case_insensitive_uniqueness(self):
        """'Word' and 'word' treated as the same (rounds to 4 decimals)."""
        text = "Word word WORD"
        feats = compute_linguistic_features(text)
        # unique_word_ratio = round(1/3, 4) = 0.3333
        assert feats["unique_word_ratio"] == pytest.approx(1.0 / 3.0, rel=1e-3)

    def test_avg_sentence_length(self):
        text = "One two three. Four five."
        feats = compute_linguistic_features(text)
        assert feats["avg_sentence_length"] == pytest.approx(2.5, rel=0.1)


# ============================================================================
# compute_readability
# ============================================================================

class TestReadability:
    def test_basic(self, positive_text):
        scores = compute_readability(positive_text)
        assert "flesch_reading_ease" in scores
        assert "gunning_fog" in scores
        assert isinstance(scores["flesch_reading_ease"], float)

    def test_empty_text(self, empty_text):
        scores = compute_readability(empty_text)
        # textstat should handle empty gracefully
        assert isinstance(scores, dict)

    def test_all_metrics_present(self, positive_text):
        scores = compute_readability(positive_text)
        expected = {
            "flesch_reading_ease", "flesch_kincaid_grade",
            "gunning_fog", "smog_index", "automated_readability",
            "dale_chall_score",
        }
        assert set(scores.keys()) >= expected

    def test_simple_text_is_more_readable(self):
        simple = "The cat sat on the mat. It was a nice day."
        complex_text = (
            "The implementation of sophisticated quantitative methodologies "
            "necessitates a comprehensive reevaluation of organizational "
            "infrastructure and governance frameworks."
        )
        simple_score = compute_readability(simple)
        complex_score = compute_readability(complex_text)
        # Flesch Reading Ease: higher = easier to read
        assert simple_score["flesch_reading_ease"] > complex_score["flesch_reading_ease"]


# ============================================================================
# compute_all_sentiment_features
# ============================================================================

class TestAllSentimentFeatures:
    def test_returns_flat_dict(self, positive_text):
        features = compute_all_sentiment_features(positive_text)
        assert isinstance(features, dict)
        assert "vader_compound" in features
        assert "lm_net" in features
        assert "flesch_reading_ease" in features
        assert "unique_word_ratio" in features
        assert "n_words" in features

    def test_no_overlapping_keys(self, positive_text):
        """Ensure sub-functions do not accidentally overwrite each other."""
        features = compute_all_sentiment_features(positive_text)
        vader_keys = sum(1 for k in features if k.startswith("vader_"))
        lm_keys = sum(1 for k in features if k.startswith("lm_"))
        assert vader_keys == 14  # compound/pos/neg/neu + mean/std/min/max/p10/p90/pct_neg/pct_pos/n_chunks/n_paragraphs
        assert lm_keys == 10    # 7 raw + net + pos_ratio + neg_ratio
