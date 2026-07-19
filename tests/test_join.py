"""Unit tests for src/join.py — transcript-to-returns matching logic."""

import pandas as pd
import pytest
from datetime import datetime

from src.join import (
    match_transcript_to_returns,
    join_sentiment_to_returns,
    DEFAULT_MAX_DAY_DIFF,
)


# ============================================================================
# match_transcript_to_returns
# ============================================================================

class TestMatchTranscriptToReturns:
    def test_exact_date_match(self, tx_df, returns_df, df_sent):
        """Transcript pub_date equals earnings_date → exact match."""
        row = df_sent.iloc[0]  # MSFT, q1, 2024
        result, unmatched = match_transcript_to_returns(row, tx_df, returns_df)
        assert not unmatched
        assert result is not None
        assert result["day_diff"] == 0
        assert result["ticker"] == "MSFT"
        assert "abnormal_30d" in result
        assert "vix_close" in result

    def test_nearby_date_match(self, tx_df, df_sent):
        """Earnings date within a few days of pub_date still matches."""
        # Build returns where the closest date is 3 days off
        custom_returns = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "earnings_date": pd.to_datetime(["2024-01-27"]),
                "return_1d": [0.01], "return_30d": [0.05], "return_90d": [0.10],
                "abnormal_1d": [0.005], "abnormal_30d": [0.03], "abnormal_90d": [0.05],
                "vix_close": [15.0], "is_covid": [False],
            }
        )
        row = df_sent.iloc[0]  # MSFT q1 2024, pub_date 2024-01-30
        result, unmatched = match_transcript_to_returns(
            row, tx_df, custom_returns
        )
        assert not unmatched
        assert result["day_diff"] == 3

    def test_beyond_max_day_diff(self, tx_df, df_sent):
        """Returns date is > 30 days away → unmatched."""
        custom_returns = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "earnings_date": pd.to_datetime(["2024-03-15"]),
                "return_1d": [0.01], "return_30d": [0.05], "return_90d": [0.10],
                "abnormal_1d": [0.005], "abnormal_30d": [0.03], "abnormal_90d": [0.05],
                "vix_close": [15.0], "is_covid": [False],
            }
        )
        row = df_sent.iloc[0]  # pub_date 2024-01-30, earnings 2024-03-15 = 45 days
        result, unmatched = match_transcript_to_returns(
            row, tx_df, custom_returns
        )
        assert unmatched
        assert result is None

    def test_custom_max_day_diff(self, tx_df, df_sent):
        """User can override max_day_diff."""
        custom_returns = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "earnings_date": pd.to_datetime(["2024-03-15"]),
                "return_1d": [0.01], "return_30d": [0.05], "return_90d": [0.10],
                "abnormal_1d": [0.005], "abnormal_30d": [0.03], "abnormal_90d": [0.05],
                "vix_close": [15.0], "is_covid": [False],
            }
        )
        row = df_sent.iloc[0]
        # 45 days is within a 60-day limit
        result, unmatched = match_transcript_to_returns(
            row, tx_df, custom_returns, max_day_diff=60
        )
        assert not unmatched

    def test_no_returns_for_ticker(self, tx_df, df_sent):
        """Ticker not present in returns_df → unmatched."""
        returns_no_appl = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "earnings_date": pd.to_datetime(["2024-01-30"]),
                "return_1d": [0.01], "return_30d": [0.05], "return_90d": [0.10],
                "abnormal_1d": [0.005], "abnormal_30d": [0.03], "abnormal_90d": [0.05],
                "vix_close": [15.0], "is_covid": [False],
            }
        )
        row = df_sent.iloc[3]  # AAPL
        result, unmatched = match_transcript_to_returns(
            row, tx_df, returns_no_appl
        )
        assert unmatched

    def test_no_transcript_metadata(self, returns_df, df_sent):
        """Transcript row not found in tx_df → unmatched."""
        custom_tx = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "quarter": ["q1"],
                "year": [2024],
                "pub_date": pd.to_datetime(["2024-02-01"]),
                "word_count": [4000],
            }
        )
        row = df_sent.iloc[0]  # MSFT, but tx_df has only AAPL
        result, unmatched = match_transcript_to_returns(
            row, custom_tx, returns_df
        )
        assert unmatched

    def test_picks_closest_when_multiple(self, tx_df):
        """When multiple earnings dates exist for a ticker, pick the closest."""
        multi_returns = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "earnings_date": pd.to_datetime(["2024-01-28", "2024-02-05"]),
                "return_1d": [0.01, 0.02], "return_30d": [0.05, 0.06],
                "return_90d": [0.10, 0.11],
                "abnormal_1d": [0.005, 0.006], "abnormal_30d": [0.03, 0.04],
                "abnormal_90d": [0.05, 0.06],
                "vix_close": [15.0, 16.0], "is_covid": [False, False],
            }
        )
        row = pd.Series({"ticker": "MSFT", "quarter": "q1", "year": 2024})
        result, unmatched = match_transcript_to_returns(
            row, tx_df, multi_returns
        )
        assert not unmatched
        # pub_date is 2024-01-30 → closest is 2024-01-28 (2 days) vs 2024-02-05 (6 days)
        assert result["day_diff"] == 2
        assert result["vix_close"] == 15.0

    def test_preserves_sentiment_fields(self, tx_df, returns_df, df_sent):
        """Merged result should carry forward all sentiment columns."""
        row = df_sent.iloc[1]  # MSFT q2 2024
        result, unmatched = match_transcript_to_returns(row, tx_df, returns_df)
        assert not unmatched
        assert result["vader_compound"] == 0.35
        assert result["finbert_net"] == 0.10
        assert result["lm_net"] == 0.03


# ============================================================================
# join_sentiment_to_returns
# ============================================================================

class TestJoinSentimentToReturns:
    def test_batch_match(self, tx_df, returns_df, df_sent):
        df_merged, unmatched = join_sentiment_to_returns(
            df_sent, tx_df, returns_df
        )
        # All 5 rows should match (perfect date alignments in fixtures)
        assert len(df_merged) == 5
        assert unmatched == 0
        assert "abnormal_30d" in df_merged.columns
        assert "vader_compound" in df_merged.columns

    def test_partial_match(self, tx_df, returns_df, df_sent):
        """When some transcripts don't have returns data."""
        # Remove one ticker from returns
        partial_returns = returns_df[returns_df["ticker"] != "AAPL"].copy()
        df_merged, unmatched = join_sentiment_to_returns(
            df_sent, tx_df, partial_returns
        )
        assert len(df_merged) == 3  # Only MSFT rows
        assert unmatched == 2         # 2 AAPL rows unmatched

    def test_empty_sentiment(self, tx_df, returns_df):
        empty_sent = pd.DataFrame(
            columns=["ticker", "quarter", "year", "vader_compound"]
        )
        df_merged, unmatched = join_sentiment_to_returns(
            empty_sent, tx_df, returns_df
        )
        assert len(df_merged) == 0
        assert unmatched == 0

    def test_empty_returns(self, tx_df, df_sent):
        empty_returns = pd.DataFrame(
            columns=[
                "ticker", "earnings_date",
                "return_1d", "return_30d", "return_90d",
                "abnormal_1d", "abnormal_30d", "abnormal_90d",
                "vix_close", "is_covid",
            ]
        )
        df_merged, unmatched = join_sentiment_to_returns(
            df_sent, tx_df, empty_returns
        )
        assert len(df_merged) == 0
        assert unmatched == 5

    def test_returned_dataframe_has_expected_columns(
        self, tx_df, returns_df, df_sent
    ):
        df_merged, _ = join_sentiment_to_returns(df_sent, tx_df, returns_df)
        # Should have sentiment columns + return columns + match metadata
        assert "matched_earnings_date" in df_merged.columns
        assert "day_diff" in df_merged.columns
        assert "abnormal_30d" in df_merged.columns


# ============================================================================
# Edge cases & regression
# ============================================================================

class TestJoinEdgeCases:
    def test_single_row_each(self, tx_df, returns_df):
        """Minimal inputs: one sentiment row, one return row, one tx row."""
        single_sent = pd.DataFrame(
            {"ticker": ["MSFT"], "quarter": ["q1"], "year": [2024]}
        )
        df_merged, unmatched = join_sentiment_to_returns(
            single_sent, tx_df, returns_df
        )
        assert len(df_merged) == 1
        assert unmatched == 0

    def test_default_max_day_diff(self):
        """Verify DEFAULT_MAX_DAY_DIFF is 30."""
        assert DEFAULT_MAX_DAY_DIFF == 30
