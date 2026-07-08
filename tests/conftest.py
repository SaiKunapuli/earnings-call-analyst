"""Test fixtures shared across test modules."""

import pandas as pd
import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# Sample texts for sentiment testing
# ---------------------------------------------------------------------------

@pytest.fixture
def positive_text() -> str:
    return """The company delivered record revenue this quarter with strong
    growth across all business segments. We are very optimistic about the
    pipeline and expect continued margin expansion. Customer satisfaction
    remains at an all-time high."""

@pytest.fixture
def negative_text() -> str:
    return """The company missed expectations this quarter due to supply chain
    disruptions and declining customer demand. We are reducing guidance and
    implementing cost-cutting measures. The competitive landscape remains
    challenging and uncertain."""

@pytest.fixture
def neutral_text() -> str:
    return """The quarterly results were in line with expectations. Revenue
    was flat compared to the prior quarter. Management will provide additional
    details during the conference call at 5:00 PM Eastern."""

@pytest.fixture
def mixed_text() -> str:
    return """We are pleased with the strong revenue growth this quarter,
    but operating margins declined due to increased investment in R&D.
    The pipeline is robust, yet macroeconomic headwinds persist. We remain
    cautiously optimistic about the second half of the year."""

@pytest.fixture
def boilerplate_text() -> str:
    return """Image source: Getty Images. This article was originally published
    on The Motley Fool. The Motley Fool has a disclosure policy.
    Should you invest $1,000 in Microsoft right now?
    Before you buy stock in Microsoft, consider this: The Motley Fool Stock
    Advisor analyst team just identified what they believe are the 10 best
    stocks for investors to buy now… and Microsoft wasn't one of them.
    See the 10 stocks. When our analyst team has a stock tip, it can pay to
    listen. © 2024 All rights reserved.

    The company reported strong earnings this quarter with revenue up 15%.
    Management highlighted cloud growth as a key driver of performance.
    Operating income increased 12% year over year."""

@pytest.fixture
def short_text() -> str:
    return "Strong quarter. Revenue up."

@pytest.fixture
def empty_text() -> str:
    return ""


# ---------------------------------------------------------------------------
# Synthetic DataFrames for join testing
# ---------------------------------------------------------------------------

@pytest.fixture
def tx_df() -> pd.DataFrame:
    """Synthetic transcripts metadata (mimics the DB ``transcripts`` table)."""
    return pd.DataFrame(
        {
            "ticker": ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL"],
            "quarter": ["q1", "q2", "q3", "q1", "q2"],
            "year": [2024, 2024, 2024, 2024, 2024],
            "pub_date": pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-04-25",
                    "2024-07-30",
                    "2024-02-01",
                    "2024-05-02",
                ]
            ),
            "word_count": [5000, 6000, 5500, 4000, 4200],
        }
    )


@pytest.fixture
def returns_df() -> pd.DataFrame:
    """Synthetic returns data (mimics the DB ``returns`` table)."""
    return pd.DataFrame(
        {
            "ticker": [
                "MSFT", "MSFT", "MSFT", "MSFT",
                "AAPL", "AAPL", "AAPL", "AAPL",
            ],
            "earnings_date": pd.to_datetime(
                [
                    "2024-01-30", "2024-04-25", "2024-07-30", "2024-10-22",
                    "2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31",
                ]
            ),
            "return_1d":  [0.01, -0.02, 0.03, 0.01, 0.02, -0.01, 0.01, 0.02],
            "return_30d": [0.05, -0.03, 0.08, 0.02, 0.06, -0.02, 0.04, 0.03],
            "return_90d": [0.10, -0.05, 0.12, 0.04, 0.08, -0.03, 0.06, 0.05],
            "abnormal_1d":  [0.005, -0.025, 0.025, 0.005, 0.015, -0.015, 0.005, 0.015],
            "abnormal_30d": [0.03, -0.04, 0.06, 0.01, 0.04, -0.03, 0.02, 0.02],
            "abnormal_90d": [0.05, -0.06, 0.08, 0.02, 0.05, -0.04, 0.03, 0.03],
            "vix_close": [15.0, 18.0, 16.0, 20.0, 15.0, 18.0, 16.0, 20.0],
            "is_covid": [False, False, False, False, False, False, False, False],
        }
    )


@pytest.fixture
def df_sent() -> pd.DataFrame:
    """Synthetic sentiment DataFrame (mimics ``df_sent`` after all
    sentiment features have been computed)."""
    return pd.DataFrame(
        {
            "ticker": ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL"],
            "quarter": ["q1", "q2", "q3", "q1", "q2"],
            "year": [2024, 2024, 2024, 2024, 2024],
            "vader_compound": [0.42, 0.35, 0.51, 0.28, 0.33],
            "finbert_net": [0.15, 0.10, 0.20, 0.08, 0.12],
            "lm_net": [0.05, 0.03, 0.06, 0.02, 0.04],
        }
    )
