"""Transcript-to-returns matching logic.

Extracted from 03_sentiment.ipynb cell ``sen_join_returns_001`` so the
join algorithm can be unit-tested without a running notebook kernel or
live SQLite database.
"""

import pandas as pd

# Maximum day difference allowed between transcript pub_date and
# the closest earnings_date for a match to be accepted.
DEFAULT_MAX_DAY_DIFF = 30


def match_transcript_to_returns(
    tx_row: pd.Series,
    tx_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    max_day_diff: int = DEFAULT_MAX_DAY_DIFF,
) -> tuple[dict | None, bool]:
    """Match a single transcript row to its nearest earnings-return event.

    Parameters
    ----------
    tx_row : pd.Series
        One row from the sentiment DataFrame (must contain 'ticker',
        'quarter', 'year').
    tx_df : pd.DataFrame
        The original transcripts metadata DataFrame (must contain
        'ticker', 'quarter', 'year', 'pub_date').
    returns_df : pd.DataFrame
        Returns data (must contain 'ticker', 'earnings_date', and all
        return/abnormal/vix/is_covid columns).
    max_day_diff : int
        Maximum allowed day difference between pub_date and
        earnings_date.  Default: 30.

    Returns
    -------
    (row_dict, was_unmatched)
        ``row_dict`` is the merged dict on success, ``None`` on failure.
        ``was_unmatched`` is ``True`` when no match could be found.
    """
    ticker = tx_row["ticker"]

    # ---- filter returns to this ticker ------------------------------------
    ticker_returns = returns_df[returns_df["ticker"] == ticker].copy()
    if ticker_returns.empty:
        return None, True

    # ---- find the transcript metadata row ---------------------------------
    matched_tx = tx_df[
        (tx_df["ticker"] == ticker)
        & (tx_df["quarter"] == tx_row["quarter"])
        & (tx_df["year"] == tx_row["year"])
    ]
    if matched_tx.empty:
        return None, True

    pub_date = matched_tx.iloc[0]["pub_date"]

    # ---- find the closest earnings date -----------------------------------
    ticker_returns["day_diff"] = abs(
        (ticker_returns["earnings_date"] - pub_date).dt.days
    )
    closest = ticker_returns.loc[ticker_returns["day_diff"].idxmin()]

    if closest["day_diff"] > max_day_diff:
        return None, True

    # ---- build merged row -------------------------------------------------
    row_dict = {k: v for k, v in tx_row.items()}
    row_dict.update(
        {
            "matched_earnings_date": closest["earnings_date"],
            "day_diff": closest["day_diff"],
            "return_1d": closest["return_1d"],
            "return_30d": closest["return_30d"],
            "return_90d": closest["return_90d"],
            "abnormal_1d": closest["abnormal_1d"],
            "abnormal_30d": closest["abnormal_30d"],
            "abnormal_90d": closest["abnormal_90d"],
            "vix_close": closest["vix_close"],
            "is_covid": closest["is_covid"],
        }
    )
    return row_dict, False


def join_sentiment_to_returns(
    df_sent: pd.DataFrame,
    tx_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    max_day_diff: int = DEFAULT_MAX_DAY_DIFF,
) -> tuple[pd.DataFrame, int]:
    """Join sentiment features to returns data by nearest earnings date.

    Parameters
    ----------
    df_sent : pd.DataFrame
        Sentiment features (must contain 'ticker', 'quarter', 'year').
    tx_df : pd.DataFrame
        Transcripts metadata (must contain 'ticker', 'quarter', 'year',
        'pub_date').
    returns_df : pd.DataFrame
        Returns data (must contain 'ticker', 'earnings_date', and all
        return columns).
    max_day_diff : int
        Maximum allowed day difference.  Default: 30.

    Returns
    -------
    (df_merged, unmatched_count)
        ``df_merged`` contains the successfully joined rows.
        ``unmatched_count`` is the number of rows that could not be
        matched.
    """
    joined: list[dict] = []
    unmatched = 0

    for _, tx_row in df_sent.iterrows():
        row_dict, was_unmatched = match_transcript_to_returns(
            tx_row, tx_df, returns_df, max_day_diff
        )
        if was_unmatched:
            unmatched += 1
            continue
        if row_dict is not None:
            joined.append(row_dict)

    if joined:
        df_merged = pd.DataFrame(joined)
    else:
        df_merged = pd.DataFrame()

    return df_merged, unmatched
