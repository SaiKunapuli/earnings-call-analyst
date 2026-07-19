"""Integration test for src/features_parallel.py — multiprocess scoring.

Uses a real (temporary) SQLite DB and a real worker process, so it also
verifies the Windows spawn path: picklable worker, per-process init,
order preservation.
"""

import pytest

from src.features_parallel import compute_sentiment_features_parallel
from src.sentiment import clean_transcript, compute_all_sentiment_features
from src.transcripts_io import (
    fetch_sections,
    make_row,
    open_db,
    split_turns,
    write_batch,
)


@pytest.fixture
def db_path(tmp_path):
    turns = [
        {"speaker": "Operator", "text": "Good afternoon, welcome to the call."},
        {"speaker": "Jane CEO", "text": "Revenue grew 15% with strong margins. " * 20},
        {"speaker": "Operator", "text": "We will now begin the question-and-answer session."},
        {"speaker": "Analyst Bob", "text": "What drove the growth this quarter?"},
        {"speaker": "Jane CEO", "text": "I think cloud adoption drove it. It depends on macro. " * 10},
    ]
    path = tmp_path / "test.db"
    conn = open_db(path)
    rows = [
        make_row("MSFT", "q1", 2024, "2024-01-30", "hf", split_turns(turns)),
        make_row("MSFT", "q2", 2024, "2024-04-25", "hf", split_turns(turns)),
        make_row("AAPL", "q1", 2024, "2024-02-01", "hf", split_turns(turns)),
    ]
    write_batch(conn, rows)
    conn.close()
    return path


def test_parallel_matches_serial(db_path):
    keys = [("MSFT", "q1", 2024), ("MSFT", "q2", 2024), ("AAPL", "q1", 2024)]
    df = compute_sentiment_features_parallel(
        db_path, keys, max_workers=1, log_every=0)

    # One row per key, in input order
    assert len(df) == 3
    assert list(zip(df["ticker"], df["quarter"], df["year"])) == keys
    assert df["has_qa"].tolist() == [1, 1, 1]

    # Per-section features present
    for col in ("full_vader_mean", "prepared_remarks_lm_net", "qa_gunning_fog"):
        assert col in df.columns, col

    # Exactly reproduces the serial code path
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    sections = fetch_sections(conn, "MSFT", "q1", 2024)
    conn.close()
    expected = {}
    for name, text in sections.items():
        expected.update(compute_all_sentiment_features(
            clean_transcript(text), prefix=name))
    row = df.iloc[0]
    for col in ("full_vader_mean", "full_lm_net", "qa_vader_mean",
                "qa_unique_word_ratio"):
        assert row[col] == pytest.approx(expected[col]), col


def test_empty_keys_returns_empty_frame(db_path):
    df = compute_sentiment_features_parallel(db_path, [], max_workers=1)
    assert df.empty
