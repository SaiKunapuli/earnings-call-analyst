"""Unit tests for src/transcripts_io.py — sectioning, stats, storage."""

import sqlite3

import pytest

from src.transcripts_io import (
    compress_text,
    decompress_text,
    normalize_turns,
    find_qa_boundary,
    split_turns,
    open_db,
    make_row,
    write_batch,
    fetch_sections,
    META_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def call_turns() -> list[dict]:
    """A miniature earnings call with prepared remarks and Q&A."""
    long_answer = (
        "I think revenue growth was driven by cloud adoption and, sort of, "
        "continued enterprise demand. It depends on macro conditions but we "
        "believe the pipeline remains strong. " + "Detail. " * 30
    )
    return [
        {"speaker": "Operator", "text": "Good afternoon, welcome to the call."},
        {"speaker": "Jane Smith", "text": "Thank you. Revenue grew 15% this quarter with strong margins. " * 5},
        {"speaker": "John CFO", "text": "Operating income was up 12% and cash flow was robust. " * 5},
        {"speaker": "Operator", "text": "We will now begin the question-and-answer session."},
        {"speaker": "Analyst Bob", "text": "Can you explain what drove the revenue growth this quarter?"},
        {"speaker": "Jane Smith", "text": long_answer},
        {"speaker": "Analyst Alice", "text": "How should we think about margins next year?"},
        {"speaker": "John CFO", "text": "Too early to say. We don't guide on segment margins. " * 5},
    ]


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

class TestCompression:
    def test_roundtrip(self):
        text = "Revenue grew 15% — margins expanded. " * 100
        assert decompress_text(compress_text(text)) == text

    def test_compresses(self):
        text = "Revenue grew 15% year over year. " * 500
        assert len(compress_text(text)) < len(text.encode()) / 3

    def test_empty(self):
        assert decompress_text(compress_text("")) == ""
        assert decompress_text(None) == ""
        assert decompress_text(b"") == ""


# ---------------------------------------------------------------------------
# normalize_turns
# ---------------------------------------------------------------------------

class TestNormalizeTurns:
    def test_list_of_dicts(self, call_turns):
        turns = normalize_turns(call_turns)
        assert len(turns) == len(call_turns)
        assert [t["speaker"] for t in turns] == [t["speaker"] for t in call_turns]
        # text preserved modulo whitespace stripping
        assert turns[1]["text"] == call_turns[1]["text"].strip()

    def test_json_string(self):
        turns = normalize_turns('[{"speaker": "CEO", "text": "Hello."}]')
        assert turns == [{"speaker": "CEO", "text": "Hello."}]

    def test_bad_input(self):
        assert normalize_turns(None) == []
        assert normalize_turns("not json{") == []
        assert normalize_turns(42) == []
        assert normalize_turns([{"speaker": "X", "text": "  "}]) == []

    def test_missing_speaker(self):
        turns = normalize_turns([{"text": "Anonymous remark."}])
        assert turns[0]["speaker"] == ""


# ---------------------------------------------------------------------------
# Q&A boundary + section split
# ---------------------------------------------------------------------------

class TestSplitTurns:
    def test_boundary_after_operator_announcement(self, call_turns):
        assert find_qa_boundary(call_turns) == 4

    def test_no_qa(self):
        turns = [
            {"speaker": "CEO", "text": "Revenue grew 15%. " * 20},
            {"speaker": "CFO", "text": "Margins expanded. " * 20},
        ]
        assert find_qa_boundary(turns) == 2
        sections = split_turns(turns)
        assert sections["has_qa"] == 0
        assert sections["qa_text"] == ""
        assert "Revenue grew" in sections["full_text"]

    def test_sections_and_stats(self, call_turns):
        s = split_turns(call_turns)
        assert s["has_qa"] == 1
        assert "Revenue grew 15%" in s["prepared_text"]
        assert "what drove the revenue growth" in s["qa_text"]
        assert "what drove the revenue growth" not in s["prepared_text"]
        # Two analysts asked one question each
        assert s["n_analyst_speakers"] == 2
        assert s["n_analyst_questions"] == 2
        # Jane and John are execs (spoke in prepared remarks)
        assert s["n_exec_speakers"] == 2
        assert s["exec_a_words"] > s["analyst_q_words"]
        assert s["answer_question_ratio"] > 1.0
        # Hedging present ("I think", "sort of", "It depends", "Too early to", ...)
        assert s["hedge_per_1k"] > 0

    def test_empty(self):
        s = split_turns([])
        assert s["has_qa"] == 0
        assert s["full_text"] == ""
        assert s["answer_question_ratio"] == 0.0
        assert s["hedge_per_1k"] == 0.0


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------

class TestStorage:
    def test_write_and_fetch(self, tmp_path, call_turns):
        db = tmp_path / "t.db"
        conn = open_db(db)
        sections = split_turns(call_turns)
        row = make_row("MSFT", "Q1", 2024, "2024-01-30", "hf", sections)
        write_batch(conn, [row])

        # quarter normalized to lowercase
        got = fetch_sections(conn, "MSFT", "q1", 2024)
        assert set(got) == {"prepared_remarks", "qa", "full"}
        assert "Revenue grew 15%" in got["prepared_remarks"]
        assert "margins next year" in got["qa"]

        # metadata columns exist and BLOBs are excluded from META_COLS
        cols = [r[1] for r in conn.execute("PRAGMA table_info(transcripts_text)")]
        assert set(META_COLS) <= set(cols)
        assert not any(c.endswith("_z") for c in META_COLS)
        conn.close()

    def test_fetch_missing(self, tmp_path):
        conn = open_db(tmp_path / "t.db")
        assert fetch_sections(conn, "AAPL", "q1", 2024) == {}
        conn.close()

    def test_replace_on_conflict(self, tmp_path, call_turns):
        db = tmp_path / "t.db"
        conn = open_db(db)
        sections = split_turns(call_turns)
        write_batch(conn, [make_row("MSFT", "q1", 2024, "2024-01-30", "edgar", sections)])
        write_batch(conn, [make_row("MSFT", "q1", 2024, "2024-01-30", "hf", sections)])
        n, src = conn.execute(
            "SELECT COUNT(*), MAX(source) FROM transcripts_text"
        ).fetchone()
        assert n == 1
        assert src == "hf"
        conn.close()
