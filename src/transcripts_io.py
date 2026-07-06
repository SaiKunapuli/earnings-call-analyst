"""Transcript ingestion, speaker-aware sectioning, and compressed storage.

Built for the HuggingFace ``kurry/sp500_earnings_transcripts`` dataset
(33k+ real earnings calls with per-speaker turns), but source-agnostic:
anything that can produce a list of ``{speaker, text}`` turns works.

Design notes (performance / resource use):
- Transcript text is stored zlib-compressed (~4-5x smaller) as BLOBs in a
  single SQLite table, NOT as 100k small files (OneDrive sync poison).
- All regexes are compiled once at module load.
- Writes are batched via executemany inside explicit transactions with
  WAL journaling — ingesting 33k transcripts is I/O-bound, not CPU-bound.
- Evasiveness/Q&A stats are computed ONCE at ingest time from the speaker
  turns, so the (expensive) sentiment notebook never has to re-derive them.
"""

from __future__ import annotations

import json
import re
import sqlite3
import zlib

# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

_ZLEVEL = 6  # good ratio/speed tradeoff for English text


def compress_text(text: str) -> bytes:
    """zlib-compress a transcript string for BLOB storage."""
    return zlib.compress(text.encode("utf-8"), _ZLEVEL)


def decompress_text(blob: bytes | None) -> str:
    """Inverse of :func:`compress_text`. None/empty-safe."""
    if not blob:
        return ""
    return zlib.decompress(blob).decode("utf-8")


# ---------------------------------------------------------------------------
# Speaker-turn sectioning
# ---------------------------------------------------------------------------

_OPERATOR_RE = re.compile(r"\boperator\b|\bcoordinator\b|\bmoderator\b", re.IGNORECASE)

# Phrases that mark the transition from prepared remarks to the Q&A session.
_QA_START_RE = re.compile(
    r"question[-\s]*and[-\s]*answer|q\s*&\s*a\s+session"
    r"|first\s+question|begin\s+the\s+question|open\s+the\s+(call|line|floor)s?\s+for\s+question"
    r"|take\s+(our|your|the)\s+first\s+question|question[-\s]*answer\s+session",
    re.IGNORECASE,
)

# Hedging / non-answer phrases used for the evasiveness score.
# Compiled as one alternation — a single scan per answer instead of N scans.
_HEDGE_RE = re.compile(
    r"\b(?:i\s+think|we\s+believe|sort\s+of|kind\s+of|it\s+depends"
    r"|too\s+early\s+to|we'?ll\s+(?:have\s+to\s+)?see|hard\s+to\s+say"
    r"|difficult\s+to\s+(?:say|predict|estimate)|not\s+going\s+to\s+(?:comment|guide|speculate)"
    r"|can'?t\s+comment|won'?t\s+comment|we\s+don'?t\s+(?:guide|disclose|break\s+out)"
    r"|as\s+i\s+(?:said|mentioned)|remains?\s+to\s+be\s+seen|wait\s+and\s+see)\b",
    re.IGNORECASE,
)


def normalize_turns(structured_content) -> list[dict]:
    """Coerce a dataset's structured_content into ``[{speaker, text}, ...]``.

    Accepts a list of dicts, a JSON string, or None. Turns with empty text
    are dropped. Speaker defaults to "" when missing.
    """
    if structured_content is None:
        return []
    if isinstance(structured_content, str):
        try:
            structured_content = json.loads(structured_content)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(structured_content, list):
        return []
    turns = []
    for t in structured_content:
        if not isinstance(t, dict):
            continue
        text = (t.get("text") or "").strip()
        if not text:
            continue
        turns.append({"speaker": (t.get("speaker") or "").strip(), "text": text})
    return turns


def find_qa_boundary(turns: list[dict]) -> int:
    """Index of the first Q&A turn, or len(turns) if no Q&A detected.

    Strategy: scan operator turns for a Q&A-transition phrase; the section
    starts at the NEXT turn. Falls back to any turn matching the phrase.
    """
    n = len(turns)
    for i, t in enumerate(turns):
        if _QA_START_RE.search(t["text"]):
            # Operator announcing Q&A: section starts after the announcement.
            # A non-operator match usually IS already Q&A content.
            return i + 1 if _OPERATOR_RE.search(t["speaker"] or "") else i
    return n


_NAME_SPLIT_RE = re.compile(r"--|—|\(|,")
_NAME_TOKEN_RE = re.compile(r"[a-z]+")


def _name_key(speaker: str) -> str:
    """Normalize a speaker label to 'first last' for cross-section matching.

    Transcript sources are inconsistent: the same person appears as
    "John Doe", "John T. Doe" and "John Doe -- Chief Executive Officer".
    Keying on (first token, last token) of the pre-title part makes all
    three collide, which is exactly what we want.
    """
    if not speaker:
        return ""
    base = _NAME_SPLIT_RE.split(speaker.lower(), 1)[0]
    toks = _NAME_TOKEN_RE.findall(base)
    if not toks:
        return ""
    return toks[0] if len(toks) == 1 else f"{toks[0]} {toks[-1]}"


def split_turns(turns: list[dict]) -> dict:
    """Split speaker turns into prepared-remarks and Q&A sections + stats.

    Q&A speaker classification (exec vs analyst) uses two signals:
      1. Name match against prepared-remarks speakers (normalized via
         :func:`_name_key`, robust to title suffixes / middle initials).
      2. Behavior: a Q&A speaker whose turns are mostly questions is an
         analyst; one who mostly makes statements is an executive
         (executives answer, they don't ask).

    Returns a dict with:
      prepared_text, qa_text, full_text          — plain-text sections
      has_qa                                     — 1/0
      n_turns, n_exec_speakers, n_analyst_speakers, n_analyst_questions
      analyst_q_words, exec_a_words              — word counts in Q&A
      answer_question_ratio                      — exec words per analyst word
      hedge_per_1k                               — hedging phrases per 1000
                                                   exec-answer words (evasiveness)
    """
    boundary = find_qa_boundary(turns)
    prep = turns[:boundary]
    qa = turns[boundary:]

    # Executives = non-operator speakers heard during prepared remarks.
    exec_keys = {
        _name_key(t["speaker"]) for t in prep
        if t["speaker"] and not _OPERATOR_RE.search(t["speaker"])
    }
    exec_keys.discard("")

    # Pass 1 — profile each Q&A speaker: how often do they ask questions?
    qa_speaker_stats: dict[str, list[int]] = {}  # key -> [n_turns, n_question_turns]
    for t in qa:
        sp = t["speaker"]
        if not sp or _OPERATOR_RE.search(sp):
            continue
        key = _name_key(sp)
        st = qa_speaker_stats.setdefault(key, [0, 0])
        st[0] += 1
        st[1] += "?" in t["text"]

    def _is_exec(key: str) -> bool:
        if key in exec_keys:
            return True
        n, q = qa_speaker_stats.get(key, (0, 0))
        return n > 0 and q / n < 0.5  # mostly statements => executive

    # Pass 2 — accumulate word counts and evasiveness stats.
    analyst_keys: set[str] = set()
    n_analyst_questions = 0
    analyst_q_words = 0
    exec_a_words = 0
    hedge_count = 0

    for t in qa:
        sp = t["speaker"]
        if not sp or _OPERATOR_RE.search(sp):
            continue
        wc = len(t["text"].split())
        if _is_exec(_name_key(sp)):
            exec_a_words += wc
            hedge_count += len(_HEDGE_RE.findall(t["text"]))
        else:
            analyst_keys.add(_name_key(sp))
            analyst_q_words += wc
            if "?" in t["text"]:
                n_analyst_questions += 1

    exec_speakers = exec_keys
    analyst_speakers = analyst_keys

    prepared_text = " ".join(t["text"] for t in prep)
    qa_text = " ".join(t["text"] for t in qa)
    has_qa = int(len(qa_text) >= 250)  # ignore trivial closings

    return {
        "prepared_text": prepared_text,
        "qa_text": qa_text if has_qa else "",
        "full_text": (prepared_text + " " + qa_text).strip(),
        "has_qa": has_qa,
        "n_turns": len(turns),
        "n_exec_speakers": len(exec_speakers),
        "n_analyst_speakers": len(analyst_speakers),
        "n_analyst_questions": n_analyst_questions,
        "analyst_q_words": analyst_q_words,
        "exec_a_words": exec_a_words,
        "answer_question_ratio": (exec_a_words / analyst_q_words) if analyst_q_words else 0.0,
        "hedge_per_1k": (hedge_count / exec_a_words * 1000.0) if exec_a_words else 0.0,
    }


def flat_sections(text: str) -> dict:
    """Sections dict for a transcript with no usable speaker structure.

    Everything goes into prepared/full; Q&A stats are zeroed rather than
    guessed (a regex guess on flat text misclassifies whole documents).
    """
    text = (text or "").strip()
    return {
        "prepared_text": text, "qa_text": "", "full_text": text,
        "has_qa": 0, "n_turns": 0, "n_exec_speakers": 0,
        "n_analyst_speakers": 0, "n_analyst_questions": 0,
        "analyst_q_words": 0, "exec_a_words": 0,
        "answer_question_ratio": 0.0, "hedge_per_1k": 0.0,
    }


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts_text (
    ticker                TEXT NOT NULL,
    quarter               TEXT NOT NULL,   -- 'q1'..'q4'
    year                  INTEGER NOT NULL,
    pub_date              TEXT,            -- 'YYYY-MM-DD'
    source                TEXT,            -- 'hf' | 'defeatbeta' | 'edgar'
    has_qa                INTEGER,
    n_turns               INTEGER,
    n_exec_speakers       INTEGER,
    n_analyst_speakers    INTEGER,
    n_analyst_questions   INTEGER,
    analyst_q_words       INTEGER,
    exec_a_words          INTEGER,
    answer_question_ratio REAL,
    hedge_per_1k          REAL,
    n_words_full          INTEGER,
    full_z                BLOB,            -- zlib-compressed text
    prep_z                BLOB,
    qa_z                  BLOB,
    PRIMARY KEY (ticker, quarter, year)
)
"""

_INSERT = """
INSERT OR REPLACE INTO transcripts_text
  (ticker, quarter, year, pub_date, source, has_qa, n_turns,
   n_exec_speakers, n_analyst_speakers, n_analyst_questions,
   analyst_q_words, exec_a_words, answer_question_ratio, hedge_per_1k,
   n_words_full, full_z, prep_z, qa_z)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

# Metadata columns (everything except the BLOBs) — used by readers that
# must not drag hundreds of MB of compressed text into pandas.
META_COLS = [
    "ticker", "quarter", "year", "pub_date", "source", "has_qa",
    "n_turns", "n_exec_speakers", "n_analyst_speakers",
    "n_analyst_questions", "analyst_q_words", "exec_a_words",
    "answer_question_ratio", "hedge_per_1k", "n_words_full",
]


def open_db(db_path) -> sqlite3.Connection:
    """Open the DB tuned for bulk writes (WAL, relaxed fsync)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_SCHEMA)
    return conn


def make_row(ticker: str, quarter: str, year: int, pub_date: str,
             source: str, sections: dict) -> tuple:
    """Build an insert tuple from :func:`split_turns` output."""
    full_text = sections["full_text"]
    return (
        ticker, quarter.lower(), int(year), pub_date, source,
        sections["has_qa"], sections["n_turns"],
        sections["n_exec_speakers"], sections["n_analyst_speakers"],
        sections["n_analyst_questions"], sections["analyst_q_words"],
        sections["exec_a_words"], round(sections["answer_question_ratio"], 4),
        round(sections["hedge_per_1k"], 4), len(full_text.split()),
        compress_text(full_text),
        compress_text(sections["prepared_text"]),
        compress_text(sections["qa_text"]),
    )


def write_batch(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Insert a batch of rows in one transaction."""
    if not rows:
        return
    with conn:  # implicit BEGIN/COMMIT
        conn.executemany(_INSERT, rows)


def fetch_sections(conn: sqlite3.Connection, ticker: str, quarter: str,
                   year: int) -> dict[str, str]:
    """Decompress one transcript's sections. Keys match 03's convention:
    'full' always; 'prepared_remarks' and 'qa' when Q&A was detected."""
    row = conn.execute(
        "SELECT full_z, prep_z, qa_z, has_qa FROM transcripts_text "
        "WHERE ticker=? AND quarter=? AND year=?",
        (ticker, quarter, year),
    ).fetchone()
    if row is None:
        return {}
    full_z, prep_z, qa_z, has_qa = row
    if has_qa:
        return {
            "prepared_remarks": decompress_text(prep_z),
            "qa": decompress_text(qa_z),
            "full": decompress_text(full_z),
        }
    return {"full": decompress_text(full_z)}
