"""Sentiment, readability, and linguistic feature computation.

Pure functions extracted from 03_sentiment.ipynb so they can be unit-tested
without requiring a running notebook kernel, SQLite database, or GPU.
"""

import re
import html as html_mod
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import pysentiment2 as ps2
import textstat


# ---------------------------------------------------------------------------
# Transcript cleaning
# ---------------------------------------------------------------------------
#
# Transcript files are single-line blobs (newlines collapsed during HTML
# extraction).  We use *surgical phrase removal* (regex substitution)
# instead of line-based filtering so we only remove boilerplate, not the
# content that happens to be on the same line.

# (pattern, replacement) pairs — patterns are removed from the text.
# Order matters: broader patterns should come after more specific ones.
# Pre-compiled at module load for performance.
_RAW_BOILERPLATE_REMOVALS: list[tuple[str, str]] = [
    # -- Motley Fool header -------------------------------------------------
    # "Image source: The Motley Fool."
    (r"image\s*source\s*:\s*the\s+motley\s+fool\s*\.\s*", " "),
    # "Image source: Getty Images."
    (r"image\s*source\s*:\s*getty\s+images\s*\.\s*", " "),
    # "Nvidia ( NVDA 2.23% )" — ticker/price noise in MF headers
    (r"\(\s*[A-Z]{1,5}\s+[\d.]+%\s*\)", " "),
    # "Q1 2025 Earnings Call May 22, 2024 , 5:00 p.m. ET"
    (r"(Q[1-4]\s+\d{4}\s+)?Earnings\s+Call\s+[A-Z][a-z]{2}\s+\d{1,2}\s*,\s*\d{4}\s*,\s*\d{1,2}:\d{2}\s*(a\.?m\.?|p\.?m\.?)\s*(ET|PT|CT)", " "),
    # "Contents: Prepared Remarks Questions and Answers Call Participants"
    (r"Contents\s*:\s*Prepared\s+Remarks\s+Questions\s+and\s+Answers\s+Call\s+Participants", " "),
    # "Prepared Remarks:" / "Questions and Answers:" section headers
    (r"Prepared\s+Remarks\s*:\s*", " "),
    (r"Questions\s+and\s+Answers\s*:\s*", " "),
    # "Call Participants:" / "Call participants"
    (r"Call\s+Participants\s*:\s*", " "),
    # "Takeaways" section header
    (r"Takeaways\s*", " "),
    # "More NVDA analysis All earnings call transcripts" (footer)
    (r"More\s+[A-Z]{1,5}\s+analysis\s+All\s+earnings\s+call\s+transcripts\s*", " "),

    # "The Motley Fool has a disclosure policy."
    (r"The\s+Motley\s+Fool\s+has\s+a\s+disclosure\s+policy\s*\.\s*", " "),
    # "This article was originally published on The Motley Fool."
    (r"This\s+article\s+was\s+originally\s+published\s+on\s+The\s+Motley\s+Fool\s*\.\s*", " "),
    # "Should you invest $1,000 in Microsoft right now?"
    (r"Should\s+you\s+invest\s+\$[\d,]+\s+in\s+[A-Z][a-z]+\s+right\s+now\s*\?\s*", " "),
    # "Before you buy stock in Microsoft, consider this:"
    (r"Before\s+you\s+buy\s+stock\s+in\s+[A-Z][a-z]+[,.]?\s+consider\s+this\s*:?\s*", " "),
    # "The Motley Fool Stock Advisor analyst team just identified what they believe are the 10 best stocks..."
    (r"The\s+Motley\s+Fool\s+Stock\s+Advisor\s+analyst\s+team\s+just\s+identified[^.?!]*[.?!]", " "),
    # "See the 10 stocks."
    (r"See\s+the\s+10\s+stocks\s*\.\s*", " "),
    # "When our analyst team has a stock tip, it can pay to listen."
    (r"When\s+our\s+analyst\s+team\s+has\s+a\s+stock\s+tip[^.?!]*[.?!]", " "),
    # "© 2024 All rights reserved."
    (r"©\s*20\d{2}\s*All\s+rights\s+reserved\s*\.\s*", " "),

    # -- SEC EDGAR boilerplate -----------------------------------------------
    # "Exhibit 99.1" / "Exhibit 99.2"
    (r"Exhibit\s+99\.\d\s*", " "),
    # "FOR IMMEDIATE RELEASE"
    (r"FOR\s+IMMEDIATE\s+RELEASE\s*", " "),
    # City, State — Date line: "REDMOND, Wash. — October 30, 2024"
    (r"[A-Z][A-Z\s]+[,.]?\s+[A-Z][a-z]+\.?\s*(—|--|–)\s*[A-Z][a-z]+\s+\d{1,2}\s*,?\s*20\d{2}\s*", " "),
    # "Item 2.02 Results of Operations and Financial Condition"
    (r"Item\s+\d+\.\d+\s+[A-Za-z\s]+(and\s+)?[A-Za-z\s]+", " "),
    # Forward-looking statements disclaimers (paragraph-level removal)
    # Match from "This press release contains forward-looking" through to
    # "actual results (could|may|might|will|to) (differ|vary|be)"
    (r"(This|The)\s+(press\s+release|communication|report|filing|document|call|presentation)\s+(contains|includes|may\s+contain)\s+(forward[\s-]*looking|\"forward-looking\")\s+statements.*?actual\s+results\s+(could|may|might|will|to)\s+(differ|vary|be)\s+[^.?!]*[.?!]", " "),
    # Simpler variant: "This call contains forward-looking statements."
    (r"(This|The)\s+(call|conference|presentation)\s+(contains|includes|may\s+contain)\s+forward[\s-]*looking\s+statements[^.?!]*[.?!]", " "),
    # "Safe Harbor" statements
    (r"Safe\s+Harbor\s+(Statement|Provision)[^.]*\.", " "),
    # "All rights reserved."
    (r"All\s+rights\s+reserved\s*\.\s*", " "),
    # Copyright symbol lines
    (r"©\s*20\d{2}\s*[A-Za-z\s,]+\s*", " "),

    # -- SEC contact / metadata lines ---------------------------------------
    # "Investor Relations Contact:" / "Media Contact:" / "Press Contact:"
    (r"(Investor|Media|Press)\s+(Relations?|Contact)\s*:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+[^.]*\.?", " "),
    # "SOURCE Microsoft Corp."
    (r"SOURCE\s+[A-Za-z\s]+\s*", " "),
    # "For more information, contact:" / "For further information:"
    (r"For\s+(more|further)\s+information[,.]?\s*(contact\s*:?\s*)?[A-Z][a-z]+\s+[A-Z][a-z]+[^.]*\.?", " "),

    # -- Motley Fool operator / call-procedure lines ------------------------
    # "Operator Good afternoon." — Operator intro lines (with or without colon)
    (r"(Operator|Coordinator|Conference\s+Operator)\s*:?\s*Good\s+(morning|afternoon|evening|day)[^.?!]*[.?!]", " "),
    # "Operator My name is..." — alternative operator intro
    (r"(Operator|Coordinator)\s+[A-Z][a-z]+\s+name\s+is[^.?!]*[.?!]", " "),
    # "Good day, and welcome to the Q1 FY '25 Adobe earnings conference call.
    #  Today's conference is being recorded."
    (r"Good\s+(morning|afternoon|evening|day)[,.]?\s+(and\s+)?welcome\s+to[^.?!]*[.?!]", " "),
    # "Today's conference is being recorded."
    (r"Today'?s\s+conference\s+(call\s+)?is\s+being\s+recorded\s*\.\s*", " "),
    # "All lines have been placed on mute..."
    (r"All\s+lines\s+have\s+been\s+placed\s+on\s+mute[^.?!]*[.?!]", " "),
    # "I would now like to turn the conference over to..."
    (r"I\s+would\s+(now\s+)?like\s+to\s+(turn|hand)\s+the\s+(conference|call)\s+over\s+to[^.?!]*[.?!]", " "),
    # "At this time, I would like to welcome everyone to..."
    (r"At\s+this\s+time[,.]?\s+I\s+would\s+like\s+to\s+welcome\s+everyone[^.?!]*[.?!]", " "),
    # "Please go ahead, sir/ma'am/Mr./Ms."
    (r"Please\s+go\s+ahead[,.]?\s*(sir|ma'am|Mr\.?|Ms\.?|Dr\.?)\s*\.?", " "),
    # "Thank you. We will now begin..."
    (r"Thank\s+you[,.!]?\s*(We\s+will\s+now\s+begin|I\s+will\s+now\s+turn)[^.?!]*[.?!]", " "),

    # -- Analyst introductions during Q&A -----------------------------------
    # "Your next question comes from John Smith from Goldman Sachs."
    (r"(Your|Our|The|Next)\s+question\s+(comes\s+from|is\s+from)\s+[A-Z][a-z]+\s+[A-Z][a-z]+[^.?!]*[.?!]", " "),
    # "Our next question comes from the line of..."
    (r"(Our|The)\s+(next\s+)?question\s+(comes\s+from|is\s+from)\s+the\s+line\s+of[^.?!]*[.?!]", " "),

    # -- Generic operator/call-procedure lines (catch-all) --------------------
    # "Operator: [anything until a period]"
    (r"Operator\s*:?\s*[A-Z][a-z]+\s+(name\s+is|here|will|would|now|today|welcome|thank)[^.?!]*[.?!]\s*", " "),
    # "I would now like to turn the (call|conference) over to..."
    (r"I\s+(would|will|want|'d)\s+(now\s+)?like\s+to\s+(turn|hand)\s+(the\s+)?(call|conference)\s+over\s+to[^.?!]*[.?!]\s*", " "),

    # -- Q&A transition section headers -------------------------------------
    # "Question-and-Answer Session"
    (r"Question[\s-]*and[\s-]*Answer\s+Session\s*", " "),

    # -- HTML entities not caught by html.unescape --------------------------
    (r"&(amp|lt|gt|quot|apos|nbsp);", " "),
]

# Pre-compile all patterns once at module load (instead of re-compiling
# on every call to clean_transcript).
_BOILERPLATE_REMOVALS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _RAW_BOILERPLATE_REMOVALS
]


def _is_boilerplate_line(line: str) -> bool:
    """Return True if *line* matches any known boilerplate pattern.

    Kept for backward compatibility with existing tests.  Prefer
    :func:`clean_transcript` for new code.
    """
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    phrases = [
        "image source: getty images",
        "this article was originally published",
        "the motley fool has a disclosure policy",
        "should you invest",
        "where to invest",
        "before you buy stock in",
        "the motley fool stock advisor",
        "see the 10 stocks",
        "when our analyst team has",
        "©",
        "all rights reserved",
    ]
    return any(phrase in lowered for phrase in phrases)


def clean_transcript(raw_text: str) -> str:
    """Strip boilerplate from SEC EDGAR 8-K filings and Motley Fool transcripts.

    Transcript files are typically single-line blobs (newlines collapsed
    during HTML extraction).  This function uses **surgical phrase removal**
    (regex substitution) to strip known boilerplate without removing the
    surrounding content.

    Handles:
    1. SEC EDGAR press releases (Exhibit 99.1, contact info, disclaimers)
    2. Motley Fool earnings call transcripts (headers, operator lines,
       analyst introductions, Q&A transitions)
    3. Partially-parsed HTML from either source

    Returns cleaned text with normalized whitespace suitable for NLP.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # 1. Decode HTML entities (&amp; → &, &lt; → <, etc.)
    text = html_mod.unescape(text)

    # 2. Strip any remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Normalize Unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")

    # 4. Surgical phrase removal — apply each (pattern, replacement) pair
    for pattern, replacement in _BOILERPLATE_REMOVALS:
        text = pattern.sub(replacement, text)

    # 5. Collapse and normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Clean up artifacts: repeated punctuation, orphaned periods
    text = re.sub(r"\s*\.\s*\.\s*\.\s*", ". ", text)  # collapse ...
    text = re.sub(r"\s*,\s*,\s*", ", ", text)        # collapse ,,
    text = re.sub(r"\s{2,}", " ", text)                # final whitespace pass

    return text


# ---------------------------------------------------------------------------
# Text chunking (for FinBERT)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_words: int = 256) -> list[str]:
    """Split *text* into chunks of roughly *max_words* at sentence boundaries.

    Returns a list of chunk strings.  Each chunk is guaranteed to end at a
    sentence boundary (when ``sent_tokenize`` can detect one).
    """
    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_count = 0
    for sent in sentences:
        wc = len(sent.split())
        if current_count + wc > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_count = wc
        else:
            current_chunk.append(sent)
            current_count += wc
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ---------------------------------------------------------------------------
# VADER sentiment — paragraph-level aggregation
# ---------------------------------------------------------------------------

def compute_paragraph_vader(
    text: str,
    analyzer: SentimentIntensityAnalyzer | None = None,
    sentences_per_chunk: int = 5,
) -> dict[str, float]:
    """Compute VADER sentiment on sentence‑groups then aggregate.

    VADER was designed for short texts (tweets, sentences).  Running it on a
    50 000‑character earnings‑call transcript produces a compound score that
    always converges to ≈1.0 regardless of content.

    Instead we:
    1. Split *text* into sentences via ``sent_tokenize``.
    2. Group sentences into chunks of *sentences_per_chunk* (default 5).
    3. Run VADER on each chunk individually.
    4. Aggregate the per‑chunk scores into a feature vector.

    Returns a dict with keys **vader_compound** (whole‑document compound, kept
    for backward‑compatibility), plus per‑chunk statistics:
    ``vader_mean``, ``vader_std``, ``vader_min``, ``vader_max``,
    ``vader_p10``, ``vader_p90``, ``vader_pct_neg``, ``vader_pct_pos``,
    ``vader_n_chunks``.
    """
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()

    if not text or not text.strip():
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 0.0,
            "vader_mean": 0.0,
            "vader_std": 0.0,
            "vader_min": 0.0,
            "vader_max": 0.0,
            "vader_p10": 0.0,
            "vader_p90": 0.0,
            "vader_pct_neg": 0.0,
            "vader_pct_pos": 0.0,
            "vader_n_chunks": 0,
            "vader_n_paragraphs": 0,
        }

    # Whole-document score (backward compatibility)
    doc_scores = analyzer.polarity_scores(text)

    # Split into sentences, then group into chunks of N sentences
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = re.split(r"[.!?]+", text)

    if not sentences:
        sentences = [text]

    # Group sentences into chunks
    chunks: list[str] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)

    compounds: list[float] = []
    pos_list: list[float] = []
    neg_list: list[float] = []
    neu_list: list[float] = []

    for chunk in chunks:
        scores = analyzer.polarity_scores(chunk)
        compounds.append(scores["compound"])
        pos_list.append(scores["pos"])
        neg_list.append(scores["neg"])
        neu_list.append(scores["neu"])

    n = len(compounds)
    arr = np.array(compounds)
    pos_arr = np.array(pos_list)
    neg_arr = np.array(neg_list)
    neu_arr = np.array(neu_list)

    # Percentage of chunks that are negative / positive
    pct_neg = float(np.mean(arr < -0.05))
    pct_pos = float(np.mean(arr > 0.05))

    return {
        "vader_compound": doc_scores["compound"],
        "vader_pos": float(np.mean(pos_arr)),
        "vader_neg": float(np.mean(neg_arr)),
        "vader_neu": float(np.mean(neu_arr)),
        "vader_mean": float(np.mean(arr)),
        "vader_std": float(np.std(arr, ddof=0)),
        "vader_min": float(np.min(arr)),
        "vader_max": float(np.max(arr)),
        "vader_p10": float(np.percentile(arr, 10)),
        "vader_p90": float(np.percentile(arr, 90)),
        "vader_pct_neg": pct_neg,
        "vader_pct_pos": pct_pos,
        "vader_n_chunks": n,
        "vader_n_paragraphs": n,  # alias for backward compat
    }


def compute_vader_sentiment(
    text: str, analyzer: SentimentIntensityAnalyzer | None = None
) -> dict[str, float]:
    """Return VADER paragraph-level features for *text*.

    Delegates to :func:`compute_paragraph_vader` which provides richer
    paragraph‑aggregated features instead of raw whole‑document scores.
    """
    return compute_paragraph_vader(text, analyzer)


# ---------------------------------------------------------------------------
# Loughran-McDonald sentiment
# ---------------------------------------------------------------------------

def compute_lm_sentiment(
    text: str, lm: ps2.LM | None = None
) -> dict[str, int | float]:
    """Return Loughran-McDonald financial-dictionary scores for *text*.

    Parameters
    ----------
    text : str
        Input text.
    lm : pysentiment2.LM, optional
        Pre-constructed LM instance.

    Returns
    -------
    dict with keys ``lm_positive``, ``lm_negative``, ``lm_uncertainty``,
    ``lm_litigious``, ``lm_constraining``, ``lm_strong_modal``,
    ``lm_weak_modal``, plus derived ``lm_net``, ``lm_pos_ratio``,
    ``lm_neg_ratio``.

    Notes
    -----
    pysentiment2 v0.1.1 only exposes ``Positive`` and ``Negative`` counts
    via ``get_score()``.  The fine-grained categories are always returned
    as 0 until a newer version of the library restores them.
    """
    if lm is None:
        lm = ps2.LM()
    tokens = lm.tokenize(text)
    scores = lm.get_score(tokens)
    total = scores["Positive"] + scores["Negative"] + 1  # +1 avoids div-by-zero
    return {
        "lm_positive": scores.get("Positive", 0),
        "lm_negative": scores.get("Negative", 0),
        "lm_uncertainty": scores.get("Uncertainty", 0),
        "lm_litigious": scores.get("Litigious", 0),
        "lm_constraining": scores.get("Constraining", 0),
        "lm_strong_modal": scores.get("StrongModal", 0),
        "lm_weak_modal": scores.get("WeakModal", 0),
        "lm_net": (scores["Positive"] - scores["Negative"]) / total,
        "lm_pos_ratio": scores["Positive"] / total,
        "lm_neg_ratio": scores["Negative"] / total,
    }


# ---------------------------------------------------------------------------
# Linguistic / readability features
# ---------------------------------------------------------------------------

def compute_linguistic_features(text: str) -> dict[str, int | float]:
    """Return sentence/word counts and lexical-diversity ratios."""
    sentences = sent_tokenize(text)
    words = text.split()
    n_sentences = len(sentences)
    n_words = len(words)
    unique_ratio = len(set(w.lower() for w in words)) / max(n_words, 1)
    avg_sent_len = n_words / max(n_sentences, 1)
    return {
        "n_sentences": n_sentences,
        "n_words": n_words,
        "unique_word_ratio": round(unique_ratio, 4),
        "avg_sentence_length": round(avg_sent_len, 1),
    }


def compute_readability(text: str) -> dict[str, float]:
    """Return standard readability metrics for *text*.

    Metrics: Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog,
    SMOG Index, Automated Readability Index, Dale-Chall score.
    """
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability": textstat.automated_readability_index(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text),
    }


# ---------------------------------------------------------------------------
# Transcript section splitting (Prepared Remarks vs Q&A)
# ---------------------------------------------------------------------------

# Regex to match Q&A section boundaries in earnings call transcripts.
# Captures variations like "Questions & Answers:", "Questions and Answers:",
# and "Question-and-Answer Session".
#
# Note: this requires a colon after "Answers" (or "Session"), so the
# Motley Fool "Contents:" metadata line (which says "Questions and Answers"
# *without* a colon) will NOT match — no false positives.
_QA_BOUNDARY_RE = re.compile(
    r"(?i)\bQuestions\s*(?:and|&)\s*Answers\s*:"
    r"|\bQuestion-and-Answer\s+Session"
)


def split_transcript_sections(raw_text: str) -> dict[str, str]:
    """Split a raw transcript into prepared-remarks and Q&A sections.

    Earnings call transcripts from sources like Motley Fool contain two
    distinct sections: the scripted *prepared remarks* (read by executives)
    and the unscripted *Q&A* (analyst questions + management answers).
    The Q&A section is where genuine sentiment signals tend to leak.

    If no Q&A boundary is found (e.g. a plain SEC 8-K press release),
    the entire text is returned under the ``"full"`` key.

    Parameters
    ----------
    raw_text : str
        Raw transcript text, *before* any boilerplate cleaning.

    Returns
    -------
    dict
        * If Q&A detected: ``{"prepared_remarks": …, "qa": …}``
        * If no Q&A detected: ``{"full": …}``
        * If *raw_text* is empty: ``{"full": ""}``
    """
    if not raw_text or not raw_text.strip():
        return {"full": ""}

    # ------------------------------------------------------------------
    # 1.  Search for the first real Q&A section boundary.
    #     The regex requires a colon, so the "Contents:" metadata line
    #     (which says "Questions and Answers" without a colon) won't
    #     trigger a false positive.
    # ------------------------------------------------------------------
    match = _QA_BOUNDARY_RE.search(raw_text)
    if match is None:
        return {"full": raw_text}

    marker = match.group(0)
    parts = raw_text.split(marker, 1)
    if len(parts) < 2:
        return {"full": raw_text}

    prepared = parts[0].strip()
    qa = parts[1].strip()

    # ------------------------------------------------------------------
    # 2.  Validate that the Q&A section has substantial content.
    #     If it's only a few words, treat as a false positive.
    # ------------------------------------------------------------------
    if len(qa) < 250:
        return {"full": raw_text}

    return {"prepared_remarks": prepared, "qa": qa}


def clean_and_split_transcript(raw_text: str) -> dict[str, str]:
    """Split a raw transcript into sections, then clean each independently.

    This is the recommended entry point for sentiment pipelines that
    want per-section features.  It combines :func:`split_transcript_sections`
    and :func:`clean_transcript`.

    Parameters
    ----------
    raw_text : str
        Raw transcript text.

    Returns
    -------
    dict
        Same keys as :func:`split_transcript_sections`, but every value
        has been run through :func:`clean_transcript`.
    """
    sections = split_transcript_sections(raw_text)
    return {key: clean_transcript(text) for key, text in sections.items()}


# ---------------------------------------------------------------------------
# Sector mapping for the expanded ticker universe
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    # Technology
    "MSFT": "Technology", "GOOGL": "Technology", "META": "Technology",
    "AMZN": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "ADBE": "Technology", "CSCO": "Technology", "IBM": "Technology",
    "NOW": "Technology", "NFLX": "Technology", "DOCU": "Technology",
    "TWLO": "Technology", "PINS": "Technology", "SNAP": "Technology",
    "NET": "Technology", "DDOG": "Technology", "SQ": "Technology",
    "ROKU": "Technology", "CRWD": "Technology", "ZM": "Technology",
    "TEAM": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "COF": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABT": "Healthcare",
    # Consumer Cyclical
    "HD": "Consumer_Cyclical", "SBUX": "Consumer_Cyclical",
    "NKE": "Consumer_Cyclical", "UBER": "Consumer_Cyclical",
    "DASH": "Consumer_Cyclical", "DIS": "Consumer_Cyclical",
    # Consumer Defensive
    "WMT": "Consumer_Defensive", "COST": "Consumer_Defensive",
    "PG": "Consumer_Defensive", "KO": "Consumer_Defensive",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "DE": "Industrials", "DAL": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "DVN": "Energy",
    # Materials
    "FCX": "Materials", "NEM": "Materials",
    # Real Estate
    "PLD": "Real_Estate", "SPG": "Real_Estate",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities",
    # Communication Services
    "VZ": "Communication_Services",
}


def add_sector_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'sector' column to *df* based on ticker.

    Tickers not found in :data:`SECTOR_MAP` are labelled ``"Other"``.
    """
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Other")
    return df


# ---------------------------------------------------------------------------
# Cross-sectional normalization (z-scores by ticker)
# ---------------------------------------------------------------------------

SENTIMENT_Z_COLS: list[str] = [
    # VADER
    "vader_compound", "vader_mean", "vader_std", "vader_pct_neg",
    "vader_pct_pos", "vader_pos", "vader_neg", "vader_neu",
    # LM
    "lm_net", "lm_pos_ratio", "lm_neg_ratio",
    "lm_positive", "lm_negative", "lm_uncertainty",
    # FinBERT
    "finbert_net", "finbert_positive", "finbert_negative",
    "finbert_neutral",
    # Readability
    "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
    "smog_index", "automated_readability", "dale_chall_score",
    "unique_word_ratio", "avg_sentence_length",
]


def compute_ticker_z_scores(
    df: pd.DataFrame,
    prefix: str = "full",
    min_obs: int = 3,
) -> pd.DataFrame:
    """Compute within-ticker z-scores for key sentiment features.

    Cross-sectional normalization makes sentiment scores comparable across
    tickers.  A VADER compound of 0.3 might be "very positive" for one
    company's earnings calls (where the mean is 0.5) but "neutral" for a more
    effusive CEO (where the mean is 0.7).

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``"ticker"`` column and the columns ``f"{prefix}_{col}"``
        for each *col* in :data:`SENTIMENT_Z_COLS`.
    prefix : str
        Section prefix (e.g. ``"full"``, ``"qa"``).
    min_obs : int
        Minimum observations per ticker to compute z-scores.  Tickers with
        fewer observations get NaN z-scores.

    Returns
    -------
    pd.DataFrame
        New columns ``f"{prefix}_{col}_z"`` are added.
    """
    df = df.copy()
    for col in SENTIMENT_Z_COLS:
        src = f"{prefix}_{col}"
        dst = f"{prefix}_{col}_z"
        if src not in df.columns:
            continue

        def _z_score(group: pd.Series) -> pd.Series:
            if len(group) < min_obs:
                return pd.Series([np.nan] * len(group), index=group.index)
            mu = group.mean()
            sigma = group.std()
            if sigma < 1e-12:
                return pd.Series([np.nan] * len(group), index=group.index)
            return (group - mu) / sigma

        df[dst] = df.groupby("ticker")[src].transform(_z_score)
    return df


# ---------------------------------------------------------------------------
# Convenience: all sentiment features in one call
# ---------------------------------------------------------------------------

def compute_all_sentiment_features(
    text: str,
    vader: SentimentIntensityAnalyzer | None = None,
    lm: ps2.LM | None = None,
    prefix: str = "",
) -> dict:
    """Compute VADER + LM + linguistic + readability features for *text*.

    Parameters
    ----------
    text : str
        Input (cleaned) transcript text.
    vader : SentimentIntensityAnalyzer, optional
    lm : pysentiment2.LM, optional
    prefix : str
        If provided, all feature keys are prefixed with this string
        followed by an underscore.  Useful when computing features
        per transcript section (e.g. ``"prepared_remarks"``, ``"qa"``).

    Returns
    -------
    dict
        Flat dict suitable for building a pandas DataFrame row.
    """
    features: dict = {}
    features.update(compute_paragraph_vader(text, vader))
    features.update(compute_lm_sentiment(text, lm))
    features.update(compute_linguistic_features(text))
    features.update(compute_readability(text))
    if prefix:
        return {f"{prefix}_{k}": v for k, v in features.items()}
    return features
