"""Tests for the shared attach_llm_features() helper in scripts/llm_ab_test.py
(used by both the A/B test and retrain_model.py — the two must stay identical,
so the helper gets its own tests)."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.llm_ab_test as ab


@pytest.fixture()
def db_with_scores(tmp_path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE llm_qa_scores (
        ticker TEXT, quarter TEXT, year INTEGER, scores TEXT, model TEXT,
        PRIMARY KEY (ticker, quarter, year))""")
    rows = [
        ("AAA", "q1", 2024, {"guidance_direction": 1, "guidance_confidence": 2,
                             "demand_outlook": 1, "margin_outlook": -1,
                             "n_questions_dodged": 0, "tone_numbers_gap": 0,
                             "unexpected_negative": 0, "analyst_pushback": 1}),
        ("AAA", "q2", 2024, {"guidance_direction": -1, "guidance_confidence": 0,
                             "demand_outlook": -2, "margin_outlook": 0,
                             "n_questions_dodged": 3, "tone_numbers_gap": 2,
                             "unexpected_negative": 1, "analyst_pushback": 2}),
    ]
    conn.executemany(
        "INSERT INTO llm_qa_scores VALUES (?,?,?,?,?)",
        [(t, q, y, json.dumps(s), "test-model") for t, q, y, s in rows])
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def panel():
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "quarter": ["q1", "q2", "q1"],
        "year": [2024, 2024, 2024],
        "matched_earnings_date": pd.to_datetime(
            ["2024-02-01", "2024-05-01", "2024-02-15"]),
    })


def test_attach_llm_features(db_with_scores, panel, monkeypatch):
    # isolate from the real credibility pipeline (needs more tables)
    monkeypatch.setattr(ab, "load_credibility_features",
                        lambda *_a, **_k: pd.DataFrame())
    df, feats = ab.attach_llm_features(panel, db_with_scores)

    # all 8 ordinals + optimism + 5 qoq present; cred absent (empty cred table)
    assert "llm_guidance_direction" in feats and "llm_optimism" in feats
    assert "llm_guidance_direction_qoq" in feats
    assert "credibility" not in feats and "cred_weighted_optimism" not in feats

    a1 = df[(df["ticker"] == "AAA") & (df["quarter"] == "q1")].iloc[0]
    a2 = df[(df["ticker"] == "AAA") & (df["quarter"] == "q2")].iloc[0]
    # optimism = mean of [1.0*gd, 0.5*demand, 0.5*margin]
    assert np.isclose(a1["llm_optimism"], np.mean([1.0, 0.5, -0.5]))
    assert np.isclose(a2["llm_optimism"], np.mean([-1.0, -1.0, 0.0]))
    # QoQ delta = q2 minus q1 within the ticker, ordered by date
    assert np.isclose(a2["llm_guidance_direction_qoq"], -2.0)
    # unscored ticker stays NaN (LightGBM-friendly), never dropped
    b = df[df["ticker"] == "BBB"].iloc[0]
    assert np.isnan(b["llm_guidance_direction"])
    assert len(df) == len(panel)


def test_attach_llm_features_empty_db(tmp_path, panel):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE llm_qa_scores "
                 "(ticker TEXT, quarter TEXT, year INTEGER, scores TEXT)")
    conn.commit()
    conn.close()
    df, feats = ab.attach_llm_features(panel, db)
    assert feats == []
    assert len(df) == len(panel)
