"""Contract tests for the ensemble module (constants + light logic; no DB)."""
import pytest

from src import ensemble as E


def test_signal_weights_sum_to_one():
    assert abs(sum(E.SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9


def test_eca_decay_matches_pead_window():
    assert E.ECA_DECAY_DAYS == 30


def test_eca_empty_predictions_returns_empty(tmp_path):
    # A DB with no model_predictions table -> empty scores, not a crash.
    import sqlite3
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE prices (date TEXT, ticker TEXT, close REAL)")
    conn.commit()
    conn.close()
    out = E.eca_daily_scores(db_path=db)
    assert list(out.columns) == ["date", "ticker", "eca_z"]
    assert out.empty


def test_linear_decay_factor_shape():
    # Reproduce the decay factors the module applies: 1.0 at event, ->0 at day 30.
    factors = [1.0 - i / E.ECA_DECAY_DAYS for i in range(E.ECA_DECAY_DAYS)]
    assert factors[0] == 1.0
    assert factors[-1] == pytest.approx(1 / E.ECA_DECAY_DAYS)   # last kept day
    assert all(factors[i] > factors[i + 1] for i in range(len(factors) - 1))
