import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_manager import DatabaseManager


def _run_payload(timestamp: str) -> dict:
    return {
        "metadata": {
            "timestamp": timestamp,
            "lookback_days": 30,
        },
        "summary": {
            "fred": {"success": True, "series_count": 1},
            "market": {"success": True, "ticker_count": 1},
            "crypto": {"success": False, "asset_count": 0},
            "korea": {"success": False, "asset_count": 0},
        },
    }


def test_build_ohlcv_records_normalizes_types(tmp_path: Path):
    db = DatabaseManager(db_path=str(tmp_path / "fi_test.db"))

    df = pd.DataFrame(
        {
            "Open": [100],
            "High": [101],
            "Low": [99],
            "Close": [np.nan],
            "Volume": [None],
        },
        index=[pd.Timestamp("2026-02-01")],
    )

    records = db._build_ohlcv_records(
        collection_run_id=7,
        data={"SPY": df},
        value_columns=db._OHLCV_DF_COLUMNS,
        category_map={"SPY": "indices"},
    )

    assert records == [
        (7, "2026-02-01", "SPY", "indices", 100.0, 101.0, 99.0, None, None)
    ]


def test_save_korea_data_persists_advanced_columns(tmp_path: Path):
    db_path = tmp_path / "fi_korea.db"
    db = DatabaseManager(db_path=str(db_path))

    df = pd.DataFrame(
        {
            "Open": [50000],
            "High": [51000],
            "Low": [49000],
            "Close": [50500],
            "Volume": [1000000],
            "institutional_net": [150000000.0],
            "foreign_net": [-50000000.0],
            "market_cap": [300000000000.0],
        },
        index=[pd.Timestamp("2026-02-02")],
    )

    db.save_korea_data(
        collection_run_id=3,
        korea_data={"005930.KS": df},
        category_map={"005930.KS": "large_caps"},
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT collection_run_id, date, ticker, category,
               open, high, low, close, volume,
               institutional_net, foreign_net, market_cap
        FROM korea_data
        """
    ).fetchone()
    conn.close()

    assert row == (
        3,
        "2026-02-02",
        "005930.KS",
        "large_caps",
        50000.0,
        51000.0,
        49000.0,
        50500.0,
        1000000.0,
        150000000.0,
        -50000000.0,
        300000000000.0,
    )


def test_get_collection_runs_enforces_minimum_limit(tmp_path: Path):
    db = DatabaseManager(db_path=str(tmp_path / "fi_runs.db"))

    db.save_collection_run(_run_payload("2026-02-01T10:00:00"))
    db.save_collection_run(_run_payload("2026-02-02T10:00:00"))

    runs = db.get_collection_runs(limit=0)

    assert len(runs) == 1
    assert runs.iloc[0]["timestamp"] == "2026-02-02T10:00:00"


def test_get_collection_runs_handles_non_numeric_limit(tmp_path: Path):
    db = DatabaseManager(db_path=str(tmp_path / "fi_runs_non_numeric.db"))

    db.save_collection_run(_run_payload("2026-02-01T10:00:00"))
    db.save_collection_run(_run_payload("2026-02-02T10:00:00"))

    runs = db.get_collection_runs(limit="invalid")

    assert len(runs) == 1
    assert runs.iloc[0]["timestamp"] == "2026-02-02T10:00:00"
