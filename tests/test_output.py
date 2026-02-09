from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fi_app import output


def _minimal_results():
    return {
        "metadata": {
            "timestamp": "2026-02-09T12:00:00",
            "lookback_days": 7,
        },
        "data": {},
        "summary": {},
    }


def test_save_results_continues_after_known_db_error(monkeypatch, tmp_path: Path):
    class BrokenDB:
        def __init__(self):
            self.db_path = str(tmp_path / "broken.db")

        def save_collection_run(self, _):
            raise sqlite3.OperationalError("db-down")

    monkeypatch.setattr(output, "DatabaseManager", BrokenDB)
    monkeypatch.setattr(output, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(output, "OUTPUT_DIR", str(tmp_path / "outputs"))

    output.save_results(_minimal_results(), save_to_db=True)

    json_files = list((tmp_path / "outputs").glob("indicators_*.json"))
    assert len(json_files) == 1


def test_save_results_propagates_unexpected_db_error(monkeypatch, tmp_path: Path):
    class ExplodingDB:
        def __init__(self):
            self.db_path = str(tmp_path / "exploding.db")

        def save_collection_run(self, _):
            raise ZeroDivisionError("unexpected-db-error")

    monkeypatch.setattr(output, "DatabaseManager", ExplodingDB)
    monkeypatch.setattr(output, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(output, "OUTPUT_DIR", str(tmp_path / "outputs"))

    with pytest.raises(ZeroDivisionError, match="unexpected-db-error"):
        output.save_results(_minimal_results(), save_to_db=True)
