from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fi_app import collector_runner


def _args(**overrides):
    defaults = {
        "days": 7,
        "quick": False,
        "full": False,
        "fred_only": False,
        "market_only": False,
        "crypto_only": False,
        "korea_only": False,
        "no_companies": False,
        "no_etfs": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_collect_data_handles_known_collector_error(monkeypatch):
    class BrokenFRED:
        def __init__(self, lookback_days):
            self.lookback_days = lookback_days

        def collect_all(self):
            raise ValueError("known-fred-error")

    monkeypatch.setattr(collector_runner, "FREDCollector", BrokenFRED)

    results = collector_runner.collect_data(_args(fred_only=True))

    assert "fred" in results["summary"]
    assert results["summary"]["fred"]["success"] is False
    assert "known-fred-error" in results["summary"]["fred"]["error"]


def test_collect_data_propagates_unexpected_collector_error(monkeypatch):
    class ExplodingFRED:
        def __init__(self, lookback_days):
            self.lookback_days = lookback_days

        def collect_all(self):
            raise ZeroDivisionError("unexpected-fred-error")

    monkeypatch.setattr(collector_runner, "FREDCollector", ExplodingFRED)

    with pytest.raises(ZeroDivisionError, match="unexpected-fred-error"):
        collector_runner.collect_data(_args(fred_only=True))
