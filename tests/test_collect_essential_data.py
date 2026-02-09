from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import collect_essential_data as ced


class _DummyDB:
    def save_collection_run(self, _results):
        return 1

    def save_fred_data(self, *_args, **_kwargs):
        return None

    def save_market_data(self, *_args, **_kwargs):
        return None

    def save_crypto_data(self, *_args, **_kwargs):
        return None

    def get_db_stats(self):
        return {
            "fred_data_count": 1,
            "market_data_count": 1,
            "crypto_data_count": 1,
            "db_size_mb": 0.1,
        }


class _DummyMarketCollector:
    def __init__(self, lookback_days, use_alpha_vantage=False):
        self.lookback_days = lookback_days
        self.use_alpha_vantage = use_alpha_vantage

    def collect_category(self, _name, tickers):
        return {next(iter(tickers.keys())): {"Close": [1.0]}} if tickers else {}

    def get_latest_prices(self, data):
        return {ticker: 1.0 for ticker in data.keys()}

    def calculate_sector_performance(self, _data):
        return {}


class _DummyCryptoCollector:
    def __init__(self, lookback_days):
        self.lookback_days = lookback_days

    def collect_all(self):
        return {"BTC-USD": {"Close": [100000.0]}}

    def get_latest_prices(self, data):
        return {ticker: 100000.0 for ticker in data.keys()}

    def calculate_volatility(self, _data):
        return {"BTC-USD": 10.0}


def test_collect_essential_data_handles_known_errors(monkeypatch):
    class _KnownErrorFRED:
        def __init__(self, lookback_days):
            self.lookback_days = lookback_days

        def collect_all(self):
            raise ValueError("known-fred-error")

        def get_latest_values(self, _data):
            return {}

        def calculate_liquidity_metrics(self, _data):
            return {}

    monkeypatch.setattr(ced, "FREDCollector", _KnownErrorFRED)
    monkeypatch.setattr(ced, "MarketCollector", _DummyMarketCollector)
    monkeypatch.setattr(ced, "CryptoCollector", _DummyCryptoCollector)
    monkeypatch.setattr(ced, "DatabaseManager", lambda: _DummyDB())

    results = ced.collect_essential_data()

    assert results["summary"]["fred"]["success"] is False
    assert "known-fred-error" in results["summary"]["fred"]["error"]


def test_collect_essential_data_propagates_unexpected_errors(monkeypatch):
    class _UnexpectedErrorFRED:
        def __init__(self, lookback_days):
            self.lookback_days = lookback_days

        def collect_all(self):
            raise ZeroDivisionError("unexpected-fred-error")

        def get_latest_values(self, _data):
            return {}

        def calculate_liquidity_metrics(self, _data):
            return {}

    monkeypatch.setattr(ced, "FREDCollector", _UnexpectedErrorFRED)
    monkeypatch.setattr(ced, "MarketCollector", _DummyMarketCollector)
    monkeypatch.setattr(ced, "CryptoCollector", _DummyCryptoCollector)
    monkeypatch.setattr(ced, "DatabaseManager", lambda: _DummyDB())

    with pytest.raises(ZeroDivisionError, match="unexpected-fred-error"):
        ced.collect_essential_data()
