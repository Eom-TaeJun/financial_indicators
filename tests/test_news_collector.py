from pathlib import Path
import sys

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import news_collector
from news_collector import PerplexityNewsCollector


def test_call_api_handles_known_request_errors(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    def _raise_timeout(*_args, **_kwargs):
        raise requests.exceptions.Timeout("known-timeout-error")

    monkeypatch.setattr(news_collector.requests, "post", _raise_timeout)

    collector = PerplexityNewsCollector()
    assert collector._call_api("hello") == ""


def test_call_api_propagates_unexpected_errors(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    def _raise_unexpected(*_args, **_kwargs):
        raise ZeroDivisionError("unexpected-api-error")

    monkeypatch.setattr(news_collector.requests, "post", _raise_unexpected)

    collector = PerplexityNewsCollector()
    with pytest.raises(ZeroDivisionError, match="unexpected-api-error"):
        collector._call_api("hello")
