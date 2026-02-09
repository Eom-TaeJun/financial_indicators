#!/usr/bin/env python3
"""
Base class for multi-source market collectors.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
YFINANCE_FETCH_ERRORS = (ValueError, KeyError, TypeError, RuntimeError, OSError)


class BaseMultiSourceCollector:
    """Shared utilities for collectors that fetch OHLCV time series."""

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        self.collection_status: Dict[str, Dict] = {}

    def _fetch_via_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data via yfinance fallback."""
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data
        except YFINANCE_FETCH_ERRORS as exc:
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return None

    def get_latest_prices(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract latest close per ticker."""
        latest: Dict[str, float] = {}
        for ticker, df in data.items():
            if not df.empty and "Close" in df.columns:
                latest[ticker] = float(df["Close"].iloc[-1])
        return latest

    def get_source_statistics(self) -> Dict:
        """Aggregate source-level success stats from collection_status."""
        stats = {
            "total": len(self.collection_status),
            "successful": 0,
            "failed": 0,
            "by_source": {},
        }

        for status in self.collection_status.values():
            if status.get("success"):
                stats["successful"] += 1
                source = status.get("source")
                if source:
                    stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            else:
                stats["failed"] += 1

        return stats
