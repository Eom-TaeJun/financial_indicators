#!/usr/bin/env python3
"""
Crypto Collector - 암호화폐 및 RWA 데이터 수집
Multi-source with fallback: CoinGecko → Binance → yfinance
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings

try:
    from ..config import CRYPTO_TICKERS
except ImportError:
    from config import CRYPTO_TICKERS

try:
    from .base_multi_source import BaseMultiSourceCollector
except ImportError:
    from base_multi_source import BaseMultiSourceCollector

try:
    from .crypto_sources import CoinGeckoSource, BinanceSource
except ImportError:
    from crypto_sources import CoinGeckoSource, BinanceSource

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CryptoCollector(BaseMultiSourceCollector):
    """
    암호화폐 및 RWA 데이터 수집기

    Data Source Priority:
    1. Binance API (무료, OHLC 제공 ⭐)
    2. CoinGecko API (무료, Close+Volume만)
    3. yfinance (fallback)
    """

    def __init__(self, lookback_days: int = 90, use_multi_source: bool = True):
        """
        Args:
            lookback_days: 데이터 수집 기간 (일)
            use_multi_source: True면 CoinGecko/Binance 사용, False면 yfinance만 사용
        """
        super().__init__(lookback_days=lookback_days)
        self.use_multi_source = use_multi_source

        # Data sources 초기화
        if use_multi_source:
            self.coingecko = CoinGeckoSource()
            self.binance = BinanceSource()

    def _fetch_via_coingecko(self, ticker: str) -> Optional[pd.DataFrame]:
        """CoinGecko를 통한 데이터 수집 (primary)"""
        if not self.use_multi_source:
            return None

        try:
            return self.coingecko.fetch_historical_data(ticker, self.lookback_days)
        except (ValueError, KeyError, TypeError):
            return None

    def _fetch_via_binance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Binance를 통한 데이터 수집 (secondary)"""
        if not self.use_multi_source:
            return None

        try:
            return self.binance.fetch_historical_data(ticker, self.lookback_days)
        except (ValueError, KeyError, TypeError):
            return None

    def fetch_ticker(self, ticker: str, name: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        단일 티커 데이터 수집 (with multi-source fallback)

        Args:
            ticker: 티커 심볼 (예: 'BTC-USD')
            name: 자산 이름

        Returns:
            (DataFrame, status_dict)
        """
        status = {
            'ticker': ticker,
            'name': name,
            'success': False,
            'source': None,
            'attempts': [],
        }

        # 1. Binance 시도 (Primary - OHLC 제공)
        if self.use_multi_source:
            data = self._fetch_via_binance(ticker)
            status['attempts'].append('binance')

            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'binance'
                logger.info("%s (%s) - Binance (OHLC): %s days", ticker, name, len(data))
                return data, status

        # 2. CoinGecko 시도 (Secondary - Close+Volume만)
        if self.use_multi_source:
            data = self._fetch_via_coingecko(ticker)
            status['attempts'].append('coingecko')

            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'coingecko'
                logger.info("%s (%s) - CoinGecko: %s days", ticker, name, len(data))
                return data, status

        # 3. yfinance 시도 (Fallback)
        data = self._fetch_via_yfinance(ticker)
        status['attempts'].append('yfinance')

        if data is not None and not data.empty:
            status['success'] = True
            status['source'] = 'yfinance'
            logger.info("%s (%s) - yfinance (fallback): %s days", ticker, name, len(data))
            return data, status

        # 모두 실패
        status['success'] = False
        logger.warning("%s (%s) - all sources failed", ticker, name)
        return None, status

    def collect_category(self, category_name: str, tickers: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        특정 카테고리의 모든 티커 수집

        Args:
            category_name: 카테고리 이름
            tickers: {ticker: name} dictionary

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        logger.info("Collecting %s (%s assets)", category_name, len(tickers))
        results = {}

        for ticker, name in tickers.items():
            data, status = self.fetch_ticker(ticker, name)
            self.collection_status[ticker] = status

            if data is not None:
                results[ticker] = data

        success_rate = len(results) / len(tickers) * 100 if tickers else 0
        logger.info("Success: %s/%s (%.1f%%)", len(results), len(tickers), success_rate)

        # Source 통계
        sources = {}
        for status in self.collection_status.values():
            if status['success']:
                source = status['source']
                sources[source] = sources.get(source, 0) + 1

        if sources:
            logger.info("Sources used: %s", sources)

        return results

    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """
        모든 암호화폐 및 RWA 데이터 수집

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        logger.info("Crypto & RWA Data Collection")
        logger.info("Period: %s to %s", self.start_date.date(), self.end_date.date())
        logger.info(
            "Multi-source: %s",
            "Enabled ✓" if self.use_multi_source else "Disabled (yfinance only)",
        )
        logger.info("=" * 60)

        all_data = {}
        self.collection_status = {}

        for category, tickers in CRYPTO_TICKERS.items():
            results = self.collect_category(category.upper(), tickers)
            all_data.update(results)

        logger.info("Total collected: %s assets", len(all_data))
        return all_data

    def calculate_volatility(self, data: Dict[str, pd.DataFrame], window: int = 30) -> Dict[str, float]:
        """변동성 계산 (30일 표준편차)"""
        volatility = {}

        for ticker, df in data.items():
            if df.empty or 'Close' not in df.columns:
                continue

            if len(df) < window:
                continue

            # 일일 수익률
            returns = df['Close'].pct_change()

            # 30일 변동성 (연율화)
            vol = returns.tail(window).std() * (252 ** 0.5) * 100
            volatility[ticker] = vol

        return volatility

    def calculate_correlations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """자산 간 상관관계 계산"""
        # 종가 데이터 결합
        prices = pd.DataFrame()
        for ticker, df in data.items():
            if not df.empty and 'Close' in df.columns:
                prices[ticker] = df['Close']

        if prices.empty:
            return pd.DataFrame()

        # 수익률 계산
        returns = prices.pct_change().dropna()

        # 상관관계 계산
        corr = returns.corr()

        return corr


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import os

    print("\n" + "="*70)
    print("🧪 Testing Crypto Collector with Multi-Source")
    print("="*70)

    # Multi-source 모드 테스트
    print("\n1️⃣  Multi-Source Mode (CoinGecko → Binance → yfinance)")
    print("-"*70)
    collector = CryptoCollector(lookback_days=30, use_multi_source=True)
    data = collector.collect_all()

    # 통계
    stats = collector.get_source_statistics()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total: {stats['total']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   By Source: {stats['by_source']}")

    # 최신 가격
    print("\n📈 Latest Prices:")
    print("="*60)
    latest = collector.get_latest_prices(data)
    for ticker, price in latest.items():
        source = collector.collection_status[ticker]['source']
        print(f"   {ticker:12s}: ${price:>12,.2f}  [{source}]")

    # 변동성
    print("\n📊 30-Day Volatility:")
    print("="*60)
    volatility = collector.calculate_volatility(data)
    for ticker, vol in sorted(volatility.items(), key=lambda x: x[1], reverse=True):
        print(f"   {ticker:12s}: {vol:>6.2f}%")

    # yfinance only 모드 비교
    print("\n\n2️⃣  yfinance Only Mode (for comparison)")
    print("-"*70)
    collector_yf = CryptoCollector(lookback_days=30, use_multi_source=False)
    data_yf = collector_yf.collect_all()

    stats_yf = collector_yf.get_source_statistics()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total: {stats_yf['total']}")
    print(f"   Successful: {stats_yf['successful']}")
    print(f"   Failed: {stats_yf['failed']}")

    # 저장
    os.makedirs('data', exist_ok=True)
    output_file = f"data/crypto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    close_prices = pd.DataFrame()
    for ticker, df in data.items():
        if not df.empty and 'Close' in df.columns:
            close_prices[ticker] = df['Close']

    close_prices.to_csv(output_file)
    print(f"\n💾 Saved to: {output_file}")
