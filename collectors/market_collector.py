#!/usr/bin/env python3
"""
Market Collector - 시장 데이터 수집
Multi-source with fallback: Alpha Vantage → yfinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import warnings

from config import MARKET_TICKERS, US_MAJOR_COMPANIES

try:
    from .market_sources import AlphaVantageSource
except ImportError:
    from market_sources import AlphaVantageSource

warnings.filterwarnings('ignore')


class MarketCollector:
    """
    미국 시장 데이터 수집기

    Data Source Priority:
    1. Alpha Vantage API (무료, 고품질)
    2. yfinance (fallback)
    """

    def __init__(self, lookback_days: int = 90, use_alpha_vantage: bool = True):
        """
        Args:
            lookback_days: 데이터 수집 기간 (일)
            use_alpha_vantage: True면 Alpha Vantage 사용, False면 yfinance만 사용
        """
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        self.use_alpha_vantage = use_alpha_vantage

        # Alpha Vantage 초기화
        if use_alpha_vantage:
            self.alpha_vantage = AlphaVantageSource()
        else:
            self.alpha_vantage = None

        self.collection_status = {}

    def _fetch_via_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """yfinance를 통한 데이터 수집 (fallback)"""
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                return None

            # MultiIndex 처리
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data

        except Exception:
            return None

    def _fetch_via_alpha_vantage(self, ticker: str) -> Optional[pd.DataFrame]:
        """Alpha Vantage를 통한 데이터 수집 (primary)"""
        if not self.use_alpha_vantage or not self.alpha_vantage or not self.alpha_vantage.available:
            return None

        try:
            # 100일 이하면 compact, 아니면 full
            outputsize = 'compact' if self.lookback_days <= 100 else 'full'
            data = self.alpha_vantage.fetch_daily_data(ticker, outputsize=outputsize)

            if data is not None and not data.empty:
                # 날짜 범위 필터링
                data = data[data.index >= self.start_date]

            return data

        except Exception:
            return None

    def fetch_ticker(self, ticker: str, name: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        단일 티커 데이터 수집 (with multi-source fallback)

        Args:
            ticker: 티커 심볼 (예: 'AAPL')
            name: 종목 이름

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

        # 1. Alpha Vantage 시도 (Primary)
        if self.use_alpha_vantage and self.alpha_vantage and self.alpha_vantage.available:
            data = self._fetch_via_alpha_vantage(ticker)
            status['attempts'].append('alpha_vantage')

            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'alpha_vantage'
                print(f"   ✅ {ticker:6s} ({name}) - AlphaVantage: {len(data)} days")
                return data, status

        # 2. yfinance 시도 (Fallback)
        data = self._fetch_via_yfinance(ticker)
        status['attempts'].append('yfinance')

        if data is not None and not data.empty:
            status['success'] = True
            status['source'] = 'yfinance'
            source_label = " (fallback)" if 'alpha_vantage' in status['attempts'] else ""
            print(f"   ✅ {ticker:6s} ({name}) - yfinance{source_label}: {len(data)} days")
            return data, status

        # 모두 실패
        status['success'] = False
        print(f"   ❌ {ticker:6s} ({name}) - All sources failed")
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
        print(f"\n📈 Collecting {category_name} ({len(tickers)} tickers)...")
        results = {}

        for ticker, name in tickers.items():
            data, status = self.fetch_ticker(ticker, name)
            self.collection_status[ticker] = status

            if data is not None:
                results[ticker] = data

        success_rate = len(results) / len(tickers) * 100 if tickers else 0
        print(f"   Success: {len(results)}/{len(tickers)} ({success_rate:.1f}%)")

        # Source 통계
        sources = {}
        for status in self.collection_status.values():
            if status['success']:
                source = status['source']
                sources[source] = sources.get(source, 0) + 1

        if sources:
            print(f"   Sources used: {sources}")

        return results

    def collect_all_etfs(self) -> Dict[str, pd.DataFrame]:
        """모든 ETF 데이터 수집"""
        all_results = {}

        for category, tickers in MARKET_TICKERS.items():
            results = self.collect_category(category.upper(), tickers)
            all_results.update(results)

        return all_results

    def collect_all_companies(self) -> Dict[str, pd.DataFrame]:
        """모든 주요 기업 데이터 수집"""
        all_results = {}

        for sector, companies in US_MAJOR_COMPANIES.items():
            results = self.collect_category(f"US {sector.upper()}", companies)
            all_results.update(results)

        return all_results

    def collect_all(self, include_etfs: bool = True, include_companies: bool = True) -> Dict[str, pd.DataFrame]:
        """
        모든 시장 데이터 수집

        Args:
            include_etfs: ETF 포함 여부
            include_companies: 개별 기업 포함 여부

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        print(f"\n📊 Market Data Collection")
        print(f"   Period: {self.start_date.date()} to {self.end_date.date()}")
        av_status = "Enabled ✓" if (self.use_alpha_vantage and self.alpha_vantage and self.alpha_vantage.available) else "Disabled (yfinance only)"
        print(f"   Alpha Vantage: {av_status}")
        print("="*60)

        all_data = {}
        self.collection_status = {}

        if include_etfs:
            print("\n" + "="*60)
            print("ETFs Collection")
            print("="*60)
            etf_data = self.collect_all_etfs()
            all_data.update(etf_data)

        if include_companies:
            print("\n" + "="*60)
            print("Major Companies Collection")
            print("="*60)
            company_data = self.collect_all_companies()
            all_data.update(company_data)

        print(f"\n✅ Total collected: {len(all_data)} tickers\n")
        return all_data

    def get_latest_prices(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """각 티커의 최신 종가 추출"""
        latest = {}
        for ticker, df in data.items():
            if not df.empty and 'Close' in df.columns:
                latest[ticker] = df['Close'].iloc[-1]
        return latest

    def calculate_returns(self, data: Dict[str, pd.DataFrame], periods: List[int] = None) -> pd.DataFrame:
        """수익률 계산"""
        if periods is None:
            periods = [1, 5, 21, 63, 252]

        returns_data = []

        for ticker, df in data.items():
            if df.empty or 'Close' not in df.columns:
                continue

            row = {'ticker': ticker}
            current_price = df['Close'].iloc[-1]

            for period in periods:
                if len(df) > period:
                    past_price = df['Close'].iloc[-(period+1)]
                    ret = (current_price - past_price) / past_price * 100
                    row[f'return_{period}d'] = ret
                else:
                    row[f'return_{period}d'] = None

            returns_data.append(row)

        return pd.DataFrame(returns_data)

    def calculate_sector_performance(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """섹터별 성과 계산"""
        sector_etfs = MARKET_TICKERS['sectors']
        sector_data = []

        for ticker, name in sector_etfs.items():
            if ticker not in data:
                continue

            df = data[ticker]
            if df.empty or 'Close' not in df.columns:
                continue

            # 1개월 수익률
            if len(df) > 21:
                ret_1m = (df['Close'].iloc[-1] - df['Close'].iloc[-22]) / df['Close'].iloc[-22] * 100
            else:
                ret_1m = None

            # 3개월 수익률
            if len(df) > 63:
                ret_3m = (df['Close'].iloc[-1] - df['Close'].iloc[-64]) / df['Close'].iloc[-64] * 100
            else:
                ret_3m = None

            sector_data.append({
                'sector': name,
                'ticker': ticker,
                'return_1m': ret_1m,
                'return_3m': ret_3m,
            })

        return pd.DataFrame(sector_data).sort_values('return_1m', ascending=False)

    def get_source_statistics(self) -> Dict:
        """데이터 소스 통계"""
        stats = {
            'total': len(self.collection_status),
            'successful': 0,
            'failed': 0,
            'by_source': {},
        }

        for status in self.collection_status.values():
            if status['success']:
                stats['successful'] += 1
                source = status['source']
                stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            else:
                stats['failed'] += 1

        return stats


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import os

    print("\n" + "="*70)
    print("🧪 Testing Market Collector with Alpha Vantage")
    print("="*70)

    # Alpha Vantage 모드 테스트 (소수만)
    print("\n1️⃣  With Alpha Vantage (Testing 5 tickers)")
    print("-"*70)

    collector = MarketCollector(lookback_days=30, use_alpha_vantage=True)

    # 소수의 티커만 테스트
    test_tickers = {
        'SPY': 'S&P 500',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet',
        'TSLA': 'Tesla',
    }

    data = collector.collect_category("TEST", test_tickers)

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
        print(f"   {ticker:6s}: ${price:>10.2f}  [{source}]")

    print("\n" + "="*70)
