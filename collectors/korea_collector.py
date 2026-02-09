#!/usr/bin/env python3
"""
Korea Collector - 한국 시장 데이터 수집
Multi-source with fallback: FinanceDataReader → pykrx → yfinance
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings

try:
    from ..config import KOREA_TICKERS
except ImportError:
    from config import KOREA_TICKERS

try:
    from .base_multi_source import BaseMultiSourceCollector
except ImportError:
    from base_multi_source import BaseMultiSourceCollector

try:
    from .korea_sources import FinanceDataReaderSource, PyKrxSource
except ImportError:
    from korea_sources import FinanceDataReaderSource, PyKrxSource

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class KoreaCollector(BaseMultiSourceCollector):
    """
    한국 시장 데이터 수집기

    Data Source Priority:
    1. FinanceDataReader (무료, 한국 전용 추천 ⭐)
    2. pykrx (KRX 공식 데이터)
    3. yfinance (fallback)
    """

    def __init__(self, lookback_days: int = 90, use_multi_source: bool = True):
        """
        Args:
            lookback_days: 데이터 수집 기간 (일)
            use_multi_source: True면 FDR/pykrx 사용, False면 yfinance만 사용
        """
        super().__init__(lookback_days=lookback_days)
        self.use_multi_source = use_multi_source

        # Data sources 초기화
        if use_multi_source:
            self.fdr = FinanceDataReaderSource()
            self.pykrx = PyKrxSource()

    def _fetch_via_fdr(self, ticker: str) -> Optional[pd.DataFrame]:
        """FinanceDataReader를 통한 데이터 수집 (primary)"""
        if not self.use_multi_source or not self.fdr.available:
            return None

        try:
            return self.fdr.fetch_data(ticker, self.start_date, self.end_date)
        except (ValueError, KeyError, TypeError):
            return None

    def _fetch_via_pykrx(self, ticker: str) -> Optional[pd.DataFrame]:
        """pykrx를 통한 데이터 수집 (secondary)"""
        if not self.use_multi_source or not self.pykrx.available:
            return None

        try:
            # KOSPI 지수인 경우 특별 처리
            if ticker == '^KS11':
                return self.pykrx.fetch_kospi_index(self.start_date, self.end_date)
            else:
                return self.pykrx.fetch_data(ticker, self.start_date, self.end_date)
        except (ValueError, KeyError, TypeError):
            return None

    def fetch_ticker(self, ticker: str, name: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        단일 티커 데이터 수집 (with multi-source fallback + 고급 데이터)

        Args:
            ticker: 티커 심볼
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
            'has_advanced_data': False,
        }

        data = None

        # 1. FinanceDataReader 시도 (Primary)
        if self.use_multi_source:
            data = self._fetch_via_fdr(ticker)
            status['attempts'].append('fdr')

            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'fdr'
                logger.info("%s (%s) - FDR: %s days", ticker, name, len(data))

        # 2. pykrx 시도 (Secondary)
        if data is None and self.use_multi_source:
            data = self._fetch_via_pykrx(ticker)
            status['attempts'].append('pykrx')

            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'pykrx'
                logger.info("%s (%s) - pykrx: %s days", ticker, name, len(data))

        # 3. yfinance 시도 (Fallback)
        if data is None:
            data = self._fetch_via_yfinance(ticker)
            status['attempts'].append('yfinance')

            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'yfinance'
                logger.info("%s (%s) - yfinance (fallback): %s days", ticker, name, len(data))

        # 기본 데이터 수집 실패
        if data is None or data.empty:
            status['success'] = False
            logger.warning("%s (%s) - all sources failed", ticker, name)
            return None, status

        # 고급 데이터 추가 (pykrx 사용)
        if self.use_multi_source and self.pykrx.available:
            try:
                # 기관/외국인 매매 데이터
                trading_data = self.pykrx.fetch_institutional_trading(ticker, self.start_date, self.end_date)
                if trading_data is not None and not trading_data.empty:
                    # 날짜 인덱스 맞추기
                    data = data.join(trading_data, how='left')
                    status['has_advanced_data'] = True

                # 시가총액은 마지막 날짜 기준으로 수집
                # (API 호출 줄이기 위해)
                # market_cap = self.pykrx.fetch_market_cap(ticker, self.end_date)
                # if market_cap:
                #     data['market_cap'] = market_cap

            except (ValueError, KeyError, TypeError):
                # 고급 데이터 수집 실패는 무시
                pass

        return data, status

    def collect_category(self, category_name: str, tickers: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        특정 카테고리의 모든 티커 수집

        Args:
            category_name: 카테고리 이름
            tickers: {name: ticker} dictionary (from config.py)

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        logger.info("Collecting %s (%s items)", category_name, len(tickers))
        results = {}

        # config.py의 KOREA_TICKERS는 {name: ticker} 형식
        for name, ticker in tickers.items():
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
        모든 한국 시장 데이터 수집

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        logger.info("Korea Market Data Collection")
        logger.info("Period: %s to %s", self.start_date.date(), self.end_date.date())
        logger.info(
            "Multi-source: %s",
            "Enabled ✓" if self.use_multi_source else "Disabled (yfinance only)",
        )
        logger.info("=" * 60)

        all_data = {}
        self.collection_status = {}

        for category, tickers in KOREA_TICKERS.items():
            results = self.collect_category(category.upper(), tickers)
            all_data.update(results)

        logger.info("Total collected: %s assets", len(all_data))
        return all_data

    def calculate_kospi_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """KOSPI 주요 지표 계산"""
        metrics = {}

        # KOSPI 지수
        kospi_ticker = 'KOSPI' if 'KOSPI' in data else '^KS11'
        if kospi_ticker in data:
            kospi_df = data[kospi_ticker]
            if not kospi_df.empty and 'Close' in kospi_df.columns:
                current = kospi_df['Close'].iloc[-1]
                metrics['kospi_current'] = current

                # 1개월 수익률
                if len(kospi_df) > 21:
                    past = kospi_df['Close'].iloc[-22]
                    ret_1m = (current - past) / past * 100
                    metrics['kospi_return_1m'] = ret_1m

                # 3개월 수익률
                if len(kospi_df) > 63:
                    past = kospi_df['Close'].iloc[-64]
                    ret_3m = (current - past) / past * 100
                    metrics['kospi_return_3m'] = ret_3m

        # USD/KRW 환율
        if 'USDKRW' in data:
            fx_df = data['USDKRW']
            if not fx_df.empty and 'Close' in fx_df.columns:
                metrics['usdkrw_current'] = fx_df['Close'].iloc[-1]

        return metrics

    def calculate_sector_performance(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """한국 섹터 ETF 성과 계산"""
        sector_etfs = KOREA_TICKERS['sector_etfs']
        sector_data = []

        for key, ticker in sector_etfs.items():
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

            sector_data.append({
                'sector': key,
                'ticker': ticker,
                'return_1m': ret_1m,
            })

        return pd.DataFrame(sector_data).sort_values('return_1m', ascending=False)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import os

    print("\n" + "="*70)
    print("🧪 Testing Korea Collector with Multi-Source")
    print("="*70)

    # Multi-source 모드 테스트
    print("\n1️⃣  Multi-Source Mode (FDR → pykrx → yfinance)")
    print("-"*70)
    collector = KoreaCollector(lookback_days=30, use_multi_source=True)
    data = collector.collect_all()

    # 통계
    stats = collector.get_source_statistics()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total: {stats['total']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   By Source: {stats['by_source']}")

    # 최신 가격
    print("\n📈 Latest Prices (sample):")
    print("="*60)
    latest = collector.get_latest_prices(data)
    for ticker, price in list(latest.items())[:10]:
        source = collector.collection_status.get(ticker, {}).get('source', 'unknown')
        print(f"   {ticker:15s}: {price:>12,.2f}  [{source}]")

    # KOSPI 지표
    print("\n📊 KOSPI Metrics:")
    print("="*60)
    metrics = collector.calculate_kospi_metrics(data)
    for key, value in metrics.items():
        print(f"   {key:20s}: {value:>10.2f}")

    # 섹터 성과
    print("\n🏆 Sector Performance:")
    print("="*60)
    sector_perf = collector.calculate_sector_performance(data)
    if not sector_perf.empty:
        print(sector_perf.to_string(index=False))

    # 저장
    os.makedirs('data', exist_ok=True)
    output_file = f"data/korea_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    close_prices = pd.DataFrame()
    for ticker, df in data.items():
        if not df.empty and 'Close' in df.columns:
            close_prices[ticker] = df['Close']

    close_prices.to_csv(output_file)
    print(f"\n💾 Saved to: {output_file}")
