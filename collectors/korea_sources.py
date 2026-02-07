#!/usr/bin/env python3
"""
Korea Market Data Sources
한국 시장 데이터 소스 (FinanceDataReader, pykrx)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class FinanceDataReaderSource:
    """
    FinanceDataReader를 통한 한국 시장 데이터 수집

    장점:
    - 무료
    - 한국 시장 전용 최적화
    - KRX, Naver 등 다중 소스 지원
    - 높은 안정성

    설치: pip install finance-datareader
    """

    # 심볼 매핑 (yfinance -> FDR)
    SYMBOL_MAP = {
        # 지수
        '^KS11': 'KS11',        # KOSPI
        '^KS200': 'KS200',      # KOSPI 200
        '^KQ11': 'KQ11',        # KOSDAQ

        # 개별 종목 (yfinance .KS 제거)
        '005930.KS': '005930',  # 삼성전자
        '000660.KS': '000660',  # SK하이닉스
        '373220.KS': '373220',  # LG에너지솔루션
        '207940.KS': '207940',  # 삼성바이오로직스
        '005380.KS': '005380',  # 현대차
        '035420.KS': '035420',  # 네이버
        '035720.KS': '035720',  # 카카오
        '005490.KS': '005490',  # 포스코홀딩스

        # ETF
        '091170.KS': '091170',  # KODEX 은행
        '091180.KS': '091180',  # KODEX 반도체
        '228790.KS': '228790',  # KODEX 바이오
        '305720.KS': '305720',  # KODEX 2차전지
        '091160.KS': '091160',  # KODEX 자동차
        '153130.KS': '153130',  # KODEX 국고채3년
        '148070.KS': '148070',  # KODEX 국고채10년
    }

    def __init__(self):
        try:
            import FinanceDataReader as fdr
            self.fdr = fdr
            self.available = True
        except ImportError:
            self.fdr = None
            self.available = False
            print("   ⚠️  FinanceDataReader not installed. Run: pip install finance-datareader")

    def fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        데이터 수집

        Args:
            ticker: yfinance 형식 티커
            start_date: 시작일
            end_date: 종료일

        Returns:
            DataFrame with OHLCV data
        """
        if not self.available:
            return None

        # 심볼 변환
        fdr_symbol = self.SYMBOL_MAP.get(ticker, ticker)

        try:
            df = self.fdr.DataReader(
                fdr_symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            if df.empty:
                return None

            # 컬럼명 표준화 (yfinance와 동일하게)
            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume',
                })

            return df

        except Exception as e:
            print(f"      FDR error for {ticker}: {e}")
            return None


class PyKrxSource:
    """
    pykrx를 통한 한국 시장 데이터 수집

    장점:
    - KRX 공식 데이터
    - 무료
    - 정확한 데이터

    단점:
    - 실시간 데이터 아님 (일봉만)

    설치: pip install pykrx
    """

    # 심볼 매핑 (yfinance -> pykrx)
    SYMBOL_MAP = {
        '005930.KS': '005930',  # 삼성전자
        '000660.KS': '000660',  # SK하이닉스
        '373220.KS': '373220',  # LG에너지솔루션
        '207940.KS': '207940',  # 삼성바이오로직스
        '005380.KS': '005380',  # 현대차
        '035420.KS': '035420',  # 네이버
        '035720.KS': '035720',  # 카카오
        '005490.KS': '005490',  # 포스코홀딩스
    }

    def __init__(self):
        try:
            from pykrx import stock
            self.stock = stock
            self.available = True
        except ImportError:
            self.stock = None
            self.available = False
            print("   ⚠️  pykrx not installed. Run: pip install pykrx")

    def fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        데이터 수집

        Args:
            ticker: yfinance 형식 티커
            start_date: 시작일
            end_date: 종료일

        Returns:
            DataFrame with OHLCV data
        """
        if not self.available:
            return None

        # 심볼 변환
        krx_symbol = self.SYMBOL_MAP.get(ticker)
        if not krx_symbol:
            return None

        try:
            df = self.stock.get_market_ohlcv_by_date(
                fromdate=start_date.strftime('%Y%m%d'),
                todate=end_date.strftime('%Y%m%d'),
                ticker=krx_symbol
            )

            if df.empty:
                return None

            # 컬럼명 영문으로 변환
            df = df.rename(columns={
                '시가': 'Open',
                '고가': 'High',
                '저가': 'Low',
                '종가': 'Close',
                '거래량': 'Volume',
            })

            return df

        except Exception as e:
            print(f"      pykrx error for {ticker}: {e}")
            return None

    def fetch_kospi_index(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        KOSPI 지수 데이터 수집

        Args:
            start_date: 시작일
            end_date: 종료일

        Returns:
            DataFrame with index data
        """
        if not self.available:
            return None

        try:
            df = self.stock.get_index_ohlcv_by_date(
                fromdate=start_date.strftime('%Y%m%d'),
                todate=end_date.strftime('%Y%m%d'),
                ticker='1001'  # KOSPI
            )

            if df.empty:
                return None

            # 컬럼명 영문으로 변환
            df = df.rename(columns={
                '시가': 'Open',
                '고가': 'High',
                '저가': 'Low',
                '종가': 'Close',
                '거래량': 'Volume',
            })

            return df

        except Exception as e:
            print(f"      pykrx KOSPI error: {e}")
            return None


def test_korea_sources():
    """한국 데이터 소스 테스트"""
    print("\n" + "="*70)
    print("🧪 Testing Korea Market Data Sources")
    print("="*70)

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    # FinanceDataReader 테스트
    print("\n1️⃣  FinanceDataReader")
    print("-" * 70)
    fdr_source = FinanceDataReaderSource()

    if fdr_source.available:
        ticker = '005930.KS'  # 삼성전자
        print(f"   Testing {ticker} (Samsung Electronics)...")

        data = fdr_source.fetch_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            print(f"   ✅ Data collected: {len(data)} days")
            print(f"      Latest close: {data['Close'].iloc[-1]:,.0f} KRW")
        else:
            print(f"   ❌ Failed to fetch data")

        # KOSPI 지수
        ticker = '^KS11'
        print(f"\n   Testing {ticker} (KOSPI)...")
        data = fdr_source.fetch_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            print(f"   ✅ Data collected: {len(data)} days")
            print(f"      Latest: {data['Close'].iloc[-1]:,.2f}")
        else:
            print(f"   ❌ Failed to fetch data")
    else:
        print("   ❌ Not available")

    # pykrx 테스트
    print("\n2️⃣  pykrx")
    print("-" * 70)
    pykrx_source = PyKrxSource()

    if pykrx_source.available:
        ticker = '005930.KS'  # 삼성전자
        print(f"   Testing {ticker} (Samsung Electronics)...")

        data = pykrx_source.fetch_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            print(f"   ✅ Data collected: {len(data)} days")
            print(f"      Latest close: {data['Close'].iloc[-1]:,.0f} KRW")
        else:
            print(f"   ❌ Failed to fetch data")

        # KOSPI 지수
        print(f"\n   Testing KOSPI index...")
        data = pykrx_source.fetch_kospi_index(start_date, end_date)
        if data is not None and not data.empty:
            print(f"   ✅ Data collected: {len(data)} days")
            print(f"      Latest: {data['Close'].iloc[-1]:,.2f}")
        else:
            print(f"   ❌ Failed to fetch data")
    else:
        print("   ❌ Not available")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_korea_sources()
