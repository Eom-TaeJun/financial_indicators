#!/usr/bin/env python3
"""
US Market Data Sources
미국 시장 데이터 소스 (Alpha Vantage, Polygon.io)
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
import time
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class AlphaVantageSource:
    """
    Alpha Vantage API를 통한 미국 시장 데이터 수집

    장점:
    - 공식 데이터 제공자 (NYSE, NASDAQ)
    - 기술 지표 내장 (RSI, MACD, SMA 등 80+ 지표)
    - 펀더멘털 데이터 (재무제표, EPS, P/E)
    - 외환(Forex), 암호화폐 지원

    Rate Limit: 5 calls/min (무료 tier)
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            self.available = False
            logger.warning("ALPHA_VANTAGE_API_KEY not found")
        else:
            self.available = True

        self.session = requests.Session()
        self.last_call_time = 0
        self.min_interval = 12  # 5 calls/min = 12초 간격

    def _rate_limit(self):
        """Rate limiting (5 calls/min)"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    def fetch_daily_data(self, ticker: str, outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """
        일봉 데이터 수집

        Args:
            ticker: 티커 심볼 (예: 'AAPL')
            outputsize: 'compact' (최근 100일) or 'full' (20년)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.available:
            return None

        self._rate_limit()

        try:
            params = {
                'function': 'TIME_SERIES_DAILY',  # 무료 tier
                'symbol': ticker,
                'outputsize': outputsize,
                'apikey': self.api_key,
            }

            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Error check
            if 'Error Message' in data:
                logger.error("AlphaVantage error: %s", data["Error Message"])
                return None

            if 'Note' in data:
                # Rate limit exceeded
                logger.warning("AlphaVantage rate limit: %s", data["Note"])
                return None

            if 'Information' in data:
                # API limit message
                logger.warning("AlphaVantage info: %s", data["Information"])
                return None

            if 'Time Series (Daily)' not in data:
                logger.warning("AlphaVantage unexpected response keys: %s", list(data.keys()))
                return None

            # DataFrame 생성
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')

            # 컬럼명 변경 (무료 tier는 5개 컬럼)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # 타입 변환
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # 인덱스를 datetime으로
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            return df

        except requests.exceptions.RequestException as e:
            logger.error("AlphaVantage request error: %s", e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("AlphaVantage parse error: %s", e)
            return None

    def fetch_quote(self, ticker: str) -> Optional[Dict]:
        """
        실시간 시세 조회

        Args:
            ticker: 티커 심볼

        Returns:
            Dictionary with quote data
        """
        if not self.available:
            return None

        self._rate_limit()

        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': ticker,
                'apikey': self.api_key,
            }

            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'Global Quote' not in data:
                return None

            quote = data['Global Quote']

            return {
                'symbol': quote.get('01. symbol'),
                'price': float(quote.get('05. price', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
            }

        except requests.exceptions.RequestException as e:
            logger.error("AlphaVantage quote request error: %s", e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("AlphaVantage quote parse error: %s", e)
            return None


class PolygonSource:
    """
    Polygon.io API를 통한 미국 시장 데이터 수집

    장점:
    - 기관투자자급 데이터
    - 분단위 데이터 (1분, 5분, 15분)
    - 옵션 데이터, Greeks
    - 뉴스 & 센티먼트

    Rate Limit: 5 calls/min (무료 tier)
    Note: 무료 tier는 15분 지연 데이터
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            self.available = False
        else:
            self.available = True

        self.session = requests.Session()

    def fetch_daily_data(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        일봉 데이터 수집

        Args:
            ticker: 티커 심볼
            days: 데이터 기간

        Returns:
            DataFrame with OHLCV data
        """
        if not self.available:
            return None

        try:
            # 날짜 계산
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apiKey': self.api_key,
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK':
                return None

            if 'results' not in data or not data['results']:
                return None

            # DataFrame 생성
            results = data['results']
            df = pd.DataFrame(results)

            # 컬럼명 변경
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
            })

            # 타임스탬프를 datetime으로
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # 필요한 컬럼만 선택
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            return df

        except requests.exceptions.RequestException as e:
            logger.error("Polygon request error: %s", e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Polygon parse error: %s", e)
            return None


def test_alpha_vantage():
    """Alpha Vantage 테스트"""
    print("\n" + "="*70)
    print("🧪 Testing Alpha Vantage API")
    print("="*70)

    source = AlphaVantageSource()

    if not source.available:
        print("❌ API key not found")
        return

    # 실시간 시세 테스트
    print("\n1️⃣  Real-time Quote")
    print("-" * 70)
    ticker = 'AAPL'
    print(f"   Testing {ticker}...")

    quote = source.fetch_quote(ticker)
    if quote:
        print(f"   ✅ Price: ${quote['price']:.2f}")
        print(f"      Change: {quote['change_percent']}")
        print(f"      Volume: {quote['volume']:,}")
    else:
        print(f"   ❌ Failed")

    # 일봉 데이터 테스트
    print("\n2️⃣  Daily Historical Data")
    print("-" * 70)
    print(f"   Testing {ticker} (last 100 days)...")

    data = source.fetch_daily_data(ticker, outputsize='compact')
    if data is not None and not data.empty:
        print(f"   ✅ Data collected: {len(data)} days")
        print(f"      Latest close: ${data['Close'].iloc[-1]:.2f}")
        print(f"      Date range: {data.index[0].date()} to {data.index[-1].date()}")
    else:
        print(f"   ❌ Failed")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_alpha_vantage()
