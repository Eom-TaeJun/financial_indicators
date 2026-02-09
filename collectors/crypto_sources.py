#!/usr/bin/env python3
"""
Crypto Data Sources
암호화폐 데이터 소스 (CoinGecko, Binance, yfinance)
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class CoinGeckoSource:
    """
    CoinGecko API를 통한 암호화폐 데이터 수집

    장점:
    - 무료, API key 불필요
    - 높은 품질과 안정성
    - 광범위한 코인 커버리지

    Rate Limit: 50 calls/min (free tier)
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # CoinGecko ID 매핑
    COIN_IDS = {
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'BNB-USD': 'binancecoin',
        'SOL-USD': 'solana',
        'XRP-USD': 'ripple',
        'USDC-USD': 'usd-coin',
        'USDT-USD': 'tether',
        'DAI-USD': 'dai',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
        })

    def fetch_historical_data(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        과거 데이터 수집

        Args:
            ticker: 'BTC-USD' 형식
            days: 데이터 기간 (일)

        Returns:
            DataFrame with OHLCV data
        """
        coin_id = self.COIN_IDS.get(ticker)
        if not coin_id:
            return None

        try:
            # Market chart 엔드포인트 사용
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily',
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'prices' not in data:
                return None

            # DataFrame 생성
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # Volume 추가 (있는 경우)
            if 'total_volumes' in data:
                volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
                volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
                volumes = volumes.set_index('timestamp')
                df = df.join(volumes)

            # OHLC는 CoinGecko의 OHLC 엔드포인트로 별도 수집 필요
            # 무료 tier에서는 daily만 가능
            return df

        except requests.exceptions.RequestException as e:
            logger.error("CoinGecko request error for %s: %s", ticker, e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("CoinGecko parse error for %s: %s", ticker, e)
            return None

    def fetch_current_price(self, ticker: str) -> Optional[float]:
        """
        현재 가격 조회

        Args:
            ticker: 'BTC-USD' 형식

        Returns:
            Current price in USD
        """
        coin_id = self.COIN_IDS.get(ticker)
        if not coin_id:
            return None

        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return data.get(coin_id, {}).get('usd')

        except requests.exceptions.RequestException as e:
            logger.error("CoinGecko price request error for %s: %s", ticker, e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("CoinGecko price parse error for %s: %s", ticker, e)
            return None


class BinanceSource:
    """
    Binance API를 통한 암호화폐 데이터 수집

    장점:
    - 무료, API key 불필요 (공개 데이터)
    - 높은 품질
    - 실시간 데이터

    Rate Limit: 1200 requests/min
    """

    BASE_URL = "https://api.binance.com/api/v3"

    # Binance 심볼 매핑
    SYMBOLS = {
        'BTC-USD': 'BTCUSDT',
        'ETH-USD': 'ETHUSDT',
        'BNB-USD': 'BNBUSDT',
        'SOL-USD': 'SOLUSDT',
        'XRP-USD': 'XRPUSDT',
    }

    def __init__(self):
        self.session = requests.Session()

    def fetch_historical_data(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        과거 데이터 수집 (Klines)

        Args:
            ticker: 'BTC-USD' 형식
            days: 데이터 기간 (일)

        Returns:
            DataFrame with OHLCV data
        """
        symbol = self.SYMBOLS.get(ticker)
        if not symbol:
            return None

        try:
            url = f"{self.BASE_URL}/klines"

            # 시작/종료 시간 계산
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            params = {
                'symbol': symbol,
                'interval': '1d',  # Daily
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000,
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            # DataFrame 생성
            # Binance klines format: [Open time, Open, High, Low, Close, Volume, ...]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col])

            # 필요한 컬럼만 선택
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            return df

        except requests.exceptions.RequestException as e:
            logger.error("Binance request error for %s: %s", ticker, e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Binance parse error for %s: %s", ticker, e)
            return None


def test_crypto_sources():
    """데이터 소스 테스트"""
    print("\n" + "="*70)
    print("🧪 Testing Crypto Data Sources")
    print("="*70)

    # CoinGecko 테스트
    print("\n1️⃣  CoinGecko API")
    print("-" * 70)
    cg = CoinGeckoSource()

    ticker = 'BTC-USD'
    print(f"   Testing {ticker}...")

    # 현재 가격
    price = cg.fetch_current_price(ticker)
    if price:
        print(f"   ✅ Current price: ${price:,.2f}")
    else:
        print(f"   ❌ Failed to fetch current price")

    # 과거 데이터
    data = cg.fetch_historical_data(ticker, days=30)
    if data is not None and not data.empty:
        print(f"   ✅ Historical data: {len(data)} days")
        print(f"      Latest: ${data['Close'].iloc[-1]:,.2f}")
    else:
        print(f"   ❌ Failed to fetch historical data")

    # Binance 테스트
    print("\n2️⃣  Binance API")
    print("-" * 70)
    binance = BinanceSource()

    data = binance.fetch_historical_data(ticker, days=30)
    if data is not None and not data.empty:
        print(f"   ✅ Historical data: {len(data)} days")
        print(f"      Latest: ${data['Close'].iloc[-1]:,.2f}")
        print(f"      Has OHLC: ✓")
    else:
        print(f"   ❌ Failed to fetch historical data")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_crypto_sources()
