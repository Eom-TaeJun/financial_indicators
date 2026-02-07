#!/usr/bin/env python3
"""
Data Sources Configuration
각 테마별 최적의 데이터 소스 및 fallback 정의
"""

from enum import Enum
from typing import Dict, List


class DataSource(Enum):
    """데이터 소스 열거형"""
    # 거시경제
    FRED = "fred"                    # Federal Reserve Economic Data

    # 미국 주식
    YFINANCE = "yfinance"            # Yahoo Finance
    ALPHA_VANTAGE = "alpha_vantage"  # Alpha Vantage API
    POLYGON = "polygon"              # Polygon.io
    FINNHUB = "finnhub"              # Finnhub
    IEX = "iex"                      # IEX Cloud

    # 암호화폐
    COINGECKO = "coingecko"          # CoinGecko API (무료, 추천)
    COINMARKETCAP = "coinmarketcap"  # CoinMarketCap API
    BINANCE = "binance"              # Binance API
    COINBASE = "coinbase"            # Coinbase API

    # 한국 시장
    FINANCE_DATA_READER = "fdr"      # FinanceDataReader (한국 전용)
    KRX = "krx"                      # 한국거래소
    NAVER = "naver"                  # 네이버 증권
    PYKRX = "pykrx"                  # pykrx 라이브러리


# 각 테마별 데이터 소스 우선순위
DATA_SOURCE_PRIORITY = {
    'fred': [
        DataSource.FRED,  # Primary (only source)
    ],

    'us_market': [
        DataSource.ALPHA_VANTAGE,  # Primary (higher quality, API key needed)
        DataSource.POLYGON,        # Secondary (API key needed)
        DataSource.YFINANCE,       # Tertiary (free, no API key)
    ],

    'crypto': [
        DataSource.COINGECKO,      # Primary (free, no API key, 추천)
        DataSource.BINANCE,        # Secondary (high quality, free)
        DataSource.YFINANCE,       # Tertiary (fallback)
    ],

    'korea': [
        DataSource.FINANCE_DATA_READER,  # Primary (한국 전용, 무료)
        DataSource.PYKRX,                # Secondary (KRX 공식 데이터)
        DataSource.YFINANCE,             # Tertiary (fallback)
    ],
}


# API 엔드포인트 및 설정
API_CONFIG = {
    # Alpha Vantage
    'alpha_vantage': {
        'base_url': 'https://www.alphavantage.co/query',
        'requires_key': True,
        'env_var': 'ALPHA_VANTAGE_API_KEY',
        'rate_limit': '5 calls/min (free tier)',
        'signup_url': 'https://www.alphavantage.co/support/#api-key',
    },

    # Polygon.io
    'polygon': {
        'base_url': 'https://api.polygon.io',
        'requires_key': True,
        'env_var': 'POLYGON_API_KEY',
        'rate_limit': '5 calls/min (free tier)',
        'signup_url': 'https://polygon.io/dashboard/signup',
    },

    # CoinGecko (무료, 추천)
    'coingecko': {
        'base_url': 'https://api.coingecko.com/api/v3',
        'requires_key': False,  # 무료 tier는 API key 불필요
        'rate_limit': '50 calls/min (free tier)',
        'docs_url': 'https://www.coingecko.com/en/api/documentation',
    },

    # Binance
    'binance': {
        'base_url': 'https://api.binance.com',
        'requires_key': False,  # 공개 데이터는 API key 불필요
        'rate_limit': '1200 requests/min',
        'docs_url': 'https://binance-docs.github.io/apidocs/',
    },

    # FinanceDataReader
    'fdr': {
        'library': 'FinanceDataReader',
        'install_cmd': 'pip install finance-datareader',
        'docs_url': 'https://github.com/FinanceData/FinanceDataReader',
        'note': '한국 시장 전용, 무료',
    },

    # pykrx
    'pykrx': {
        'library': 'pykrx',
        'install_cmd': 'pip install pykrx',
        'docs_url': 'https://github.com/sharebook-kr/pykrx',
        'note': 'KRX 공식 데이터, 무료',
    },
}


# 데이터 품질 평가 (1-5 scale)
DATA_QUALITY_RATING = {
    # 미국 주식
    ('us_market', DataSource.ALPHA_VANTAGE): {
        'quality': 5,
        'reliability': 5,
        'coverage': 5,
        'cost': 2,  # Free tier 제한적
    },
    ('us_market', DataSource.POLYGON): {
        'quality': 5,
        'reliability': 5,
        'coverage': 5,
        'cost': 2,
    },
    ('us_market', DataSource.YFINANCE): {
        'quality': 3,
        'reliability': 3,
        'coverage': 4,
        'cost': 5,  # 완전 무료
    },

    # 암호화폐
    ('crypto', DataSource.COINGECKO): {
        'quality': 5,
        'reliability': 5,
        'coverage': 5,
        'cost': 5,  # 완전 무료
    },
    ('crypto', DataSource.BINANCE): {
        'quality': 5,
        'reliability': 5,
        'coverage': 4,
        'cost': 5,
    },
    ('crypto', DataSource.YFINANCE): {
        'quality': 3,
        'reliability': 2,
        'coverage': 3,
        'cost': 5,
    },

    # 한국 시장
    ('korea', DataSource.FINANCE_DATA_READER): {
        'quality': 5,
        'reliability': 5,
        'coverage': 5,
        'cost': 5,  # 완전 무료
    },
    ('korea', DataSource.PYKRX): {
        'quality': 5,
        'reliability': 5,
        'coverage': 5,
        'cost': 5,
    },
    ('korea', DataSource.YFINANCE): {
        'quality': 3,
        'reliability': 3,
        'coverage': 3,
        'cost': 5,
    },
}


def get_recommended_sources(theme: str) -> List[DataSource]:
    """
    테마별 추천 데이터 소스 반환

    Args:
        theme: 'fred', 'us_market', 'crypto', 'korea'

    Returns:
        List of DataSource in priority order
    """
    return DATA_SOURCE_PRIORITY.get(theme, [DataSource.YFINANCE])


def get_api_config(source: DataSource) -> Dict:
    """
    데이터 소스의 API 설정 반환

    Args:
        source: DataSource enum

    Returns:
        API configuration dictionary
    """
    return API_CONFIG.get(source.value, {})


def print_data_source_guide():
    """데이터 소스 가이드 출력"""
    print("\n" + "="*70)
    print("📊 DATA SOURCE GUIDE")
    print("="*70)

    for theme, sources in DATA_SOURCE_PRIORITY.items():
        print(f"\n🎯 {theme.upper()}")
        print("-" * 70)

        for idx, source in enumerate(sources, 1):
            quality = DATA_QUALITY_RATING.get((theme, source), {})
            config = API_CONFIG.get(source.value, {})

            print(f"\n  {idx}. {source.value.upper()}")

            if quality:
                print(f"     Quality: {'⭐' * quality.get('quality', 0)}")
                print(f"     Reliability: {'⭐' * quality.get('reliability', 0)}")
                print(f"     Cost: {'💰' * (6 - quality.get('cost', 0))}")

            if config:
                if config.get('requires_key'):
                    print(f"     ⚠️  Requires API Key: {config.get('env_var')}")
                    print(f"     📝 Sign up: {config.get('signup_url')}")
                else:
                    print(f"     ✅ No API Key Required")

                if 'rate_limit' in config:
                    print(f"     ⏱️  Rate Limit: {config['rate_limit']}")

    print("\n" + "="*70)


if __name__ == "__main__":
    # 가이드 출력
    print_data_source_guide()

    # 추천 소스 확인
    print("\n\n📌 RECOMMENDED SETUP (무료):")
    print("="*70)
    print("1. FRED: FRED_API_KEY (무료)")
    print("   → https://fred.stlouisfed.org/docs/api/api_key.html")
    print("\n2. US Market: yfinance (무료, API key 불필요)")
    print("   또는 Alpha Vantage API (더 나은 품질)")
    print("\n3. Crypto: CoinGecko (무료, API key 불필요, 추천 ⭐)")
    print("   → https://www.coingecko.com/en/api/documentation")
    print("\n4. Korea: FinanceDataReader (무료)")
    print("   → pip install finance-datareader")
    print("\n" + "="*70)
