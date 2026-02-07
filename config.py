#!/usr/bin/env python3
"""
Configuration for Financial Indicators Collection System
설정 파일
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API Keys
# ============================================================================

FRED_API_KEY = os.getenv('FRED_API_KEY', '')

# ============================================================================
# Data Collection Settings
# ============================================================================

# 기본 수집 기간 (일)
DEFAULT_LOOKBACK_DAYS = 90
QUICK_LOOKBACK_DAYS = 30
FULL_LOOKBACK_DAYS = 365

# 데이터 저장 경로
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'

# ============================================================================
# FRED Series Configuration
# ============================================================================

FRED_SERIES = {
    # 금리 (Rates)
    'fed_funds': 'DFF',                   # Effective Federal Funds Rate
    'fed_target_upper': 'DFEDTARU',       # Fed Target Upper
    'fed_target_lower': 'DFEDTARL',       # Fed Target Lower
    'treasury_3m': 'DGS3MO',              # 3-Month Treasury
    'treasury_2y': 'DGS2',                # 2-Year Treasury
    'treasury_5y': 'DGS5',                # 5-Year Treasury
    'treasury_10y': 'DGS10',              # 10-Year Treasury
    'treasury_30y': 'DGS30',              # 30-Year Treasury

    # 스프레드 (Spreads)
    'spread_10y2y': 'T10Y2Y',             # 10Y-2Y Spread
    'spread_10y3m': 'T10Y3M',             # 10Y-3M Spread
    'hy_oas': 'BAMLH0A0HYM2',             # High Yield OAS
    'ig_oas': 'BAMLC0A4CBBB',             # Investment Grade OAS

    # 인플레이션 (Inflation)
    'cpi': 'CPIAUCSL',                    # CPI All Urban
    'core_cpi': 'CPILFESL',               # Core CPI
    'pce': 'PCEPI',                       # PCE Price Index
    'core_pce': 'PCEPILFE',               # Core PCE
    'breakeven_5y': 'T5YIE',              # 5Y Breakeven Inflation
    'breakeven_10y': 'T10YIE',            # 10Y Breakeven Inflation

    # 고용 (Employment)
    'unemployment': 'UNRATE',              # Unemployment Rate
    'payrolls': 'PAYEMS',                  # Nonfarm Payrolls
    'initial_claims': 'ICSA',              # Initial Jobless Claims

    # 경제활동 (Economic Activity)
    'gdp': 'GDP',                          # GDP
    'industrial_prod': 'INDPRO',           # Industrial Production
    'retail_sales': 'RSAFS',               # Retail Sales

    # 유동성 (Liquidity)
    'rrp': 'RRPONTSYD',                    # Reverse Repo
    'tga': 'WTREGEN',                      # Treasury General Account
    'fed_assets': 'WALCL',                 # Fed Total Assets
    'reserves': 'TOTRESNS',                # Total Reserves
    'iorb': 'IORB',                        # Interest on Reserve Balances

    # 시장 (Markets)
    'vix': 'VIXCLS',                       # VIX
    'dxy': 'DTWEXBGS',                     # Dollar Index
}

# ============================================================================
# Market Tickers Configuration
# ============================================================================

MARKET_TICKERS = {
    # 주요 지수 (Major Indices)
    'indices': {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ 100',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000',
        'VTI': 'Total Stock Market',
    },

    # 섹터 ETF (Sector ETFs)
    'sectors': {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services',
    },

    # 채권 (Bonds)
    'bonds': {
        'TLT': '20+ Year Treasury',
        'IEF': '7-10 Year Treasury',
        'SHY': '1-3 Year Treasury',
        'LQD': 'Investment Grade Corporate',
        'HYG': 'High Yield Corporate',
        'TIP': 'TIPS',
        'MUB': 'Municipal Bonds',
    },

    # 원자재 (Commodities)
    'commodities': {
        'GLD': 'Gold',
        'SLV': 'Silver',
        'USO': 'Oil',
        'DBA': 'Agriculture',
        'UNG': 'Natural Gas',
    },

    # 국제 (International)
    'international': {
        'EFA': 'Developed Markets',
        'EEM': 'Emerging Markets',
        'FXI': 'China',
        'EWJ': 'Japan',
        'EWG': 'Germany',
        'EWU': 'UK',
    },

    # 테마 ETF (Thematic ETFs)
    'thematic': {
        'ARK': 'Innovation',
        'SOXX': 'Semiconductors',
        'IBB': 'Biotech',
        'XHB': 'Homebuilders',
        'XRT': 'Retail',
        'XME': 'Metals & Mining',
    },
}

# ============================================================================
# 미국 주요 기업 (Top US Companies by Sector)
# ============================================================================

US_MAJOR_COMPANIES = {
    # Technology (기술)
    'tech': {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet (Google)',
        'NVDA': 'NVIDIA',
        'META': 'Meta (Facebook)',
        'TSLA': 'Tesla',
        'AVGO': 'Broadcom',
        'ORCL': 'Oracle',
        'ADBE': 'Adobe',
        'CRM': 'Salesforce',
        'AMD': 'AMD',
        'INTC': 'Intel',
    },

    # Financials (금융)
    'financials': {
        'JPM': 'JPMorgan Chase',
        'BAC': 'Bank of America',
        'WFC': 'Wells Fargo',
        'GS': 'Goldman Sachs',
        'MS': 'Morgan Stanley',
        'C': 'Citigroup',
        'BLK': 'BlackRock',
        'SCHW': 'Charles Schwab',
        'AXP': 'American Express',
        'V': 'Visa',
        'MA': 'Mastercard',
    },

    # Healthcare (헬스케어)
    'healthcare': {
        'UNH': 'UnitedHealth',
        'JNJ': 'Johnson & Johnson',
        'LLY': 'Eli Lilly',
        'PFE': 'Pfizer',
        'ABBV': 'AbbVie',
        'TMO': 'Thermo Fisher',
        'MRK': 'Merck',
        'ABT': 'Abbott',
        'DHR': 'Danaher',
        'BMY': 'Bristol Myers Squibb',
    },

    # Energy (에너지)
    'energy': {
        'XOM': 'Exxon Mobil',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'SLB': 'Schlumberger',
        'EOG': 'EOG Resources',
        'MPC': 'Marathon Petroleum',
        'PSX': 'Phillips 66',
    },

    # Consumer Discretionary (임의소비재)
    'consumer_discretionary': {
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'HD': 'Home Depot',
        'MCD': 'McDonald\'s',
        'NKE': 'Nike',
        'SBUX': 'Starbucks',
        'LOW': 'Lowe\'s',
        'TGT': 'Target',
        'TJX': 'TJX Companies',
    },

    # Consumer Staples (필수소비재)
    'consumer_staples': {
        'WMT': 'Walmart',
        'PG': 'Procter & Gamble',
        'COST': 'Costco',
        'KO': 'Coca-Cola',
        'PEP': 'PepsiCo',
        'PM': 'Philip Morris',
        'MO': 'Altria',
    },

    # Industrials (산업재)
    'industrials': {
        'BA': 'Boeing',
        'CAT': 'Caterpillar',
        'HON': 'Honeywell',
        'UPS': 'UPS',
        'RTX': 'Raytheon',
        'LMT': 'Lockheed Martin',
        'GE': 'General Electric',
        'MMM': '3M',
    },

    # Communication Services (통신 서비스)
    'communication': {
        'META': 'Meta',
        'GOOGL': 'Alphabet',
        'DIS': 'Disney',
        'NFLX': 'Netflix',
        'CMCSA': 'Comcast',
        'T': 'AT&T',
        'VZ': 'Verizon',
    },

    # Utilities (유틸리티)
    'utilities': {
        'NEE': 'NextEra Energy',
        'DUK': 'Duke Energy',
        'SO': 'Southern Company',
        'D': 'Dominion Energy',
    },

    # Real Estate (부동산)
    'real_estate': {
        'AMT': 'American Tower',
        'PLD': 'Prologis',
        'CCI': 'Crown Castle',
        'EQIX': 'Equinix',
        'SPG': 'Simon Property',
    },
}

# ============================================================================
# Crypto & RWA Configuration
# ============================================================================

CRYPTO_TICKERS = {
    # 주요 암호화폐 (Major Cryptocurrencies)
    'majors': {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'BNB-USD': 'Binance Coin',
        'SOL-USD': 'Solana',
        'XRP-USD': 'Ripple',
    },

    # 스테이블코인 (Stablecoins) - 프록시로 관련 자산
    'stablecoins': {
        'USDC-USD': 'USD Coin',
        'USDT-USD': 'Tether',
        'DAI-USD': 'Dai',
    },

    # RWA 토큰 (Real World Assets)
    'rwa': {
        'ONDO-USD': 'Ondo Finance',
        'PAXG-USD': 'PAX Gold',
    },

    # 크립토 관련 주식 (Crypto-related Stocks)
    'crypto_stocks': {
        'COIN': 'Coinbase',
        'MSTR': 'MicroStrategy',
        'RIOT': 'Riot Platforms',
        'MARA': 'Marathon Digital',
    },
}

# ============================================================================
# Korea Market Configuration
# ============================================================================

KOREA_TICKERS = {
    # 지수 (Indices)
    'indices': {
        'KOSPI': '^KS11',
        'KOSPI200': '^KS200',
        'KOSDAQ': '^KQ11',
    },

    # 섹터 ETF (Sector ETFs)
    'sector_etfs': {
        'KODEX_Bank': '091170.KS',         # 은행
        'KODEX_Semi': '091180.KS',         # 반도체
        'KODEX_Bio': '228790.KS',          # 바이오
        'KODEX_Battery': '305720.KS',      # 2차전지
        'KODEX_Auto': '091160.KS',         # 자동차
    },

    # 대형주 (Large Caps)
    'large_caps': {
        'Samsung_Electronics': '005930.KS',
        'SK_Hynix': '000660.KS',
        'LG_Energy': '373220.KS',
        'Samsung_Bio': '207940.KS',
        'Hyundai_Motor': '005380.KS',
        'NAVER': '035420.KS',
        'Kakao': '035720.KS',
        'POSCO': '005490.KS',
    },

    # 채권 ETF (Bond ETFs)
    'bond_etfs': {
        'KODEX_KTB3Y': '153130.KS',
        'KODEX_KTB10Y': '148070.KS',
    },

    # 환율 (Currency)
    'currency': {
        'USDKRW': 'USDKRW=X',
    },
}

# ============================================================================
# Collection Options
# ============================================================================

COLLECTION_OPTIONS = {
    'fred': {
        'enabled': True,
        'series': list(FRED_SERIES.keys()),
    },
    'market': {
        'enabled': True,
        'include_indices': True,
        'include_sectors': True,
        'include_bonds': True,
        'include_commodities': True,
        'include_international': True,
        'include_thematic': True,
    },
    'us_companies': {
        'enabled': True,
        'sectors': list(US_MAJOR_COMPANIES.keys()),
    },
    'crypto': {
        'enabled': True,
        'include_majors': True,
        'include_stablecoins': True,
        'include_rwa': True,
        'include_crypto_stocks': True,
    },
    'korea': {
        'enabled': True,
        'include_indices': True,
        'include_sectors': True,
        'include_large_caps': True,
        'include_bonds': True,
    },
}
