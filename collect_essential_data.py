#!/usr/bin/env python3
"""
Essential Data Collection - 365 Days
핵심 데이터만 빠르게 수집 (섹터 분석용)
"""

import pandas as pd
from datetime import datetime
from collectors import FREDCollector, MarketCollector, CryptoCollector
from db_manager import DatabaseManager
from config import MARKET_TICKERS, CRYPTO_TICKERS


def collect_essential_data():
    """핵심 데이터만 365일 수집"""

    lookback_days = 365
    db = DatabaseManager()

    print("="*70)
    print("📊 ESSENTIAL DATA COLLECTION - 365 DAYS")
    print("="*70)
    print(f"Period: {lookback_days} days")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'lookback_days': lookback_days,
        },
        'data': {},
        'summary': {},
    }

    # ========================================================================
    # 1. FRED DATA (전체)
    # ========================================================================
    print("\n1️⃣  FRED Data...")
    try:
        fred_collector = FREDCollector(lookback_days=lookback_days)
        fred_data = fred_collector.collect_all()

        results['data']['fred'] = {
            'raw_data': fred_data,
            'latest_values': fred_collector.get_latest_values(fred_data),
            'liquidity_metrics': fred_collector.calculate_liquidity_metrics(fred_data),
        }
        results['summary']['fred'] = {
            'series_count': len(fred_data),
            'success': True,
        }
        print(f"✅ FRED: {len(fred_data)} series")
    except Exception as e:
        print(f"❌ FRED failed: {e}")
        results['summary']['fred'] = {'success': False, 'error': str(e)}

    # ========================================================================
    # 2. MARKET DATA (주요 ETF만 - 개별 기업 제외)
    # ========================================================================
    print("\n2️⃣  Market ETFs (Essential)...")
    try:
        market_collector = MarketCollector(lookback_days=lookback_days, use_alpha_vantage=False)

        # 핵심 티커만 선택
        essential_tickers = {}

        # 주요 지수
        essential_tickers.update(MARKET_TICKERS['indices'])

        # 11개 섹터 ETF (섹터 로테이션 분석용)
        essential_tickers.update(MARKET_TICKERS['sectors'])

        # 채권
        essential_tickers.update({
            'TLT': '20+ Year Treasury',
            'IEF': '7-10 Year Treasury',
            'SHY': '1-3 Year Treasury',
            'LQD': 'Investment Grade Corporate',
            'HYG': 'High Yield Corporate',
        })

        # 원자재
        essential_tickers.update({
            'GLD': 'Gold',
            'USO': 'Oil',
        })

        print(f"   Collecting {len(essential_tickers)} ETFs...")
        market_data = market_collector.collect_category("ESSENTIAL ETFS", essential_tickers)

        results['data']['market'] = {
            'raw_data': market_data,
            'latest_prices': market_collector.get_latest_prices(market_data),
            'sector_performance': market_collector.calculate_sector_performance(market_data),
        }
        results['summary']['market'] = {
            'ticker_count': len(market_data),
            'success': True,
        }
        print(f"✅ Market: {len(market_data)} ETFs")
    except Exception as e:
        print(f"❌ Market failed: {e}")
        results['summary']['market'] = {'success': False, 'error': str(e)}

    # ========================================================================
    # 3. MAJOR US STOCKS (TOP 10 only - 대표 기업만)
    # ========================================================================
    print("\n3️⃣  Major US Stocks (Top 10)...")
    try:
        top_stocks = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Alphabet',
            'NVDA': 'NVIDIA',
            'META': 'Meta',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'BRK-B': 'Berkshire Hathaway',
            'JPM': 'JPMorgan Chase',
            'V': 'Visa',
        }

        stock_data = market_collector.collect_category("TOP US STOCKS", top_stocks)
        results['data']['market']['raw_data'].update(stock_data)
        print(f"✅ Stocks: {len(stock_data)} companies")
    except Exception as e:
        print(f"⚠️  Stocks partially failed: {e}")

    # ========================================================================
    # 4. CRYPTO (전체)
    # ========================================================================
    print("\n4️⃣  Crypto & RWA...")
    try:
        crypto_collector = CryptoCollector(lookback_days=lookback_days)
        crypto_data = crypto_collector.collect_all()

        results['data']['crypto'] = {
            'raw_data': crypto_data,
            'latest_prices': crypto_collector.get_latest_prices(crypto_data),
            'volatility': crypto_collector.calculate_volatility(crypto_data),
        }
        results['summary']['crypto'] = {
            'asset_count': len(crypto_data),
            'success': True,
        }
        print(f"✅ Crypto: {len(crypto_data)} assets")
    except Exception as e:
        print(f"❌ Crypto failed: {e}")
        results['summary']['crypto'] = {'success': False, 'error': str(e)}

    # ========================================================================
    # 5. SAVE TO DATABASE
    # ========================================================================
    print("\n5️⃣  Saving to database...")
    try:
        collection_run_id = db.save_collection_run(results)
        print(f"   Collection run ID: {collection_run_id}")

        # FRED
        if 'fred' in results['data']:
            db.save_fred_data(collection_run_id, results['data']['fred']['raw_data'])
            print(f"   ✅ FRED data saved")

        # Market
        if 'market' in results['data']:
            category_map = {}
            for ticker in essential_tickers.keys():
                category_map[ticker] = 'essential_etfs'
            for ticker in top_stocks.keys():
                category_map[ticker] = 'top_stocks'

            db.save_market_data(
                collection_run_id,
                results['data']['market']['raw_data'],
                category_map
            )
            print(f"   ✅ Market data saved")

        # Crypto
        if 'crypto' in results['data']:
            category_map = {}
            for category, tickers in CRYPTO_TICKERS.items():
                for ticker in tickers.keys():
                    category_map[ticker] = category

            db.save_crypto_data(
                collection_run_id,
                results['data']['crypto']['raw_data'],
                category_map
            )
            print(f"   ✅ Crypto data saved")

        # Stats
        stats = db.get_db_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total records: {stats['fred_data_count'] + stats['market_data_count'] + stats['crypto_data_count']:,}")
        print(f"   Database size: {stats['db_size_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ Database save failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("📈 COLLECTION SUMMARY")
    print("="*70)

    total_success = 0
    total_failed = 0

    for category, summary in results['summary'].items():
        if summary.get('success'):
            total_success += 1
            status = "✅"
            if 'series_count' in summary:
                count = f"{summary['series_count']} series"
            elif 'ticker_count' in summary:
                count = f"{summary['ticker_count']} tickers"
            elif 'asset_count' in summary:
                count = f"{summary['asset_count']} assets"
            else:
                count = "unknown"
            print(f"{status} {category.upper()}: {count}")
        else:
            total_failed += 1
            print(f"❌ {category.upper()}: {summary.get('error', 'Unknown error')}")

    print("\n" + "="*70)
    print(f"✅ Collection completed: {total_success}/{total_success+total_failed} categories")
    print("="*70)

    return results


if __name__ == "__main__":
    results = collect_essential_data()
