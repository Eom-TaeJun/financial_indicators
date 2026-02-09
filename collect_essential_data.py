#!/usr/bin/env python3
"""
Essential Data Collection - 365 Days
핵심 데이터만 빠르게 수집 (섹터 분석용)
"""

import logging
import sqlite3
from datetime import datetime

from collectors import FREDCollector, MarketCollector, CryptoCollector
from db_manager import DatabaseManager
from config import MARKET_TICKERS, CRYPTO_TICKERS

logger = logging.getLogger(__name__)
COLLECTION_ERRORS = (ValueError, TypeError, KeyError, RuntimeError, OSError)
DB_SAVE_ERRORS = (sqlite3.Error, ValueError, TypeError, KeyError, RuntimeError, OSError)


def collect_essential_data():
    """핵심 데이터만 365일 수집"""

    lookback_days = 365
    db = DatabaseManager()

    logger.info("=" * 70)
    logger.info("ESSENTIAL DATA COLLECTION - 365 DAYS")
    logger.info("=" * 70)
    logger.info("Period: %s days", lookback_days)
    logger.info("Timestamp: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

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
    logger.info("1) FRED Data")
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
        logger.info("FRED: %s series", len(fred_data))
    except COLLECTION_ERRORS as e:
        logger.exception("FRED failed: %s", e)
        results['summary']['fred'] = {'success': False, 'error': str(e)}

    # ========================================================================
    # 2. MARKET DATA (주요 ETF만 - 개별 기업 제외)
    # ========================================================================
    logger.info("2) Market ETFs (Essential)")
    market_collector = MarketCollector(lookback_days=lookback_days, use_alpha_vantage=False)
    essential_tickers = {}
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

    try:
        # 핵심 티커만 선택
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

        logger.info("Collecting %s ETFs", len(essential_tickers))
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
        logger.info("Market: %s ETFs", len(market_data))
    except COLLECTION_ERRORS as e:
        logger.exception("Market failed: %s", e)
        results['summary']['market'] = {'success': False, 'error': str(e)}

    # ========================================================================
    # 3. MAJOR US STOCKS (TOP 10 only - 대표 기업만)
    # ========================================================================
    logger.info("3) Major US Stocks (Top 10)")
    try:
        stock_data = market_collector.collect_category("TOP US STOCKS", top_stocks)

        if 'market' not in results['data']:
            results['data']['market'] = {
                'raw_data': {},
                'latest_prices': {},
                'sector_performance': {},
            }
        results['data']['market']['raw_data'].update(stock_data)
        results['data']['market']['latest_prices'] = market_collector.get_latest_prices(
            results['data']['market']['raw_data']
        )

        market_summary = results['summary'].get('market')
        if not market_summary or market_summary.get('success'):
            results['summary']['market'] = {
                'ticker_count': len(results['data']['market']['raw_data']),
                'success': True,
            }
        logger.info("Stocks: %s companies", len(stock_data))
    except COLLECTION_ERRORS as e:
        logger.warning("Stocks partially failed: %s", e)

    # ========================================================================
    # 4. CRYPTO (전체)
    # ========================================================================
    logger.info("4) Crypto & RWA")
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
        logger.info("Crypto: %s assets", len(crypto_data))
    except COLLECTION_ERRORS as e:
        logger.exception("Crypto failed: %s", e)
        results['summary']['crypto'] = {'success': False, 'error': str(e)}

    # ========================================================================
    # 5. SAVE TO DATABASE
    # ========================================================================
    logger.info("5) Saving to database")
    try:
        collection_run_id = db.save_collection_run(results)
        logger.info("Collection run ID: %s", collection_run_id)

        # FRED
        if 'fred' in results['data']:
            db.save_fred_data(collection_run_id, results['data']['fred']['raw_data'])
            logger.info("FRED data saved")

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
            logger.info("Market data saved")

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
            logger.info("Crypto data saved")

        # Stats
        stats = db.get_db_stats()
        logger.info("Database Statistics:")
        logger.info(
            "Total records: %s",
            f"{stats['fred_data_count'] + stats['market_data_count'] + stats['crypto_data_count']:,}",
        )
        logger.info("Database size: %.2f MB", stats['db_size_mb'])

    except DB_SAVE_ERRORS as e:
        logger.exception("Database save failed: %s", e)

    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    logger.info("=" * 70)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 70)

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
            logger.info("%s %s: %s", status, category.upper(), count)
        else:
            total_failed += 1
            logger.error("%s: %s", category.upper(), summary.get('error', 'Unknown error'))

    logger.info("=" * 70)
    logger.info(
        "Collection completed: %s/%s categories",
        total_success,
        total_success + total_failed,
    )
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    results = collect_essential_data()
