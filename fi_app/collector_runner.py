#!/usr/bin/env python3
"""Collection orchestration for financial_indicators."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any

from collectors import FREDCollector, MarketCollector, CryptoCollector, KoreaCollector

from .cli import determine_lookback_days

logger = logging.getLogger(__name__)

COLLECTOR_ERRORS = (ValueError, TypeError, KeyError, RuntimeError, OSError)


def collect_data(args) -> Dict[str, Any]:
    """데이터 수집 실행"""
    lookback_days = determine_lookback_days(args)

    logger.info("=" * 70)
    logger.info("FINANCIAL INDICATORS COLLECTION SYSTEM")
    logger.info("Collection Period: %s days", lookback_days)
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

    # 수집 대상 결정
    collect_all = not any([args.fred_only, args.market_only, args.crypto_only, args.korea_only])

    # FRED 수집
    if collect_all or args.fred_only:
        try:
            logger.info("=" * 70)
            logger.info("1) FRED DATA COLLECTION")
            logger.info("=" * 70)
            collector = FREDCollector(lookback_days=lookback_days)
            fred_data = collector.collect_all()

            results['data']['fred'] = {
                'raw_data': fred_data,
                'latest_values': collector.get_latest_values(fred_data),
                'liquidity_metrics': collector.calculate_liquidity_metrics(fred_data),
            }
            results['summary']['fred'] = {
                'series_count': len(fred_data),
                'success': True,
            }
        except COLLECTOR_ERRORS as e:
            logger.exception("FRED collection failed: %s", e)
            results['summary']['fred'] = {'success': False, 'error': str(e)}

    # Market 수집
    if collect_all or args.market_only:
        try:
            logger.info("=" * 70)
            logger.info("2) MARKET DATA COLLECTION")
            logger.info("=" * 70)
            collector = MarketCollector(lookback_days=lookback_days)

            include_etfs = not args.no_etfs
            include_companies = not args.no_companies

            market_data = collector.collect_all(
                include_etfs=include_etfs,
                include_companies=include_companies
            )

            results['data']['market'] = {
                'raw_data': market_data,
                'latest_prices': collector.get_latest_prices(market_data),
                'returns': collector.calculate_returns(market_data),
                'sector_performance': collector.calculate_sector_performance(market_data),
            }
            results['summary']['market'] = {
                'ticker_count': len(market_data),
                'success': True,
            }
        except COLLECTOR_ERRORS as e:
            logger.exception("Market collection failed: %s", e)
            results['summary']['market'] = {'success': False, 'error': str(e)}

    # Crypto 수집
    if collect_all or args.crypto_only:
        try:
            logger.info("=" * 70)
            logger.info("3) CRYPTO & RWA DATA COLLECTION")
            logger.info("=" * 70)
            collector = CryptoCollector(lookback_days=lookback_days)
            crypto_data = collector.collect_all()

            results['data']['crypto'] = {
                'raw_data': crypto_data,
                'latest_prices': collector.get_latest_prices(crypto_data),
                'volatility': collector.calculate_volatility(crypto_data),
                'correlations': collector.calculate_correlations(crypto_data),
            }
            results['summary']['crypto'] = {
                'asset_count': len(crypto_data),
                'success': True,
            }
        except COLLECTOR_ERRORS as e:
            logger.exception("Crypto collection failed: %s", e)
            results['summary']['crypto'] = {'success': False, 'error': str(e)}

    # Korea 수집
    if collect_all or args.korea_only:
        try:
            logger.info("=" * 70)
            logger.info("4) KOREA MARKET DATA COLLECTION")
            logger.info("=" * 70)
            collector = KoreaCollector(lookback_days=lookback_days)
            korea_data = collector.collect_all()

            results['data']['korea'] = {
                'raw_data': korea_data,
                'latest_prices': collector.get_latest_prices(korea_data),
                'kospi_metrics': collector.calculate_kospi_metrics(korea_data),
                'sector_performance': collector.calculate_sector_performance(korea_data),
            }
            results['summary']['korea'] = {
                'asset_count': len(korea_data),
                'success': True,
            }
        except COLLECTOR_ERRORS as e:
            logger.exception("Korea collection failed: %s", e)
            results['summary']['korea'] = {'success': False, 'error': str(e)}

    return results
