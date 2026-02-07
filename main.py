#!/usr/bin/env python3
"""
Financial Indicators Collection System - Main Entry Point
금융 지표 수집 시스템 메인 실행 파일
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

from collectors import FREDCollector, MarketCollector, CryptoCollector, KoreaCollector
from config import (
    DEFAULT_LOOKBACK_DAYS,
    QUICK_LOOKBACK_DAYS,
    FULL_LOOKBACK_DAYS,
    DATA_DIR,
    OUTPUT_DIR,
    MARKET_TICKERS,
    CRYPTO_TICKERS,
    KOREA_TICKERS,
)
from db_manager import DatabaseManager


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Financial Indicators Collection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # 모든 지표 수집 (90일)
  python main.py --quick            # 빠른 수집 (30일)
  python main.py --full             # 전체 수집 (1년)
  python main.py --fred-only        # FRED만 수집
  python main.py --market-only      # 시장 데이터만
  python main.py --crypto-only      # 암호화폐만
  python main.py --korea-only       # 한국 시장만
        """
    )

    # 수집 범위
    parser.add_argument('--quick', action='store_true',
                        help='빠른 수집 (30일 데이터)')
    parser.add_argument('--full', action='store_true',
                        help='전체 수집 (1년 데이터)')
    parser.add_argument('--days', type=int,
                        help='사용자 지정 기간 (일)')

    # 수집 대상 선택
    parser.add_argument('--fred-only', action='store_true',
                        help='FRED 데이터만 수집')
    parser.add_argument('--market-only', action='store_true',
                        help='시장 데이터만 수집')
    parser.add_argument('--crypto-only', action='store_true',
                        help='암호화폐 데이터만 수집')
    parser.add_argument('--korea-only', action='store_true',
                        help='한국 시장 데이터만 수집')

    # 시장 데이터 세부 옵션
    parser.add_argument('--no-companies', action='store_true',
                        help='개별 기업 제외 (ETF만 수집)')
    parser.add_argument('--no-etfs', action='store_true',
                        help='ETF 제외 (기업만 수집)')

    return parser.parse_args()


def determine_lookback_days(args) -> int:
    """수집 기간 결정"""
    if args.days:
        return args.days
    elif args.quick:
        return QUICK_LOOKBACK_DAYS
    elif args.full:
        return FULL_LOOKBACK_DAYS
    else:
        return DEFAULT_LOOKBACK_DAYS


def collect_data(args) -> Dict[str, Any]:
    """데이터 수집 실행"""
    lookback_days = determine_lookback_days(args)

    print("\n" + "="*70)
    print("📊 FINANCIAL INDICATORS COLLECTION SYSTEM")
    print("="*70)
    print(f"Collection Period: {lookback_days} days")
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

    # 수집 대상 결정
    collect_all = not any([args.fred_only, args.market_only, args.crypto_only, args.korea_only])

    # FRED 수집
    if collect_all or args.fred_only:
        try:
            print("\n" + "="*70)
            print("1️⃣  FRED DATA COLLECTION")
            print("="*70)
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
        except Exception as e:
            print(f"❌ FRED collection failed: {e}")
            results['summary']['fred'] = {'success': False, 'error': str(e)}

    # Market 수집
    if collect_all or args.market_only:
        try:
            print("\n" + "="*70)
            print("2️⃣  MARKET DATA COLLECTION")
            print("="*70)
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
        except Exception as e:
            print(f"❌ Market collection failed: {e}")
            results['summary']['market'] = {'success': False, 'error': str(e)}

    # Crypto 수집
    if collect_all or args.crypto_only:
        try:
            print("\n" + "="*70)
            print("3️⃣  CRYPTO & RWA DATA COLLECTION")
            print("="*70)
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
        except Exception as e:
            print(f"❌ Crypto collection failed: {e}")
            results['summary']['crypto'] = {'success': False, 'error': str(e)}

    # Korea 수집
    if collect_all or args.korea_only:
        try:
            print("\n" + "="*70)
            print("4️⃣  KOREA MARKET DATA COLLECTION")
            print("="*70)
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
        except Exception as e:
            print(f"❌ Korea collection failed: {e}")
            results['summary']['korea'] = {'success': False, 'error': str(e)}

    return results


def convert_to_native_types(obj):
    """numpy 타입을 Python native 타입으로 변환"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj


def save_results(results: Dict[str, Any], save_to_db: bool = True) -> None:
    """결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 디렉토리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    # ========================================================================
    # 1. DATABASE SAVING
    # ========================================================================
    if save_to_db:
        try:
            print("\n📦 Saving to database...")
            db = DatabaseManager()

            # Save collection run metadata
            collection_run_id = db.save_collection_run(results)
            print(f"✅ Collection run saved (ID: {collection_run_id})")

            # Save FRED data
            if 'fred' in results['data']:
                db.save_fred_data(collection_run_id, results['data']['fred']['raw_data'])

            # Save Market data
            if 'market' in results['data']:
                # Category mapping from config
                category_map = {}
                for category, tickers in MARKET_TICKERS.items():
                    for ticker in tickers.keys():
                        category_map[ticker] = category

                db.save_market_data(
                    collection_run_id,
                    results['data']['market']['raw_data'],
                    category_map
                )

            # Save Crypto data
            if 'crypto' in results['data']:
                # Category mapping from config
                category_map = {}
                for category, tickers in CRYPTO_TICKERS.items():
                    for ticker in tickers.keys():
                        category_map[ticker] = category

                db.save_crypto_data(
                    collection_run_id,
                    results['data']['crypto']['raw_data'],
                    category_map
                )

            # Save Korea data
            if 'korea' in results['data']:
                # Category mapping from config
                category_map = {}
                for category, tickers in KOREA_TICKERS.items():
                    for ticker in tickers.keys():
                        category_map[ticker] = category

                db.save_korea_data(
                    collection_run_id,
                    results['data']['korea']['raw_data'],
                    category_map
                )

            # Print DB stats
            stats = db.get_db_stats()
            print(f"\n📊 Database Statistics:")
            print(f"   Total records: {stats['fred_data_count'] + stats['market_data_count'] + stats['crypto_data_count'] + stats['korea_data_count']:,}")
            print(f"   Database size: {stats['db_size_mb']:.2f} MB")
            print(f"   Database path: {db.db_path}")

        except Exception as e:
            print(f"⚠️  Database save failed: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # 2. FILE SAVING (CSV & JSON)
    # ========================================================================

    # JSON 저장 (DataFrame은 제외)
    json_results = {
        'metadata': results['metadata'],
        'summary': results['summary'],
    }

    # 숫자 데이터만 JSON에 포함
    if 'fred' in results['data']:
        json_results['fred'] = {
            'latest_values': results['data']['fred']['latest_values'],
            'liquidity_metrics': results['data']['fred']['liquidity_metrics'],
        }

    if 'market' in results['data']:
        json_results['market'] = {
            'latest_prices': results['data']['market']['latest_prices'],
        }

    if 'crypto' in results['data']:
        json_results['crypto'] = {
            'latest_prices': results['data']['crypto']['latest_prices'],
            'volatility': results['data']['crypto']['volatility'],
        }

    if 'korea' in results['data']:
        json_results['korea'] = {
            'latest_prices': results['data']['korea']['latest_prices'],
            'kospi_metrics': results['data']['korea']['kospi_metrics'],
        }

    # numpy 타입 변환
    json_results = convert_to_native_types(json_results)

    # JSON 저장
    json_file = os.path.join(OUTPUT_DIR, f'indicators_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved: {json_file}")

    # CSV 저장 (각 카테고리별)
    if 'fred' in results['data']:
        from collectors import FREDCollector
        collector = FREDCollector()
        combined_fred = collector.combine_to_dataframe(results['data']['fred']['raw_data'])
        csv_file = os.path.join(DATA_DIR, f'fred_{timestamp}.csv')
        combined_fred.to_csv(csv_file)
        print(f"✅ FRED CSV saved: {csv_file}")

    if 'market' in results['data']:
        import pandas as pd
        prices = pd.DataFrame()
        for ticker, df in results['data']['market']['raw_data'].items():
            if not df.empty and 'Close' in df.columns:
                prices[ticker] = df['Close']
        csv_file = os.path.join(DATA_DIR, f'market_{timestamp}.csv')
        prices.to_csv(csv_file)
        print(f"✅ Market CSV saved: {csv_file}")

    if 'crypto' in results['data']:
        import pandas as pd
        prices = pd.DataFrame()
        for ticker, df in results['data']['crypto']['raw_data'].items():
            if not df.empty and 'Close' in df.columns:
                prices[ticker] = df['Close']
        csv_file = os.path.join(DATA_DIR, f'crypto_{timestamp}.csv')
        prices.to_csv(csv_file)
        print(f"✅ Crypto CSV saved: {csv_file}")

    if 'korea' in results['data']:
        import pandas as pd
        prices = pd.DataFrame()
        for ticker, df in results['data']['korea']['raw_data'].items():
            if not df.empty and 'Close' in df.columns:
                prices[ticker] = df['Close']
        csv_file = os.path.join(DATA_DIR, f'korea_{timestamp}.csv')
        prices.to_csv(csv_file)
        print(f"✅ Korea CSV saved: {csv_file}")


def print_summary(results: Dict[str, Any]) -> None:
    """결과 요약 출력"""
    print("\n" + "="*70)
    print("📈 COLLECTION SUMMARY")
    print("="*70)

    for category, summary in results['summary'].items():
        status = "✅" if summary.get('success') else "❌"
        print(f"\n{status} {category.upper()}")

        if summary.get('success'):
            if 'series_count' in summary:
                print(f"   Series collected: {summary['series_count']}")
            if 'ticker_count' in summary:
                print(f"   Tickers collected: {summary['ticker_count']}")
            if 'asset_count' in summary:
                print(f"   Assets collected: {summary['asset_count']}")
        else:
            print(f"   Error: {summary.get('error', 'Unknown error')}")

    # 주요 지표 출력
    if 'fred' in results['data']:
        print("\n💡 Key FRED Indicators:")
        latest = results['data']['fred']['latest_values']
        print(f"   Fed Funds Rate: {latest.get('fed_funds', 0):.2f}%")
        print(f"   10Y Treasury: {latest.get('treasury_10y', 0):.2f}%")
        print(f"   10Y-2Y Spread: {latest.get('spread_10y2y', 0):.2f}%")

        liquidity = results['data']['fred']['liquidity_metrics']
        print(f"\n💧 Liquidity:")
        print(f"   Net Liquidity: ${liquidity.get('net_liquidity_billions', 0):,.1f}B")

    if 'market' in results['data']:
        print("\n📊 Major Indices:")
        prices = results['data']['market']['latest_prices']
        for ticker in ['SPY', 'QQQ', 'DIA']:
            if ticker in prices:
                print(f"   {ticker}: ${prices[ticker]:.2f}")

    if 'crypto' in results['data']:
        print("\n🪙 Crypto:")
        prices = results['data']['crypto']['latest_prices']
        for ticker in ['BTC-USD', 'ETH-USD']:
            if ticker in prices:
                print(f"   {ticker}: ${prices[ticker]:,.2f}")

    print("\n" + "="*70)


def main():
    """메인 실행 함수"""
    args = parse_args()

    try:
        # 데이터 수집
        results = collect_data(args)

        # 결과 저장
        save_results(results)

        # 요약 출력
        print_summary(results)

        print("\n✅ Collection completed successfully!\n")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
