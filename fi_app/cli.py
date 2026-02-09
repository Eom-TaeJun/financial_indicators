#!/usr/bin/env python3
"""CLI helpers for financial_indicators entrypoint."""

from __future__ import annotations

import argparse

from config import (
    DEFAULT_LOOKBACK_DAYS,
    QUICK_LOOKBACK_DAYS,
    FULL_LOOKBACK_DAYS,
)


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
