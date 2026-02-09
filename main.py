#!/usr/bin/env python3
"""
Financial Indicators Collection System - Main Entry Point
금융 지표 수집 시스템 메인 실행 파일
"""

import logging

from fi_app.cli import parse_args, determine_lookback_days
from fi_app.collector_runner import collect_data
from fi_app.logging_config import configure_logging
from fi_app.output import convert_to_native_types, save_results, print_summary

logger = logging.getLogger(__name__)

APP_FATAL_ERRORS = (ValueError, TypeError, KeyError, RuntimeError, OSError)


__all__ = [
    "parse_args",
    "determine_lookback_days",
    "configure_logging",
    "collect_data",
    "convert_to_native_types",
    "save_results",
    "print_summary",
    "main",
]


def main() -> int:
    """메인 실행 함수"""
    configure_logging()
    args = parse_args()

    try:
        # 데이터 수집
        results = collect_data(args)

        # 결과 저장
        save_results(results)

        # 요약 출력
        print_summary(results)

        logger.info("Collection completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Collection interrupted by user")
        return 1
    except APP_FATAL_ERRORS as e:
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
