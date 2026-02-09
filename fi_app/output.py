#!/usr/bin/env python3
"""Output/persistence helpers for financial_indicators."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from collectors import FREDCollector
from config import DATA_DIR, OUTPUT_DIR, MARKET_TICKERS, CRYPTO_TICKERS, KOREA_TICKERS
from db_manager import DatabaseManager

logger = logging.getLogger(__name__)
DB_SAVE_ERRORS = (sqlite3.Error, ValueError, TypeError, KeyError, OSError)


def convert_to_native_types(obj: Any) -> Any:
    """numpy 타입을 Python native 타입으로 변환"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    return obj


def _build_category_map(mapping: Dict[str, Dict[str, str]], use_values: bool = False) -> Dict[str, str]:
    """Build ticker -> category map from config mapping."""
    category_map: Dict[str, str] = {}
    for category, tickers in mapping.items():
        iterable = tickers.values() if use_values else tickers.keys()
        for ticker in iterable:
            category_map[ticker] = category
    return category_map


def _save_price_csv(raw_data: Dict[str, pd.DataFrame], prefix: str, timestamp: str) -> None:
    """Save close-price matrix CSV for a raw_data payload."""
    prices = pd.DataFrame()
    for ticker, df in raw_data.items():
        if not df.empty and "Close" in df.columns:
            prices[ticker] = df["Close"]

    csv_file = os.path.join(DATA_DIR, f"{prefix}_{timestamp}.csv")
    prices.to_csv(csv_file)
    logger.info("%s CSV saved: %s", prefix.capitalize(), csv_file)


def save_results(results: Dict[str, Any], save_to_db: bool = True) -> None:
    """결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 디렉토리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    # ========================================================================
    # 1. DATABASE SAVING
    # ========================================================================
    if save_to_db:
        try:
            logger.info("Saving to database...")
            db = DatabaseManager()

            # Save collection run metadata
            collection_run_id = db.save_collection_run(results)
            logger.info("Collection run saved (ID: %s)", collection_run_id)

            # Save FRED data
            if "fred" in results["data"]:
                db.save_fred_data(collection_run_id, results["data"]["fred"]["raw_data"])

            # Save Market data
            if "market" in results["data"]:
                market_category_map = _build_category_map(MARKET_TICKERS, use_values=False)
                db.save_market_data(
                    collection_run_id,
                    results["data"]["market"]["raw_data"],
                    market_category_map,
                )

            # Save Crypto data
            if "crypto" in results["data"]:
                crypto_category_map = _build_category_map(CRYPTO_TICKERS, use_values=False)
                db.save_crypto_data(
                    collection_run_id,
                    results["data"]["crypto"]["raw_data"],
                    crypto_category_map,
                )

            # Save Korea data (config shape: {name: ticker})
            if "korea" in results["data"]:
                korea_category_map = _build_category_map(KOREA_TICKERS, use_values=True)
                db.save_korea_data(
                    collection_run_id,
                    results["data"]["korea"]["raw_data"],
                    korea_category_map,
                )

            # Print DB stats
            stats = db.get_db_stats()
            logger.info("Database Statistics:")
            logger.info(
                "Total records: %s",
                f"{stats['fred_data_count'] + stats['market_data_count'] + stats['crypto_data_count'] + stats['korea_data_count']:,}",
            )
            logger.info("Database size: %.2f MB", stats["db_size_mb"])
            logger.info("Database path: %s", db.db_path)

        except DB_SAVE_ERRORS as e:
            logger.exception("Database save failed: %s", e)

    # ========================================================================
    # 2. FILE SAVING (CSV & JSON)
    # ========================================================================

    # JSON 저장 (DataFrame은 제외)
    json_results = {
        "metadata": results["metadata"],
        "summary": results["summary"],
    }

    # 숫자 데이터만 JSON에 포함
    if "fred" in results["data"]:
        json_results["fred"] = {
            "latest_values": results["data"]["fred"]["latest_values"],
            "liquidity_metrics": results["data"]["fred"]["liquidity_metrics"],
        }

    if "market" in results["data"]:
        json_results["market"] = {
            "latest_prices": results["data"]["market"]["latest_prices"],
        }

    if "crypto" in results["data"]:
        json_results["crypto"] = {
            "latest_prices": results["data"]["crypto"]["latest_prices"],
            "volatility": results["data"]["crypto"]["volatility"],
        }

    if "korea" in results["data"]:
        json_results["korea"] = {
            "latest_prices": results["data"]["korea"]["latest_prices"],
            "kospi_metrics": results["data"]["korea"]["kospi_metrics"],
        }

    # numpy 타입 변환
    json_results = convert_to_native_types(json_results)

    # JSON 저장
    json_file = os.path.join(OUTPUT_DIR, f"indicators_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved: %s", json_file)

    # CSV 저장 (각 카테고리별)
    if "fred" in results["data"]:
        collector = FREDCollector()
        combined_fred = collector.combine_to_dataframe(results["data"]["fred"]["raw_data"])
        csv_file = os.path.join(DATA_DIR, f"fred_{timestamp}.csv")
        combined_fred.to_csv(csv_file)
        logger.info("FRED CSV saved: %s", csv_file)

    if "market" in results["data"]:
        _save_price_csv(results["data"]["market"]["raw_data"], "market", timestamp)

    if "crypto" in results["data"]:
        _save_price_csv(results["data"]["crypto"]["raw_data"], "crypto", timestamp)

    if "korea" in results["data"]:
        _save_price_csv(results["data"]["korea"]["raw_data"], "korea", timestamp)


def print_summary(results: Dict[str, Any]) -> None:
    """결과 요약 출력"""
    logger.info("=" * 70)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 70)

    for category, summary in results["summary"].items():
        status = "✅" if summary.get("success") else "❌"
        logger.info("%s %s", status, category.upper())

        if summary.get("success"):
            if "series_count" in summary:
                logger.info("Series collected: %s", summary["series_count"])
            if "ticker_count" in summary:
                logger.info("Tickers collected: %s", summary["ticker_count"])
            if "asset_count" in summary:
                logger.info("Assets collected: %s", summary["asset_count"])
        else:
            logger.error("Error: %s", summary.get("error", "Unknown error"))

    # 주요 지표 출력
    if "fred" in results["data"]:
        logger.info("Key FRED Indicators:")
        latest = results["data"]["fred"]["latest_values"]
        logger.info("Fed Funds Rate: %.2f%%", latest.get("fed_funds", 0))
        logger.info("10Y Treasury: %.2f%%", latest.get("treasury_10y", 0))
        logger.info("10Y-2Y Spread: %.2f%%", latest.get("spread_10y2y", 0))

        liquidity = results["data"]["fred"]["liquidity_metrics"]
        logger.info("Liquidity:")
        logger.info(
            "Net Liquidity: $%sB",
            f"{liquidity.get('net_liquidity_billions', 0):,.1f}",
        )

    if "market" in results["data"]:
        logger.info("Major Indices:")
        prices = results["data"]["market"]["latest_prices"]
        for ticker in ["SPY", "QQQ", "DIA"]:
            if ticker in prices:
                logger.info("%s: $%.2f", ticker, prices[ticker])

    if "crypto" in results["data"]:
        logger.info("Crypto:")
        prices = results["data"]["crypto"]["latest_prices"]
        for ticker in ["BTC-USD", "ETH-USD"]:
            if ticker in prices:
                logger.info("%s: $%s", ticker, f"{prices[ticker]:,.2f}")

    logger.info("=" * 70)
