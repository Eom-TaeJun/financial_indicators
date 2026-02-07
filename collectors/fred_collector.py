#!/usr/bin/env python3
"""
FRED Collector - 거시경제 지표 수집
Federal Reserve Economic Data (FRED) API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time

from config import FRED_API_KEY, FRED_SERIES


class FREDCollector:
    """FRED API를 통한 거시경제 지표 수집기"""

    def __init__(self, api_key: str = None, lookback_days: int = 90):
        """
        Args:
            api_key: FRED API 키 (없으면 환경변수에서 로드)
            lookback_days: 데이터 수집 기간 (일)
        """
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError("FRED_API_KEY is required. Get one from https://fred.stlouisfed.org/docs/api/api_key.html")

        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)

    def fetch_series(self, series_id: str, series_name: str) -> Optional[pd.DataFrame]:
        """
        단일 FRED 시리즈 데이터 수집

        Args:
            series_id: FRED 시리즈 ID (예: 'DFF')
            series_name: 시리즈 이름 (예: 'fed_funds')

        Returns:
            DataFrame with date and value columns
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': self.start_date.strftime('%Y-%m-%d'),
            'observation_end': self.end_date.strftime('%Y-%m-%d'),
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'observations' not in data:
                print(f"   ⚠️  {series_name}: No observations found")
                return None

            observations = data['observations']
            if not observations:
                print(f"   ⚠️  {series_name}: Empty data")
                return None

            # DataFrame 생성
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].dropna()

            if df.empty:
                print(f"   ⚠️  {series_name}: All values are NaN")
                return None

            df = df.rename(columns={'value': series_name})
            df = df.set_index('date')

            print(f"   ✅ {series_name}: {len(df)} data points")
            return df

        except requests.exceptions.RequestException as e:
            print(f"   ❌ {series_name}: Request failed - {e}")
            return None
        except Exception as e:
            print(f"   ❌ {series_name}: Error - {e}")
            return None

    def collect_all(self, rate_limit_delay: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        모든 FRED 시리즈 데이터 수집

        Args:
            rate_limit_delay: API 호출 간 대기 시간 (초)

        Returns:
            Dictionary of {series_name: DataFrame}
        """
        print(f"\n📊 Collecting FRED data (last {self.lookback_days} days)...")
        print(f"   Period: {self.start_date.date()} to {self.end_date.date()}\n")

        results = {}
        total = len(FRED_SERIES)

        for idx, (series_name, series_id) in enumerate(FRED_SERIES.items(), 1):
            print(f"   [{idx}/{total}] {series_name} ({series_id})")
            df = self.fetch_series(series_id, series_name)

            if df is not None:
                results[series_name] = df

            # Rate limiting
            if idx < total:
                time.sleep(rate_limit_delay)

        print(f"\n✅ FRED collection complete: {len(results)}/{total} series\n")
        return results

    def get_latest_values(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        각 시리즈의 최신 값 추출

        Args:
            data: collect_all()의 결과

        Returns:
            Dictionary of {series_name: latest_value}
        """
        latest = {}
        for series_name, df in data.items():
            if not df.empty:
                latest[series_name] = df[series_name].iloc[-1]
        return latest

    def combine_to_dataframe(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        모든 시리즈를 하나의 DataFrame으로 결합

        Args:
            data: collect_all()의 결과

        Returns:
            Combined DataFrame with all series
        """
        if not data:
            return pd.DataFrame()

        # 모든 데이터를 날짜 기준으로 병합
        combined = pd.DataFrame()
        for series_name, df in data.items():
            if combined.empty:
                combined = df
            else:
                combined = combined.join(df, how='outer')

        # Forward fill (최신 값으로 채우기)
        combined = combined.fillna(method='ffill')

        return combined

    def calculate_liquidity_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        유동성 지표 계산

        Net Liquidity = Fed Assets - RRP - TGA

        Args:
            data: collect_all()의 결과

        Returns:
            Dictionary with liquidity metrics
        """
        latest = self.get_latest_values(data)

        # FRED 데이터 단위 변환
        # fed_assets: millions -> billions
        # rrp: billions (그대로)
        # tga: billions (그대로)

        fed_assets_b = latest.get('fed_assets', 0) / 1000  # millions to billions
        rrp_b = latest.get('rrp', 0)
        tga_b = latest.get('tga', 0)

        net_liquidity = fed_assets_b - rrp_b - tga_b

        return {
            'fed_assets_billions': fed_assets_b,
            'rrp_billions': rrp_b,
            'tga_billions': tga_b,
            'net_liquidity_billions': net_liquidity,
        }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import os
    import sys

    # API 키 확인
    if not os.getenv('FRED_API_KEY'):
        print("❌ FRED_API_KEY not found in environment variables")
        print("   Get your API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    # 수집기 생성
    collector = FREDCollector(lookback_days=30)

    # 데이터 수집
    data = collector.collect_all()

    # 최신 값 출력
    print("\n📈 Latest Values:")
    print("=" * 60)
    latest = collector.get_latest_values(data)
    for key, value in latest.items():
        print(f"   {key:20s}: {value:>10.2f}")

    # 유동성 지표 출력
    print("\n💧 Liquidity Metrics:")
    print("=" * 60)
    liquidity = collector.calculate_liquidity_metrics(data)
    for key, value in liquidity.items():
        print(f"   {key:30s}: ${value:>10.2f}B")

    # 결합된 DataFrame 저장
    combined = collector.combine_to_dataframe(data)
    output_file = f"data/fred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs('data', exist_ok=True)
    combined.to_csv(output_file)
    print(f"\n💾 Saved to: {output_file}")
