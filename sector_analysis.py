#!/usr/bin/env python3
"""
Sector Rotation & Risk Factor Analysis
섹터 로테이션 및 리스크 팩터 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SectorScore:
    """섹터 스코어"""
    sector: str
    ticker: str
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    relative_strength: float
    volatility: float
    sharpe_ratio: float
    total_score: float
    rank: int


@dataclass
class RiskFactors:
    """리스크 팩터 분석"""
    ticker: str
    market_beta: float
    alpha: float
    r_squared: float
    volatility: float
    max_drawdown: float
    correlation_to_market: float


class SectorRotationAnalyzer:
    """
    섹터 로테이션 분석기

    Economic Cycle Framework:
    - Early Expansion: Technology, Consumer Discretionary, Financials
    - Mid Expansion: Industrials, Materials, Energy
    - Late Expansion: Energy, Materials
    - Contraction: Consumer Staples, Healthcare, Utilities
    """

    # 경기 사이클별 선호 섹터
    CYCLE_SECTORS = {
        'early_expansion': ['XLK', 'XLY', 'XLF'],  # Tech, Consumer Disc, Financials
        'mid_expansion': ['XLI', 'XLB', 'XLE'],    # Industrials, Materials, Energy
        'late_expansion': ['XLE', 'XLB'],           # Energy, Materials
        'contraction': ['XLP', 'XLV', 'XLU'],      # Staples, Healthcare, Utilities
    }

    SECTOR_NAMES = {
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
    }

    def __init__(self, sector_data: Dict[str, pd.DataFrame], market_data: pd.DataFrame):
        """
        Args:
            sector_data: {ticker: DataFrame} - 섹터 ETF 데이터
            market_data: DataFrame - 시장 지수 (SPY) 데이터
        """
        self.sector_data = sector_data
        self.market_data = market_data
        self.sector_scores = []

    def calculate_momentum(self, prices: pd.Series, period: int) -> float:
        """모멘텀 계산 (기간별 수익률)"""
        if len(prices) < period:
            return 0.0

        return (prices.iloc[-1] / prices.iloc[-period] - 1) * 100

    def calculate_relative_strength(self, sector_prices: pd.Series, market_prices: pd.Series) -> float:
        """상대 강도 계산 (vs 시장)"""
        if len(sector_prices) < 63 or len(market_prices) < 63:
            return 0.0

        # 3개월 수익률 기준
        sector_ret = (sector_prices.iloc[-1] / sector_prices.iloc[-63] - 1)
        market_ret = (market_prices.iloc[-1] / market_prices.iloc[-63] - 1)

        return ((sector_ret - market_ret) / abs(market_ret)) * 100 if market_ret != 0 else 0.0

    def calculate_sharpe_ratio(self, prices: pd.Series, risk_free_rate: float = 0.045) -> float:
        """샤프 비율 계산"""
        returns = prices.pct_change().dropna()

        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        return sharpe

    def analyze_sector(self, ticker: str, df: pd.DataFrame) -> Optional[SectorScore]:
        """단일 섹터 분석"""
        if df.empty or 'Close' not in df.columns:
            return None

        prices = df['Close']

        # 모멘텀 (1M, 3M, 6M)
        momentum_1m = self.calculate_momentum(prices, 21)
        momentum_3m = self.calculate_momentum(prices, 63)
        momentum_6m = self.calculate_momentum(prices, 126)

        # 상대 강도 (vs SPY)
        relative_strength = self.calculate_relative_strength(
            prices,
            self.market_data['Close']
        )

        # 변동성 (annualized)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0

        # 샤프 비율
        sharpe = self.calculate_sharpe_ratio(prices)

        # 종합 스코어 (가중 평균)
        # 모멘텀 40%, 상대 강도 30%, 샤프 20%, 변동성(역) 10%
        total_score = (
            momentum_3m * 0.4 +
            relative_strength * 0.3 +
            sharpe * 5 * 0.2 -  # Sharpe 정규화
            volatility * 0.1
        )

        return SectorScore(
            sector=self.SECTOR_NAMES.get(ticker, ticker),
            ticker=ticker,
            momentum_1m=momentum_1m,
            momentum_3m=momentum_3m,
            momentum_6m=momentum_6m,
            relative_strength=relative_strength,
            volatility=volatility,
            sharpe_ratio=sharpe,
            total_score=total_score,
            rank=0,  # 나중에 할당
        )

    def analyze_all_sectors(self) -> List[SectorScore]:
        """모든 섹터 분석"""
        scores = []

        for ticker, df in self.sector_data.items():
            if ticker not in self.SECTOR_NAMES:
                continue

            score = self.analyze_sector(ticker, df)
            if score:
                scores.append(score)

        # 스코어순 정렬 및 랭킹 부여
        scores.sort(key=lambda x: x.total_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1

        self.sector_scores = scores
        return scores

    def detect_economic_cycle(self) -> Tuple[str, float]:
        """
        경기 사이클 감지

        Returns:
            (cycle_phase, confidence)
        """
        if not self.sector_scores:
            self.analyze_all_sectors()

        # 상위 5개 섹터
        top_sectors = [s.ticker for s in self.sector_scores[:5]]

        # 각 사이클별 매칭 스코어 계산
        cycle_scores = {}

        for cycle, preferred_sectors in self.CYCLE_SECTORS.items():
            # 선호 섹터가 상위권에 있는지 확인
            matches = sum(1 for s in preferred_sectors if s in top_sectors)
            cycle_scores[cycle] = matches / len(preferred_sectors)

        # 가장 높은 스코어의 사이클
        best_cycle = max(cycle_scores.items(), key=lambda x: x[1])

        return best_cycle[0], best_cycle[1]

    def get_rotation_signals(self) -> Dict:
        """섹터 로테이션 신호 생성"""
        if not self.sector_scores:
            self.analyze_all_sectors()

        cycle, confidence = self.detect_economic_cycle()

        # 현재 선호 섹터
        preferred_sectors = self.CYCLE_SECTORS.get(cycle, [])

        # 실제 성과 상위 섹터
        top_performers = self.sector_scores[:3]

        # 하위 성과 섹터
        bottom_performers = self.sector_scores[-3:]

        return {
            'economic_cycle': cycle.replace('_', ' ').title(),
            'cycle_confidence': confidence,
            'preferred_sectors': [
                {
                    'ticker': ticker,
                    'name': self.SECTOR_NAMES.get(ticker, ticker)
                }
                for ticker in preferred_sectors
            ],
            'top_performers': [
                {
                    'rank': s.rank,
                    'ticker': s.ticker,
                    'sector': s.sector,
                    'score': s.total_score,
                    'momentum_3m': s.momentum_3m,
                }
                for s in top_performers
            ],
            'bottom_performers': [
                {
                    'rank': s.rank,
                    'ticker': s.ticker,
                    'sector': s.sector,
                    'score': s.total_score,
                    'momentum_3m': s.momentum_3m,
                }
                for s in bottom_performers
            ],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """섹터 스코어를 DataFrame으로 변환"""
        if not self.sector_scores:
            self.analyze_all_sectors()

        return pd.DataFrame([
            {
                'Rank': s.rank,
                'Sector': s.sector,
                'Ticker': s.ticker,
                'Score': s.total_score,
                '1M %': s.momentum_1m,
                '3M %': s.momentum_3m,
                '6M %': s.momentum_6m,
                'RS': s.relative_strength,
                'Vol %': s.volatility,
                'Sharpe': s.sharpe_ratio,
            }
            for s in self.sector_scores
        ])


class RiskFactorAnalyzer:
    """
    리스크 팩터 분석기

    Simple Factor Model (Fama-French style):
    - Market Factor (Beta)
    - Size Factor (not implemented - requires market cap data)
    - Value Factor (not implemented - requires P/B data)
    """

    def __init__(self, market_data: pd.DataFrame, risk_free_rate: float = 0.045):
        """
        Args:
            market_data: DataFrame - 시장 지수 (SPY) 데이터
            risk_free_rate: 무위험 이자율 (annual)
        """
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate

        # 시장 수익률 계산 (중복 인덱스 제거)
        market_prices = market_data['Close']
        market_prices = market_prices[~market_prices.index.duplicated(keep='last')]
        self.market_returns = market_prices.pct_change().dropna()

    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float, float]:
        """
        베타 계산 (CAPM)

        Returns:
            (beta, alpha, r_squared)
        """
        # 중복 인덱스 제거
        asset_returns = asset_returns[~asset_returns.index.duplicated(keep='last')]
        market_returns = market_returns[~market_returns.index.duplicated(keep='last')]

        # 날짜 맞추기
        combined = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()

        if len(combined) < 30:
            return 0.0, 0.0, 0.0

        # 공분산 / 분산
        covariance = combined['asset'].cov(combined['market'])
        variance = combined['market'].var()

        beta = covariance / variance if variance > 0 else 0

        # Alpha (초과 수익)
        asset_mean = combined['asset'].mean() * 252  # annualized
        market_mean = combined['market'].mean() * 252
        alpha = asset_mean - (self.risk_free_rate + beta * (market_mean - self.risk_free_rate))

        # R-squared
        correlation = combined['asset'].corr(combined['market'])
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0

        return beta, alpha, r_squared

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """최대 낙폭 계산 (%)"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax * 100
        return drawdown.min()

    def analyze_asset(self, ticker: str, df: pd.DataFrame) -> Optional[RiskFactors]:
        """단일 자산의 리스크 팩터 분석"""
        if df.empty or 'Close' not in df.columns:
            return None

        # 중복 인덱스 제거
        prices = df['Close']
        prices = prices[~prices.index.duplicated(keep='last')]
        returns = prices.pct_change().dropna()

        # 베타, 알파, R²
        beta, alpha, r_squared = self.calculate_beta(returns, self.market_returns)

        # 변동성 (annualized)
        volatility = returns.std() * np.sqrt(252) * 100

        # 최대 낙폭
        max_dd = self.calculate_max_drawdown(prices)

        # 시장 상관계수
        combined = pd.DataFrame({
            'asset': returns,
            'market': self.market_returns
        }).dropna()

        correlation = combined['asset'].corr(combined['market']) if len(combined) > 0 else 0

        return RiskFactors(
            ticker=ticker,
            market_beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            volatility=volatility,
            max_drawdown=max_dd,
            correlation_to_market=correlation,
        )

    def analyze_portfolio(self, assets: Dict[str, pd.DataFrame]) -> List[RiskFactors]:
        """포트폴리오 리스크 팩터 분석"""
        results = []

        for ticker, df in assets.items():
            factors = self.analyze_asset(ticker, df)
            if factors:
                results.append(factors)

        return results

    def to_dataframe(self, factors: List[RiskFactors]) -> pd.DataFrame:
        """리스크 팩터를 DataFrame으로 변환"""
        return pd.DataFrame([
            {
                'Ticker': f.ticker,
                'Beta': f.market_beta,
                'Alpha %': f.alpha * 100,
                'R²': f.r_squared,
                'Vol %': f.volatility,
                'Max DD %': f.max_drawdown,
                'Corr': f.correlation_to_market,
            }
            for f in factors
        ])


def test_sector_rotation():
    """섹터 로테이션 분석 테스트"""
    from db_manager import DatabaseManager

    print("="*70)
    print("🧪 Testing Sector Rotation Analysis")
    print("="*70)

    db = DatabaseManager()

    # 섹터 ETF 로드
    sector_tickers = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    sector_data = {}

    print("\n📊 Loading sector data...")
    for ticker in sector_tickers:
        df = db.get_latest_market_data(ticker)
        if not df.empty:
            df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            sector_data[ticker] = df
            print(f"   ✅ {ticker}: {len(df)} days")

    # SPY 로드
    spy = db.get_latest_market_data('SPY')
    if spy.empty:
        print("❌ SPY data not found")
        return

    spy_df = spy.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
    spy_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    spy_df.index = pd.to_datetime(spy_df.index)
    print(f"   ✅ SPY: {len(spy_df)} days")

    # 섹터 로테이션 분석
    print("\n🔄 Running sector rotation analysis...")
    analyzer = SectorRotationAnalyzer(sector_data, spy_df)
    scores = analyzer.analyze_all_sectors()

    # 결과 출력
    print("\n" + "="*70)
    print("📊 SECTOR RANKINGS")
    print("="*70)
    df = analyzer.to_dataframe()
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # 로테이션 신호
    print("\n" + "="*70)
    print("🔄 ROTATION SIGNALS")
    print("="*70)
    signals = analyzer.get_rotation_signals()

    print(f"\n📈 Economic Cycle: {signals['economic_cycle']}")
    print(f"   Confidence: {signals['cycle_confidence']:.0%}")

    print(f"\n✅ Preferred Sectors (by cycle):")
    for sector in signals['preferred_sectors']:
        print(f"   - {sector['ticker']}: {sector['name']}")

    print(f"\n🏆 Top Performers:")
    for perf in signals['top_performers']:
        print(f"   #{perf['rank']} {perf['ticker']:6s} ({perf['sector']:25s}): Score {perf['score']:>7.2f}, 3M {perf['momentum_3m']:>6.2f}%")

    print(f"\n⚠️  Bottom Performers:")
    for perf in signals['bottom_performers']:
        print(f"   #{perf['rank']} {perf['ticker']:6s} ({perf['sector']:25s}): Score {perf['score']:>7.2f}, 3M {perf['momentum_3m']:>6.2f}%")

    print("\n" + "="*70)


def test_risk_factors():
    """리스크 팩터 분석 테스트"""
    from db_manager import DatabaseManager

    print("\n" + "="*70)
    print("🧪 Testing Risk Factor Analysis")
    print("="*70)

    db = DatabaseManager()

    # SPY 로드
    spy = db.get_latest_market_data('SPY')
    if spy.empty:
        print("❌ SPY data not found")
        return

    spy_df = spy.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
    spy_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    spy_df.index = pd.to_datetime(spy_df.index)

    # 주요 종목 로드
    test_tickers = ['AAPL', 'NVDA', 'TSLA', 'JPM', 'XLE', 'TLT', 'BTC-USD']
    assets = {}

    print("\n📊 Loading assets...")
    for ticker in test_tickers:
        # Market data
        df = db.get_latest_market_data(ticker)

        # Crypto data (fallback)
        if df.empty:
            conn = db._get_connection()
            query = f'''
                SELECT date, open, high, low, close, volume
                FROM crypto_data
                WHERE ticker = '{ticker}'
                AND collection_run_id = (SELECT MAX(id) FROM collection_runs WHERE crypto_success = 1)
                ORDER BY date ASC
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.columns = [col.capitalize() for col in df.columns]
            assets[ticker] = df
            print(f"   ✅ {ticker}: {len(df)} days")

    # 리스크 팩터 분석
    print("\n📊 Running risk factor analysis...")
    analyzer = RiskFactorAnalyzer(spy_df)
    factors = analyzer.analyze_portfolio(assets)

    # 결과 출력
    print("\n" + "="*70)
    print("📊 RISK FACTORS")
    print("="*70)
    df = analyzer.to_dataframe(factors)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    print("\n" + "="*70)


if __name__ == "__main__":
    test_sector_rotation()
    test_risk_factors()
