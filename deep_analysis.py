#!/usr/bin/env python3
"""
Deep Dive Asset Analysis
특정 자산/섹터 심층 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SupportResistance:
    """지지/저항 레벨"""
    level: float
    strength: int  # 터치 횟수
    level_type: str  # 'support' or 'resistance'


@dataclass
class TrendAnalysis:
    """트렌드 분석"""
    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-100
    timeframe: str  # '1M', '3M', '6M', '1Y'
    slope: float  # 추세선 기울기


@dataclass
class PositionSizing:
    """포지션 사이징"""
    kelly_fraction: float
    risk_based_pct: float
    suggested_allocation: float
    max_loss_per_trade: float
    shares_to_buy: int


@dataclass
class TradeIdea:
    """트레이딩 아이디어"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: float
    rationale: str


class DeepDiveAnalyzer:
    """심층 분석기"""

    def __init__(self, ticker: str, price_data: pd.DataFrame, market_data: pd.DataFrame):
        """
        Args:
            ticker: 분석 대상 티커
            price_data: OHLCV DataFrame
            market_data: 시장 지수 DataFrame (비교용)
        """
        self.ticker = ticker
        self.data = price_data
        self.market_data = market_data

        if self.data.empty or 'Close' not in self.data.columns:
            raise ValueError(f"Invalid price data for {ticker}")

    def calculate_support_resistance(self, lookback: int = 100, threshold: float = 0.02) -> List[SupportResistance]:
        """
        지지/저항 레벨 계산

        Args:
            lookback: 분석 기간
            threshold: 레벨 인식 임계값 (2%)

        Returns:
            지지/저항 레벨 리스트
        """
        if len(self.data) < lookback:
            lookback = len(self.data)

        recent_data = self.data.tail(lookback)

        # 고점/저점 찾기
        highs = recent_data['High'].values
        lows = recent_data['Low'].values

        # 주요 레벨 추출 (단순화된 방법)
        levels = []

        # 최근 고점들
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append(('resistance', highs[i]))

        # 최근 저점들
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append(('support', lows[i]))

        # 레벨 클러스터링
        clustered = []
        for level_type, price in levels:
            # 기존 레벨과 가까운지 확인
            found = False
            for sr in clustered:
                if abs(sr.level - price) / price < threshold and sr.level_type == level_type:
                    sr.strength += 1
                    found = True
                    break

            if not found:
                clustered.append(SupportResistance(
                    level=price,
                    strength=1,
                    level_type=level_type
                ))

        # 강도순 정렬
        clustered.sort(key=lambda x: x.strength, reverse=True)

        return clustered[:5]  # 상위 5개만

    def analyze_trend(self, timeframe_days: int) -> TrendAnalysis:
        """
        트렌드 분석

        Args:
            timeframe_days: 분석 기간 (일)

        Returns:
            트렌드 분석 결과
        """
        if len(self.data) < timeframe_days:
            timeframe_days = len(self.data)

        recent_data = self.data.tail(timeframe_days)
        prices = recent_data['Close'].values

        # 선형 회귀로 추세선 계산
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)

        # 추세 방향
        if slope > 0:
            direction = 'bullish'
        elif slope < 0:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # 추세 강도 (R² 기반)
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        strength = max(0, min(100, r_squared * 100))

        # 시간대 레이블
        if timeframe_days <= 30:
            tf_label = '1M'
        elif timeframe_days <= 90:
            tf_label = '3M'
        elif timeframe_days <= 180:
            tf_label = '6M'
        else:
            tf_label = '1Y'

        return TrendAnalysis(
            direction=direction,
            strength=strength,
            timeframe=tf_label,
            slope=slope
        )

    def multi_timeframe_analysis(self) -> Dict[str, TrendAnalysis]:
        """다중 시간대 트렌드 분석"""
        timeframes = {
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '1Y': 252,
        }

        results = {}
        for label, days in timeframes.items():
            if len(self.data) >= days:
                results[label] = self.analyze_trend(days)

        return results

    def calculate_volume_profile(self, bins: int = 20) -> pd.DataFrame:
        """
        거래량 프로파일 계산

        Args:
            bins: 가격 구간 수

        Returns:
            가격대별 거래량 DataFrame
        """
        if 'Volume' not in self.data.columns:
            return pd.DataFrame()

        recent_data = self.data.tail(100)

        # 가격 범위 구간 나누기
        price_min = recent_data['Low'].min()
        price_max = recent_data['High'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)

        # 각 구간별 거래량 집계
        volume_profile = []
        for i in range(len(price_bins) - 1):
            low_bound = price_bins[i]
            high_bound = price_bins[i + 1]

            # 해당 구간에 속하는 거래량 합계
            mask = (recent_data['Close'] >= low_bound) & (recent_data['Close'] < high_bound)
            total_volume = recent_data.loc[mask, 'Volume'].sum()

            volume_profile.append({
                'price_low': low_bound,
                'price_high': high_bound,
                'price_mid': (low_bound + high_bound) / 2,
                'volume': total_volume,
            })

        return pd.DataFrame(volume_profile).sort_values('volume', ascending=False)

    def relative_performance(self) -> Dict:
        """시장 대비 상대 성과"""
        if self.market_data.empty:
            return {}

        # 공통 날짜 찾기
        common_dates = self.data.index.intersection(self.market_data.index)
        if len(common_dates) < 2:
            return {}

        asset_prices = self.data.loc[common_dates, 'Close']
        market_prices = self.market_data.loc[common_dates, 'Close']

        # 수익률 계산
        timeframes = [21, 63, 126, 252]
        performance = {}

        for days in timeframes:
            if len(common_dates) < days:
                continue

            asset_ret = (asset_prices.iloc[-1] / asset_prices.iloc[-days] - 1) * 100
            market_ret = (market_prices.iloc[-1] / market_prices.iloc[-days] - 1) * 100

            label = f"{days}D"
            performance[label] = {
                'asset_return': asset_ret,
                'market_return': market_ret,
                'outperformance': asset_ret - market_ret,
            }

        return performance

    def calculate_kelly_criterion(self, win_rate: float = None, avg_win: float = None,
                                   avg_loss: float = None) -> float:
        """
        Kelly Criterion 계산

        Args:
            win_rate: 승률 (0-1), None이면 과거 데이터로 추정
            avg_win: 평균 수익률, None이면 과거 데이터로 추정
            avg_loss: 평균 손실률, None이면 과거 데이터로 추정

        Returns:
            Kelly fraction (0-1)
        """
        if win_rate is None or avg_win is None or avg_loss is None:
            # 과거 데이터로 추정
            returns = self.data['Close'].pct_change().dropna()

            if len(returns) < 30:
                return 0.1  # 기본값

            wins = returns[returns > 0]
            losses = returns[returns < 0]

            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
            avg_win = wins.mean() if len(wins) > 0 else 0.01
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01

        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        # 보수적으로 절반만 사용 (Half Kelly)
        kelly_fraction = max(0, min(0.25, kelly * 0.5))

        return kelly_fraction

    def position_sizing(self, portfolio_value: float = 100000,
                        risk_per_trade: float = 0.02) -> PositionSizing:
        """
        포지션 사이징 계산

        Args:
            portfolio_value: 포트폴리오 총액
            risk_per_trade: 거래당 리스크 (2% = 0.02)

        Returns:
            포지션 사이징 결과
        """
        current_price = self.data['Close'].iloc[-1]

        # Kelly Criterion
        kelly_fraction = self.calculate_kelly_criterion()
        kelly_allocation = kelly_fraction * 100

        # Risk-based sizing
        # ATR(14)로 변동성 측정
        if len(self.data) >= 14:
            high_low = self.data['High'] - self.data['Low']
            high_close = abs(self.data['High'] - self.data['Close'].shift())
            low_close = abs(self.data['Low'] - self.data['Close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

            # 2 * ATR를 스탑로스로 사용
            stop_distance = 2 * atr
            risk_per_share = stop_distance
        else:
            # ATR을 계산할 수 없으면 5% 사용
            risk_per_share = current_price * 0.05

        # 리스크 기반 포지션 크기
        max_loss = portfolio_value * risk_per_trade
        shares = int(max_loss / risk_per_share) if risk_per_share > 0 else 0
        risk_based_pct = (shares * current_price / portfolio_value * 100) if portfolio_value > 0 else 0

        # 최종 권장 (Kelly와 Risk-based의 평균)
        suggested_allocation = (kelly_allocation + risk_based_pct) / 2
        suggested_allocation = min(suggested_allocation, 15)  # 최대 15%로 제한

        return PositionSizing(
            kelly_fraction=kelly_fraction,
            risk_based_pct=risk_based_pct,
            suggested_allocation=suggested_allocation,
            max_loss_per_trade=max_loss,
            shares_to_buy=int(portfolio_value * suggested_allocation / 100 / current_price)
        )

    def generate_trade_idea(self) -> TradeIdea:
        """트레이딩 아이디어 생성"""
        current_price = self.data['Close'].iloc[-1]

        # 다중 시간대 분석
        trends = self.multi_timeframe_analysis()

        # 지지/저항 레벨
        sr_levels = self.calculate_support_resistance()

        # 트렌드 점수 계산 (1M, 3M에 더 높은 가중치)
        trend_score = 0
        weights = {'1M': 0.4, '3M': 0.3, '6M': 0.2, '1Y': 0.1}

        for tf, weight in weights.items():
            if tf in trends:
                trend = trends[tf]
                if trend.direction == 'bullish':
                    trend_score += weight * trend.strength
                elif trend.direction == 'bearish':
                    trend_score -= weight * trend.strength

        # 액션 결정
        if trend_score > 40:
            action = 'BUY'
            confidence = 'HIGH' if trend_score > 60 else 'MEDIUM'
        elif trend_score < -40:
            action = 'SELL'
            confidence = 'HIGH' if trend_score < -60 else 'MEDIUM'
        else:
            action = 'HOLD'
            confidence = 'LOW'

        # 지지/저항 기반 가격 레벨
        supports = [sr for sr in sr_levels if sr.level_type == 'support' and sr.level < current_price]
        resistances = [sr for sr in sr_levels if sr.level_type == 'resistance' and sr.level > current_price]

        # Entry, Stop, Target 설정
        if action == 'BUY':
            entry = current_price
            stop_loss = supports[0].level if supports else current_price * 0.95
            target_1 = resistances[0].level if resistances else current_price * 1.05
            target_2 = resistances[1].level if len(resistances) > 1 else current_price * 1.10

            rationale = f"Bullish trend ({trend_score:.1f}/100). "
            if supports:
                rationale += f"Strong support at ${supports[0].level:.2f}. "
            if resistances:
                rationale += f"First resistance at ${resistances[0].level:.2f}."

        elif action == 'SELL':
            entry = current_price
            stop_loss = resistances[0].level if resistances else current_price * 1.05
            target_1 = supports[0].level if supports else current_price * 0.95
            target_2 = supports[1].level if len(supports) > 1 else current_price * 0.90

            rationale = f"Bearish trend ({trend_score:.1f}/100). "
            if resistances:
                rationale += f"Strong resistance at ${resistances[0].level:.2f}. "
            if supports:
                rationale += f"First support at ${supports[0].level:.2f}."

        else:  # HOLD
            entry = current_price
            stop_loss = current_price * 0.95
            target_1 = current_price * 1.05
            target_2 = current_price * 1.10
            rationale = f"Neutral trend ({trend_score:.1f}/100). Wait for clearer signal."

        # Risk/Reward Ratio
        risk = abs(entry - stop_loss)
        reward = abs(target_1 - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        return TradeIdea(
            action=action,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            risk_reward_ratio=rr_ratio,
            rationale=rationale
        )

    def create_report(self) -> str:
        """종합 리포트 생성"""
        report = []
        report.append("=" * 70)
        report.append(f"📊 DEEP DIVE ANALYSIS: {self.ticker}")
        report.append("=" * 70)
        report.append(f"Current Price: ${self.data['Close'].iloc[-1]:,.2f}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 1. 다중 시간대 트렌드
        report.append("1️⃣  MULTI-TIMEFRAME TREND ANALYSIS")
        report.append("-" * 70)
        trends = self.multi_timeframe_analysis()
        for tf, trend in trends.items():
            emoji = "📈" if trend.direction == 'bullish' else "📉" if trend.direction == 'bearish' else "➡️"
            report.append(f"   {tf:3s} {emoji} {trend.direction.upper():8s} (Strength: {trend.strength:.1f}/100)")
        report.append("")

        # 2. 지지/저항 레벨
        report.append("2️⃣  SUPPORT & RESISTANCE LEVELS")
        report.append("-" * 70)
        sr_levels = self.calculate_support_resistance()
        current_price = self.data['Close'].iloc[-1]

        for sr in sr_levels:
            symbol = "🟢" if sr.level_type == 'support' else "🔴"
            distance = (sr.level - current_price) / current_price * 100
            report.append(f"   {symbol} ${sr.level:>10,.2f} ({sr.level_type.upper():10s}) "
                         f"[{distance:>+6.2f}%] (Strength: {sr.strength})")
        report.append("")

        # 3. 상대 성과
        report.append("3️⃣  RELATIVE PERFORMANCE (vs Market)")
        report.append("-" * 70)
        perf = self.relative_performance()
        for label, data in perf.items():
            outperf = data['outperformance']
            symbol = "✅" if outperf > 0 else "❌"
            report.append(f"   {label} {symbol} Asset: {data['asset_return']:>+7.2f}% | "
                         f"Market: {data['market_return']:>+7.2f}% | "
                         f"Alpha: {outperf:>+7.2f}%")
        report.append("")

        # 4. 포지션 사이징
        report.append("4️⃣  POSITION SIZING (Portfolio: $100,000)")
        report.append("-" * 70)
        sizing = self.position_sizing()
        report.append(f"   Kelly Criterion: {sizing.kelly_fraction*100:.2f}%")
        report.append(f"   Risk-Based: {sizing.risk_based_pct:.2f}%")
        report.append(f"   ⭐ Suggested Allocation: {sizing.suggested_allocation:.2f}%")
        report.append(f"   Shares to Buy: {sizing.shares_to_buy:,} shares")
        report.append(f"   Max Loss per Trade: ${sizing.max_loss_per_trade:,.2f}")
        report.append("")

        # 5. 트레이딩 아이디어
        report.append("5️⃣  TRADING IDEA")
        report.append("-" * 70)
        idea = self.generate_trade_idea()

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}

        report.append(f"   {action_emoji.get(idea.action, '⚪')} Action: {idea.action}")
        report.append(f"   {conf_emoji.get(idea.confidence, '⚪')} Confidence: {idea.confidence}")
        report.append(f"   Entry: ${idea.entry_price:,.2f}")
        report.append(f"   Stop Loss: ${idea.stop_loss:,.2f} ({(idea.stop_loss/idea.entry_price-1)*100:+.2f}%)")
        report.append(f"   Target 1: ${idea.target_1:,.2f} ({(idea.target_1/idea.entry_price-1)*100:+.2f}%)")
        report.append(f"   Target 2: ${idea.target_2:,.2f} ({(idea.target_2/idea.entry_price-1)*100:+.2f}%)")
        report.append(f"   Risk/Reward: 1:{idea.risk_reward_ratio:.2f}")
        report.append(f"   Rationale: {idea.rationale}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)


def test_deep_analysis():
    """심층 분석 테스트"""
    from db_manager import DatabaseManager

    print("="*70)
    print("🧪 Testing Deep Dive Analysis")
    print("="*70)

    db = DatabaseManager()

    # SPY (시장 지수) 로드
    spy = db.get_latest_market_data('SPY')
    spy_df = spy.set_index('date')[['close']]
    spy_df.columns = ['Close']
    spy_df.index = pd.to_datetime(spy_df.index)

    # 분석 대상
    targets = [
        ('XLU', 'Utilities ETF - Top Performer'),
        ('NVDA', 'NVIDIA - High Beta Growth'),
        ('BTC-USD', 'Bitcoin - Independent Asset'),
    ]

    for ticker, description in targets:
        print(f"\n{'='*70}")
        print(f"Analyzing: {ticker} - {description}")
        print('='*70)

        # 데이터 로드
        df = db.get_latest_market_data(ticker)

        # Crypto fallback
        if df.empty and 'USD' in ticker:
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

        if df.empty:
            print(f"❌ No data for {ticker}")
            continue

        # 데이터 준비
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.columns = [col.capitalize() for col in df.columns]

        # 심층 분석
        try:
            analyzer = DeepDiveAnalyzer(ticker, df, spy_df)
            report = analyzer.create_report()
            print(report)
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ Deep analysis completed!")
    print("="*70)


if __name__ == "__main__":
    test_deep_analysis()
