#!/usr/bin/env python3
"""
Backtesting Framework
트레이딩 전략 백테스팅 프레임워크
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from analysis import TechnicalAnalysis


@dataclass
class Trade:
    """거래 기록"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    quantity: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy_name: str
    ticker: str
    start_date: str
    end_date: str

    # 성과 지표
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # 거래 통계
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # 거래 기록
    trades: List[Trade]

    # 포트폴리오 가치 시계열
    portfolio_value: pd.Series

    def __str__(self) -> str:
        return f"""
{'='*70}
📊 BACKTEST RESULTS: {self.strategy_name}
{'='*70}
Ticker: {self.ticker}
Period: {self.start_date} ~ {self.end_date}

💰 Performance:
   Total Return: ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)
   Sharpe Ratio: {self.sharpe_ratio:.2f}
   Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)

📈 Trading Statistics:
   Total Trades: {self.num_trades}
   Win Rate: {self.win_rate:.1f}%
   Avg Win: ${self.avg_win:.2f}
   Avg Loss: ${self.avg_loss:.2f}
   Profit Factor: {self.profit_factor:.2f}
{'='*70}
"""


class Strategy(ABC):
    """트레이딩 전략 추상 클래스"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        매매 신호 생성

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with values: 1 (buy), -1 (sell), 0 (hold)
        """
        pass


class MovingAverageCrossover(Strategy):
    """이동평균 크로스오버 전략"""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"MA_Crossover_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ta = TechnicalAnalysis(data)

        fast_ma = ta.moving_average(self.fast_period, 'SMA')
        slow_ma = ta.moving_average(self.slow_period, 'SMA')

        signals = pd.Series(0, index=data.index)

        # 골든 크로스 (빠른 MA가 느린 MA를 상향 돌파)
        signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1

        # 데드 크로스 (빠른 MA가 느린 MA를 하향 돌파)
        signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1

        return signals


class RSIMeanReversion(Strategy):
    """RSI 평균회귀 전략"""

    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_MeanReversion_{rsi_period}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ta = TechnicalAnalysis(data)
        rsi = ta.rsi(self.rsi_period)

        signals = pd.Series(0, index=data.index)

        # RSI가 과매도에서 벗어날 때 매수
        signals[(rsi > self.oversold) & (rsi.shift(1) <= self.oversold)] = 1

        # RSI가 과매수에서 벗어날 때 매도
        signals[(rsi < self.overbought) & (rsi.shift(1) >= self.overbought)] = -1

        return signals


class MACDStrategy(Strategy):
    """MACD 전략"""

    def __init__(self):
        super().__init__("MACD_Strategy")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ta = TechnicalAnalysis(data)
        macd = ta.macd()

        signals = pd.Series(0, index=data.index)

        macd_line = macd['macd']
        signal_line = macd['signal']

        # MACD 골든 크로스
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1

        # MACD 데드 크로스
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1

        return signals


class BollingerBandsStrategy(Strategy):
    """볼린저 밴드 전략"""

    def __init__(self, period: int = 20, std_dev: int = 2):
        super().__init__(f"BB_Strategy_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ta = TechnicalAnalysis(data)
        bb = ta.bollinger_bands(self.period, self.std_dev)

        signals = pd.Series(0, index=data.index)
        price = data['Close']

        # 하단 밴드 터치 후 반등 시 매수
        signals[(price > bb['lower']) & (price.shift(1) <= bb['lower'].shift(1))] = 1

        # 상단 밴드 터치 후 하락 시 매도
        signals[(price < bb['upper']) & (price.shift(1) >= bb['upper'].shift(1))] = -1

        return signals


class Backtester:
    """백테스팅 엔진"""

    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Args:
            initial_capital: 초기 자본
            commission: 거래 수수료 (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, strategy: Strategy, data: pd.DataFrame) -> BacktestResult:
        """
        백테스트 실행

        Args:
            strategy: 트레이딩 전략
            data: OHLCV DataFrame

        Returns:
            BacktestResult
        """
        # 신호 생성
        signals = strategy.generate_signals(data)

        # 포지션 추적
        position = 0  # 0: no position, 1: long
        trades = []
        portfolio_values = []

        cash = self.initial_capital
        shares = 0
        entry_price = 0
        entry_date = None

        for date, signal in signals.items():
            if date not in data.index:
                continue

            price = data.loc[date, 'Close']

            # 매수 신호
            if signal == 1 and position == 0:
                # 전액 매수
                shares = cash / (price * (1 + self.commission))
                cash = 0
                position = 1
                entry_price = price
                entry_date = date

            # 매도 신호
            elif signal == -1 and position == 1:
                # 전량 매도
                cash = shares * price * (1 - self.commission)

                # 거래 기록
                pnl = cash - self.initial_capital
                pnl_pct = (price / entry_price - 1) * 100

                trades.append(Trade(
                    entry_date=str(entry_date),
                    exit_date=str(date),
                    entry_price=entry_price,
                    exit_price=price,
                    position='long',
                    quantity=shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct
                ))

                shares = 0
                position = 0

            # 포트폴리오 가치 계산
            portfolio_value = cash + shares * price
            portfolio_values.append({'date': date, 'value': portfolio_value})

        # 마지막 포지션 청산
        if position == 1:
            final_price = data['Close'].iloc[-1]
            cash = shares * final_price * (1 - self.commission)

            pnl = cash - self.initial_capital
            pnl_pct = (final_price / entry_price - 1) * 100

            trades.append(Trade(
                entry_date=str(entry_date),
                exit_date=str(data.index[-1]),
                entry_price=entry_price,
                exit_price=final_price,
                position='long',
                quantity=shares,
                pnl=pnl,
                pnl_pct=pnl_pct
            ))

        # 포트폴리오 가치 시계열
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_series = portfolio_df.set_index('date')['value']

        # 성과 지표 계산
        result = self._calculate_metrics(
            strategy.name,
            data,
            trades,
            portfolio_series
        )

        return result

    def _calculate_metrics(self, strategy_name: str, data: pd.DataFrame,
                          trades: List[Trade], portfolio_value: pd.Series) -> BacktestResult:
        """성과 지표 계산"""

        # 기본 수익
        total_return = portfolio_value.iloc[-1] - self.initial_capital
        total_return_pct = (portfolio_value.iloc[-1] / self.initial_capital - 1) * 100

        # 샤프 비율
        returns = portfolio_value.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # 최대 낙폭
        cummax = portfolio_value.cummax()
        drawdown = portfolio_value - cummax
        max_dd = drawdown.min()
        max_dd_pct = (drawdown / cummax).min() * 100

        # 거래 통계
        num_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        total_wins = sum([t.pnl for t in winning_trades])
        total_losses = abs(sum([t.pnl for t in losing_trades]))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        return BacktestResult(
            strategy_name=strategy_name,
            ticker=data.index.name or 'Unknown',
            start_date=str(data.index[0]),
            end_date=str(data.index[-1]),
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=trades,
            portfolio_value=portfolio_value
        )


def test_backtesting():
    """백테스팅 테스트"""
    from db_manager import DatabaseManager

    print("="*70)
    print("🧪 Testing Backtesting Framework")
    print("="*70)

    # DB에서 SPY 데이터 로드
    db = DatabaseManager()
    spy = db.get_latest_market_data('SPY')

    if spy.empty:
        print("❌ No SPY data")
        return

    spy_df = spy.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
    spy_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    spy_df.index = pd.to_datetime(spy_df.index)
    spy_df = spy_df.sort_index()

    print(f"\n📊 Data: SPY ({len(spy_df)} days)")
    print(f"   Period: {spy_df.index[0]} to {spy_df.index[-1]}")

    # 백테스터 초기화
    backtester = Backtester(initial_capital=100000, commission=0.001)

    # 여러 전략 테스트
    strategies = [
        MovingAverageCrossover(fast_period=10, slow_period=20),
        RSIMeanReversion(rsi_period=14, oversold=30, overbought=70),
        MACDStrategy(),
        BollingerBandsStrategy(period=20, std_dev=2),
    ]

    results = []
    for strategy in strategies:
        print(f"\n🔄 Running {strategy.name}...")
        try:
            result = backtester.run(strategy, spy_df)
            results.append(result)
            print(result)
        except Exception as e:
            print(f"❌ Failed: {e}")

    # 전략 비교
    if results:
        print("\n" + "="*70)
        print("📊 STRATEGY COMPARISON")
        print("="*70)
        print(f"{'Strategy':<30} {'Return %':>12} {'Sharpe':>10} {'Trades':>10}")
        print("-"*70)
        for r in results:
            print(f"{r.strategy_name:<30} {r.total_return_pct:>11.2f}% {r.sharpe_ratio:>10.2f} {r.num_trades:>10}")


if __name__ == "__main__":
    test_backtesting()
