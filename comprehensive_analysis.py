#!/usr/bin/env python3
"""
Comprehensive Financial Analysis
종합 금융 분석 리포트 생성

통합 기능:
1. 시장 데이터 수집
2. 기술적 분석
3. 알림 탐지
4. 백테스팅
5. 포트폴리오 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path

from db_manager import DatabaseManager
from analysis import TechnicalAnalysis, PortfolioAnalysis
from alert_system import AlertDetector, AlertConfig
from backtesting import (
    Backtester,
    MovingAverageCrossover,
    RSIMeanReversion,
    MACDStrategy,
    BollingerBandsStrategy
)


class ComprehensiveAnalyzer:
    """종합 분석 엔진"""

    def __init__(self, db: DatabaseManager = None):
        self.db = db or DatabaseManager()
        self.output_dir = Path('outputs/comprehensive')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_market_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """DB에서 시장 데이터 로드"""
        data = {}

        for ticker in tickers:
            # Market data 테이블에서 조회
            df = self.db.get_latest_market_data(ticker)

            if df.empty:
                # Crypto data 테이블에서 조회
                conn = self.db._get_connection()
                query = f'''
                    SELECT date, open, high, low, close, volume
                    FROM crypto_data
                    WHERE ticker = '{ticker}'
                    AND collection_run_id = (
                        SELECT MAX(id) FROM collection_runs WHERE crypto_success = 1
                    )
                    ORDER BY date ASC
                '''
                df = pd.read_sql_query(query, conn)
                conn.close()

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

                # 컬럼명 표준화
                df.columns = [col.capitalize() for col in df.columns]

                data[ticker] = df

        return data

    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> Dict:
        """단일 티커 종합 분석"""
        ta = TechnicalAnalysis(df)

        current_price = df['Close'].iloc[-1]

        # 기술적 지표
        rsi = ta.rsi(14)
        macd = ta.macd()
        bb = ta.bollinger_bands(20, 2)

        # 수익률
        returns_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) > 21 else None
        returns_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-64] - 1) * 100 if len(df) > 63 else None

        # 변동성 (annualized)
        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100

        analysis = {
            'ticker': ticker,
            'current_price': current_price,
            'indicators': {
                'rsi': rsi.iloc[-1] if not rsi.empty else None,
                'macd': macd['macd'].iloc[-1] if len(macd) > 0 else None,
                'macd_signal': macd['signal'].iloc[-1] if len(macd) > 0 else None,
                'bb_upper': bb['upper'].iloc[-1],
                'bb_lower': bb['lower'].iloc[-1],
                'bb_position': (current_price - bb['lower'].iloc[-1]) / (bb['upper'].iloc[-1] - bb['lower'].iloc[-1])
            },
            'performance': {
                'return_1m': returns_1m,
                'return_3m': returns_3m,
                'volatility_annual': volatility,
            }
        }

        return analysis

    def run_alerts(self, data: Dict[str, pd.DataFrame]) -> List:
        """알림 시스템 실행"""
        detector = AlertDetector()
        alerts = detector.scan_and_alert(data)
        return alerts

    def run_backtests(self, ticker: str, df: pd.DataFrame) -> Dict:
        """백테스팅 실행"""
        backtester = Backtester(initial_capital=100000, commission=0.001)

        strategies = [
            MovingAverageCrossover(fast_period=10, slow_period=20),
            MovingAverageCrossover(fast_period=20, slow_period=50),
            RSIMeanReversion(rsi_period=14, oversold=30, overbought=70),
            MACDStrategy(),
            BollingerBandsStrategy(period=20, std_dev=2),
        ]

        results = {}
        for strategy in strategies:
            try:
                result = backtester.run(strategy, df)
                results[strategy.name] = {
                    'total_return_pct': result.total_return_pct,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown_pct': result.max_drawdown_pct,
                    'num_trades': result.num_trades,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                }
            except Exception as e:
                results[strategy.name] = {'error': str(e)}

        return results

    def analyze_portfolio(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """포트폴리오 분석"""
        # 종가 데이터프레임 생성
        close_prices = pd.DataFrame()
        for ticker, df in data.items():
            if 'Close' in df.columns:
                close_prices[ticker] = df['Close']

        if close_prices.empty or len(close_prices.columns) < 2:
            return {'error': 'Not enough data for portfolio analysis'}

        pa = PortfolioAnalysis(close_prices)

        # 동일 가중 포트폴리오
        equal_weights = np.array([1.0 / len(close_prices.columns)] * len(close_prices.columns))

        # 상관관계 매트릭스
        corr_matrix = pa.correlation_matrix()

        analysis = {
            'assets': list(close_prices.columns),
            'equal_weight': {
                'return': pa.portfolio_return(equal_weights),
                'volatility': pa.portfolio_volatility(equal_weights),
                'sharpe': pa.sharpe_ratio(equal_weights),
            },
            'correlation': {
                'average': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'min': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min(),
                'max': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
            }
        }

        return analysis

    def generate_report(self, tickers: List[str]) -> Dict:
        """종합 리포트 생성"""
        print("="*70)
        print("📊 COMPREHENSIVE FINANCIAL ANALYSIS")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Tickers: {', '.join(tickers)}")
        print()

        # 1. 데이터 로드
        print("1️⃣  Loading market data...")
        data = self.load_market_data(tickers)
        print(f"   Loaded: {len(data)}/{len(tickers)} tickers")
        print()

        # 2. 개별 분석
        print("2️⃣  Running technical analysis...")
        ticker_analyses = {}
        for ticker, df in data.items():
            analysis = self.analyze_ticker(ticker, df)
            ticker_analyses[ticker] = analysis

            print(f"\n   📈 {ticker}")
            print(f"      Price: ${analysis['current_price']:,.2f}")
            if analysis['indicators']['rsi']:
                print(f"      RSI: {analysis['indicators']['rsi']:.1f}")
            if analysis['performance']['return_1m']:
                print(f"      1M Return: {analysis['performance']['return_1m']:.2f}%")
            print(f"      Volatility: {analysis['performance']['volatility_annual']:.1f}%")
        print()

        # 3. 알림 탐지
        print("3️⃣  Scanning for trading alerts...")
        alerts = self.run_alerts(data)
        print(f"   Found {len(alerts)} alerts")
        print()

        # 4. 백테스팅 (주요 ETF만)
        backtest_tickers = [t for t in ['SPY', 'QQQ'] if t in data]
        backtest_results = {}

        if backtest_tickers:
            print("4️⃣  Running backtests...")
            for ticker in backtest_tickers:
                print(f"\n   🔄 Backtesting {ticker}...")
                results = self.run_backtests(ticker, data[ticker])
                backtest_results[ticker] = results

                # 최고 성과 전략
                valid_results = {k: v for k, v in results.items() if 'error' not in v and v.get('num_trades', 0) > 0}
                if valid_results:
                    best = max(valid_results.items(), key=lambda x: x[1]['total_return_pct'])
                    print(f"      Best: {best[0]} ({best[1]['total_return_pct']:.2f}%)")
            print()

        # 5. 포트폴리오 분석
        print("5️⃣  Portfolio analysis...")
        portfolio_analysis = self.analyze_portfolio(data)
        if 'error' not in portfolio_analysis:
            print(f"   Equal-weight Sharpe: {portfolio_analysis['equal_weight']['sharpe']:.2f}")
            print(f"   Avg Correlation: {portfolio_analysis['correlation']['average']:.2f}")
        print()

        # 종합 리포트
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'tickers': tickers,
                'data_loaded': list(data.keys()),
            },
            'ticker_analysis': ticker_analyses,
            'alerts': [alert.to_dict() for alert in alerts],
            'backtests': backtest_results,
            'portfolio': portfolio_analysis,
        }

        # 저장
        output_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"💾 Report saved: {output_file}")
        print("="*70)

        return report


def main():
    """메인 실행"""

    # 분석 대상 티커
    tickers = [
        # US Market
        'SPY', 'QQQ', 'IWM',
        # Crypto
        'BTC-USD', 'ETH-USD',
        # 개별 종목
        'AAPL', 'NVDA', 'TSLA',
    ]

    analyzer = ComprehensiveAnalyzer()
    report = analyzer.generate_report(tickers)

    # 주요 발견사항 요약
    print("\n" + "="*70)
    print("🔍 KEY FINDINGS")
    print("="*70)

    # 고위험 알림
    high_alerts = [a for a in report['alerts'] if a['severity'] == 'high']
    if high_alerts:
        print(f"\n🔴 High Priority Alerts ({len(high_alerts)}):")
        for alert in high_alerts[:5]:
            print(f"   - [{alert['ticker']}] {alert['message']}")

    # 포트폴리오
    if 'error' not in report['portfolio']:
        print(f"\n💼 Portfolio Metrics:")
        print(f"   Equal-Weight Return: {report['portfolio']['equal_weight']['return']:.2f}%")
        print(f"   Equal-Weight Volatility: {report['portfolio']['equal_weight']['volatility']:.2f}%")
        print(f"   Sharpe Ratio: {report['portfolio']['equal_weight']['sharpe']:.2f}")

    # 최고/최저 수익률
    returns = {}
    for ticker, analysis in report['ticker_analysis'].items():
        ret = analysis['performance'].get('return_1m')
        if ret is not None:
            returns[ticker] = ret

    if returns:
        best = max(returns.items(), key=lambda x: x[1])
        worst = min(returns.items(), key=lambda x: x[1])
        print(f"\n📊 1-Month Performance:")
        print(f"   Best: {best[0]} ({best[1]:+.2f}%)")
        print(f"   Worst: {worst[0]} ({worst[1]:+.2f}%)")

    print("="*70)


if __name__ == "__main__":
    main()
