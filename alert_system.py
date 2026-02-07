#!/usr/bin/env python3
"""
Trading Alert System
트레이딩 신호 감지 및 알림 시스템
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from analysis import TechnicalAnalysis


@dataclass
class Alert:
    """알림 데이터 클래스"""
    timestamp: str
    ticker: str
    signal_type: str
    severity: str  # 'high', 'medium', 'low'
    message: str
    price: float
    indicator_value: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'ticker': self.ticker,
            'signal_type': self.signal_type,
            'severity': self.severity,
            'message': self.message,
            'price': self.price,
            'indicator_value': self.indicator_value,
        }

    def __str__(self) -> str:
        emoji = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢',
        }.get(self.severity, '⚪')

        return f"{emoji} [{self.ticker}] {self.signal_type}: {self.message} (${self.price:.2f})"


class AlertConfig:
    """알림 설정"""

    # RSI 임계값
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_EXTREME_OVERSOLD = 20
    RSI_EXTREME_OVERBOUGHT = 80

    # 볼린저 밴드
    BB_BREAKOUT_ENABLED = True

    # 거래량
    VOLUME_SURGE_THRESHOLD = 2.0  # 평균 대비 2배

    # 변동성
    VOLATILITY_SURGE_THRESHOLD = 1.5  # 평균 대비 1.5배

    # 알림 저장 경로
    ALERT_DIR = 'outputs/alerts'

    # 알림 활성화
    CONSOLE_ALERTS = True
    FILE_ALERTS = True
    EMAIL_ALERTS = False  # 추후 구현


class AlertDetector:
    """알림 신호 감지기"""

    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alerts: List[Alert] = []

        # 알림 디렉토리 생성
        Path(self.config.ALERT_DIR).mkdir(parents=True, exist_ok=True)

    def detect_rsi_signals(self, ticker: str, price_data: pd.DataFrame) -> List[Alert]:
        """RSI 신호 감지"""
        alerts = []

        try:
            ta = TechnicalAnalysis(price_data)
            rsi = ta.rsi(14)

            if rsi.empty:
                return alerts

            current_rsi = rsi.iloc[-1]
            current_price = price_data['Close'].iloc[-1]

            # 극도의 과매도
            if current_rsi < self.config.RSI_EXTREME_OVERSOLD:
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='RSI_EXTREME_OVERSOLD',
                    severity='high',
                    message=f'극도의 과매도 (RSI: {current_rsi:.1f})',
                    price=current_price,
                    indicator_value=current_rsi
                ))

            # 과매도
            elif current_rsi < self.config.RSI_OVERSOLD:
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='RSI_OVERSOLD',
                    severity='medium',
                    message=f'과매도 (RSI: {current_rsi:.1f})',
                    price=current_price,
                    indicator_value=current_rsi
                ))

            # 극도의 과매수
            elif current_rsi > self.config.RSI_EXTREME_OVERBOUGHT:
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='RSI_EXTREME_OVERBOUGHT',
                    severity='high',
                    message=f'극도의 과매수 (RSI: {current_rsi:.1f})',
                    price=current_price,
                    indicator_value=current_rsi
                ))

            # 과매수
            elif current_rsi > self.config.RSI_OVERBOUGHT:
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='RSI_OVERBOUGHT',
                    severity='medium',
                    message=f'과매수 (RSI: {current_rsi:.1f})',
                    price=current_price,
                    indicator_value=current_rsi
                ))

        except Exception as e:
            pass

        return alerts

    def detect_macd_signals(self, ticker: str, price_data: pd.DataFrame) -> List[Alert]:
        """MACD 크로스오버 감지"""
        alerts = []

        try:
            ta = TechnicalAnalysis(price_data)
            macd = ta.macd()

            if len(macd) < 2:
                return alerts

            current_price = price_data['Close'].iloc[-1]

            # MACD 라인
            macd_line = macd['macd']
            signal_line = macd['signal']

            # 골든 크로스 (MACD가 시그널을 상향 돌파)
            if (macd_line.iloc[-2] < signal_line.iloc[-2] and
                macd_line.iloc[-1] > signal_line.iloc[-1]):
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='MACD_GOLDEN_CROSS',
                    severity='high',
                    message=f'MACD 골든 크로스 (강세 신호)',
                    price=current_price,
                    indicator_value=macd_line.iloc[-1]
                ))

            # 데드 크로스 (MACD가 시그널을 하향 돌파)
            elif (macd_line.iloc[-2] > signal_line.iloc[-2] and
                  macd_line.iloc[-1] < signal_line.iloc[-1]):
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='MACD_DEAD_CROSS',
                    severity='high',
                    message=f'MACD 데드 크로스 (약세 신호)',
                    price=current_price,
                    indicator_value=macd_line.iloc[-1]
                ))

        except Exception:
            pass

        return alerts

    def detect_bollinger_breakout(self, ticker: str, price_data: pd.DataFrame) -> List[Alert]:
        """볼린저 밴드 돌파 감지"""
        alerts = []

        if not self.config.BB_BREAKOUT_ENABLED:
            return alerts

        try:
            ta = TechnicalAnalysis(price_data)
            bb = ta.bollinger_bands(20, 2)

            current_price = price_data['Close'].iloc[-1]
            upper_band = bb['upper'].iloc[-1]
            lower_band = bb['lower'].iloc[-1]

            # 상단 밴드 돌파
            if current_price > upper_band:
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='BB_UPPER_BREAKOUT',
                    severity='medium',
                    message=f'볼린저 상단 돌파 (과매수 가능성)',
                    price=current_price,
                    indicator_value=upper_band
                ))

            # 하단 밴드 돌파
            elif current_price < lower_band:
                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='BB_LOWER_BREAKOUT',
                    severity='medium',
                    message=f'볼린저 하단 돌파 (과매도 가능성)',
                    price=current_price,
                    indicator_value=lower_band
                ))

        except Exception:
            pass

        return alerts

    def detect_volume_surge(self, ticker: str, price_data: pd.DataFrame) -> List[Alert]:
        """거래량 급증 감지"""
        alerts = []

        try:
            if 'Volume' not in price_data.columns:
                return alerts

            current_volume = price_data['Volume'].iloc[-1]
            avg_volume = price_data['Volume'].iloc[-20:].mean()

            if current_volume > avg_volume * self.config.VOLUME_SURGE_THRESHOLD:
                current_price = price_data['Close'].iloc[-1]
                surge_ratio = current_volume / avg_volume

                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='VOLUME_SURGE',
                    severity='medium',
                    message=f'거래량 급증 ({surge_ratio:.1f}x 평균)',
                    price=current_price,
                    indicator_value=current_volume
                ))

        except Exception:
            pass

        return alerts

    def detect_volatility_surge(self, ticker: str, price_data: pd.DataFrame) -> List[Alert]:
        """변동성 급증 감지"""
        alerts = []

        try:
            returns = price_data['Close'].pct_change().dropna()

            if len(returns) < 20:
                return alerts

            current_vol = returns.iloc[-5:].std()
            avg_vol = returns.iloc[-20:].std()

            if current_vol > avg_vol * self.config.VOLATILITY_SURGE_THRESHOLD:
                current_price = price_data['Close'].iloc[-1]
                vol_ratio = current_vol / avg_vol

                alerts.append(Alert(
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    signal_type='VOLATILITY_SURGE',
                    severity='low',
                    message=f'변동성 급증 ({vol_ratio:.1f}x 평균)',
                    price=current_price,
                    indicator_value=current_vol
                ))

        except Exception:
            pass

        return alerts

    def scan_ticker(self, ticker: str, price_data: pd.DataFrame) -> List[Alert]:
        """단일 티커 전체 스캔"""
        all_alerts = []

        # 각종 신호 감지
        all_alerts.extend(self.detect_rsi_signals(ticker, price_data))
        all_alerts.extend(self.detect_macd_signals(ticker, price_data))
        all_alerts.extend(self.detect_bollinger_breakout(ticker, price_data))
        all_alerts.extend(self.detect_volume_surge(ticker, price_data))
        all_alerts.extend(self.detect_volatility_surge(ticker, price_data))

        return all_alerts

    def send_alerts(self, alerts: List[Alert]):
        """알림 전송"""
        if not alerts:
            return

        # 콘솔 출력
        if self.config.CONSOLE_ALERTS:
            print(f"\n{'='*70}")
            print(f"🔔 TRADING ALERTS ({len(alerts)} signals)")
            print(f"{'='*70}")

            for alert in alerts:
                print(alert)

        # 파일 저장
        if self.config.FILE_ALERTS:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config.ALERT_DIR}/alerts_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump([alert.to_dict() for alert in alerts], f, indent=2)

            print(f"\n💾 Alerts saved to: {filename}")

        # 이메일 알림 (추후 구현)
        if self.config.EMAIL_ALERTS:
            # TODO: Implement email alerts
            pass

    def scan_and_alert(self, tickers_data: Dict[str, pd.DataFrame]):
        """여러 티커 스캔 및 알림"""
        all_alerts = []

        for ticker, price_data in tickers_data.items():
            if price_data is None or price_data.empty:
                continue

            alerts = self.scan_ticker(ticker, price_data)
            all_alerts.extend(alerts)

        # 중요도 순으로 정렬
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_alerts.sort(key=lambda x: severity_order.get(x.severity, 3))

        self.send_alerts(all_alerts)
        self.alerts.extend(all_alerts)

        return all_alerts


def test_alert_system():
    """알림 시스템 테스트"""
    from db_manager import DatabaseManager

    print("="*70)
    print("🧪 Testing Alert System")
    print("="*70)

    # DB에서 데이터 로드
    db = DatabaseManager()

    test_tickers = ['SPY', 'BTC-USD', 'AAPL']
    tickers_data = {}

    print("\n📊 Loading data...")
    for ticker in test_tickers:
        if 'USD' in ticker:
            data = db.get_latest_market_data(ticker) if ticker not in ['BTC-USD', 'ETH-USD'] else None
            if data is None or data.empty:
                # Try crypto table
                conn = db._get_connection()
                query = f'''
                    SELECT date, open, high, low, close, volume
                    FROM crypto_data
                    WHERE ticker = '{ticker}' AND collection_run_id = (SELECT MAX(id) FROM collection_runs WHERE crypto_success = 1)
                    ORDER BY date ASC
                '''
                data = pd.read_sql_query(query, conn)
                conn.close()

                if not data.empty:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            data = db.get_latest_market_data(ticker)
            if not data.empty:
                data = data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                data.index = pd.to_datetime(data.index)

        if data is not None and not data.empty:
            tickers_data[ticker] = data
            print(f"   ✅ {ticker}: {len(data)} days")

    # 알림 시스템 실행
    print("\n🔍 Scanning for signals...")
    detector = AlertDetector()
    alerts = detector.scan_and_alert(tickers_data)

    print(f"\n✅ Found {len(alerts)} alerts")
    print("="*70)


if __name__ == "__main__":
    test_alert_system()
