"""
Technical Analysis Indicators

Implements traditional technical analysis methods:
- Trend Indicators (Moving Averages, MACD, ADX)
- Momentum Indicators (RSI, Stochastic, ROC)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators (OBV, Volume MA)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class TechnicalAnalysis:
    """Traditional technical analysis indicators"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data

        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume
        """
        self.df = df.copy()
        self.results = {}

    # ============== Trend Indicators ==============

    def moving_average(self, period: int = 20, ma_type: str = 'SMA') -> pd.Series:
        """
        Calculate Moving Average

        Args:
            period: Period for MA calculation
            ma_type: 'SMA' (Simple) or 'EMA' (Exponential)
        """
        if ma_type == 'SMA':
            ma = self.df['Close'].rolling(window=period).mean()
        elif ma_type == 'EMA':
            ma = self.df['Close'].ewm(span=period, adjust=False).mean()
        else:
            raise ValueError("ma_type must be 'SMA' or 'EMA'")

        self.results[f'{ma_type}_{period}'] = ma
        return ma

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)

        Returns:
            dict with 'macd', 'signal', 'histogram'
        """
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        result = {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

        self.results['MACD'] = result
        return result

    def adx(self, period: int = 14) -> Dict[str, pd.Series]:
        """
        Average Directional Index (ADX) - Trend Strength

        Returns:
            dict with 'adx', 'plus_di', 'minus_di'
        """
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Smoothed indicators
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        result = {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

        self.results['ADX'] = result
        return result

    # ============== Momentum Indicators ==============

    def rsi(self, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)

        Traditional overbought/oversold indicator
        RSI > 70: Overbought
        RSI < 30: Oversold
        """
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        self.results['RSI'] = rsi
        return rsi

    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator

        Returns:
            dict with '%K' and '%D' lines
        """
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()

        k = 100 * (self.df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()

        result = {
            'K': k,
            'D': d
        }

        self.results['Stochastic'] = result
        return result

    def roc(self, period: int = 12) -> pd.Series:
        """
        Rate of Change (ROC)

        Momentum indicator showing % change
        """
        roc = 100 * (self.df['Close'] - self.df['Close'].shift(period)) / self.df['Close'].shift(period)

        self.results['ROC'] = roc
        return roc

    def williams_r(self, period: int = 14) -> pd.Series:
        """
        Williams %R

        Momentum indicator (similar to Stochastic but reversed scale)
        """
        high_max = self.df['High'].rolling(window=period).max()
        low_min = self.df['Low'].rolling(window=period).min()

        williams = -100 * (high_max - self.df['Close']) / (high_max - low_min)

        self.results['Williams_R'] = williams
        return williams

    # ============== Volatility Indicators ==============

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands

        Returns:
            dict with 'middle', 'upper', 'lower' bands
        """
        middle = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        result = {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'bandwidth': (upper - lower) / middle * 100
        }

        self.results['Bollinger_Bands'] = result
        return result

    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)

        Volatility indicator
        """
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        self.results['ATR'] = atr
        return atr

    def keltner_channels(self, ema_period: int = 20, atr_period: int = 10,
                        multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Keltner Channels

        Similar to Bollinger Bands but uses ATR
        """
        middle = self.df['Close'].ewm(span=ema_period, adjust=False).mean()
        atr = self.atr(atr_period)

        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)

        result = {
            'middle': middle,
            'upper': upper,
            'lower': lower
        }

        self.results['Keltner_Channels'] = result
        return result

    # ============== Volume Indicators ==============

    def obv(self) -> pd.Series:
        """
        On-Balance Volume (OBV)

        Cumulative volume-based indicator
        """
        obv = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

        self.results['OBV'] = obv
        return obv

    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)

        Important for institutional trading
        """
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        vwap = (typical_price * self.df['Volume']).cumsum() / self.df['Volume'].cumsum()

        self.results['VWAP'] = vwap
        return vwap

    def mfi(self, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI)

        Volume-weighted RSI
        """
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow))

        self.results['MFI'] = mfi
        return mfi

    # ============== Support/Resistance ==============

    def pivot_points(self) -> Dict[str, float]:
        """
        Classic Pivot Points (daily)

        Support and resistance levels
        """
        high = self.df['High'].iloc[-1]
        low = self.df['Low'].iloc[-1]
        close = self.df['Close'].iloc[-1]

        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        result = {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }

        self.results['Pivot_Points'] = result
        return result

    # ============== Pattern Recognition ==============

    def detect_crossover(self, fast_period: int = 50, slow_period: int = 200) -> pd.Series:
        """
        Golden Cross / Death Cross Detection

        Golden Cross: Fast MA crosses above Slow MA (bullish)
        Death Cross: Fast MA crosses below Slow MA (bearish)
        """
        fast_ma = self.moving_average(fast_period, 'SMA')
        slow_ma = self.moving_average(slow_period, 'SMA')

        # 1: Golden Cross, -1: Death Cross, 0: No cross
        crossover = pd.Series(0, index=self.df.index)

        # Detect crosses
        prev_diff = (fast_ma - slow_ma).shift(1)
        curr_diff = fast_ma - slow_ma

        crossover[(prev_diff < 0) & (curr_diff > 0)] = 1  # Golden Cross
        crossover[(prev_diff > 0) & (curr_diff < 0)] = -1  # Death Cross

        self.results['MA_Crossover'] = crossover
        return crossover

    def get_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators and return as DataFrame
        """
        # Trend
        self.moving_average(20, 'SMA')
        self.moving_average(50, 'SMA')
        self.moving_average(200, 'SMA')
        self.moving_average(20, 'EMA')
        macd_data = self.macd()
        adx_data = self.adx()

        # Momentum
        self.rsi()
        stoch_data = self.stochastic()
        self.roc()
        self.williams_r()

        # Volatility
        bb_data = self.bollinger_bands()
        self.atr()

        # Volume
        self.obv()
        self.vwap()
        self.mfi()

        # Combine into single DataFrame
        result_df = self.df.copy()

        # Add all calculated indicators
        for key, value in self.results.items():
            if isinstance(value, pd.Series):
                result_df[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, pd.Series):
                        result_df[f'{key}_{subkey}'] = subvalue

        return result_df

    def get_signals(self) -> Dict[str, str]:
        """
        Generate trading signals based on current indicator values

        Returns:
            Dictionary of indicator signals (bullish/bearish/neutral)
        """
        signals = {}

        # RSI Signal
        if 'RSI' in self.results:
            rsi_current = self.results['RSI'].iloc[-1]
            if rsi_current > 70:
                signals['RSI'] = 'Overbought'
            elif rsi_current < 30:
                signals['RSI'] = 'Oversold'
            else:
                signals['RSI'] = 'Neutral'

        # MACD Signal
        if 'MACD' in self.results:
            macd_hist = self.results['MACD']['histogram'].iloc[-1]
            if macd_hist > 0:
                signals['MACD'] = 'Bullish'
            else:
                signals['MACD'] = 'Bearish'

        # Bollinger Bands Signal
        if 'Bollinger_Bands' in self.results:
            current_price = self.df['Close'].iloc[-1]
            upper = self.results['Bollinger_Bands']['upper'].iloc[-1]
            lower = self.results['Bollinger_Bands']['lower'].iloc[-1]

            if current_price > upper:
                signals['Bollinger'] = 'Overbought'
            elif current_price < lower:
                signals['Bollinger'] = 'Oversold'
            else:
                signals['Bollinger'] = 'Normal'

        # ADX Trend Strength
        if 'ADX' in self.results:
            adx_current = self.results['ADX']['adx'].iloc[-1]
            if adx_current > 25:
                signals['Trend_Strength'] = 'Strong'
            elif adx_current > 20:
                signals['Trend_Strength'] = 'Moderate'
            else:
                signals['Trend_Strength'] = 'Weak'

        return signals
