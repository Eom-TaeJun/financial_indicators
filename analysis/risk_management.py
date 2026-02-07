"""
Risk Management

Traditional risk management and measurement:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Expected Shortfall (ES/CVaR)
- Volatility Measures
- Correlation and Tail Risk
- Stress Testing
- Risk Limits and Controls
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


class RiskManagement:
    """Traditional risk management techniques"""

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with returns data

        Args:
            returns: DataFrame with asset returns
        """
        self.returns = returns

    # ============== Value at Risk (VaR) ==============

    def historical_var(self, confidence_level: float = 0.95,
                      portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Historical Value at Risk

        Non-parametric method using historical distribution
        Most straightforward VaR calculation
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]  # Single asset

        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def parametric_var(self, confidence_level: float = 0.95,
                      portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Parametric VaR (Variance-Covariance Method)

        Assumes normal distribution
        VaR = μ - z*σ

        Faster but less accurate for non-normal returns
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)

        var = mean - z_score * std
        return var

    def monte_carlo_var(self, confidence_level: float = 0.95,
                       num_simulations: int = 10000,
                       portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Monte Carlo VaR

        Simulates future returns using random sampling
        Most flexible but computationally intensive
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        mean = returns.mean()
        std = returns.std()

        # Generate simulations
        simulated_returns = np.random.normal(mean, std, num_simulations)

        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return var

    def compare_var_methods(self, confidence_level: float = 0.95,
                           portfolio_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compare all VaR calculation methods
        """
        return {
            'historical': self.historical_var(confidence_level, portfolio_weights),
            'parametric': self.parametric_var(confidence_level, portfolio_weights),
            'monte_carlo': self.monte_carlo_var(confidence_level, portfolio_weights)
        }

    # ============== Expected Shortfall (CVaR) ==============

    def expected_shortfall(self, confidence_level: float = 0.95,
                          portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Expected Shortfall (ES) / Conditional Value at Risk (CVaR)

        Average loss beyond VaR
        More conservative than VaR (captures tail risk better)
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        var = self.historical_var(confidence_level, portfolio_weights)
        es = returns[returns <= var].mean()

        return es

    # ============== Volatility Measures ==============

    def realized_volatility(self, window: int = 30,
                           portfolio_weights: Optional[np.ndarray] = None) -> pd.Series:
        """
        Realized Volatility (Historical Volatility)

        Rolling standard deviation of returns
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        return returns.rolling(window=window).std() * np.sqrt(252)

    def ewma_volatility(self, lambda_param: float = 0.94,
                       portfolio_weights: Optional[np.ndarray] = None) -> pd.Series:
        """
        Exponentially Weighted Moving Average (EWMA) Volatility

        Gives more weight to recent observations
        RiskMetrics uses λ = 0.94 for daily data
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        # EWMA variance
        var_ewma = returns.ewm(alpha=1-lambda_param).var()

        return np.sqrt(var_ewma * 252)

    def garch_volatility(self, returns_series: pd.Series) -> pd.Series:
        """
        GARCH(1,1) Volatility Forecast

        Simple implementation of GARCH model
        σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
        """
        # Simplified GARCH parameters (typical values)
        omega = 0.000001
        alpha = 0.1
        beta = 0.85

        variance = pd.Series(index=returns_series.index, dtype=float)
        variance.iloc[0] = returns_series.var()

        for i in range(1, len(returns_series)):
            variance.iloc[i] = (omega +
                              alpha * returns_series.iloc[i-1]**2 +
                              beta * variance.iloc[i-1])

        return np.sqrt(variance * 252)

    # ============== Downside Risk ==============

    def downside_deviation(self, min_acceptable_return: float = 0,
                          portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Downside Deviation (Semi-Deviation)

        Only considers returns below MAR (Minimum Acceptable Return)
        Used in Sortino Ratio
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        downside_returns = returns[returns < min_acceptable_return]
        downside_std = np.sqrt(np.mean(downside_returns**2))

        return downside_std * np.sqrt(252)

    def semi_variance(self, target_return: float = 0,
                     portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Semi-Variance

        Variance of returns below target
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        downside_returns = returns[returns < target_return] - target_return
        return np.var(downside_returns)

    # ============== Tail Risk ==============

    def skewness(self, portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Skewness

        Measures asymmetry of return distribution
        Negative skew: More extreme negative returns
        Positive skew: More extreme positive returns
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        return stats.skew(returns)

    def kurtosis(self, portfolio_weights: Optional[np.ndarray] = None) -> float:
        """
        Kurtosis (Excess Kurtosis)

        Measures "tailedness" of distribution
        Kurtosis > 0: Fat tails (more extreme events)
        Kurtosis < 0: Thin tails
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        return stats.kurtosis(returns)

    def jarque_bera_test(self, portfolio_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Jarque-Bera Test for Normality

        Tests if returns follow normal distribution
        Important for validating parametric VaR assumptions
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        jb_stat, p_value = stats.jarque_bera(returns.dropna())

        return {
            'statistic': jb_stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': self.skewness(portfolio_weights),
            'kurtosis': self.kurtosis(portfolio_weights)
        }

    # ============== Correlation Risk ==============

    def correlation_breakdown(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Identify high correlation pairs

        High correlation increases portfolio risk
        """
        corr_matrix = self.returns.corr()

        # Extract upper triangle
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        'asset_1': corr_matrix.columns[i],
                        'asset_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        return pd.DataFrame(high_corr)

    def rolling_correlation(self, asset1: str, asset2: str, window: int = 60) -> pd.Series:
        """
        Rolling correlation between two assets

        Monitors correlation stability over time
        """
        return self.returns[asset1].rolling(window).corr(self.returns[asset2])

    # ============== Drawdown Analysis ==============

    def calculate_drawdowns(self, portfolio_weights: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate all drawdowns

        Drawdown = (Current Value - Peak Value) / Peak Value
        """
        if portfolio_weights is not None:
            returns = (self.returns @ portfolio_weights)
        else:
            returns = self.returns.iloc[:, 0]

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return pd.DataFrame({
            'cumulative_return': cumulative,
            'running_max': running_max,
            'drawdown': drawdown
        })

    def maximum_drawdown_duration(self, portfolio_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Maximum Drawdown Duration

        Longest time to recover from peak
        """
        dd_df = self.calculate_drawdowns(portfolio_weights)

        # Find drawdown periods
        is_drawdown = dd_df['drawdown'] < 0
        drawdown_periods = []
        start = None

        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                drawdown_periods.append((start, i))
                start = None

        if start is not None:
            drawdown_periods.append((start, len(is_drawdown)))

        # Find longest period
        if drawdown_periods:
            max_duration_period = max(drawdown_periods, key=lambda x: x[1] - x[0])
            max_duration = max_duration_period[1] - max_duration_period[0]

            return {
                'max_duration_days': max_duration,
                'start_date': dd_df.index[max_duration_period[0]],
                'end_date': dd_df.index[max_duration_period[1]-1] if max_duration_period[1] < len(dd_df) else dd_df.index[-1]
            }
        else:
            return {'max_duration_days': 0}

    # ============== Stress Testing ==============

    def stress_test_scenarios(self, scenarios: Dict[str, float],
                             portfolio_weights: np.ndarray) -> pd.DataFrame:
        """
        Stress Test with Custom Scenarios

        Args:
            scenarios: Dict of {scenario_name: market_shock}
            portfolio_weights: Current portfolio weights

        Returns:
            Impact on portfolio for each scenario
        """
        results = []

        for scenario_name, shock in scenarios.items():
            # Apply shock to returns
            shocked_return = shock
            portfolio_impact = np.sum(portfolio_weights * shocked_return)

            results.append({
                'scenario': scenario_name,
                'shock': shock,
                'portfolio_impact': portfolio_impact
            })

        return pd.DataFrame(results)

    def historical_stress_test(self, crisis_period: Tuple[str, str],
                               portfolio_weights: np.ndarray) -> Dict:
        """
        Historical Stress Test

        Applies returns from historical crisis period to current portfolio

        Args:
            crisis_period: (start_date, end_date) tuple
            portfolio_weights: Current portfolio weights
        """
        crisis_returns = self.returns.loc[crisis_period[0]:crisis_period[1]]

        portfolio_crisis_returns = (crisis_returns @ portfolio_weights)
        cumulative_impact = (1 + portfolio_crisis_returns).prod() - 1

        return {
            'period': crisis_period,
            'cumulative_return': cumulative_impact,
            'worst_day': portfolio_crisis_returns.min(),
            'best_day': portfolio_crisis_returns.max(),
            'volatility': portfolio_crisis_returns.std() * np.sqrt(252)
        }

    # ============== Risk Limits ==============

    def check_risk_limits(self, portfolio_weights: np.ndarray,
                         limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if portfolio breaches risk limits

        Args:
            limits: Dict of {limit_name: threshold_value}
                e.g., {'max_var': -0.02, 'max_volatility': 0.20}

        Returns:
            Dict of {limit_name: is_within_limit}
        """
        results = {}

        # VaR limit
        if 'max_var' in limits:
            var_95 = self.historical_var(0.95, portfolio_weights)
            results['var_limit'] = var_95 >= limits['max_var']

        # Volatility limit
        if 'max_volatility' in limits:
            returns = (self.returns @ portfolio_weights)
            vol = returns.std() * np.sqrt(252)
            results['volatility_limit'] = vol <= limits['max_volatility']

        # Maximum position size
        if 'max_position' in limits:
            max_weight = np.max(portfolio_weights)
            results['position_size_limit'] = max_weight <= limits['max_position']

        # Concentration limit (sum of top 3)
        if 'max_concentration' in limits:
            top3_sum = np.sum(np.sort(portfolio_weights)[-3:])
            results['concentration_limit'] = top3_sum <= limits['max_concentration']

        return results

    def portfolio_risk_dashboard(self, portfolio_weights: np.ndarray) -> Dict:
        """
        Comprehensive risk dashboard

        Returns all key risk metrics
        """
        returns_series = (self.returns @ portfolio_weights)

        return {
            'var_95': self.historical_var(0.95, portfolio_weights),
            'var_99': self.historical_var(0.99, portfolio_weights),
            'expected_shortfall': self.expected_shortfall(0.95, portfolio_weights),
            'volatility': returns_series.std() * np.sqrt(252),
            'downside_deviation': self.downside_deviation(0, portfolio_weights),
            'max_drawdown': self.calculate_drawdowns(portfolio_weights)['drawdown'].min(),
            'skewness': self.skewness(portfolio_weights),
            'kurtosis': self.kurtosis(portfolio_weights),
            'sharpe_ratio': (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252))
        }
