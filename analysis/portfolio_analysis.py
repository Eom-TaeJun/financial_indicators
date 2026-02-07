"""
Portfolio Analysis

Traditional portfolio management and Modern Portfolio Theory:
- Portfolio Returns and Risk
- Sharpe Ratio, Sortino Ratio, Information Ratio
- Modern Portfolio Theory (Markowitz)
- Efficient Frontier
- Portfolio Optimization
- CAPM and Beta
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize


class PortfolioAnalysis:
    """Traditional portfolio analysis and optimization"""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize with asset returns

        Args:
            returns: DataFrame with asset returns (columns = assets, rows = time periods)
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.num_assets = len(returns.columns)

    # ============== Portfolio Metrics ==============

    def portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate expected portfolio return

        Return = Σ(weight_i × return_i)
        """
        return np.sum(weights * self.mean_returns)

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility (standard deviation)

        Volatility = √(W^T × Σ × W)
        where Σ is covariance matrix
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio variance
        """
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    # ============== Performance Ratios ==============

    def sharpe_ratio(self, weights: np.ndarray, annualization_factor: int = 252) -> float:
        """
        Sharpe Ratio (Risk-Adjusted Return)

        Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

        Most widely used risk-adjusted performance metric
        Annualization: 252 for daily, 52 for weekly, 12 for monthly
        """
        portfolio_ret = self.portfolio_return(weights) * annualization_factor
        portfolio_vol = self.portfolio_volatility(weights) * np.sqrt(annualization_factor)

        return (portfolio_ret - self.risk_free_rate) / portfolio_vol

    def sortino_ratio(self, weights: np.ndarray, annualization_factor: int = 252) -> float:
        """
        Sortino Ratio

        Like Sharpe but only penalizes downside volatility
        Sortino = (Return - Risk-Free Rate) / Downside Deviation
        """
        portfolio_returns = (self.returns @ weights)
        excess_returns = portfolio_returns - (self.risk_free_rate / annualization_factor)

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))

        avg_excess = excess_returns.mean() * annualization_factor
        downside_vol = downside_std * np.sqrt(annualization_factor)

        return avg_excess / downside_vol if downside_vol > 0 else np.nan

    def information_ratio(self, weights: np.ndarray, benchmark_returns: pd.Series) -> float:
        """
        Information Ratio

        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        Measures active return per unit of active risk
        """
        portfolio_returns = (self.returns @ weights)
        active_returns = portfolio_returns - benchmark_returns

        tracking_error = active_returns.std()
        active_return = active_returns.mean()

        return active_return / tracking_error if tracking_error > 0 else np.nan

    def treynor_ratio(self, weights: np.ndarray, portfolio_beta: float,
                     annualization_factor: int = 252) -> float:
        """
        Treynor Ratio

        Treynor = (Portfolio Return - Risk-Free Rate) / Beta
        Risk-adjusted return using systematic risk (beta)
        """
        portfolio_ret = self.portfolio_return(weights) * annualization_factor

        return (portfolio_ret - self.risk_free_rate) / portfolio_beta

    def calmar_ratio(self, weights: np.ndarray, max_drawdown: float) -> float:
        """
        Calmar Ratio

        Calmar = Annualized Return / Maximum Drawdown
        """
        if max_drawdown == 0:
            return np.nan
        annual_return = self.portfolio_return(weights) * 252
        return annual_return / abs(max_drawdown)

    # ============== Risk Metrics ==============

    def maximum_drawdown(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Maximum Drawdown (MDD)

        Largest peak-to-trough decline
        """
        portfolio_returns = (self.returns @ weights)
        cumulative = (1 + portfolio_returns).cumprod()

        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'drawdown_series': drawdown
        }

    def value_at_risk(self, weights: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Value at Risk (VaR)

        VaR at 95% confidence: maximum loss expected 95% of the time
        """
        portfolio_returns = (self.returns @ weights)
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    def conditional_var(self, weights: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR) / Expected Shortfall

        Average loss beyond VaR
        """
        portfolio_returns = (self.returns @ weights)
        var = self.value_at_risk(weights, confidence_level)

        # Returns worse than VaR
        tail_returns = portfolio_returns[portfolio_returns <= var]
        return tail_returns.mean()

    # ============== CAPM and Beta ==============

    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Beta (Systematic Risk)

        Beta = Cov(Asset, Market) / Var(Market)

        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        Beta = 1: Moves with market
        """
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        return covariance / market_variance if market_variance > 0 else np.nan

    def calculate_alpha(self, asset_return: float, asset_beta: float,
                       market_return: float) -> float:
        """
        Calculate Alpha (Jensen's Alpha)

        Alpha = Actual Return - Expected Return (from CAPM)
        Expected Return = Risk-Free Rate + Beta × (Market Return - Risk-Free Rate)

        Alpha > 0: Outperformance
        Alpha < 0: Underperformance
        """
        expected_return = self.risk_free_rate + asset_beta * (market_return - self.risk_free_rate)
        return asset_return - expected_return

    def capm_expected_return(self, beta: float, market_return: float) -> float:
        """
        CAPM Expected Return

        E(R) = Rf + β × (Rm - Rf)
        """
        return self.risk_free_rate + beta * (market_return - self.risk_free_rate)

    # ============== Modern Portfolio Theory ==============

    def efficient_frontier(self, num_portfolios: int = 10000) -> pd.DataFrame:
        """
        Generate Efficient Frontier (Markowitz)

        Monte Carlo simulation of random portfolios
        Returns portfolios with best return for each risk level
        """
        results = {
            'returns': [],
            'volatility': [],
            'sharpe': [],
            'weights': []
        }

        for _ in range(num_portfolios):
            # Random weights
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)

            # Calculate metrics
            ret = self.portfolio_return(weights) * 252
            vol = self.portfolio_volatility(weights) * np.sqrt(252)
            sharpe = (ret - self.risk_free_rate) / vol

            results['returns'].append(ret)
            results['volatility'].append(vol)
            results['sharpe'].append(sharpe)
            results['weights'].append(weights)

        return pd.DataFrame(results)

    def minimum_variance_portfolio(self) -> Dict:
        """
        Find Minimum Variance Portfolio

        Portfolio with lowest possible risk
        """
        def portfolio_variance_func(weights):
            return self.portfolio_variance(weights)

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1/self.num_assets] * self.num_assets)

        result = minimize(
            portfolio_variance_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        return {
            'weights': weights,
            'return': self.portfolio_return(weights) * 252,
            'volatility': self.portfolio_volatility(weights) * np.sqrt(252),
            'sharpe': self.sharpe_ratio(weights)
        }

    def maximum_sharpe_portfolio(self) -> Dict:
        """
        Find Maximum Sharpe Ratio Portfolio (Tangency Portfolio)

        Optimal risk-adjusted portfolio
        """
        def negative_sharpe(weights):
            return -self.sharpe_ratio(weights)

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1/self.num_assets] * self.num_assets)

        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        return {
            'weights': weights,
            'return': self.portfolio_return(weights) * 252,
            'volatility': self.portfolio_volatility(weights) * np.sqrt(252),
            'sharpe': self.sharpe_ratio(weights)
        }

    def target_return_portfolio(self, target_return: float) -> Dict:
        """
        Find Minimum Variance Portfolio for Target Return

        Portfolio with lowest risk for desired return
        """
        def portfolio_variance_func(weights):
            return self.portfolio_variance(weights)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) * 252 - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1/self.num_assets] * self.num_assets)

        result = minimize(
            portfolio_variance_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x
            return {
                'weights': weights,
                'return': self.portfolio_return(weights) * 252,
                'volatility': self.portfolio_volatility(weights) * np.sqrt(252),
                'sharpe': self.sharpe_ratio(weights)
            }
        else:
            return {'error': 'Optimization failed', 'message': result.message}

    def risk_parity_portfolio(self) -> Dict:
        """
        Risk Parity Portfolio (Equal Risk Contribution)

        Each asset contributes equally to portfolio risk
        """
        def risk_parity_objective(weights):
            portfolio_vol = self.portfolio_volatility(weights)
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Minimize difference from equal risk
            target_risk = portfolio_vol / self.num_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1/self.num_assets] * self.num_assets)

        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        return {
            'weights': weights,
            'return': self.portfolio_return(weights) * 252,
            'volatility': self.portfolio_volatility(weights) * np.sqrt(252),
            'sharpe': self.sharpe_ratio(weights)
        }

    # ============== Portfolio Rebalancing ==============

    def rebalancing_drift(self, current_weights: np.ndarray,
                         target_weights: np.ndarray) -> Dict:
        """
        Calculate portfolio drift from target weights
        """
        drift = current_weights - target_weights
        total_drift = np.sum(np.abs(drift))

        return {
            'weight_drift': drift,
            'total_drift': total_drift,
            'needs_rebalancing': total_drift > 0.05  # 5% threshold
        }

    def rebalancing_trades(self, current_weights: np.ndarray,
                          target_weights: np.ndarray,
                          portfolio_value: float) -> pd.DataFrame:
        """
        Calculate required trades to rebalance portfolio
        """
        current_values = current_weights * portfolio_value
        target_values = target_weights * portfolio_value
        trade_values = target_values - current_values

        trades_df = pd.DataFrame({
            'asset': self.returns.columns,
            'current_weight': current_weights,
            'target_weight': target_weights,
            'current_value': current_values,
            'target_value': target_values,
            'trade_value': trade_values,
            'action': ['Buy' if x > 0 else 'Sell' if x < 0 else 'Hold' for x in trade_values]
        })

        return trades_df

    # ============== Portfolio Attribution ==============

    def performance_attribution(self, portfolio_weights: np.ndarray,
                               benchmark_weights: np.ndarray) -> pd.DataFrame:
        """
        Performance Attribution Analysis

        Decomposes excess return into:
        - Allocation effect (weight differences)
        - Selection effect (return differences)
        """
        portfolio_ret = self.returns @ portfolio_weights
        benchmark_ret = self.returns @ benchmark_weights

        # Asset-level attribution
        asset_returns = self.returns.mean() * 252

        allocation_effect = (portfolio_weights - benchmark_weights) * benchmark_ret.mean() * 252
        selection_effect = benchmark_weights * (asset_returns - benchmark_ret.mean() * 252)
        interaction_effect = (portfolio_weights - benchmark_weights) * (asset_returns - benchmark_ret.mean() * 252)

        attribution_df = pd.DataFrame({
            'asset': self.returns.columns,
            'portfolio_weight': portfolio_weights,
            'benchmark_weight': benchmark_weights,
            'asset_return': asset_returns,
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_effect': allocation_effect + selection_effect + interaction_effect
        })

        return attribution_df

    # ============== Correlation Analysis ==============

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between assets
        """
        return self.returns.corr()

    def diversification_ratio(self, weights: np.ndarray) -> float:
        """
        Diversification Ratio

        DR = (Σ weight_i × volatility_i) / Portfolio Volatility

        DR > 1: Portfolio is diversified
        DR = 1: No diversification benefit
        """
        asset_vols = self.returns.std()
        weighted_vol_sum = np.sum(weights * asset_vols)
        portfolio_vol = self.portfolio_volatility(weights)

        return weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else np.nan

    def get_optimal_portfolios(self) -> Dict[str, Dict]:
        """
        Calculate all optimal portfolio strategies

        Returns:
            Dictionary with different optimization strategies
        """
        return {
            'minimum_variance': self.minimum_variance_portfolio(),
            'maximum_sharpe': self.maximum_sharpe_portfolio(),
            'risk_parity': self.risk_parity_portfolio()
        }
