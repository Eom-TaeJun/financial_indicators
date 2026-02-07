"""
Valuation Models

Traditional valuation techniques:
- Discounted Cash Flow (DCF)
- Dividend Discount Model (DDM)
- Comparables Valuation (Multiples)
- Asset-Based Valuation
- Option Pricing (Black-Scholes)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class ValuationModels:
    """Traditional valuation models"""

    def __init__(self, discount_rate: float = 0.10):
        """
        Args:
            discount_rate: Required rate of return / WACC
        """
        self.discount_rate = discount_rate

    # ============== Discounted Cash Flow (DCF) ==============

    def dcf_valuation(self, free_cash_flows: List[float], terminal_growth_rate: float = 0.02,
                     discount_rate: Optional[float] = None) -> Dict:
        """
        Discounted Cash Flow Valuation

        PV = Σ(FCF_t / (1 + r)^t) + Terminal Value

        Most fundamental valuation method
        """
        if discount_rate is None:
            discount_rate = self.discount_rate

        present_values = []
        for year, fcf in enumerate(free_cash_flows, start=1):
            pv = fcf / ((1 + discount_rate) ** year)
            present_values.append(pv)

        # Terminal Value (Gordon Growth Model)
        terminal_fcf = free_cash_flows[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        terminal_pv = terminal_value / ((1 + discount_rate) ** len(free_cash_flows))

        enterprise_value = sum(present_values) + terminal_pv

        return {
            'enterprise_value': enterprise_value,
            'pv_forecast_period': sum(present_values),
            'pv_terminal_value': terminal_pv,
            'terminal_value': terminal_value,
            'discount_rate': discount_rate
        }

    def unlevered_fcf(self, ebit: float, tax_rate: float, depreciation: float,
                     capex: float, change_in_nwc: float) -> float:
        """
        Calculate Free Cash Flow to Firm (FCFF)

        FCFF = EBIT × (1 - Tax Rate) + D&A - CapEx - ΔNWC
        """
        nopat = ebit * (1 - tax_rate)
        fcf = nopat + depreciation - capex - change_in_nwc
        return fcf

    def levered_fcf(self, net_income: float, depreciation: float,
                   capex: float, change_in_nwc: float,
                   net_borrowing: float) -> float:
        """
        Calculate Free Cash Flow to Equity (FCFE)

        FCFE = Net Income + D&A - CapEx - ΔNWC + Net Borrowing
        """
        fcfe = net_income + depreciation - capex - change_in_nwc + net_borrowing
        return fcfe

    # ============== Dividend Discount Model (DDM) ==============

    def gordon_growth_model(self, current_dividend: float, growth_rate: float,
                           required_return: Optional[float] = None) -> float:
        """
        Gordon Growth Model (Constant Growth DDM)

        P0 = D1 / (r - g)
        where D1 = D0 × (1 + g)

        Works for mature companies with stable dividends
        """
        if required_return is None:
            required_return = self.discount_rate

        if required_return <= growth_rate:
            raise ValueError("Required return must be greater than growth rate")

        next_dividend = current_dividend * (1 + growth_rate)
        value = next_dividend / (required_return - growth_rate)

        return value

    def multi_stage_ddm(self, current_dividend: float,
                       high_growth_rate: float, high_growth_years: int,
                       stable_growth_rate: float,
                       required_return: Optional[float] = None) -> Dict:
        """
        Two-Stage Dividend Discount Model

        Stage 1: High growth period
        Stage 2: Stable growth (Gordon Growth)

        Used for companies transitioning from growth to maturity
        """
        if required_return is None:
            required_return = self.discount_rate

        # Stage 1: High growth
        stage1_value = 0
        dividend = current_dividend

        for year in range(1, high_growth_years + 1):
            dividend = dividend * (1 + high_growth_rate)
            pv = dividend / ((1 + required_return) ** year)
            stage1_value += pv

        # Stage 2: Stable growth (terminal value)
        terminal_dividend = dividend * (1 + stable_growth_rate)
        terminal_value = terminal_dividend / (required_return - stable_growth_rate)
        stage2_value = terminal_value / ((1 + required_return) ** high_growth_years)

        total_value = stage1_value + stage2_value

        return {
            'total_value': total_value,
            'stage1_value': stage1_value,
            'stage2_value': stage2_value,
            'terminal_value': terminal_value
        }

    def h_model(self, current_dividend: float, initial_growth_rate: float,
               stable_growth_rate: float, half_life_years: float,
               required_return: Optional[float] = None) -> float:
        """
        H-Model (Linear Declining Growth)

        P0 = D0 × (1 + gL) / (r - gL) + D0 × H × (gS - gL) / (r - gL)

        Assumes growth declines linearly from high to stable
        """
        if required_return is None:
            required_return = self.discount_rate

        d0 = current_dividend
        gl = stable_growth_rate
        gs = initial_growth_rate
        r = required_return
        h = half_life_years / 2

        value = (d0 * (1 + gl) / (r - gl) +
                d0 * h * (gs - gl) / (r - gl))

        return value

    # ============== Relative Valuation (Comparables) ==============

    def comparable_valuation(self, company_metric: float, peer_multiples: List[float],
                           method: str = 'median') -> Dict:
        """
        Comparable Company Valuation

        Value = Company Metric × Peer Multiple

        Args:
            company_metric: Company's base metric (EBITDA, Earnings, Sales, etc.)
            peer_multiples: List of peer multiples (P/E, EV/EBITDA, etc.)
            method: 'median', 'mean', or 'harmonic_mean'
        """
        if method == 'median':
            multiple = np.median(peer_multiples)
        elif method == 'mean':
            multiple = np.mean(peer_multiples)
        elif method == 'harmonic_mean':
            multiple = len(peer_multiples) / np.sum(1.0 / np.array(peer_multiples))
        else:
            raise ValueError("Method must be 'median', 'mean', or 'harmonic_mean'")

        implied_value = company_metric * multiple

        return {
            'implied_value': implied_value,
            'multiple_used': multiple,
            'peer_multiples': {
                'min': min(peer_multiples),
                'max': max(peer_multiples),
                'mean': np.mean(peer_multiples),
                'median': np.median(peer_multiples),
                'std': np.std(peer_multiples)
            }
        }

    def transaction_multiples(self, company_metric: float,
                            transaction_multiples: List[float]) -> Dict:
        """
        Precedent Transaction Analysis

        Similar to comparable but uses M&A transaction multiples
        Usually shows premium to trading multiples
        """
        return self.comparable_valuation(company_metric, transaction_multiples, 'median')

    # ============== Weighted Average Cost of Capital (WACC) ==============

    def wacc(self, equity_value: float, debt_value: float,
            cost_of_equity: float, cost_of_debt: float,
            tax_rate: float) -> float:
        """
        Weighted Average Cost of Capital

        WACC = (E/V) × Re + (D/V) × Rd × (1 - Tc)

        where:
        - E = Equity value
        - D = Debt value
        - V = E + D (Total value)
        - Re = Cost of equity
        - Rd = Cost of debt
        - Tc = Corporate tax rate
        """
        total_value = equity_value + debt_value

        if total_value == 0:
            return 0

        equity_weight = equity_value / total_value
        debt_weight = debt_value / total_value

        wacc = (equity_weight * cost_of_equity +
               debt_weight * cost_of_debt * (1 - tax_rate))

        return wacc

    def cost_of_equity_capm(self, risk_free_rate: float, beta: float,
                           market_return: float) -> float:
        """
        Cost of Equity using CAPM

        Re = Rf + β × (Rm - Rf)
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)

    def cost_of_equity_ddm(self, current_price: float, expected_dividend: float,
                          growth_rate: float) -> float:
        """
        Cost of Equity using Dividend Growth Model

        Re = (D1 / P0) + g
        """
        return (expected_dividend / current_price) + growth_rate

    # ============== Asset-Based Valuation ==============

    def book_value_valuation(self, total_assets: float, total_liabilities: float,
                           adjustments: Optional[Dict[str, float]] = None) -> Dict:
        """
        Book Value (Net Asset Value)

        NAV = Total Assets - Total Liabilities + Adjustments

        Useful for asset-heavy companies
        """
        nav = total_assets - total_liabilities

        if adjustments:
            for item, value in adjustments.items():
                nav += value

        return {
            'net_asset_value': nav,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'adjustments': adjustments or {}
        }

    def liquidation_value(self, assets: Dict[str, float],
                         recovery_rates: Dict[str, float],
                         liabilities: float) -> Dict:
        """
        Liquidation Value

        Value if company were liquidated today
        Uses recovery rates for each asset class
        """
        liquidation_proceeds = 0
        asset_breakdown = {}

        for asset_type, asset_value in assets.items():
            recovery_rate = recovery_rates.get(asset_type, 0.5)  # Default 50%
            recoverable = asset_value * recovery_rate
            liquidation_proceeds += recoverable
            asset_breakdown[asset_type] = recoverable

        net_liquidation_value = liquidation_proceeds - liabilities

        return {
            'net_liquidation_value': net_liquidation_value,
            'gross_proceeds': liquidation_proceeds,
            'liabilities': liabilities,
            'asset_breakdown': asset_breakdown
        }

    # ============== Sum-of-the-Parts Valuation ==============

    def sum_of_parts_valuation(self, segments: List[Dict]) -> Dict:
        """
        Sum-of-the-Parts (SOTP) Valuation

        For conglomerates with multiple business segments

        Args:
            segments: List of dicts with 'name', 'value' for each segment
        """
        total_value = sum(seg['value'] for seg in segments)

        return {
            'total_value': total_value,
            'segments': segments,
            'segment_count': len(segments)
        }

    # ============== Option Pricing ==============

    def black_scholes_call(self, stock_price: float, strike_price: float,
                          time_to_expiry: float, risk_free_rate: float,
                          volatility: float) -> Dict:
        """
        Black-Scholes Option Pricing (Call)

        C = S × N(d1) - K × e^(-rT) × N(d2)

        where:
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T
        """
        S = stock_price
        K = strike_price
        T = time_to_expiry
        r = risk_free_rate
        sigma = volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

        # Greeks
        delta = stats.norm.cdf(d1)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * stats.norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)

        return {
            'call_price': call_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # Per 1% change in volatility
            'theta': theta / 365,  # Per day
            'rho': rho / 100  # Per 1% change in rate
        }

    def black_scholes_put(self, stock_price: float, strike_price: float,
                         time_to_expiry: float, risk_free_rate: float,
                         volatility: float) -> Dict:
        """
        Black-Scholes Option Pricing (Put)

        P = K × e^(-rT) × N(-d2) - S × N(-d1)
        """
        S = stock_price
        K = strike_price
        T = time_to_expiry
        r = risk_free_rate
        sigma = volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

        # Greeks
        delta = stats.norm.cdf(d1) - 1
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)

        return {
            'put_price': put_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,
            'theta': theta / 365,
            'rho': rho / 100
        }

    # ============== Sensitivity Analysis ==============

    def dcf_sensitivity_analysis(self, base_case: Dict,
                                 wacc_range: List[float],
                                 growth_range: List[float]) -> pd.DataFrame:
        """
        DCF Sensitivity Analysis

        Creates sensitivity table for WACC and terminal growth rate
        """
        sensitivity_matrix = []

        for growth in growth_range:
            row = []
            for wacc in wacc_range:
                # Recalculate DCF with different assumptions
                result = self.dcf_valuation(
                    base_case['free_cash_flows'],
                    terminal_growth_rate=growth,
                    discount_rate=wacc
                )
                row.append(result['enterprise_value'])
            sensitivity_matrix.append(row)

        df = pd.DataFrame(
            sensitivity_matrix,
            index=[f'{g:.1%}' for g in growth_range],
            columns=[f'{w:.1%}' for w in wacc_range]
        )
        df.index.name = 'Terminal Growth'
        df.columns.name = 'WACC'

        return df

    def monte_carlo_valuation(self, base_fcf: float, num_simulations: int = 10000,
                             revenue_growth_mean: float = 0.05,
                             revenue_growth_std: float = 0.10,
                             margin_mean: float = 0.15,
                             margin_std: float = 0.05,
                             num_years: int = 5) -> Dict:
        """
        Monte Carlo DCF Simulation

        Simulates range of valuations under uncertainty
        """
        valuations = []

        for _ in range(num_simulations):
            fcfs = []
            fcf = base_fcf

            for year in range(num_years):
                # Random growth and margin
                growth = np.random.normal(revenue_growth_mean, revenue_growth_std)
                margin = np.random.normal(margin_mean, margin_std)

                fcf = fcf * (1 + growth) * (1 + np.random.normal(0, 0.1))
                fcfs.append(fcf)

            # Random terminal growth
            terminal_growth = np.random.normal(0.02, 0.01)
            terminal_growth = max(0, min(terminal_growth, 0.05))  # Bound 0-5%

            # Random WACC
            wacc = np.random.normal(0.10, 0.02)
            wacc = max(0.05, min(wacc, 0.20))  # Bound 5-20%

            result = self.dcf_valuation(fcfs, terminal_growth, wacc)
            valuations.append(result['enterprise_value'])

        valuations = np.array(valuations)

        return {
            'mean': np.mean(valuations),
            'median': np.median(valuations),
            'std': np.std(valuations),
            'percentile_5': np.percentile(valuations, 5),
            'percentile_25': np.percentile(valuations, 25),
            'percentile_75': np.percentile(valuations, 75),
            'percentile_95': np.percentile(valuations, 95),
            'min': np.min(valuations),
            'max': np.max(valuations)
        }
