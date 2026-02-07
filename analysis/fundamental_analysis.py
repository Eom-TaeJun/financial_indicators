"""
Fundamental Analysis

Traditional fundamental analysis metrics and ratios:
- Profitability Ratios (ROE, ROA, Profit Margins)
- Valuation Ratios (P/E, P/B, EV/EBITDA)
- Liquidity Ratios (Current, Quick)
- Leverage Ratios (Debt/Equity, Interest Coverage)
- Efficiency Ratios (Asset Turnover, Inventory Turnover)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class FundamentalAnalysis:
    """Traditional fundamental analysis ratios and metrics"""

    def __init__(self, financials: Optional[Dict] = None):
        """
        Initialize with financial statement data

        Args:
            financials: Dict with 'income_statement', 'balance_sheet', 'cash_flow'
        """
        self.financials = financials or {}
        self.ratios = {}

    # ============== Profitability Ratios ==============

    def roe(self, net_income: float, shareholders_equity: float) -> float:
        """
        Return on Equity (ROE)

        ROE = Net Income / Shareholders' Equity
        Measures profitability relative to equity
        """
        if shareholders_equity == 0:
            return np.nan
        return (net_income / shareholders_equity) * 100

    def roa(self, net_income: float, total_assets: float) -> float:
        """
        Return on Assets (ROA)

        ROA = Net Income / Total Assets
        Measures how efficiently assets generate profit
        """
        if total_assets == 0:
            return np.nan
        return (net_income / total_assets) * 100

    def roic(self, nopat: float, invested_capital: float) -> float:
        """
        Return on Invested Capital (ROIC)

        ROIC = NOPAT / Invested Capital
        NOPAT = Net Operating Profit After Tax
        """
        if invested_capital == 0:
            return np.nan
        return (nopat / invested_capital) * 100

    def gross_margin(self, revenue: float, cogs: float) -> float:
        """
        Gross Profit Margin

        Gross Margin = (Revenue - COGS) / Revenue
        """
        if revenue == 0:
            return np.nan
        return ((revenue - cogs) / revenue) * 100

    def operating_margin(self, operating_income: float, revenue: float) -> float:
        """
        Operating Profit Margin

        Operating Margin = Operating Income / Revenue
        """
        if revenue == 0:
            return np.nan
        return (operating_income / revenue) * 100

    def net_margin(self, net_income: float, revenue: float) -> float:
        """
        Net Profit Margin

        Net Margin = Net Income / Revenue
        """
        if revenue == 0:
            return np.nan
        return (net_income / revenue) * 100

    # ============== Valuation Ratios ==============

    def pe_ratio(self, price: float, earnings_per_share: float) -> float:
        """
        Price-to-Earnings Ratio (P/E)

        P/E = Stock Price / EPS
        Traditional valuation multiple
        """
        if earnings_per_share == 0:
            return np.nan
        return price / earnings_per_share

    def pb_ratio(self, price: float, book_value_per_share: float) -> float:
        """
        Price-to-Book Ratio (P/B)

        P/B = Stock Price / Book Value per Share
        Value investing metric
        """
        if book_value_per_share == 0:
            return np.nan
        return price / book_value_per_share

    def ps_ratio(self, market_cap: float, revenue: float) -> float:
        """
        Price-to-Sales Ratio (P/S)

        P/S = Market Cap / Revenue
        """
        if revenue == 0:
            return np.nan
        return market_cap / revenue

    def peg_ratio(self, pe_ratio: float, earnings_growth_rate: float) -> float:
        """
        PEG Ratio

        PEG = P/E Ratio / Earnings Growth Rate
        Adjusts P/E for growth
        """
        if earnings_growth_rate == 0:
            return np.nan
        return pe_ratio / earnings_growth_rate

    def ev_ebitda(self, enterprise_value: float, ebitda: float) -> float:
        """
        EV/EBITDA Ratio

        EV/EBITDA = Enterprise Value / EBITDA
        Popular valuation metric
        """
        if ebitda == 0:
            return np.nan
        return enterprise_value / ebitda

    def calculate_enterprise_value(self, market_cap: float, total_debt: float,
                                   cash: float) -> float:
        """
        Enterprise Value (EV)

        EV = Market Cap + Total Debt - Cash
        """
        return market_cap + total_debt - cash

    # ============== Liquidity Ratios ==============

    def current_ratio(self, current_assets: float, current_liabilities: float) -> float:
        """
        Current Ratio

        Current Ratio = Current Assets / Current Liabilities
        Measures short-term liquidity
        """
        if current_liabilities == 0:
            return np.nan
        return current_assets / current_liabilities

    def quick_ratio(self, current_assets: float, inventory: float,
                   current_liabilities: float) -> float:
        """
        Quick Ratio (Acid-Test Ratio)

        Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        Conservative liquidity measure
        """
        if current_liabilities == 0:
            return np.nan
        return (current_assets - inventory) / current_liabilities

    def cash_ratio(self, cash: float, current_liabilities: float) -> float:
        """
        Cash Ratio

        Cash Ratio = Cash / Current Liabilities
        Most conservative liquidity ratio
        """
        if current_liabilities == 0:
            return np.nan
        return cash / current_liabilities

    # ============== Leverage Ratios ==============

    def debt_to_equity(self, total_debt: float, shareholders_equity: float) -> float:
        """
        Debt-to-Equity Ratio (D/E)

        D/E = Total Debt / Shareholders' Equity
        Measures financial leverage
        """
        if shareholders_equity == 0:
            return np.nan
        return total_debt / shareholders_equity

    def debt_to_assets(self, total_debt: float, total_assets: float) -> float:
        """
        Debt-to-Assets Ratio

        D/A = Total Debt / Total Assets
        """
        if total_assets == 0:
            return np.nan
        return total_debt / total_assets

    def equity_multiplier(self, total_assets: float, shareholders_equity: float) -> float:
        """
        Equity Multiplier

        EM = Total Assets / Shareholders' Equity
        Part of DuPont analysis
        """
        if shareholders_equity == 0:
            return np.nan
        return total_assets / shareholders_equity

    def interest_coverage(self, ebit: float, interest_expense: float) -> float:
        """
        Interest Coverage Ratio

        ICR = EBIT / Interest Expense
        Measures ability to pay interest
        """
        if interest_expense == 0:
            return np.nan
        return ebit / interest_expense

    # ============== Efficiency Ratios ==============

    def asset_turnover(self, revenue: float, average_total_assets: float) -> float:
        """
        Asset Turnover Ratio

        Asset Turnover = Revenue / Average Total Assets
        Measures asset efficiency
        """
        if average_total_assets == 0:
            return np.nan
        return revenue / average_total_assets

    def inventory_turnover(self, cogs: float, average_inventory: float) -> float:
        """
        Inventory Turnover Ratio

        Inventory Turnover = COGS / Average Inventory
        """
        if average_inventory == 0:
            return np.nan
        return cogs / average_inventory

    def days_inventory_outstanding(self, inventory_turnover: float) -> float:
        """
        Days Inventory Outstanding (DIO)

        DIO = 365 / Inventory Turnover
        """
        if inventory_turnover == 0:
            return np.nan
        return 365 / inventory_turnover

    def receivables_turnover(self, revenue: float, average_receivables: float) -> float:
        """
        Receivables Turnover Ratio

        Receivables Turnover = Revenue / Average Receivables
        """
        if average_receivables == 0:
            return np.nan
        return revenue / average_receivables

    def days_sales_outstanding(self, receivables_turnover: float) -> float:
        """
        Days Sales Outstanding (DSO)

        DSO = 365 / Receivables Turnover
        """
        if receivables_turnover == 0:
            return np.nan
        return 365 / receivables_turnover

    # ============== DuPont Analysis ==============

    def dupont_analysis(self, net_income: float, revenue: float,
                       total_assets: float, shareholders_equity: float) -> Dict[str, float]:
        """
        DuPont Analysis (3-Factor Model)

        ROE = Net Margin × Asset Turnover × Equity Multiplier

        Breaks down ROE into:
        - Profitability (Net Margin)
        - Efficiency (Asset Turnover)
        - Leverage (Equity Multiplier)
        """
        net_margin = self.net_margin(net_income, revenue)
        asset_turnover = revenue / total_assets if total_assets > 0 else np.nan
        equity_multiplier = self.equity_multiplier(total_assets, shareholders_equity)

        roe = self.roe(net_income, shareholders_equity)

        return {
            'roe': roe,
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'dupont_roe': net_margin * asset_turnover * equity_multiplier / 100
        }

    # ============== Growth Metrics ==============

    def revenue_growth(self, current_revenue: float, previous_revenue: float) -> float:
        """
        Revenue Growth Rate

        Growth = (Current - Previous) / Previous
        """
        if previous_revenue == 0:
            return np.nan
        return ((current_revenue - previous_revenue) / previous_revenue) * 100

    def earnings_growth(self, current_earnings: float, previous_earnings: float) -> float:
        """
        Earnings Growth Rate
        """
        if previous_earnings == 0:
            return np.nan
        return ((current_earnings - previous_earnings) / previous_earnings) * 100

    def cagr(self, beginning_value: float, ending_value: float, num_periods: float) -> float:
        """
        Compound Annual Growth Rate (CAGR)

        CAGR = (Ending Value / Beginning Value)^(1/n) - 1
        """
        if beginning_value == 0 or num_periods == 0:
            return np.nan
        return (pow(ending_value / beginning_value, 1 / num_periods) - 1) * 100

    # ============== Dividend Metrics ==============

    def dividend_yield(self, annual_dividend: float, stock_price: float) -> float:
        """
        Dividend Yield

        Dividend Yield = Annual Dividend / Stock Price
        """
        if stock_price == 0:
            return np.nan
        return (annual_dividend / stock_price) * 100

    def dividend_payout_ratio(self, dividends: float, net_income: float) -> float:
        """
        Dividend Payout Ratio

        Payout Ratio = Dividends / Net Income
        """
        if net_income == 0:
            return np.nan
        return (dividends / net_income) * 100

    def retention_ratio(self, payout_ratio: float) -> float:
        """
        Retention Ratio (Plowback Ratio)

        Retention = 1 - Payout Ratio
        """
        return 100 - payout_ratio

    # ============== Altman Z-Score ==============

    def altman_z_score(self, working_capital: float, retained_earnings: float,
                      ebit: float, market_cap: float, total_liabilities: float,
                      revenue: float, total_assets: float) -> Dict[str, float]:
        """
        Altman Z-Score (Bankruptcy Prediction)

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Cap / Total Liabilities
        X5 = Revenue / Total Assets

        Z > 2.99: Safe
        1.81 < Z < 2.99: Grey zone
        Z < 1.81: Distress
        """
        if total_assets == 0 or total_liabilities == 0:
            return {'z_score': np.nan, 'status': 'Unknown'}

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities
        x5 = revenue / total_assets

        z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

        if z_score > 2.99:
            status = 'Safe'
        elif z_score > 1.81:
            status = 'Grey Zone'
        else:
            status = 'Distress'

        return {
            'z_score': z_score,
            'status': status,
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5
        }

    # ============== Piotroski F-Score ==============

    def piotroski_f_score(self, current: Dict, previous: Dict) -> Dict[str, any]:
        """
        Piotroski F-Score (Value Stock Screening)

        9-point score based on:
        - Profitability (4 points)
        - Leverage/Liquidity (3 points)
        - Operating Efficiency (2 points)

        Score 8-9: Strong
        Score 0-2: Weak
        """
        score = 0
        details = {}

        # Profitability (4 points)
        # 1. Positive ROA
        if current.get('roa', 0) > 0:
            score += 1
            details['positive_roa'] = True

        # 2. Positive Operating Cash Flow
        if current.get('operating_cash_flow', 0) > 0:
            score += 1
            details['positive_ocf'] = True

        # 3. Increasing ROA
        if current.get('roa', 0) > previous.get('roa', 0):
            score += 1
            details['increasing_roa'] = True

        # 4. Quality of Earnings (OCF > Net Income)
        if current.get('operating_cash_flow', 0) > current.get('net_income', 0):
            score += 1
            details['quality_earnings'] = True

        # Leverage/Liquidity (3 points)
        # 5. Decreasing Long-term Debt
        if current.get('long_term_debt', float('inf')) < previous.get('long_term_debt', float('inf')):
            score += 1
            details['decreasing_debt'] = True

        # 6. Increasing Current Ratio
        if current.get('current_ratio', 0) > previous.get('current_ratio', 0):
            score += 1
            details['increasing_current_ratio'] = True

        # 7. No New Shares Issued
        if current.get('shares_outstanding', 0) <= previous.get('shares_outstanding', float('inf')):
            score += 1
            details['no_dilution'] = True

        # Operating Efficiency (2 points)
        # 8. Increasing Gross Margin
        if current.get('gross_margin', 0) > previous.get('gross_margin', 0):
            score += 1
            details['increasing_margin'] = True

        # 9. Increasing Asset Turnover
        if current.get('asset_turnover', 0) > previous.get('asset_turnover', 0):
            score += 1
            details['increasing_turnover'] = True

        if score >= 8:
            rating = 'Strong'
        elif score >= 5:
            rating = 'Moderate'
        else:
            rating = 'Weak'

        return {
            'f_score': score,
            'rating': rating,
            'details': details
        }

    def get_all_ratios(self, financial_data: Dict) -> Dict[str, float]:
        """
        Calculate all fundamental ratios from financial statement data

        Args:
            financial_data: Dict containing all necessary financial metrics

        Returns:
            Dictionary of all calculated ratios
        """
        ratios = {}

        # Extract data
        net_income = financial_data.get('net_income', 0)
        revenue = financial_data.get('revenue', 0)
        total_assets = financial_data.get('total_assets', 0)
        shareholders_equity = financial_data.get('shareholders_equity', 0)
        total_debt = financial_data.get('total_debt', 0)
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        cogs = financial_data.get('cogs', 0)
        operating_income = financial_data.get('operating_income', 0)

        # Profitability
        ratios['roe'] = self.roe(net_income, shareholders_equity)
        ratios['roa'] = self.roa(net_income, total_assets)
        ratios['gross_margin'] = self.gross_margin(revenue, cogs)
        ratios['operating_margin'] = self.operating_margin(operating_income, revenue)
        ratios['net_margin'] = self.net_margin(net_income, revenue)

        # Liquidity
        ratios['current_ratio'] = self.current_ratio(current_assets, current_liabilities)

        # Leverage
        ratios['debt_to_equity'] = self.debt_to_equity(total_debt, shareholders_equity)
        ratios['debt_to_assets'] = self.debt_to_assets(total_debt, total_assets)

        # Efficiency
        ratios['asset_turnover'] = self.asset_turnover(revenue, total_assets)

        return ratios
