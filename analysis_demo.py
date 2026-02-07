"""
Financial Analysis Demonstration

Shows how to use traditional finance methods with collected data
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import (
    TechnicalAnalysis,
    FundamentalAnalysis,
    PortfolioAnalysis,
    RiskManagement,
    ValuationModels
)
from collectors.market_collector import MarketCollector
from collectors.crypto_collector import CryptoCollector


def demo_technical_analysis():
    """Demonstrate technical analysis indicators"""
    print("="*80)
    print("TECHNICAL ANALYSIS DEMONSTRATION")
    print("="*80)

    # Collect sample data
    print("\n1. Collecting price data for AAPL...")
    collector = MarketCollector(lookback_days=100)
    data, status = collector.fetch_ticker('AAPL', 'Apple Inc.')

    if data is None:
        print("Failed to fetch data")
        return

    print(f"✓ Fetched {len(data)} days of data")

    # Initialize technical analysis
    ta = TechnicalAnalysis(data)

    # Calculate indicators
    print("\n2. Calculating Technical Indicators...")

    # Moving Averages
    sma_20 = ta.moving_average(20, 'SMA')
    sma_50 = ta.moving_average(50, 'SMA')
    print(f"   SMA(20): ${sma_20.iloc[-1]:.2f}")
    print(f"   SMA(50): ${sma_50.iloc[-1]:.2f}")

    # MACD
    macd = ta.macd()
    print(f"   MACD: {macd['macd'].iloc[-1]:.2f}")
    print(f"   Signal: {macd['signal'].iloc[-1]:.2f}")
    print(f"   Histogram: {macd['histogram'].iloc[-1]:.2f}")

    # RSI
    rsi = ta.rsi()
    print(f"   RSI: {rsi.iloc[-1]:.2f}")

    # Bollinger Bands
    bb = ta.bollinger_bands()
    current_price = data['Close'].iloc[-1]
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   BB Upper: ${bb['upper'].iloc[-1]:.2f}")
    print(f"   BB Lower: ${bb['lower'].iloc[-1]:.2f}")

    # Trading Signals
    print("\n3. Trading Signals:")
    signals = ta.get_signals()
    for indicator, signal in signals.items():
        print(f"   {indicator}: {signal}")

    print("\n✓ Technical Analysis Complete")


def demo_portfolio_analysis():
    """Demonstrate portfolio analysis and optimization"""
    print("\n" + "="*80)
    print("PORTFOLIO ANALYSIS DEMONSTRATION")
    print("="*80)

    # Collect data for multiple assets
    print("\n1. Collecting data for portfolio assets...")
    collector = MarketCollector(lookback_days=252)  # 1 year

    tickers = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla'
    }

    price_data = {}
    for ticker, name in tickers.items():
        data, status = collector.fetch_ticker(ticker, name)
        if data is not None:
            price_data[ticker] = data['Close']
            print(f"   ✓ {ticker}")

    if len(price_data) < 2:
        print("Not enough data for portfolio analysis")
        return

    # Create returns DataFrame
    prices_df = pd.DataFrame(price_data)
    returns_df = prices_df.pct_change().dropna()

    print(f"\n✓ Collected data for {len(price_data)} assets")

    # Initialize portfolio analysis
    pa = PortfolioAnalysis(returns_df, risk_free_rate=0.04)

    # Equal weight portfolio
    equal_weights = np.array([1/len(price_data)] * len(price_data))

    print("\n2. Portfolio Metrics (Equal Weight):")
    print(f"   Expected Return: {pa.portfolio_return(equal_weights) * 252:.2%}")
    print(f"   Volatility: {pa.portfolio_volatility(equal_weights) * np.sqrt(252):.2%}")
    print(f"   Sharpe Ratio: {pa.sharpe_ratio(equal_weights):.2f}")
    print(f"   Sortino Ratio: {pa.sortino_ratio(equal_weights):.2f}")

    # Maximum Drawdown
    mdd_info = pa.maximum_drawdown(equal_weights)
    print(f"   Max Drawdown: {mdd_info['max_drawdown']:.2%}")

    # Risk metrics
    print("\n3. Risk Metrics:")
    rm = RiskManagement(returns_df)
    print(f"   VaR (95%): {rm.historical_var(0.95, equal_weights):.2%}")
    print(f"   CVaR (95%): {rm.expected_shortfall(0.95, equal_weights):.2%}")

    # Optimization
    print("\n4. Portfolio Optimization:")

    print("\n   a) Minimum Variance Portfolio:")
    min_var = pa.minimum_variance_portfolio()
    for i, ticker in enumerate(tickers.keys()):
        print(f"      {ticker}: {min_var['weights'][i]:.1%}")
    print(f"      Return: {min_var['return']:.2%}")
    print(f"      Volatility: {min_var['volatility']:.2%}")
    print(f"      Sharpe: {min_var['sharpe']:.2f}")

    print("\n   b) Maximum Sharpe Portfolio:")
    max_sharpe = pa.maximum_sharpe_portfolio()
    for i, ticker in enumerate(tickers.keys()):
        print(f"      {ticker}: {max_sharpe['weights'][i]:.1%}")
    print(f"      Return: {max_sharpe['return']:.2%}")
    print(f"      Volatility: {max_sharpe['volatility']:.2%}")
    print(f"      Sharpe: {max_sharpe['sharpe']:.2f}")

    print("\n   c) Risk Parity Portfolio:")
    risk_parity = pa.risk_parity_portfolio()
    for i, ticker in enumerate(tickers.keys()):
        print(f"      {ticker}: {risk_parity['weights'][i]:.1%}")
    print(f"      Return: {risk_parity['return']:.2%}")
    print(f"      Volatility: {risk_parity['volatility']:.2%}")
    print(f"      Sharpe: {risk_parity['sharpe']:.2f}")

    # Correlation Analysis
    print("\n5. Correlation Matrix:")
    corr = pa.correlation_matrix()
    print(corr.round(2))

    print("\n✓ Portfolio Analysis Complete")


def demo_fundamental_analysis():
    """Demonstrate fundamental analysis ratios"""
    print("\n" + "="*80)
    print("FUNDAMENTAL ANALYSIS DEMONSTRATION")
    print("="*80)

    # Example financial data (in millions)
    financial_data = {
        'revenue': 394_000,
        'cogs': 223_000,
        'operating_income': 119_000,
        'net_income': 99_000,
        'total_assets': 352_000,
        'current_assets': 135_000,
        'shareholders_equity': 63_000,
        'total_debt': 111_000,
        'current_liabilities': 125_000,
        'cash': 48_000,
        'inventory': 6_000,
    }

    fa = FundamentalAnalysis()

    print("\n1. Profitability Ratios:")
    print(f"   ROE: {fa.roe(financial_data['net_income'], financial_data['shareholders_equity']):.2f}%")
    print(f"   ROA: {fa.roa(financial_data['net_income'], financial_data['total_assets']):.2f}%")
    print(f"   Gross Margin: {fa.gross_margin(financial_data['revenue'], financial_data['cogs']):.2f}%")
    print(f"   Operating Margin: {fa.operating_margin(financial_data['operating_income'], financial_data['revenue']):.2f}%")
    print(f"   Net Margin: {fa.net_margin(financial_data['net_income'], financial_data['revenue']):.2f}%")

    print("\n2. Liquidity Ratios:")
    print(f"   Current Ratio: {fa.current_ratio(financial_data['current_assets'], financial_data['current_liabilities']):.2f}")
    print(f"   Quick Ratio: {fa.quick_ratio(financial_data['current_assets'], financial_data['inventory'], financial_data['current_liabilities']):.2f}")

    print("\n3. Leverage Ratios:")
    print(f"   Debt-to-Equity: {fa.debt_to_equity(financial_data['total_debt'], financial_data['shareholders_equity']):.2f}")
    print(f"   Debt-to-Assets: {fa.debt_to_assets(financial_data['total_debt'], financial_data['total_assets']):.2f}")

    print("\n4. DuPont Analysis:")
    dupont = fa.dupont_analysis(
        financial_data['net_income'],
        financial_data['revenue'],
        financial_data['total_assets'],
        financial_data['shareholders_equity']
    )
    print(f"   ROE: {dupont['roe']:.2f}%")
    print(f"   = Net Margin ({dupont['net_margin']:.2f}%)")
    print(f"   × Asset Turnover ({dupont['asset_turnover']:.2f})")
    print(f"   × Equity Multiplier ({dupont['equity_multiplier']:.2f})")

    print("\n5. Altman Z-Score:")
    market_cap = 3_000_000  # $3T
    z_score = fa.altman_z_score(
        working_capital=financial_data['current_assets'] - financial_data['current_liabilities'],
        retained_earnings=financial_data['shareholders_equity'] * 0.4,  # Estimate
        ebit=financial_data['operating_income'],
        market_cap=market_cap,
        total_liabilities=financial_data['total_debt'] + financial_data['current_liabilities'],
        revenue=financial_data['revenue'],
        total_assets=financial_data['total_assets']
    )
    print(f"   Z-Score: {z_score['z_score']:.2f}")
    print(f"   Status: {z_score['status']}")

    print("\n✓ Fundamental Analysis Complete")


def demo_valuation():
    """Demonstrate valuation models"""
    print("\n" + "="*80)
    print("VALUATION MODELS DEMONSTRATION")
    print("="*80)

    vm = ValuationModels(discount_rate=0.10)

    # DCF Valuation
    print("\n1. Discounted Cash Flow (DCF):")
    fcf_forecast = [10_000, 11_000, 12_100, 13_310, 14_641]  # Millions
    dcf = vm.dcf_valuation(fcf_forecast, terminal_growth_rate=0.03)
    print(f"   Enterprise Value: ${dcf['enterprise_value']:,.0f}M")
    print(f"   PV of Forecast: ${dcf['pv_forecast_period']:,.0f}M")
    print(f"   PV of Terminal: ${dcf['pv_terminal_value']:,.0f}M")

    # Dividend Discount Model
    print("\n2. Gordon Growth Model:")
    value = vm.gordon_growth_model(current_dividend=2.50, growth_rate=0.05)
    print(f"   Stock Value: ${value:.2f}")

    # Two-Stage DDM
    print("\n3. Two-Stage DDM:")
    two_stage = vm.multi_stage_ddm(
        current_dividend=2.50,
        high_growth_rate=0.15,
        high_growth_years=5,
        stable_growth_rate=0.04
    )
    print(f"   Total Value: ${two_stage['total_value']:.2f}")
    print(f"   Stage 1: ${two_stage['stage1_value']:.2f}")
    print(f"   Stage 2: ${two_stage['stage2_value']:.2f}")

    # Comparable Valuation
    print("\n4. Comparable Valuation (P/E):")
    peer_pe_multiples = [25.5, 28.3, 22.1, 30.2, 26.8]
    company_earnings = 5_000  # Millions
    comp_val = vm.comparable_valuation(company_earnings, peer_pe_multiples)
    print(f"   Implied Value: ${comp_val['implied_value']:,.0f}M")
    print(f"   Multiple Used: {comp_val['multiple_used']:.1f}x")
    print(f"   Peer Range: {comp_val['peer_multiples']['min']:.1f}x - {comp_val['peer_multiples']['max']:.1f}x")

    # WACC Calculation
    print("\n5. WACC Calculation:")
    wacc = vm.wacc(
        equity_value=200_000,
        debt_value=50_000,
        cost_of_equity=0.12,
        cost_of_debt=0.05,
        tax_rate=0.21
    )
    print(f"   WACC: {wacc:.2%}")

    # Black-Scholes Option Pricing
    print("\n6. Black-Scholes Option Pricing:")
    call_option = vm.black_scholes_call(
        stock_price=150,
        strike_price=155,
        time_to_expiry=0.25,  # 3 months
        risk_free_rate=0.04,
        volatility=0.30
    )
    print(f"   Call Price: ${call_option['call_price']:.2f}")
    print(f"   Delta: {call_option['delta']:.3f}")
    print(f"   Gamma: {call_option['gamma']:.4f}")
    print(f"   Vega: {call_option['vega']:.2f}")
    print(f"   Theta: {call_option['theta']:.2f}")

    print("\n✓ Valuation Models Complete")


def demo_risk_management():
    """Demonstrate risk management techniques"""
    print("\n" + "="*80)
    print("RISK MANAGEMENT DEMONSTRATION")
    print("="*80)

    # Generate sample return data
    print("\n1. Generating sample portfolio returns...")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    returns = pd.DataFrame({
        'Asset': np.random.normal(0.0005, 0.02, 252)
    }, index=dates)

    rm = RiskManagement(returns)

    print("\n2. Value at Risk (VaR) Comparison:")
    var_methods = rm.compare_var_methods(confidence_level=0.95)
    for method, value in var_methods.items():
        print(f"   {method.capitalize()}: {value:.2%}")

    print("\n3. Expected Shortfall (CVaR):")
    es = rm.expected_shortfall(0.95)
    print(f"   CVaR (95%): {es:.2%}")

    print("\n4. Volatility Measures:")
    realized_vol = rm.realized_volatility(window=30).iloc[-1]
    ewma_vol = rm.ewma_volatility().iloc[-1]
    print(f"   Realized Vol (30-day): {realized_vol:.2%}")
    print(f"   EWMA Vol: {ewma_vol:.2%}")

    print("\n5. Distribution Analysis:")
    jb_test = rm.jarque_bera_test()
    print(f"   Skewness: {jb_test['skewness']:.3f}")
    print(f"   Kurtosis: {jb_test['kurtosis']:.3f}")
    print(f"   Normal Distribution: {jb_test['is_normal']}")

    print("\n6. Downside Risk:")
    downside_dev = rm.downside_deviation(min_acceptable_return=0)
    print(f"   Downside Deviation: {downside_dev:.2%}")

    print("\n7. Risk Dashboard:")
    dashboard = rm.portfolio_risk_dashboard(np.array([1.0]))
    print(f"   VaR (95%): {dashboard['var_95']:.2%}")
    print(f"   VaR (99%): {dashboard['var_99']:.2%}")
    print(f"   Expected Shortfall: {dashboard['expected_shortfall']:.2%}")
    print(f"   Volatility: {dashboard['volatility']:.2%}")
    print(f"   Max Drawdown: {dashboard['max_drawdown']:.2%}")
    print(f"   Sharpe Ratio: {dashboard['sharpe_ratio']:.2f}")

    print("\n✓ Risk Management Complete")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("TRADITIONAL FINANCIAL ANALYSIS METHODS")
    print("Implementation and Demonstration")
    print("="*80)

    try:
        # Run demonstrations
        demo_technical_analysis()
        demo_portfolio_analysis()
        demo_fundamental_analysis()
        demo_valuation()
        demo_risk_management()

        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETE")
        print("="*80)
        print("\nTraditional finance methods successfully implemented:")
        print("  ✓ Technical Analysis (20+ indicators)")
        print("  ✓ Fundamental Analysis (30+ ratios)")
        print("  ✓ Portfolio Analysis (MPT, optimization)")
        print("  ✓ Risk Management (VaR, volatility)")
        print("  ✓ Valuation Models (DCF, DDM, options)")
        print("\nAll methods are ready to use with your financial data!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
