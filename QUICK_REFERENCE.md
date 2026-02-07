# Quick Reference Guide - Traditional Finance Methods

## 📖 Table of Contents
1. [Technical Analysis](#technical-analysis)
2. [Fundamental Analysis](#fundamental-analysis)
3. [Portfolio Analysis](#portfolio-analysis)
4. [Risk Management](#risk-management)
5. [Valuation Models](#valuation-models)

---

## Technical Analysis

### Basic Setup
```python
from analysis import TechnicalAnalysis
import pandas as pd

# Load your price data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

ta = TechnicalAnalysis(df)
```

### Moving Averages
```python
# Simple Moving Average
sma_20 = ta.moving_average(period=20, ma_type='SMA')
sma_50 = ta.moving_average(period=50, ma_type='SMA')

# Exponential Moving Average (reacts faster to price changes)
ema_12 = ta.moving_average(period=12, ma_type='EMA')

# Detect Golden Cross / Death Cross
crossover = ta.detect_crossover(fast_period=50, slow_period=200)
# Returns: 1 = Golden Cross (bullish), -1 = Death Cross (bearish)
```

### Momentum Indicators
```python
# RSI - Overbought (>70) / Oversold (<30)
rsi = ta.rsi(period=14)
current_rsi = rsi.iloc[-1]

if current_rsi > 70:
    print("Overbought - potential sell signal")
elif current_rsi < 30:
    print("Oversold - potential buy signal")

# MACD - Trend and momentum
macd = ta.macd(fast=12, slow=26, signal=9)
# macd['histogram'] > 0 = Bullish momentum

# Stochastic - %K and %D lines
stoch = ta.stochastic(k_period=14, d_period=3)
# Both above 80 = Overbought, both below 20 = Oversold
```

### Volatility Bands
```python
# Bollinger Bands
bb = ta.bollinger_bands(period=20, std_dev=2.0)
current_price = df['Close'].iloc[-1]

if current_price > bb['upper'].iloc[-1]:
    print("Above upper band - overbought")
elif current_price < bb['lower'].iloc[-1]:
    print("Below lower band - oversold")

# ATR - Measure volatility for position sizing
atr = ta.atr(period=14)
stop_loss = current_price - (2 * atr.iloc[-1])  # 2 ATR stop
```

### All-in-One
```python
# Get all indicators at once
all_data = ta.get_all_indicators()

# Get simple trading signals
signals = ta.get_signals()
# Returns: {'RSI': 'Oversold', 'MACD': 'Bullish', 'Bollinger': 'Normal', ...}
```

---

## Fundamental Analysis

### Basic Setup
```python
from analysis import FundamentalAnalysis

fa = FundamentalAnalysis()

# Your financial data (from financial statements)
financial_data = {
    'revenue': 100_000,
    'cogs': 60_000,
    'operating_income': 25_000,
    'net_income': 20_000,
    'total_assets': 150_000,
    'current_assets': 50_000,
    'shareholders_equity': 75_000,
    'total_debt': 40_000,
    'current_liabilities': 30_000,
}
```

### Profitability Analysis
```python
# Return on Equity - How much profit per dollar of equity?
roe = fa.roe(
    net_income=financial_data['net_income'],
    shareholders_equity=financial_data['shareholders_equity']
)
# Good: >15%, Excellent: >20%

# Profit Margins
gross_margin = fa.gross_margin(
    revenue=financial_data['revenue'],
    cogs=financial_data['cogs']
)
net_margin = fa.net_margin(
    net_income=financial_data['net_income'],
    revenue=financial_data['revenue']
)
```

### Valuation Multiples
```python
# P/E Ratio - Price per dollar of earnings
pe_ratio = fa.pe_ratio(price=150, earnings_per_share=10)
# Market average: ~15-20x

# EV/EBITDA - Enterprise value multiple
ev = fa.calculate_enterprise_value(
    market_cap=500_000,
    total_debt=40_000,
    cash=20_000
)
ev_ebitda = fa.ev_ebitda(enterprise_value=ev, ebitda=30_000)
```

### Financial Health
```python
# Liquidity - Can company pay short-term obligations?
current_ratio = fa.current_ratio(
    current_assets=financial_data['current_assets'],
    current_liabilities=financial_data['current_liabilities']
)
# Good: >1.5, Excellent: >2.0

# Leverage - How much debt?
debt_equity = fa.debt_to_equity(
    total_debt=financial_data['total_debt'],
    shareholders_equity=financial_data['shareholders_equity']
)
# Conservative: <0.5, Moderate: 0.5-1.0, High: >1.0
```

### Advanced Analysis
```python
# DuPont Analysis - Break down ROE
dupont = fa.dupont_analysis(
    net_income=20_000,
    revenue=100_000,
    total_assets=150_000,
    shareholders_equity=75_000
)
# Shows: ROE = Margin × Turnover × Leverage

# Altman Z-Score - Bankruptcy risk
z_score = fa.altman_z_score(
    working_capital=20_000,
    retained_earnings=30_000,
    ebit=25_000,
    market_cap=500_000,
    total_liabilities=75_000,
    revenue=100_000,
    total_assets=150_000
)
# Z > 2.99: Safe, 1.81-2.99: Gray zone, <1.81: Distress
```

---

## Portfolio Analysis

### Basic Setup
```python
from analysis import PortfolioAnalysis
import pandas as pd
import numpy as np

# Asset returns (daily, as decimal not %)
returns_df = pd.DataFrame({
    'AAPL': [0.01, -0.02, 0.015, ...],
    'MSFT': [0.005, 0.01, -0.005, ...],
    'GOOGL': [-0.01, 0.02, 0.01, ...]
})

pa = PortfolioAnalysis(returns_df, risk_free_rate=0.04)
```

### Portfolio Metrics
```python
# Equal weight portfolio
weights = np.array([0.33, 0.33, 0.34])

# Expected return (annualized)
annual_return = pa.portfolio_return(weights) * 252
print(f"Expected return: {annual_return:.2%}")

# Volatility (annualized)
annual_vol = pa.portfolio_volatility(weights) * np.sqrt(252)
print(f"Volatility: {annual_vol:.2%}")

# Sharpe Ratio (higher is better)
sharpe = pa.sharpe_ratio(weights)
print(f"Sharpe ratio: {sharpe:.2f}")
# Excellent: >2, Good: 1-2, Poor: <1
```

### Portfolio Optimization
```python
# Find portfolio with maximum Sharpe ratio
max_sharpe = pa.maximum_sharpe_portfolio()
print("Optimal Weights:")
for i, asset in enumerate(returns_df.columns):
    print(f"  {asset}: {max_sharpe['weights'][i]:.1%}")
print(f"Return: {max_sharpe['return']:.2%}")
print(f"Sharpe: {max_sharpe['sharpe']:.2f}")

# Find minimum risk portfolio
min_var = pa.minimum_variance_portfolio()

# Risk Parity (equal risk contribution)
risk_parity = pa.risk_parity_portfolio()

# All strategies at once
all_strategies = pa.get_optimal_portfolios()
```

### Risk Metrics
```python
# Maximum Drawdown - worst peak-to-trough loss
mdd = pa.maximum_drawdown(weights)
print(f"Max drawdown: {mdd['max_drawdown']:.2%}")

# Sortino Ratio - like Sharpe but only penalizes downside
sortino = pa.sortino_ratio(weights)

# Diversification benefit
div_ratio = pa.diversification_ratio(weights)
# >1 means diversification is working
```

### CAPM Analysis
```python
# Calculate beta for an asset
asset_returns = returns_df['AAPL']
market_returns = returns_df['SPY']  # S&P 500 proxy

beta = pa.calculate_beta(asset_returns, market_returns)
# Beta > 1: More volatile than market
# Beta < 1: Less volatile than market
# Beta = 1: Moves with market

# Calculate alpha (excess return)
alpha = pa.calculate_alpha(
    asset_return=0.15,  # 15% return
    asset_beta=1.2,
    market_return=0.10  # 10% market return
)
# Positive alpha = outperformance
```

---

## Risk Management

### Basic Setup
```python
from analysis import RiskManagement
import pandas as pd

# Portfolio returns
returns_df = pd.DataFrame({'Portfolio': [...]})
rm = RiskManagement(returns_df)
```

### Value at Risk (VaR)
```python
# Historical VaR - 95% confidence
var_95 = rm.historical_var(confidence_level=0.95)
print(f"VaR (95%): {var_95:.2%}")
# "We are 95% confident losses won't exceed {var_95}%"

# VaR at different confidence levels
var_99 = rm.historical_var(confidence_level=0.99)

# Compare all VaR methods
var_methods = rm.compare_var_methods(confidence_level=0.95)
# {'historical': -0.02, 'parametric': -0.021, 'monte_carlo': -0.019}

# Expected Shortfall (CVaR) - average loss beyond VaR
cvar = rm.expected_shortfall(confidence_level=0.95)
# More conservative than VaR
```

### Volatility Measurement
```python
# Realized volatility (30-day rolling)
vol_30d = rm.realized_volatility(window=30)
current_vol = vol_30d.iloc[-1]
print(f"30-day volatility: {current_vol:.2%}")

# EWMA volatility (gives more weight to recent data)
ewma_vol = rm.ewma_volatility(lambda_param=0.94)
# RiskMetrics uses λ=0.94 for daily data
```

### Distribution Analysis
```python
# Check if returns are normally distributed
jb_test = rm.jarque_bera_test()

if not jb_test['is_normal']:
    print("Returns are NOT normal - use Historical VaR")
    print(f"Skewness: {jb_test['skewness']:.2f}")
    print(f"Kurtosis: {jb_test['kurtosis']:.2f}")

    # Negative skew = more extreme losses than gains
    # High kurtosis = fat tails (more extreme events)
```

### Downside Risk
```python
# Downside deviation (used in Sortino ratio)
downside_dev = rm.downside_deviation(min_acceptable_return=0)

# Focus only on negative returns
semi_var = rm.semi_variance(target_return=0)
```

### Risk Dashboard
```python
# Get all risk metrics at once
portfolio_weights = np.array([1.0])  # Single asset or portfolio
dashboard = rm.portfolio_risk_dashboard(portfolio_weights)

print(f"VaR (95%): {dashboard['var_95']:.2%}")
print(f"VaR (99%): {dashboard['var_99']:.2%}")
print(f"Expected Shortfall: {dashboard['expected_shortfall']:.2%}")
print(f"Volatility: {dashboard['volatility']:.2%}")
print(f"Max Drawdown: {dashboard['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {dashboard['sharpe_ratio']:.2f}")
```

### Stress Testing
```python
# Historical stress test (e.g., 2008 crisis)
crisis_test = rm.historical_stress_test(
    crisis_period=('2008-09-01', '2008-12-31'),
    portfolio_weights=weights
)
print(f"Crisis return: {crisis_test['cumulative_return']:.2%}")
print(f"Worst day: {crisis_test['worst_day']:.2%}")
```

---

## Valuation Models

### Basic Setup
```python
from analysis import ValuationModels

vm = ValuationModels(discount_rate=0.10)  # 10% WACC
```

### Discounted Cash Flow (DCF)
```python
# Free cash flow forecast (next 5 years)
fcf_forecast = [
    10_000,  # Year 1
    11_000,  # Year 2
    12_100,  # Year 3
    13_310,  # Year 4
    14_641   # Year 5
]

dcf = vm.dcf_valuation(
    free_cash_flows=fcf_forecast,
    terminal_growth_rate=0.03,  # 3% perpetual growth
    discount_rate=0.10  # 10% WACC
)

print(f"Enterprise Value: ${dcf['enterprise_value']:,.0f}M")
print(f"PV of forecasts: ${dcf['pv_forecast_period']:,.0f}M")
print(f"PV of terminal: ${dcf['pv_terminal_value']:,.0f}M")

# Convert to equity value
equity_value = dcf['enterprise_value'] - net_debt
shares_outstanding = 100_000_000
price_per_share = equity_value / shares_outstanding
```

### Calculate WACC
```python
# Weighted Average Cost of Capital
wacc = vm.wacc(
    equity_value=200_000,    # Market cap
    debt_value=50_000,       # Total debt
    cost_of_equity=0.12,     # Required return (CAPM)
    cost_of_debt=0.05,       # Interest rate on debt
    tax_rate=0.21            # Corporate tax rate
)

# Cost of Equity using CAPM
cost_of_equity = vm.cost_of_equity_capm(
    risk_free_rate=0.04,     # 10-year Treasury
    beta=1.2,                # Stock's beta
    market_return=0.10       # Expected market return
)
```

### Dividend Discount Model (DDM)
```python
# Gordon Growth Model (for stable dividend payers)
value = vm.gordon_growth_model(
    current_dividend=2.50,   # Last annual dividend
    growth_rate=0.05,        # 5% perpetual growth
    required_return=0.10     # Required return
)

# Two-Stage DDM (for growing companies)
two_stage = vm.multi_stage_ddm(
    current_dividend=2.50,
    high_growth_rate=0.15,   # 15% growth for 5 years
    high_growth_years=5,
    stable_growth_rate=0.04, # Then 4% forever
    required_return=0.10
)
print(f"Fair value: ${two_stage['total_value']:.2f}")
```

### Comparable Valuation
```python
# Using P/E multiples from peer companies
peer_pe_ratios = [25.5, 28.3, 22.1, 30.2, 26.8]
company_earnings = 5_000  # Company's earnings (millions)

comp_val = vm.comparable_valuation(
    company_metric=company_earnings,
    peer_multiples=peer_pe_ratios,
    method='median'  # or 'mean', 'harmonic_mean'
)

print(f"Implied value: ${comp_val['implied_value']:,.0f}M")
print(f"Multiple used: {comp_val['multiple_used']:.1f}x")
print(f"Peer range: {comp_val['peer_multiples']['min']:.1f}x - {comp_val['peer_multiples']['max']:.1f}x")
```

### Black-Scholes Option Pricing
```python
# Call option valuation
call = vm.black_scholes_call(
    stock_price=150,         # Current stock price
    strike_price=155,        # Strike price
    time_to_expiry=0.25,     # 3 months (0.25 years)
    risk_free_rate=0.04,     # 4% risk-free rate
    volatility=0.30          # 30% annual volatility
)

print(f"Call price: ${call['call_price']:.2f}")
print(f"Delta: {call['delta']:.3f}")  # Hedge ratio
print(f"Gamma: {call['gamma']:.4f}")  # Delta sensitivity
print(f"Vega: {call['vega']:.2f}")    # Vol sensitivity
print(f"Theta: {call['theta']:.2f}")  # Time decay per day

# Put option
put = vm.black_scholes_put(...)
```

### Sensitivity Analysis
```python
# DCF sensitivity to WACC and growth
sensitivity = vm.dcf_sensitivity_analysis(
    base_case={'free_cash_flows': fcf_forecast},
    wacc_range=[0.08, 0.09, 0.10, 0.11, 0.12],
    growth_range=[0.02, 0.025, 0.03, 0.035, 0.04]
)
# Returns DataFrame showing EV for each combination

# Monte Carlo simulation for uncertainty
mc_results = vm.monte_carlo_valuation(
    base_fcf=10_000,
    num_simulations=10_000,
    revenue_growth_mean=0.05,
    revenue_growth_std=0.10
)
print(f"Mean value: ${mc_results['mean']:,.0f}M")
print(f"5th percentile: ${mc_results['percentile_5']:,.0f}M")
print(f"95th percentile: ${mc_results['percentile_95']:,.0f}M")
```

---

## 💡 Common Workflows

### Stock Screening
```python
# Find undervalued stocks with good fundamentals
stocks = ['AAPL', 'MSFT', 'GOOGL', ...]

for ticker in stocks:
    data = collect_data(ticker)
    fa = FundamentalAnalysis()

    # Screen criteria
    pe = fa.pe_ratio(price, eps)
    roe = fa.roe(ni, equity)
    debt_equity = fa.debt_to_equity(debt, equity)

    if pe < 20 and roe > 15 and debt_equity < 1.0:
        print(f"{ticker}: Good candidate")
```

### Portfolio Construction
```python
# 1. Collect data for candidates
# 2. Optimize portfolio
pa = PortfolioAnalysis(returns_df)
optimal = pa.maximum_sharpe_portfolio()

# 3. Check risk
rm = RiskManagement(returns_df)
var = rm.historical_var(0.95, optimal['weights'])

if var > -0.03:  # Max 3% daily loss
    print("Risk acceptable")
    implement_portfolio(optimal['weights'])
```

### Risk Monitoring
```python
# Daily risk check
rm = RiskManagement(returns_df)

risk_limits = {
    'max_var': -0.02,        # Max 2% daily VaR
    'max_volatility': 0.20,  # Max 20% annual vol
    'max_position': 0.30,    # Max 30% in single asset
}

compliance = rm.check_risk_limits(current_weights, risk_limits)

if not all(compliance.values()):
    print("RISK LIMIT BREACH!")
    rebalance_portfolio()
```

---

## 📊 Rule of Thumb Guidelines

### Technical Analysis
- **RSI**: >70 overbought, <30 oversold
- **MACD Histogram**: Positive = bullish momentum
- **Bollinger Bands**: Price at upper = overbought, lower = oversold

### Fundamental Analysis
- **ROE**: >15% good, >20% excellent
- **Current Ratio**: >1.5 healthy, >2.0 excellent
- **Debt/Equity**: <0.5 conservative, 0.5-1.0 moderate, >1.0 aggressive
- **P/E Ratio**: Market average ~15-20x

### Portfolio Analysis
- **Sharpe Ratio**: >2 excellent, 1-2 good, <1 poor
- **Diversification**: 15-30 stocks for individual portfolios

### Risk Management
- **VaR**: Typically 95% or 99% confidence
- **Max Drawdown**: Expect 20-30% in equity portfolios

### Valuation
- **DCF Terminal Growth**: Usually 2-3% (GDP growth)
- **WACC**: Typically 8-12% for equity
- **P/E Multiples**: Compare to industry average

---

This quick reference covers the most commonly used methods. For detailed documentation, see `analysis/README.md`.
