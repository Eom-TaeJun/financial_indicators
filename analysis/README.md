># Traditional Financial Analysis Methods

This module implements traditional finance methods used in investment management, portfolio analysis, and risk management.

## 📚 Overview

The analysis module provides five comprehensive categories of traditional financial analysis:

1. **Technical Analysis** - Price-based indicators and patterns
2. **Fundamental Analysis** - Financial statement analysis and ratios
3. **Portfolio Analysis** - Modern Portfolio Theory and optimization
4. **Risk Management** - VaR, volatility, and risk metrics
5. **Valuation Models** - DCF, DDM, and option pricing

---

## 🔧 Modules

### 1. Technical Analysis (`technical_indicators.py`)

Implements 20+ technical indicators used by traders and analysts.

#### Trend Indicators
- **Moving Averages** (SMA, EMA)
- **MACD** (Moving Average Convergence Divergence)
- **ADX** (Average Directional Index) - Trend strength

#### Momentum Indicators
- **RSI** (Relative Strength Index) - Overbought/oversold
- **Stochastic Oscillator**
- **ROC** (Rate of Change)
- **Williams %R**

#### Volatility Indicators
- **Bollinger Bands**
- **ATR** (Average True Range)
- **Keltner Channels**

#### Volume Indicators
- **OBV** (On-Balance Volume)
- **VWAP** (Volume Weighted Average Price)
- **MFI** (Money Flow Index)

#### Pattern Recognition
- **Golden Cross / Death Cross** detection
- **Support/Resistance** (Pivot Points)

**Example Usage:**
```python
from analysis import TechnicalAnalysis
import pandas as pd

# Load price data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

ta = TechnicalAnalysis(df)

# Calculate indicators
rsi = ta.rsi(period=14)
macd = ta.macd()
bollinger = ta.bollinger_bands()

# Get trading signals
signals = ta.get_signals()
print(signals)  # {'RSI': 'Oversold', 'MACD': 'Bullish', ...}
```

---

### 2. Fundamental Analysis (`fundamental_analysis.py`)

Implements 30+ financial ratios and metrics from financial statements.

#### Profitability Ratios
- **ROE** (Return on Equity)
- **ROA** (Return on Assets)
- **ROIC** (Return on Invested Capital)
- **Gross, Operating, Net Margins**

#### Valuation Ratios
- **P/E Ratio** (Price-to-Earnings)
- **P/B Ratio** (Price-to-Book)
- **P/S Ratio** (Price-to-Sales)
- **PEG Ratio** (P/E to Growth)
- **EV/EBITDA**

#### Liquidity Ratios
- **Current Ratio**
- **Quick Ratio** (Acid-Test)
- **Cash Ratio**

#### Leverage Ratios
- **Debt-to-Equity**
- **Debt-to-Assets**
- **Interest Coverage**

#### Efficiency Ratios
- **Asset Turnover**
- **Inventory Turnover**
- **Days Sales Outstanding (DSO)**

#### Advanced Analysis
- **DuPont Analysis** (3-factor ROE decomposition)
- **Altman Z-Score** (Bankruptcy prediction)
- **Piotroski F-Score** (Value stock screening)

**Example Usage:**
```python
from analysis import FundamentalAnalysis

fa = FundamentalAnalysis()

# Calculate profitability
roe = fa.roe(net_income=10_000, shareholders_equity=50_000)
roa = fa.roa(net_income=10_000, total_assets=100_000)

# DuPont Analysis
dupont = fa.dupont_analysis(
    net_income=10_000,
    revenue=100_000,
    total_assets=100_000,
    shareholders_equity=50_000
)

# Altman Z-Score (bankruptcy risk)
z_score = fa.altman_z_score(
    working_capital=20_000,
    retained_earnings=15_000,
    ebit=12_000,
    market_cap=150_000,
    total_liabilities=40_000,
    revenue=100_000,
    total_assets=100_000
)
```

---

### 3. Portfolio Analysis (`portfolio_analysis.py`)

Modern Portfolio Theory (MPT) and portfolio optimization techniques.

#### Portfolio Metrics
- **Portfolio Return** (expected return)
- **Portfolio Volatility** (standard deviation)
- **Portfolio Variance**

#### Performance Ratios
- **Sharpe Ratio** (risk-adjusted return)
- **Sortino Ratio** (downside risk-adjusted)
- **Information Ratio** (active management)
- **Treynor Ratio** (beta-adjusted)
- **Calmar Ratio** (drawdown-adjusted)

#### Modern Portfolio Theory
- **Efficient Frontier** (Monte Carlo simulation)
- **Minimum Variance Portfolio**
- **Maximum Sharpe Portfolio** (Tangency Portfolio)
- **Risk Parity Portfolio** (Equal Risk Contribution)
- **Target Return Portfolio**

#### CAPM Analysis
- **Beta Calculation** (systematic risk)
- **Alpha Calculation** (Jensen's Alpha)
- **Expected Return** (CAPM)

#### Portfolio Management
- **Rebalancing Analysis**
- **Performance Attribution** (allocation vs selection)
- **Correlation Analysis**
- **Diversification Ratio**

**Example Usage:**
```python
from analysis import PortfolioAnalysis
import pandas as pd
import numpy as np

# Create returns DataFrame
returns_df = pd.DataFrame({
    'AAPL': [...],
    'MSFT': [...],
    'GOOGL': [...]
})

pa = PortfolioAnalysis(returns_df, risk_free_rate=0.04)

# Equal weight portfolio
weights = np.array([0.33, 0.33, 0.34])
sharpe = pa.sharpe_ratio(weights)

# Optimize portfolio
max_sharpe = pa.maximum_sharpe_portfolio()
print(f"Optimal weights: {max_sharpe['weights']}")
print(f"Expected return: {max_sharpe['return']:.2%}")
print(f"Sharpe ratio: {max_sharpe['sharpe']:.2f}")

# Get all optimal strategies
strategies = pa.get_optimal_portfolios()
```

---

### 4. Risk Management (`risk_management.py`)

Comprehensive risk measurement and management techniques.

#### Value at Risk (VaR)
- **Historical VaR** (non-parametric)
- **Parametric VaR** (variance-covariance)
- **Monte Carlo VaR** (simulation-based)
- **Expected Shortfall / CVaR** (tail risk)

#### Volatility Measures
- **Realized Volatility** (historical)
- **EWMA Volatility** (exponentially weighted)
- **GARCH Volatility** (time-series model)

#### Downside Risk
- **Downside Deviation** (semi-deviation)
- **Semi-Variance**

#### Tail Risk
- **Skewness** (distribution asymmetry)
- **Kurtosis** (fat tails)
- **Jarque-Bera Test** (normality test)

#### Correlation Risk
- **Correlation Breakdown** (identify high correlations)
- **Rolling Correlation** (time-varying)

#### Drawdown Analysis
- **Maximum Drawdown**
- **Drawdown Duration**
- **Recovery Analysis**

#### Stress Testing
- **Scenario Analysis**
- **Historical Stress Tests**
- **Risk Limits** (compliance checking)

**Example Usage:**
```python
from analysis import RiskManagement
import pandas as pd

returns_df = pd.DataFrame({'Asset': [...]})
rm = RiskManagement(returns_df)

# Value at Risk
var_95 = rm.historical_var(confidence_level=0.95)
var_99 = rm.historical_var(confidence_level=0.99)

# Compare VaR methods
var_comparison = rm.compare_var_methods()

# Expected Shortfall (CVaR)
cvar = rm.expected_shortfall(confidence_level=0.95)

# Volatility
vol = rm.realized_volatility(window=30)
ewma_vol = rm.ewma_volatility(lambda_param=0.94)

# Tail risk
skew = rm.skewness()
kurt = rm.kurtosis()

# Risk dashboard
dashboard = rm.portfolio_risk_dashboard(weights)
```

---

### 5. Valuation Models (`valuation.py`)

Traditional corporate and equity valuation methods.

#### Discounted Cash Flow (DCF)
- **DCF Valuation** (enterprise value)
- **Free Cash Flow to Firm** (FCFF)
- **Free Cash Flow to Equity** (FCFE)

#### Dividend Discount Models
- **Gordon Growth Model** (constant growth DDM)
- **Two-Stage DDM** (high growth + stable)
- **H-Model** (declining growth)

#### Relative Valuation
- **Comparable Company Analysis** (multiples)
- **Precedent Transaction Analysis**

#### Cost of Capital
- **WACC** (Weighted Average Cost of Capital)
- **Cost of Equity** (CAPM)
- **Cost of Equity** (DDM method)

#### Asset-Based Valuation
- **Book Value / NAV**
- **Liquidation Value**
- **Sum-of-the-Parts** (SOTP)

#### Option Pricing
- **Black-Scholes Call Option**
- **Black-Scholes Put Option**
- **Option Greeks** (Delta, Gamma, Vega, Theta, Rho)

#### Sensitivity Analysis
- **DCF Sensitivity** (WACC vs Growth)
- **Monte Carlo Valuation** (probabilistic)

**Example Usage:**
```python
from analysis import ValuationModels

vm = ValuationModels(discount_rate=0.10)

# DCF Valuation
fcf_forecast = [10_000, 11_000, 12_100, 13_310, 14_641]
dcf = vm.dcf_valuation(
    free_cash_flows=fcf_forecast,
    terminal_growth_rate=0.03
)
print(f"Enterprise Value: ${dcf['enterprise_value']:,.0f}")

# Dividend Discount Model
value = vm.gordon_growth_model(
    current_dividend=2.50,
    growth_rate=0.05
)

# Two-Stage DDM
two_stage = vm.multi_stage_ddm(
    current_dividend=2.50,
    high_growth_rate=0.15,
    high_growth_years=5,
    stable_growth_rate=0.04
)

# Comparable Valuation
peer_multiples = [25.5, 28.3, 22.1, 30.2, 26.8]
comp_val = vm.comparable_valuation(
    company_metric=5_000,
    peer_multiples=peer_multiples
)

# WACC
wacc = vm.wacc(
    equity_value=200_000,
    debt_value=50_000,
    cost_of_equity=0.12,
    cost_of_debt=0.05,
    tax_rate=0.21
)

# Black-Scholes Option Pricing
call = vm.black_scholes_call(
    stock_price=150,
    strike_price=155,
    time_to_expiry=0.25,
    risk_free_rate=0.04,
    volatility=0.30
)
```

---

## 🚀 Quick Start

### Installation

Ensure required packages are installed:

```bash
pip install pandas numpy scipy
```

### Running the Demo

```bash
cd /home/tj/projects/autoai/financial_indicators
python analysis_demo.py
```

This will demonstrate:
- Technical analysis on real stock data
- Portfolio optimization with multiple assets
- Fundamental ratio calculations
- Valuation model examples
- Risk management techniques

---

## 📊 Integration with Data Collection

All analysis methods work seamlessly with data from the collectors:

```python
from collectors.market_collector import MarketCollector
from analysis import TechnicalAnalysis, PortfolioAnalysis

# Collect data
collector = MarketCollector(lookback_days=100)
data, status = collector.fetch_ticker('AAPL', 'Apple')

# Analyze
ta = TechnicalAnalysis(data)
rsi = ta.rsi()
macd = ta.macd()
signals = ta.get_signals()
```

---

## 📖 Traditional Finance Concepts

### Key Financial Concepts Implemented

1. **Efficient Market Hypothesis (EMH)** - Foundation for many models
2. **Modern Portfolio Theory (MPT)** - Markowitz optimization
3. **Capital Asset Pricing Model (CAPM)** - Risk-return relationship
4. **Random Walk Theory** - Price movement assumptions
5. **Risk-Return Tradeoff** - Core principle throughout
6. **Time Value of Money** - Discounting in DCF
7. **Option Pricing Theory** - Black-Scholes model
8. **Value Investing** - Graham & Dodd principles (ratios, multiples)
9. **Technical Analysis** - Dow Theory, trend following
10. **Risk Management** - VaR, diversification

### Financial Ratios by Category

**Profitability (9 metrics)**
- ROE, ROA, ROIC, Gross/Operating/Net Margins, etc.

**Valuation (6 metrics)**
- P/E, P/B, P/S, PEG, EV/EBITDA, Enterprise Value

**Liquidity (3 metrics)**
- Current, Quick, Cash Ratios

**Leverage (4 metrics)**
- Debt/Equity, Debt/Assets, Equity Multiplier, Interest Coverage

**Efficiency (5 metrics)**
- Asset Turnover, Inventory Turnover, DIO, DSO, Receivables Turnover

**Total: 27+ fundamental ratios**

### Portfolio Metrics

- 5 Performance Ratios (Sharpe, Sortino, Information, Treynor, Calmar)
- 4 Optimization Strategies (Min Variance, Max Sharpe, Risk Parity, Target Return)
- 3 Risk Metrics (VaR, CVaR, Max Drawdown)
- CAPM Analysis (Alpha, Beta, Expected Return)

### Risk Metrics

- 3 VaR Methods (Historical, Parametric, Monte Carlo)
- 3 Volatility Measures (Realized, EWMA, GARCH)
- 4 Tail Risk Metrics (Skewness, Kurtosis, ES, JB Test)
- Drawdown Analysis (Max DD, Duration)

### Valuation Methods

- 3 DCF Approaches (Standard, FCFF, FCFE)
- 3 DDM Models (Gordon, Two-Stage, H-Model)
- 2 Relative Methods (Comps, Transactions)
- 2 Option Models (Call, Put with Greeks)

---

## 📚 References

### Books
- "Security Analysis" - Benjamin Graham & David Dodd
- "The Intelligent Investor" - Benjamin Graham
- "A Random Walk Down Wall Street" - Burton Malkiel
- "Options, Futures, and Other Derivatives" - John Hull
- "Investment Valuation" - Aswath Damodaran
- "Active Portfolio Management" - Grinold & Kahn

### Academic Papers
- Markowitz (1952) - Portfolio Selection
- Sharpe (1964) - Capital Asset Pricing Model
- Black & Scholes (1973) - Option Pricing
- Fama & French (1993) - Three-Factor Model

### Industry Standards
- CFA Institute - CFA Curriculum
- GARP - FRM (Financial Risk Manager)
- RiskMetrics - VaR methodology

---

## 🎯 Use Cases

### For Traders
- Technical indicators for entry/exit signals
- Momentum and trend analysis
- Support/resistance levels

### For Portfolio Managers
- Portfolio optimization
- Risk-adjusted performance measurement
- Rebalancing analysis
- Performance attribution

### For Analysts
- Company valuation (DCF, DDM)
- Comparable analysis
- Financial health assessment (ratios)
- Credit risk (Z-Score)

### For Risk Managers
- VaR and CVaR calculation
- Stress testing
- Risk limit monitoring
- Tail risk assessment

---

## 💡 Tips

1. **Technical Analysis**: Use multiple indicators for confirmation
2. **Fundamental Analysis**: Compare ratios to industry averages
3. **Portfolio Optimization**: Consider transaction costs in real implementation
4. **Risk Management**: Use multiple VaR methods for robustness
5. **Valuation**: Perform sensitivity analysis on key assumptions

---

## ⚠️ Limitations

- **Technical Analysis**: Past performance doesn't guarantee future results
- **Parametric VaR**: Assumes normal distribution (may underestimate tail risk)
- **DCF**: Highly sensitive to assumptions (WACC, growth rate)
- **CAPM**: Single-factor model (consider Fama-French for robustness)
- **Black-Scholes**: Assumes constant volatility (consider stochastic models for accuracy)

---

## 🔮 Future Enhancements

- Machine Learning integration (prediction models)
- Factor models (Fama-French, Carhart)
- Advanced options (Exotic options, Greeks surface)
- Credit risk models (Merton, KMV)
- Alternative data integration
- Real-time monitoring dashboard

---

## 📝 License

Part of the AutoAI Financial Indicators project.

---

**Author**: Financial Analysis Module
**Version**: 1.0
**Last Updated**: 2025
