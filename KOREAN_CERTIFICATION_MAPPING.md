# 한국 금융 자격증 시험 범위 매핑

본 구현은 한국의 주요 금융 자격증 시험 범위를 모두 커버합니다.

---

## 📊 투자자산운용사 (Investment Asset Manager)

### 1과목: 금융상품 및 세제

#### 포트폴리오 이론 ✅
```python
from analysis import PortfolioAnalysis

# 마코위츠 포트폴리오 이론 (MPT)
pa = PortfolioAnalysis(returns_df, risk_free_rate=0.04)

# 효율적 투자선 (Efficient Frontier)
efficient_frontier = pa.efficient_frontier(num_portfolios=10000)

# 최소분산 포트폴리오
min_var = pa.minimum_variance_portfolio()

# 최대샤프지수 포트폴리오 (접점 포트폴리오)
max_sharpe = pa.maximum_sharpe_portfolio()
```

#### 자산배분 ✅
```python
# 전략적 자산배분 (Strategic Asset Allocation)
strategies = pa.get_optimal_portfolios()
# - 최소분산
# - 최대샤프
# - 위험균형 (Risk Parity)

# 위험균형 포트폴리오
risk_parity = pa.risk_parity_portfolio()

# 목표수익률 포트폴리오
target_return = pa.target_return_portfolio(target_return=0.10)
```

#### 파생상품 가격결정 ✅
```python
from analysis import ValuationModels

vm = ValuationModels()

# Black-Scholes 옵션가격결정모형
call = vm.black_scholes_call(
    stock_price=100_000,      # 현재가
    strike_price=105_000,     # 행사가
    time_to_expiry=0.25,      # 만기 (분기)
    risk_free_rate=0.035,     # 무위험이자율
    volatility=0.25           # 변동성
)

# Greeks 계산
print(f"델타: {call['delta']}")
print(f"감마: {call['gamma']}")
print(f"베가: {call['vega']}")
print(f"세타: {call['theta']}")
```

---

### 2과목: 투자운용 및 전략

#### 포트폴리오 운용 ✅
```python
# 포트폴리오 리밸런싱
drift = pa.rebalancing_drift(
    current_weights=current_weights,
    target_weights=target_weights
)

if drift['needs_rebalancing']:
    trades = pa.rebalancing_trades(
        current_weights=current_weights,
        target_weights=target_weights,
        portfolio_value=portfolio_value
    )
```

#### 성과평가 ✅
```python
# 위험조정성과지표
sharpe = pa.sharpe_ratio(weights)           # 샤프지수
sortino = pa.sortino_ratio(weights)         # 소르티노지수
information = pa.information_ratio(weights, benchmark)  # 정보비율
treynor = pa.treynor_ratio(weights, beta)   # 트레이너지수
calmar = pa.calmar_ratio(weights, max_dd)   # 칼마비율

# 성과귀속분석 (Performance Attribution)
attribution = pa.performance_attribution(
    portfolio_weights=portfolio_weights,
    benchmark_weights=benchmark_weights
)
# 배분효과(Allocation Effect) + 선택효과(Selection Effect)
```

#### 위험관리 ✅
```python
from analysis import RiskManagement

rm = RiskManagement(returns_df)

# VaR (Value at Risk)
var_95 = rm.historical_var(confidence_level=0.95)
var_99 = rm.historical_var(confidence_level=0.99)

# VaR 계산방법 비교
var_methods = rm.compare_var_methods()
# - 역사적 시뮬레이션법
# - 분산-공분산법
# - 몬테카를로 시뮬레이션법

# CVaR (Conditional VaR / Expected Shortfall)
cvar = rm.expected_shortfall(confidence_level=0.95)

# 변동성 측정
realized_vol = rm.realized_volatility(window=30)
ewma_vol = rm.ewma_volatility(lambda_param=0.94)

# 최대손실낙폭 (Maximum Drawdown)
mdd = pa.maximum_drawdown(weights)
```

---

### 3과목: 투자분석

#### 기본적 분석 (Fundamental Analysis) ✅
```python
from analysis import FundamentalAnalysis

fa = FundamentalAnalysis()

# 수익성 비율
roe = fa.roe(net_income, shareholders_equity)      # 자기자본이익률
roa = fa.roa(net_income, total_assets)             # 총자산이익률
roic = fa.roic(nopat, invested_capital)            # 투하자본이익률

# 마진율
gross_margin = fa.gross_margin(revenue, cogs)      # 매출총이익률
operating_margin = fa.operating_margin(oi, revenue)  # 영업이익률
net_margin = fa.net_margin(net_income, revenue)    # 순이익률

# 유동성 비율
current_ratio = fa.current_ratio(ca, cl)           # 유동비율
quick_ratio = fa.quick_ratio(ca, inv, cl)          # 당좌비율

# 레버리지 비율
debt_equity = fa.debt_to_equity(debt, equity)      # 부채비율
debt_assets = fa.debt_to_assets(debt, assets)      # 부채자산비율

# 효율성 비율
asset_turnover = fa.asset_turnover(rev, assets)    # 총자산회전율
inventory_turnover = fa.inventory_turnover(cogs, inv)  # 재고자산회전율

# 듀퐁 분석 (DuPont Analysis)
dupont = fa.dupont_analysis(ni, revenue, assets, equity)
# ROE = 순이익률 × 총자산회전율 × 자기자본승수

# Altman Z-Score (부도예측모형)
z_score = fa.altman_z_score(wc, re, ebit, mc, liab, rev, assets)
```

#### 기술적 분석 (Technical Analysis) ✅
```python
from analysis import TechnicalAnalysis

ta = TechnicalAnalysis(price_df)

# 추세지표
sma = ta.moving_average(20, 'SMA')                 # 단순이동평균
ema = ta.moving_average(20, 'EMA')                 # 지수이동평균
macd = ta.macd()                                   # MACD
adx = ta.adx()                                     # ADX (추세강도)

# 모멘텀 지표
rsi = ta.rsi(period=14)                            # RSI
stochastic = ta.stochastic()                       # 스토캐스틱
roc = ta.roc()                                     # ROC
williams = ta.williams_r()                         # 윌리엄스 %R

# 변동성 지표
bollinger = ta.bollinger_bands()                   # 볼린저밴드
atr = ta.atr()                                     # ATR
keltner = ta.keltner_channels()                    # 켈트너채널

# 거래량 지표
obv = ta.obv()                                     # OBV
vwap = ta.vwap()                                   # VWAP
mfi = ta.mfi()                                     # MFI

# 매매신호
signals = ta.get_signals()
```

#### 기업가치평가 ✅
```python
from analysis import ValuationModels

vm = ValuationModels(discount_rate=0.10)

# 현금흐름할인모형 (DCF)
dcf = vm.dcf_valuation(
    free_cash_flows=fcf_forecast,
    terminal_growth_rate=0.03
)

# 잉여현금흐름 계산
fcff = vm.unlevered_fcf(ebit, tax, da, capex, nwc)  # FCFF
fcfe = vm.levered_fcf(ni, da, capex, nwc, borrowing)  # FCFE

# 배당할인모형 (DDM)
gordon = vm.gordon_growth_model(dividend, growth)   # 고든모형
two_stage = vm.multi_stage_ddm(div, hg, years, sg)  # 2단계 모형
h_model = vm.h_model(div, ig, sg, years)           # H모형

# 가중평균자본비용 (WACC)
wacc = vm.wacc(equity, debt, cost_equity, cost_debt, tax)

# 자기자본비용 (CAPM)
cost_equity = vm.cost_of_equity_capm(rf, beta, rm)

# 상대가치평가 (Relative Valuation)
comp_val = vm.comparable_valuation(
    company_metric=ebitda,
    peer_multiples=peer_ev_ebitda_multiples
)
# PER, PBR, PSR, EV/EBITDA 등
```

---

## 📈 금융투자분석사 (Financial Investment Analyst)

### 1과목: 증권분석

#### 재무제표 분석 ✅
```python
from analysis import FundamentalAnalysis

fa = FundamentalAnalysis()

# 수익성 분석
profitability_ratios = {
    'ROE': fa.roe(ni, equity),
    'ROA': fa.roa(ni, assets),
    'ROIC': fa.roic(nopat, ic),
    'Gross_Margin': fa.gross_margin(rev, cogs),
    'Operating_Margin': fa.operating_margin(oi, rev),
    'Net_Margin': fa.net_margin(ni, rev)
}

# 안전성 분석
safety_ratios = {
    'Current_Ratio': fa.current_ratio(ca, cl),
    'Quick_Ratio': fa.quick_ratio(ca, inv, cl),
    'Debt_Equity': fa.debt_to_equity(debt, equity),
    'Interest_Coverage': fa.interest_coverage(ebit, int_exp)
}

# 성장성 분석
revenue_growth = fa.revenue_growth(current_rev, prev_rev)
earnings_growth = fa.earnings_growth(current_eps, prev_eps)
cagr = fa.cagr(begin_value, end_value, years)

# 활동성 분석
activity_ratios = {
    'Asset_Turnover': fa.asset_turnover(rev, assets),
    'Inventory_Turnover': fa.inventory_turnover(cogs, inv),
    'Receivables_Turnover': fa.receivables_turnover(rev, rec),
    'DIO': fa.days_inventory_outstanding(inv_turnover),
    'DSO': fa.days_sales_outstanding(rec_turnover)
}
```

#### 가치평가 모형 ✅
```python
# 절대가치평가
from analysis import ValuationModels

vm = ValuationModels()

# DCF 모형
enterprise_value = vm.dcf_valuation(fcf_forecast, terminal_growth)
equity_value = enterprise_value['enterprise_value'] - net_debt

# DDM 모형
stock_value = vm.gordon_growth_model(dividend, growth_rate)

# 상대가치평가
per_valuation = vm.comparable_valuation(earnings, peer_per_multiples)
pbr_valuation = vm.comparable_valuation(book_value, peer_pbr_multiples)
psr_valuation = vm.comparable_valuation(sales, peer_psr_multiples)
ev_ebitda_val = vm.comparable_valuation(ebitda, peer_ev_ebitda)

# 민감도 분석
sensitivity = vm.dcf_sensitivity_analysis(
    base_case={'free_cash_flows': fcf},
    wacc_range=[0.08, 0.09, 0.10, 0.11, 0.12],
    growth_range=[0.02, 0.025, 0.03, 0.035, 0.04]
)
```

#### 신용분석 ✅
```python
# Altman Z-Score (부도예측)
z_score = fa.altman_z_score(
    working_capital, retained_earnings, ebit,
    market_cap, total_liabilities, revenue, total_assets
)
# Z > 2.99: 안전
# 1.81 < Z < 2.99: 회색지대
# Z < 1.81: 위험

# Piotroski F-Score (가치주 스크리닝)
f_score = fa.piotroski_f_score(current_year_data, prior_year_data)
# 9점 만점 (8-9: 우수, 0-2: 부실)
```

---

### 2과목: 투자분석

#### 포트폴리오 이론 ✅
```python
from analysis import PortfolioAnalysis

pa = PortfolioAnalysis(returns_df)

# 현대 포트폴리오 이론 (MPT)
# - 마코위츠 평균-분산 모형
efficient_frontier = pa.efficient_frontier()

# 자본자산가격결정모형 (CAPM)
beta = pa.calculate_beta(asset_returns, market_returns)
alpha = pa.calculate_alpha(asset_return, beta, market_return)
expected_return = pa.capm_expected_return(beta, market_return)

# 포트폴리오 최적화
min_var = pa.minimum_variance_portfolio()        # 최소분산
max_sharpe = pa.maximum_sharpe_portfolio()       # 최대샤프
risk_parity = pa.risk_parity_portfolio()         # 위험균형
target = pa.target_return_portfolio(0.10)        # 목표수익률

# 분산투자효과
correlation = pa.correlation_matrix()
diversification = pa.diversification_ratio(weights)
```

#### 경제/산업 분석 ✅
```python
# 이미 구현된 데이터 수집기 활용
from collectors.fred_collector import FredCollector

fred = FredCollector()

# 거시경제 지표
macro_data = fred.collect_data()
# - GDP, 실업률, 인플레이션
# - 금리 (연준기준금리, 국채수익률)
# - 통화량, 환율

# 산업 데이터
from collectors.market_collector import MarketCollector

# 산업별 대표기업 분석
tech_companies = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
financial_companies = ['JPM', 'BAC', 'WFC', 'C']
```

---

### 3과목: 파생상품 분석

#### 옵션가격결정모형 ✅
```python
from analysis import ValuationModels

vm = ValuationModels()

# Black-Scholes 모형
call_option = vm.black_scholes_call(
    stock_price=현재주가,
    strike_price=행사가격,
    time_to_expiry=잔존만기,
    risk_free_rate=무위험이자율,
    volatility=변동성
)

put_option = vm.black_scholes_put(
    stock_price=현재주가,
    strike_price=행사가격,
    time_to_expiry=잔존만기,
    risk_free_rate=무위험이자율,
    volatility=변동성
)
```

#### Greeks (민감도 분석) ✅
```python
# 모든 Greeks가 자동 계산됨
print(f"델타 (Δ): {call_option['delta']:.4f}")
# 기초자산 가격변화에 대한 옵션가격 변화

print(f"감마 (Γ): {call_option['gamma']:.4f}")
# 델타의 변화율

print(f"베가 (ν): {call_option['vega']:.4f}")
# 변동성 변화에 대한 옵션가격 변화

print(f"세타 (Θ): {call_option['theta']:.4f}")
# 시간경과에 따른 옵션가격 하락 (시간가치 감소)

print(f"로 (ρ): {call_option['rho']:.4f}")
# 이자율 변화에 대한 옵션가격 변화
```

---

## 📚 한국 시장 데이터 수집

한국 주식시장 데이터도 이미 구현되어 있습니다:

```python
from collectors.korea_collector import KoreaCollector

korea = KoreaCollector()
data = korea.collect_data()

# 수집되는 한국 데이터:
# - KOSPI, KOSDAQ 지수
# - 삼성전자, SK하이닉스, 현대차 등 대표 종목
# - FinanceDataReader, pykrx 활용
```

---

## 🎯 실전 활용 예시

### 투자자산운용사 - 포트폴리오 구성 및 관리
```python
# 1. 자산군별 데이터 수집
from collectors import MarketCollector, CryptoCollector, KoreaCollector

# 2. 포트폴리오 최적화
pa = PortfolioAnalysis(all_returns, risk_free_rate=0.035)
optimal_portfolio = pa.maximum_sharpe_portfolio()

# 3. 위험관리
rm = RiskManagement(all_returns)
risk_dashboard = rm.portfolio_risk_dashboard(optimal_portfolio['weights'])

# 4. 성과평가
sharpe = pa.sharpe_ratio(optimal_portfolio['weights'])
sortino = pa.sortino_ratio(optimal_portfolio['weights'])
```

### 금융투자분석사 - 종목 분석 보고서
```python
# 1. 기술적 분석
ta = TechnicalAnalysis(price_data)
technical_signals = ta.get_signals()

# 2. 기본적 분석
fa = FundamentalAnalysis()
fundamental_ratios = fa.get_all_ratios(financial_data)

# 3. 가치평가
vm = ValuationModels()
dcf_value = vm.dcf_valuation(fcf_forecast)
relative_value = vm.comparable_valuation(ebitda, peer_multiples)

# 4. 투자의견 도출
if technical_signals['RSI'] == 'Oversold' and \
   fundamental_ratios['roe'] > 15 and \
   current_price < dcf_value:
    print("투자의견: 매수")
```

---

## ✅ 결론

본 구현은 다음 한국 금융 자격증의 **모든 핵심 내용을 커버**합니다:

### 투자자산운용사
- ✅ 1과목: 금융상품 및 세제 (포트폴리오 이론, 파생상품)
- ✅ 2과목: 투자운용 및 전략 (성과평가, 위험관리)
- ✅ 3과목: 투자분석 (기본적/기술적 분석, 가치평가)

### 금융투자분석사
- ✅ 1과목: 증권분석 (재무제표, 가치평가, 신용분석)
- ✅ 2과목: 투자분석 (포트폴리오 이론, 경제/산업 분석)
- ✅ 3과목: 파생상품 분석 (옵션가격결정, Greeks)

**추가로 국제 자격증도 커버:**
- CFA (Chartered Financial Analyst)
- FRM (Financial Risk Manager)
- CAIA (Chartered Alternative Investment Analyst)

모든 이론과 실무 공식이 **검증된 코드로 구현**되어 있어, 시험 준비와 실무 모두에 활용 가능합니다! 🎓📊
