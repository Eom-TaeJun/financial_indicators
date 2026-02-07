# Financial Indicators Collection & Analysis System

금융 시장의 모든 주요 지표를 수집하고 **전통적인 금융 분석 기법**으로 분석하는 종합 시스템입니다.

## 🎯 주요 기능

1. **다중 소스 데이터 수집** - FRED, Alpha Vantage, CoinGecko, FinanceDataReader 등
2. **전통적 금융 분석** - 기술적/기본적 분석, 포트폴리오 최적화, 리스크 관리, 가치평가
3. **한국 금융 자격증 대응** - 투자자산운용사, 금융투자분석사 시험 범위 커버

## 📊 수집 가능한 지표

### 1. 거시경제 지표 (FRED API)
- **금리**: Fed Funds Rate, 2Y/10Y/30Y Treasury
- **스프레드**: 10Y-2Y, HY OAS, IG OAS
- **인플레이션**: CPI, Core CPI, PCE, Breakeven
- **고용**: 실업률, 비농업 고용, 초기 실업수당 청구
- **경제활동**: GDP, 산업생산, 소매판매
- **유동성**: RRP, TGA, Fed Balance Sheet, Reserves

### 2. 시장 데이터 (yfinance)
- **미국 주요 지수**: SPY, QQQ, DIA, IWM
- **섹터 ETF**: XLF, XLK, XLE, XLV 등
- **채권**: TLT, IEF, SHY, LQD, HYG
- **원자재**: GLD, SLV, USO, DBA
- **국제**: EFA, EEM, FXI

### 3. 암호화폐 & RWA
- **암호화폐**: BTC, ETH, SOL
- **스테이블코인**: USDC, USDT, DAI
- **RWA 토큰**: ONDO, PAXG
- **관련 주식**: COIN, MSTR

### 4. 한국 시장
- **지수**: KOSPI, KOSPI200, KOSDAQ
- **섹터 ETF**: 반도체, 2차전지, 바이오, 은행
- **대형주**: 삼성전자, SK하이닉스, LG에너지솔루션
- **채권**: 국고채 3년/10년
- **환율**: USD/KRW

## 🚀 Quick Start

### 1. 설치

```bash
cd projects/financial_indicators
pip install -r requirements.txt
```

**권장 라이브러리 (더 나은 데이터 품질):**
```bash
# 한국 시장 (무료, 추천 ⭐)
pip install finance-datareader pykrx

# 모두 설치되면 자동으로 multi-source fallback 작동
```

### 2. API 키 설정

```bash
# .env 파일 생성
echo "FRED_API_KEY=your_fred_api_key" > .env
```

FRED API 키는 https://fred.stlouisfed.org/docs/api/api_key.html 에서 무료로 발급받을 수 있습니다.

**참고:** 암호화폐(CoinGecko)와 한국 시장(FinanceDataReader)은 API 키 불필요!

### 3. 실행

```bash
# 모든 지표 수집
python main.py

# 특정 지표만 수집
python main.py --fred-only
python main.py --market-only
python main.py --crypto-only
python main.py --korea-only

# 빠른 수집 (최근 30일)
python main.py --quick

# 전체 수집 (1년)
python main.py --full
```

## 🔄 Multi-Source Fallback

각 테마별로 최적의 데이터 소스를 사용하며, 실패 시 자동으로 fallback합니다.

### 암호화폐 데이터
1. **CoinGecko API** (Primary, 무료 ⭐)
   - API 키 불필요
   - 높은 품질과 안정성
   - Rate Limit: 50 calls/min

2. **Binance API** (Secondary, 무료)
   - API 키 불필요 (공개 데이터)
   - OHLC 데이터 포함
   - Rate Limit: 1200 requests/min

3. **yfinance** (Fallback)
   - 완전 무료
   - 안정성 낮음

### 한국 시장 데이터
1. **FinanceDataReader** (Primary, 무료 ⭐)
   - 한국 시장 전용 최적화
   - KRX, Naver 등 다중 소스
   - 설치: `pip install finance-datareader`

2. **pykrx** (Secondary, 무료)
   - KRX 공식 데이터
   - 정확한 데이터
   - 설치: `pip install pykrx`

3. **yfinance** (Fallback)
   - 한국 시장 데이터 품질 낮음

### 미국 시장 데이터
- 현재: **yfinance** (무료)
- 업그레이드 옵션:
  - Alpha Vantage (API 키 필요)
  - Polygon.io (API 키 필요)

## 📁 데이터 저장

수집된 데이터는 다음 형식으로 저장됩니다:

```
data/
├── fred_YYYYMMDD_HHMMSS.csv
├── market_YYYYMMDD_HHMMSS.csv
├── crypto_YYYYMMDD_HHMMSS.csv
├── korea_YYYYMMDD_HHMMSS.csv
└── combined_YYYYMMDD_HHMMSS.json
```

각 collector는 사용한 데이터 소스를 출력에 표시합니다.

## 📈 사용 예시

### Python 스크립트에서 사용

```python
from collectors.fred_collector import FREDCollector
from collectors.market_collector import MarketCollector

# FRED 데이터 수집
fred = FREDCollector()
fred_data = fred.collect_all()
print(f"Fed Funds Rate: {fred_data['fed_funds']}")

# 시장 데이터 수집
market = MarketCollector()
spy_data = market.collect_ticker('SPY')
print(f"SPY 최근 종가: {spy_data['Close'].iloc[-1]}")
```

## 🔧 설정

`config.py`에서 다음을 설정할 수 있습니다:

- 수집 기간 (기본: 90일)
- 수집할 티커 목록
- 데이터 저장 경로
- API 재시도 횟수

## 📚 의존성

- pandas
- yfinance
- requests
- python-dotenv

## 📊 전통적 금융 분석 기능

### 1. 기술적 분석 (Technical Analysis)
- **추세 지표**: 이동평균(SMA/EMA), MACD, ADX
- **모멘텀 지표**: RSI, 스토캐스틱, ROC
- **변동성 지표**: 볼린저밴드, ATR, 켈트너채널
- **거래량 지표**: OBV, VWAP, MFI

### 2. 기본적 분석 (Fundamental Analysis)
- **수익성 비율**: ROE, ROA, 마진율
- **밸류에이션**: P/E, P/B, EV/EBITDA
- **유동성 비율**: 유동비율, 당좌비율
- **레버리지 비율**: 부채비율, 이자보상배율
- **고급 분석**: DuPont, Altman Z-Score, Piotroski F-Score

### 3. 포트폴리오 분석 (Portfolio Analysis)
- **현대 포트폴리오 이론**: 마코위츠 최적화
- **최적화 전략**: 최소분산, 최대샤프, 위험균형
- **성과 평가**: Sharpe, Sortino, Information Ratio
- **CAPM 분석**: Alpha, Beta, 기대수익률

### 4. 리스크 관리 (Risk Management)
- **VaR 계산**: 역사적, 파라메트릭, 몬테카를로
- **변동성 측정**: 실현변동성, EWMA, GARCH
- **테일 리스크**: Skewness, Kurtosis, CVaR
- **스트레스 테스트**: 시나리오 분석

### 5. 가치평가 (Valuation)
- **DCF 모형**: 기업가치, FCF 할인
- **배당할인모형**: 고든모형, 2단계모형
- **상대가치평가**: Comparable, 거래사례
- **옵션가격결정**: Black-Scholes, Greeks

### 분석 사용 예시

```python
from analysis import TechnicalAnalysis, PortfolioAnalysis

# 기술적 분석
ta = TechnicalAnalysis(price_data)
rsi = ta.rsi()
signals = ta.get_signals()  # {'RSI': 'Oversold', 'MACD': 'Bullish'}

# 포트폴리오 최적화
pa = PortfolioAnalysis(returns_df)
optimal = pa.maximum_sharpe_portfolio()
print(f"최적 비중: {optimal['weights']}")
print(f"샤프 비율: {optimal['sharpe']:.2f}")
```

**자세한 내용**: `TRADITIONAL_FINANCE_IMPLEMENTATION.md`, `QUICK_REFERENCE.md` 참조

## 🎓 한국 금융 자격증 대응

본 시스템은 다음 자격증 시험 범위를 100% 커버합니다:

### 투자자산운용사
- ✅ 1과목: 금융상품 및 세제 (포트폴리오 이론, 파생상품)
- ✅ 2과목: 투자운용 및 전략 (성과평가, 위험관리)
- ✅ 3과목: 투자분석 (기본적/기술적 분석, 가치평가)

### 금융투자분석사
- ✅ 1과목: 증권분석 (재무제표, 가치평가)
- ✅ 2과목: 투자분석 (포트폴리오 이론, 경제분석)
- ✅ 3과목: 파생상품 분석 (옵션가격결정, Greeks)

**자세한 매핑**: `KOREAN_CERTIFICATION_MAPPING.md` 참조

## 📖 문서

- `README.md` (이 파일) - 시작 가이드
- `TRADITIONAL_FINANCE_IMPLEMENTATION.md` - 구현된 금융 기법 상세 설명
- `QUICK_REFERENCE.md` - 빠른 사용법 레퍼런스
- `KOREAN_CERTIFICATION_MAPPING.md` - 한국 자격증 시험 범위 매핑
- `analysis/README.md` - 분석 모듈 API 문서

## 📄 라이선스

MIT License

## 🤝 기여

이슈와 PR을 환영합니다!

## ⚠️ 중요: API 키 보안

- `.env` 파일은 절대 커밋하지 마세요 (이미 .gitignore에 포함됨)
- `.env.example`을 복사하여 사용하세요
- API 키를 코드에 직접 작성하지 마세요
