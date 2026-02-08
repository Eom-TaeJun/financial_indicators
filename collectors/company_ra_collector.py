#!/usr/bin/env python3
"""
Company RA Collector
====================
RA(리서치 어시스턴트) 업무에 맞춘 기업 재무/회계 + 밸류에이션 + ETF 스냅샷 수집기.

핵심 기능:
- 기업별 재무제표(손익/재무상태/현금흐름) 핵심 항목 추출
- 기본 회계/재무 비율(ROE, ROA, 마진, D/E, 유동비율) 계산
- 밸류에이션 신호(동일 섹터 Peer P/E 대비) 생성
- 매크로/ETF 전략 보조용 ETF 모멘텀 스냅샷 생성
- PostgreSQL(fi_ra.company_fundamentals) 업서트 저장 (옵션)

PostgreSQL 환경 변수:
- FI_RA_POSTGRES_ENABLED=true|false
- FI_PG_DSN=postgresql://user:pass@host:5432/dbname
- FI_PG_SCHEMA=fi_ra (optional)
- FI_PG_TABLE=company_fundamentals (optional)
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from ..analysis import FundamentalAnalysis, ValuationModels
except ImportError:
    from analysis import FundamentalAnalysis, ValuationModels


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _calc_return_pct(series: pd.Series, periods: int) -> Optional[float]:
    if series is None or len(series) <= periods:
        return None
    past = _safe_float(series.iloc[-(periods + 1)])
    latest = _safe_float(series.iloc[-1])
    if past is None or latest is None or abs(past) < 1e-12:
        return None
    return (latest / past - 1.0) * 100.0


class CompanyRACollector:
    """RA용 기업 재무/회계/밸류에이션 데이터 수집기."""

    DEFAULT_COMPANY_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    DEFAULT_ETF_TICKERS = ["SPY", "QQQ", "IWM", "XLF", "TLT", "GLD"]

    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days
        self.fundamental = FundamentalAnalysis()
        self.valuation = ValuationModels(discount_rate=0.10)

        self.enable_postgres = _env_flag("FI_RA_POSTGRES_ENABLED", default=True)
        self.pg_dsn = os.getenv("FI_PG_DSN", "").strip() or os.getenv("DATABASE_URL", "").strip()
        self.pg_schema = self._sanitize_identifier(os.getenv("FI_PG_SCHEMA", "fi_ra"), "fi_ra")
        self.pg_table = self._sanitize_identifier(
            os.getenv("FI_PG_TABLE", "company_fundamentals"),
            "company_fundamentals",
        )

        self._psycopg = None

    @staticmethod
    def _sanitize_identifier(raw: Optional[str], fallback: str) -> str:
        candidate = (raw or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
            return candidate
        return fallback

    @staticmethod
    def _extract_statement_value(frame: Any, keys: List[str]) -> Optional[float]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None

        for key in keys:
            if key not in frame.index:
                continue
            row = frame.loc[key]
            if isinstance(row, pd.Series):
                non_na = row.dropna()
                if non_na.empty:
                    continue
                return _safe_float(non_na.iloc[0])
            value = _safe_float(row)
            if value is not None:
                return value
        return None

    @staticmethod
    def _extract_info_value(info: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            value = _safe_float(info.get(key))
            if value is not None:
                return value
        return None

    def _resolve_company_tickers(self, tickers: Optional[List[str]]) -> List[str]:
        if tickers:
            normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
            return list(dict.fromkeys(normalized))

        raw = os.getenv("FI_RA_COMPANY_TICKERS", "").strip()
        if raw:
            items = [item.strip().upper() for item in raw.split(",") if item.strip()]
            items = list(dict.fromkeys(items))
            if items:
                return items

        return list(self.DEFAULT_COMPANY_TICKERS)

    def _resolve_etf_tickers(self, tickers: Optional[List[str]]) -> List[str]:
        if tickers:
            normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
            return list(dict.fromkeys(normalized))

        raw = os.getenv("FI_RA_ETF_TICKERS", "").strip()
        if raw:
            items = [item.strip().upper() for item in raw.split(",") if item.strip()]
            items = list(dict.fromkeys(items))
            if items:
                return items

        return list(self.DEFAULT_ETF_TICKERS)

    def _build_company_record(self, ticker: str) -> Dict[str, Any]:
        obj = yf.Ticker(ticker)

        try:
            info = obj.info or {}
        except Exception:
            info = {}

        try:
            hist = obj.history(period="6mo", auto_adjust=True)
        except Exception:
            hist = pd.DataFrame()

        close_series = hist["Close"] if isinstance(hist, pd.DataFrame) and "Close" in hist.columns else pd.Series(dtype=float)
        last_close = _safe_float(close_series.iloc[-1]) if len(close_series) else None

        # Financial statements (annual, latest first)
        try:
            income_stmt = obj.financials
        except Exception:
            income_stmt = pd.DataFrame()

        try:
            balance_sheet = obj.balance_sheet
        except Exception:
            balance_sheet = pd.DataFrame()

        try:
            cashflow = obj.cashflow
        except Exception:
            cashflow = pd.DataFrame()

        revenue = self._extract_statement_value(income_stmt, ["Total Revenue", "Revenue"])
        operating_income = self._extract_statement_value(income_stmt, ["Operating Income", "EBIT"])
        net_income = self._extract_statement_value(income_stmt, ["Net Income", "Net Income Common Stockholders"])

        total_assets = self._extract_statement_value(balance_sheet, ["Total Assets"])
        total_liabilities = self._extract_statement_value(
            balance_sheet,
            ["Total Liab", "Total Liabilities Net Minority Interest", "Total Liabilities"],
        )
        total_equity = self._extract_statement_value(
            balance_sheet,
            ["Stockholders Equity", "Total Equity Gross Minority Interest", "Total Equity"],
        )
        current_assets = self._extract_statement_value(balance_sheet, ["Current Assets", "Total Current Assets"])
        current_liabilities = self._extract_statement_value(
            balance_sheet,
            ["Current Liabilities", "Total Current Liabilities"],
        )

        total_debt = self._extract_statement_value(balance_sheet, ["Total Debt"])
        if total_debt is None:
            long_debt = self._extract_statement_value(balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
            short_debt = self._extract_statement_value(balance_sheet, ["Current Debt", "Current Debt And Capital Lease Obligation"])
            if long_debt is not None or short_debt is not None:
                total_debt = (long_debt or 0.0) + (short_debt or 0.0)

        operating_cf = self._extract_statement_value(
            cashflow,
            ["Total Cash From Operating Activities", "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
        )
        capex = self._extract_statement_value(cashflow, ["Capital Expenditures", "Capital Expenditure"])
        free_cf = self._extract_statement_value(cashflow, ["Free Cash Flow"])
        if free_cf is None and operating_cf is not None and capex is not None:
            free_cf = operating_cf + capex if capex < 0 else operating_cf - capex

        market_cap = self._extract_info_value(info, ["marketCap"])
        trailing_pe = self._extract_info_value(info, ["trailingPE"])
        forward_pe = self._extract_info_value(info, ["forwardPE"])
        price_to_book = self._extract_info_value(info, ["priceToBook"])
        ev_to_ebitda = self._extract_info_value(info, ["enterpriseToEbitda"])

        if trailing_pe is None and last_close is not None:
            trailing_eps = self._extract_info_value(info, ["trailingEps"])
            if trailing_eps is not None and abs(trailing_eps) > 1e-12:
                trailing_pe = last_close / trailing_eps

        roe = None
        roa = None
        op_margin = None
        net_margin = None
        debt_to_equity = None
        curr_ratio = None

        if net_income is not None and total_equity is not None:
            roe = _safe_float(self.fundamental.roe(net_income, total_equity))
        if net_income is not None and total_assets is not None:
            roa = _safe_float(self.fundamental.roa(net_income, total_assets))
        if operating_income is not None and revenue is not None:
            op_margin = _safe_float(self.fundamental.operating_margin(operating_income, revenue))
        if net_income is not None and revenue is not None:
            net_margin = _safe_float(self.fundamental.net_margin(net_income, revenue))
        if total_debt is not None and total_equity is not None:
            debt_to_equity = _safe_float(self.fundamental.debt_to_equity(total_debt, total_equity))
        if current_assets is not None and current_liabilities is not None:
            curr_ratio = _safe_float(self.fundamental.current_ratio(current_assets, current_liabilities))

        record = {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "currency": info.get("currency") or "USD",
            "last_close": _round_or_none(last_close, 4),
            "market_cap": _round_or_none(market_cap, 2),
            "valuation": {
                "trailing_pe": _round_or_none(trailing_pe, 3),
                "forward_pe": _round_or_none(forward_pe, 3),
                "price_to_book": _round_or_none(price_to_book, 3),
                "ev_to_ebitda": _round_or_none(ev_to_ebitda, 3),
            },
            "accounting": {
                "revenue": _round_or_none(revenue, 2),
                "operating_income": _round_or_none(operating_income, 2),
                "net_income": _round_or_none(net_income, 2),
                "operating_cash_flow": _round_or_none(operating_cf, 2),
                "free_cash_flow": _round_or_none(free_cf, 2),
                "total_assets": _round_or_none(total_assets, 2),
                "total_liabilities": _round_or_none(total_liabilities, 2),
                "total_equity": _round_or_none(total_equity, 2),
            },
            "ratios": {
                "roe": _round_or_none(roe, 2),
                "roa": _round_or_none(roa, 2),
                "operating_margin": _round_or_none(op_margin, 2),
                "net_margin": _round_or_none(net_margin, 2),
                "debt_to_equity": _round_or_none(debt_to_equity, 3),
                "current_ratio": _round_or_none(curr_ratio, 3),
            },
            "price_momentum": {
                "ret_1d_pct": _round_or_none(_calc_return_pct(close_series, 1), 2),
                "ret_5d_pct": _round_or_none(_calc_return_pct(close_series, 5), 2),
                "ret_20d_pct": _round_or_none(_calc_return_pct(close_series, 20), 2),
            },
            "valuation_signal": "N/A",
            "peer_pe_median": None,
            "ra_takeaway": "",
        }

        return record

    @staticmethod
    def _infer_valuation_signal(trailing_pe: Optional[float], peer_pe: Optional[float]) -> str:
        if trailing_pe is None or peer_pe is None or trailing_pe <= 0 or peer_pe <= 0:
            return "N/A"
        if trailing_pe <= peer_pe * 0.85:
            return "UNDERVALUED"
        if trailing_pe >= peer_pe * 1.15:
            return "OVERVALUED"
        return "FAIR"

    @staticmethod
    def _build_ra_takeaway(company: Dict[str, Any]) -> str:
        signal = company.get("valuation_signal", "N/A")
        ratios = company.get("ratios", {}) if isinstance(company.get("ratios"), dict) else {}
        roe = _safe_float(ratios.get("roe"))
        debt_to_equity = _safe_float(ratios.get("debt_to_equity"))
        ret_20d = _safe_float(company.get("price_momentum", {}).get("ret_20d_pct"))

        if signal == "UNDERVALUED" and (roe is not None and roe >= 12):
            return "동종 대비 밸류 할인 + 수익성 우수. 리서치 노트 업사이드 가설 점검 권고."
        if signal == "OVERVALUED" and (ret_20d is not None and ret_20d >= 8):
            return "단기 모멘텀 과열 가능성. 밸류/이익 컨센서스 갭 점검 필요."
        if debt_to_equity is not None and debt_to_equity > 2.0:
            return "레버리지 비율이 높아 금리 민감도 점검 필요."
        return "핵심 재무지표와 이익 모멘텀의 정합성 확인 후 포지션 판단 권고."

    def _build_etf_strategy_snapshot(self, etf_tickers: List[str]) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []

        for ticker in etf_tickers:
            try:
                hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
            except Exception:
                continue

            if not isinstance(hist, pd.DataFrame) or hist.empty or "Close" not in hist.columns:
                continue

            close = hist["Close"]
            ret_5 = _calc_return_pct(close, 5)
            ret_20 = _calc_return_pct(close, 20)
            momentum_label = "NEUTRAL"
            if ret_20 is not None:
                if ret_20 >= 5:
                    momentum_label = "UPTREND"
                elif ret_20 <= -5:
                    momentum_label = "DOWNTREND"

            snapshot.append(
                {
                    "ticker": ticker,
                    "last_close": _round_or_none(_safe_float(close.iloc[-1]), 4),
                    "ret_5d_pct": _round_or_none(ret_5, 2),
                    "ret_20d_pct": _round_or_none(ret_20, 2),
                    "momentum_label": momentum_label,
                }
            )

        return snapshot

    @staticmethod
    def _build_ra_work_support(companies: List[Dict[str, Any]], etf_snapshot: List[Dict[str, Any]]) -> Dict[str, Any]:
        top_names = [row.get("ticker", "N/A") for row in companies[:3]]
        top_line = ", ".join(top_names) if top_names else "주요 종목 미확보"

        return {
            "role_focus": "RA 1명 (매크로/ETF 전략)",
            "research_tasks": [
                f"기업 재무/회계 핵심지표 업데이트: {top_line}",
                "섹터/지수 ETF 모멘텀 비교표 작성 (5일/20일 수익률)",
                "밸류에이션 편차(동종 대비 P/E) 점검 및 코멘트 작성",
            ],
            "seminar_material_points": [
                "최근 매크로 환경과 ETF 자금흐름 요약 슬라이드",
                "기업 실적/밸류에이션 체커보드(ROE, 마진, D/E, P/E) 업데이트",
                "리스크 시나리오(금리/유가/달러)별 대응 포인트 정리",
            ],
            "cross_department_support_points": [
                "영업/운용 협조용 기업별 one-page 팩트시트 제공",
                "외부 세미나용 핵심 데이터 출처 및 산식 명시",
                "SQL(PostgreSQL) 기반 데이터 추출 요청 대응",
            ],
            "data_update_note": (
                "재무제표 기반 지표는 공시 갱신 주기(분기/연간)와 시차가 있으므로 "
                "가격지표와 업데이트 주기를 구분해 운용"
            ),
            "etf_coverage_count": len(etf_snapshot),
            "company_coverage_count": len(companies),
        }

    def _load_psycopg(self):
        if self._psycopg is not None:
            return self._psycopg

        try:
            import psycopg  # type: ignore

            self._psycopg = psycopg
            return psycopg
        except Exception:
            self._psycopg = None
            return None

    def _postgres_table_qualified(self) -> str:
        return f"{self.pg_schema}.{self.pg_table}"

    def _ensure_postgres_table(self, conn) -> None:
        table = self._postgres_table_qualified()
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.pg_schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    as_of_date DATE NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    sector TEXT,
                    currency TEXT,
                    market_cap DOUBLE PRECISION,
                    trailing_pe DOUBLE PRECISION,
                    forward_pe DOUBLE PRECISION,
                    price_to_book DOUBLE PRECISION,
                    ev_to_ebitda DOUBLE PRECISION,
                    revenue DOUBLE PRECISION,
                    operating_income DOUBLE PRECISION,
                    net_income DOUBLE PRECISION,
                    operating_cash_flow DOUBLE PRECISION,
                    free_cash_flow DOUBLE PRECISION,
                    total_assets DOUBLE PRECISION,
                    total_liabilities DOUBLE PRECISION,
                    total_equity DOUBLE PRECISION,
                    roe DOUBLE PRECISION,
                    roa DOUBLE PRECISION,
                    operating_margin DOUBLE PRECISION,
                    net_margin DOUBLE PRECISION,
                    debt_to_equity DOUBLE PRECISION,
                    current_ratio DOUBLE PRECISION,
                    ret_1d_pct DOUBLE PRECISION,
                    ret_5d_pct DOUBLE PRECISION,
                    ret_20d_pct DOUBLE PRECISION,
                    last_close DOUBLE PRECISION,
                    valuation_signal TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (as_of_date, ticker)
                )
                """
            )
        conn.commit()

    def _query_sector_peer_median(self, conn, sector: str) -> Optional[float]:
        table = self._postgres_table_qualified()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY trailing_pe)
                FROM {table}
                WHERE sector = %s
                  AND trailing_pe > 0
                  AND as_of_date >= CURRENT_DATE - INTERVAL '180 days'
                """,
                (sector,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return _safe_float(row[0])

    def _to_postgres_row(self, company: Dict[str, Any]) -> Dict[str, Any]:
        val = company.get("valuation", {}) if isinstance(company.get("valuation"), dict) else {}
        acc = company.get("accounting", {}) if isinstance(company.get("accounting"), dict) else {}
        ratios = company.get("ratios", {}) if isinstance(company.get("ratios"), dict) else {}
        mom = company.get("price_momentum", {}) if isinstance(company.get("price_momentum"), dict) else {}

        return {
            "as_of_date": date.today(),
            "ticker": company.get("ticker"),
            "company_name": company.get("company_name"),
            "sector": company.get("sector"),
            "currency": company.get("currency"),
            "market_cap": _safe_float(company.get("market_cap")),
            "trailing_pe": _safe_float(val.get("trailing_pe")),
            "forward_pe": _safe_float(val.get("forward_pe")),
            "price_to_book": _safe_float(val.get("price_to_book")),
            "ev_to_ebitda": _safe_float(val.get("ev_to_ebitda")),
            "revenue": _safe_float(acc.get("revenue")),
            "operating_income": _safe_float(acc.get("operating_income")),
            "net_income": _safe_float(acc.get("net_income")),
            "operating_cash_flow": _safe_float(acc.get("operating_cash_flow")),
            "free_cash_flow": _safe_float(acc.get("free_cash_flow")),
            "total_assets": _safe_float(acc.get("total_assets")),
            "total_liabilities": _safe_float(acc.get("total_liabilities")),
            "total_equity": _safe_float(acc.get("total_equity")),
            "roe": _safe_float(ratios.get("roe")),
            "roa": _safe_float(ratios.get("roa")),
            "operating_margin": _safe_float(ratios.get("operating_margin")),
            "net_margin": _safe_float(ratios.get("net_margin")),
            "debt_to_equity": _safe_float(ratios.get("debt_to_equity")),
            "current_ratio": _safe_float(ratios.get("current_ratio")),
            "ret_1d_pct": _safe_float(mom.get("ret_1d_pct")),
            "ret_5d_pct": _safe_float(mom.get("ret_5d_pct")),
            "ret_20d_pct": _safe_float(mom.get("ret_20d_pct")),
            "last_close": _safe_float(company.get("last_close")),
            "valuation_signal": company.get("valuation_signal"),
        }

    def _upsert_companies(self, conn, companies: List[Dict[str, Any]]) -> int:
        table = self._postgres_table_qualified()
        rows = [self._to_postgres_row(c) for c in companies if c.get("ticker")]
        if not rows:
            return 0

        query = f"""
            INSERT INTO {table} (
                as_of_date, ticker, company_name, sector, currency,
                market_cap, trailing_pe, forward_pe, price_to_book, ev_to_ebitda,
                revenue, operating_income, net_income, operating_cash_flow, free_cash_flow,
                total_assets, total_liabilities, total_equity,
                roe, roa, operating_margin, net_margin, debt_to_equity, current_ratio,
                ret_1d_pct, ret_5d_pct, ret_20d_pct, last_close, valuation_signal
            ) VALUES (
                %(as_of_date)s, %(ticker)s, %(company_name)s, %(sector)s, %(currency)s,
                %(market_cap)s, %(trailing_pe)s, %(forward_pe)s, %(price_to_book)s, %(ev_to_ebitda)s,
                %(revenue)s, %(operating_income)s, %(net_income)s, %(operating_cash_flow)s, %(free_cash_flow)s,
                %(total_assets)s, %(total_liabilities)s, %(total_equity)s,
                %(roe)s, %(roa)s, %(operating_margin)s, %(net_margin)s, %(debt_to_equity)s, %(current_ratio)s,
                %(ret_1d_pct)s, %(ret_5d_pct)s, %(ret_20d_pct)s, %(last_close)s, %(valuation_signal)s
            )
            ON CONFLICT (as_of_date, ticker)
            DO UPDATE SET
                company_name = EXCLUDED.company_name,
                sector = EXCLUDED.sector,
                currency = EXCLUDED.currency,
                market_cap = EXCLUDED.market_cap,
                trailing_pe = EXCLUDED.trailing_pe,
                forward_pe = EXCLUDED.forward_pe,
                price_to_book = EXCLUDED.price_to_book,
                ev_to_ebitda = EXCLUDED.ev_to_ebitda,
                revenue = EXCLUDED.revenue,
                operating_income = EXCLUDED.operating_income,
                net_income = EXCLUDED.net_income,
                operating_cash_flow = EXCLUDED.operating_cash_flow,
                free_cash_flow = EXCLUDED.free_cash_flow,
                total_assets = EXCLUDED.total_assets,
                total_liabilities = EXCLUDED.total_liabilities,
                total_equity = EXCLUDED.total_equity,
                roe = EXCLUDED.roe,
                roa = EXCLUDED.roa,
                operating_margin = EXCLUDED.operating_margin,
                net_margin = EXCLUDED.net_margin,
                debt_to_equity = EXCLUDED.debt_to_equity,
                current_ratio = EXCLUDED.current_ratio,
                ret_1d_pct = EXCLUDED.ret_1d_pct,
                ret_5d_pct = EXCLUDED.ret_5d_pct,
                ret_20d_pct = EXCLUDED.ret_20d_pct,
                last_close = EXCLUDED.last_close,
                valuation_signal = EXCLUDED.valuation_signal,
                created_at = NOW()
        """

        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
        return len(rows)

    def collect_all(
        self,
        tickers: Optional[List[str]] = None,
        etf_tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        company_tickers = self._resolve_company_tickers(tickers)
        etf_list = self._resolve_etf_tickers(etf_tickers)

        companies: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []

        print(f"\n🏢 RA Company Analysis ({len(company_tickers)} tickers)")
        for ticker in company_tickers:
            try:
                record = self._build_company_record(ticker)
                companies.append(record)
                print(f"   ✅ {ticker}: accounting/valuation snapshot")
            except Exception as exc:
                failures.append({"ticker": ticker, "error": str(exc)[:180]})
                print(f"   ❌ {ticker}: {exc}")

        # Local sector median trailing P/E
        sector_map: Dict[str, List[float]] = {}
        for row in companies:
            sector = row.get("sector") or "Unknown"
            pe = _safe_float(row.get("valuation", {}).get("trailing_pe"))
            if pe is not None and pe > 0:
                sector_map.setdefault(sector, []).append(pe)

        local_sector_medians = {
            sector: float(np.median(values))
            for sector, values in sector_map.items()
            if values
        }

        pg_status = {
            "enabled": bool(self.enable_postgres),
            "dsn_configured": bool(self.pg_dsn),
            "driver_available": False,
            "stored_rows": 0,
            "error": "",
            "table": self._postgres_table_qualified(),
        }

        conn = None
        if self.enable_postgres and self.pg_dsn:
            psycopg = self._load_psycopg()
            pg_status["driver_available"] = bool(psycopg)
            if psycopg is None:
                pg_status["error"] = "psycopg_not_installed"
            else:
                try:
                    conn = psycopg.connect(self.pg_dsn)
                    self._ensure_postgres_table(conn)
                except Exception as exc:
                    pg_status["error"] = f"postgres_connect_error:{type(exc).__name__}:{exc}"
                    conn = None

        # Peer median and valuation signal
        for row in companies:
            sector = row.get("sector") or "Unknown"
            peer_median = None

            if conn is not None:
                try:
                    peer_median = self._query_sector_peer_median(conn, sector)
                except Exception:
                    peer_median = None

            if peer_median is None:
                peer_median = local_sector_medians.get(sector)

            trailing_pe = _safe_float(row.get("valuation", {}).get("trailing_pe"))
            row["peer_pe_median"] = _round_or_none(peer_median, 3)
            row["valuation_signal"] = self._infer_valuation_signal(trailing_pe, peer_median)
            row["ra_takeaway"] = self._build_ra_takeaway(row)

        if conn is not None:
            try:
                stored = self._upsert_companies(conn, companies)
                pg_status["stored_rows"] = int(stored)
            except Exception as exc:
                pg_status["error"] = f"postgres_upsert_error:{type(exc).__name__}:{exc}"
            finally:
                conn.close()

        etf_snapshot = self._build_etf_strategy_snapshot(etf_list)
        ra_work_support = self._build_ra_work_support(companies, etf_snapshot)

        return {
            "timestamp": datetime.now().isoformat(),
            "role_focus": "RA 1명 (매크로/ETF 전략)",
            "companies": companies,
            "etf_strategy_snapshot": etf_snapshot,
            "ra_work_support": ra_work_support,
            "postgresql": pg_status,
            "failures": failures,
        }


if __name__ == "__main__":
    collector = CompanyRACollector(lookback_days=365)
    output = collector.collect_all()
    print(f"\nCollected companies: {len(output.get('companies', []))}")
    print(f"PostgreSQL stored rows: {output.get('postgresql', {}).get('stored_rows', 0)}")
