#!/usr/bin/env python3
"""
Database Manager for Financial Indicators
SQLite를 사용한 금융 데이터 저장 및 조회
"""

import sqlite3
import logging
import pandas as pd
from typing import Dict, Any, Optional, Iterable, Sequence
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 데이터베이스 관리 클래스"""

    _OHLCV_DB_COLUMNS = ("open", "high", "low", "close", "volume")
    _OHLCV_DF_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
    _KOREA_EXTRA_COLUMNS = ("institutional_net", "foreign_net", "market_cap")
    _OHLCV_TABLE_COLUMNS = {
        "market_data": _OHLCV_DB_COLUMNS,
        "crypto_data": _OHLCV_DB_COLUMNS,
        "korea_data": _OHLCV_DB_COLUMNS + _KOREA_EXTRA_COLUMNS,
    }

    def __init__(self, db_path: str = 'data/financial_indicators.db'):
        """
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path

        # 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 연결 및 스키마 초기화
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """DB 연결 반환"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Dict 형태로 결과 반환
        return conn

    def _init_schema(self):
        """데이터베이스 스키마 초기화"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Collection Runs 테이블 - 수집 실행 메타데이터
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                lookback_days INTEGER,
                fred_success BOOLEAN,
                market_success BOOLEAN,
                crypto_success BOOLEAN,
                korea_success BOOLEAN,
                fred_series_count INTEGER,
                market_ticker_count INTEGER,
                crypto_asset_count INTEGER,
                korea_asset_count INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. FRED 데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fred_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_run_id INTEGER,
                date DATE NOT NULL,
                series_name TEXT NOT NULL,
                series_code TEXT NOT NULL,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_run_id) REFERENCES collection_runs(id),
                UNIQUE(collection_run_id, date, series_code)
            )
        """)

        # 3. Market 데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_run_id INTEGER,
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                category TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_run_id) REFERENCES collection_runs(id),
                UNIQUE(collection_run_id, date, ticker)
            )
        """)

        # 4. Crypto 데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_run_id INTEGER,
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                category TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_run_id) REFERENCES collection_runs(id),
                UNIQUE(collection_run_id, date, ticker)
            )
        """)

        # 5. Korea 데이터 테이블 (고급 데이터 포함)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS korea_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_run_id INTEGER,
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                category TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                institutional_net REAL,
                foreign_net REAL,
                market_cap REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_run_id) REFERENCES collection_runs(id),
                UNIQUE(collection_run_id, date, ticker)
            )
        """)

        # 기존 테이블에 새 컬럼 추가 (ALTER TABLE - 안전하게)
        try:
            cursor.execute("ALTER TABLE korea_data ADD COLUMN institutional_net REAL")
        except sqlite3.OperationalError:
            pass  # 컬럼이 이미 존재

        try:
            cursor.execute("ALTER TABLE korea_data ADD COLUMN foreign_net REAL")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE korea_data ADD COLUMN market_cap REAL")
        except sqlite3.OperationalError:
            pass

        # 인덱스 생성 (조회 성능 향상)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fred_date ON fred_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fred_series ON fred_data(series_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_data(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crypto_date ON crypto_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crypto_ticker ON crypto_data(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_korea_date ON korea_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_korea_ticker ON korea_data(ticker)")

        conn.commit()
        conn.close()

    @staticmethod
    def _to_db_date(value: Any) -> str:
        """DataFrame index 값을 DB 저장용 YYYY-MM-DD 문자열로 변환."""
        return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        """값을 float로 변환. 결측/변환 불가 값은 None."""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_positive_int(value: Any, default: int = 1) -> int:
        """양의 정수로 변환. 실패 시 default 반환."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _build_ohlcv_records(
        self,
        collection_run_id: int,
        data: Dict[str, pd.DataFrame],
        value_columns: Sequence[str],
        category_map: Optional[Dict[str, str]] = None,
    ) -> list[tuple]:
        """OHLCV 계열 DataFrame 딕셔너리를 INSERT 레코드로 변환."""
        category_map = category_map or {}
        records: list[tuple] = []

        for ticker, df in data.items():
            if df.empty:
                continue

            category = category_map.get(ticker, "unknown")

            for idx, row in df.iterrows():
                values = tuple(self._to_optional_float(row.get(col)) for col in value_columns)
                records.append(
                    (
                        collection_run_id,
                        self._to_db_date(idx),
                        ticker,
                        category,
                        *values,
                    )
                )

        return records

    def _save_ohlcv_table(
        self,
        table: str,
        collection_run_id: int,
        data: Dict[str, pd.DataFrame],
        category_map: Optional[Dict[str, str]] = None,
    ) -> int:
        """market/crypto/korea OHLCV 계열 테이블 저장 공통 함수."""
        value_columns = self._OHLCV_TABLE_COLUMNS.get(table)
        if value_columns is None:
            raise ValueError(f"Unsupported OHLCV table: {table}")

        df_columns: Iterable[str] = self._OHLCV_DF_COLUMNS
        if table == "korea_data":
            df_columns = self._OHLCV_DF_COLUMNS + self._KOREA_EXTRA_COLUMNS

        records = self._build_ohlcv_records(
            collection_run_id=collection_run_id,
            data=data,
            value_columns=tuple(df_columns),
            category_map=category_map,
        )

        if not records:
            return 0

        placeholders = ", ".join(["?"] * (4 + len(value_columns)))
        columns_sql = ", ".join(("collection_run_id", "date", "ticker", "category", *value_columns))

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.executemany(
            f"""
            INSERT OR REPLACE INTO {table}
            ({columns_sql})
            VALUES ({placeholders})
        """,
            records,
        )
        conn.commit()
        conn.close()

        return len(records)

    def save_collection_run(self, results: Dict[str, Any]) -> int:
        """
        수집 실행 메타데이터 저장

        Args:
            results: collect_data() 결과 딕셔너리

        Returns:
            collection_run_id
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        metadata = results.get('metadata', {})
        summary = results.get('summary', {})

        cursor.execute("""
            INSERT INTO collection_runs (
                timestamp, lookback_days,
                fred_success, market_success, crypto_success, korea_success,
                fred_series_count, market_ticker_count,
                crypto_asset_count, korea_asset_count,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.get('timestamp'),
            metadata.get('lookback_days'),
            summary.get('fred', {}).get('success', False),
            summary.get('market', {}).get('success', False),
            summary.get('crypto', {}).get('success', False),
            summary.get('korea', {}).get('success', False),
            summary.get('fred', {}).get('series_count', 0),
            summary.get('market', {}).get('ticker_count', 0),
            summary.get('crypto', {}).get('asset_count', 0),
            summary.get('korea', {}).get('asset_count', 0),
            json.dumps(metadata)
        ))

        collection_run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return collection_run_id

    def save_fred_data(self, collection_run_id: int, fred_data: Dict[str, pd.DataFrame]):
        """
        FRED 데이터 저장

        Args:
            collection_run_id: 수집 실행 ID
            fred_data: {series_name: DataFrame} 형태
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        records = []
        for series_name, df in fred_data.items():
            if df.empty:
                continue

            # DataFrame을 리스트로 변환
            for idx, row in df.iterrows():
                if pd.notna(row.get('value')) or pd.notna(row.iloc[0]):
                    value = row.get('value', row.iloc[0])
                    records.append((
                        collection_run_id,
                        idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        series_name,
                        series_name,  # series_code는 series_name과 동일
                        float(value) if pd.notna(value) else None
                    ))

        # Batch insert
        cursor.executemany("""
            INSERT OR REPLACE INTO fred_data
            (collection_run_id, date, series_name, series_code, value)
            VALUES (?, ?, ?, ?, ?)
        """, records)

        conn.commit()
        conn.close()

        logger.info("Saved %s FRED data points to database", len(records))

    def save_market_data(self, collection_run_id: int, market_data: Dict[str, pd.DataFrame],
                        category_map: Optional[Dict[str, str]] = None):
        """
        Market 데이터 저장

        Args:
            collection_run_id: 수집 실행 ID
            market_data: {ticker: DataFrame} 형태
            category_map: {ticker: category} 매핑
        """
        saved_count = self._save_ohlcv_table(
            table="market_data",
            collection_run_id=collection_run_id,
            data=market_data,
            category_map=category_map,
        )
        logger.info("Saved %s market data points to database", saved_count)

    def save_crypto_data(self, collection_run_id: int, crypto_data: Dict[str, pd.DataFrame],
                        category_map: Optional[Dict[str, str]] = None):
        """
        Crypto 데이터 저장

        Args:
            collection_run_id: 수집 실행 ID
            crypto_data: {ticker: DataFrame} 형태
            category_map: {ticker: category} 매핑
        """
        saved_count = self._save_ohlcv_table(
            table="crypto_data",
            collection_run_id=collection_run_id,
            data=crypto_data,
            category_map=category_map,
        )
        logger.info("Saved %s crypto data points to database", saved_count)

    def save_korea_data(self, collection_run_id: int, korea_data: Dict[str, pd.DataFrame],
                       category_map: Optional[Dict[str, str]] = None):
        """
        Korea 데이터 저장 (고급 데이터 포함)

        Args:
            collection_run_id: 수집 실행 ID
            korea_data: {ticker: DataFrame} 형태
            category_map: {ticker: category} 매핑
        """
        saved_count = self._save_ohlcv_table(
            table="korea_data",
            collection_run_id=collection_run_id,
            data=korea_data,
            category_map=category_map,
        )
        logger.info("Saved %s korea data points to database", saved_count)

    def get_latest_fred_data(self, series_code: Optional[str] = None) -> pd.DataFrame:
        """
        최신 FRED 데이터 조회

        Args:
            series_code: 특정 시리즈만 조회 (None이면 전체)

        Returns:
            DataFrame
        """
        conn = self._get_connection()

        if series_code:
            query = """
                SELECT date, series_name, series_code, value
                FROM fred_data
                WHERE series_code = ?
                ORDER BY date DESC
                LIMIT 100
            """
            df = pd.read_sql_query(query, conn, params=(series_code,))
        else:
            query = """
                SELECT date, series_name, series_code, value
                FROM fred_data
                WHERE collection_run_id = (SELECT MAX(id) FROM collection_runs)
                ORDER BY date DESC
            """
            df = pd.read_sql_query(query, conn)

        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_latest_market_data(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        최신 시장 데이터 조회

        Args:
            ticker: 특정 티커만 조회 (None이면 전체)

        Returns:
            DataFrame
        """
        conn = self._get_connection()

        if ticker:
            query = """
                SELECT date, ticker, category, open, high, low, close, volume
                FROM market_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 100
            """
            df = pd.read_sql_query(query, conn, params=(ticker,))
        else:
            query = """
                SELECT date, ticker, category, open, high, low, close, volume
                FROM market_data
                WHERE collection_run_id = (SELECT MAX(id) FROM collection_runs)
                ORDER BY date DESC
            """
            df = pd.read_sql_query(query, conn)

        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_collection_runs(self, limit: int = 10) -> pd.DataFrame:
        """
        수집 실행 이력 조회

        Args:
            limit: 조회할 개수

        Returns:
            DataFrame
        """
        conn = self._get_connection()

        safe_limit = self._to_positive_int(limit, default=1)
        query = """
            SELECT *
            FROM collection_runs
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(safe_limit,))

        conn.close()

        return df

    def get_db_stats(self) -> Dict[str, Any]:
        """
        데이터베이스 통계 조회

        Returns:
            통계 딕셔너리
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # 각 테이블 레코드 수
        for table in ['fred_data', 'market_data', 'crypto_data', 'korea_data', 'collection_runs']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f'{table}_count'] = cursor.fetchone()[0]

        # 날짜 범위
        for table in ['fred_data', 'market_data', 'crypto_data', 'korea_data']:
            cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
            result = cursor.fetchone()
            stats[f'{table}_date_range'] = {
                'min': result[0],
                'max': result[1]
            }

        # 데이터베이스 파일 크기
        stats['db_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)

        conn.close()

        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # 테스트 코드
    db = DatabaseManager()

    logger.info("=" * 70)
    logger.info("DATABASE MANAGER TEST")
    logger.info("=" * 70)

    logger.info("Database initialized: %s", db.db_path)

    stats = db.get_db_stats()
    logger.info("Database Statistics:")
    for key, value in stats.items():
        logger.info("%s: %s", key, value)

    logger.info("Database manager is ready")
