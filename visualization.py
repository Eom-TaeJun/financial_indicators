#!/usr/bin/env python3
"""
Financial Data Visualization
금융 데이터 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (선택적)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


class FinancialVisualizer:
    """금융 데이터 시각화 클래스"""

    def __init__(self, output_dir: str = "outputs/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 색상 팔레트
        self.colors = {
            'positive': '#2ecc71',  # Green
            'negative': '#e74c3c',  # Red
            'neutral': '#95a5a6',   # Gray
            'primary': '#3498db',   # Blue
            'warning': '#f39c12',   # Orange
        }

    def plot_sector_performance(self, sector_df: pd.DataFrame, save: bool = True) -> Optional[str]:
        """
        섹터 성과 비교 차트

        Args:
            sector_df: 섹터 스코어 DataFrame (from SectorRotationAnalyzer)
            save: 파일 저장 여부

        Returns:
            파일 경로 (save=True인 경우)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 섹터별 3개월 수익률
        data = sector_df.sort_values('3M %', ascending=True)
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] for x in data['3M %']]

        ax1.barh(data['Sector'], data['3M %'], color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('3-Month Return (%)', fontsize=12)
        ax1.set_title('Sector Performance (3M)', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 2. 종합 스코어
        data = sector_df.sort_values('Score', ascending=True)
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] for x in data['Score']]

        ax2.barh(data['Sector'], data['Score'], color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Composite Score', fontsize=12)
        ax2.set_title('Sector Composite Score', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"sector_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)

        plt.show()
        return None

    def plot_risk_return_scatter(self, risk_df: pd.DataFrame, returns_dict: Dict[str, float],
                                  save: bool = True) -> Optional[str]:
        """
        리스크-수익 산점도

        Args:
            risk_df: 리스크 팩터 DataFrame
            returns_dict: {ticker: return_pct}
            save: 파일 저장 여부

        Returns:
            파일 경로
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # 데이터 준비
        data = risk_df.copy()
        data['Return %'] = data['Ticker'].map(returns_dict)
        data = data.dropna(subset=['Return %'])

        # 산점도
        scatter = ax.scatter(
            data['Vol %'],
            data['Return %'],
            s=200,
            c=data['Beta'],
            cmap='RdYlGn_r',
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5
        )

        # 레이블
        for idx, row in data.iterrows():
            ax.annotate(
                row['Ticker'],
                (row['Vol %'], row['Return %']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )

        # 기준선
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=data['Vol %'].median(), color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # 색상바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Market Beta', fontsize=12)

        ax.set_xlabel('Volatility (Annual %)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"risk_return_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)

        plt.show()
        return None

    def plot_cumulative_returns(self, price_data: Dict[str, pd.DataFrame],
                                 tickers: List[str], save: bool = True) -> Optional[str]:
        """
        누적 수익률 차트

        Args:
            price_data: {ticker: DataFrame with 'Close'}
            tickers: 표시할 티커 리스트
            save: 파일 저장 여부

        Returns:
            파일 경로
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        for ticker in tickers:
            if ticker not in price_data:
                continue

            df = price_data[ticker]
            if df.empty or 'Close' not in df.columns:
                continue

            # 누적 수익률 계산
            prices = df['Close']
            cum_returns = (prices / prices.iloc[0] - 1) * 100

            ax.plot(cum_returns.index, cum_returns, label=ticker, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"cumulative_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)

        plt.show()
        return None

    def plot_volatility_comparison(self, risk_df: pd.DataFrame, save: bool = True) -> Optional[str]:
        """
        변동성 비교 차트

        Args:
            risk_df: 리스크 팩터 DataFrame
            save: 파일 저장 여부

        Returns:
            파일 경로
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        data = risk_df.sort_values('Vol %', ascending=True)

        # 변동성에 따른 색상
        colors = []
        for vol in data['Vol %']:
            if vol < 15:
                colors.append(self.colors['positive'])
            elif vol < 25:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['negative'])

        ax.barh(data['Ticker'], data['Vol %'], color=colors, alpha=0.7)
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_title('Volatility Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 기준선
        ax.axvline(x=15, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low Vol')
        ax.axvline(x=25, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Med Vol')
        ax.legend(loc='best')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"volatility_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)

        plt.show()
        return None

    def plot_correlation_heatmap(self, price_data: Dict[str, pd.DataFrame],
                                  tickers: List[str], save: bool = True) -> Optional[str]:
        """
        상관계수 히트맵

        Args:
            price_data: {ticker: DataFrame with 'Close'}
            tickers: 분석할 티커 리스트
            save: 파일 저장 여부

        Returns:
            파일 경로
        """
        # 가격 데이터 결합
        prices = pd.DataFrame()
        for ticker in tickers:
            if ticker in price_data and not price_data[ticker].empty:
                if 'Close' in price_data[ticker].columns:
                    # 중복 인덱스 제거
                    close_prices = price_data[ticker]['Close']
                    close_prices = close_prices[~close_prices.index.duplicated(keep='last')]
                    prices[ticker] = close_prices

        if prices.empty or len(prices.columns) < 2:
            print("⚠️  Not enough data for correlation heatmap")
            return None

        # 수익률 상관계수
        returns = prices.pct_change().dropna()
        corr = returns.corr()

        # 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title('Return Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            filepath = self.output_dir / f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)

        plt.show()
        return None

    def create_dashboard_html(self, charts: Dict[str, str], summary_data: Dict) -> str:
        """
        HTML 대시보드 생성

        Args:
            charts: {chart_name: filepath}
            summary_data: 요약 데이터

        Returns:
            HTML 파일 경로
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Financial Analysis Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px;
        }}
        .section {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .positive {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Financial Analysis Dashboard</h1>
        <p>{timestamp}</p>
    </div>

    <div class="section">
        <h2>📈 Sector Performance</h2>
        {sector_chart}
    </div>

    <div class="section">
        <h2>💹 Risk-Return Analysis</h2>
        {risk_return_chart}
    </div>

    <div class="section">
        <h2>📊 Cumulative Returns</h2>
        {cumulative_chart}
    </div>

    <div class="section">
        <h2>⚡ Volatility Comparison</h2>
        {volatility_chart}
    </div>

    <div class="section">
        <h2>🔗 Correlation Matrix</h2>
        {correlation_chart}
    </div>

    <div class="section">
        <h2>💡 Summary</h2>
        {summary}
    </div>
</body>
</html>
"""

        # 차트 HTML 생성
        chart_html = {}
        for name, filepath in charts.items():
            if filepath and Path(filepath).exists():
                chart_html[name] = f'<div class="chart"><img src="{Path(filepath).name}" /></div>'
            else:
                chart_html[name] = '<p>Chart not available</p>'

        # 요약 데이터 HTML
        summary_html = "<ul>"
        for key, value in summary_data.items():
            summary_html += f"<li><strong>{key}:</strong> {value}</li>"
        summary_html += "</ul>"

        # HTML 생성
        html_content = html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            sector_chart=chart_html.get('sector', ''),
            risk_return_chart=chart_html.get('risk_return', ''),
            cumulative_chart=chart_html.get('cumulative', ''),
            volatility_chart=chart_html.get('volatility', ''),
            correlation_chart=chart_html.get('correlation', ''),
            summary=summary_html
        )

        # 파일 저장
        filepath = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)


def test_visualization():
    """시각화 테스트"""
    from db_manager import DatabaseManager
    from sector_analysis import SectorRotationAnalyzer, RiskFactorAnalyzer

    print("="*70)
    print("🧪 Testing Financial Visualization")
    print("="*70)

    db = DatabaseManager()
    viz = FinancialVisualizer()

    # 1. 섹터 데이터 로드
    print("\n1️⃣  Loading sector data...")
    sector_tickers = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    sector_data = {}

    for ticker in sector_tickers:
        df = db.get_latest_market_data(ticker)
        if not df.empty:
            df = df.set_index('date')[['close']]
            df.columns = ['Close']
            df.index = pd.to_datetime(df.index)
            sector_data[ticker] = df

    # SPY 로드
    spy = db.get_latest_market_data('SPY')
    spy_df = spy.set_index('date')[['close']]
    spy_df.columns = ['Close']
    spy_df.index = pd.to_datetime(spy_df.index)

    # 2. 섹터 분석
    print("\n2️⃣  Running sector analysis...")
    sector_analyzer = SectorRotationAnalyzer(sector_data, spy_df)
    sector_analyzer.analyze_all_sectors()
    sector_df = sector_analyzer.to_dataframe()

    # 3. 섹터 성과 차트
    print("\n3️⃣  Creating sector performance chart...")
    chart1 = viz.plot_sector_performance(sector_df)
    print(f"   ✅ Saved: {chart1}")

    # 4. 리스크 팩터 분석
    print("\n4️⃣  Running risk factor analysis...")
    test_tickers = ['AAPL', 'NVDA', 'TSLA', 'JPM', 'TLT']
    assets = {}

    for ticker in test_tickers:
        df = db.get_latest_market_data(ticker)
        if not df.empty:
            df = df.set_index('date')[['close']]
            df.columns = ['Close']
            df.index = pd.to_datetime(df.index)
            assets[ticker] = df

    risk_analyzer = RiskFactorAnalyzer(spy_df)
    factors = risk_analyzer.analyze_portfolio(assets)
    risk_df = risk_analyzer.to_dataframe(factors)

    # 3개월 수익률
    returns_dict = {}
    for ticker, df in assets.items():
        if len(df) > 63:
            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100
            returns_dict[ticker] = ret

    # 5. 리스크-수익 차트
    print("\n5️⃣  Creating risk-return scatter...")
    chart2 = viz.plot_risk_return_scatter(risk_df, returns_dict)
    print(f"   ✅ Saved: {chart2}")

    # 6. 누적 수익률
    print("\n6️⃣  Creating cumulative returns chart...")
    chart3 = viz.plot_cumulative_returns(assets, test_tickers)
    print(f"   ✅ Saved: {chart3}")

    # 7. 변동성 비교
    print("\n7️⃣  Creating volatility comparison...")
    chart4 = viz.plot_volatility_comparison(risk_df)
    print(f"   ✅ Saved: {chart4}")

    # 8. 상관계수 히트맵
    print("\n8️⃣  Creating correlation heatmap...")
    chart5 = viz.plot_correlation_heatmap(assets, test_tickers)
    print(f"   ✅ Saved: {chart5}")

    # 9. HTML 대시보드
    print("\n9️⃣  Creating HTML dashboard...")
    charts = {
        'sector': chart1,
        'risk_return': chart2,
        'cumulative': chart3,
        'volatility': chart4,
        'correlation': chart5,
    }

    summary = {
        'Analysis Date': datetime.now().strftime('%Y-%m-%d'),
        'Sectors Analyzed': len(sector_data),
        'Assets Analyzed': len(assets),
        'Top Sector': sector_df.iloc[0]['Sector'],
        'Economic Cycle': 'Contraction',
    }

    dashboard_path = viz.create_dashboard_html(charts, summary)
    print(f"   ✅ Dashboard saved: {dashboard_path}")

    print("\n" + "="*70)
    print("✅ Visualization test completed!")
    print(f"📁 Output directory: {viz.output_dir}")
    print("="*70)


if __name__ == "__main__":
    test_visualization()
