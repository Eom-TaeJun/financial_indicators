#!/usr/bin/env python3
"""
News Collector using Perplexity API
Perplexity API를 사용한 뉴스 수집
"""

import os
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
API_CALL_ERRORS = (
    requests.exceptions.RequestException,
    ValueError,
    KeyError,
    TypeError,
    json.JSONDecodeError,
)
JSON_PARSE_ERRORS = (json.JSONDecodeError, ValueError, TypeError)


@dataclass
class NewsArticle:
    """뉴스 기사"""
    title: str
    summary: str
    source: str
    relevance: str  # 'high', 'medium', 'low'
    sentiment: str  # 'positive', 'negative', 'neutral'
    timestamp: str


class PerplexityNewsCollector:
    """Perplexity API를 사용한 뉴스 수집기"""

    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment")

        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-small-128k-online"

    def _call_api(self, prompt: str) -> str:
        """Perplexity API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial news analyst. Provide accurate, concise summaries of recent financial news."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2000,
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']
        except API_CALL_ERRORS as e:
            logger.warning("Perplexity API error: %s", e)
            return ""

    def get_ticker_news(self, ticker: str, max_articles: int = 5) -> List[NewsArticle]:
        """
        특정 티커의 최신 뉴스 수집

        Args:
            ticker: 티커 심볼 (예: 'NVDA', 'BTC-USD')
            max_articles: 수집할 뉴스 개수

        Returns:
            뉴스 기사 리스트
        """
        # 티커에 따른 검색어 조정
        if 'USD' in ticker:
            search_term = ticker.replace('-USD', '')
            if search_term == 'BTC':
                search_term = 'Bitcoin'
            elif search_term == 'ETH':
                search_term = 'Ethereum'
        else:
            search_term = ticker

        prompt = f"""
Find the latest financial news about {search_term} ({ticker}) from the past 7 days.

For each article, provide:
1. Title
2. Brief summary (2-3 sentences)
3. Source
4. Sentiment (positive/negative/neutral)
5. Relevance to investors (high/medium/low)

Format your response as a JSON array with exactly {max_articles} articles.
Each article should have: title, summary, source, sentiment, relevance

Example format:
[
  {{
    "title": "Article Title",
    "summary": "Brief summary here",
    "source": "Bloomberg",
    "sentiment": "positive",
    "relevance": "high"
  }}
]

Return ONLY the JSON array, no additional text.
"""

        response = self._call_api(prompt)

        if not response:
            return []

        # JSON 파싱
        try:
            # JSON 부분만 추출
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                articles_data = json.loads(json_str)

                articles = []
                for article_data in articles_data[:max_articles]:
                    articles.append(NewsArticle(
                        title=article_data.get('title', 'No title'),
                        summary=article_data.get('summary', 'No summary'),
                        source=article_data.get('source', 'Unknown'),
                        relevance=article_data.get('relevance', 'medium'),
                        sentiment=article_data.get('sentiment', 'neutral'),
                        timestamp=datetime.now().isoformat()
                    ))

                return articles
            else:
                logger.warning("No JSON found in response for %s", ticker)
                return []

        except JSON_PARSE_ERRORS as e:
            logger.warning("JSON parse error for %s: %s", ticker, e)
            logger.debug("Raw response prefix for %s: %s", ticker, response[:200])
            return []

    def get_market_sentiment(self) -> Dict:
        """
        전체 시장 센티먼트 분석

        Returns:
            시장 센티먼트 딕셔너리
        """
        prompt = """
Analyze the current stock market sentiment based on recent news (past 3 days).

Consider:
1. Major market indices (S&P 500, NASDAQ, Dow Jones)
2. Economic indicators
3. Federal Reserve policy
4. Geopolitical events
5. Sector rotation

Provide:
1. Overall market sentiment (bullish/bearish/neutral)
2. Confidence level (high/medium/low)
3. Key drivers (3-5 bullet points)
4. Sector outlook (which sectors are favored)
5. Risk factors (main concerns)

Format as JSON:
{
  "sentiment": "bullish/bearish/neutral",
  "confidence": "high/medium/low",
  "drivers": ["point 1", "point 2", ...],
  "favored_sectors": ["sector 1", "sector 2", ...],
  "risk_factors": ["risk 1", "risk 2", ...]
}

Return ONLY the JSON object.
"""

        response = self._call_api(prompt)

        if not response:
            return {
                "sentiment": "neutral",
                "confidence": "low",
                "drivers": [],
                "favored_sectors": [],
                "risk_factors": []
            }

        try:
            # JSON 부분만 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in market sentiment response")
                return {}

        except JSON_PARSE_ERRORS as e:
            logger.warning("JSON parse error in market sentiment: %s", e)
            return {}

    def get_economic_calendar(self) -> List[Dict]:
        """
        다가오는 주요 경제 이벤트

        Returns:
            경제 이벤트 리스트
        """
        prompt = """
List the most important upcoming economic events and data releases in the next 2 weeks that could impact financial markets.

Include:
- Date
- Event name
- Expected impact (high/medium/low)
- Brief description

Format as JSON array:
[
  {
    "date": "2026-02-10",
    "event": "CPI Report",
    "impact": "high",
    "description": "Monthly inflation data"
  }
]

Return ONLY the JSON array, maximum 10 events.
"""

        response = self._call_api(prompt)

        if not response:
            return []

        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return []

        except JSON_PARSE_ERRORS:
            return []


def test_news_collector():
    """뉴스 수집기 테스트"""
    print("="*70)
    print("🧪 Testing Perplexity News Collector")
    print("="*70)

    collector = PerplexityNewsCollector()

    # 1. 티커별 뉴스
    test_tickers = ['NVDA', 'BTC-USD', 'XLU']

    for ticker in test_tickers:
        print(f"\n📰 News for {ticker}:")
        print("-"*70)

        news = collector.get_ticker_news(ticker, max_articles=3)

        if news:
            for i, article in enumerate(news, 1):
                emoji = "📈" if article.sentiment == 'positive' else "📉" if article.sentiment == 'negative' else "➡️"
                relevance_emoji = "🔥" if article.relevance == 'high' else "⚡" if article.relevance == 'medium' else "💤"

                print(f"\n{i}. {emoji} {relevance_emoji} {article.title}")
                print(f"   Source: {article.source}")
                print(f"   Summary: {article.summary}")
                print(f"   Sentiment: {article.sentiment} | Relevance: {article.relevance}")
        else:
            print("   ❌ No news found")

    # 2. 시장 센티먼트
    print(f"\n\n📊 Market Sentiment Analysis:")
    print("="*70)

    sentiment = collector.get_market_sentiment()

    if sentiment:
        emoji = "📈" if sentiment.get('sentiment') == 'bullish' else "📉" if sentiment.get('sentiment') == 'bearish' else "➡️"

        print(f"{emoji} Overall Sentiment: {sentiment.get('sentiment', 'N/A').upper()}")
        print(f"   Confidence: {sentiment.get('confidence', 'N/A')}")

        if sentiment.get('drivers'):
            print(f"\n   Key Drivers:")
            for driver in sentiment['drivers']:
                print(f"   - {driver}")

        if sentiment.get('favored_sectors'):
            print(f"\n   Favored Sectors: {', '.join(sentiment['favored_sectors'])}")

        if sentiment.get('risk_factors'):
            print(f"\n   Risk Factors:")
            for risk in sentiment['risk_factors']:
                print(f"   - {risk}")

    # 3. 경제 캘린더
    print(f"\n\n📅 Economic Calendar:")
    print("="*70)

    events = collector.get_economic_calendar()

    if events:
        for event in events[:5]:
            impact_emoji = "🔴" if event.get('impact') == 'high' else "🟡" if event.get('impact') == 'medium' else "🟢"
            print(f"{impact_emoji} {event.get('date')}: {event.get('event')}")
            print(f"   {event.get('description')}")
    else:
        print("   ❌ No events found")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_news_collector()
