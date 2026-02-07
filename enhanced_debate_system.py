#!/usr/bin/env python3
"""
Enhanced Multi-Agent AI Debate System
뉴스 통합 + 추가 에이전트 + 다중 라운드 토론
"""

import os
import openai
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from dotenv import load_dotenv
import requests

load_dotenv()


@dataclass
class NewsContext:
    """뉴스 컨텍스트"""
    headlines: List[str]
    sentiment: str  # 'positive', 'negative', 'neutral'
    key_events: List[str]
    market_mood: str


@dataclass
class AgentOpinion:
    """에이전트 의견"""
    agent_name: str
    role: str
    round_number: int
    stance: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: int  # 0-100
    reasoning: str
    key_points: List[str]
    recommended_action: str  # 'BUY', 'SELL', 'HOLD'
    position_size: str  # 'AGGRESSIVE', 'MODERATE', 'CONSERVATIVE', 'NONE'
    rebuttals: List[str] = field(default_factory=list)  # 다른 에이전트에 대한 반박


@dataclass
class DebateRound:
    """토론 라운드"""
    round_number: int
    opinions: List[AgentOpinion]
    consensus_level: float  # 0-1
    main_disagreements: List[str]


@dataclass
class EnhancedDebateResult:
    """강화된 토론 결과"""
    ticker: str
    timestamp: str
    news_context: NewsContext
    rounds: List[DebateRound]
    final_consensus: Optional[str]
    final_recommendation: str
    confidence_score: float
    agents_changed_mind: List[str]  # 입장을 바꾼 에이전트
    full_transcript: str


class SimpleNewsCollector:
    """간단한 뉴스 수집기 (NewsAPI 사용)"""

    def __init__(self):
        # NewsAPI는 무료 티어 제공
        self.api_key = os.getenv('NEWS_API_KEY', '')
        self.base_url = "https://newsapi.org/v2/everything"

    def get_ticker_news(self, ticker: str) -> NewsContext:
        """티커 관련 뉴스 수집"""

        # 티커를 회사명으로 변환
        ticker_to_name = {
            'NVDA': 'NVIDIA',
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum',
            'XLU': 'Utilities sector',
        }

        search_term = ticker_to_name.get(ticker, ticker)

        # Mock news (실제 API 없을 경우)
        if not self.api_key:
            return self._get_mock_news(ticker, search_term)

        # 실제 NewsAPI 호출
        try:
            params = {
                'q': search_term,
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get('articles', [])

            if not articles:
                return self._get_mock_news(ticker, search_term)

            headlines = [article['title'] for article in articles[:5]]

            # 간단한 센티먼트 분석 (키워드 기반)
            positive_words = ['surge', 'gains', 'rises', 'bullish', 'growth', 'strong', 'beats', 'outperform']
            negative_words = ['falls', 'drops', 'bearish', 'decline', 'weak', 'misses', 'concern', 'risk']

            text = ' '.join(headlines).lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count > neg_count:
                sentiment = 'positive'
            elif neg_count > pos_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            key_events = headlines[:3]

            return NewsContext(
                headlines=headlines,
                sentiment=sentiment,
                key_events=key_events,
                market_mood='cautious' if sentiment == 'negative' else 'optimistic'
            )

        except Exception as e:
            print(f"⚠️  News API error: {e}, using mock data")
            return self._get_mock_news(ticker, search_term)

    def _get_mock_news(self, ticker: str, search_term: str) -> NewsContext:
        """Mock 뉴스 데이터"""
        mock_data = {
            'NVDA': NewsContext(
                headlines=[
                    "NVIDIA faces headwinds from AI chip competition",
                    "Tech sector volatility impacts NVDA stock",
                    "Analysts divided on NVIDIA's Q1 outlook",
                    "AI demand remains strong despite market concerns",
                    "NVIDIA announces new data center partnerships"
                ],
                sentiment='neutral',
                key_events=[
                    "Increased competition in AI chip market",
                    "Mixed analyst ratings",
                    "Strong data center demand"
                ],
                market_mood='cautious'
            ),
            'BTC-USD': NewsContext(
                headlines=[
                    "Bitcoin drops below $70K amid regulatory concerns",
                    "Crypto market faces selling pressure",
                    "Institutional investors remain on sidelines",
                    "Bitcoin ETF flows turn negative",
                    "Analysts warn of further downside risk"
                ],
                sentiment='negative',
                key_events=[
                    "Regulatory uncertainty increases",
                    "ETF outflows accelerate",
                    "Technical support levels broken"
                ],
                market_mood='fearful'
            ),
            'XLU': NewsContext(
                headlines=[
                    "Utilities sector gains as investors seek safety",
                    "Defensive stocks outperform in volatile market",
                    "XLU hits new 52-week high",
                    "Dividend yields attract income investors",
                    "Utilities benefit from rate stabilization"
                ],
                sentiment='positive',
                key_events=[
                    "Flight to safety boosts defensive sectors",
                    "Stable dividends attract investors",
                    "Rate environment favorable"
                ],
                market_mood='risk-averse'
            ),
        }

        return mock_data.get(ticker, NewsContext(
            headlines=[f"No specific news for {search_term}"],
            sentiment='neutral',
            key_events=[],
            market_mood='neutral'
        ))


class EnhancedGPTAgent:
    """강화된 GPT 에이전트 (뉴스 포함)"""

    def __init__(self, name: str, role: str, personality: str, expertise: str):
        self.name = name
        self.role = role
        self.personality = personality
        self.expertise = expertise
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def analyze(self, data: Dict, news: NewsContext, round_num: int = 1,
                other_opinions: List[AgentOpinion] = None) -> AgentOpinion:
        """데이터 + 뉴스 분석"""

        # 다른 에이전트 의견 요약
        debate_context = ""
        if other_opinions and round_num > 1:
            debate_context = "\n\nOTHER AGENTS' OPINIONS:\n"
            for op in other_opinions:
                debate_context += f"\n{op.agent_name} ({op.role}):\n"
                debate_context += f"- Stance: {op.stance} ({op.confidence}% confidence)\n"
                debate_context += f"- Recommendation: {op.recommended_action}\n"
                debate_context += f"- Key reasoning: {op.reasoning[:200]}...\n"

        prompt = f"""You are {self.name}, a {self.role}.

Your personality: {self.personality}
Your expertise: {self.expertise}

This is ROUND {round_num} of the debate.

FINANCIAL DATA:
{json.dumps(data, indent=2)}

RECENT NEWS CONTEXT:
- Market Mood: {news.market_mood}
- Sentiment: {news.sentiment}
- Key Headlines:
{chr(10).join(f"  • {h}" for h in news.headlines[:3])}
- Key Events:
{chr(10).join(f"  • {e}" for e in news.key_events)}

{debate_context}

Based on the data AND news context, provide your analysis in JSON format:
{{
  "stance": "BULLISH/BEARISH/NEUTRAL",
  "confidence": 0-100,
  "reasoning": "Your detailed reasoning here (consider both data AND news)",
  "key_points": ["point 1", "point 2", "point 3"],
  "recommended_action": "BUY/SELL/HOLD",
  "position_size": "AGGRESSIVE/MODERATE/CONSERVATIVE/NONE"{"," + chr(10) + '  "rebuttals": ["rebuttal to other agent 1", ...]' if round_num > 1 else ""}
}}

{"If this is round 2+, also provide rebuttals to other agents' opinions where you disagree." if round_num > 1 else ""}

Return ONLY the JSON object.
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are {self.name}, {self.role}. {self.personality}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # JSON 추출
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)

                return AgentOpinion(
                    agent_name=self.name,
                    role=self.role,
                    round_number=round_num,
                    stance=result.get('stance', 'NEUTRAL'),
                    confidence=result.get('confidence', 50),
                    reasoning=result.get('reasoning', ''),
                    key_points=result.get('key_points', []),
                    recommended_action=result.get('recommended_action', 'HOLD'),
                    position_size=result.get('position_size', 'NONE'),
                    rebuttals=result.get('rebuttals', [])
                )
            else:
                raise ValueError("No valid JSON in response")

        except Exception as e:
            print(f"⚠️  {self.name} analysis error: {e}")
            return AgentOpinion(
                agent_name=self.name,
                role=self.role,
                round_number=round_num,
                stance='NEUTRAL',
                confidence=0,
                reasoning=f"Error: {e}",
                key_points=[],
                recommended_action='HOLD',
                position_size='NONE'
            )


class EnhancedDebateSystem:
    """강화된 Multi-Agent 토론 시스템"""

    def __init__(self):
        # 6명의 에이전트 생성
        self.agents = [
            EnhancedGPTAgent(
                name="Dr. Sarah Chen",
                role="Conservative Fundamental Analyst",
                personality="Data-driven, risk-averse, long-term focus",
                expertise="Fundamental analysis, value investing, risk assessment"
            ),
            EnhancedGPTAgent(
                name="Alex Rivers",
                role="Aggressive Momentum Trader",
                personality="Opportunistic, high-risk tolerance, trend-following",
                expertise="Technical analysis, momentum trading, short-term patterns"
            ),
            EnhancedGPTAgent(
                name="Michael Foster",
                role="Risk Management Specialist",
                personality="Balanced, systematic, capital preservation",
                expertise="Risk management, portfolio allocation, drawdown control"
            ),
            EnhancedGPTAgent(
                name="Emily Zhang",
                role="Technical Chart Analyst",
                personality="Pattern-focused, data visualization expert",
                expertise="Chart patterns, support/resistance, volume analysis"
            ),
            EnhancedGPTAgent(
                name="Dr. Robert Klein",
                role="Macro Economist",
                personality="Big-picture thinker, fundamental macro analysis",
                expertise="Economic cycles, monetary policy, sector rotation"
            ),
            EnhancedGPTAgent(
                name="Jessica Martinez",
                role="Contrarian Investor",
                personality="Skeptical, challenges consensus, finds hidden value",
                expertise="Contrarian strategies, market psychology, crowd behavior"
            ),
        ]

        self.news_collector = SimpleNewsCollector()

    def conduct_enhanced_debate(self, ticker: str, analysis_data: Dict,
                                 num_rounds: int = 3) -> EnhancedDebateResult:
        """
        강화된 토론 진행 (다중 라운드)

        Args:
            ticker: 분석 대상
            analysis_data: 데이터
            num_rounds: 토론 라운드 수 (기본 3)

        Returns:
            강화된 토론 결과
        """
        print(f"\n{'='*70}")
        print(f"🎭 ENHANCED MULTI-AGENT DEBATE: {ticker}")
        print(f"{'='*70}\n")

        # 뉴스 수집
        print("📰 Collecting news context...")
        news = self.news_collector.get_ticker_news(ticker)

        print(f"\n📊 News Sentiment: {news.sentiment.upper()}")
        print(f"   Market Mood: {news.market_mood}")
        print(f"   Key Headlines:")
        for headline in news.headlines[:3]:
            print(f"   • {headline}")
        print()

        rounds = []
        all_opinions = []

        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*70}")
            print(f"🔄 ROUND {round_num}/{num_rounds}")
            print(f"{'='*70}\n")

            round_opinions = []

            # 이전 라운드 의견들
            previous_opinions = [op for op in all_opinions if op.round_number == round_num - 1]

            for agent in self.agents:
                print(f"🤖 {agent.name} ({agent.role})...")

                opinion = agent.analyze(
                    analysis_data,
                    news,
                    round_num,
                    previous_opinions if round_num > 1 else None
                )
                round_opinions.append(opinion)

                # 의견 출력
                stance_emoji = "📈" if opinion.stance == 'BULLISH' else "📉" if opinion.stance == 'BEARISH' else "➡️"
                action_emoji = "🟢" if opinion.recommended_action == 'BUY' else "🔴" if opinion.recommended_action == 'SELL' else "🟡"

                print(f"   {stance_emoji} {opinion.stance} ({opinion.confidence}%)")
                print(f"   {action_emoji} {opinion.recommended_action} ({opinion.position_size})")

                if round_num > 1 and opinion.rebuttals:
                    print(f"   💬 Rebuttals: {len(opinion.rebuttals)}")
                print()

            # 라운드 요약
            consensus_level = self._calculate_consensus(round_opinions)
            disagreements = self._identify_disagreements(round_opinions)

            round_result = DebateRound(
                round_number=round_num,
                opinions=round_opinions,
                consensus_level=consensus_level,
                main_disagreements=disagreements
            )
            rounds.append(round_result)
            all_opinions.extend(round_opinions)

            print(f"   📊 Consensus Level: {consensus_level*100:.1f}%")
            if disagreements:
                print(f"   ⚠️  Main Disagreements:")
                for dis in disagreements[:3]:
                    print(f"      • {dis}")

        # 최종 분석
        final_round = rounds[-1]
        consensus = self._reach_consensus(final_round.opinions)
        final_rec = self._final_recommendation(final_round.opinions)
        confidence = self._calculate_final_confidence(final_round.opinions, final_round.consensus_level)

        # 입장 변경 추적
        changed_minds = self._track_mind_changes(all_opinions)

        # 전체 트랜스크립트
        transcript = self._generate_transcript(rounds, news, consensus, final_rec)

        result = EnhancedDebateResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            news_context=news,
            rounds=rounds,
            final_consensus=consensus,
            final_recommendation=final_rec,
            confidence_score=confidence,
            agents_changed_mind=changed_minds,
            full_transcript=transcript
        )

        return result

    def _calculate_consensus(self, opinions: List[AgentOpinion]) -> float:
        """합의 수준 계산"""
        stances = [op.stance for op in opinions]
        max_count = max(stances.count(s) for s in set(stances))
        return max_count / len(opinions)

    def _identify_disagreements(self, opinions: List[AgentOpinion]) -> List[str]:
        """주요 이견 식별"""
        disagreements = []

        # 입장 차이
        stances = [op.stance for op in opinions]
        if 'BULLISH' in stances and 'BEARISH' in stances:
            disagreements.append("Fundamental disagreement on market direction")

        # 행동 차이
        actions = [op.recommended_action for op in opinions]
        if 'BUY' in actions and 'SELL' in actions:
            disagreements.append("Conflicting action recommendations")

        return disagreements

    def _reach_consensus(self, opinions: List[AgentOpinion]) -> Optional[str]:
        """최종 합의"""
        stances = [op.stance for op in opinions]
        for stance in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            if stances.count(stance) >= len(opinions) // 2 + 1:
                return stance
        return None

    def _final_recommendation(self, opinions: List[AgentOpinion]) -> str:
        """최종 권고"""
        actions = [op.recommended_action for op in opinions]
        for action in ['BUY', 'SELL', 'HOLD']:
            if actions.count(action) >= len(opinions) // 2 + 1:
                return action
        return 'HOLD'

    def _calculate_final_confidence(self, opinions: List[AgentOpinion], consensus_level: float) -> float:
        """최종 신뢰도"""
        avg_confidence = sum(op.confidence for op in opinions) / len(opinions)
        return avg_confidence * consensus_level

    def _track_mind_changes(self, all_opinions: List[AgentOpinion]) -> List[str]:
        """입장 변경 추적"""
        changed = []

        # 에이전트별로 그룹화
        by_agent = {}
        for op in all_opinions:
            if op.agent_name not in by_agent:
                by_agent[op.agent_name] = []
            by_agent[op.agent_name].append(op)

        # 입장 변경 확인
        for agent_name, ops in by_agent.items():
            if len(ops) < 2:
                continue

            initial_stance = ops[0].stance
            final_stance = ops[-1].stance

            if initial_stance != final_stance:
                changed.append(f"{agent_name}: {initial_stance} → {final_stance}")

        return changed

    def _generate_transcript(self, rounds: List[DebateRound], news: NewsContext,
                            consensus: Optional[str], final_rec: str) -> str:
        """전체 트랜스크립트 생성"""
        lines = []

        lines.append("="*70)
        lines.append("ENHANCED DEBATE TRANSCRIPT")
        lines.append("="*70)

        lines.append(f"\n📰 NEWS CONTEXT:")
        lines.append(f"   Sentiment: {news.sentiment}")
        lines.append(f"   Market Mood: {news.market_mood}")

        for i, round_result in enumerate(rounds, 1):
            lines.append(f"\n{'='*70}")
            lines.append(f"ROUND {i}")
            lines.append(f"{'='*70}")
            lines.append(f"Consensus Level: {round_result.consensus_level*100:.1f}%")

            for op in round_result.opinions:
                lines.append(f"\n{op.agent_name} ({op.role}):")
                lines.append(f"  Stance: {op.stance} ({op.confidence}%)")
                lines.append(f"  Action: {op.recommended_action}")
                lines.append(f"  Key Points:")
                for point in op.key_points:
                    lines.append(f"    • {point}")

                if op.rebuttals:
                    lines.append(f"  Rebuttals:")
                    for reb in op.rebuttals:
                        lines.append(f"    → {reb}")

        lines.append(f"\n{'='*70}")
        lines.append(f"FINAL CONSENSUS: {consensus or 'NO CONSENSUS'}")
        lines.append(f"FINAL RECOMMENDATION: {final_rec}")
        lines.append("="*70)

        return "\n".join(lines)

    def print_final_report(self, result: EnhancedDebateResult):
        """최종 리포트 출력"""
        print(f"\n{'='*70}")
        print(f"📊 ENHANCED DEBATE FINAL REPORT: {result.ticker}")
        print(f"{'='*70}")

        print(f"\n📰 News Context: {result.news_context.sentiment} ({result.news_context.market_mood})")
        print(f"🔄 Total Rounds: {len(result.rounds)}")
        print(f"🎯 Final Consensus: {result.final_consensus or 'NO CONSENSUS'}")
        print(f"📌 Final Recommendation: {result.final_recommendation}")
        print(f"💪 Confidence: {result.confidence_score:.1f}%")

        if result.agents_changed_mind:
            print(f"\n🔄 Agents Who Changed Their Mind:")
            for change in result.agents_changed_mind:
                print(f"   • {change}")

        print(f"\n{'='*70}")
        print("ROUND-BY-ROUND SUMMARY")
        print(f"{'='*70}")

        for round_result in result.rounds:
            print(f"\nRound {round_result.round_number}:")
            print(f"  Consensus: {round_result.consensus_level*100:.1f}%")

            # 입장 분포
            stances = [op.stance for op in round_result.opinions]
            print(f"  Stances: ", end="")
            for stance in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                count = stances.count(stance)
                if count > 0:
                    print(f"{stance}({count}) ", end="")
            print()

        print(f"\n{'='*70}\n")


def test_enhanced_debate():
    """강화된 토론 시스템 테스트"""
    from db_manager import DatabaseManager
    from deep_analysis import DeepDiveAnalyzer
    import pandas as pd

    print("="*70)
    print("🧪 Testing Enhanced Multi-Agent Debate System")
    print("="*70)

    db = DatabaseManager()
    debate_system = EnhancedDebateSystem()

    # SPY 로드
    spy = db.get_latest_market_data('SPY')
    spy_df = spy.set_index('date')[['close']]
    spy_df.columns = ['Close']
    spy_df.index = pd.to_datetime(spy_df.index)

    # 테스트 티커
    test_ticker = 'NVDA'

    # 데이터 로드
    df = db.get_latest_market_data(test_ticker)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.columns = [col.capitalize() for col in df.columns]

    # 심층 분석
    analyzer = DeepDiveAnalyzer(test_ticker, df, spy_df)
    trends = analyzer.multi_timeframe_analysis()
    sr_levels = analyzer.calculate_support_resistance()
    perf = analyzer.relative_performance()
    trade_idea = analyzer.generate_trade_idea()

    analysis_data = {
        "ticker": test_ticker,
        "current_price": df['Close'].iloc[-1],
        "trends": {
            tf: {"direction": trend.direction, "strength": trend.strength}
            for tf, trend in trends.items()
        },
        "support_resistance": [
            {"type": sr.level_type, "level": sr.level, "strength": sr.strength}
            for sr in sr_levels[:3]
        ],
        "relative_performance": perf,
    }

    # 강화된 토론 (3라운드)
    result = debate_system.conduct_enhanced_debate(test_ticker, analysis_data, num_rounds=3)

    # 결과 출력
    debate_system.print_final_report(result)

    # 트랜스크립트 저장
    output_file = f"outputs/debates/enhanced_debate_{test_ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("outputs/debates", exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(result.full_transcript)

    print(f"\n💾 Full transcript saved: {output_file}\n")


if __name__ == "__main__":
    test_enhanced_debate()
