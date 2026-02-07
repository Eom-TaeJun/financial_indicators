#!/usr/bin/env python3
"""
Multi-Agent AI Debate System
여러 AI 에이전트가 금융 데이터를 분석하고 토론
"""

import os
import anthropic
import openai
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentOpinion:
    """에이전트 의견"""
    agent_name: str
    role: str
    stance: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: int  # 0-100
    reasoning: str
    key_points: List[str]
    recommended_action: str  # 'BUY', 'SELL', 'HOLD'
    position_size: str  # 'AGGRESSIVE', 'MODERATE', 'CONSERVATIVE', 'NONE'


@dataclass
class DebateResult:
    """토론 결과"""
    ticker: str
    timestamp: str
    agents: List[AgentOpinion]
    consensus: Optional[str]
    final_recommendation: str
    confidence_score: float
    debate_summary: str


class AIAgent:
    """AI 에이전트 베이스 클래스"""

    def __init__(self, name: str, role: str, personality: str):
        self.name = name
        self.role = role
        self.personality = personality

    def analyze(self, data: Dict, context: str = "") -> AgentOpinion:
        """데이터 분석 (서브클래스에서 구현)"""
        raise NotImplementedError


class ClaudeAgent(AIAgent):
    """Claude 기반 에이전트"""

    def __init__(self, name: str, role: str, personality: str):
        super().__init__(name, role, personality)
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def analyze(self, data: Dict, context: str = "") -> AgentOpinion:
        """Claude로 데이터 분석"""

        prompt = f"""You are {self.name}, a {self.role}.

Your personality: {self.personality}

Analyze the following financial data and provide your opinion:

{json.dumps(data, indent=2)}

{context}

Based on this data, provide your analysis in the following JSON format:
{{
  "stance": "BULLISH/BEARISH/NEUTRAL",
  "confidence": 0-100,
  "reasoning": "Your detailed reasoning here",
  "key_points": ["point 1", "point 2", "point 3"],
  "recommended_action": "BUY/SELL/HOLD",
  "position_size": "AGGRESSIVE/MODERATE/CONSERVATIVE/NONE"
}}

Be specific and data-driven. Return ONLY the JSON object.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text

            # JSON 추출
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)

                return AgentOpinion(
                    agent_name=self.name,
                    role=self.role,
                    stance=result.get('stance', 'NEUTRAL'),
                    confidence=result.get('confidence', 50),
                    reasoning=result.get('reasoning', ''),
                    key_points=result.get('key_points', []),
                    recommended_action=result.get('recommended_action', 'HOLD'),
                    position_size=result.get('position_size', 'NONE')
                )
            else:
                raise ValueError("No valid JSON in response")

        except Exception as e:
            print(f"⚠️  {self.name} analysis error: {e}")
            return AgentOpinion(
                agent_name=self.name,
                role=self.role,
                stance='NEUTRAL',
                confidence=0,
                reasoning=f"Error: {e}",
                key_points=[],
                recommended_action='HOLD',
                position_size='NONE'
            )


class GPTAgent(AIAgent):
    """GPT 기반 에이전트"""

    def __init__(self, name: str, role: str, personality: str):
        super().__init__(name, role, personality)
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def analyze(self, data: Dict, context: str = "") -> AgentOpinion:
        """GPT로 데이터 분석"""

        prompt = f"""You are {self.name}, a {self.role}.

Your personality: {self.personality}

Analyze the following financial data and provide your opinion:

{json.dumps(data, indent=2)}

{context}

Based on this data, provide your analysis in the following JSON format:
{{
  "stance": "BULLISH/BEARISH/NEUTRAL",
  "confidence": 0-100,
  "reasoning": "Your detailed reasoning here",
  "key_points": ["point 1", "point 2", "point 3"],
  "recommended_action": "BUY/SELL/HOLD",
  "position_size": "AGGRESSIVE/MODERATE/CONSERVATIVE/NONE"
}}

Be specific and data-driven. Return ONLY the JSON object.
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
                    stance=result.get('stance', 'NEUTRAL'),
                    confidence=result.get('confidence', 50),
                    reasoning=result.get('reasoning', ''),
                    key_points=result.get('key_points', []),
                    recommended_action=result.get('recommended_action', 'HOLD'),
                    position_size=result.get('position_size', 'NONE')
                )
            else:
                raise ValueError("No valid JSON in response")

        except Exception as e:
            print(f"⚠️  {self.name} analysis error: {e}")
            return AgentOpinion(
                agent_name=self.name,
                role=self.role,
                stance='NEUTRAL',
                confidence=0,
                reasoning=f"Error: {e}",
                key_points=[],
                recommended_action='HOLD',
                position_size='NONE'
            )


class MultiAgentDebateSystem:
    """Multi-Agent 토론 시스템"""

    def __init__(self):
        # 에이전트 생성 (모두 GPT 사용, 다른 성격)
        self.agents = [
            GPTAgent(
                name="Dr. Sarah Chen",
                role="Conservative Fundamental Analyst",
                personality="Data-driven, risk-averse, focuses on fundamentals and long-term value. Prefers defensive positions during uncertainty. Always asks 'what could go wrong?'"
            ),
            GPTAgent(
                name="Alex Rivers",
                role="Aggressive Momentum Trader",
                personality="Opportunistic, high-risk tolerance, focuses on technical patterns and market momentum. Quick to capitalize on trends. Believes in riding winners."
            ),
            GPTAgent(
                name="Michael Foster",
                role="Risk Management Specialist",
                personality="Balanced, systematic, focuses on risk-adjusted returns and portfolio protection. Always considers worst-case scenarios. Prioritizes capital preservation."
            ),
        ]

    def conduct_debate(self, ticker: str, analysis_data: Dict) -> DebateResult:
        """
        토론 진행

        Args:
            ticker: 분석 대상 티커
            analysis_data: 분석 데이터 (가격, 지표, 트렌드 등)

        Returns:
            토론 결과
        """
        print(f"\n{'='*70}")
        print(f"🎭 MULTI-AGENT DEBATE: {ticker}")
        print(f"{'='*70}\n")

        # 각 에이전트 의견 수집
        opinions = []

        for agent in self.agents:
            print(f"🤖 {agent.name} ({agent.role}) is analyzing...")

            opinion = agent.analyze(analysis_data)
            opinions.append(opinion)

            # 의견 출력
            stance_emoji = "📈" if opinion.stance == 'BULLISH' else "📉" if opinion.stance == 'BEARISH' else "➡️"
            action_emoji = "🟢" if opinion.recommended_action == 'BUY' else "🔴" if opinion.recommended_action == 'SELL' else "🟡"

            print(f"   {stance_emoji} Stance: {opinion.stance} (Confidence: {opinion.confidence}%)")
            print(f"   {action_emoji} Recommendation: {opinion.recommended_action} ({opinion.position_size})")
            print(f"   💭 Reasoning: {opinion.reasoning[:150]}...")
            print()

        # 합의 도출
        consensus = self._reach_consensus(opinions)
        final_recommendation = self._final_recommendation(opinions)
        confidence_score = self._calculate_confidence(opinions)

        # 토론 요약
        debate_summary = self._generate_summary(opinions, consensus, final_recommendation)

        result = DebateResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            agents=opinions,
            consensus=consensus,
            final_recommendation=final_recommendation,
            confidence_score=confidence_score,
            debate_summary=debate_summary
        )

        return result

    def _reach_consensus(self, opinions: List[AgentOpinion]) -> Optional[str]:
        """합의 도출"""
        stances = [op.stance for op in opinions]

        # 과반수 합의
        for stance in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            if stances.count(stance) >= 2:
                return stance

        return None  # 합의 실패

    def _final_recommendation(self, opinions: List[AgentOpinion]) -> str:
        """최종 권고"""
        actions = [op.recommended_action for op in opinions]

        # 다수결
        for action in ['BUY', 'SELL', 'HOLD']:
            if actions.count(action) >= 2:
                return action

        # 동률이면 가장 보수적인 선택
        return 'HOLD'

    def _calculate_confidence(self, opinions: List[AgentOpinion]) -> float:
        """신뢰도 계산"""
        # 평균 신뢰도
        avg_confidence = sum(op.confidence for op in opinions) / len(opinions)

        # 합의 여부에 따라 가중치
        stances = [op.stance for op in opinions]
        if len(set(stances)) == 1:
            # 만장일치
            return avg_confidence * 1.2
        elif len(set(stances)) == 2:
            # 과반수 합의
            return avg_confidence
        else:
            # 의견 분산
            return avg_confidence * 0.8

    def _generate_summary(self, opinions: List[AgentOpinion],
                          consensus: Optional[str], final_rec: str) -> str:
        """토론 요약 생성"""
        summary = []

        summary.append("DEBATE SUMMARY")
        summary.append("=" * 70)

        # 합의 상태
        if consensus:
            summary.append(f"\n✅ Consensus Reached: {consensus}")
        else:
            summary.append(f"\n⚠️  No Consensus (Split Opinion)")

        summary.append(f"📊 Final Recommendation: {final_rec}\n")

        # 각 에이전트 핵심 포인트
        for opinion in opinions:
            summary.append(f"\n{opinion.agent_name}:")
            for point in opinion.key_points:
                summary.append(f"  • {point}")

        return "\n".join(summary)

    def print_report(self, result: DebateResult):
        """리포트 출력"""
        print(f"\n{'='*70}")
        print(f"📊 FINAL DEBATE REPORT: {result.ticker}")
        print(f"{'='*70}")
        print(f"Timestamp: {result.timestamp}")
        print(f"\n{result.debate_summary}")

        print(f"\n{'='*70}")
        print(f"🎯 FINAL DECISION")
        print(f"{'='*70}")

        action_emoji = "🟢" if result.final_recommendation == 'BUY' else "🔴" if result.final_recommendation == 'SELL' else "🟡"
        print(f"{action_emoji} Recommendation: {result.final_recommendation}")
        print(f"💪 Confidence: {result.confidence_score:.1f}%")

        if result.consensus:
            print(f"✅ Team Consensus: {result.consensus}")
        else:
            print(f"⚠️  Split Decision - Exercise Caution")

        print(f"\n{'='*70}\n")


def test_multi_agent_debate():
    """Multi-Agent 토론 테스트"""
    from db_manager import DatabaseManager
    from deep_analysis import DeepDiveAnalyzer

    print("="*70)
    print("🧪 Testing Multi-Agent Debate System")
    print("="*70)

    db = DatabaseManager()
    debate_system = MultiAgentDebateSystem()

    # SPY (시장 지수) 로드
    spy = db.get_latest_market_data('SPY')
    spy_df = spy.set_index('date')[['close']]
    spy_df.columns = ['Close']
    spy_df.index = pd.to_datetime(spy_df.index)

    # 테스트 대상
    test_ticker = 'NVDA'

    # 데이터 로드
    df = db.get_latest_market_data(test_ticker)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.columns = [col.capitalize() for col in df.columns]

    # 심층 분석
    analyzer = DeepDiveAnalyzer(test_ticker, df, spy_df)

    # 데이터 준비
    trends = analyzer.multi_timeframe_analysis()
    sr_levels = analyzer.calculate_support_resistance()
    perf = analyzer.relative_performance()
    trade_idea = analyzer.generate_trade_idea()

    analysis_data = {
        "ticker": test_ticker,
        "current_price": df['Close'].iloc[-1],
        "trends": {
            tf: {
                "direction": trend.direction,
                "strength": trend.strength
            }
            for tf, trend in trends.items()
        },
        "support_resistance": [
            {
                "type": sr.level_type,
                "level": sr.level,
                "strength": sr.strength
            }
            for sr in sr_levels[:3]
        ],
        "relative_performance": perf,
        "trade_idea": {
            "action": trade_idea.action,
            "confidence": trade_idea.confidence,
            "entry": trade_idea.entry_price,
            "stop_loss": trade_idea.stop_loss,
            "target": trade_idea.target_1,
            "rationale": trade_idea.rationale
        }
    }

    # 토론 진행
    result = debate_system.conduct_debate(test_ticker, analysis_data)

    # 결과 출력
    debate_system.print_report(result)


if __name__ == "__main__":
    import pandas as pd
    test_multi_agent_debate()
