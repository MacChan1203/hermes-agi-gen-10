"""自己省察エンジン: AGIの「考える時間」を実装する。

タスク実行の間に能動的な反省フェーズを挟むことで、
システムが自分自身の認知パターン・知識ギャップ・成長軌跡を把握できるようにする。

AGI的観点: 単に経験を記録するだけでなく、経験から「洞察」を抽出し、
それを次の行動・目標生成に活かす高次メタ認知ループ。

使い方:
    engine = ReflectionEngine(llm=llm)
    insights = engine.reflect(ltm)
    new_goals = engine.generate_strategic_goals(insights, ltm)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory
    from .meta_cognition import QueuedGoal
    from .mistral_client import MistralClient


# ------------------------------------------------------------------
# データクラス
# ------------------------------------------------------------------

@dataclass
class Insight:
    """反省フェーズで抽出された洞察。"""
    category: str        # "strength" / "weakness" / "gap" / "pattern" / "opportunity"
    content: str         # 洞察の内容
    confidence: float    # 確信度 (0.0〜1.0)
    source: str          # 洞察の根拠 (LTMキー・パターン説明)
    actionable: bool     # 行動につながるか
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "content": self.content,
            "confidence": self.confidence,
            "source": self.source,
            "actionable": self.actionable,
            "timestamp": self.timestamp,
        }


@dataclass
class GrowthMetrics:
    """AGIの成長指標。"""
    total_sessions: int = 0
    success_rate: float = 0.0
    avg_iterations: float = 0.0
    knowledge_breadth: int = 0    # LTMのユニークキー数
    strategy_diversity: int = 0   # 記録された戦略の多様性
    prediction_accuracy: float = 0.0
    reflection_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"セッション={self.total_sessions} 成功率={self.success_rate:.0%} "
            f"知識={self.knowledge_breadth}件 反省回数={self.reflection_count}"
        )


# ------------------------------------------------------------------
# プロンプト
# ------------------------------------------------------------------

_REFLECTION_PROMPT = """\
あなたはAGIシステムの自己省察エンジンです。
過去のセッション履歴・成功/失敗パターン・知識ベースを分析し、
システムの強み・弱み・知識ギャップ・改善機会を特定してください。

=== 分析データ ===
成功した戦略 (上位5件):
{successful_strategies}

失敗したステップ (上位5件):
{failed_steps}

最近の学習事実 (上位10件):
{recent_facts}

成長指標:
{growth_metrics}

=== 指示 ===
以下のJSON形式で洞察を返してください:
{{
  "insights": [
    {{
      "category": "strength|weakness|gap|pattern|opportunity",
      "content": "洞察の内容 (日本語、具体的に)",
      "confidence": 0.0〜1.0,
      "actionable": true|false
    }}
  ],
  "summary": "全体的な自己評価 (1〜2文)",
  "priority_focus": "次に最も重要な改善点"
}}

最大5個の洞察を返してください。具体的で実行可能なものを優先してください。
"""

_STRATEGIC_GOALS_PROMPT = """\
あなたはAGIシステムの戦略的目標生成エンジンです。
以下の洞察に基づいて、AGIシステムが自律的に取り組むべき目標を提案してください。

=== 洞察 ===
{insights}

=== 現在の知識ギャップ ===
{knowledge_gaps}

=== 指示 ===
以下のJSON形式で目標を返してください:
{{
  "goals": [
    {{
      "goal": "具体的な目標 (日本語)",
      "rationale": "なぜこの目標が重要か",
      "priority": 0.0〜1.0,
      "domain": "general|coding|analysis|learning|self_improvement",
      "estimated_value": 0.0〜1.0,
      "estimated_difficulty": 0.0〜1.0
    }}
  ]
}}

最大3個の目標を返してください。実行可能で具体的なものにしてください。
"""


# ------------------------------------------------------------------
# ReflectionEngine
# ------------------------------------------------------------------

class ReflectionEngine:
    """AGIの能動的自己省察エンジン。

    タスク完了後や定期的に呼び出し、LTMを分析して洞察を生成する。
    生成された洞察はGoalQueueへの戦略的目標として変換される。

    Args:
        llm: MistralClient インスタンス (None の場合はルールベースの省察のみ)
        reflection_interval: 何ゴールごとに省察を行うか
    """

    _LTM_REFLECTION_KEY = "reflection_history_v1"
    _LTM_GROWTH_KEY = "growth_metrics_v1"

    def __init__(
        self,
        llm: Optional[Any] = None,
        reflection_interval: int = 5,
    ) -> None:
        self.llm = llm
        self.reflection_interval = reflection_interval
        self._goal_counter: int = 0
        self._last_reflection_time: float = 0.0
        self._reflection_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public: 省察サイクル
    # ------------------------------------------------------------------

    def should_reflect(self) -> bool:
        """省察を行うべきタイミングかどうか。"""
        self._goal_counter += 1
        return self._goal_counter % self.reflection_interval == 0

    def reflect(self, ltm: "LongTermMemory") -> List[Insight]:
        """LTMを分析して洞察リストを生成する。

        Returns:
            Insight のリスト
        """
        print("[ReflectionEngine] 自己省察を開始...", flush=True)

        # LTMからデータを収集
        strategies = ltm.recall_strategies("", limit=20)
        failures = ltm.get_known_failures(limit=10)
        recent_facts = ltm.recall_recent(limit=15)

        # ルールベースの省察（LLM不要）
        rule_insights = self._rule_based_reflection(strategies, failures, recent_facts)

        # LLMによる深層省察
        llm_insights: List[Insight] = []
        if self.llm is not None:
            try:
                llm_insights = self._llm_reflection(strategies, failures, recent_facts, ltm)
            except Exception as e:
                print(f"[ReflectionEngine] LLM省察エラー: {e}", flush=True)

        insights = rule_insights + llm_insights
        self._last_reflection_time = time.time()

        # LTMに省察記録を保存
        self._save_reflection(ltm, insights)

        print(
            f"[ReflectionEngine] 省察完了: {len(insights)}件の洞察を生成",
            flush=True,
        )
        return insights

    def generate_strategic_goals(
        self,
        insights: List[Insight],
        ltm: "LongTermMemory",
    ) -> List["QueuedGoal"]:
        """洞察から戦略的目標を生成する。"""
        from .meta_cognition import QueuedGoal

        # ルールベースのゴール生成（常に実行）
        rule_goals = self._rule_based_goal_generation(insights, ltm)

        # LLMによるゴール生成
        llm_goals: List[QueuedGoal] = []
        if self.llm is not None and insights:
            try:
                llm_goals = self._llm_goal_generation(insights, ltm)
            except Exception as e:
                print(f"[ReflectionEngine] LLMゴール生成エラー: {e}", flush=True)

        all_goals = rule_goals + llm_goals
        # 重複排除
        seen = set()
        unique_goals = []
        for g in all_goals:
            if g.goal not in seen:
                seen.add(g.goal)
                unique_goals.append(g)

        return unique_goals[:5]  # 最大5件

    def compute_growth_metrics(self, ltm: "LongTermMemory") -> GrowthMetrics:
        """AGIの成長指標を計算する。"""
        strategies = ltm.recall_strategies("", limit=100)
        failures = ltm.get_known_failures(limit=100)
        recent = ltm.recall_recent(limit=200)

        success_count = sum(1 for s in strategies if s.get("outcome") == "success")
        fail_count = len(failures)
        total = success_count + fail_count

        # 過去の省察回数
        reflection_count = sum(
            1 for f in recent if f.get("key", "").startswith(self._LTM_REFLECTION_KEY)
        )

        metrics = GrowthMetrics(
            total_sessions=total,
            success_rate=success_count / total if total > 0 else 0.0,
            knowledge_breadth=len(recent),
            strategy_diversity=len({s.get("step", "")[:30] for s in strategies}),
            reflection_count=reflection_count,
        )
        return metrics

    def identify_knowledge_gaps(self, ltm: "LongTermMemory") -> List[str]:
        """知識ギャップを特定する（失敗パターンから）。"""
        failures = ltm.get_known_failures(limit=20)
        gaps: List[str] = []

        # 繰り返しパターンの検出
        error_types: Dict[str, int] = {}
        for f in failures:
            err_type = f.get("error_type", "unknown")
            error_types[err_type] = error_types.get(err_type, 0) + 1

        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            if count >= 2:
                gaps.append(f"{err_type}エラーが{count}回発生（知識ギャップの可能性）")

        # 試みたが失敗した領域
        failed_steps = [f.get("step", "")[:50] for f in failures[:5]]
        for step in failed_steps:
            if step:
                gaps.append(f"未解決: {step}")

        return gaps[:5]

    # ------------------------------------------------------------------
    # ルールベース省察
    # ------------------------------------------------------------------

    def _rule_based_reflection(
        self,
        strategies: List[Dict],
        failures: List[Dict],
        facts: List[Dict],
    ) -> List[Insight]:
        """LLM不要のルールベース省察。"""
        insights: List[Insight] = []

        # 成功率の分析
        success_count = sum(1 for s in strategies if s.get("outcome") == "success")
        total = len(strategies)
        if total > 0:
            rate = success_count / total
            if rate >= 0.7:
                insights.append(Insight(
                    category="strength",
                    content=f"成功率が高い ({rate:.0%}) — 戦略が効果的に機能している",
                    confidence=0.8,
                    source="strategy_analysis",
                    actionable=False,
                ))
            elif rate < 0.4:
                insights.append(Insight(
                    category="weakness",
                    content=f"成功率が低い ({rate:.0%}) — 戦略の見直しが必要",
                    confidence=0.8,
                    source="strategy_analysis",
                    actionable=True,
                ))

        # 繰り返しエラーパターンの検出
        error_types: Dict[str, int] = {}
        for f in failures:
            err_type = f.get("error_type", "unknown")
            error_types[err_type] = error_types.get(err_type, 0) + 1

        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:2]:
            if count >= 2:
                insights.append(Insight(
                    category="gap",
                    content=f"「{err_type}」エラーが繰り返し発生 ({count}回) — 対策が必要",
                    confidence=0.9,
                    source=f"failure_pattern:{err_type}",
                    actionable=True,
                ))

        # 知識ベースの成長
        if len(facts) > 50:
            insights.append(Insight(
                category="strength",
                content=f"豊富な経験データを保有 ({len(facts)}件) — 過去の知識を活用できる",
                confidence=0.7,
                source="knowledge_base",
                actionable=False,
            ))
        elif len(facts) < 10:
            insights.append(Insight(
                category="opportunity",
                content="経験データが少ない — 積極的な探索で学習を加速できる",
                confidence=0.7,
                source="knowledge_base",
                actionable=True,
            ))

        return insights

    # ------------------------------------------------------------------
    # LLMベース省察
    # ------------------------------------------------------------------

    def _llm_reflection(
        self,
        strategies: List[Dict],
        failures: List[Dict],
        facts: List[Dict],
        ltm: "LongTermMemory",
    ) -> List[Insight]:
        """LLMを使った深層省察。"""
        metrics = self.compute_growth_metrics(ltm)

        def fmt_list(items: List[Dict], key: str, limit: int = 5) -> str:
            lines = []
            for item in items[:limit]:
                val = item.get(key, "")
                if val:
                    lines.append(f"- {str(val)[:80]}")
            return "\n".join(lines) if lines else "（なし）"

        prompt = _REFLECTION_PROMPT.format(
            successful_strategies=fmt_list(
                [s for s in strategies if s.get("outcome") == "success"], "step"
            ),
            failed_steps=fmt_list(failures, "step"),
            recent_facts=fmt_list(facts, "value"),
            growth_metrics=metrics.summary(),
        )

        try:
            data = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception:
            return []

        if not isinstance(data, dict):
            return []

        insights = []
        for item in data.get("insights", [])[:5]:
            if not isinstance(item, dict):
                continue
            insights.append(Insight(
                category=item.get("category", "pattern"),
                content=item.get("content", ""),
                confidence=float(item.get("confidence", 0.5)),
                source="llm_reflection",
                actionable=bool(item.get("actionable", False)),
            ))

        return insights

    # ------------------------------------------------------------------
    # ゴール生成
    # ------------------------------------------------------------------

    def _rule_based_goal_generation(
        self,
        insights: List[Insight],
        ltm: "LongTermMemory",
    ) -> List["QueuedGoal"]:
        """ルールベースのゴール生成。"""
        from .meta_cognition import QueuedGoal

        goals: List[QueuedGoal] = []

        for insight in insights:
            if not insight.actionable:
                continue

            if insight.category == "gap":
                goals.append(QueuedGoal(
                    goal=f"繰り返しエラーへの対策を調査・実装する: {insight.content[:60]}",
                    priority_score=0.7,
                    source="reflection",
                    rationale=insight.content,
                    domain="self_improvement",
                    estimated_value=0.8,
                    estimated_difficulty=0.5,
                ))
            elif insight.category == "weakness":
                goals.append(QueuedGoal(
                    goal=f"弱点を改善するための戦略を立案する: {insight.content[:60]}",
                    priority_score=0.6,
                    source="reflection",
                    rationale=insight.content,
                    domain="self_improvement",
                    estimated_value=0.7,
                    estimated_difficulty=0.6,
                ))
            elif insight.category == "opportunity":
                goals.append(QueuedGoal(
                    goal=f"改善機会を活かす: {insight.content[:60]}",
                    priority_score=0.5,
                    source="reflection",
                    rationale=insight.content,
                    domain="general",
                    estimated_value=0.6,
                    estimated_difficulty=0.4,
                ))

        return goals

    def _llm_goal_generation(
        self,
        insights: List[Insight],
        ltm: "LongTermMemory",
    ) -> List["QueuedGoal"]:
        """LLMを使った戦略的ゴール生成。"""
        from .meta_cognition import QueuedGoal

        insights_text = "\n".join(
            f"- [{i.category}] {i.content}" for i in insights
        )
        gaps = self.identify_knowledge_gaps(ltm)
        gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "（特定されたギャップなし）"

        prompt = _STRATEGIC_GOALS_PROMPT.format(
            insights=insights_text,
            knowledge_gaps=gaps_text,
        )

        try:
            data = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )
        except Exception:
            return []

        if not isinstance(data, dict):
            return []

        goals = []
        for item in data.get("goals", [])[:3]:
            if not isinstance(item, dict) or not item.get("goal"):
                continue
            goals.append(QueuedGoal(
                goal=item["goal"],
                priority_score=float(item.get("priority", 0.5)),
                source="strategic_reflection",
                rationale=item.get("rationale", ""),
                domain=item.get("domain", "general"),
                estimated_value=float(item.get("estimated_value", 0.5)),
                estimated_difficulty=float(item.get("estimated_difficulty", 0.5)),
            ))

        return goals

    # ------------------------------------------------------------------
    # LTMへの省察記録
    # ------------------------------------------------------------------

    def _save_reflection(self, ltm: "LongTermMemory", insights: List[Insight]) -> None:
        """省察結果をLTMに保存する。"""
        key = f"{self._LTM_REFLECTION_KEY}_{int(time.time())}"
        data = {
            "timestamp": time.time(),
            "insight_count": len(insights),
            "insights": [i.to_dict() for i in insights[:3]],  # 上位3件のみ保存
        }
        ltm.learn(key, json.dumps(data, ensure_ascii=False))

    def get_recent_reflections(self, ltm: "LongTermMemory", limit: int = 3) -> List[Dict]:
        """最近の省察記録を取得する。"""
        facts = ltm.recall_recent(limit=100)
        reflections = []
        for f in facts:
            if f.get("key", "").startswith(self._LTM_REFLECTION_KEY):
                try:
                    data = json.loads(f["value"])
                    reflections.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
        return reflections[:limit]

    def summary(self) -> str:
        """省察エンジンの状態サマリー。"""
        elapsed = time.time() - self._last_reflection_time if self._last_reflection_time else None
        if elapsed:
            return f"前回省察: {int(elapsed)}秒前 | ゴールカウンタ: {self._goal_counter}"
        return f"未省察 | ゴールカウンタ: {self._goal_counter}"
