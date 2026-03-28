"""メタ認知エンジン。エージェント自身の状態を観察し、戦略を自己調整する。

GoalQueueによる自律的ゴール生成・優先付けを実装。
LLMによる高品質なゴール提案と、好奇心駆動の探索を実現する。
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .agent_state import AgentState

if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory
    from .mistral_client import MistralClient


# ------------------------------------------------------------------
# GoalQueue: 自律的ゴール管理
# ------------------------------------------------------------------

@dataclass
class QueuedGoal:
    """優先度付きゴールキューのエントリ。"""
    goal: str
    priority_score: float   # 0.0〜1.0 (高いほど重要)
    source: str             # "meta_cognition" / "user" / "curiosity" / "priority_upgrade"
    rationale: str          # なぜこのゴールが重要か
    domain: str = "general"
    created_at: float = field(default_factory=time.time)
    estimated_value: float = 0.5   # 達成価値の推定 (0〜1)
    estimated_difficulty: float = 0.5  # 難易度の推定 (0〜1, 低いほど易しい)

    @property
    def composite_score(self) -> float:
        """優先度・価値・難易度を総合したスコア。"""
        return (self.priority_score * 0.5) + (self.estimated_value * 0.3) + ((1 - self.estimated_difficulty) * 0.2)


class GoalQueue:
    """自律的ゴールの優先度付きキュー。"""

    def __init__(self, max_size: int = 20) -> None:
        self._queue: List[QueuedGoal] = []
        self.max_size = max_size

    def add(self, goal: QueuedGoal) -> None:
        """ゴールを追加する。重複は追加しない。"""
        existing_goals = {g.goal for g in self._queue}
        if goal.goal in existing_goals:
            return
        self._queue.append(goal)
        # 総合スコアでソート
        self._queue.sort(key=lambda g: g.composite_score, reverse=True)
        # 最大サイズを超えたら低優先度を削除
        if len(self._queue) > self.max_size:
            self._queue = self._queue[:self.max_size]

    def pop_best(self) -> Optional[QueuedGoal]:
        """最高優先度のゴールを取り出す。"""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def peek_best(self) -> Optional[QueuedGoal]:
        """最高優先度のゴールを取り出さずに確認する。"""
        return self._queue[0] if self._queue else None

    def size(self) -> int:
        return len(self._queue)

    def to_list(self) -> List[QueuedGoal]:
        return list(self._queue)


_GOAL_GENERATION_PROMPT = """\
あなたはAGIエージェントのメタ認知システムです。
エージェントのセッション結果を分析し、次に追求すべきゴールを提案してください。

=== セッション分析 ===
達成したゴール: {completed_goal}
成功したステップ: {completed_steps}
失敗したステップ: {failed_steps}
観測メモ: {observations}
学習した事実: {learned_facts}
パフォーマンス: {performance}%

=== 指示 ===
以下の観点から次のゴールを2〜3個提案してください:
1. 今回のゴールの自然な発展・深化
2. 失敗から気づいた知識ギャップの補完
3. 学習した事実から派生する新しい探索

各ゴールに優先度スコア(0.0〜1.0)、達成価値(0.0〜1.0)、難易度(0.0〜1.0)を付けてください。

JSON配列のみで返答:
[
  {{
    "goal": "次のゴール (日本語)",
    "priority_score": 0.8,
    "rationale": "なぜ重要か (日本語)",
    "domain": "general/coding/research/writing/data/ops",
    "estimated_value": 0.7,
    "estimated_difficulty": 0.4
  }}
]\
"""

_CURIOSITY_PROMPT = """\
あなたはAGIエージェントの好奇心システムです。
以下の知識ギャップや未探索領域を特定し、探索すべきゴールを提案してください。

既知の情報:
{known_facts}

失敗パターン:
{failure_patterns}

世界モデルの状態:
{world_model_summary}

まだ理解できていないことや、探索する価値がある領域を特定し、
好奇心駆動のゴールを2個提案してください。

JSON配列のみで返答:
[
  {{"goal": "探索ゴール", "priority_score": 0.6, "rationale": "なぜ探索すべきか", "domain": "general"}}
]\
"""


class MetaCognition:
    """エージェントの実行状態を監視し、行き詰まりの検出・戦略転換・次ゴール提案を行う。

    GoalQueueによる自律的ゴール管理と、LLMによる高品質なゴール提案を実装。
    """

    STUCK_FAILURE_THRESHOLD = 3
    REPEATED_ERROR_THRESHOLD = 2

    def __init__(self, llm: Optional[MistralClient] = None) -> None:
        self.llm = llm
        self.goal_queue = GoalQueue()

    # ------------------------------------------------------------------
    # 行き詰まり検出
    # ------------------------------------------------------------------

    def is_stuck(self, state: AgentState) -> bool:
        """エージェントが行き詰まっているか判定する。"""
        if (
            len(state.failed_steps) >= self.STUCK_FAILURE_THRESHOLD
            and len(state.completed_steps) == 0
        ):
            return True

        error_history = state.working_memory.get("error_history", [])
        if len(error_history) >= self.REPEATED_ERROR_THRESHOLD:
            recent = error_history[-self.REPEATED_ERROR_THRESHOLD:]
            if len(set(recent)) == 1:
                return True

        return False

    # ------------------------------------------------------------------
    # パフォーマンス評価
    # ------------------------------------------------------------------

    def performance_score(self, state: AgentState) -> float:
        """成功率を 0.0〜1.0 で返す。"""
        total = len(state.completed_steps) + len(state.failed_steps)
        if total == 0:
            return 0.5
        return len(state.completed_steps) / total

    # ------------------------------------------------------------------
    # 戦略転換
    # ------------------------------------------------------------------

    def suggest_pivot(self, state: AgentState, memory: LongTermMemory) -> Optional[str]:
        """行き詰まり時に、別のアプローチを提案する。"""
        successful = memory.get_successful_strategies(limit=5)
        for s in successful:
            strategy = s.get("strategy", "")
            if strategy and strategy not in state.failed_steps and strategy not in state.completed_steps:
                return f"[メタ認知] 過去の成功戦略を参考: {strategy}"

        error_history = state.working_memory.get("error_history", [])
        if error_history:
            top_error = Counter(error_history).most_common(1)[0][0]
            pivot_map = {
                "missing_command": "CMD: which python3 && python3 --version && ls -la",
                "missing_file": "CMD: find . -maxdepth 3 -not -path '*/__pycache__/*' | sort | head -60",
                "permission_error": "CMD: ls -la && id",
                "missing_python_module": "CMD: python3 -m pip list | head -30",
                "connection_error": "CMD: ls -la && cat requirements.txt 2>/dev/null | head -20",
            }
            if top_error in pivot_map:
                return pivot_map[top_error]

        return "CMD: ls -la && find . -maxdepth 2 -not -path '*/__pycache__/*' | sort | head -40"

    # ------------------------------------------------------------------
    # 自律的ゴール生成 (LLMベース)
    # ------------------------------------------------------------------

    def generate_next_goal(self, state: AgentState, memory: LongTermMemory) -> Optional[str]:
        """現在のゴール達成後、次に取り組むべきゴールを自律的に提案する。

        LLMが利用可能な場合はLLMベースの高品質な提案、
        そうでなければルールベースの提案を行う。
        """
        if not state.is_done:
            return None

        # LLMベースの提案
        if self.llm is not None:
            self._llm_generate_goals(state, memory)

        # ルールベースのフォールバック
        self._rule_based_generate_goals(state, memory)

        # キューから最良のゴールを取得
        best = self.goal_queue.peek_best()
        if best:
            return best.goal

        return None

    def _llm_generate_goals(self, state: AgentState, memory: LongTermMemory) -> None:
        """LLMを使って次のゴールを生成しGoalQueueに追加する。"""
        assert self.llm is not None
        score = self.performance_score(state)
        learned_facts = state.working_memory.get("learned_facts", [])

        prompt = _GOAL_GENERATION_PROMPT.format(
            completed_goal=state.user_goal,
            completed_steps=", ".join(state.completed_steps[-5:]) or "なし",
            failed_steps=", ".join(state.failed_steps[-3:]) or "なし",
            observations="; ".join(state.observations[-3:]) or "なし",
            learned_facts="; ".join(learned_facts[-3:]) or "なし",
            performance=int(score * 100),
        )

        try:
            data = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("goal"):
                        self.goal_queue.add(QueuedGoal(
                            goal=item["goal"],
                            priority_score=float(item.get("priority_score", 0.5)),
                            source="meta_cognition",
                            rationale=item.get("rationale", ""),
                            domain=item.get("domain", "general"),
                            estimated_value=float(item.get("estimated_value", 0.5)),
                            estimated_difficulty=float(item.get("estimated_difficulty", 0.5)),
                        ))
        except Exception:
            pass

        # 好奇心駆動のゴール生成
        self._generate_curiosity_goals(state, memory)

    def _generate_curiosity_goals(self, state: AgentState, memory: LongTermMemory) -> None:
        """知識ギャップから好奇心駆動のゴールを生成する。"""
        assert self.llm is not None

        known_facts = memory.recall_recent(limit=5)
        facts_text = "\n".join(f"- {f['key']}: {f['value'][:50]}" for f in known_facts) or "なし"

        failures = memory.get_known_failures(limit=5)
        failures_text = "\n".join(
            f"- {f['command_pattern'][:40]}: {f['error_type']}({f['count']}回)"
            for f in failures
        ) or "なし"

        world_model = getattr(state, "world_model", None)
        wm_summary = world_model.summary() if world_model else "未初期化"

        prompt = _CURIOSITY_PROMPT.format(
            known_facts=facts_text,
            failure_patterns=failures_text,
            world_model_summary=wm_summary,
        )

        try:
            data = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=512,
            )
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("goal"):
                        self.goal_queue.add(QueuedGoal(
                            goal=item["goal"],
                            priority_score=float(item.get("priority_score", 0.4)),
                            source="curiosity",
                            rationale=item.get("rationale", "好奇心駆動探索"),
                            domain=item.get("domain", "general"),
                        ))
        except Exception:
            pass

    def _rule_based_generate_goals(self, state: AgentState, memory: LongTermMemory) -> None:
        """ルールベースのゴール生成 (LLMフォールバック)。"""
        # priority_upgradesから
        priority_upgrades = state.working_memory.get("priority_upgrades", [])
        for i, upgrade in enumerate(priority_upgrades):
            self.goal_queue.add(QueuedGoal(
                goal=upgrade,
                priority_score=0.8 - i * 0.1,
                source="priority_upgrade",
                rationale="前回セッションで特定された改善候補",
            ))

        # 長期記憶から
        recent = memory.recall_recent(limit=10)
        for item in recent:
            if item["key"].startswith("priority_upgrade_"):
                self.goal_queue.add(QueuedGoal(
                    goal=item["value"],
                    priority_score=0.6,
                    source="long_term_memory",
                    rationale="長期記憶からの改善候補",
                ))

    def get_queued_goals(self) -> List[QueuedGoal]:
        """キュー内のゴール一覧を返す。"""
        return self.goal_queue.to_list()

    # ------------------------------------------------------------------
    # 自己振り返り
    # ------------------------------------------------------------------

    def reflection_summary(self, state: AgentState) -> str:
        """セッション終了時の自己振り返りサマリを生成する。"""
        score = self.performance_score(state)
        lines = [
            "[メタ認知レポート]",
            f"パフォーマンス: {score:.0%}  ({len(state.completed_steps)} 成功 / {len(state.failed_steps)} 失敗)",
        ]

        error_history = state.working_memory.get("error_history", [])
        if error_history:
            top_errors = Counter(error_history).most_common(2)
            lines.append("主要エラー: " + ", ".join(f"{e}({c}回)" for e, c in top_errors))

        # CoT推論のサマリー
        cot = state.working_memory.get("last_cot_reasoning")
        if cot:
            lines.append(f"最後の推論: {cot[:80]}...")

        # 信頼度の低いステップへの警告
        warnings = state.working_memory.get("confidence_warnings", [])
        if warnings:
            lines.append(f"信頼度警告: {len(warnings)}件")

        queued = self.goal_queue.size()
        if queued > 0:
            best = self.goal_queue.peek_best()
            if best:
                lines.append(f"次の推奨ゴール [{best.source}]: {best.goal}")
                lines.append(f"  理由: {best.rationale}")
            lines.append(f"ゴールキュー: {queued}件待機中")

        next_goal = state.working_memory.get("suggested_next_goal")
        if next_goal:
            lines.append(f"次の推奨ゴール: {next_goal}")
        elif state.working_memory.get("priority_upgrades"):
            lines.append(f"次の改善候補: {state.working_memory['priority_upgrades'][0]}")

        return "\n".join(lines)
