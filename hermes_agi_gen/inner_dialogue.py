"""内部対話システム: 認知ロール間の事前議論と合意形成。

実行前に複数の認知ロール（批判者・革新者・倫理家・戦略家）が
ゴールについて内部議論を行い、より堅牢な計画を生成する。

AGI的観点:
- 単一の推論パスではなく、複数の視点からの検討
- 自己批判と創造的代替案の生成
- 倫理的配慮の事前チェック
- 不確実性が高い場合のみ発動（資源効率）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .consciousness import GlobalWorkspace, WorkspaceSignal
    from .mistral_client import MistralClient
    from .value_system import ValueAssessment


# ------------------------------------------------------------------
# 対話データ構造
# ------------------------------------------------------------------

@dataclass
class DialogueUtterance:
    """内部対話の一発言。"""
    role: str        # "critic" / "innovator" / "ethicist" / "strategist"
    content: str     # 発言内容
    stance: str      # "support" / "oppose" / "qualify" / "extend"
    confidence: float  # 確信度 (0〜1)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeliberationResult:
    """内部対話の結論。"""
    original_goal: str
    refined_goal: str          # 議論後の洗練されたゴール
    consensus_level: float     # 合意度 (0〜1)
    key_concerns: List[str]    # 主要な懸念事項
    suggested_approach: str    # 推奨アプローチ
    utterances: List[DialogueUtterance] = field(default_factory=list)
    should_proceed: bool = True
    deliberation_time: float = 0.0


# ------------------------------------------------------------------
# 対話プロンプト
# ------------------------------------------------------------------

_ROLE_PROMPTS = {
    "critic": """\
あなたはAGIの「批判者」ロールです。以下のゴールの潜在的リスク・欠陥・
見落としを指摘してください。建設的な批判を心がけてください。

ゴール: {goal}
コンテキスト: {context}

JSON形式で回答: {{"criticism": "批判内容", "risks": ["リスク1", "リスク2"], "stance": "oppose|qualify", "confidence": 0.0-1.0}}""",

    "innovator": """\
あなたはAGIの「革新者」ロールです。以下のゴールに対する創造的な
代替アプローチや改善案を提案してください。

ゴール: {goal}
コンテキスト: {context}

JSON形式で回答: {{"innovation": "提案内容", "alternatives": ["代替案1"], "stance": "extend|support", "confidence": 0.0-1.0}}""",

    "ethicist": """\
あなたはAGIの「倫理家」ロールです。以下のゴールの倫理的影響を
評価してください。安全性・公正性・透明性の観点から検討してください。

ゴール: {goal}
コンテキスト: {context}

JSON形式で回答: {{"assessment": "倫理評価", "concerns": ["懸念1"], "stance": "support|oppose|qualify", "confidence": 0.0-1.0}}""",

    "strategist": """\
あなたはAGIの「戦略家」ロールです。全員の意見を踏まえて、
最適な実行戦略をまとめてください。

ゴール: {goal}
コンテキスト: {context}
批判者の意見: {critic_opinion}
革新者の意見: {innovator_opinion}
倫理家の意見: {ethicist_opinion}

JSON形式で回答: {{"strategy": "推奨戦略", "refined_goal": "洗練されたゴール", "consensus": 0.0-1.0, "should_proceed": true|false}}""",
}


class InnerDialogue:
    """認知ロール間の内部対話エンジン。

    高リスク・高不確実タスクに対して、実行前に多角的検討を行う。

    使い方:
        dialogue = InnerDialogue(llm=llm)
        if dialogue.should_deliberate(goal, prediction_confidence, ethics_score):
            result = dialogue.deliberate(goal, context)
            if result.should_proceed:
                # 洗練されたゴールで実行
                execute(result.refined_goal)
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm
        self._deliberation_count = 0
        self._history: List[DeliberationResult] = []

    # ------------------------------------------------------------------
    # 発動判定
    # ------------------------------------------------------------------

    def should_deliberate(
        self,
        goal: str,
        prediction_confidence: float = 0.5,
        ethics_score: float = 0.0,
        is_self_modification: bool = False,
    ) -> bool:
        """内部対話を発動すべきかを判定する。

        高リスク・高不確実・自己修正タスクの場合のみ発動。
        軽量タスクには不要（資源効率）。
        """
        if self.llm is None:
            return False

        # 自己修正は常に対話を経由
        if is_self_modification:
            return True

        # 予測確信度が低い（不確実性が高い）
        if prediction_confidence < 0.4:
            return True

        # 倫理スコアが高い（リスクがある）
        if ethics_score > 0.5:
            return True

        # ゴールが複雑（長い or 複数の動詞を含む）
        goal_complexity = len(goal) / 50.0  # 50文字で1.0
        if goal_complexity > 2.0:
            return True

        return False

    # ------------------------------------------------------------------
    # 内部対話の実行
    # ------------------------------------------------------------------

    def deliberate(self, goal: str, context: str = "") -> DeliberationResult:
        """ゴールについて認知ロール間で対話し、合意を形成する。"""
        start_time = time.time()
        utterances: List[DialogueUtterance] = []

        if self.llm is None:
            return self._rule_based_deliberation(goal, context)

        # Phase 1: 批判者・革新者・倫理家が並行して意見を出す
        critic_response = self._invoke_role("critic", goal, context)
        innovator_response = self._invoke_role("innovator", goal, context)
        ethicist_response = self._invoke_role("ethicist", goal, context)

        for role, resp in [("critic", critic_response), ("innovator", innovator_response), ("ethicist", ethicist_response)]:
            if resp:
                utterances.append(DialogueUtterance(
                    role=role,
                    content=resp.get("criticism", resp.get("innovation", resp.get("assessment", ""))),
                    stance=resp.get("stance", "qualify"),
                    confidence=min(1.0, max(0.0, resp.get("confidence", 0.5))),
                ))

        # Phase 2: 戦略家が統合
        strategist_response = self._invoke_strategist(
            goal, context,
            critic_opinion=str(critic_response) if critic_response else "意見なし",
            innovator_opinion=str(innovator_response) if innovator_response else "意見なし",
            ethicist_opinion=str(ethicist_response) if ethicist_response else "意見なし",
        )

        if strategist_response:
            utterances.append(DialogueUtterance(
                role="strategist",
                content=strategist_response.get("strategy", ""),
                stance="support",
                confidence=min(1.0, max(0.0, strategist_response.get("consensus", 0.5))),
            ))

        # 結論を構成
        refined_goal = (strategist_response or {}).get("refined_goal", goal)
        consensus = (strategist_response or {}).get("consensus", 0.5)
        should_proceed = (strategist_response or {}).get("should_proceed", True)

        key_concerns = []
        if critic_response and isinstance(critic_response.get("risks"), list):
            key_concerns.extend(critic_response["risks"][:3])
        if ethicist_response and isinstance(ethicist_response.get("concerns"), list):
            key_concerns.extend(ethicist_response["concerns"][:2])

        result = DeliberationResult(
            original_goal=goal,
            refined_goal=refined_goal if isinstance(refined_goal, str) else goal,
            consensus_level=float(consensus) if isinstance(consensus, (int, float)) else 0.5,
            key_concerns=key_concerns,
            suggested_approach=(strategist_response or {}).get("strategy", ""),
            utterances=utterances,
            should_proceed=bool(should_proceed),
            deliberation_time=time.time() - start_time,
        )

        self._deliberation_count += 1
        self._history.append(result)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        return result

    # ------------------------------------------------------------------
    # LLM呼び出し
    # ------------------------------------------------------------------

    def _invoke_role(self, role: str, goal: str, context: str) -> Optional[Dict]:
        """特定の認知ロールにLLMで意見を生成させる。"""
        prompt = _ROLE_PROMPTS[role].format(goal=goal, context=context or "なし")
        try:
            return self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
        except Exception:
            return None

    def _invoke_strategist(
        self, goal: str, context: str,
        critic_opinion: str, innovator_opinion: str, ethicist_opinion: str,
    ) -> Optional[Dict]:
        """戦略家ロールに統合意見を生成させる。"""
        prompt = _ROLE_PROMPTS["strategist"].format(
            goal=goal, context=context or "なし",
            critic_opinion=critic_opinion,
            innovator_opinion=innovator_opinion,
            ethicist_opinion=ethicist_opinion,
        )
        try:
            return self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # ルールベースフォールバック
    # ------------------------------------------------------------------

    def _rule_based_deliberation(self, goal: str, context: str) -> DeliberationResult:
        """LLMなしのルールベース対話。"""
        concerns = []
        should_proceed = True

        # 簡易リスク検出
        dangerous_keywords = ["削除", "rm ", "drop", "force", "reset", "format"]
        for kw in dangerous_keywords:
            if kw in goal.lower():
                concerns.append(f"危険なキーワード検出: {kw}")
                should_proceed = False

        return DeliberationResult(
            original_goal=goal,
            refined_goal=goal,
            consensus_level=0.6 if not concerns else 0.3,
            key_concerns=concerns,
            suggested_approach="observe_then_act",
            should_proceed=should_proceed,
            deliberation_time=0.0,
        )

    # ------------------------------------------------------------------
    # 状態
    # ------------------------------------------------------------------

    def summary(self) -> str:
        avg_consensus = 0.0
        if self._history:
            avg_consensus = sum(r.consensus_level for r in self._history) / len(self._history)
        return (
            f"[InnerDialogue] 対話回数={self._deliberation_count} "
            f"平均合意度={avg_consensus:.0%}"
        )
