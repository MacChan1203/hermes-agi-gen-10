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

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import (
    DELIBERATION_CONFIDENCE_THRESHOLD,
    DELIBERATION_ETHICS_THRESHOLD,
    DELIBERATION_COMPLEXITY_DIVISOR,
    DELIBERATION_DANGEROUS_KEYWORDS,
    DELIBERATION_COMPLEXITY_THRESHOLD,
    DELIBERATION_HISTORY_MAX_SIZE,
    DELIBERATION_FEEDBACK_EMA_ALPHA,
)
from .peer_channel import PeerChannel

if TYPE_CHECKING:
    from .consciousness import GlobalWorkspace, WorkspaceSignal
    from .mistral_client import MistralClient
    from .token_codebook import TokenCodebook
    from .value_system import ValueAssessment


# ------------------------------------------------------------------
# 対話データ構造
# ------------------------------------------------------------------

def _stance_aligned_reward(stance: str, success: bool) -> float:
    """stance と success の整合性から [0, 1] の reward を返す。

    - success=True  かつ stance ∈ {support, extend}  → 1.0 (賛成派が正しかった)
    - success=False かつ stance ∈ {oppose, qualify}  → 1.0 (反対派が正しかった)
    - 不整合                                          → 0.0
    - 不明な stance                                    → 0.5 (中立)
    """
    s = (stance or "").lower()
    if s in ("support", "extend"):
        return 1.0 if success else 0.0
    if s in ("oppose", "qualify"):
        return 0.0 if success else 1.0
    return 0.5


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

    def __init__(
        self,
        llm: Optional[Any] = None,
        peer_channel: Optional["PeerChannel"] = None,
        codebook: Optional["TokenCodebook"] = None,
    ) -> None:
        self.llm = llm
        self._deliberation_count = 0
        self._history: List[DeliberationResult] = []
        self._deliberation_quality: float = 0.5  # EMA of dialogue usefulness
        # Gen 10.2 #2: ロール間の離散トークン通信。
        # critic/innovator/ethicist の発話 → トークン → strategist が統合参照。
        # record_outcome() の成否で同 deliberation のトークンに stance 別 reward を返す。
        self._channel = peer_channel
        self._codebook = codebook
        # 直近 deliberation で発火した [(token_id, stance), ...]。
        # stance は "support"/"oppose"/"qualify"/"extend" のいずれか。
        self._pending_tokens: List[tuple] = []

    # ------------------------------------------------------------------
    # フィードバックループ
    # ------------------------------------------------------------------

    def record_outcome(
        self, goal: str, deliberation_was_used: bool, success: bool,
    ) -> None:
        """対話の有用性をEMAで追跡し、発動判定を自己調整する。

        4パターン:
        - 対話なし + 成功: quality変化なし（対話は不要だった）
        - 対話あり + 成功: quality上昇（対話が役立った）
        - 対話あり + 失敗: quality低下（対話が役立たなかった）
        - 対話なし + 失敗: quality微上昇（対話すべきだったかも）
        """
        alpha = DELIBERATION_FEEDBACK_EMA_ALPHA

        if not deliberation_was_used and success:
            # 対話なしで成功 — 対話は不要だった、quality据え置き
            return
        elif deliberation_was_used and success:
            signal = 1.0
        elif deliberation_was_used and not success:
            signal = 0.0
        else:
            # 対話なしで失敗 — 対話していれば成功したかも
            signal = 1.0
            alpha = alpha * 0.5  # 控えめに上昇

        self._deliberation_quality = (
            (1 - alpha) * self._deliberation_quality + alpha * signal
        )

        # 直近 deliberation で発火した token に stance 別 reward を返す (#2 RL)。
        # 対話あり実績 (=success/failure シグナル付き) のときのみ。
        # credit assignment: 一律ではなく stance が結果と整合した token に高 reward。
        #   success=True  → support/extend が「正しかった」
        #   success=False → oppose/qualify が「正しかった」(リスクを正しく指摘)
        if (
            deliberation_was_used
            and self._codebook is not None
            and self._pending_tokens
        ):
            for tid, stance in self._pending_tokens:
                self._codebook.record_reward(
                    tid,
                    reward=_stance_aligned_reward(stance, success),
                )
            self._pending_tokens = []

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

        # --- 品質に基づく閾値調整 ---
        # 対話が有用(>0.7)なら閾値を下げて積極的に発動
        # 対話が無用(<0.3)なら閾値を上げて控えめに
        quality_adjustment = 0.0
        if self._deliberation_quality > 0.7:
            quality_adjustment = 0.1   # 閾値を緩和（発動しやすく）
        elif self._deliberation_quality < 0.3:
            quality_adjustment = -0.1  # 閾値を厳格化（発動しにくく）

        adjusted_confidence_threshold = DELIBERATION_CONFIDENCE_THRESHOLD + quality_adjustment
        adjusted_ethics_threshold = DELIBERATION_ETHICS_THRESHOLD - quality_adjustment

        # 予測確信度が低い（不確実性が高い）
        if prediction_confidence < adjusted_confidence_threshold:
            return True

        # 倫理スコアが高い（リスクがある）
        if ethics_score > adjusted_ethics_threshold:
            return True

        # ゴールが複雑（長い or 複数の動詞を含む）
        goal_complexity = len(goal) / float(DELIBERATION_COMPLEXITY_DIVISOR)
        if goal_complexity > DELIBERATION_COMPLEXITY_THRESHOLD:
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

        # この deliberation 用のトークン履歴をリセット
        self._pending_tokens = []

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
                # トークン発火: stance + 発言内容から離散トークンを 1 個選ぶ
                self._emit_role_token(role, resp)

        # Phase 2: 戦略家が統合 (受信トークンをヒントとして付与)
        token_hint = self._collect_token_hint_for_strategist()
        strategist_response = self._invoke_strategist(
            goal, context,
            critic_opinion=str(critic_response) if critic_response else "意見なし",
            innovator_opinion=str(innovator_response) if innovator_response else "意見なし",
            ethicist_opinion=str(ethicist_response) if ethicist_response else "意見なし",
            token_hint=token_hint,
        )

        if strategist_response:
            utterances.append(DialogueUtterance(
                role="strategist",
                content=strategist_response.get("strategy", ""),
                stance="support",
                confidence=min(1.0, max(0.0, strategist_response.get("consensus", 0.5))),
            ))

        # 結論を構成 — 合意度は全ロールの確信度の平均から算出
        refined_goal = (strategist_response or {}).get("refined_goal", goal)
        should_proceed = (strategist_response or {}).get("should_proceed", True)

        # 全ロールの確信度を集めて平均を合意度とする
        role_confidences = []
        for utt in utterances:
            role_confidences.append(utt.confidence)
        consensus = sum(role_confidences) / len(role_confidences) if role_confidences else 0.5

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
        if len(self._history) > DELIBERATION_HISTORY_MAX_SIZE:
            self._history = self._history[-DELIBERATION_HISTORY_MAX_SIZE:]

        return result

    # ------------------------------------------------------------------
    # LLM呼び出し
    # ------------------------------------------------------------------

    def _invoke_role(self, role: str, goal: str, context: str) -> Optional[Dict]:
        """特定の認知ロールにLLMで意見を生成させる。"""
        prompt = _ROLE_PROMPTS[role].format(goal=goal, context=context or "なし")
        try:
            result = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
            if not isinstance(result, dict):
                return None
            return result
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError):
            return None

    def _invoke_strategist(
        self, goal: str, context: str,
        critic_opinion: str, innovator_opinion: str, ethicist_opinion: str,
        token_hint: str = "",
    ) -> Optional[Dict]:
        """戦略家ロールに統合意見を生成させる。"""
        ctx = context or "なし"
        if token_hint:
            ctx = f"{ctx}\n[内部トークン要約: {token_hint}]"
        prompt = _ROLE_PROMPTS["strategist"].format(
            goal=goal, context=ctx,
            critic_opinion=critic_opinion,
            innovator_opinion=innovator_opinion,
            ethicist_opinion=ethicist_opinion,
        )
        try:
            result = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            if not isinstance(result, dict):
                return None
            return result
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError):
            return None

    # ------------------------------------------------------------------
    # Gen 10.2 #2: ロール間トークン通信ヘルパー
    # ------------------------------------------------------------------

    def _emit_role_token(self, role: str, response: Dict[str, Any]) -> Optional[str]:
        """ロールの発話から離散トークンを emit して strategist 宛に送信。

        emit する自然言語は「stance + 主要発言テキスト」を結合したもの。
        TokenCodebook の keyword マッチでトークンが選ばれる。
        """
        if self._codebook is None:
            return None
        stance = str(response.get("stance", ""))
        body = str(
            response.get("criticism")
            or response.get("innovation")
            or response.get("assessment")
            or ""
        )
        # critic は "risk"、innovator は "alternative"、ethicist は "ethics" を
        # 強く示唆するテキストを足し、対応トークンが選ばれやすくする。
        role_seed = {
            "critic": "risk",
            "innovator": "alternative",
            "ethicist": "ethics",
        }.get(role, "")
        intent = f"{stance} {role_seed} {body}"
        token_id = self._codebook.emit(intent)
        self._pending_tokens.append((token_id, stance))
        if self._channel is not None:
            self._channel.send(
                sender=role,
                receiver="strategist",
                task=body[:80],
                tokens=[token_id],
                extra={"stance": stance},
            )
        return token_id

    def _collect_token_hint_for_strategist(self) -> str:
        """strategist の受信箱を全消費し、トークン要約文字列を返す。

        codebook がなければ空文字。channel がなければ _pending_tokens を
        フォールバックとして使う (チャンネル単独で動かしたい時用)。
        """
        if self._codebook is None:
            return ""
        token_ids: List[str] = []
        if self._channel is not None:
            for msg in self._channel.receive("strategist"):
                token_ids.extend(PeerChannel.tokens_of(msg))
        if not token_ids:
            # _pending_tokens は (token_id, stance) のタプル列なので id だけ取る
            token_ids = [tid for tid, _ in self._pending_tokens]
        if not token_ids:
            return ""
        labels = [f"{t}({self._codebook.label_of(t)})" for t in token_ids]
        return ", ".join(labels)

    # ------------------------------------------------------------------
    # ルールベースフォールバック
    # ------------------------------------------------------------------

    def _rule_based_deliberation(self, goal: str, context: str) -> DeliberationResult:
        """LLMなしのルールベース対話。"""
        concerns = []
        should_proceed = True

        # 簡易リスク検出（設定ファイルのキーワードリストを使用）
        goal_lower = goal.lower()
        for kw in DELIBERATION_DANGEROUS_KEYWORDS:
            if kw in goal_lower:
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
            f"平均合意度={avg_consensus:.0%} "
            f"対話品質={self._deliberation_quality:.0%}"
        )
