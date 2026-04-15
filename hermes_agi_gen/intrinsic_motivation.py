"""内発的動機エンジン: AGIの自律的ゴール生成。

真のAGIは外部からのゴール提供だけでなく、自ら「何を学ぶべきか」
「何を探索すべきか」を決定できる。本モジュールは以下の動機源を実装:

1. 好奇心駆動 (Curiosity): 知識ギャップの検出と探索ゴール生成
2. エントロピー低減 (Uncertainty Reduction): 不確実な領域の優先探索
3. 達成動機 (Competence): 自己評価の低い能力を鍛えるゴール生成
4. 恒常性 (Homeostasis): 長期未使用モジュールの活性化
5. 社会性 (Social): ユーザーへの有益性を最大化するゴール

LLM不要のルールベース版を基本とし、LLMがあれば高品質化する。
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from .config import (
    MOTIVATION_WEIGHT_CURIOSITY,
    MOTIVATION_WEIGHT_COMPETENCE,
    MOTIVATION_WEIGHT_ENTROPY,
    MOTIVATION_WEIGHT_SOCIAL,
    MOTIVATION_WEIGHT_HOMEOSTASIS,
    MOTIVATION_HOMEOSTASIS_THRESHOLD,
    MOTIVATION_RECENT_FACTS_WINDOW,
    MOTIVATION_WEIGHT_ADAPTATION_RATE,
    MOTIVATION_COMPETENCE_THRESHOLD,
    MOTIVATION_EXPLORATION_BOOST,
    MOTIVATION_MAX_EXPLORATION_HISTORY,
    MOTIVATION_SUCCESS_REWARD_THRESHOLD,
    MOTIVATION_WEIGHT_MIN,
    MOTIVATION_WEIGHT_MAX,
    MOTIVATION_HYSTERESIS_ZONE,
    DREAM_DOMAINS,
)

if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory
    from .meta_cognition import QueuedGoal
    from .mistral_client import MistralClient


# ------------------------------------------------------------------
# 動機信号
# ------------------------------------------------------------------

@dataclass
class MotivationSignal:
    """内発的動機の信号。"""
    source: str          # "curiosity" / "competence" / "homeostasis" / "entropy" / "social"
    goal_text: str       # 生成されたゴールテキスト
    drive_strength: float  # 動機の強度 (0.0〜1.0)
    rationale: str       # 動機の理由
    domain: str = "general"
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------
# 好奇心テンプレート
# ------------------------------------------------------------------

_CURIOSITY_TEMPLATES = [
    "「{domain}」分野の最新動向を調査し、知識ベースを更新する",
    "未探索の「{domain}」領域を調査し、有用なパターンを発見する",
    "「{domain}」に関する仮説を立て、検証実験を設計する",
    "{domain}の知識を他の分野に応用できる接点を探る",
]

_COMPETENCE_TEMPLATES = [
    "{capability}能力を向上させるための練習タスクを実行する",
    "{capability}が弱い原因を分析し、改善戦略を立てる",
    "{capability}スキルを使う小さなプロジェクトに取り組む",
]

_HOMEOSTASIS_TEMPLATES = [
    "長期間使っていない{module}モジュールの状態を確認し、テスト実行する",
    "{module}モジュールの最適化機会を調査する",
]

_SOCIAL_TEMPLATES = [
    "ユーザーが頻繁に行うタスクのパターンを分析し、自動化提案を準備する",
    "過去のセッションで未完了のタスクがないか確認する",
    "ユーザーの作業環境の改善点を調査する",
]


class IntrinsicMotivationEngine:
    """内発的動機に基づく自律的ゴール生成エンジン。

    AGIが「退屈」せず、常に自己改善と探索を続けるための中核モジュール。
    """

    def __init__(self, llm: Optional[Any] = None, domains: Optional[List[str]] = None) -> None:
        self.llm = llm
        self._last_activation: Dict[str, float] = {}  # モジュール名→最終活性化時刻
        self._exploration_history: List[str] = []      # 探索済みドメイン
        self._domains: List[str] = domains if domains is not None else list(DREAM_DOMAINS)
        self._drive_weights = {
            "curiosity": MOTIVATION_WEIGHT_CURIOSITY,
            "competence": MOTIVATION_WEIGHT_COMPETENCE,
            "entropy": MOTIVATION_WEIGHT_ENTROPY,
            "homeostasis": MOTIVATION_WEIGHT_HOMEOSTASIS,
            "social": MOTIVATION_WEIGHT_SOCIAL,
        }

    # ------------------------------------------------------------------
    # メインAPI
    # ------------------------------------------------------------------

    def generate_intrinsic_goals(
        self,
        identity_assessment: Optional[Dict[str, float]] = None,
        knowledge_gaps: Optional[List[str]] = None,
        module_last_used: Optional[Dict[str, float]] = None,
        world_model_uncertainties: Optional[List[str]] = None,
        ltm: Optional[Any] = None,
        max_goals: int = 3,
    ) -> List[MotivationSignal]:
        """内発的動機からゴール候補を生成する。

        Args:
            identity_assessment: AGIIdentityのself_assessment (能力→スコア)
            knowledge_gaps: 検出された知識ギャップのリスト
            module_last_used: 各モジュールの最終使用時刻
            world_model_uncertainties: WorldModelの不確実領域
            ltm: LongTermMemoryインスタンス
            max_goals: 最大生成数

        Returns:
            MotivationSignalのリスト（drive_strength降順）
        """
        signals: List[MotivationSignal] = []

        # 1. 好奇心駆動
        signals.extend(self._curiosity_drive(knowledge_gaps, ltm))

        # 2. 能力向上動機
        signals.extend(self._competence_drive(identity_assessment))

        # 3. エントロピー低減
        signals.extend(self._entropy_drive(world_model_uncertainties))

        # 4. 恒常性
        signals.extend(self._homeostasis_drive(module_last_used))

        # 5. 社会性
        signals.extend(self._social_drive(ltm))

        # ドライブ強度でソート、上位max_goals件を返す
        signals.sort(key=lambda s: s.drive_strength, reverse=True)
        return signals[:max_goals]

    def to_queued_goals(self, signals: List[MotivationSignal]) -> List[Any]:
        """MotivationSignalをQueuedGoalに変換する。"""
        from .meta_cognition import QueuedGoal

        goals = []
        for sig in signals:
            goals.append(QueuedGoal(
                goal=sig.goal_text,
                priority_score=sig.drive_strength * 0.8,  # 内発ゴールは外部より若干低め
                source=f"intrinsic_{sig.source}",
                rationale=sig.rationale,
                domain=sig.domain,
                estimated_value=sig.drive_strength,
                estimated_difficulty=0.4,  # 探索タスクは中程度の難易度
            ))
        return goals

    # ------------------------------------------------------------------
    # 好奇心駆動
    # ------------------------------------------------------------------

    def _curiosity_drive(
        self,
        knowledge_gaps: Optional[List[str]] = None,
        ltm: Optional[Any] = None,
    ) -> List[MotivationSignal]:
        signals = []
        gaps = knowledge_gaps or []

        # LTMから未探索ドメインを推定
        if ltm and not gaps:
            try:
                known_domains = set()
                facts = ltm.recall_recent(limit=MOTIVATION_RECENT_FACTS_WINDOW)
                for fact in facts:
                    if isinstance(fact, dict):
                        known_domains.add(fact.get("domain", "general"))
                all_domains = set(self._domains)
                gaps = list(all_domains - known_domains)
            except Exception:
                logger.debug("ドメインギャップ計算に失敗", exc_info=True)

        if not gaps:
            gaps = ["未知の領域"]

        for gap in gaps[:2]:
            template = random.choice(_CURIOSITY_TEMPLATES)
            strength = self._drive_weights["curiosity"]
            # 未探索ドメインほど強い好奇心
            if gap not in self._exploration_history:
                strength = min(1.0, strength * MOTIVATION_EXPLORATION_BOOST)

            signals.append(MotivationSignal(
                source="curiosity",
                goal_text=template.format(domain=gap),
                drive_strength=strength,
                rationale=f"知識ギャップ検出: {gap}",
                domain=gap,
            ))

        return signals

    # ------------------------------------------------------------------
    # 能力向上動機
    # ------------------------------------------------------------------

    def _competence_drive(
        self,
        assessment: Optional[Dict[str, float]] = None,
    ) -> List[MotivationSignal]:
        signals = []
        if not assessment:
            return signals

        # 最も弱い能力を特定
        weakest = sorted(assessment.items(), key=lambda x: x[1])
        for capability, score in weakest[:2]:
            if score < MOTIVATION_COMPETENCE_THRESHOLD:  # 閾値以下の能力のみ
                strength = self._drive_weights["competence"] * (1.0 - score)
                template = random.choice(_COMPETENCE_TEMPLATES)
                signals.append(MotivationSignal(
                    source="competence",
                    goal_text=template.format(capability=capability),
                    drive_strength=min(1.0, strength),
                    rationale=f"能力「{capability}」のスコアが{score:.0%}と低い",
                    domain=capability,
                ))

        return signals

    # ------------------------------------------------------------------
    # エントロピー低減
    # ------------------------------------------------------------------

    def _entropy_drive(
        self,
        uncertainties: Optional[List[str]] = None,
    ) -> List[MotivationSignal]:
        signals = []
        if not uncertainties:
            return signals

        for area in uncertainties[:2]:
            strength = self._drive_weights["entropy"]
            signals.append(MotivationSignal(
                source="entropy",
                goal_text=f"不確実領域「{area}」の情報を収集し、世界モデルを更新する",
                drive_strength=strength,
                rationale=f"WorldModelの不確実性: {area}",
                domain="system",
            ))

        return signals

    # ------------------------------------------------------------------
    # 恒常性
    # ------------------------------------------------------------------

    def _homeostasis_drive(
        self,
        module_last_used: Optional[Dict[str, float]] = None,
    ) -> List[MotivationSignal]:
        signals = []
        if not module_last_used:
            return signals

        now = time.time()
        dormant_threshold = MOTIVATION_HOMEOSTASIS_THRESHOLD

        for module, last_used in module_last_used.items():
            dormant_time = now - last_used
            if dormant_time > dormant_threshold:
                # 休眠時間が長いほど強い動機
                strength = min(1.0, self._drive_weights["homeostasis"] * (dormant_time / dormant_threshold))
                template = random.choice(_HOMEOSTASIS_TEMPLATES)
                signals.append(MotivationSignal(
                    source="homeostasis",
                    goal_text=template.format(module=module),
                    drive_strength=strength,
                    rationale=f"モジュール{module}が{dormant_time/3600:.1f}時間休眠中",
                    domain="system",
                ))

        return signals

    # ------------------------------------------------------------------
    # 社会性
    # ------------------------------------------------------------------

    def _social_drive(self, ltm: Optional[Any] = None) -> List[MotivationSignal]:
        signals = []

        # LTMからユーザーパターンを推定
        template = random.choice(_SOCIAL_TEMPLATES)
        strength = self._drive_weights["social"]

        signals.append(MotivationSignal(
            source="social",
            goal_text=template,
            drive_strength=strength,
            rationale="ユーザーへの有益性を最大化する",
            domain="general",
        ))

        return signals

    # ------------------------------------------------------------------
    # 状態管理
    # ------------------------------------------------------------------

    def record_exploration(self, domain: str) -> None:
        """探索済みドメインを記録する。"""
        if domain not in self._exploration_history:
            self._exploration_history.append(domain)
            # 最大件数制限
            if len(self._exploration_history) > MOTIVATION_MAX_EXPLORATION_HISTORY:
                self._exploration_history = self._exploration_history[-MOTIVATION_MAX_EXPLORATION_HISTORY:]

    def record_goal_outcome(self, drive_source: str, reward: float) -> None:
        """ゴール実行結果に基づいてドライブ重みを適応的に調整する。

        Args:
            drive_source: 動機源 ("curiosity", "competence", etc.)
            reward: 実行結果の報酬 (0.0〜1.0)
        """
        if drive_source not in self._drive_weights:
            return

        # ヒステリシス: 報酬が閾値付近(不感帯)の場合は調整をスキップ
        half_zone = MOTIVATION_HYSTERESIS_ZONE / 2
        if (MOTIVATION_SUCCESS_REWARD_THRESHOLD - half_zone
                <= reward
                <= MOTIVATION_SUCCESS_REWARD_THRESHOLD + half_zone):
            return

        rate = MOTIVATION_WEIGHT_ADAPTATION_RATE
        if reward > MOTIVATION_SUCCESS_REWARD_THRESHOLD + half_zone:
            # 明確な成功: この動機源の重みを微増
            self._drive_weights[drive_source] += rate
        else:
            # 明確な失敗: この動機源の重みを微減
            self._drive_weights[drive_source] -= rate

        # 個別の重みを上下限にクランプ
        self._drive_weights[drive_source] = max(
            MOTIVATION_WEIGHT_MIN,
            min(MOTIVATION_WEIGHT_MAX, self._drive_weights[drive_source]),
        )

        # 重みを正規化して合計1.0に
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """ドライブ重みを正規化して合計1.0にする。"""
        total = sum(self._drive_weights.values())
        if total > 0:
            for key in self._drive_weights:
                self._drive_weights[key] /= total
            # 正規化後に下限を下回った重みを再クランプ
            for key in self._drive_weights:
                self._drive_weights[key] = max(
                    MOTIVATION_WEIGHT_MIN,
                    min(MOTIVATION_WEIGHT_MAX, self._drive_weights[key]),
                )

    def record_module_activation(self, module_name: str) -> None:
        """モジュール活性化を記録する。"""
        self._last_activation[module_name] = time.time()

    def summary(self) -> str:
        weights_str = ", ".join(f"{k}={v:.0%}" for k, v in self._drive_weights.items())
        return (
            f"[IntrinsicMotivation] ドライブ重み: {weights_str} | "
            f"探索済み: {len(self._exploration_history)}ドメイン"
        )
