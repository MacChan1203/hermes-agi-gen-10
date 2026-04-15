"""予測的処理エンジン (Predictive Processing Engine)。

Clark & Friston (2013) の予測的符号化 (Predictive Coding) 理論に基づき、
AGIが行動を実行する前に結果を予測し、予測誤差から学習する仕組みを実装する。

予測的処理の流れ:
1. 行動候補を受け取る
2. 過去の経験 (LTM) から結果を予測する
3. 予測確信度が低い場合は代替行動を推奨する
4. 行動実行後、実際の結果と予測を比較する
5. 予測誤差が大きい場合、モデルを更新する

これにより:
- 失敗しやすい行動を事前に特定できる
- 経験から自動的に精度が向上する
- 不確実な状況での慎重な行動を促す
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .config import (
    PREDICTION_BASE_PROBABILITY,
    PREDICTION_HISTORY_WEIGHT,
    PREDICTION_HISTORY_MAX_FACTOR,
    PREDICTION_HISTORY_DIVISOR,
    PREDICTION_FAILURE_RISK_BASE,
    PREDICTION_FAILURE_RISK_INCREMENT,
    PREDICTION_FAILURE_RISK_CAP,
    PREDICTION_WRONG_DIRECTION_PENALTY,
    PREDICTION_BAYESIAN_WEIGHT,
    PREDICTION_BAYESIAN_DECAY,
    PREDICTION_SUCCESS_THRESHOLD,
    PREDICTION_WRONG_DIRECTION_MAGNITUDE,
    PREDICTION_HISTORY_MAX_SIZE,
    PREDICTION_ACTION_TYPE_PRIORS,
)


@dataclass
class Prediction:
    """行動結果の予測。"""
    action: str
    predicted_outcome: str       # 予測結果の説明
    success_probability: float   # 成功確率 (0.0〜1.0)
    confidence: float            # 予測自体の確信度 (0.0〜1.0)
    predicted_side_effects: List[str] = field(default_factory=list)
    basis: str = ""              # 予測の根拠
    timestamp: float = field(default_factory=time.time)

    @property
    def should_proceed(self) -> bool:
        """この予測に基づいて実行すべきか判定する。"""
        # 成功確率が低く、かつ確信度が高い場合は実行しない
        if self.confidence > 0.7 and self.success_probability < 0.3:
            return False
        return True

    @property
    def uncertainty(self) -> float:
        """予測の不確実性 (1 - confidence)。"""
        return 1.0 - self.confidence


@dataclass
class PredictionRecord:
    """予測と実際の結果の記録 (学習データ)。"""
    prediction: Prediction
    actual_outcome: str          # 実際の結果
    actual_success: bool         # 実際に成功したか
    prediction_error: float      # 予測誤差 (0.0=完全一致, 1.0=完全不一致)
    recorded_at: float = field(default_factory=time.time)


class PredictiveEngine:
    """予測的処理エンジン: 行動前に結果を予測し、経験から精度を向上させる。

    長期記憶 (LTM) の失敗パターンと成功戦略を活用して予測を生成する。
    予測精度を継続的にトラッキングし、信頼性の高い予測モデルを構築する。
    """

    def __init__(self, ltm: Optional[Any] = None) -> None:
        """
        Args:
            ltm: LongTermMemory インスタンス (省略時は記憶なしで動作)
        """
        self.ltm = ltm
        self._prediction_history: List[PredictionRecord] = []
        self._accuracy_by_action_type: Dict[str, List[float]] = {}
        self._bayesian_accuracy: Dict[str, List[float]] = {}  # action_type → accuracy history for Bayesian updating

    def predict(self, action: str, goal: str = "", context: str = "") -> Prediction:
        """行動の結果を予測する。

        Args:
            action: 予測対象の行動
            goal: 達成しようとしているゴール
            context: 現在のコンテキスト

        Returns:
            Prediction オブジェクト
        """
        action_type = self._classify_action(action)

        # LTMから失敗パターンを確認
        failure_risk = self._assess_failure_risk(action)

        # LTMから成功戦略を確認
        success_evidence = self._assess_success_evidence(action, goal)

        # 予測確信度を計算
        confidence = self._calculate_confidence(action_type, failure_risk, success_evidence)

        # 成功確率を計算
        success_probability = self._calculate_success_probability(
            failure_risk, success_evidence, confidence, action_type=action_type
        )

        # 予測結果の説明を生成
        predicted_outcome = self._generate_outcome_description(
            action, action_type, success_probability
        )

        # 副作用を予測
        side_effects = self._predict_side_effects(action, action_type)

        # 予測根拠を生成
        basis_parts = []
        if failure_risk > 0.3:
            basis_parts.append(f"過去の失敗パターンあり(risk={failure_risk:.2f})")
        if success_evidence > 0.3:
            basis_parts.append(f"過去の成功事例あり(evidence={success_evidence:.2f})")
        if not basis_parts:
            basis_parts.append("記憶ベースの根拠なし（先験的推定）")

        return Prediction(
            action=action,
            predicted_outcome=predicted_outcome,
            success_probability=success_probability,
            confidence=confidence,
            predicted_side_effects=side_effects,
            basis=" | ".join(basis_parts),
        )

    def record_outcome(
        self, prediction: Prediction, actual_outcome: str, actual_success: bool
    ) -> PredictionRecord:
        """実際の結果を記録して学習データを蓄積する。

        Args:
            prediction: 事前に生成した予測
            actual_outcome: 実際の結果テキスト
            actual_success: 実際に成功したか

        Returns:
            PredictionRecord
        """
        # 予測誤差を計算
        predicted_success = prediction.success_probability >= PREDICTION_SUCCESS_THRESHOLD
        if predicted_success == actual_success:
            # 方向性は合っていた
            error = abs(prediction.success_probability - (1.0 if actual_success else 0.0))
        else:
            # 方向性が間違っていた（大きな誤差）
            error = PREDICTION_WRONG_DIRECTION_PENALTY + abs(prediction.success_probability - (1.0 if actual_success else 0.0)) * PREDICTION_WRONG_DIRECTION_MAGNITUDE

        record = PredictionRecord(
            prediction=prediction,
            actual_outcome=actual_outcome,
            actual_success=actual_success,
            prediction_error=error,
        )

        self._prediction_history.append(record)

        # アクションタイプ別精度を更新 (temporal decay を適用)
        action_type = self._classify_action(prediction.action)
        # Clamp reward to [0, 1]
        reward = 1.0 - error
        reward = max(0.0, min(1.0, reward))
        accuracy = reward

        # Apply temporal decay to existing entries before appending
        if action_type in self._accuracy_by_action_type:
            self._accuracy_by_action_type[action_type] = [
                v * PREDICTION_BAYESIAN_DECAY for v in self._accuracy_by_action_type[action_type]
            ]
            # Prune entries that have decayed below 0.01
            self._accuracy_by_action_type[action_type] = [
                v for v in self._accuracy_by_action_type[action_type] if v >= 0.01
            ]
        self._accuracy_by_action_type.setdefault(action_type, []).append(accuracy)

        # Update Bayesian accuracy tracker (with temporal decay)
        if action_type in self._bayesian_accuracy:
            self._bayesian_accuracy[action_type] = [
                v * PREDICTION_BAYESIAN_DECAY for v in self._bayesian_accuracy[action_type]
            ]
            # Prune entries that have decayed below 0.01
            self._bayesian_accuracy[action_type] = [
                v for v in self._bayesian_accuracy[action_type] if v >= 0.01
            ]
        self._bayesian_accuracy.setdefault(action_type, []).append(1.0 if actual_success else 0.0)

        # 履歴を最新200件に制限
        if len(self._prediction_history) > PREDICTION_HISTORY_MAX_SIZE:
            self._prediction_history = self._prediction_history[-PREDICTION_HISTORY_MAX_SIZE:]

        return record

    def get_accuracy(self, action_type: Optional[str] = None) -> float:
        """予測精度を返す。

        Args:
            action_type: 特定のアクションタイプ (省略時は全体)

        Returns:
            精度 (0.0〜1.0)
        """
        if action_type and action_type in self._accuracy_by_action_type:
            scores = self._accuracy_by_action_type[action_type]
        elif self._prediction_history:
            scores = [1.0 - r.prediction_error for r in self._prediction_history]
        else:
            return 0.5  # データなし

        return sum(scores) / len(scores) if scores else 0.5

    def get_risky_actions(self, threshold: float = 0.4) -> List[str]:
        """予測成功確率が閾値未満の最近の行動を返す。"""
        risky = []
        for record in self._prediction_history[-20:]:
            if record.prediction.success_probability < threshold:
                risky.append(record.prediction.action)
        return risky

    def summary(self) -> str:
        """予測エンジンの状態サマリを返す。"""
        if not self._prediction_history:
            return "[PredictiveEngine] 記録なし"

        total = len(self._prediction_history)
        overall_accuracy = self.get_accuracy()
        type_stats = {
            t: f"{self.get_accuracy(t):.1%}"
            for t in self._accuracy_by_action_type
        }
        stats_str = ", ".join(f"{k}={v}" for k, v in type_stats.items())

        return (
            f"[PredictiveEngine] 予測記録={total}件 | "
            f"全体精度={overall_accuracy:.1%} | "
            f"タイプ別={stats_str}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_action(self, action: str) -> str:
        """行動を種類に分類する。"""
        action_upper = action.upper().strip()
        if action_upper.startswith("CMD:"):
            return "cmd"
        elif action_upper.startswith("READ:"):
            return "read"
        elif action_upper.startswith("WRITE:"):
            return "write"
        elif action_upper.startswith("PYTHON:"):
            return "python"
        elif action_upper.startswith("SEARCH:"):
            return "search"
        elif action_upper.startswith("PLAN:"):
            return "plan"
        elif action_upper.startswith("ANSWER:"):
            return "answer"
        elif action_upper.startswith("DONE:"):
            return "done"
        return "unknown"

    def _assess_failure_risk(self, action: str) -> float:
        """LTMの失敗パターンからリスクを評価する。"""
        if self.ltm is None:
            return 0.1  # デフォルト低リスク

        try:
            failures = self.ltm.get_known_failures(limit=20)
            action_lower = action.lower()
            for failure in failures:
                pattern = failure.get("command_pattern", "").lower()
                if pattern and len(pattern) > 3:
                    # Word-boundary regex to prevent "rm" matching "form"
                    if re.search(r'\b' + re.escape(pattern) + r'\b', action_lower):
                        count = failure.get("count", 1)
                        return min(PREDICTION_FAILURE_RISK_CAP, PREDICTION_FAILURE_RISK_BASE + count * PREDICTION_FAILURE_RISK_INCREMENT)
        except Exception:
            logger.debug("失敗リスク評価に失敗", exc_info=True)

        return 0.1

    def _assess_success_evidence(self, action: str, goal: str) -> float:
        """LTMの成功戦略から成功根拠を評価する。"""
        if self.ltm is None:
            return 0.3

        try:
            strategies = self.ltm.recall_strategies(goal, limit=5)
            action_lower = action.lower()
            for strategy in strategies:
                s = strategy.get("strategy", "").lower()
                outcome = strategy.get("outcome", "")
                if s and len(s) > 3 and s[:20] in action_lower and outcome == "success":
                    return 0.7
        except Exception:
            logger.debug("成功根拠評価に失敗", exc_info=True)

        return 0.3

    def _calculate_confidence(
        self, action_type: str, failure_risk: float, success_evidence: float
    ) -> float:
        """予測確信度を計算する。"""
        # データが多いほど確信度が上がる
        history_factor = min(PREDICTION_HISTORY_MAX_FACTOR, len(self._prediction_history) / PREDICTION_HISTORY_DIVISOR)
        base_confidence = 0.3 + history_factor * PREDICTION_HISTORY_WEIGHT

        # 失敗パターンや成功根拠があれば確信度が上がる
        if failure_risk > 0.3 or success_evidence > 0.3:
            base_confidence = min(0.9, base_confidence + 0.2)

        return base_confidence

    def _calculate_success_probability(
        self, failure_risk: float, success_evidence: float, confidence: float,
        action_type: str = "unknown",
    ) -> float:
        """成功確率を計算する (情報量ある事前分布 + Bayesian updating)。"""
        # アクションタイプ別の経験的事前確率 (一律 0.5 ではなく情報量あり)
        base = PREDICTION_ACTION_TYPE_PRIORS.get(action_type, PREDICTION_BASE_PROBABILITY)
        # 失敗リスクと成功根拠で調整
        base += (success_evidence - failure_risk) * PREDICTION_HISTORY_WEIGHT

        # Bayesian updating: blend base probability with historical accuracy per action type
        historical_records = self._bayesian_accuracy.get(action_type, [])
        if historical_records:
            historical_accuracy = sum(historical_records) / len(historical_records)
            base = base * (1 - PREDICTION_BAYESIAN_WEIGHT) + historical_accuracy * PREDICTION_BAYESIAN_WEIGHT

        return max(0.05, min(0.95, base))

    def _generate_outcome_description(
        self, action: str, action_type: str, success_probability: float
    ) -> str:
        """予測結果の説明文を生成する。"""
        level = (
            "高い確率で成功" if success_probability >= 0.7
            else "成功する可能性あり" if success_probability >= 0.5
            else "失敗リスクあり" if success_probability >= 0.3
            else "失敗する可能性が高い"
        )

        action_desc = {
            "cmd": "シェルコマンドが実行される",
            "read": "ファイルが読み込まれる",
            "write": "ファイルに書き込まれる",
            "python": "Pythonコードが実行される",
            "search": "Web検索が行われる",
            "plan": "計画が立案される",
            "answer": "直接回答が生成される",
        }.get(action_type, "行動が実行される")

        return f"{action_desc} — {level} (確率={success_probability:.0%})"

    def _predict_side_effects(self, action: str, action_type: str) -> List[str]:
        """行動の副作用を予測する。"""
        effects = []

        if action_type == "write":
            effects.append("既存ファイルが上書きされる可能性")
        elif action_type == "cmd":
            action_lower = action.lower()
            if "rm" in action_lower or "delete" in action_lower:
                effects.append("ファイルが削除される")
            if "install" in action_lower or "pip" in action_lower:
                effects.append("パッケージがインストールされる")
            if "git" in action_lower:
                effects.append("Gitリポジトリの状態が変わる")
        elif action_type == "python":
            effects.append("Pythonの状態が変わる可能性")

        return effects
