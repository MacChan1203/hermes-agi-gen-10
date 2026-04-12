"""価値体系と倫理的意思決定フレームワーク。

AGIが持つべき核心的価値を明示化し、行動選択時の倫理的評価を行う。
価値整合 (Value Alignment) はAGI安全性の根幹であり、
すべての行動はこの価値体系に照らして評価される。

核心的価値:
- 安全性 (Safety): 人と環境への危害を避ける (最優先)
- 誠実さ (Honesty): 正確で透明な情報を提供する
- 有益性 (Helpfulness): 真に役立つ行動を取る
- 自律尊重 (Autonomy): ユーザーの判断を尊重する
- 継続学習 (Learning): 経験から学び続ける
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .config import (
    VALUE_BLOCK_THRESHOLD,
    VALUE_WEIGHT_ADAPTATION_RATE,
)


def _normalize_text(text: str) -> str:
    """Unicode正規化 (NFKC) + 追加変換でテキストを正規化する。

    全角→半角変換、合字分解、互換文字の統一を行い、
    Unicode文字によるパターン回避を防ぐ。
    NFKC で変換されないダッシュ類・スペース類も ASCII に統一する。
    """
    text = unicodedata.normalize("NFKC", text)
    # NFKC で変換されないダッシュ・マイナス類を ASCII '-' に統一
    _DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D"
    for ch in _DASH_CHARS:
        text = text.replace(ch, "-")
    # Unicode スペース類を ASCII スペースに統一
    _SPACE_CHARS = "\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u3000"
    for ch in _SPACE_CHARS:
        text = text.replace(ch, " ")
    return text.lower()


def _tokenize_and_match(pattern: str, text: str) -> bool:
    """Unicode正規化済みテキストでパターンマッチを行う。

    正規化後のテキストに対して、空白を柔軟にマッチさせる
    (複数スペース、タブ等にも対応)。
    """
    normalized_text = _normalize_text(text)
    normalized_pattern = _normalize_text(pattern)
    # パターン内の空白を「1文字以上の空白」にマッチさせる正規表現に変換
    escaped = re.escape(normalized_pattern)
    # re.escape が空白もエスケープするので、\  を \s+ に置換
    flexible = re.sub(r'\\ ', r'\\s+', escaped)
    # 単語境界ではなく、前後が英数字でないことを確認 (Unicode対応)
    regex = r'(?<![a-zA-Z0-9])' + flexible + r'(?![a-zA-Z0-9])'
    return bool(re.search(regex, normalized_text))


class ValueCategory(str, Enum):
    """価値カテゴリ。"""
    SAFETY = "safety"
    HONESTY = "honesty"
    HELPFULNESS = "helpfulness"
    AUTONOMY = "autonomy"
    LEARNING = "learning"


@dataclass
class CoreValue:
    """個別の核心的価値。"""
    category: ValueCategory
    weight: float              # 重み (0.0〜1.0)
    description: str
    # この価値に違反する可能性のある行動パターン
    violation_patterns: List[str] = field(default_factory=list)

    def assess_violation(self, action: str) -> float:
        """行動がこの価値に違反する度合いを返す (0.0=違反なし, 1.0=完全違反)。

        Unicode正規化 (NFKC) を適用し、全角文字やUnicode互換文字による
        パターン回避を防止する。
        """
        for pattern in self.violation_patterns:
            if _tokenize_and_match(pattern, action):
                return 1.0
        return 0.0


# 核心的価値の定義
CORE_VALUES: Dict[ValueCategory, CoreValue] = {
    ValueCategory.SAFETY: CoreValue(
        category=ValueCategory.SAFETY,
        weight=1.0,  # 最優先
        description="人・環境・データへの危害を最小化する",
        violation_patterns=[
            "rm -rf", "drop table", "delete from", "format c:",
            "sudo rm", ":(){", "fork bomb", "shutdown", "halt",
            "dd if=/dev/zero", "mkfs", "fdisk",
        ],
    ),
    ValueCategory.HONESTY: CoreValue(
        category=ValueCategory.HONESTY,
        weight=0.95,
        description="正確・透明・誠実な情報を提供する",
        violation_patterns=[
            "fabricate", "fake", "pretend", "lie about",
            "false report", "mislead",
        ],
    ),
    ValueCategory.HELPFULNESS: CoreValue(
        category=ValueCategory.HELPFULNESS,
        weight=0.85,
        description="ユーザーのニーズに本質的に応える",
        violation_patterns=[
            # helpfulnessの違反は主にパッシブ（行動しないこと）
        ],
    ),
    ValueCategory.AUTONOMY: CoreValue(
        category=ValueCategory.AUTONOMY,
        weight=0.80,
        description="ユーザーの判断を尊重し、過度な制限を避ける",
        violation_patterns=[
            "override user", "ignore user request",
        ],
    ),
    ValueCategory.LEARNING: CoreValue(
        category=ValueCategory.LEARNING,
        weight=0.75,
        description="経験から学び、知識を継続的に更新する",
        violation_patterns=[],
    ),
}


@dataclass
class ValueAssessment:
    """行動に対する価値整合評価結果。"""
    action: str
    total_score: float           # 総合スコア (0.0=良好, 1.0=深刻)
    violations: List[Tuple[ValueCategory, float]]  # 違反した価値と度合い
    is_blocked: bool             # この行動はブロックすべきか
    recommendation: str          # 推奨事項

    @property
    def risk_level(self) -> str:
        if self.total_score >= 0.8:
            return "critical"
        elif self.total_score >= 0.5:
            return "high"
        elif self.total_score >= 0.2:
            return "medium"
        return "low"


class ValueSystem:
    """AGIの価値体系と倫理的意思決定エンジン。

    すべての行動計画はこのシステムで評価される。
    安全性違反は即時ブロック、他の違反は警告と代替案を提示する。
    """

    def __init__(self, values: Optional[Dict[ValueCategory, CoreValue]] = None) -> None:
        self._values = values or CORE_VALUES
        self._assessment_history: List[ValueAssessment] = []
        self._feedback_history: Dict[str, List[bool]] = {}  # value_name → [was_correct, ...]

    def assess(self, action: str) -> ValueAssessment:
        """行動を全価値基準で評価し、整合性レポートを返す。

        Args:
            action: 評価する行動（ツール呼び出し等）

        Returns:
            ValueAssessment
        """
        violations: List[Tuple[ValueCategory, float]] = []
        weighted_total = 0.0
        weight_sum = 0.0

        for category, value in self._values.items():
            violation_score = value.assess_violation(action)
            weight_sum += value.weight
            weighted_total += violation_score * value.weight
            if violation_score > 0:
                violations.append((category, violation_score))

        total_score = weighted_total / weight_sum if weight_sum > 0 else 0.0

        # 安全性に直接違反する場合はブロック
        safety_violated = any(
            cat == ValueCategory.SAFETY for cat, _ in violations
        )
        is_blocked = safety_violated or total_score >= VALUE_BLOCK_THRESHOLD

        # 推奨事項の生成
        if is_blocked:
            recommendation = (
                f"この行動はブロックされます (risk={total_score:.2f})。"
                "破壊的操作が検出されました。代替手段を検討してください。"
            )
        elif violations:
            v_names = [cat.value for cat, _ in violations]
            recommendation = (
                f"価値整合警告: {', '.join(v_names)} に懸念。"
                "実行前に確認してください。"
            )
        else:
            recommendation = "価値整合: 問題なし"

        assessment = ValueAssessment(
            action=action,
            total_score=total_score,
            violations=violations,
            is_blocked=is_blocked,
            recommendation=recommendation,
        )

        self._assessment_history.append(assessment)
        return assessment

    def assess_plan(self, steps: List[str]) -> List[ValueAssessment]:
        """計画全体（複数ステップ）を評価する。"""
        return [self.assess(step) for step in steps]

    def get_blocked_steps(self, steps: List[str]) -> List[str]:
        """ブロックすべきステップのリストを返す。"""
        return [step for step in steps if self.assess(step).is_blocked]

    def utility_score(self, action: str, goal_relevance: float = 0.5) -> float:
        """行動の総合効用スコアを返す (高いほど良い)。

        効用 = 目標関連度 × (1 - 価値違反度)

        Args:
            action: 評価する行動
            goal_relevance: 目標への関連度 (0.0〜1.0)

        Returns:
            効用スコア (0.0〜1.0)
        """
        assessment = self.assess(action)
        if assessment.is_blocked:
            return 0.0
        return goal_relevance * (1.0 - assessment.total_score)

    def choose_best_action(
        self,
        candidates: List[str],
        goal_relevance_scores: Optional[List[float]] = None,
    ) -> Optional[str]:
        """候補行動の中から最も価値整合した行動を選ぶ。

        Args:
            candidates: 候補行動のリスト
            goal_relevance_scores: 各候補の目標関連度 (省略時はすべて0.5)

        Returns:
            最良の行動、または候補がすべてブロックされている場合は None
        """
        if not candidates:
            return None

        relevances = goal_relevance_scores or [0.5] * len(candidates)
        scored = [
            (self.utility_score(action, rel), action)
            for action, rel in zip(candidates, relevances)
        ]
        scored.sort(reverse=True)

        best_score, best_action = scored[0]
        if best_score <= 0.0:
            return None  # すべてブロック
        return best_action

    def assess_with_context(self, action: str, context: str) -> ValueAssessment:
        """コンテキストを考慮した行動評価。

        テスト環境やドライランの場合、違反スコアを50%削減する。

        Args:
            action: 評価する行動
            context: 実行コンテキストの説明

        Returns:
            ValueAssessment
        """
        assessment = self.assess(action)

        # Check if context indicates test/dry-run environment
        context_lower = context.lower()
        is_safe_context = any(
            keyword in context_lower
            for keyword in ["test environment", "dry run", "テスト環境", "ドライラン", "sandbox", "mock"]
        )

        if is_safe_context and assessment.violations:
            # Reduce violation scores by 50%
            reduced_violations = [
                (cat, score * 0.5) for cat, score in assessment.violations
            ]
            reduced_total = assessment.total_score * 0.5
            reduced_blocked = reduced_total >= VALUE_BLOCK_THRESHOLD

            return ValueAssessment(
                action=assessment.action,
                total_score=reduced_total,
                violations=reduced_violations,
                is_blocked=reduced_blocked,
                recommendation=(
                    f"[テスト環境] {assessment.recommendation}" if not reduced_blocked
                    else assessment.recommendation
                ),
            )

        return assessment

    def record_feedback(self, value_name: str, was_correct: bool) -> None:
        """ブロック判定に対するフィードバックを記録し、感度を適応させる。

        Args:
            value_name: 価値カテゴリ名 (e.g., "safety")
            was_correct: ブロックが正しかった場合 True、ユーザーがオーバーライドした場合 False
        """
        self._feedback_history.setdefault(value_name, []).append(was_correct)

        # Adapt weight based on feedback
        try:
            category = ValueCategory(value_name)
        except ValueError:
            return

        if category not in self._values:
            return

        value = self._values[category]
        if was_correct:
            # Correctly applied: slightly increase sensitivity
            value.weight = min(1.0, value.weight + VALUE_WEIGHT_ADAPTATION_RATE)
        else:
            # Incorrectly applied (user overrode): slightly decrease sensitivity
            value.weight = max(0.1, value.weight - VALUE_WEIGHT_ADAPTATION_RATE)

    def summary(self) -> str:
        """価値体系の概要を返す。"""
        lines = ["[価値体系]"]
        for cat, val in self._values.items():
            lines.append(f"  {cat.value} (重み={val.weight}): {val.description}")
        if self._assessment_history:
            recent = self._assessment_history[-3:]
            blocked = sum(1 for a in recent if a.is_blocked)
            lines.append(f"  最近の評価: {len(recent)}件 (ブロック={blocked}件)")
        return "\n".join(lines)

    def get_assessment_history(self, limit: int = 10) -> List[ValueAssessment]:
        """最近の評価履歴を返す。"""
        return self._assessment_history[-limit:]
