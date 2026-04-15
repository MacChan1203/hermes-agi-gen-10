"""実行結果レビュアー。

信頼度スコアとリスクレベルを追加し、低信頼・高リスク操作に確認フラグを立てる。
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .agent_state import AgentState
from .config import (
    REVIEWER_STATIC_CONFIDENCE_PASS,
    REVIEWER_STATIC_CONFIDENCE_PARTIAL,
    REVIEWER_STATIC_CONFIDENCE_FAIL,
)
from .errors import classify_error, should_retry_error_type, should_retry_step
from .memory import remember_failure

if TYPE_CHECKING:
    from .mistral_client import MistralClient

# 破壊的操作のパターン
_DESTRUCTIVE_PATTERNS = [
    "rm -rf", "rm -r", "drop table", "truncate", "format", "mkfs",
    "dd if=", ":(){:|:&};:", "chmod 777", "sudo rm", "git reset --hard",
    "git push --force", "> /dev/", "shred", "wipefs",
]

# リスクレベルしきい値
_CONFIDENCE_WARN_THRESHOLD = 0.6   # 60%未満: 警告
_CONFIDENCE_BLOCK_THRESHOLD = 0.4  # 40%未満: ブロック推奨


def _assess_risk(step: str) -> str:
    """ステップのリスクレベルを評価する。

    単語境界マッチングを使用し、"rm -rf" が "form-rfid" 等に
    誤マッチしないようにする。
    """
    for pattern in _DESTRUCTIVE_PATTERNS:
        if re.search(r'\b' + re.escape(pattern) + r'\b', step, re.IGNORECASE):
            return "critical"
    if step.upper().startswith(("WRITE:", "CMD:")):
        return "medium"
    if step.upper().startswith(("READ:", "SEARCH:", "CALC:", "ANSWER:")):
        return "low"
    return "low"


_REVIEW_SYSTEM = """\
あなたは汎用 AGI エージェントの実行結果を評価するレビュアーです。

ドメイン: {domain}
目標: {goal}
実行したステップ: {step}
標準出力: {stdout}
標準エラー: {stderr}
終了コード: {returncode}

【評価基準】
- 終了コード 0 かつ stdout に有用な情報があれば "success"
- エラーがあっても目標に対して十分な情報が得られていれば "success" でよい
- "goal_achieved" は目標全体が達成されたと判断できる場合のみ true
- recovery_action はツール形式で返す: ANSWER: / SEARCH: / CMD: / READ: / PYTHON: / DONE: のいずれか
- ブラウザ・GUI は使えない。ウェブ情報が必要なら SEARCH: を使う
- tree コマンドは使えない → 代わりに CMD: find . -maxdepth 3 | sort | head -60

【信頼度スコア (confidence)】
- 0.0〜1.0 の浮動小数点数
- stdout に具体的な結果がある: 0.8〜1.0
- 部分的な結果: 0.5〜0.7
- エラーや空の結果: 0.1〜0.4

【リスク評価 (risk_level)】
- "low": 読み取り専用・情報収集
- "medium": ファイル変更・コード実行
- "high": 外部サービス操作・設定変更
- "critical": 削除・フォーマット・不可逆操作

以下の JSON のみを返してください (説明不要):
{{
  "status": "success" または "failed",
  "goal_achieved": true または false,
  "summary": "日本語の簡潔な要約 (60文字以内)",
  "learned_fact": "このステップで学んだ重要な事実 (なければ null)",
  "recovery_action": "失敗時のみ: ツール形式の1行。成功時: null",
  "confidence": 0.0〜1.0,
  "risk_level": "low" / "medium" / "high" / "critical",
  "causal_effect": "このアクションが世界に与えた変化 (なければ null)"
}}\
"""


class Reviewer:
    def __init__(
        self,
        llm: Optional[MistralClient] = None,
        role: str = "worker",
        ltm: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.role = role
        self.ltm = ltm  # LongTermMemory (リカバリ戦略の履歴検索用)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if self.llm:
            review = self._llm_evaluate(step, result, state)
        else:
            review = self._static_evaluate(step, result, state)

        # リスクチェックを後付けで強制適用
        self._apply_risk_check(step, review, state)
        return review

    # ------------------------------------------------------------------
    # リスクチェック
    # ------------------------------------------------------------------

    def _apply_risk_check(self, step: str, review: Dict[str, Any], state: AgentState) -> None:
        """信頼度とリスクレベルに基づいて警告/ブロックフラグを設定する。"""
        confidence = review.get("confidence", 1.0)
        risk_level = review.get("risk_level", _assess_risk(step))
        review["risk_level"] = risk_level

        # 破壊的操作の検出
        if risk_level == "critical":
            review["requires_confirmation"] = True
            review.setdefault("warnings", []).append(
                f"[危険] 破壊的操作が検出されました: {step[:60]}"
            )

        # 低信頼度の警告
        if confidence < _CONFIDENCE_WARN_THRESHOLD and review.get("status") == "success":
            review.setdefault("warnings", []).append(
                f"[低信頼] 信頼度 {confidence:.0%} — 結果を確認することを推奨します"
            )

        if confidence < _CONFIDENCE_BLOCK_THRESHOLD:
            review.setdefault("warnings", []).append(
                f"[ブロック推奨] 信頼度が非常に低い ({confidence:.0%})"
            )
            review["low_confidence"] = True

        # 世界モデルに因果関係を記録
        causal_effect = review.get("causal_effect")
        if causal_effect:
            world_model = getattr(state, "world_model", None)
            if world_model:
                world_model.add_causal_effect(step, causal_effect)

    # ------------------------------------------------------------------
    # LLM ベースの評価
    # ------------------------------------------------------------------

    def _llm_evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if step.upper().startswith("PLAN:"):
            return {
                "status": "success",
                "summary": result.get("stdout", "計画を立案しました"),
                "goal_achieved": False,
                "recovery_action": None,
                "evidence": result.get("stdout", "")[:400],
                "improvement_hints": [],
                "confidence": 0.9,
                "risk_level": "low",
            }
        stdout = (result.get("stdout", "") or "")[:800]
        stderr = (result.get("stderr", "") or "")[:400]
        returncode = result.get("returncode", -1)

        domain = getattr(state, "domain", "general")

        assert self.llm is not None
        data = self.llm.chat_json(
            [
                {
                    "role": "user",
                    "content": _REVIEW_SYSTEM.format(
                        step=step,
                        stdout=stdout,
                        stderr=stderr,
                        returncode=returncode,
                        goal=state.user_goal,
                        domain=domain,
                    ),
                }
            ],
            temperature=0.1,
            max_tokens=1536,
        )

        if not isinstance(data, dict):
            return self._fallback_review(step, result, state)

        status = data.get("status", "failed" if not result.get("ok") else "success")
        goal_achieved = bool(data.get("goal_achieved", False))
        summary = str(data.get("summary", f"{step} を実行しました"))
        recovery_action = data.get("recovery_action") or None
        learned_fact = data.get("learned_fact") or None
        confidence = float(data.get("confidence", 0.8 if result.get("ok") else 0.3))
        risk_level = data.get("risk_level", _assess_risk(step))
        causal_effect = data.get("causal_effect") or None

        review: Dict[str, Any] = {
            "status": status,
            "summary": summary,
            "goal_achieved": goal_achieved,
            "recovery_action": recovery_action,
            "evidence": stdout[:400],
            "improvement_hints": [],
            "learned_fact": learned_fact,
            "confidence": confidence,
            "risk_level": risk_level,
            "causal_effect": causal_effect,
        }

        if goal_achieved:
            review["priority_upgrades"] = self._priority_upgrades(state)

        return review

    def _fallback_review(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if result.get("ok"):
            return {
                "status": "success",
                "summary": f"{step} を完了しました",
                "goal_achieved": False,
                "recovery_action": None,
                "evidence": (result.get("stdout", "") or "")[:400],
                "improvement_hints": [],
                "confidence": 0.7,
                "risk_level": _assess_risk(step),
            }
        return self._static_failure_review(step, result, state)

    # ------------------------------------------------------------------
    # 静的評価 (LLM なし fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_goal_completion_step(step: str) -> bool:
        """ステップがゴール達成を示すかどうかを動的に判定する。

        特定の文字列にハードコードせず、結果/要約/回答を示すインジケータで判定。
        """
        step_upper = step.upper()
        # ANSWER: や DONE: プレフィックスは明確なゴール達成
        if step_upper.startswith(("ANSWER:", "DONE:")):
            return True
        # 「まとめ」「総括」「結果」「提案」などの要約・結論キーワードを含む場合
        completion_indicators = [
            "summarize", "summary", "findings", "結果", "総括", "まとめ",
            "conclude", "conclusion", "propose", "提案", "報告",
        ]
        step_lower = step.lower()
        return any(indicator in step_lower for indicator in completion_indicators)

    def _static_evaluate(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if result["ok"]:
            is_answer = step.upper().startswith("ANSWER:")
            is_plan = step.upper().startswith("PLAN:")
            is_final = self._is_goal_completion_step(step)

            summary = result.get("stdout", "") if (is_answer or is_plan) else self._success_summary(step)
            hints = [] if (is_answer or is_plan) else self._improvement_hints(step)

            # 静的信頼度: ANSWER/PLANは高め、その他は中程度
            confidence = REVIEWER_STATIC_CONFIDENCE_PASS if (is_answer or is_plan) else REVIEWER_STATIC_CONFIDENCE_PARTIAL

            review: Dict[str, Any] = {
                "status": "success",
                "summary": summary,
                "goal_achieved": is_answer or is_final,
                "recovery_action": None,
                "evidence": (result.get("stdout", "") or "")[:400],
                "improvement_hints": hints,
                "confidence": confidence,
                "risk_level": _assess_risk(step),
            }

            if is_final:
                review["priority_upgrades"] = self._priority_upgrades(state)

            return review

        return self._static_failure_review(step, result, state)

    def _lookup_ltm_recovery(self, error_type: str) -> Optional[str]:
        """LTMから過去に成功したリカバリ戦略を検索する。

        Returns:
            過去に成功した戦略があればその文字列、なければ None。
        """
        if self.ltm is None:
            return None
        try:
            if hasattr(self.ltm, "recall_strategies"):
                strategies = self.ltm.recall_strategies(
                    f"recovery:{error_type}", limit=3
                )
                for s in strategies:
                    val = s.get("value", "")
                    if val:
                        return val
            # フォールバック: recall_similar で検索
            if hasattr(self.ltm, "recall_similar"):
                similar = self.ltm.recall_similar(
                    f"successful recovery for {error_type}", limit=3
                )
                for s in similar:
                    val = s.get("value", "")
                    if "recovery" in s.get("key", "").lower() and val:
                        return val
        except Exception:
            logger.debug("類似リカバリー戦略の参照に失敗", exc_info=True)
        return None

    def _static_failure_review(self, step: str, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        stderr = result.get("stderr", "")
        error_type = classify_error(stderr)

        remember_failure(state, step, error_type, stderr)
        error_history = state.working_memory.get("error_history", [])

        # まず LTM から過去に成功したリカバリ戦略を検索
        ltm_recovery = self._lookup_ltm_recovery(error_type)

        # 静的フォールバックマップ
        recovery_map = {
            "missing_command": "Check installed commands and PATH",
            "permission_error": "Inspect file permissions",
            "missing_python_module": "Check Python environment and pip packages",
            "connection_error": "Check running services and ports",
            "missing_file": "Inspect project structure",
            "syntax_error": "Read main entry point",
            "unknown_error": "Inspect project structure",
        }

        recovery_action = ltm_recovery or recovery_map.get(error_type, "Inspect project structure")

        can_retry_same_step = should_retry_step(step, state.failed_steps)
        can_retry_same_error = should_retry_error_type(error_type, error_history)

        if not can_retry_same_step or not can_retry_same_error:
            recovery_action = "ANSWER: リトライ上限に到達。調査結果を報告します。"

        return {
            "status": "failed",
            "summary": f"{step} で失敗しました: {error_type}",
            "goal_achieved": False,
            "recovery_action": recovery_action,
            "evidence": stderr[:400],
            "error_type": error_type,
            "improvement_hints": [
                f"失敗分類: {error_type}",
                f"次は {recovery_action} を試す",
            ],
            "confidence": REVIEWER_STATIC_CONFIDENCE_FAIL,
            "risk_level": _assess_risk(step),
        }

    # ------------------------------------------------------------------
    # ヘルパー (静的)
    # ------------------------------------------------------------------

    def _success_summary(self, step: str) -> str:
        summaries = {
            "Inspect project structure": "プロジェクト構造を確認しました",
            "Read README": "README を確認しました",
            "Read requirements": "requirements.txt を確認しました",
            "Read pyproject config": "pyproject.toml を確認しました",
            "Read main entry point": "メインの入口処理を確認しました",
            "Inspect CLI entry point": "CLI の入口処理を確認しました",
            "Inspect tests": "テスト構成を確認しました",
            "Inspect state store": "状態保存の仕組みを確認しました",
            "Inspect toolsets": "toolset 定義を確認しました",
            "Inspect tool distributions": "tool distribution 定義を確認しました",
            "Inspect model tools": "model tool 定義を確認しました",
            "Inspect time handling": "時刻処理を確認しました",
            "Inspect constants": "定数定義を確認しました",
            "Inspect mini-swe-agent path support": "mini-swe-agent 連携用パス処理を確認しました",
            "Summarize findings and propose next upgrade": "全体を総括し、次の改善候補を整理しました",
        }
        return summaries.get(step, f"{step} を確認しました")

    def _improvement_hints(self, step: str) -> List[str]:
        hints_map: Dict[str, List[str]] = {
            "Inspect project structure": ["重要ファイルの優先順位づけをさらに明確化する"],
            "Read README": ["README に起動手順と設計概要が不足していないか確認する"],
            "Summarize findings and propose next upgrade": ["Reviewer に改善優先順位づけを持たせる"],
        }
        return hints_map.get(step, ["改善候補を整理する"])

    def _priority_upgrades(self, state: AgentState) -> List[str]:
        completed = set(state.completed_steps)
        suggestions: List[str] = []

        if "Inspect CLI entry point" in completed:
            suggestions.append("CLI 引数名を --max-turns / --repo-root のようなハイフン形式へ統一する")
        if "Inspect tests" in completed:
            suggestions.append("Planner / Executor / Reviewer の単体テストを追加する")
        if "Inspect toolsets" in completed:
            suggestions.append("toolset 選択を固定定義から動的選択へ発展させる")
        if "Inspect state store" in completed:
            suggestions.append("総括結果を session に保存して次回再利用できるようにする")
        if "Read main entry point" in completed:
            suggestions.append("run_agent.py を薄くして内部 API と CLI の責務を分離する")

        if not suggestions:
            suggestions.append("Planner と Reviewer の連携を強化する")

        return suggestions[:3]
