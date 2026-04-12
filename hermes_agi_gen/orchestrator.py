"""AGI マルチ認知ロール オーケストレーター (Gen 6)。

Gen 5 との主な差分:
- 3ロール (researcher/developer/critic) → 8認知ロール
- GlobalWorkspace (GWT) による統合的認知
- ValueSystem による倫理的行動評価
- PredictiveEngine による事前予測
- 目標に応じた動的ロール選択 (select_roles_for_goal)
- 自己モデル (self_model) によるシステム状態把握

使い方:
    from hermes_agi_gen import AgentOrchestrator, MistralClient

    llm = MistralClient(model="qwen3")   # Ollama ローカル
    orch = AgentOrchestrator(llm=llm)
    result = orch.run("このプロジェクトの構造を調べて改善案をまとめてください")
    print(result)
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_message import AgentMessage
from .config import ORCHESTRATOR_MAX_CONTEXT_LEN, ORCHESTRATOR_MAX_GOAL_LEN, ORCHESTRATOR_RESULT_TRUNCATE, PLANNER_MAX_PARALLEL, PLANNER_THREAD_TIMEOUT
from .agent_runner import HermesAgentV9
from .agent_state import AgentState
from .cognitive_roles import (
    COGNITIVE_ROLES,
    ROLE_SYSTEM_PROMPTS,
    decompose_into_roles,
    select_roles_for_goal,
)
from .consciousness import GlobalWorkspace, SignalSource, WorkspaceSignal
from .hierarchical_planner import GoalNode, GoalTree, HierarchicalPlanner
from .mistral_client import MistralClient
from .predictive_engine import PredictiveEngine
from .state_store import SessionDB
from .value_system import ValueSystem

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 合成プロンプト
# ------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
あなたはローカルマシン上で動作するAGIシステムのオーケストレーターです。
ユーザーの目標を 1〜3 個のサブタスクに分解してください。
各サブタスクには以下のいずれかのロールを割り当ててください:
- perceiver:    入力の意図・要件を明確化する
- memorist:     ローカルファイル・コードを調査して情報を収集する
- ethicist:     実行計画の安全性・倫理的問題を評価する
- strategist:   実行計画を立案する
- innovator:    創造的・代替的アプローチを提案する
- executor:     コード実行・ファイル操作・テスト実行を担当する
- critic:       成果物の評価・確認・改善提案を行う
- goal_manager: 追加ゴールの特定と優先付けを行う

【重要】インターネット・Web検索は使えません。すべてローカル操作のみです。
必ず JSON 配列のみで返してください (説明不要):
[{"role": "memorist", "task": "..."}, {"role": "executor", "task": "..."}, ...]\
"""

_SYNTHESIZE_SYSTEM = """\
あなたはAGIシステムのオーケストレーターです。
複数の専門認知エージェントの実行結果を受け取り、最終的な成果を日本語で簡潔にまとめてください。
各エージェントの貢献を統合し、ユーザーへの明確な回答を生成してください。\
"""

_SELF_MODEL_TEMPLATE = """\
[AGI自己モデル]
バージョン: Hermes AGI Gen 6
アーキテクチャ: GlobalWorkspace + 8認知ロール + ValueSystem + PredictiveEngine
認知ロール: {roles}
予測精度: {prediction_accuracy:.1%}
ゴールキュー: {goal_queue_size}件
倫理評価: {ethics_summary}
ワークスペース: {workspace_summary}
"""


class AgentOrchestrator:
    """AGI マルチ認知ロール オーケストレーター。

    GlobalWorkspace Theory に基づく統合的認知を実現する。
    8つの専門認知ロールが協調して複雑な目標を達成する。

    Args:
        llm: MistralClient インスタンス
        repo_root: 作業ディレクトリ
        session_db: セッション DB (省略時は新規作成)
        max_worker_iterations: 各ワーカーの最大イテレーション数
        use_hierarchical: 階層的プランニングを使用するか
    """

    def __init__(
        self,
        llm: MistralClient,
        repo_root: str | Path = ".",
        session_db: Optional[SessionDB] = None,
        max_worker_iterations: int = 4,
        use_hierarchical: bool = True,
    ) -> None:
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.session_db = session_db or SessionDB()
        self.max_worker_iterations = max_worker_iterations
        self.orchestrator_session_id = str(uuid.uuid4())

        # Gen 6 新モジュール
        self.workspace = GlobalWorkspace()
        self.value_system = ValueSystem()
        self.predictor = PredictiveEngine()
        self.hierarchical_planner = HierarchicalPlanner(llm=llm) if use_hierarchical else None

        # 自己モデル: システム自身の状態追跡
        self._self_model: Dict[str, Any] = {
            "total_runs": 0,
            "successful_runs": 0,
            "blocked_actions": 0,
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, goal: str, context: str = "") -> str:
        """目標を受け取り、認知ロールに委任して最終結果を返す。

        フロー:
        1. GlobalWorkspace でシグナルを統合 → 優先度を決定
        2. ValueSystem で倫理評価
        3. 最適なロール構成を選択
        4. 各ロールが順次実行
        5. 結果を統合
        """
        self._self_model["total_runs"] += 1

        # --- Input validation: truncate goal if too long ---
        if len(goal) > ORCHESTRATOR_MAX_GOAL_LEN:
            logger.warning(
                "Goal truncated from %d to %d chars", len(goal), ORCHESTRATOR_MAX_GOAL_LEN
            )
            goal = goal[:ORCHESTRATOR_MAX_GOAL_LEN]

        self.session_db.create_session(
            self.orchestrator_session_id,
            source="orchestrator_gen6",
            model=self.llm.model,
            title=f"[AGI Orchestrator] {goal}",
        )
        self.session_db.append_message(self.orchestrator_session_id, "user", goal)

        # --- Step 1: GlobalWorkspace にシグナルを送信 ---
        ethics_risk = self._quick_ethics_check(goal)
        self.workspace.build_signals_from_state(
            goal=goal,
            context=context,
            is_stuck=False,
            value_risk=ethics_risk,
        )
        broadcast = self.workspace.broadcast()
        if broadcast:
            logger.info(
                "[GlobalWorkspace] 注意焦点: [%s] %s",
                broadcast.winner.source.value,
                broadcast.winner.content[:60],
            )

        # --- Step 2: 倫理評価 ---
        if ethics_risk >= 0.8:
            blocked_msg = (
                f"[ValueSystem] この目標は倫理基準に違反する可能性があります (risk={ethics_risk:.2f})。"
                "安全な代替手段を検討してください。"
            )
            self.session_db.append_message(
                self.orchestrator_session_id, "assistant", blocked_msg
            )
            self._self_model["blocked_actions"] += 1
            return blocked_msg

        # --- Step 3: ロール構成の選択 ---
        roles = select_roles_for_goal(goal, context)
        logger.info("[AGI] 認知ロール構成: %s", " → ".join(roles))

        # Gen 6: ロール数に応じて実行パスを選択
        # - 1ロール: 単純クエリ → 直接実行（高速）
        # - 2〜3ロール: 認知パイプライン
        # - 4ロール以上 + 階層プランナーあり: 階層型ゴールツリー
        if len(roles) == 1:
            result = self._run_single_role(goal, context, roles[0])
        elif self.hierarchical_planner is not None and len(roles) >= 4:
            result = self._run_hierarchical(goal, context)
        else:
            result = self._run_cognitive_pipeline(goal, context, roles)

        # 自己モデルを更新
        if result and "エラー" not in result[:50]:
            self._self_model["successful_runs"] += 1

        # 自己モデルをログに記録
        self_model_log = self._generate_self_model_report(roles)
        self.session_db.append_message(
            self.orchestrator_session_id, "assistant", self_model_log
        )

        self.session_db.end_session(self.orchestrator_session_id, "completed")
        return result

    def run_with_prediction(self, goal: str, context: str = "") -> Dict[str, Any]:
        """予測情報付きで目標を実行する。

        Returns:
            {"result": str, "predictions": [...], "prediction_accuracy": float}
        """
        # 目標に対する事前予測
        prediction = self.predictor.predict(
            action=f"PLAN: {goal}",
            goal=goal,
            context=context,
        )
        logger.info(
            "[PredictiveEngine] 予測: %s (確信=%.0f%%)",
            prediction.predicted_outcome,
            prediction.confidence * 100,
        )

        if not prediction.should_proceed:
            logger.warning(
                "[PredictiveEngine] 警告: 成功確率=%.0f%% — 慎重に実行します",
                prediction.success_probability * 100,
            )

        result = self.run(goal, context)

        # 予測を記録
        actual_success = bool(result and len(result) > 50)
        record = self.predictor.record_outcome(prediction, result[:ORCHESTRATOR_RESULT_TRUNCATE], actual_success)

        return {
            "result": result,
            "prediction": {
                "predicted_outcome": prediction.predicted_outcome,
                "success_probability": prediction.success_probability,
                "confidence": prediction.confidence,
                "actual_success": actual_success,
                "prediction_error": record.prediction_error,
            },
            "prediction_accuracy": self.predictor.get_accuracy(),
        }

    # ------------------------------------------------------------------
    # 認知パイプライン実行
    # ------------------------------------------------------------------

    def _run_single_role(self, goal: str, context: str, role: str) -> str:
        """単純クエリ用の単一ロール直接実行（パイプラインオーバーヘッドなし）。"""
        logger.info("[AGI] 単一ロール実行: %s", role)
        msg = AgentMessage(
            sender="orchestrator",
            receiver=role,
            task=goal,
            context=context[:300] if context else "",
            session_id=self.orchestrator_session_id,
        )
        completed = self._run_worker(msg)
        result = completed.result or "（結果なし）"
        self.session_db.append_message(
            self.orchestrator_session_id, "assistant", result
        )
        return result

    def _run_cognitive_pipeline(
        self, goal: str, context: str, roles: List[str]
    ) -> str:
        """8認知ロールのパイプラインで目標を実行する。

        各ロールは前のロールの結果をコンテキストとして受け取る。
        """
        accumulated_context = context
        results: List[AgentMessage] = []

        for i, role in enumerate(roles, 1):
            logger.info("[Pipeline %d/%d] %s: 実行中...", i, len(roles), role)

            # 前のロールの結果をコンテキストに追加
            # sender は常に "orchestrator" なので receiver（ロール名）を使う
            if results:
                prev_result = results[-1]
                accumulated_context = (
                    f"前のステップ ({prev_result.receiver}) の結果:\n"
                    f"{prev_result.result or '（なし）'}\n\n"
                    + accumulated_context
                )

            # --- Truncate accumulated context if too long ---
            if len(accumulated_context) > ORCHESTRATOR_MAX_CONTEXT_LEN:
                accumulated_context = (
                    accumulated_context[:2000]
                    + "\n\n[...truncated...]\n\n"
                    + accumulated_context[-2000:]
                )

            # タスクを構築
            task = self._build_role_task(role, goal, accumulated_context)

            msg = AgentMessage(
                sender="orchestrator",
                receiver=role,
                task=task,
                context=accumulated_context[:500] if accumulated_context else "",
                session_id=self.orchestrator_session_id,
            )

            completed = self._run_worker(msg)
            results.append(completed)

            # GlobalWorkspace にシグナルを送信
            self.workspace.receive(WorkspaceSignal(
                source=self._role_to_signal_source(role),
                content=(completed.result or "（結果なし）")[:200],
                relevance=0.7,
                urgency=0.5,
                confidence=0.7 if completed.status == "success" else 0.3,
                tags=[role, "execution_result"],
            ))

            self.session_db.append_message(
                self.orchestrator_session_id, "tool",
                f"[{role}] {completed.result or '（結果なし）'}",
                tool_name=role,
            )

            logger.info("[Pipeline %d/%d] %s: 完了 (status=%s)", i, len(roles), role, completed.status)

        # 中間ブロードキャスト（全ロール完了後）
        self.workspace.broadcast()

        final = self._synthesize(goal, results)
        self.session_db.append_message(
            self.orchestrator_session_id, "assistant", final
        )
        return final

    def _run_hierarchical(self, goal: str, context: str = "") -> str:
        """階層的GoalTreeを使って目標を実行する (gen-5 から継承・強化)。"""
        logger.info("[AGI Orchestrator] 階層的ゴールツリーを生成中...")
        assert self.hierarchical_planner is not None
        tree = self.hierarchical_planner.decompose(goal, context=context)

        tree_summary = tree.summary()
        self.session_db.append_message(
            self.orchestrator_session_id, "assistant",
            f"ゴールツリー: {tree_summary}"
        )
        logger.info("[AGI Orchestrator] %s", tree_summary)

        def worker_fn(node: GoalNode) -> str:
            # ValueSystem で倫理評価
            risk = self.value_system.assess(node.goal).total_score
            if risk >= 0.8:
                return f"[ValueSystem] このノードはブロックされました (risk={risk:.2f})"

            msg = AgentMessage(
                sender="orchestrator",
                receiver=node.role,
                task=node.goal,
                context="\n".join(
                    f"[{n.role}の結果] {n.result[:ORCHESTRATOR_RESULT_TRUNCATE]}"
                    for n in tree._nodes.values()
                    if n.goal_id in node.depends_on and n.result
                ),
                session_id=self.orchestrator_session_id,
            )
            logger.info("  [GoalNode:%s] %s: %s", node.goal_id, node.role, node.goal[:60])
            completed = self._run_worker(msg)
            self.session_db.append_message(
                self.orchestrator_session_id, "tool",
                f"[{node.role}:{node.goal_id}] {completed.result or '（結果なし）'}",
                tool_name="worker",
            )
            return completed.result or ""

        self.hierarchical_planner.execute_tree(tree, worker_fn, max_parallel=PLANNER_MAX_PARALLEL)

        messages = []
        for node in tree._nodes.values():
            if node.goal_id == tree.root.goal_id:
                continue
            msg = AgentMessage(
                sender=node.role, receiver="orchestrator",
                task=node.goal, result=node.result,
                status=node.status.value,
                session_id=self.orchestrator_session_id,
            )
            messages.append(msg)

        final = self._synthesize(goal, messages)
        self.session_db.append_message(self.orchestrator_session_id, "assistant", final)
        return final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_role_task(self, role: str, goal: str, context: str) -> str:
        """ロールに応じたタスク文字列を構築する。"""
        role_tasks = {
            "perceiver": f"目標「{goal}」の意図・要件・制約を明確化してください",
            "memorist": f"「{goal}」に関連するローカルファイル・コード・情報を調査してください",
            "ethicist": f"「{goal}」の実行計画の安全性・倫理的問題を評価してください",
            "strategist": f"「{goal}」を達成するための詳細な実行計画を立案してください",
            "innovator": f"「{goal}」に対する創造的・代替的アプローチを提案してください",
            "executor": f"「{goal}」を実際に実行してください: {context[:100] if context else ''}",
            "critic": f"「{goal}」の実行結果を評価し、品質と改善点を報告してください",
            "goal_manager": f"「{goal}」に関連する追加目標を特定し、優先順位を付けてください",
        }
        return role_tasks.get(role, f"{goal} を担当してください")

    def _quick_ethics_check(self, text: str) -> float:
        """テキストの倫理リスクを簡易評価する (0.0=安全, 1.0=危険)。"""
        assessment = self.value_system.assess(text)
        return assessment.total_score

    def _run_worker(self, msg: AgentMessage) -> AgentMessage:
        """ワーカーエージェントを生成してサブタスクを実行する。"""
        role = msg.receiver
        system_prompt = ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS["executor"])
        cognitive_role = COGNITIVE_ROLES.get(role)
        max_iter = cognitive_role.max_iterations if cognitive_role else self.max_worker_iterations

        agent = HermesAgentV9(
            repo_root=self.repo_root,
            model=self.llm.model,
            max_iterations=max_iter,
            session_db=self.session_db,
            source=f"cognitive/{role}",
            llm=self.llm,
            agent_role=role,
            system_prompt=system_prompt,
        )

        state = AgentState(
            user_goal=msg.task,
            success_criteria=["タスクを完了できた", "結果を日本語で説明できる"],
            constraints=["破壊的操作はしない", "まず読んで把握する"],
            max_iterations=max_iter,
            agent_role=role,
            parent_session_id=self.orchestrator_session_id,
        )

        final_state = agent.run(state)
        result_text = "\n".join(final_state.observations) if final_state.observations else "（観測なし）"

        msg.result = result_text
        msg.status = "success" if final_state.is_done else "partial"
        return msg

    def _synthesize(self, goal: str, results: List[AgentMessage]) -> str:
        """全ワーカーの結果を統合して最終回答を生成する。"""
        results_text = "\n\n".join(
            f"=== [{r.receiver}] の結果 ===\n{r.result or '（なし）'}" for r in results
        )
        # GlobalWorkspace の共有コンテキストを追加
        ws_context = self.workspace.get_context()
        ws_summary = ws_context.get("last_broadcast_content", "")

        response = self.llm.chat(
            [
                {"role": "system", "content": _SYNTHESIZE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"目標: {goal}\n\n"
                        f"{results_text}\n\n"
                        f"ワークスペース状態: {ws_summary[:100] if ws_summary else 'なし'}\n\n"
                        "上記の結果をまとめてください。"
                    ),
                },
            ],
            max_tokens=1024,
        )
        return response or "（統合結果を生成できませんでした）"

    def _role_to_signal_source(self, role: str) -> SignalSource:
        """ロール名を SignalSource に変換する。"""
        mapping = {
            "perceiver": SignalSource.PERCEIVER,
            "strategist": SignalSource.STRATEGIST,
            "executor": SignalSource.EXECUTOR,
            "critic": SignalSource.CRITIC,
            "memorist": SignalSource.MEMORIST,
            "goal_manager": SignalSource.GOAL_MANAGER,
            "innovator": SignalSource.INNOVATOR,
            "ethicist": SignalSource.ETHICIST,
        }
        return mapping.get(role, SignalSource.EXECUTOR)

    def _generate_self_model_report(self, roles: List[str]) -> str:
        """自己モデルレポートを生成する。"""
        total = self._self_model["total_runs"]
        success = self._self_model["successful_runs"]
        success_rate = success / total if total > 0 else 0.0

        return _SELF_MODEL_TEMPLATE.format(
            roles=", ".join(roles),
            prediction_accuracy=self.predictor.get_accuracy(),
            goal_queue_size=0,  # 将来の拡張用
            ethics_summary=f"評価={total}件, ブロック={self._self_model['blocked_actions']}件",
            workspace_summary=self.workspace.summary(),
        ).strip() + f"\n成功率: {success_rate:.0%} ({success}/{total})"

    def get_system_status(self) -> Dict[str, Any]:
        """システム全体のステータスを返す。"""
        return {
            "version": "Hermes AGI Gen 6",
            "self_model": self._self_model,
            "workspace": self.workspace.summary(),
            "predictor": self.predictor.summary(),
            "value_system": self.value_system.summary(),
            "cognitive_roles": list(COGNITIVE_ROLES.keys()),
            "attention_stats": self.workspace.attention.source_attention_stats(),
        }
