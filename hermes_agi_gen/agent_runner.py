from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_state import AgentState
from .bellman_planner import BellmanEvaluator, BellmanPlanner, QTable
from .config import AGENT_MAX_TOOL_OUTPUTS, AGENT_TOOL_OUTPUT_MAX_LEN
from .executor import Executor
from .long_term_memory import LongTermMemory
from .memory import initialize_working_memory
from .meta_cognition import MetaCognition, _is_executable_step
from .mistral_client import MistralClient
from .planner import Planner
from .predictive_engine import PredictiveEngine
from .reviewer import Reviewer
from .self_improvement import SelfImprovementEngine
from .state_store import SessionDB
from .token_codebook import TokenCodebook
from .state_store import load_latest_run_summary, save_run_summary
from .value_system import ValueSystem
from .world_model import WorldModel


logger = logging.getLogger(__name__)


class HermesAgentV10:
    """旧 Hermes の運用感と v10 の plan-act-review を合わせた軽量 AGI 版。

    llm を渡すと Planner/Reviewer が Mistral (または Ollama) を使う LLM モードで動作する。
    llm=None の場合は静的ルールによる従来モードで動作する。
    長期記憶とメタ認知により、セッションをまたいで経験を蓄積し自律的に改善する。
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        model: str = "local/mock-model",
        max_iterations: int = 8,
        session_db: SessionDB | None = None,
        source: str = "cli",
        llm: MistralClient | None = None,
        agent_role: str = "worker",
        system_prompt: str | None = None,
        ltm: LongTermMemory | None = None,
        use_bellman: bool = False,
        codebook: TokenCodebook | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.model = model
        self.max_iterations = max_iterations
        self.session_db = session_db or SessionDB()
        self.source = source
        self.llm = llm
        self.agent_role = agent_role
        self.system_prompt = system_prompt
        self.planner = Planner(llm=llm, role=agent_role)
        self.executor = Executor(self.repo_root)
        self.reviewer = Reviewer(llm=llm, role=agent_role)
        self.ltm = ltm or LongTermMemory()
        self.meta = MetaCognition(llm=llm)
        self.self_improver = SelfImprovementEngine(llm=llm)
        # Gen 6: 予測エンジン & 価値体系
        self.predictor = PredictiveEngine(ltm=self.ltm)
        self.value_system = ValueSystem()
        # Gen 10.2: トークン語彙 (省略時は配線しない)。
        self.codebook = codebook
        # Bellman 最適方程式に基づく行動選択 (オプトイン)
        self.use_bellman = use_bellman
        self.bellman_planner: BellmanPlanner | None = None
        if use_bellman:
            # Gen 10.2: codebook がある時のみ peer_reward_hook を実配線。
            # 候補 action ごとに lookup() (副作用なし) → bonus_for() を流す。
            peer_hook = None
            if self.codebook is not None:
                cb = self.codebook
                def peer_hook(action: str, _cb=cb) -> float:
                    return _cb.bonus_for(_cb.lookup(action))
            evaluator = BellmanEvaluator(
                value_system=self.value_system,
                predictor=self.predictor,
                peer_reward_hook=peer_hook,
            )
            qtable = QTable(ltm=self.ltm)
            self.bellman_planner = BellmanPlanner(
                planner=self.planner,
                evaluator=evaluator,
                qtable=qtable,
            )

    def run(self, state: AgentState) -> AgentState:
        initialize_working_memory(state)

        # 世界モデルを初期化
        if state.world_model is None:
            state.world_model = WorldModel()

        state.max_iterations = state.max_iterations or self.max_iterations
        if state.agent_role == "worker":
            state.agent_role = self.agent_role

        if not state.session_id:
            state.session_id = str(uuid.uuid4())

        state.working_memory["session_id"] = state.session_id

        # GoalQueueをLTMから復元 (セッション間の継続性)
        self.meta.goal_queue.load_from_ltm(self.ltm)

        # 前回セッションの総括を読み込む
        latest_summary = load_latest_run_summary(self.repo_root)
        if latest_summary:
            state.working_memory["latest_run_summary"] = latest_summary

        # --- 長期記憶をワーキングメモリに注入 ---
        state.working_memory["ltm_strategies"] = self.ltm.recall_strategies(
            state.user_goal, limit=5
        )
        state.working_memory["ltm_known_failures"] = self.ltm.get_known_failures(limit=10)
        # セマンティック検索による関連記憶
        state.working_memory["ltm_semantic"] = self.ltm.recall_similar(
            state.user_goal, limit=3
        )
        # カスタムツール一覧をワーキングメモリに注入
        custom_tool_descs = self.executor.tool_registry.get_tool_descriptions_for_prompt()
        if custom_tool_descs:
            state.working_memory["custom_tool_descriptions"] = custom_tool_descs

        # 自己改善: few-shot例とanti-patternを注入
        self.self_improver.inject_into_state(state)

        self.session_db.create_session(
            state.session_id,
            source=self.source,
            model=self.model,
            title=state.user_goal,
        )
        self.session_db.append_message(state.session_id, "user", state.user_goal)

        while not state.is_done and state.iteration_count < state.max_iterations:
            state.iteration_count += 1

            # --- メタ認知: 行き詰まり検出 ---
            if self.meta.is_stuck(state):
                pivot = self.meta.suggest_pivot(state, self.ltm)
                if pivot and _is_executable_step(pivot):
                    logger.info("[メタ認知] 行き詰まりを検出 → 戦略転換: %s", pivot[:60])
                    state.current_plan.insert(0, pivot)
                    state.observations.append(
                        f"[メタ認知] 行き詰まりを検出。戦略を転換します: {pivot[:80]}"
                    )
                elif pivot:
                    logger.debug("[メタ認知] pivot 候補が実行可能形式でないため破棄: %s", pivot[:80])

            active_planner = self.bellman_planner or self.planner
            step = active_planner.next_step(state, self.repo_root)
            if not step:
                state.is_done = True
                state.last_status = "finished"
                break

            # --- Gen 6: ValueSystem による倫理評価 ---
            ethics = self.value_system.assess(step)
            if ethics.is_blocked:
                logger.info("[ValueSystem] ブロック: %s", ethics.recommendation[:60])
                state.observations.append(f"[ValueSystem] {ethics.recommendation}")
                state.failed_steps.append(step)
                state.working_memory.setdefault("blocked_by_ethics", []).append(step)
                continue

            # --- Gen 6: PredictiveEngine による事前予測 ---
            prediction = self.predictor.predict(
                action=step,
                goal=state.user_goal,
            )
            if prediction.confidence > 0.6 and prediction.success_probability < 0.25:
                logger.info(
                    "[PredictiveEngine] 低成功予測 (%s): %s",
                    f"{prediction.success_probability:.0%}",
                    prediction.basis[:60],
                )
                state.observations.append(
                    f"[予測] {step[:40]} — 成功確率={prediction.success_probability:.0%}"
                )

            state.last_step = step
            logger.info("[%d/%d] %s", state.iteration_count, state.max_iterations, step[:80])
            self.session_db.append_message(state.session_id, "assistant", f"次の一手: {step}")
            result = self.executor.execute(step, state)
            if not result.get("ok") and result.get("stderr"):
                logger.warning("[エラー] %s", result['stderr'][:200])
            elif result.get("stdout") and step.upper().startswith("PYTHON:"):
                logger.debug("[出力] %s", result['stdout'][:200])

            # Gen 10: ツール出力をworking_memoryに記録 (CLI表示用)
            stdout = result.get("stdout", "")
            if stdout and stdout.strip():
                step_upper = step.upper()
                if any(step_upper.startswith(p) for p in ("FETCH:", "PYTHON:", "CMD:", "SEARCH:", "CALC:")):
                    if "tool_outputs" not in state.working_memory:
                        state.working_memory["tool_outputs"] = []
                    tool_outputs = state.working_memory["tool_outputs"]
                    tool_outputs.append(stdout[:AGENT_TOOL_OUTPUT_MAX_LEN])
                    # FIFO: remove oldest entries when exceeding max
                    while len(tool_outputs) > AGENT_MAX_TOOL_OUTPUTS:
                        tool_outputs.pop(0)

            review = self.reviewer.evaluate(step, result, state)

            # --- Gen 6: 予測誤差の記録 (学習) ---
            actual_success = review.get("status") == "success"
            self.predictor.record_outcome(
                prediction=prediction,
                actual_outcome=review.get("summary", "")[:200],
                actual_success=actual_success,
            )

            # Bellman: TD 更新で Q-table を学習
            if self.bellman_planner is not None:
                terminal = bool(review.get("goal_achieved", False)) and not state.current_plan
                self.bellman_planner.update_after_step(
                    state=state,
                    action=step,
                    success=actual_success,
                    terminal=terminal,
                    session_id=state.session_id,
                )

            state.observations.append(review["summary"])
            state.last_status = review["status"]
            state.working_memory["last_improvement_hints"] = review.get("improvement_hints", [])

            if review.get("priority_upgrades"):
                state.working_memory["priority_upgrades"] = review["priority_upgrades"]
                for i, upgrade in enumerate(review["priority_upgrades"]):
                    self.ltm.learn(
                        f"priority_upgrade_{state.session_id}_{i}",
                        upgrade,
                        session_id=state.session_id,
                    )

            # 学んだ事実を長期記憶に保存
            learned_fact = review.get("learned_fact")
            if learned_fact:
                fact_key = f"fact_{state.session_id}_{state.iteration_count}"
                self.ltm.learn(fact_key, learned_fact, session_id=state.session_id)
                state.working_memory.setdefault("learned_facts", []).append(learned_fact)

            if review.get("goal_achieved", False):
                summary_text = review.get("summary", "")
                priority_upgrades = review.get("priority_upgrades", [])
                save_run_summary(
                    self.repo_root,
                    session_id=state.session_id,
                    goal=state.user_goal,
                    summary=summary_text,
                    priority_upgrades=priority_upgrades,
                )

            self.session_db.append_message(
                state.session_id,
                "tool",
                result.get("stdout", "") or result.get("stderr", ""),
                tool_name="terminal",
            )
            self.session_db.append_message(state.session_id, "assistant", review["summary"])

            if review["status"] == "success":
                state.completed_steps.append(step)
                # 成功した戦略を長期記憶に記録
                self.ltm.log_strategy(
                    state.user_goal, step, "success", session_id=state.session_id
                )
            else:
                state.failed_steps.append(step)
                # 失敗パターンを長期記憶に記録
                error_type = review.get("error_type", "unknown")
                self.ltm.log_failure(step, error_type, session_id=state.session_id)
                self.ltm.log_strategy(
                    state.user_goal, step, "failed", session_id=state.session_id
                )

                recovery_action = review.get("recovery_action")
                if recovery_action and _is_executable_step(recovery_action):
                    state.current_plan.insert(0, recovery_action)
                elif recovery_action:
                    logger.debug(
                        "[Reviewer] recovery_action が実行可能形式でないため破棄: %s",
                        str(recovery_action)[:80],
                    )

                if state.failed_steps.count(step) >= 2:
                    state.observations.append(f"[中断] {step} が繰り返し失敗したため終了します")
                    state.is_done = True
                elif len(state.failed_steps) >= 3 and len(state.completed_steps) == 0:
                    state.observations.append("[中断] 連続失敗が続いたため終了します")
                    state.is_done = True

            # PLAN: で積まれたサブステップが残っている間は終了しない
            if review.get("goal_achieved", False) and not state.current_plan:
                state.is_done = True

        # --- セッション終了処理 ---
        self.session_db.end_session(state.session_id, "completed" if state.is_done else "stopped")

        # 自己改善: 軌跡を分析してfew-shot例を更新
        self.self_improver.analyze_session(state)

        # パフォーマンス記録 (自律改善ループ用)
        perf_score = self.meta.performance_score(state)
        if state.session_id:
            self.self_improver.record_session_performance(
                session_id=state.session_id,
                goal=state.user_goal,
                domain=state.domain or "general",
                score=perf_score,
            )

        # メタ認知: 次ゴールの自律提案
        next_goal = self.meta.generate_next_goal(state, self.ltm)
        if next_goal:
            state.suggested_next_goal = next_goal
            state.working_memory["suggested_next_goal"] = next_goal

        # ゴールキューをワーキングメモリに保存
        queued_goals = self.meta.get_queued_goals()
        if queued_goals:
            state.working_memory["goal_queue"] = [
                {"goal": g.goal, "score": g.composite_score, "source": g.source}
                for g in queued_goals[:5]
            ]

        # GoalQueueをLTMに永続化 (次回セッションで復元するため)
        self.meta.goal_queue.save_to_ltm(self.ltm)

        return state

    def chat(self, message: str) -> str:
        state = AgentState(
            user_goal=message,
            success_criteria=["次の一手を出せる", "失敗時に立て直せる", "進捗を日本語で説明できる"],
            constraints=["破壊的操作はしない", "まず読んで把握する"],
            max_iterations=self.max_iterations,
        )
        final_state = self.run(state)
        return self.render_progress(final_state)

    def render_progress(self, state: AgentState) -> str:
        lines: List[str] = []
        lines.append("=== Hermes AGI Gen 10 進捗 ===")
        lines.append(f"目的: {state.user_goal}")
        lines.append(f"反復回数: {state.iteration_count}/{state.max_iterations}")
        lines.append(f"最後のステップ: {state.last_step}")
        lines.append(f"最後の状態: {state.last_status}")
        lines.append("")
        lines.append("[完了したステップ]")
        lines.extend([f"- {step}" for step in state.completed_steps] or ["- なし"])
        lines.append("")
        lines.append("[失敗したステップ]")
        lines.extend([f"- {step}" for step in state.failed_steps] or ["- なし"])
        lines.append("")
        lines.append("[観測メモ]")
        lines.extend([f"- {obs}" for obs in state.observations] or ["- なし"])
        env = state.working_memory.get("environment", {})
        lines.append("")
        lines.append("[作業メモ]")
        lines.append(f"- cwd: {env.get('cwd')}")
        lines.append(f"- python_version: {env.get('python_version')}")
        lines.append(f"- python_executable: {env.get('python_executable')}")
        lines.append(f"- session_id: {state.session_id}")
        lines.append("")
        lines.append("[直近の改善ヒント]")
        hints = state.working_memory.get("last_improvement_hints", [])
        lines.extend([f"- {h}" for h in hints] or ["- なし"])
        lines.append("")
        lines.append("[優先改善案]")
        upgrades = state.working_memory.get("priority_upgrades", [])
        lines.extend([f"- {u}" for u in upgrades] or ["- なし"])
        lines.append("")
        lines.append("[Gen 6: 予測エンジン]")
        lines.append(f"- {self.predictor.summary()}")
        blocked_ethics = state.working_memory.get("blocked_by_ethics", [])
        if blocked_ethics:
            lines.append(f"- 倫理ブロック: {len(blocked_ethics)}件")
        lines.append("")
        lines.append("[メタ認知レポート]")
        lines.append(self.meta.reflection_summary(state))
        lines.append("")
        lines.append("[次の推奨ゴール]")
        lines.append(f"- {state.suggested_next_goal or 'なし'}")
        lines.append("")
        lines.append("[前回の総括]")
        latest = state.working_memory.get("latest_run_summary")
        if latest:
            lines.append(f"- session_id: {latest.get('session_id')}")
            lines.append(f"- created_at: {latest.get('created_at')}")
            lines.append(f"- summary: {latest.get('summary')}")
            prev_upgrades = latest.get("priority_upgrades", [])
            if prev_upgrades:
                lines.append("- 前回の優先改善案:")
                for item in prev_upgrades:
                    lines.append(f"  - {item}")
        else:
            lines.append("- なし")

        return "\n".join(lines)
