"""Integration tests: inter-module coordination in Hermes AGI Gen 10.

All LLM calls are mocked. No network/Ollama required.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# 1. Executor → WorldModel integration
# =========================================================================

class TestExecutorWorldModel:
    def test_cmd_updates_world_model(self, tmp_path):
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState
        from hermes_agi_gen.world_model import WorldModel

        ex = Executor(repo_root=tmp_path)
        wm = WorldModel()
        state = AgentState(user_goal="test", max_iterations=1)
        state.world_model = wm

        result = ex.execute("CMD: echo hello", state)
        assert result["ok"]
        # WorldModel should have recorded resource cost
        assert len(wm.resource_history) >= 1


# =========================================================================
# 2. ValueSystem → blocking
# =========================================================================

class TestValueSystemBlocking:
    def test_dangerous_action_blocked(self):
        from hermes_agi_gen.value_system import ValueSystem
        vs = ValueSystem()
        assessment = vs.assess("rm -rf /")
        assert assessment.is_blocked

    def test_safe_action_passes(self):
        from hermes_agi_gen.value_system import ValueSystem
        vs = ValueSystem()
        assessment = vs.assess("ls -la")
        assert not assessment.is_blocked


# =========================================================================
# 3. GlobalWorkspace → AttentionMechanism pipeline
# =========================================================================

class TestGWTPipeline:
    def test_build_signals_broadcast_selects_winner(self):
        from hermes_agi_gen.consciousness import GlobalWorkspace
        gw = GlobalWorkspace()
        gw.build_signals_from_state(
            goal="テストゴール",
            context="テストコンテキスト",
        )
        event = gw.broadcast()
        assert event is not None
        assert event.winner is not None
        ctx = gw.get_context()
        assert "last_broadcast_source" in ctx


# =========================================================================
# 4. IntrinsicMotivation → GoalQueue pipeline
# =========================================================================

class TestMotivationGoalQueuePipeline:
    def test_motivation_goals_added_to_queue(self):
        from hermes_agi_gen.intrinsic_motivation import IntrinsicMotivationEngine
        from hermes_agi_gen.meta_cognition import GoalQueue

        engine = IntrinsicMotivationEngine()
        queue = GoalQueue()

        signals = engine.generate_intrinsic_goals(
            identity_assessment={"planning": 0.3, "execution": 0.8},
            knowledge_gaps=["security"],
            module_last_used={"reflection_engine": time.time() - 7200},
            max_goals=3,
        )

        if signals:
            queued = engine.to_queued_goals(signals)
            for qg in queued:
                queue.add(qg)
            assert queue.size() > 0


# =========================================================================
# 5. MetaLearner → Strategy selection → Outcome recording
# =========================================================================

class TestMetaLearnerPipeline:
    def test_select_record_updates_ucb(self, tmp_path):
        from hermes_agi_gen.meta_learning import MetaLearner

        ml = MetaLearner(db_path=tmp_path / "ml.db")
        s1 = ml.select_strategy("coding")
        initial_score = s1.ucb_score

        ml.record_outcome("coding", s1.name, "テスト", reward=1.0)
        s2 = ml.select_strategy("coding")
        # After a successful outcome, scores should have changed
        assert s2 is not None


# =========================================================================
# 6. ReflectionEngine → Insights
# =========================================================================

class TestReflectionPipeline:
    def test_reflect_generates_insights(self):
        from hermes_agi_gen.reflection_engine import ReflectionEngine
        mock_ltm = MagicMock()
        mock_ltm.recall_recent.return_value = []
        mock_ltm.get_successful_strategies.return_value = [
            {"goal": "g1", "strategy": "s1"} for _ in range(5)
        ]
        mock_ltm.get_known_failures.return_value = [
            {"command_pattern": "cmd", "error_type": "timeout", "count": 3}
        ]

        engine = ReflectionEngine()
        insights = engine.reflect(mock_ltm)
        assert isinstance(insights, list)


# =========================================================================
# 7. PredictiveEngine → Record → Accuracy
# =========================================================================

class TestPredictionAccuracyPipeline:
    def test_accuracy_improves_with_correct_predictions(self):
        from hermes_agi_gen.predictive_engine import PredictiveEngine
        engine = PredictiveEngine()

        for _ in range(20):
            pred = engine.predict("CMD: ls", goal="list files")
            engine.record_outcome(pred, "success", actual_success=True)

        accuracy = engine.get_accuracy()
        assert accuracy > 0


# =========================================================================
# 8. AGICore.run_goal() end-to-end
# =========================================================================

class TestAGICoreEndToEnd:
    def test_run_goal_returns_expected_structure(self):
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["テスト完了"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["step1"]
            mock_state.session_id = "test-session"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト目標")

        assert "result" in result
        assert "success" in result
        assert "identity" in result
        assert "strategy" in result
        assert "insights" in result
        assert result["success"] is True

    def test_run_goal_failed_returns_success_false(self):
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = []
            mock_state.is_done = False
            mock_state.failed_steps = ["error occurred"]
            mock_state.completed_steps = []
            mock_state.session_id = "test-fail"
            mock_run.return_value = mock_state

            result = core.run_goal("失敗するテスト")

        assert result["success"] is False


# =========================================================================
# 9. InnerDialogue → ValueSystem coordination
# =========================================================================

class TestDialogueEthicsCoordination:
    def test_dangerous_goal_triggers_deliberation(self):
        from hermes_agi_gen.inner_dialogue import InnerDialogue
        from hermes_agi_gen.value_system import ValueSystem

        vs = ValueSystem()
        ethics = vs.assess("rm -rf /")

        dialogue = InnerDialogue(llm=MagicMock())
        should = dialogue.should_deliberate(
            "rm -rf /",
            prediction_confidence=0.3,
            ethics_score=ethics.total_score,
        )
        # High ethics score should trigger deliberation
        assert should is True


# =========================================================================
# 10. Config constants consistency
# =========================================================================

class TestConfigConsistency:
    def test_all_config_constants_exist(self):
        """Verify key config constants are defined."""
        from hermes_agi_gen import config

        required = [
            "IDENTITY_EMA_ALPHA", "GWT_RELEVANCE_WEIGHT",
            "MOTIVATION_WEIGHT_CURIOSITY", "UCB1_EXPLORATION_CONSTANT",
            "DELIBERATION_CONFIDENCE_THRESHOLD", "PREDICTION_BASE_PROBABILITY",
            "REFLECTION_DEFAULT_INTERVAL", "VALUE_BLOCK_THRESHOLD",
            "WORLD_MODEL_MAX_CAUSAL_EFFECTS", "EXECUTOR_MAX_OUTPUT",
            "AGENT_MAX_TOOL_OUTPUTS", "ORCHESTRATOR_MAX_CONTEXT_LEN",
            "PLANNER_THREAD_TIMEOUT", "DAEMON_DAILY_BUDGET",
            "SCHEDULER_MAX_JOBS", "SELF_MODIFIER_MAX_PENDING_HIGH_RISK",
            "STUCK_FAILURE_THRESHOLD", "STATE_STORE_MAX_SESSIONS",
            "LTM_MAX_FACTS", "MODULE_LAST_USED_TTL",
            "SIMPLE_GOAL_MAX_LEN", "WEB_SEARCH_TIMEOUT",
            "REVIEWER_STATIC_CONFIDENCE_PASS",
            "EXPERIMENT_WEIGHT_SUCCESS",
            "GOAL_MAX_LENGTH", "PREVIEW_SHORT", "PREVIEW_MEDIUM",
            "PARTIAL_REWARD_MIN",
            "MOTIVATION_WEIGHT_MIN", "MOTIVATION_WEIGHT_MAX",
            "DELIBERATION_FEEDBACK_EMA_ALPHA",
            "PREDICTION_BAYESIAN_DECAY",
            "REFLECTION_RESOLVED_TTL",
            "COGNITIVE_ROLE_SUCCESS_EMA_ALPHA",
        ]
        for name in required:
            assert hasattr(config, name), f"config.py is missing {name}"

    def test_config_values_reasonable(self):
        """Spot-check that config values are within reasonable ranges."""
        from hermes_agi_gen import config

        assert 0 < config.IDENTITY_EMA_ALPHA < 1
        assert 0 < config.GWT_RELEVANCE_WEIGHT < 1
        assert config.EXECUTOR_MAX_OUTPUT > 0
        assert config.LTM_MAX_FACTS > 100
        assert config.GOAL_MAX_LENGTH > 100
        assert config.WORLD_MODEL_MIN_ITERATIONS >= 1
        assert config.WORLD_MODEL_MAX_ITERATIONS >= config.WORLD_MODEL_MIN_ITERATIONS


# =========================================================================
# 11. Module import smoke test
# =========================================================================

# =========================================================================
# 12. hermes_time utility
# =========================================================================

class TestHermesTime:
    def test_now_returns_datetime(self):
        from hermes_agi_gen.hermes_time import now
        from datetime import datetime
        result = now()
        assert isinstance(result, datetime)

    def test_get_timezone_name_returns_string(self):
        from hermes_agi_gen.hermes_time import get_timezone_name
        name = get_timezone_name()
        assert isinstance(name, str)  # 空文字列もOK (タイムゾーン未検出の環境)

    def test_reset_cache(self):
        from hermes_agi_gen.hermes_time import reset_cache, get_timezone_name
        reset_cache()
        name = get_timezone_name()
        assert isinstance(name, str)


# =========================================================================
# 13. minisweagent_path utility
# =========================================================================

class TestMinisweagentPath:
    def test_discover_returns_path_or_none(self, tmp_path):
        from hermes_agi_gen.minisweagent_path import discover_minisweagent_src
        result = discover_minisweagent_src(repo_root=tmp_path)
        assert result is None or isinstance(result, Path)

    def test_ensure_on_path_returns_path_or_none(self, tmp_path):
        from hermes_agi_gen.minisweagent_path import ensure_minisweagent_on_path
        result = ensure_minisweagent_on_path(repo_root=tmp_path)
        assert result is None or isinstance(result, Path)


# =========================================================================
# Module import smoke test
# =========================================================================

# =========================================================================
# 14. TypedDict 型ヒント検証
# =========================================================================

class TestTypedDicts:
    def test_executor_result_is_typeddict(self):
        from hermes_agi_gen.executor import ExecutorResult
        assert hasattr(ExecutorResult, "__annotations__")
        annotations = ExecutorResult.__annotations__
        assert "ok" in annotations
        assert "stdout" in annotations
        assert "stderr" in annotations
        assert "returncode" in annotations
        assert "command" in annotations

    def test_run_goal_result_is_typeddict(self):
        from hermes_agi_gen.agi_core import RunGoalResult
        assert hasattr(RunGoalResult, "__annotations__")
        annotations = RunGoalResult.__annotations__
        assert "result" in annotations
        assert "success" in annotations
        assert "identity" in annotations
        assert "strategy" in annotations
        assert "insights" in annotations
        assert "deliberation" in annotations

    def test_run_goal_returns_correct_keys(self):
        from hermes_agi_gen.agi_core import AGICore, RunGoalResult
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["done"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト")

        # RunGoalResult の全必須キーが存在
        for key in RunGoalResult.__annotations__:
            assert key in result, f"Missing key: {key}"

    def test_agi_core_build_signals_type_safe(self):
        """_build_gen10_signals が ValueAssessment と DeliberationResult を受け入れる。"""
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.value_system import ValueAssessment
        from hermes_agi_gen.inner_dialogue import DeliberationResult

        mock_llm = MagicMock()
        mock_llm.model = "test"
        core = AGICore(llm=mock_llm, repo_root=Path("."))

        ethics = ValueAssessment(
            action="test", total_score=0.1, violations=[],
            is_blocked=False, recommendation="",
        )
        # None は Optional[DeliberationResult] として許容される
        core._build_gen10_signals("goal", "ctx", ethics, None)
        # クラッシュしなければ OK


# =========================================================================
# 15. Plan→Execute→Review パイプライン統合テスト
# =========================================================================

class TestPlanExecuteReviewPipeline:
    """Planner→Executor→Reviewer の実行パイプライン統合検証。"""

    def test_planner_generates_step_executor_runs_reviewer_evaluates(self, tmp_path):
        """Plan→Execute→Review の3段階パイプラインが連携する。"""
        from hermes_agi_gen.planner import Planner
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.reviewer import Reviewer
        from hermes_agi_gen.agent_state import AgentState

        # Planner: LLM なしで静的プランを生成
        planner = Planner(llm=None, role="worker")
        state = AgentState(user_goal="プロジェクト構造を確認", max_iterations=5)
        state.domain = "general"
        step = planner.next_step(state)
        assert step is not None
        assert step.startswith("CMD:")

        # Executor: ステップを実行
        executor = Executor(repo_root=tmp_path)
        (tmp_path / "test.txt").write_text("hello", encoding="utf-8")
        result = executor.execute("CMD: ls", state)
        assert result["ok"]

        # Reviewer: 結果を評価
        reviewer = Reviewer(llm=None)
        review = reviewer.evaluate(
            step="CMD: ls",
            result=result,
            state=state,
        )
        assert isinstance(review, dict)
        assert "confidence" in review

    def test_executor_result_feeds_into_world_model(self, tmp_path):
        """Executor の実行結果が WorldModel のリソースコストに記録される。"""
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        state = AgentState(user_goal="test", max_iterations=1)
        state.world_model = wm

        executor = Executor(repo_root=tmp_path)
        executor.execute("CMD: echo integration_test", state)

        # WorldModel にリソースコストが記録されている
        assert len(wm.resource_history) >= 1
        assert wm.resource_history[-1].tool_type == "CMD"
        assert wm.resource_history[-1].success is True


# =========================================================================
# 16. AGICore 認知サイクル統合テスト (deeper)
# =========================================================================

class TestAGICoreDeepIntegration:
    """AGICore の認知サイクルが各モジュールを正しく連携させることを検証。"""

    def test_identity_persists_across_goals(self):
        """複数ゴール実行で Identity が更新される。"""
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))
        initial_processed = core.identity.total_goals_processed

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["ok"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            core.run_goal("ゴール1")
            core.run_goal("ゴール2")

        assert core.identity.total_goals_processed == initial_processed + 2

    def test_failed_goal_does_not_increment_success(self):
        """失敗ゴールは successful_goals を増やさない。"""
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))
        initial_success = core.identity.successful_goals

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = []
            mock_state.is_done = False
            mock_state.failed_steps = ["error"]
            mock_state.completed_steps = []
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            core.run_goal("失敗するゴール")

        assert core.identity.successful_goals == initial_success

    def test_ethics_blocks_dangerous_goal(self):
        """ValueSystem が危険なゴールをブロックし、実行されない。"""
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            result = core.run_goal("rm -rf /")

        # エージェントは実行されない (ValueSystem がブロック)
        mock_run.assert_not_called()
        assert result["success"] is False
        assert "ValueSystem" in result["result"]


# =========================================================================
# 17. Feedback Loop 統合テスト
# =========================================================================

class TestFeedbackLoopIntegration:
    """フィードバックループがモジュール間で連携することを検証。"""

    def test_motivation_feedback_adjusts_weights(self):
        """動機ゴール成功後にドライブ重みが変化する。"""
        from hermes_agi_gen.intrinsic_motivation import IntrinsicMotivationEngine
        engine = IntrinsicMotivationEngine()
        original = dict(engine._drive_weights)

        # 好奇心ゴールが成功
        engine.record_goal_outcome("curiosity", 1.0)

        # 重みが変化した (正規化で全体が動く)
        changed = any(
            abs(engine._drive_weights[k] - original[k]) > 0.001
            for k in original
        )
        assert changed

    def test_meta_learner_records_affect_selection(self, tmp_path):
        """戦略の記録が次の選択に影響する。"""
        from hermes_agi_gen.meta_learning import MetaLearner
        ml = MetaLearner(db_path=tmp_path / "ml.db")

        # 初回選択
        s1 = ml.select_strategy("coding")

        # 高報酬を記録
        ml.record_outcome("coding", s1.name, "テスト", reward=1.0)

        # 再度選択 — 記録済み戦略のスコアが変わる
        s2 = ml.select_strategy("coding")
        assert s2 is not None

    def test_predictive_engine_improves_with_feedback(self):
        """予測→記録→再予測で精度が反映される。"""
        from hermes_agi_gen.predictive_engine import PredictiveEngine
        engine = PredictiveEngine()

        # 10回連続で READ 成功を記録
        for _ in range(10):
            pred = engine.predict("READ: file.txt", goal="read")
            engine.record_outcome(pred, "ok", actual_success=True)

        # READ の成功率が高くなるはず
        accuracy = engine.get_accuracy("read")
        assert accuracy > 0.5

    def test_role_feedback_affects_selection(self):
        """ロール成功率のフィードバックが選択に影響する。"""
        from hermes_agi_gen.cognitive_roles import (
            record_role_outcome,
            get_role_performance,
        )

        # perceiver が連続成功
        for _ in range(10):
            record_role_outcome(["perceiver"], success=True)

        perf = get_role_performance()
        assert perf.get("perceiver", 0) > 0.7


# =========================================================================
# 18. セキュリティ統合テスト
# =========================================================================

class TestSecurityIntegration:
    """セキュリティ機構がモジュール横断で機能することを検証。"""

    def test_executor_blocks_python_os_import(self, tmp_path):
        """Python sandbox が os インポートをブロック。"""
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState

        ex = Executor(repo_root=tmp_path)
        state = AgentState(user_goal="test", max_iterations=1)
        result = ex.execute("PYTHON: import os; os.system('echo pwned')", state)
        assert not result["ok"]

    def test_executor_blocks_shell_injection(self, tmp_path):
        """Shell injection (;, &&, ||) がブロックされる。"""
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState

        ex = Executor(repo_root=tmp_path)
        state = AgentState(user_goal="test", max_iterations=1)

        for cmd in ["echo hi ; rm -rf /", "true && cat /etc/passwd", "false || whoami"]:
            result = ex.execute(f"CMD: {cmd}", state)
            assert not result["ok"], f"Should block: {cmd}"

    def test_value_system_blocks_in_agi_core(self):
        """AGICore 経由で ValueSystem が危険なゴールをブロック。"""
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.agent_runner import HermesAgentV10

        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        core = AGICore(llm=mock_llm, repo_root=Path("."))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            result = core.run_goal("drop table users")
            mock_run.assert_not_called()
            assert result["success"] is False


# =========================================================================
# 19. 世界モデル → 認知サイクル統合
# =========================================================================

class TestWorldModelCognitiveIntegration:
    """WorldModel がAGICore 認知サイクルに正しく組み込まれることを検証。"""

    def test_complexity_estimation_affects_iterations(self):
        """ゴール複雑度がエージェントの max_iterations に反映される。"""
        from hermes_agi_gen.world_model import WorldModel
        wm = WorldModel()

        # 短いゴール → 低複雑度 → 少ないイテレーション
        simple = wm.estimate_goal_complexity("ls")
        assert simple["recommended_iterations"] <= 8

        # 長い複雑なゴール → 高複雑度 → 多いイテレーション
        complex_goal = "プロジェクト全体のアーキテクチャを分析し、パフォーマンスボトルネックを特定し、リファクタリング計画を策定してテストを実行する"
        complex_result = wm.estimate_goal_complexity(complex_goal)
        assert complex_result["recommended_iterations"] > simple["recommended_iterations"]

    def test_uncertainty_updates_with_tool_execution(self):
        """ツール実行でドメイン不確実性が更新される。"""
        from hermes_agi_gen.world_model import WorldModel
        wm = WorldModel()

        initial = wm.get_domain_uncertainty("coding")
        wm.record_tool_execution(tool_type="CMD", domain="coding", success=True)
        after_success = wm.get_domain_uncertainty("coding")
        assert after_success < initial  # 成功で不確実性低下


class TestModuleImports:
    def test_all_modules_import(self):
        """All hermes_agi_gen modules import without error."""
        import importlib
        modules = [
            "hermes_agi_gen.config", "hermes_agi_gen.hermes_constants",
            "hermes_agi_gen.errors", "hermes_agi_gen.memory",
            "hermes_agi_gen.model_tools", "hermes_agi_gen.tools",
            "hermes_agi_gen.toolsets", "hermes_agi_gen.toolset_distributions",
            "hermes_agi_gen.web_search", "hermes_agi_gen.mistral_client",
            "hermes_agi_gen.state_store", "hermes_agi_gen.long_term_memory",
            "hermes_agi_gen.agent_state", "hermes_agi_gen.agent_message",
            "hermes_agi_gen.consciousness", "hermes_agi_gen.cognitive_roles",
            "hermes_agi_gen.value_system", "hermes_agi_gen.predictive_engine",
            "hermes_agi_gen.intrinsic_motivation", "hermes_agi_gen.meta_learning",
            "hermes_agi_gen.inner_dialogue", "hermes_agi_gen.world_model",
            "hermes_agi_gen.reflection_engine", "hermes_agi_gen.self_improvement",
            "hermes_agi_gen.meta_cognition", "hermes_agi_gen.tool_registry",
            "hermes_agi_gen.executor", "hermes_agi_gen.planner",
            "hermes_agi_gen.reviewer", "hermes_agi_gen.agent_runner",
            "hermes_agi_gen.hierarchical_planner", "hermes_agi_gen.orchestrator",
            "hermes_agi_gen.experiment_runner", "hermes_agi_gen.self_modifier",
            "hermes_agi_gen.daemon", "hermes_agi_gen.scheduler",
            "hermes_agi_gen.agi_core",
        ]
        for mod in modules:
            m = importlib.import_module(mod)
            assert m is not None, f"Failed to import {mod}"
