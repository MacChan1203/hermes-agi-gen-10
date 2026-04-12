"""Tests for planner.py and orchestrator.py modules."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import field

from hermes_agi_gen.agent_state import AgentState
from hermes_agi_gen.planner import (
    Planner,
    _is_conversational,
    _extract_action_from_cot,
    _extract_thinking,
    _STATIC_BOOTSTRAP,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_state(goal: str = "テスト目標", domain: str = "general", **kwargs) -> AgentState:
    """Create a minimal AgentState for testing."""
    return AgentState(user_goal=goal, domain=domain, **kwargs)


def _make_mock_llm(response: str = "ANSWER: テスト回答") -> MagicMock:
    """Create a MagicMock LLM that returns a fixed chat response."""
    llm = MagicMock()
    llm.chat.return_value = response
    llm.model = "test-model"
    return llm


# ======================================================================
# planner.py — _is_conversational
# ======================================================================

class TestIsConversational:
    """Tests for _is_conversational() helper."""

    @pytest.mark.parametrize("query", [
        "こんにちは、元気ですか",
        "あなたは何ができますか",
        "教えてください",
        "説明してほしい",
        "hello there",
        "can you help me?",
        "what is Python?",
        "how do I use git?",
        "ありがとうございます",
        "はじめまして",
    ])
    def test_conversational_queries_return_true(self, query: str):
        assert _is_conversational(query) is True

    @pytest.mark.parametrize("query", [
        "ファイルを作成してください",
        "プロジェクトのテストを実行",
        "requirements.txtを更新",
        "デプロイスクリプトを書く",
        "CSVを解析する",
        "run the build pipeline",
    ])
    def test_task_oriented_goals_return_false(self, query: str):
        assert _is_conversational(query) is False


# ======================================================================
# planner.py — _extract_action_from_cot
# ======================================================================

class TestExtractActionFromCot:
    """Tests for _extract_action_from_cot()."""

    def test_extracts_from_action_tags(self):
        response = (
            "<thinking>考え中...</thinking>\n"
            "<action>SEARCH: Python tutorial</action>"
        )
        assert _extract_action_from_cot(response) == "SEARCH: Python tutorial"

    def test_extracts_from_action_tags_multiline(self):
        response = (
            "<thinking>分析中</thinking>\n"
            "<action>\nCMD: ls -la\n</action>"
        )
        assert _extract_action_from_cot(response) == "CMD: ls -la"

    def test_extracts_from_tool_prefix_line(self):
        response = "# some comment\nSEARCH: deep learning papers"
        assert _extract_action_from_cot(response) == "SEARCH: deep learning papers"

    def test_extracts_cmd_prefix(self):
        response = "CMD: pwd && ls"
        assert _extract_action_from_cot(response) == "CMD: pwd && ls"

    def test_extracts_answer_prefix(self):
        response = "ANSWER: はい、できます。"
        assert _extract_action_from_cot(response) == "ANSWER: はい、できます。"

    def test_extracts_done_prefix(self):
        response = "DONE: 完了しました"
        assert _extract_action_from_cot(response) == "DONE: 完了しました"

    def test_returns_none_for_empty(self):
        assert _extract_action_from_cot("") is None

    def test_returns_none_for_no_action(self):
        response = "# just comments\n// and more comments"
        assert _extract_action_from_cot(response) is None

    def test_skips_comments(self):
        response = "# comment\n// another comment\nREAD: file.txt"
        assert _extract_action_from_cot(response) == "READ: file.txt"


# ======================================================================
# planner.py — _extract_thinking
# ======================================================================

class TestExtractThinking:
    """Tests for _extract_thinking()."""

    def test_extracts_thinking_content(self):
        response = "<thinking>ステップ1: 現状を把握する</thinking><action>CMD: ls</action>"
        result = _extract_thinking(response)
        assert result == "ステップ1: 現状を把握する"

    def test_extracts_multiline_thinking(self):
        response = "<thinking>\n1. 分析\n2. 計画\n3. 実行\n</thinking>"
        result = _extract_thinking(response)
        assert "1. 分析" in result
        assert "3. 実行" in result

    def test_returns_none_when_no_thinking_tag(self):
        response = "ANSWER: 直接回答"
        assert _extract_thinking(response) is None

    def test_case_insensitive(self):
        response = "<THINKING>大文字タグ</THINKING>"
        result = _extract_thinking(response)
        assert result == "大文字タグ"


# ======================================================================
# planner.py — Planner.next_step (with LLM mock)
# ======================================================================

class TestPlannerWithLLM:
    """Tests for Planner.next_step() with a mocked LLM."""

    def test_next_step_returns_action_from_llm(self):
        llm = _make_mock_llm(
            "<thinking>分析中</thinking>\n<action>CMD: pwd</action>"
        )
        planner = Planner(llm=llm, role="executor")
        state = _make_state("プロジェクト構造を調べる", domain="coding")

        step = planner.next_step(state)

        assert step == "CMD: pwd"
        llm.chat.assert_called_once()

    def test_next_step_returns_none_when_done(self):
        llm = _make_mock_llm("<action>DONE: 完了</action>")
        planner = Planner(llm=llm, role="executor")
        state = _make_state("テスト")

        step = planner.next_step(state)

        assert step is None

    def test_next_step_records_cot_reasoning(self):
        llm = _make_mock_llm(
            "<thinking>重要な推論内容</thinking>\n<action>SEARCH: test</action>"
        )
        planner = Planner(llm=llm, role="worker")
        state = _make_state("調査する")

        planner.next_step(state)

        assert "重要な推論内容" in state.working_memory.get("last_cot_reasoning", "")

    def test_next_step_pops_from_current_plan_first(self):
        llm = _make_mock_llm("should not be called")
        planner = Planner(llm=llm, role="worker")
        state = _make_state("テスト")
        state.current_plan = ["CMD: ls", "DONE: 完了"]

        step = planner.next_step(state)

        assert step == "CMD: ls"
        # LLM should NOT be called when current_plan has steps
        llm.chat.assert_not_called()

    def test_next_step_returns_plan_action(self):
        llm = _make_mock_llm("<action>PLAN: step1 || step2</action>")
        planner = Planner(llm=llm, role="strategist")
        state = _make_state("複雑なタスク")

        step = planner.next_step(state)

        assert step.startswith("PLAN:")

    def test_next_step_returns_none_when_is_done(self):
        llm = _make_mock_llm("anything")
        planner = Planner(llm=llm, role="worker")
        state = _make_state("テスト")
        state.is_done = True

        step = planner.next_step(state)

        assert step is None
        llm.chat.assert_not_called()


# ======================================================================
# planner.py — Planner.next_step (without LLM — static bootstrap)
# ======================================================================

class TestPlannerWithoutLLM:
    """Tests for Planner.next_step() without LLM (static fallback)."""

    def test_static_plan_for_general_domain(self):
        planner = Planner(llm=None, role="worker")
        state = _make_state("ファイルを確認する", domain="general")

        step = planner.next_step(state)

        assert step is not None
        assert step.startswith("CMD:")

    def test_static_plan_for_coding_domain(self):
        planner = Planner(llm=None, role="worker")
        state = _make_state("コードを改善する", domain="coding")

        step = planner.next_step(state)

        assert step is not None
        assert step.startswith("CMD:")

    def test_static_plan_for_research_domain(self):
        planner = Planner(llm=None, role="worker")
        state = _make_state("AIの最新動向を調べる", domain="research")

        step = planner.next_step(state)

        assert step is not None
        assert step.startswith("SEARCH:")

    def test_conversational_goal_gets_answer(self):
        planner = Planner(llm=None, role="worker")
        state = _make_state("こんにちは、何ができますか", domain="general")

        step = planner.next_step(state)

        assert step is not None
        assert step.startswith("ANSWER:")

    def test_static_plan_exhausts_all_steps(self):
        planner = Planner(llm=None, role="worker")
        state = _make_state("確認する", domain="general")

        steps = []
        for _ in range(10):
            step = planner.next_step(state)
            if step is None:
                break
            steps.append(step)

        assert len(steps) >= 1
        assert any("DONE:" in s for s in steps)


# ======================================================================
# planner.py — Static bootstrap plans for different domains
# ======================================================================

class TestStaticBootstrapPlans:
    """Tests for _STATIC_BOOTSTRAP domain plans."""

    def test_all_expected_domains_exist(self):
        expected = {"general", "coding", "research", "writing", "data", "ops"}
        assert expected.issubset(set(_STATIC_BOOTSTRAP.keys()))

    def test_each_plan_ends_with_done(self):
        for domain, plan in _STATIC_BOOTSTRAP.items():
            assert any("DONE:" in step for step in plan), (
                f"Domain '{domain}' plan does not contain a DONE step"
            )

    def test_coding_plan_has_readme_check(self):
        plan = _STATIC_BOOTSTRAP["coding"]
        assert any("README" in step for step in plan)

    def test_research_plan_uses_search(self):
        plan = _STATIC_BOOTSTRAP["research"]
        assert any("SEARCH:" in step for step in plan)


# ======================================================================
# planner.py — Planner._generate_summary_answer
# ======================================================================

class TestGenerateSummaryAnswer:
    """Tests for Planner._generate_summary_answer()."""

    def test_produces_answer_text(self):
        llm = _make_mock_llm("AI技術は急速に進化しています。")
        planner = Planner(llm=llm, role="worker")
        state = _make_state("AIの最新動向")
        state.working_memory["last_search_results"] = [
            {"title": "記事1", "snippet": "AIの進展"},
        ]
        state.observations = ["検索完了"]

        result = planner._generate_summary_answer(state)

        assert result.startswith("ANSWER:")
        assert "AI技術" in result
        llm.chat.assert_called_once()

    def test_returns_done_when_llm_returns_empty(self):
        llm = _make_mock_llm("")
        planner = Planner(llm=llm, role="worker")
        state = _make_state("テスト")

        result = planner._generate_summary_answer(state)

        assert result.startswith("DONE:")

    def test_uses_observations_as_evidence(self):
        llm = _make_mock_llm("まとめ結果")
        planner = Planner(llm=llm, role="worker")
        state = _make_state("テスト")
        state.observations = ["観測1", "観測2", "観測3"]

        planner._generate_summary_answer(state)

        call_args = llm.chat.call_args
        prompt_text = call_args[0][0][0]["content"]
        assert "観測" in prompt_text


# ======================================================================
# orchestrator.py — AgentOrchestrator.__init__
# ======================================================================

class TestOrchestratorInit:
    """Tests for AgentOrchestrator.__init__()."""

    def test_creates_required_components(self, tmp_path: Path):
        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            from hermes_agi_gen.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        assert orch.llm is llm
        assert orch.workspace is not None
        assert orch.value_system is not None
        assert orch.predictor is not None
        assert orch._self_model["total_runs"] == 0

    def test_hierarchical_planner_created_by_default(self, tmp_path: Path):
        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            from hermes_agi_gen.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path, use_hierarchical=True)

        assert orch.hierarchical_planner is not None

    def test_hierarchical_planner_disabled(self, tmp_path: Path):
        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            from hermes_agi_gen.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path, use_hierarchical=False)

        assert orch.hierarchical_planner is None


# ======================================================================
# orchestrator.py — Goal / Context truncation
# ======================================================================

class TestOrchestratorTruncation:
    """Tests for goal and context truncation."""

    def test_goal_truncation(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator
        from hermes_agi_gen.config import ORCHESTRATOR_MAX_GOAL_LEN

        llm = _make_mock_llm()
        long_goal = "あ" * (ORCHESTRATOR_MAX_GOAL_LEN + 1000)

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        # Mock the internals to avoid full execution
        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""

        with patch.object(orch, "_run_single_role", return_value="結果") as mock_run:
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["executor"]):
                result = orch.run(long_goal)

        # The goal passed to _run_single_role should be truncated
        called_goal = mock_run.call_args[0][0]
        assert len(called_goal) <= ORCHESTRATOR_MAX_GOAL_LEN

    def test_context_truncation_in_pipeline(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator
        from hermes_agi_gen.config import ORCHESTRATOR_MAX_CONTEXT_LEN

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        # Create a mock worker that returns a very long result
        long_result = "x" * (ORCHESTRATOR_MAX_CONTEXT_LEN + 5000)
        mock_msg = MagicMock()
        mock_msg.result = long_result
        mock_msg.receiver = "memorist"
        mock_msg.status = "success"

        orch.workspace = MagicMock()
        orch.workspace.get_context.return_value = {}

        with patch.object(orch, "_run_worker", return_value=mock_msg):
            with patch.object(orch, "_synthesize", return_value="統合結果"):
                result = orch._run_cognitive_pipeline(
                    "テスト", "", ["memorist", "executor"]
                )

        assert result == "統合結果"


# ======================================================================
# orchestrator.py — Self-model tracking
# ======================================================================

class TestOrchestratorSelfModel:
    """Tests for self_model tracking."""

    def test_total_runs_incremented(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        assert orch._self_model["total_runs"] == 0

        # Mock everything to just track the increment
        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        with patch.object(orch, "_run_single_role", return_value="成功"):
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["executor"]):
                orch.run("テスト1")
                orch.run("テスト2")

        assert orch._self_model["total_runs"] == 2

    def test_successful_runs_tracked(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        with patch.object(orch, "_run_single_role", return_value="成功した結果"):
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["executor"]):
                orch.run("テスト")

        assert orch._self_model["successful_runs"] == 1


# ======================================================================
# orchestrator.py — Ethics blocking
# ======================================================================

class TestOrchestratorEthicsBlocking:
    """Tests for ethics blocking via ValueSystem."""

    def test_high_risk_goal_blocked(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        # Make ValueSystem return high risk
        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.9)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None

        result = orch.run("危険な操作")

        assert "ValueSystem" in result
        assert "倫理基準" in result or "risk=" in result
        assert orch._self_model["blocked_actions"] == 1

    def test_low_risk_goal_not_blocked(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.1)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        with patch.object(orch, "_run_single_role", return_value="正常な結果"):
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["executor"]):
                result = orch.run("安全な操作")

        assert "ValueSystem" not in result
        assert orch._self_model["blocked_actions"] == 0


# ======================================================================
# orchestrator.py — Single-role execution path
# ======================================================================

class TestOrchestratorSingleRole:
    """Tests for single-role execution path."""

    def test_single_role_path(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        with patch.object(orch, "_run_single_role", return_value="単一ロール結果") as mock_single:
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["executor"]):
                result = orch.run("簡単なタスク")

        mock_single.assert_called_once()
        assert result == "単一ロール結果"


# ======================================================================
# orchestrator.py — Pipeline execution path
# ======================================================================

class TestOrchestratorPipeline:
    """Tests for pipeline (2-3 roles) execution path."""

    def test_pipeline_path_with_two_roles(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        with patch.object(orch, "_run_cognitive_pipeline", return_value="パイプライン結果") as mock_pipe:
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["memorist", "executor"]):
                result = orch.run("コード調査して改善する")

        mock_pipe.assert_called_once()
        assert result == "パイプライン結果"

    def test_pipeline_path_with_three_roles(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm()

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        with patch.object(orch, "_run_cognitive_pipeline", return_value="3ロール結果") as mock_pipe:
            with patch(
                "hermes_agi_gen.orchestrator.select_roles_for_goal",
                return_value=["perceiver", "strategist", "executor"],
            ):
                result = orch.run("複雑なタスク")

        mock_pipe.assert_called_once()
        roles_arg = mock_pipe.call_args[0][2]
        assert len(roles_arg) == 3
        assert result == "3ロール結果"


# ======================================================================
# orchestrator.py — run() completes without crash
# ======================================================================

class TestOrchestratorRunIntegration:
    """Integration-style test: run() with fully mocked internals."""

    def test_run_completes_without_crash(self, tmp_path: Path):
        from hermes_agi_gen.orchestrator import AgentOrchestrator

        llm = _make_mock_llm("統合された結果です。")

        with patch("hermes_agi_gen.orchestrator.SessionDB"):
            orch = AgentOrchestrator(llm=llm, repo_root=tmp_path)

        orch.value_system = MagicMock()
        orch.value_system.assess.return_value = MagicMock(total_score=0.0)
        orch.workspace = MagicMock()
        orch.workspace.broadcast.return_value = None
        orch.workspace.get_context.return_value = {}
        orch.workspace.summary.return_value = ""
        orch.predictor = MagicMock()
        orch.predictor.get_accuracy.return_value = 0.5

        mock_msg = MagicMock()
        mock_msg.result = "ワーカー結果"
        mock_msg.receiver = "executor"
        mock_msg.status = "success"

        with patch.object(orch, "_run_worker", return_value=mock_msg):
            with patch("hermes_agi_gen.orchestrator.select_roles_for_goal", return_value=["executor"]):
                result = orch.run("テスト目標")

        assert isinstance(result, str)
        assert len(result) > 0
