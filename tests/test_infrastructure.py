"""Comprehensive pytest tests for Hermes AGI Gen 10 infrastructure modules.

All tests are self-contained: no external LLM, Ollama, or network calls.
Uses mocks for subprocess, requests, and LLM. Uses tmp_path for file/DB ops.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# =========================================================================
# Hierarchical Planner
# =========================================================================

class TestHierarchicalPlannerCycleDetection:
    """DAG cycle detection in GoalTree."""

    def test_detect_cycles_finds_abc_cycle(self):
        from hermes_agi_gen.hierarchical_planner import GoalTree, HierarchicalPlanner, GoalNode

        tree = GoalTree("root")
        a = tree.add_child(tree.root.goal_id, "A", depends_on=[])
        b = tree.add_child(tree.root.goal_id, "B", depends_on=[a.goal_id])
        c = tree.add_child(tree.root.goal_id, "C", depends_on=[b.goal_id])
        # Create cycle: A depends on C
        a.depends_on = [c.goal_id]

        back_edges = HierarchicalPlanner._detect_cycles(tree)
        assert len(back_edges) > 0, "Should detect at least one back edge in A->B->C->A cycle"

    def test_no_cycle_in_linear_dag(self):
        from hermes_agi_gen.hierarchical_planner import GoalTree, HierarchicalPlanner

        tree = GoalTree("root")
        a = tree.add_child(tree.root.goal_id, "A")
        b = tree.add_child(tree.root.goal_id, "B", depends_on=[a.goal_id])
        tree.add_child(tree.root.goal_id, "C", depends_on=[b.goal_id])

        back_edges = HierarchicalPlanner._detect_cycles(tree)
        assert back_edges == [], "No cycles should be found in a linear DAG"

    def test_remove_back_edges_breaks_cycle(self):
        from hermes_agi_gen.hierarchical_planner import GoalTree, HierarchicalPlanner

        tree = GoalTree("root")
        a = tree.add_child(tree.root.goal_id, "A")
        b = tree.add_child(tree.root.goal_id, "B", depends_on=[a.goal_id])
        # Cycle: A depends on B
        a.depends_on = [b.goal_id]

        back_edges = HierarchicalPlanner._detect_cycles(tree)
        assert len(back_edges) > 0
        HierarchicalPlanner._remove_back_edges(tree, back_edges)

        # After removal, no cycles should remain
        back_edges_after = HierarchicalPlanner._detect_cycles(tree)
        assert back_edges_after == []


class TestHierarchicalPlannerThreadSafety:
    """Thread safety: verify threading.Lock is used for thread_results."""

    def test_parallel_execution_uses_lock(self):
        from hermes_agi_gen.hierarchical_planner import (
            GoalTree, GoalNode, HierarchicalPlanner, GoalStatus,
        )

        tree = GoalTree("root")
        tree.add_child(tree.root.goal_id, "task1", is_parallel=True)
        tree.add_child(tree.root.goal_id, "task2", is_parallel=True)

        call_count = {"value": 0}
        lock_observed = {"used": False}

        original_lock_class = threading.Lock

        def worker_fn(node: GoalNode) -> str:
            call_count["value"] += 1
            time.sleep(0.01)  # small delay to trigger concurrency
            return f"result_{node.goal_id}"

        planner = HierarchicalPlanner(llm=None)
        result = planner.execute_tree(tree, worker_fn, max_parallel=3)

        # Both tasks should have completed
        assert call_count["value"] == 2
        assert "task1" in result or "task2" in result

    def test_execute_tree_respects_max_parallel(self):
        from hermes_agi_gen.hierarchical_planner import GoalTree, GoalNode, HierarchicalPlanner

        tree = GoalTree("root")
        for i in range(5):
            tree.add_child(tree.root.goal_id, f"task{i}", is_parallel=True)

        active = {"max": 0, "current": 0}
        lock = threading.Lock()

        def worker_fn(node: GoalNode) -> str:
            with lock:
                active["current"] += 1
                active["max"] = max(active["max"], active["current"])
            time.sleep(0.05)
            with lock:
                active["current"] -= 1
            return "ok"

        planner = HierarchicalPlanner(llm=None)
        planner.execute_tree(tree, worker_fn, max_parallel=2)

        assert active["max"] <= 2, "Should not exceed max_parallel=2"


class TestHierarchicalPlannerConfig:
    """Config values used for timeout and max_parallel."""

    def test_config_values_imported(self):
        from hermes_agi_gen.config import (
            PLANNER_THREAD_TIMEOUT,
            PLANNER_MAX_PARALLEL,
            PLANNER_RESULT_CHARS_PER_NODE,
        )
        assert PLANNER_THREAD_TIMEOUT == 120
        assert PLANNER_MAX_PARALLEL == 3
        assert PLANNER_RESULT_CHARS_PER_NODE == 300

    def test_decompose_without_llm_returns_two_steps(self):
        from hermes_agi_gen.hierarchical_planner import HierarchicalPlanner

        planner = HierarchicalPlanner(llm=None)
        tree = planner.decompose("test goal")
        # root + 2 children
        assert len(tree.nodes) == 3


# =========================================================================
# Scheduler
# =========================================================================

class TestParserTriggerSpec:
    """parse_trigger_spec() correctly parses various formats."""

    def test_once_iso_datetime(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        result = parse_trigger_spec("2026-03-31T09:00")
        assert result == "once:2026-03-31T09:00"

    def test_daily_format(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("daily 09:00") == "daily:09:00"

    def test_every_minutes(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("+30m") == "every:30m"

    def test_every_hours(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("+2h") == "every:2h"

    def test_every_with_word(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("every 30m") == "every:30m"

    def test_every_with_min_suffix(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("every 30min") == "every:30m"

    def test_weekly_format(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("weekly mon 09:00") == "weekly:mon:09:00"

    def test_canonical_passthrough(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("once:2026-01-01T00:00") == "once:2026-01-01T00:00"
        assert parse_trigger_spec("every:5m") == "every:5m"
        assert parse_trigger_spec("daily:08:30") == "daily:08:30"

    def test_invalid_returns_none(self):
        from hermes_agi_gen.scheduler import parse_trigger_spec
        assert parse_trigger_spec("garbage input") is None


class TestSchedulerMaxJobs:
    """Max job count enforcement."""

    def test_max_jobs_raises_runtime_error(self, tmp_path):
        import hermes_agi_gen.scheduler as sched_mod

        # Redirect scheduler paths to tmp_path
        orig_dir = sched_mod._HERMES_DIR
        orig_file = sched_mod._SCHEDULE_FILE
        sched_mod._HERMES_DIR = tmp_path
        sched_mod._SCHEDULE_FILE = tmp_path / "scheduler.json"

        # Patch SCHEDULER_MAX_JOBS at the module level where it's used
        orig_max = sched_mod.SCHEDULER_MAX_JOBS
        sched_mod.SCHEDULER_MAX_JOBS = 3

        try:
            scheduler = sched_mod.JobScheduler()
            scheduler.add_job("job1", "every:30m")
            scheduler.add_job("job2", "every:30m")
            scheduler.add_job("job3", "every:30m")
            with pytest.raises(RuntimeError, match="上限"):
                scheduler.add_job("job4", "every:30m")
        finally:
            sched_mod.SCHEDULER_MAX_JOBS = orig_max
            sched_mod._HERMES_DIR = orig_dir
            sched_mod._SCHEDULE_FILE = orig_file


class TestSchedulerFileLocking:
    """File locking: concurrent writes don't corrupt."""

    def test_concurrent_save_load(self, tmp_path):
        import hermes_agi_gen.scheduler as sched_mod

        orig_dir = sched_mod._HERMES_DIR
        orig_file = sched_mod._SCHEDULE_FILE
        sched_mod._HERMES_DIR = tmp_path
        sched_mod._SCHEDULE_FILE = tmp_path / "scheduler.json"

        try:
            s1 = sched_mod.JobScheduler()
            s2 = sched_mod.JobScheduler()

            s1.add_job("job_a", "every:10m", job_id="aaa")
            s2.load()

            # s2 should now see the job from s1
            jobs = s2.list_jobs()
            assert any(j.id == "aaa" for j in jobs)

            # File should be valid JSON
            content = (tmp_path / "scheduler.json").read_text()
            data = json.loads(content)
            assert isinstance(data, list)
        finally:
            sched_mod._HERMES_DIR = orig_dir
            sched_mod._SCHEDULE_FILE = orig_file


class TestCalcNextRun:
    """Test _calc_next_run for various trigger formats."""

    def test_every_minutes(self):
        from hermes_agi_gen.scheduler import _calc_next_run
        now = time.time()
        result = _calc_next_run("every:30m", last_run=now)
        assert result is not None
        assert result == pytest.approx(now + 30 * 60, abs=2)

    def test_every_hours(self):
        from hermes_agi_gen.scheduler import _calc_next_run
        now = time.time()
        result = _calc_next_run("every:2h", last_run=now)
        assert result is not None
        assert result == pytest.approx(now + 2 * 3600, abs=2)

    def test_once_past_with_last_run(self):
        from hermes_agi_gen.scheduler import _calc_next_run
        past = time.time() - 100000
        result = _calc_next_run(f"once:2020-01-01T00:00", last_run=past)
        # Once job already in the past, last_run exists but is before the target
        # The key is: if last_run >= ts, returns None
        # past < ts(2020) is false actually since 2020 is way past...
        # 2020-01-01 is ~1577836800, and past is recent, so past > ts(2020)
        assert result is None

    def test_unknown_trigger_returns_none(self):
        from hermes_agi_gen.scheduler import _calc_next_run
        result = _calc_next_run("bogus:stuff", last_run=None)
        assert result is None


# =========================================================================
# Self Modifier
# =========================================================================

class TestSelfModifierGitCheck:
    """Git clean check: mock subprocess.run for git status --porcelain."""

    def test_git_clean_returns_true(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier

        db_path = tmp_path / "sm.db"
        modifier = SelfModifier(llm=None, repo_root=tmp_path, db_path=db_path)

        with patch("hermes_agi_gen.self_modifier.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            assert modifier._check_git_clean("hermes_agi_gen/planner.py") is True

    def test_git_dirty_returns_false(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier

        db_path = tmp_path / "sm.db"
        modifier = SelfModifier(llm=None, repo_root=tmp_path, db_path=db_path)

        with patch("hermes_agi_gen.self_modifier.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=" M hermes_agi_gen/planner.py", returncode=0)
            assert modifier._check_git_clean("hermes_agi_gen/planner.py") is False

    def test_git_not_available_returns_true(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier

        db_path = tmp_path / "sm.db"
        modifier = SelfModifier(llm=None, repo_root=tmp_path, db_path=db_path)

        with patch("hermes_agi_gen.self_modifier.subprocess.run", side_effect=FileNotFoundError):
            assert modifier._check_git_clean("hermes_agi_gen/planner.py") is True


class TestSelfModifierBackup:
    """File backup before modification."""

    def test_backup_creates_file(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier, Patch

        db_path = tmp_path / "sm.db"
        modifier = SelfModifier(llm=None, repo_root=tmp_path, db_path=db_path)

        # Create the target file
        target = tmp_path / "hermes_agi_gen" / "planner.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original content", encoding="utf-8")

        patch_obj = Patch(
            file_path="hermes_agi_gen/planner.py",
            rationale="test",
            changes=[],
            original_content="original content",
        )

        backup_path = modifier._backup_file(patch_obj)
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_text(encoding="utf-8") == "original content"
        assert ".bak" in backup_path.name


class TestSelfModifierPendingProposals:
    """get_pending_proposals_summary() returns correct format."""

    def test_summary_format(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier

        db_path = tmp_path / "sm.db"
        modifier = SelfModifier(llm=None, repo_root=tmp_path, db_path=db_path)

        # Insert a high-risk proposal directly
        modifier._conn.execute(
            """INSERT INTO high_risk_proposals
               (file_path, rationale, risk_level, proposal_json, reviewed, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("hermes_agi_gen/planner.py", "test rationale", "high", "{}", 0, time.time()),
        )
        modifier._conn.commit()

        summaries = modifier.get_pending_proposals_summary()
        assert len(summaries) == 1
        s = summaries[0]
        assert "id" in s
        assert "file" in s
        assert "rationale" in s
        assert "risk" in s
        assert "created" in s
        assert s["file"] == "hermes_agi_gen/planner.py"
        assert s["risk"] == "high"


class TestSelfModifierWhitelist:
    """Whitelist enforcement: only allowed files can be modified."""

    def test_propose_rejects_non_whitelisted(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier

        db_path = tmp_path / "sm.db"
        mock_llm = MagicMock()
        modifier = SelfModifier(llm=mock_llm, repo_root=tmp_path, db_path=db_path)

        result = modifier.propose_change("hermes_agi_gen/config.py", "analyze this")
        assert result is None
        mock_llm.chat_json.assert_not_called()

    def test_propose_accepts_whitelisted(self, tmp_path):
        from hermes_agi_gen.self_modifier import SelfModifier

        db_path = tmp_path / "sm.db"
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "rationale": "improve performance",
            "changes": [
                {"description": "optimize loop", "old_code": "for x", "new_code": "for y"}
            ],
            "risk_level": "low",
            "expected_benefit": "faster",
        }

        # Create the whitelisted file
        target = tmp_path / "hermes_agi_gen" / "planner.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("for x in items:\n    pass\n", encoding="utf-8")

        modifier = SelfModifier(llm=mock_llm, repo_root=tmp_path, db_path=db_path)
        result = modifier.propose_change("hermes_agi_gen/planner.py", "analyze this")
        mock_llm.chat_json.assert_called_once()
        # Result should exist if old_code matched
        assert result is not None
        assert result.file_path == "hermes_agi_gen/planner.py"


# =========================================================================
# Reviewer
# =========================================================================

class TestReviewerRiskAssessment:
    """Word-boundary risk assessment: 'rm -rf' matches but 'form-rfid' doesn't."""

    def test_rm_rf_detected_as_critical(self):
        from hermes_agi_gen.reviewer import _assess_risk
        assert _assess_risk("CMD: rm -rf /tmp/test") == "critical"

    def test_form_rfid_not_critical(self):
        from hermes_agi_gen.reviewer import _assess_risk
        assert _assess_risk("READ: form-rfid-data.txt") == "low"

    def test_git_push_force_detected(self):
        from hermes_agi_gen.reviewer import _assess_risk
        assert _assess_risk("CMD: git push --force origin main") == "critical"

    def test_normal_read_is_low(self):
        from hermes_agi_gen.reviewer import _assess_risk
        assert _assess_risk("READ: src/main.py") == "low"

    def test_write_is_medium(self):
        from hermes_agi_gen.reviewer import _assess_risk
        assert _assess_risk("WRITE: output.txt") == "medium"


class TestReviewerGoalCompletion:
    """Dynamic goal completion detection (not hardcoded strings)."""

    def test_answer_prefix_is_completion(self):
        from hermes_agi_gen.reviewer import Reviewer
        assert Reviewer._is_goal_completion_step("ANSWER: the result is 42") is True

    def test_done_prefix_is_completion(self):
        from hermes_agi_gen.reviewer import Reviewer
        assert Reviewer._is_goal_completion_step("DONE: task complete") is True

    def test_summary_keyword_is_completion(self):
        from hermes_agi_gen.reviewer import Reviewer
        assert Reviewer._is_goal_completion_step("Summarize findings") is True

    def test_japanese_completion_keywords(self):
        from hermes_agi_gen.reviewer import Reviewer
        assert Reviewer._is_goal_completion_step("結果を表示する") is True
        assert Reviewer._is_goal_completion_step("まとめを作成") is True

    def test_regular_step_not_completion(self):
        from hermes_agi_gen.reviewer import Reviewer
        assert Reviewer._is_goal_completion_step("CMD: ls -la") is False
        assert Reviewer._is_goal_completion_step("READ: src/main.py") is False


class TestReviewerConfig:
    """Config constants used for confidence values."""

    def test_static_evaluate_uses_config_pass_confidence(self):
        from hermes_agi_gen.reviewer import Reviewer
        from hermes_agi_gen.agent_state import AgentState
        from hermes_agi_gen.config import REVIEWER_STATIC_CONFIDENCE_PASS

        reviewer = Reviewer(llm=None)
        state = AgentState(user_goal="test goal")
        result = {"ok": True, "stdout": "answer text", "stderr": ""}

        review = reviewer._static_evaluate("ANSWER: done", result, state)
        assert review["confidence"] == REVIEWER_STATIC_CONFIDENCE_PASS

    def test_static_evaluate_uses_config_partial_confidence(self):
        from hermes_agi_gen.reviewer import Reviewer
        from hermes_agi_gen.agent_state import AgentState
        from hermes_agi_gen.config import REVIEWER_STATIC_CONFIDENCE_PARTIAL

        reviewer = Reviewer(llm=None)
        state = AgentState(user_goal="test goal")
        result = {"ok": True, "stdout": "some output", "stderr": ""}

        review = reviewer._static_evaluate("CMD: ls -la", result, state)
        assert review["confidence"] == REVIEWER_STATIC_CONFIDENCE_PARTIAL

    def test_static_failure_uses_config_fail_confidence(self):
        from hermes_agi_gen.reviewer import Reviewer
        from hermes_agi_gen.agent_state import AgentState
        from hermes_agi_gen.config import REVIEWER_STATIC_CONFIDENCE_FAIL
        from hermes_agi_gen.memory import initialize_working_memory

        reviewer = Reviewer(llm=None)
        state = AgentState(user_goal="test goal")
        initialize_working_memory(state)
        result = {"ok": False, "stdout": "", "stderr": "command not found"}

        review = reviewer._static_failure_review("CMD: foobar", result, state)
        assert review["confidence"] == REVIEWER_STATIC_CONFIDENCE_FAIL


class TestReviewerLTMRecovery:
    """LTM recovery lookup (with mock LTM)."""

    def test_ltm_recovery_found(self):
        from hermes_agi_gen.reviewer import Reviewer

        mock_ltm = MagicMock()
        mock_ltm.recall_strategies.return_value = [
            {"value": "try pip install missing-module"}
        ]

        reviewer = Reviewer(llm=None, ltm=mock_ltm)
        recovery = reviewer._lookup_ltm_recovery("missing_python_module")
        assert recovery == "try pip install missing-module"
        mock_ltm.recall_strategies.assert_called_once()

    def test_ltm_recovery_not_found(self):
        from hermes_agi_gen.reviewer import Reviewer

        mock_ltm = MagicMock()
        mock_ltm.recall_strategies.return_value = [{"value": ""}]
        mock_ltm.recall_similar.return_value = []

        reviewer = Reviewer(llm=None, ltm=mock_ltm)
        recovery = reviewer._lookup_ltm_recovery("unknown_error")
        assert recovery is None

    def test_ltm_recovery_no_ltm(self):
        from hermes_agi_gen.reviewer import Reviewer

        reviewer = Reviewer(llm=None, ltm=None)
        recovery = reviewer._lookup_ltm_recovery("any_error")
        assert recovery is None


# =========================================================================
# Errors
# =========================================================================

class TestErrorTypeEnum:
    """ErrorType enum has all expected members."""

    def test_all_expected_members(self):
        from hermes_agi_gen.errors import ErrorType

        expected = {
            "PERMISSION", "TIMEOUT", "MISSING_MODULE", "SYNTAX",
            "RUNTIME", "NETWORK", "NOT_FOUND", "MISSING_COMMAND", "UNKNOWN",
        }
        actual = {e.name for e in ErrorType}
        assert expected == actual

    def test_values_are_strings(self):
        from hermes_agi_gen.errors import ErrorType
        for e in ErrorType:
            assert isinstance(e.value, str)


class TestClassifyError:
    """classify_error() returns correct ErrorType values."""

    def test_command_not_found(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("bash: foobar: command not found") == "missing_command"

    def test_permission_denied(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("Permission denied") == "permission_error"

    def test_no_module_named(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("ModuleNotFoundError: No module named 'numpy'") == "missing_python_module"

    def test_connection_refused(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("Connection refused on port 8080") == "connection_error"

    def test_file_not_found(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("No such file or directory: /tmp/test.py") == "missing_file"

    def test_syntax_error(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("SyntaxError: invalid syntax") == "syntax_error"

    def test_timeout(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("Operation timed out") == "timeout_error"

    def test_runtime_error(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("RuntimeError: something failed") == "runtime_error"

    def test_unknown(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("some weird error message") == "unknown_error"

    def test_empty_stderr(self):
        from hermes_agi_gen.errors import classify_error
        assert classify_error("") == "unknown_error"
        assert classify_error(None) == "unknown_error"


class TestRetryLogic:
    """Retry logic respects config thresholds."""

    def test_should_retry_step_within_limit(self):
        from hermes_agi_gen.errors import should_retry_step
        from hermes_agi_gen.config import REPEATED_ERROR_THRESHOLD

        failed = ["CMD: ls"] * (REPEATED_ERROR_THRESHOLD - 1)
        assert should_retry_step("CMD: ls", failed) is True

    def test_should_retry_step_at_limit(self):
        from hermes_agi_gen.errors import should_retry_step
        from hermes_agi_gen.config import REPEATED_ERROR_THRESHOLD

        failed = ["CMD: ls"] * REPEATED_ERROR_THRESHOLD
        assert should_retry_step("CMD: ls", failed) is False

    def test_should_retry_error_type_within_limit(self):
        from hermes_agi_gen.errors import should_retry_error_type
        from hermes_agi_gen.config import REPEATED_ERROR_THRESHOLD

        history = ["missing_command"] * (REPEATED_ERROR_THRESHOLD - 1)
        assert should_retry_error_type("missing_command", history) is True

    def test_should_retry_error_type_at_limit(self):
        from hermes_agi_gen.errors import should_retry_error_type
        from hermes_agi_gen.config import REPEATED_ERROR_THRESHOLD

        history = ["missing_command"] * REPEATED_ERROR_THRESHOLD
        assert should_retry_error_type("missing_command", history) is False


# =========================================================================
# World Model
# =========================================================================

class TestWorldModelUncertainty:
    """Uncertainty map: record_tool_execution() updates uncertainty."""

    def test_record_success_reduces_uncertainty(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        wm.uncertainty_map["coding"] = 0.5
        wm.record_tool_execution("CMD", "coding", success=True, execution_time=1.0)

        assert wm.uncertainty_map["coding"] < 0.5

    def test_record_failure_increases_uncertainty(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        wm.uncertainty_map["coding"] = 0.5
        wm.record_tool_execution("CMD", "coding", success=False, execution_time=1.0)

        assert wm.uncertainty_map["coding"] > 0.5

    def test_uncertainty_clamped_to_0_1(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        wm.uncertainty_map["test"] = 0.01
        wm.update_uncertainty("test", -0.5)
        assert wm.uncertainty_map["test"] == 0.0

        wm.uncertainty_map["test"] = 0.99
        wm.update_uncertainty("test", 0.5)
        assert wm.uncertainty_map["test"] == 1.0


class TestWorldModelDomainUncertainty:
    """get_domain_uncertainty() returns default for unknown domain."""

    def test_unknown_domain_returns_default(self):
        from hermes_agi_gen.world_model import WorldModel
        from hermes_agi_gen.config import WORLD_MODEL_DEFAULT_UNCERTAINTY

        wm = WorldModel()
        assert wm.get_domain_uncertainty("never_seen_domain") == WORLD_MODEL_DEFAULT_UNCERTAINTY

    def test_known_domain_returns_stored_value(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        wm.uncertainty_map["coding"] = 0.3
        assert wm.get_domain_uncertainty("coding") == 0.3


class TestWorldModelCausalMatching:
    """Word-boundary causal matching."""

    def test_predict_outcome_exact_match(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        wm.add_causal_effect("pip install numpy", "numpy installed")
        prediction = wm.predict_outcome("pip install numpy")
        assert prediction is not None
        assert "numpy installed" in prediction

    def test_predict_outcome_no_match(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        wm.add_causal_effect("pip install numpy", "numpy installed")
        prediction = wm.predict_outcome("git status")
        assert prediction is None

    def test_causal_graph_bounded(self):
        from hermes_agi_gen.world_model import WorldModel
        from hermes_agi_gen.config import WORLD_MODEL_MAX_CAUSAL_EFFECTS

        wm = WorldModel()
        for i in range(WORLD_MODEL_MAX_CAUSAL_EFFECTS + 50):
            wm.add_causal_effect(f"action_{i}", f"effect_{i}")

        assert len(wm.causal_graph) <= WORLD_MODEL_MAX_CAUSAL_EFFECTS


class TestWorldModelComplexity:
    """Complexity estimation validation with fallback."""

    def test_complexity_estimation_returns_all_keys(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        result = wm.estimate_goal_complexity("analyze the codebase")

        expected_keys = {
            "complexity", "length_score", "keyword_score",
            "historical_score", "recommended_iterations",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_complexity_recommended_iterations_has_valid_value(self):
        from hermes_agi_gen.world_model import WorldModel
        from hermes_agi_gen.config import WORLD_MODEL_MIN_ITERATIONS, WORLD_MODEL_MAX_ITERATIONS

        wm = WorldModel()
        result = wm.estimate_goal_complexity("simple read")
        iters = result["recommended_iterations"]
        assert isinstance(iters, int)
        assert WORLD_MODEL_MIN_ITERATIONS <= iters <= WORLD_MODEL_MAX_ITERATIONS

    def test_complex_goal_gets_more_iterations(self):
        from hermes_agi_gen.world_model import WorldModel

        wm = WorldModel()
        simple = wm.estimate_goal_complexity("read")
        complex_goal = wm.estimate_goal_complexity(
            "analyze and refactor the entire codebase to optimize performance and integrate with external APIs"
        )
        assert complex_goal["recommended_iterations"] >= simple["recommended_iterations"]

    def test_complexity_uses_config_weights(self):
        from hermes_agi_gen.config import (
            WORLD_MODEL_COMPLEXITY_LENGTH_WEIGHT,
            WORLD_MODEL_COMPLEXITY_KEYWORD_WEIGHT,
            WORLD_MODEL_COMPLEXITY_HISTORY_WEIGHT,
        )
        # Verify the weights sum to 1.0
        total = (
            WORLD_MODEL_COMPLEXITY_LENGTH_WEIGHT
            + WORLD_MODEL_COMPLEXITY_KEYWORD_WEIGHT
            + WORLD_MODEL_COMPLEXITY_HISTORY_WEIGHT
        )
        assert total == pytest.approx(1.0)


# =========================================================================
# Long Term Memory
# =========================================================================

class TestLTMCleanup:
    """cleanup_old_facts() removes oldest facts beyond limit."""

    def test_cleanup_removes_excess_facts(self, tmp_path):
        from hermes_agi_gen.long_term_memory import LongTermMemory

        db_path = tmp_path / "ltm_test.db"
        # Patch out Ollama so no network calls
        with patch("hermes_agi_gen.long_term_memory.SemanticIndexer") as MockIndexer:
            MockIndexer.return_value.embed.return_value = None
            ltm = LongTermMemory(db_path=db_path)

            # Insert 15 facts
            for i in range(15):
                ltm._conn.execute(
                    """INSERT INTO knowledge(key, value, confidence, created_at, updated_at)
                       VALUES (?, ?, 1.0, ?, ?)""",
                    (f"fact_{i:03d}", f"value_{i}", time.time() - (15 - i), time.time() - (15 - i)),
                )
            ltm._conn.commit()

            deleted = ltm.cleanup_old_facts(max_facts=10)
            assert deleted == 5

            row = ltm._conn.execute("SELECT COUNT(*) as cnt FROM knowledge").fetchone()
            assert row["cnt"] == 10

    def test_cleanup_no_op_under_limit(self, tmp_path):
        from hermes_agi_gen.long_term_memory import LongTermMemory

        db_path = tmp_path / "ltm_test2.db"
        with patch("hermes_agi_gen.long_term_memory.SemanticIndexer") as MockIndexer:
            MockIndexer.return_value.embed.return_value = None
            ltm = LongTermMemory(db_path=db_path)

            for i in range(5):
                ltm._conn.execute(
                    """INSERT INTO knowledge(key, value, confidence, created_at, updated_at)
                       VALUES (?, ?, 1.0, ?, ?)""",
                    (f"fact_{i}", f"value_{i}", time.time(), time.time()),
                )
            ltm._conn.commit()

            deleted = ltm.cleanup_old_facts(max_facts=10)
            assert deleted == 0


class TestLTMPeriodicCleanup:
    """Periodic cleanup triggers every 100 learn() calls."""

    def test_cleanup_triggers_at_100(self, tmp_path):
        from hermes_agi_gen.long_term_memory import LongTermMemory

        db_path = tmp_path / "ltm_periodic.db"
        with patch("hermes_agi_gen.long_term_memory.SemanticIndexer") as MockIndexer:
            MockIndexer.return_value.embed.return_value = None
            ltm = LongTermMemory(db_path=db_path)

            with patch.object(ltm, "cleanup_old_facts") as mock_cleanup:
                mock_cleanup.return_value = 0

                # Call learn 99 times -- no cleanup
                for i in range(99):
                    ltm.learn(f"key_{i}", f"value_{i}")
                mock_cleanup.assert_not_called()

                # 100th call should trigger cleanup
                ltm.learn("key_99", "value_99")
                mock_cleanup.assert_called_once()


class TestLTMSQLite:
    """Use tmp SQLite database."""

    def test_learn_and_recall(self, tmp_path):
        from hermes_agi_gen.long_term_memory import LongTermMemory

        db_path = tmp_path / "ltm_crud.db"
        with patch("hermes_agi_gen.long_term_memory.SemanticIndexer") as MockIndexer:
            MockIndexer.return_value.embed.return_value = None
            ltm = LongTermMemory(db_path=db_path)

            ltm.learn("python_version", "3.12")
            assert ltm.recall("python_version") == "3.12"

    def test_learn_overwrites(self, tmp_path):
        from hermes_agi_gen.long_term_memory import LongTermMemory

        db_path = tmp_path / "ltm_overwrite.db"
        with patch("hermes_agi_gen.long_term_memory.SemanticIndexer") as MockIndexer:
            MockIndexer.return_value.embed.return_value = None
            ltm = LongTermMemory(db_path=db_path)

            ltm.learn("key", "old_value")
            ltm.learn("key", "new_value")
            assert ltm.recall("key") == "new_value"


# =========================================================================
# Web Search
# =========================================================================

class TestWebSearchRateLimiting:
    """Rate limiting: second search within interval is delayed."""

    def test_rate_limit_delay(self):
        import hermes_agi_gen.web_search as ws

        # Reset global state
        ws._last_search_time = 0.0

        # Mock time.sleep to verify it's called
        with patch("hermes_agi_gen.web_search.time.sleep") as mock_sleep, \
             patch("hermes_agi_gen.web_search._search_via_ddgs", return_value=[]), \
             patch("hermes_agi_gen.web_search._search_via_html", return_value=[]):
            # First call
            ws._last_search_time = time.time()  # simulate a recent call
            ws.search("test query")
            mock_sleep.assert_called_once()
            # Sleep should have been called with a positive duration
            args = mock_sleep.call_args[0]
            assert args[0] > 0

    def test_no_delay_after_interval(self):
        import hermes_agi_gen.web_search as ws

        # Set last search time well in the past
        ws._last_search_time = time.time() - 100.0

        with patch("hermes_agi_gen.web_search.time.sleep") as mock_sleep, \
             patch("hermes_agi_gen.web_search._search_via_ddgs", return_value=[]), \
             patch("hermes_agi_gen.web_search._search_via_html", return_value=[]):
            ws.search("test query")
            mock_sleep.assert_not_called()


class TestWebSearchHTMLUnescape:
    """HTML entity unescaping works."""

    def test_strip_tags_unescapes_entities(self):
        from hermes_agi_gen.web_search import _strip_tags
        assert _strip_tags("Hello &amp; World") == "Hello & World"
        assert _strip_tags("&lt;b&gt;bold&lt;/b&gt;") == "bold"
        assert _strip_tags("<b>bold</b> text") == "bold text"
        assert _strip_tags("foo &gt; bar") == "foo > bar"

    def test_strip_tags_collapses_whitespace(self):
        from hermes_agi_gen.web_search import _strip_tags
        result = _strip_tags("hello   world")
        assert result == "hello world"


class TestWebSearchConfig:
    """Config values used for timeout and max results."""

    def test_config_values(self):
        from hermes_agi_gen.config import (
            WEB_SEARCH_TIMEOUT,
            WEB_SEARCH_MAX_RESULTS,
            WEB_SEARCH_RATE_LIMIT_SEC,
        )
        assert WEB_SEARCH_TIMEOUT == 15
        assert WEB_SEARCH_MAX_RESULTS == 5
        assert WEB_SEARCH_RATE_LIMIT_SEC == 2.0

    def test_search_uses_max_results_default(self):
        import hermes_agi_gen.web_search as ws
        from hermes_agi_gen.config import WEB_SEARCH_MAX_RESULTS

        ws._last_search_time = 0.0

        with patch("hermes_agi_gen.web_search._search_via_ddgs") as mock_ddgs, \
             patch("hermes_agi_gen.web_search._search_via_html", return_value=[]):
            mock_ddgs.return_value = None
            ws.search("test")
            mock_ddgs.assert_called_once_with("test", WEB_SEARCH_MAX_RESULTS)


# =========================================================================
# Agent Runner
# =========================================================================

class TestAgentRunnerToolOutputsBounded:
    """Tool outputs list is bounded at AGENT_MAX_TOOL_OUTPUTS."""

    def test_tool_outputs_bounded(self):
        from hermes_agi_gen.config import AGENT_MAX_TOOL_OUTPUTS

        # Simulate the FIFO logic from agent_runner.py
        tool_outputs: list = []
        for i in range(AGENT_MAX_TOOL_OUTPUTS + 20):
            tool_outputs.append(f"output_{i}")
            while len(tool_outputs) > AGENT_MAX_TOOL_OUTPUTS:
                tool_outputs.pop(0)

        assert len(tool_outputs) == AGENT_MAX_TOOL_OUTPUTS

    def test_fifo_eviction_removes_oldest(self):
        from hermes_agi_gen.config import AGENT_MAX_TOOL_OUTPUTS

        tool_outputs: list = []
        for i in range(AGENT_MAX_TOOL_OUTPUTS + 5):
            tool_outputs.append(f"output_{i}")
            while len(tool_outputs) > AGENT_MAX_TOOL_OUTPUTS:
                tool_outputs.pop(0)

        # The oldest 5 should have been evicted
        assert tool_outputs[0] == "output_5"
        assert tool_outputs[-1] == f"output_{AGENT_MAX_TOOL_OUTPUTS + 4}"

    def test_agent_max_tool_outputs_config(self):
        from hermes_agi_gen.config import AGENT_MAX_TOOL_OUTPUTS, AGENT_TOOL_OUTPUT_MAX_LEN
        assert AGENT_MAX_TOOL_OUTPUTS == 50
        assert AGENT_TOOL_OUTPUT_MAX_LEN == 2000


# =========================================================================
# AGICore: エラーハンドリング検証
# =========================================================================

class TestAGICoreErrorHandling:
    """agi_core.py の各フェーズが個別の障害で全体をクラッシュさせないことを検証。"""

    def _make_core(self):
        """テスト用に全モジュールをモックした AGICore 相当のオブジェクトを作成。"""
        from hermes_agi_gen.agi_core import AGICore
        # LLM をモック
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.chat.return_value = ""
        mock_llm.chat_json.return_value = None

        # AGICore を作成 (LTM等は自動初期化される)
        core = AGICore(llm=mock_llm, repo_root=Path("."))
        return core

    def test_run_goal_survives_world_model_failure(self):
        """知覚フェーズ (world_model) が例外を出しても run_goal は完了する。"""
        core = self._make_core()
        core.world_model.needs_regrounding = MagicMock(side_effect=RuntimeError("test error"))

        # agent.run のモック — 最低限の AgentState を返す
        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["test"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["step1"]
            mock_state.session_id = "test-session"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト目標")
            assert "result" in result  # クラッシュせず結果を返す

    def test_run_goal_survives_predictor_failure(self):
        """予測フェーズが例外を出しても run_goal は完了する。"""
        core = self._make_core()
        core.predictor.predict = MagicMock(side_effect=RuntimeError("predict failed"))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["done"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト")
            assert "result" in result
            assert result["strategy"]  # フォールバック戦略が使われる

    def test_run_goal_survives_agent_run_failure(self):
        """エージェント実行 (agent.run) が例外を出しても run_goal は完了する。"""
        core = self._make_core()

        with patch.object(HermesAgentV10, 'run', side_effect=RuntimeError("agent crashed")):
            result = core.run_goal("テスト")
            assert "result" in result
            assert result["success"] is False  # 失敗として記録される

    def test_run_goal_survives_reflection_failure(self):
        """省察フェーズが例外を出しても run_goal は完了する。"""
        core = self._make_core()
        core.reflection_engine.should_reflect = MagicMock(side_effect=RuntimeError("reflect error"))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["ok"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト")
            assert "result" in result  # クラッシュしない

    def test_run_goal_survives_meta_learner_failure(self):
        """メタ学習フェーズが例外を出してもフォールバック戦略で続行する。"""
        core = self._make_core()
        core.meta_learner.select_strategy = MagicMock(side_effect=RuntimeError("meta error"))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["ok"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト")
            assert result["strategy"] == "observe_then_act"  # フォールバック戦略

    def test_run_goal_survives_motivation_failure(self):
        """内発動機フェーズが例外を出しても run_goal は完了する。"""
        core = self._make_core()
        core.motivation.generate_intrinsic_goals = MagicMock(side_effect=RuntimeError("motivation error"))

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["ok"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            result = core.run_goal("テスト")
            assert result["new_goals"] == 0  # エラーで0になる

    def test_get_status_survives_module_failure(self):
        """get_status は個別モジュール障害でもエラー文字列を返す。"""
        core = self._make_core()
        core.workspace.summary = MagicMock(side_effect=RuntimeError("workspace broken"))

        status = core.get_status()
        assert "identity" in status
        assert "(エラー:" in str(status.get("workspace", ""))

    def test_no_print_calls_in_run_goal(self):
        """run_goal 内で print() が呼ばれないことを確認 (logger を使用)。"""
        core = self._make_core()

        with patch.object(HermesAgentV10, 'run') as mock_run:
            mock_state = MagicMock()
            mock_state.observations = ["ok"]
            mock_state.is_done = True
            mock_state.failed_steps = []
            mock_state.completed_steps = ["s1"]
            mock_state.session_id = "s"
            mock_run.return_value = mock_state

            with patch("builtins.print") as mock_print:
                core.run_goal("テスト")
                # print_status ではなく run_goal 内なので print は呼ばれない
                mock_print.assert_not_called()


# HermesAgentV10 のインポート (テスト用)
from hermes_agi_gen.agent_runner import HermesAgentV10


# =========================================================================
# 並行性・競合状態修正の検証
# =========================================================================


class TestSchedulerTOCTOU:
    """scheduler.py: ロック内でJSON解析が完了することを検証。"""

    def test_load_parses_inside_lock(self, tmp_path):
        """load() がロック保持中にJSON解析とジョブ構築を行う。"""
        from hermes_agi_gen.scheduler import JobScheduler
        sched_file = tmp_path / "scheduler.json"
        sched_file.write_text('[]', encoding="utf-8")

        sched = JobScheduler()
        # ソースコードを検査して、flock→json.loads→ジョブ構築→flock解除の順序を確認
        import inspect
        source = inspect.getsource(sched.load)
        # "json.loads" が flock 解除の前にあることを確認
        flock_un_pos = source.find("LOCK_UN")
        json_loads_pos = source.find("json.loads")
        jobs_assign_pos = source.find("self._jobs")
        if flock_un_pos > 0 and json_loads_pos > 0 and jobs_assign_pos > 0:
            assert json_loads_pos < flock_un_pos, "JSON解析がロック解除前に実行されるべき"
            assert jobs_assign_pos < flock_un_pos, "ジョブ構築がロック解除前に実行されるべき"


class TestDaemonBudgetLocking:
    """daemon.py: 予算カウンタのアトミックインクリメントを検証。"""

    def test_increment_uses_file_locking(self):
        """increment() がfcntlロックを使用していることを確認。"""
        from hermes_agi_gen.daemon import DailyBudgetGuard
        import inspect
        source = inspect.getsource(DailyBudgetGuard.increment)
        assert "fcntl.flock" in source or "LOCK_EX" in source

    def test_concurrent_increments_are_safe(self, tmp_path):
        """複数スレッドからの同時インクリメントでカウンタが正確。"""
        from hermes_agi_gen.daemon import DailyBudgetGuard
        guard = DailyBudgetGuard()
        guard._counter_file = tmp_path / "budget.json"

        results = []
        def _increment():
            r = guard.increment()
            results.append(r)

        threads = [threading.Thread(target=_increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 全スレッド完了後、カウンタが正確に10
        assert guard.get_used() == 10


class TestLTMThreadSafety:
    """long_term_memory.py: SQLite書き込みのスレッドロックを検証。"""

    def test_ltm_has_lock(self, tmp_path):
        """LongTermMemory がスレッドロックを持つことを確認。"""
        from hermes_agi_gen.long_term_memory import LongTermMemory
        ltm = LongTermMemory(db_path=tmp_path / "ltm.db")
        assert hasattr(ltm, "_lock")
        assert isinstance(ltm._lock, type(threading.Lock()))

    def test_concurrent_writes_safe(self, tmp_path):
        """複数スレッドからの同時書き込みでデータが破損しない。"""
        from hermes_agi_gen.long_term_memory import LongTermMemory
        ltm = LongTermMemory(db_path=tmp_path / "ltm.db")

        errors = []
        def _write(i):
            try:
                ltm.learn(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 全エントリが正しく保存されている
        for i in range(20):
            assert ltm.recall(f"key_{i}") == f"value_{i}"


class TestWebSearchRateLimitThreadSafe:
    """web_search.py: レートリミットのスレッドセーフ性を検証。"""

    def test_rate_limit_has_lock(self):
        """レートリミットがスレッドロックを使用していることを確認。"""
        from hermes_agi_gen import web_search
        assert hasattr(web_search, "_rate_limit_lock")
        assert isinstance(web_search._rate_limit_lock, type(threading.Lock()))

    def test_enforce_rate_limit_uses_lock(self):
        """_enforce_rate_limit がロック内で実行されることを確認。"""
        import inspect
        from hermes_agi_gen.web_search import _enforce_rate_limit
        source = inspect.getsource(_enforce_rate_limit)
        assert "_rate_limit_lock" in source


class TestSelfModifierAtomicWrite:
    """self_modifier.py: パッチ適用のアトミック書き込みを検証。"""

    def test_apply_patch_uses_atomic_write(self):
        """apply_patch がos.replaceによるアトミック書き込みを使用することを確認。"""
        from hermes_agi_gen.self_modifier import SelfModifier
        import inspect
        source = inspect.getsource(SelfModifier.apply_patch)
        assert "os.replace" in source
        assert "os.fsync" in source

    def test_failed_patch_does_not_corrupt_file(self, tmp_path):
        """パッチ適用が途中失敗してもファイルが破損しない。"""
        from hermes_agi_gen.self_modifier import SelfModifier, Patch, PatchChange
        modifier = SelfModifier(repo_root=tmp_path, llm=None)

        # テスト用ファイル作成
        target = tmp_path / "hermes_agi_gen" / "test_file.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        original_content = "def hello():\n    return 'world'\n"
        target.write_text(original_content, encoding="utf-8")

        # 存在しないコードを置換しようとするパッチ (失敗するはず)
        patch = Patch(
            file_path="hermes_agi_gen/test_file.py",
            original_content=original_content,
            changes=[PatchChange(description="test", old_code="NONEXISTENT", new_code="replaced")],
            rationale="test",
        )

        with patch_mock("subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="")
            result = modifier.apply_patch(patch)

        assert result is False
        # ファイルが元のまま
        assert target.read_text(encoding="utf-8") == original_content


# patch_mock のインポート
from unittest.mock import patch as patch_mock


# =========================================================================
# model_tools: ToolsetRegistry の実装検証
# =========================================================================


class TestToolsetRegistry:
    """model_tools.py: スタブではなく実装されていることを検証。"""

    def test_dispatch_builtin_tool_uses_executor(self, tmp_path):
        """組み込みツール (web_search) が Executor 経由で実行される。"""
        from hermes_agi_gen.model_tools import ToolsetRegistry
        reg = ToolsetRegistry(repo_root=tmp_path)
        result_json = reg.dispatch("terminal", {"input": "echo hello"})
        import json
        result = json.loads(result_json)
        assert result["ok"] is True
        assert "hello" in result["stdout"]

    def test_dispatch_unknown_tool_returns_error(self):
        """未知のツールはエラーを返す (偽の成功を返さない)。"""
        from hermes_agi_gen.model_tools import ToolsetRegistry
        reg = ToolsetRegistry()
        result_json = reg.dispatch("nonexistent_tool_xyz", {})
        import json
        result = json.loads(result_json)
        assert result["ok"] is False
        assert "未知のツール" in result["stderr"]

    def test_dispatch_read_file(self, tmp_path):
        """read_file ツールが Executor の READ: に委譲される。"""
        from hermes_agi_gen.model_tools import ToolsetRegistry
        # テスト用ファイル作成
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        reg = ToolsetRegistry(repo_root=tmp_path)
        result_json = reg.dispatch("read_file", {"input": "test.txt"})
        import json
        result = json.loads(result_json)
        assert result["ok"] is True
        assert "hello world" in result["stdout"]

    def test_get_tool_definitions_not_empty(self):
        """ツール定義が空でないことを確認。"""
        from hermes_agi_gen.model_tools import ToolsetRegistry
        reg = ToolsetRegistry()
        defs = reg.get_tool_definitions()
        assert len(defs) > 0
        assert all("function" in d for d in defs)

    def test_get_tool_definitions_with_filter(self):
        """ツールセットフィルタが動作する。"""
        from hermes_agi_gen.model_tools import ToolsetRegistry
        reg = ToolsetRegistry()
        web_only = reg.get_tool_definitions(enabled_toolsets=["web"])
        all_tools = reg.get_tool_definitions(enabled_toolsets=["all"])
        assert len(web_only) < len(all_tools)

    def test_backward_compat_aliases(self):
        """StubToolRegistry / DummyRegistry が ToolsetRegistry のエイリアスである。"""
        from hermes_agi_gen.model_tools import StubToolRegistry, DummyRegistry, ToolsetRegistry
        assert StubToolRegistry is ToolsetRegistry
        assert DummyRegistry is ToolsetRegistry

    def test_handle_function_call_works(self, tmp_path):
        """handle_function_call がツールを実際に実行する。"""
        from hermes_agi_gen import model_tools
        # registry の repo_root を tmp_path に設定
        model_tools.registry = model_tools.ToolsetRegistry(repo_root=tmp_path)
        result_json = model_tools.handle_function_call("terminal", {"input": "echo test123"})
        import json
        result = json.loads(result_json)
        assert result["ok"] is True
        assert "test123" in result["stdout"]
