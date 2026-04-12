"""Tests for experiment_runner.py and self_improvement.py modules."""
from __future__ import annotations

import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_agi_gen.config import (
    EXPERIMENT_DIVERSITY_SCALE,
    EXPERIMENT_KNOWLEDGE_SCALE,
    EXPERIMENT_WEIGHT_ACCURACY,
    EXPERIMENT_WEIGHT_BREADTH,
    EXPERIMENT_WEIGHT_DIVERSITY,
    EXPERIMENT_WEIGHT_SUCCESS,
)
from hermes_agi_gen.experiment_runner import (
    ExperimentMetrics,
    ExperimentResult,
    ExperimentRunner,
    _INSIGHT_TO_TARGET,
)
from hermes_agi_gen.self_improvement import SelfImprovementEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_insight(
    category: str = "weakness",
    content: str = "テスト洞察",
    confidence: float = 0.8,
    actionable: bool = True,
    source: str = "test",
):
    return SimpleNamespace(
        category=category,
        content=content,
        confidence=confidence,
        actionable=actionable,
        source=source,
    )


def _make_mock_agi_core(tmp_path: Path) -> MagicMock:
    """Create a mock AGICore with the dependencies ExperimentRunner needs."""
    core = MagicMock()
    core.ltm = MagicMock()
    core.predictor = MagicMock()
    core.predictor.get_accuracy.return_value = 0.7
    core.reflection_engine = MagicMock()
    core.reflection_engine.compute_growth_metrics.return_value = SimpleNamespace(
        success_rate=0.6,
        knowledge_breadth=50,
        strategy_diversity=10,
        reflection_count=5,
    )
    core.self_modifier = MagicMock()
    core.self_modifier.apply_patch.return_value = True
    core.self_modifier.run_tests.return_value = SimpleNamespace(passed=True, duration=1.0)
    core.identity = SimpleNamespace(success_rate=0.7, total_goals_processed=10)
    return core


def _make_agent_state(
    session_id: str = "sess-001",
    user_goal: str = "test goal",
    domain: str = "general",
    completed_steps: list | None = None,
    failed_steps: list | None = None,
    observations: list | None = None,
    working_memory: dict | None = None,
):
    state = SimpleNamespace(
        session_id=session_id,
        user_goal=user_goal,
        domain=domain,
        completed_steps=completed_steps or [],
        failed_steps=failed_steps or [],
        observations=observations or [],
        working_memory=working_memory if working_memory is not None else {},
    )
    return state


# ======================================================================
# ExperimentMetrics tests
# ======================================================================

class TestExperimentMetrics:
    def test_score_uses_config_weights(self):
        m = ExperimentMetrics(
            success_rate=0.8,
            prediction_accuracy=0.6,
            knowledge_breadth=50,
            strategy_diversity=10,
        )
        expected = (
            0.8 * EXPERIMENT_WEIGHT_SUCCESS
            + 0.6 * EXPERIMENT_WEIGHT_ACCURACY
            + min(1.0, 50 / EXPERIMENT_KNOWLEDGE_SCALE) * EXPERIMENT_WEIGHT_BREADTH
            + min(1.0, 10 / EXPERIMENT_DIVERSITY_SCALE) * EXPERIMENT_WEIGHT_DIVERSITY
        )
        assert m.score() == pytest.approx(expected)

    def test_score_all_zeros(self):
        m = ExperimentMetrics(
            success_rate=0.0,
            prediction_accuracy=0.0,
            knowledge_breadth=0,
            strategy_diversity=0,
        )
        assert m.score() == pytest.approx(0.0)

    def test_score_all_ones_or_max(self):
        m = ExperimentMetrics(
            success_rate=1.0,
            prediction_accuracy=1.0,
            knowledge_breadth=EXPERIMENT_KNOWLEDGE_SCALE * 2,  # exceeds scale
            strategy_diversity=EXPERIMENT_DIVERSITY_SCALE * 2,
        )
        expected = (
            1.0 * EXPERIMENT_WEIGHT_SUCCESS
            + 1.0 * EXPERIMENT_WEIGHT_ACCURACY
            + 1.0 * EXPERIMENT_WEIGHT_BREADTH   # clamped to 1.0
            + 1.0 * EXPERIMENT_WEIGHT_DIVERSITY  # clamped to 1.0
        )
        assert m.score() == pytest.approx(expected)

    def test_summary_returns_string(self):
        m = ExperimentMetrics(success_rate=0.5, prediction_accuracy=0.4)
        s = m.summary()
        assert isinstance(s, str)
        assert "成功率" in s
        assert "スコア" in s


# ======================================================================
# ExperimentResult tests
# ======================================================================

class TestExperimentResult:
    def test_dataclass_creation(self):
        before = ExperimentMetrics(success_rate=0.5)
        after = ExperimentMetrics(success_rate=0.7)
        insight = _make_insight()
        result = ExperimentResult(
            insight=insight,
            patch=None,
            metrics_before=before,
            metrics_after=after,
            accepted=True,
            test_passed=True,
            duration=10.0,
            improvement=after.score() - before.score(),
        )
        assert result.accepted is True
        assert result.test_passed is True
        assert result.duration == 10.0
        assert result.improvement == pytest.approx(after.score() - before.score())

    def test_summary_contains_status(self):
        before = ExperimentMetrics()
        result = ExperimentResult(
            insight=_make_insight(),
            patch=None,
            metrics_before=before,
            metrics_after=before,
            accepted=False,
            test_passed=False,
            duration=5.0,
            improvement=0.0,
        )
        s = result.summary()
        assert "ロールバック" in s


# ======================================================================
# ExperimentRunner tests
# ======================================================================

class TestExperimentRunner:
    def test_init_creates_with_mock_dependencies(self, tmp_path):
        core = _make_mock_agi_core(tmp_path)
        db_path = tmp_path / "test_experiments.db"
        runner = ExperimentRunner(agi_core=core, db_path=db_path)
        assert runner.agi_core is core
        assert runner.experiment_timeout == 300
        assert db_path.exists()

    def test_insight_filtering_only_actionable(self, tmp_path):
        """run_experiments_from_insights filters to actionable weakness/gap/opportunity."""
        core = _make_mock_agi_core(tmp_path)
        db_path = tmp_path / "filter_test.db"
        runner = ExperimentRunner(agi_core=core, db_path=db_path)

        insights = [
            _make_insight(category="weakness", actionable=True, confidence=0.9),
            _make_insight(category="strength", actionable=True, confidence=0.95),
            _make_insight(category="gap", actionable=True, confidence=0.7),
            _make_insight(category="opportunity", actionable=False, confidence=0.8),
            _make_insight(category="weakness", actionable=False, confidence=0.6),
        ]

        # Patch run_experiment to just record which insights get through
        called_insights = []

        def fake_run(insight):
            called_insights.append(insight)
            before = ExperimentMetrics()
            return ExperimentResult(
                insight=insight, patch=None,
                metrics_before=before, metrics_after=before,
                accepted=False, test_passed=False, duration=0.1, improvement=0.0,
            )

        runner.run_experiment = fake_run
        runner.run_experiments_from_insights(insights, max_experiments=10)

        # Only actionable weakness and gap should pass (opportunity is not actionable)
        assert len(called_insights) == 2
        categories = [i.category for i in called_insights]
        assert "strength" not in categories
        assert "weakness" in categories
        assert "gap" in categories

    def test_target_file_selection_word_boundary(self, tmp_path):
        """Keyword matching uses word boundaries for ASCII keywords."""
        core = _make_mock_agi_core(tmp_path)
        db_path = tmp_path / "target_test.db"
        runner = ExperimentRunner(agi_core=core, db_path=db_path)

        # "plan" should match but not "explanation" (no word boundary match)
        insight_plan = _make_insight(content="The plan needs improvement")
        core.self_modifier.propose_change.return_value = None
        runner.run_experiment(insight_plan)
        target_arg = core.self_modifier.propose_change.call_args[0][0]
        assert target_arg == "hermes_agi_gen/planner.py"

        # "meta" keyword should match
        insight_meta = _make_insight(content="meta cognition is weak")
        runner.run_experiment(insight_meta)
        target_arg = core.self_modifier.propose_change.call_args[0][0]
        assert target_arg == "hermes_agi_gen/meta_cognition.py"

        # No keyword match -> default target
        insight_default = _make_insight(content="something completely unrelated xyz")
        runner.run_experiment(insight_default)
        target_arg = core.self_modifier.propose_change.call_args[0][0]
        assert target_arg == "hermes_agi_gen/self_improvement.py"

    def test_target_file_selection_cjk(self, tmp_path):
        """CJK keywords use substring matching."""
        core = _make_mock_agi_core(tmp_path)
        db_path = tmp_path / "cjk_test.db"
        runner = ExperimentRunner(agi_core=core, db_path=db_path)

        insight = _make_insight(content="記憶の管理を改善する必要がある")
        core.self_modifier.propose_change.return_value = None
        runner.run_experiment(insight)
        target_arg = core.self_modifier.propose_change.call_args[0][0]
        assert target_arg == "hermes_agi_gen/long_term_memory.py"

    def test_timeout_checking(self, tmp_path):
        """Experiment rolls back when timeout is exceeded."""
        core = _make_mock_agi_core(tmp_path)
        db_path = tmp_path / "timeout_test.db"
        runner = ExperimentRunner(agi_core=core, experiment_timeout=0, db_path=db_path)

        mock_patch = MagicMock()
        mock_patch.file_path = "hermes_agi_gen/test.py"
        mock_patch.rationale = "test rationale"
        core.self_modifier.propose_change.return_value = mock_patch

        insight = _make_insight(content="test timeout")
        result = runner.run_experiment(insight)

        # With timeout=0, the experiment should not be accepted
        assert result.accepted is False


# ======================================================================
# SelfImprovementEngine tests
# ======================================================================

class TestSelfImprovementEngine:
    def test_init_with_mock_llm(self, tmp_path):
        db_path = tmp_path / "improvement.db"
        llm = MagicMock()
        engine = SelfImprovementEngine(db_path=db_path, llm=llm)
        assert engine.llm is llm
        assert engine.db_path == db_path
        assert db_path.exists()

    def test_init_no_llm(self, tmp_path):
        db_path = tmp_path / "no_llm.db"
        engine = SelfImprovementEngine(db_path=db_path, llm=None)
        assert engine.llm is None

    def test_inject_into_state_populates_working_memory(self, tmp_path):
        db_path = tmp_path / "inject_test.db"
        engine = SelfImprovementEngine(db_path=db_path)

        # Insert a prompt version and anti-pattern so inject has data
        now = time.time()
        engine._conn.execute(
            "INSERT INTO prompt_versions(domain, few_shot_text, created_at, is_active) VALUES (?, ?, ?, 1)",
            ("general", "example few-shot text", now),
        )
        engine._conn.execute(
            "INSERT INTO anti_patterns(domain, bad_action, error_type, lesson, last_seen) VALUES (?, ?, ?, ?, ?)",
            ("general", "rm -rf /", "permission_error", "Do not delete root", now),
        )
        engine._conn.commit()

        state = _make_agent_state()
        engine.inject_into_state(state)

        assert "few_shot_examples" in state.working_memory
        assert state.working_memory["few_shot_examples"] == "example few-shot text"
        assert "anti_patterns" in state.working_memory
        assert "rm -rf" in state.working_memory["anti_patterns"]

    def test_analyze_session_with_completed_steps(self, tmp_path):
        db_path = tmp_path / "analyze_test.db"
        engine = SelfImprovementEngine(db_path=db_path, llm=None)

        state = _make_agent_state(
            completed_steps=["CMD: ls -la", "SEARCH: python docs", "echo hello"],
            observations=["found files"],
        )
        engine.analyze_session(state)

        # Rule-based extraction should store examples for CMD: and SEARCH: prefixed steps
        rows = engine._conn.execute("SELECT * FROM few_shot_examples").fetchall()
        actions = [r["good_action"] for r in rows]
        assert "CMD: ls -la" in actions
        assert "SEARCH: python docs" in actions
        # "echo hello" does not start with CMD:/SEARCH:/PYTHON: so should be excluded
        assert "echo hello" not in actions

    def test_analyze_session_with_failed_steps(self, tmp_path):
        db_path = tmp_path / "fail_analyze.db"
        engine = SelfImprovementEngine(db_path=db_path, llm=None)

        state = _make_agent_state(
            failed_steps=["rm -rf /tmp/test"],
            working_memory={"error_history": ["permission_error"]},
        )
        engine.analyze_session(state)

        rows = engine._conn.execute("SELECT * FROM anti_patterns").fetchall()
        assert len(rows) == 1
        assert rows[0]["error_type"] == "permission_error"
        assert "権限" in rows[0]["lesson"]

    def test_record_session_performance(self, tmp_path):
        db_path = tmp_path / "perf_test.db"
        engine = SelfImprovementEngine(db_path=db_path)

        engine.record_session_performance("s1", "goal1", "general", 0.8)
        engine.record_session_performance("s2", "goal2", "general", 0.6)

        rows = engine._conn.execute("SELECT * FROM session_performance ORDER BY created_at").fetchall()
        assert len(rows) == 2
        assert rows[0]["score"] == 0.8
        assert rows[1]["score"] == 0.6

    def test_get_performance_trend_returns_float(self, tmp_path):
        db_path = tmp_path / "trend_test.db"
        engine = SelfImprovementEngine(db_path=db_path)

        # No data -> returns default 0.5
        assert engine.get_performance_trend() == pytest.approx(0.5)

        # With data -> returns average
        engine.record_session_performance("s1", "g1", "general", 0.9)
        engine.record_session_performance("s2", "g2", "general", 0.7)
        trend = engine.get_performance_trend("general", window=10)
        assert isinstance(trend, float)
        assert trend == pytest.approx(0.8)

    def test_get_best_examples_cross_domain_weight(self, tmp_path):
        db_path = tmp_path / "cross_domain.db"
        engine = SelfImprovementEngine(db_path=db_path)

        now = time.time()
        # Insert an example in a different domain
        engine._conn.execute(
            """INSERT INTO few_shot_examples
            (domain, goal_pattern, good_action, context, outcome, quality_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("coding", "pattern", "CMD: test", "ctx", "ok", 0.9, now),
        )
        engine._conn.commit()

        # Query for "web" domain which has 0 examples -> triggers cross-domain
        results = engine.get_best_examples(
            domain="web", limit=5,
            domain_match_weight=1.0,
            cross_domain_weight=0.3,
        )
        assert len(results) >= 1
        # Cross-domain example should have discounted quality_score
        assert results[0]["quality_score"] == pytest.approx(0.9 * 0.3)

    def test_few_shot_validation_action_prefix(self, tmp_path):
        """LLM-extracted few-shot examples are validated: action must start with known prefix or match a completed step."""
        db_path = tmp_path / "fewshot_valid.db"
        llm = MagicMock()

        # LLM returns examples: one valid (CMD: prefix), one invalid
        llm.chat_json.return_value = [
            {
                "goal_pattern": "file listing",
                "good_action": "CMD: ls -la",
                "context": "list files",
                "outcome": "success",
                "quality_score": 0.9,
            },
            {
                "goal_pattern": "random thing",
                "good_action": "just do something random",
                "context": "no context",
                "outcome": "unknown",
                "quality_score": 0.5,
            },
        ]

        engine = SelfImprovementEngine(db_path=db_path, llm=llm)
        state = _make_agent_state(
            completed_steps=["CMD: ls -la", "PYTHON: print('hi')"],
            observations=["files listed"],
        )
        engine._extract_few_shot_examples(state, "general")

        rows = engine._conn.execute("SELECT * FROM few_shot_examples").fetchall()
        actions = [r["good_action"] for r in rows]
        # Valid action with CMD: prefix should be stored
        assert "CMD: ls -la" in actions
        # Invalid action without prefix and not in completed_steps should be skipped
        assert "just do something random" not in actions

    def test_anti_pattern_recording(self, tmp_path):
        db_path = tmp_path / "anti_pattern.db"
        engine = SelfImprovementEngine(db_path=db_path, llm=None)

        state = _make_agent_state(
            failed_steps=["pip install nonexistent_pkg"],
            working_memory={"error_history": ["missing_python_module"]},
        )
        engine._record_anti_patterns(state, "general")

        rows = engine._conn.execute("SELECT * FROM anti_patterns").fetchall()
        assert len(rows) == 1
        assert rows[0]["bad_action"] == "pip install nonexistent_pkg"
        assert rows[0]["error_type"] == "missing_python_module"
        assert "pip" in rows[0]["lesson"]

        # Recording the same anti-pattern again should increment frequency
        engine._record_anti_patterns(state, "general")
        rows = engine._conn.execute("SELECT * FROM anti_patterns").fetchall()
        assert len(rows) == 1
        assert rows[0]["frequency"] == 2
