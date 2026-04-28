"""Hermes AGI 正式版 (spec_full) のテスト。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_agi_gen.spec_core import PlanStep, Result, Task
from hermes_agi_gen.spec_full import (
    FullConfig,
    FullCritic,
    FullExecutor,
    FullPlanner,
    HermesAGIFull,
    SqliteMemory,
    _classify_goal,
    _parse_json_array,
    make_real_tool_runner,
    run_spec_full,
)


# ----------------------------------------------------------------------
# ヘルパー
# ----------------------------------------------------------------------
def _task(goal: str = "目標", constraints=None) -> Task:
    return Task(id="T1", goal=goal, constraints=list(constraints or []), status="running")


# ----------------------------------------------------------------------
# SqliteMemory
# ----------------------------------------------------------------------
class TestSqliteMemory:
    def test_round_trip(self, tmp_path):
        m = SqliteMemory(tmp_path / "m.sqlite")
        t = _task("テスト", ["c1"])
        m.save_task(t)
        snap = m.load()
        assert snap["tasks"] and snap["tasks"][0]["goal"] == "テスト"
        assert snap["tasks"][0]["constraints"] == ["c1"]

    def test_load_shape_when_empty(self, tmp_path):
        m = SqliteMemory(tmp_path / "empty.sqlite")
        snap = m.load()
        assert set(snap.keys()) == {"tasks", "plans", "results", "reviews"}
        assert all(v == [] for v in snap.values())

    def test_failure_rate(self, tmp_path):
        m = SqliteMemory(tmp_path / "m.sqlite")
        m.save_task(_task())
        m.save_result("T1", 1, Result(step_id=1, output="ok", success=True))
        m.save_result("T1", 1, Result(step_id=2, output="bad", success=False))
        assert m.get_failure_rate("T1") == pytest.approx(0.5)

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "m.sqlite"
        m1 = SqliteMemory(path)
        m1.save_task(_task("永続"))
        m1.close()
        m2 = SqliteMemory(path)
        assert m2.load()["tasks"][0]["goal"] == "永続"


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------
class TestFullPlanner:
    def test_static_template_for_coding_domain(self):
        p = FullPlanner(llm=None)
        plan = p.plan(_task("コードを修正する"), {"reviews": []})
        assert len(plan.steps) >= FullConfig().plan_min_steps
        assert any("リポジトリ" in s.action for s in plan.steps)

    def test_static_template_for_research_domain(self):
        p = FullPlanner(llm=None)
        plan = p.plan(_task("AI動向をリサーチする"), {"reviews": []})
        assert any("情報" in s.action for s in plan.steps)

    def test_min_max_steps_enforced(self):
        cfg = FullConfig(plan_min_steps=2, plan_max_steps=4)
        p = FullPlanner(llm=None, config=cfg)
        plan = p.plan(_task("一般的な目標"), {"reviews": []})
        assert 2 <= len(plan.steps) <= 4

    def test_uses_feedback_on_iteration_two(self):
        p = FullPlanner(llm=None)
        memory = {
            "reviews": [
                {"task_id": "T1", "feedback": "失敗ステップを見直してください"},
            ]
        }
        plan = p.plan(_task(), memory)
        assert "前回の指摘" in plan.steps[0].action

    def test_llm_plan_used_when_available(self):
        llm = MagicMock()
        llm.chat.return_value = '["A", "B", "C", "D"]'
        p = FullPlanner(llm=llm, config=FullConfig(plan_min_steps=2, plan_max_steps=10))
        plan = p.plan(_task(), {"reviews": []})
        actions = [s.action for s in plan.steps]
        assert actions == ["A", "B", "C", "D"]

    def test_llm_failure_falls_back_to_template(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")
        p = FullPlanner(llm=llm)
        plan = p.plan(_task("コード書く"), {"reviews": []})
        assert plan.steps  # フォールバックで成立


class TestParseJsonArray:
    def test_plain_array(self):
        assert _parse_json_array('["a", "b"]') == ["a", "b"]

    def test_with_code_fence(self):
        raw = '```json\n["x", "y"]\n```'
        assert _parse_json_array(raw) == ["x", "y"]

    def test_extracts_from_surrounding_text(self):
        raw = '思考の結果、計画は以下です: ["s1", "s2", "s3"] 以上です'
        assert _parse_json_array(raw) == ["s1", "s2", "s3"]

    def test_returns_none_for_non_array(self):
        assert _parse_json_array('{"k": 1}') is None

    def test_returns_none_for_garbage(self):
        assert _parse_json_array("not json at all") is None


class TestClassifyGoal:
    @pytest.mark.parametrize("goal,domain", [
        ("コードを修正する", "coding"),
        ("AIをリサーチする", "research"),
        ("ブログ記事を書く", "writing"),
        ("CSVデータを分析する", "data"),
        ("何かをする", "general"),
    ])
    def test_classify(self, goal, domain):
        assert _classify_goal(goal) == domain


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------
class TestFullExecutor:
    def test_default_runner_succeeds(self):
        ex = FullExecutor()
        result, attempts, dur = ex.execute(PlanStep(id=1, action="OK 動作"), _task())
        assert result.success is True
        assert attempts == 1
        assert dur >= 0.0

    def test_retries_on_failure(self):
        calls = {"n": 0}

        def runner(step, task):
            calls["n"] += 1
            return Result(step_id=step.id, output="err", success=calls["n"] >= 3)

        ex = FullExecutor(runner=runner, config=FullConfig(retry_max=3, retry_backoff_sec=0.0))
        result, attempts, _ = ex.execute(PlanStep(id=1, action="x"), _task())
        assert result.success is True
        assert attempts == 3

    def test_stops_after_retry_max(self):
        def always_fail(step, task):
            return Result(step_id=step.id, output="boom", success=False)

        ex = FullExecutor(runner=always_fail, config=FullConfig(retry_max=2, retry_backoff_sec=0.0))
        result, attempts, _ = ex.execute(PlanStep(id=1, action="x"), _task())
        assert result.success is False
        assert attempts == 3  # 初回 + retry_max(2)

    def test_value_system_blocks_dangerous(self):
        from hermes_agi_gen.value_system import ValueSystem
        ex = FullExecutor(value_system=ValueSystem())
        result, attempts, _ = ex.execute(
            PlanStep(id=1, action="rm -rf /"), _task("片付け"),
        )
        assert result.success is False
        assert "blocked by ValueSystem" in result.output
        assert attempts == 0  # 価値ブロックは即拒否、ランナー呼ばず

    def test_runner_exception_is_caught(self):
        def boom(step, task):
            raise RuntimeError("kaboom")

        ex = FullExecutor(runner=boom, config=FullConfig(retry_max=0, retry_backoff_sec=0.0))
        result, _, _ = ex.execute(PlanStep(id=1, action="x"), _task())
        assert result.success is False
        assert "kaboom" in result.output


# ----------------------------------------------------------------------
# Critic
# ----------------------------------------------------------------------
class TestFullCritic:
    def test_done_when_full_completion_and_high_score(self):
        c = FullCritic()
        memory = {
            "results": [
                {"task_id": "T1", "iteration": 1, "success": True, "output": "目標達成"},
                {"task_id": "T1", "iteration": 1, "success": True, "output": "目標完了"},
            ]
        }
        review, br = c.review(_task("目標"), memory, iteration=1)
        assert review.done is True
        assert br["completion"] == 1.0
        assert br["total"] >= FullConfig().done_threshold

    def test_not_done_when_partial(self):
        c = FullCritic()
        memory = {
            "results": [
                {"task_id": "T1", "iteration": 1, "success": True, "output": "x"},
                {"task_id": "T1", "iteration": 1, "success": False, "output": "y"},
            ]
        }
        review, br = c.review(_task("目標"), memory, iteration=1)
        assert review.done is False
        assert br["completion"] == 0.5

    def test_no_results_returns_zero(self):
        c = FullCritic()
        review, br = c.review(_task(), {"results": []}, iteration=1)
        assert review.score == 0.0
        assert br["total"] == 0.0


# ----------------------------------------------------------------------
# HermesAGIFull オーケストレーター
# ----------------------------------------------------------------------
class TestHermesAGIFull:
    def test_default_run_completes(self, tmp_path):
        out = run_spec_full("テスト目標", tmp_path / "m.sqlite")
        assert out["task"]["status"] == "done"
        assert out["review"]["done"] is True
        assert 1 <= out["iterations"] <= FullConfig().max_iterations

    def test_no_three_iteration_cap(self, tmp_path):
        """MVP は max_iterations を 3 でクランプしていた。正式版はしない。"""
        # 全て失敗するランナーで反復が 3 を超えることを確認
        def fail_runner(step, task):
            return Result(step_id=step.id, output="fail", success=False)

        cfg = FullConfig(max_iterations=6, patience=99, retry_max=0, retry_backoff_sec=0.0)
        app = HermesAGIFull(
            memory_path=tmp_path / "m.sqlite",
            config=cfg,
            executor=FullExecutor(runner=fail_runner, config=cfg),
        )
        out = app.run("失敗する目標")
        assert out["iterations"] == 6  # MVP の 3 を超えて回る

    def test_plateau_halts_early(self, tmp_path):
        """スコアが伸びなければ plateau で打ち切る。"""
        def fail_runner(step, task):
            return Result(step_id=step.id, output="x", success=False)

        cfg = FullConfig(max_iterations=20, patience=2, retry_max=0, retry_backoff_sec=0.0)
        app = HermesAGIFull(
            memory_path=tmp_path / "m.sqlite",
            config=cfg,
            executor=FullExecutor(runner=fail_runner, config=cfg),
        )
        out = app.run("プラトー検出")
        assert out["halted_reason"] == "plateau"
        assert out["iterations"] < 20

    def test_value_violation_halts(self, tmp_path):
        from hermes_agi_gen.value_system import ValueSystem

        # Planner が必ず "rm -rf /" を含む計画を返すようにする
        class _BadPlanner(FullPlanner):
            def plan(self, task, memory):
                from hermes_agi_gen.spec_core import Plan, PlanStep
                return Plan(steps=[PlanStep(id=1, action="rm -rf /")])

        cfg = FullConfig(max_iterations=5, halt_on_value_violation=True, retry_max=0)
        app = HermesAGIFull(
            memory_path=tmp_path / "m.sqlite",
            config=cfg,
            planner=_BadPlanner(config=cfg),
            executor=FullExecutor(config=cfg, value_system=ValueSystem()),
            value_system=ValueSystem(),
        )
        out = app.run("危ないことをする")
        assert out["halted_reason"] == "value_violation"
        assert out["iterations"] == 1

    def test_history_recorded(self, tmp_path):
        out = run_spec_full("テスト", tmp_path / "m.sqlite")
        assert isinstance(out["history"], list)
        assert all("review" in h and "breakdown" in h for h in out["history"])
        assert "metrics" in out and "best_score" in out["metrics"]

    def test_memory_persisted_across_runs(self, tmp_path):
        path = tmp_path / "shared.sqlite"
        run_spec_full("最初のタスク", path)
        run_spec_full("二つ目のタスク", path)
        m = SqliteMemory(path)
        snap = m.load()
        assert len(snap["tasks"]) == 2

    def test_constraints_threaded_into_iterations(self, tmp_path):
        """MVP では constraints に critic_feedback がそのまま追記された。
        正式版でも反復継続時は同様に追記されること。"""
        attempts = {"n": 0}

        def alt_runner(step, task):
            # 1反復目は失敗、2反復目以降は成功
            attempts["n"] += 1
            return Result(step_id=step.id, output="o", success=attempts["n"] > 5)

        cfg = FullConfig(max_iterations=4, patience=99, retry_max=0)
        app = HermesAGIFull(
            memory_path=tmp_path / "m.sqlite",
            config=cfg,
            executor=FullExecutor(runner=alt_runner, config=cfg),
        )
        out = app.run("継続する目標")
        # task.constraints に critic_feedback: が含まれる
        assert any("critic_feedback" in c for c in out["task"]["constraints"])


# ----------------------------------------------------------------------
# 実ツールアダプタ (smoke)
# ----------------------------------------------------------------------
class TestRealToolRunner:
    def test_passthrough_for_descriptive_step(self, tmp_path):
        runner = make_real_tool_runner(repo_root=tmp_path)
        result = runner(PlanStep(id=1, action="目的を確認する"), _task())
        assert result.success is True

    def test_real_read_via_executor(self, tmp_path):
        target = tmp_path / "hello.txt"
        target.write_text("HELLO_WORLD\n", encoding="utf-8")
        runner = make_real_tool_runner(repo_root=tmp_path)
        result = runner(PlanStep(id=1, action=f"READ: {target}"), _task("ファイルを読む"))
        assert result.success is True
        assert "HELLO_WORLD" in result.output
