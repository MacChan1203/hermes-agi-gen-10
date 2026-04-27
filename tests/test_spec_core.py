from __future__ import annotations

import json

from hermes_agi_gen.spec_core import (
    HermesAGIMVP,
    JsonMemory,
    PlanStep,
    Result,
    SpecExecutor,
    Task,
    run_spec_mvp,
)


def test_spec_mvp_writes_json_memory(tmp_path):
    memory_path = tmp_path / "memory.json"
    result = run_spec_mvp("テスト目標", memory_path)

    assert result["task"]["status"] == "done"
    assert result["review"]["done"] is True
    assert result["iterations"] == 1

    data = json.loads(memory_path.read_text(encoding="utf-8"))
    assert set(data) >= {"tasks", "plans", "results", "reviews"}
    assert data["tasks"][0]["goal"] == "テスト目標"
    assert len(data["results"]) == 3


def test_spec_mvp_caps_iterations_at_three(tmp_path):
    def failing_runner(step: PlanStep, task: Task) -> Result:
        return Result(step_id=step.id, output="forced failure", success=False)

    app = HermesAGIMVP(
        memory_path=tmp_path / "memory.json",
        max_iterations=99,
        executor=SpecExecutor(runner=failing_runner, max_retries=0),
    )
    result = app.run("失敗する目標")

    assert result["task"]["status"] == "running"
    assert result["review"]["done"] is False
    assert result["iterations"] == 3


def test_json_memory_initializes_shape(tmp_path):
    memory = JsonMemory(tmp_path / "missing.json")
    data = memory.load()

    assert data == {"tasks": [], "plans": [], "results": [], "reviews": []}
