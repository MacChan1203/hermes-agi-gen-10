"""Spec-aligned Hermes AGI MVP core.

This module implements the compact loop described in hermes_agi_spec.md:
Planner -> Executor -> Critic, backed by JSON memory and capped at 3 loops.
It is intentionally deterministic so it can run without an LLM or network.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional


TaskStatus = Literal["pending", "running", "done"]
StepStatus = Literal["pending", "done", "failed"]


@dataclass
class Task:
    id: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    status: TaskStatus = "pending"


@dataclass
class PlanStep:
    id: int
    action: str
    status: StepStatus = "pending"


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Result:
    step_id: int
    output: str
    success: bool


@dataclass
class CriticOutput:
    score: float
    feedback: str
    done: bool


class JsonMemory:
    """Small append-only JSON memory for MVP loop state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"tasks": [], "plans": [], "results": [], "reviews": []}
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"JSON memory must contain an object: {self.path}")
        for key in ("tasks", "plans", "results", "reviews"):
            data.setdefault(key, [])
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        tmp.replace(self.path)

    def append(self, key: str, value: Any) -> None:
        data = self.load()
        data.setdefault(key, []).append(_to_jsonable(value))
        self.save(data)


class SpecPlanner:
    """Task -> Plan with a small, bounded set of steps."""

    def __init__(self, max_steps: int = 3) -> None:
        self.max_steps = max(1, max_steps)

    def plan(self, task: Task, memory: dict[str, Any]) -> Plan:
        goal = task.goal.strip()
        constraints = ", ".join(task.constraints) if task.constraints else "制約なし"
        actions = [
            f"目的を確認する: {goal}",
            f"制約を反映して実行する: {constraints}",
            f"結果をまとめる: {goal}",
        ]
        return Plan(
            steps=[
                PlanStep(id=index + 1, action=action)
                for index, action in enumerate(actions[: self.max_steps])
            ]
        )


class SpecExecutor:
    """Executes plan steps with retry support.

    A custom runner can be supplied for real integrations. The default runner is
    deliberately side-effect-free and returns a textual observation.
    """

    def __init__(
        self,
        runner: Optional[Callable[[PlanStep, Task], Result]] = None,
        max_retries: int = 1,
    ) -> None:
        self.runner = runner or self._default_runner
        self.max_retries = max(0, max_retries)

    def execute(self, step: PlanStep, task: Task) -> Result:
        last_result: Optional[Result] = None
        for attempt in range(self.max_retries + 1):
            last_result = self.runner(step, task)
            if last_result.success:
                step.status = "done"
                return last_result
            if attempt < self.max_retries:
                time.sleep(0.01)
        step.status = "failed"
        assert last_result is not None
        return last_result

    @staticmethod
    def _default_runner(step: PlanStep, task: Task) -> Result:
        success = "fail" not in step.action.lower() and "失敗" not in step.action
        output = f"{step.action} -> {'完了' if success else '失敗'}"
        return Result(step_id=step.id, output=output, success=success)


class SpecCritic:
    """Evaluates accumulated results and decides whether the loop is done."""

    def review(self, task: Task, memory: dict[str, Any]) -> CriticOutput:
        task_results = [
            item for item in memory.get("results", [])
            if item.get("task_id") == task.id
        ]
        if not task_results:
            return CriticOutput(score=0.0, feedback="まだ結果がありません", done=False)

        successes = sum(1 for item in task_results if item.get("success") is True)
        score = successes / len(task_results)
        done = score >= 1.0
        feedback = "全ステップが成功しました" if done else "失敗したステップを見直してください"
        return CriticOutput(score=round(score, 3), feedback=feedback, done=done)


class HermesAGIMVP:
    """Minimal multi-component AGI loop from the Codex spec."""

    def __init__(
        self,
        memory_path: str | Path,
        *,
        max_iterations: int = 3,
        planner: Optional[SpecPlanner] = None,
        executor: Optional[SpecExecutor] = None,
        critic: Optional[SpecCritic] = None,
    ) -> None:
        self.memory = JsonMemory(memory_path)
        self.max_iterations = min(max(1, max_iterations), 3)
        self.planner = planner or SpecPlanner(max_steps=3)
        self.executor = executor or SpecExecutor(max_retries=1)
        self.critic = critic or SpecCritic()

    def run(self, goal: str, constraints: Optional[Iterable[str]] = None) -> dict[str, Any]:
        task = Task(
            id=str(uuid.uuid4()),
            goal=goal.strip(),
            constraints=list(constraints or []),
            status="running",
        )
        self.memory.append("tasks", task)
        final_review = CriticOutput(score=0.0, feedback="未評価", done=False)

        for iteration in range(1, self.max_iterations + 1):
            snapshot = self.memory.load()
            plan = self.planner.plan(task, snapshot)
            self.memory.append("plans", {
                "task_id": task.id,
                "iteration": iteration,
                "steps": [asdict(step) for step in plan.steps],
            })

            for step in plan.steps:
                result = self.executor.execute(step, task)
                self.memory.append("results", {
                    "task_id": task.id,
                    "iteration": iteration,
                    **asdict(result),
                })

            final_review = self.critic.review(task, self.memory.load())
            self.memory.append("reviews", {
                "task_id": task.id,
                "iteration": iteration,
                **asdict(final_review),
            })
            if final_review.done:
                task.status = "done"
                break
            task.constraints.append(f"critic_feedback: {final_review.feedback}")

        data = self.memory.load()
        _mark_task_status(data, task.id, task.status)
        self.memory.save(data)
        return {
            "task": asdict(task),
            "review": asdict(final_review),
            "iterations": min(len([
                r for r in data.get("reviews", []) if r.get("task_id") == task.id
            ]), self.max_iterations),
            "memory_path": str(self.memory.path),
        }


def run_spec_mvp(
    goal: str,
    memory_path: str | Path,
    constraints: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Convenience entry point for the spec MVP."""
    return HermesAGIMVP(memory_path=memory_path).run(goal, constraints=constraints)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _mark_task_status(data: dict[str, Any], task_id: str, status: TaskStatus) -> None:
    for item in reversed(data.get("tasks", [])):
        if item.get("id") == task_id:
            item["status"] = status
            return
