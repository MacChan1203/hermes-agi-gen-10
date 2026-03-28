from __future__ import annotations

import sys
from pathlib import Path

from .agent_state import AgentState

def initialize_working_memory(state: AgentState) -> None:
    if state.working_memory:
        return

    state.working_memory = {
        "environment": {
            "cwd": str(Path.cwd()),
            "python_version": f"Python {sys.version.split()[0]}",
            "python_executable": sys.executable,
        },
        "important_files": [],
        "known_commands_that_work": [],
        "known_failures": [],
        "assumptions": [],
        "error_history": [],
    }


def remember_successful_command(state: AgentState, command: str) -> None:
    commands = state.working_memory.setdefault("known_commands_that_work", [])
    if command not in commands:
        commands.append(command)


def remember_failure(state: AgentState, step: str, error_type: str, stderr: str) -> None:
    failures = state.working_memory.setdefault("known_failures", [])
    failures.append({
        "step": step,
        "error_type": error_type,
        "stderr": stderr.strip(),
    })

    history = state.working_memory.setdefault("error_history", [])
    history.append(error_type)


def set_environment_info(
    state: AgentState,
    *,
    cwd: str | None = None,
    python_version: str | None = None,
    python_executable: str | None = None,
) -> None:
    env = state.working_memory.setdefault("environment", {})

    if cwd is not None:
        env["cwd"] = cwd
    if python_version is not None:
        env["python_version"] = python_version
    if python_executable is not None:
        env["python_executable"] = python_executable
