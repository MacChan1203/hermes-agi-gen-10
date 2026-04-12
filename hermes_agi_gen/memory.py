"""ワーキングメモリの初期化・管理ユーティリティ。

AgentState のワーキングメモリを初期化し、実行中に得た情報
（成功コマンド、失敗パターン、環境情報）を記録する関数群。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_state import AgentState


def initialize_working_memory(state: AgentState) -> None:
    """ワーキングメモリを初期化する。既に初期化済みの場合はスキップ。

    環境情報（cwd, Pythonバージョン, 実行パス）と各種記録用リストを設定する。

    Args:
        state: 初期化対象の AgentState
    """
    if state.working_memory:
        return

    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = "(unknown)"

    try:
        python_version = f"Python {sys.version.split()[0]}"
        python_executable = sys.executable
    except Exception:
        python_version = "Python (unknown)"
        python_executable = "(unknown)"

    state.working_memory = {
        "environment": {
            "cwd": cwd,
            "python_version": python_version,
            "python_executable": python_executable,
        },
        "important_files": [],
        "known_commands_that_work": [],
        "known_failures": [],
        "assumptions": [],
        "error_history": [],
    }


def remember_successful_command(state: AgentState, command: str) -> None:
    """成功したコマンドをワーキングメモリに記録する。重複は無視。

    Args:
        state: 記録対象の AgentState
        command: 成功したコマンド文字列
    """
    commands: List[str] = state.working_memory.setdefault("known_commands_that_work", [])
    if command not in commands:
        commands.append(command)


def remember_failure(state: AgentState, step: str, error_type: str, stderr: str) -> None:
    """失敗情報をワーキングメモリに記録する。

    known_failures リストに失敗詳細を追加し、
    error_history にエラータイプを追記する。

    Args:
        state: 記録対象の AgentState
        step: 失敗したステップの説明
        error_type: エラー分類文字列
        stderr: 標準エラー出力
    """
    failures: List[Dict[str, str]] = state.working_memory.setdefault("known_failures", [])
    failures.append({
        "step": step,
        "error_type": error_type,
        "stderr": stderr.strip(),
    })

    history: List[str] = state.working_memory.setdefault("error_history", [])
    history.append(error_type)


def set_environment_info(
    state: AgentState,
    *,
    cwd: Optional[str] = None,
    python_version: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> None:
    """ワーキングメモリ内の環境情報を更新する。

    指定されたフィールドのみ上書きし、None のフィールドは変更しない。

    Args:
        state: 更新対象の AgentState
        cwd: 作業ディレクトリパス
        python_version: Python バージョン文字列
        python_executable: Python 実行ファイルパス
    """
    env: Dict[str, str] = state.working_memory.setdefault("environment", {})

    if cwd is not None:
        env["cwd"] = cwd
    if python_version is not None:
        env["python_version"] = python_version
    if python_executable is not None:
        env["python_executable"] = python_executable
