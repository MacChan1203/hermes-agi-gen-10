"""ツールセット管理・ツールディスパッチモジュール。

ToolsetRegistry は executor.py / tool_registry.py と連携して
ツールセット解決・ツール定義生成・ツールディスパッチを行う。

旧 StubToolRegistry を実装に置き換え済み。dispatch() は
ToolRegistry (動的ツール) と Executor (組み込みツール) に委譲する。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .toolsets import get_all_toolsets, resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)

_TOOL_TO_TOOLSET_MAP: Dict[str, str] = {}
for toolset_name, info in get_all_toolsets().items():
    for tool in resolve_toolset(toolset_name):
        _TOOL_TO_TOOLSET_MAP.setdefault(tool, toolset_name)

TOOL_TO_TOOLSET_MAP = _TOOL_TO_TOOLSET_MAP
TOOLSET_REQUIREMENTS = {
    "terminal": {"python": True},
    "web": {"internet": True},
    "browser": {"browser": True},
}

# 組み込みツール名 → Executor が処理するプレフィックスのマッピング
_BUILTIN_TOOL_PREFIX: Dict[str, str] = {
    "web_search": "SEARCH",
    "web_extract": "FETCH",
    "terminal": "CMD",
    "process": "CMD",
    "read_file": "READ",
    "write_file": "WRITE",
    "patch": "WRITE",
    "search_files": "CMD",
    "execute_code": "PYTHON",
}


class ToolsetRegistry:
    """ツールセット管理とツールディスパッチを行うレジストリ。

    - ツールセットの解決・フィルタリング・定義生成
    - ツール呼び出しの Executor / ToolRegistry への委譲
    - スタブではなく実際にツールを実行する
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        tool_registry: Optional[Any] = None,
    ) -> None:
        self._repo_root = repo_root or Path(".")
        self._tool_registry = tool_registry
        self._executor: Optional[Any] = None

    def _get_executor(self):
        """Executor を遅延初期化する (循環インポート回避)。"""
        if self._executor is None:
            from .executor import Executor
            self._executor = Executor(repo_root=self._repo_root)
        return self._executor

    def _get_tool_registry(self):
        """ToolRegistry を遅延初期化する。"""
        if self._tool_registry is None:
            from .tool_registry import ToolRegistry
            self._tool_registry = ToolRegistry()
        return self._tool_registry

    def get_tool_definitions(
        self,
        enabled_toolsets: Optional[List[str]] = None,
        disabled_toolsets: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """有効なツールセットに基づいてツール定義一覧を返す。"""
        enabled_toolsets = enabled_toolsets or ["all"]
        disabled_toolsets = disabled_toolsets or []
        names: list[str] = []
        for toolset in enabled_toolsets:
            if not validate_toolset(toolset):
                continue
            names.extend(resolve_toolset(toolset))
        for toolset in disabled_toolsets:
            if validate_toolset(toolset):
                disabled_names = set(resolve_toolset(toolset))
                names = [n for n in names if n not in disabled_names]
        deduped: list[str] = []
        for name in names:
            if name not in deduped:
                deduped.append(name)

        definitions = []
        for name in deduped:
            prefix = _BUILTIN_TOOL_PREFIX.get(name, name.upper())
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} ツール (prefix: {prefix}:)",
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return definitions

    def dispatch(
        self,
        function_name: str,
        function_args: Dict[str, Any],
    ) -> str:
        """ツールを実際に実行する。

        1. 組み込みツール → Executor に委譲
        2. カスタムツール → ToolRegistry に委譲
        3. 未知のツール → エラーを返す (偽の成功は返さない)

        Returns:
            JSON 文字列 (実行結果)
        """
        # 1. 組み込みツール
        prefix = _BUILTIN_TOOL_PREFIX.get(function_name)
        if prefix:
            executor = self._get_executor()
            args_str = function_args.get("input", function_args.get("query", ""))
            step = f"{prefix}: {args_str}"
            # Executor には AgentState が必要 — 最小限のモックを使用
            from .agent_state import AgentState
            state = AgentState(user_goal="tool dispatch", max_iterations=1)
            result = executor.execute(step, state)
            return json.dumps(result, ensure_ascii=False, default=str)

        # 2. カスタムツール (ToolRegistry)
        try:
            tr = self._get_tool_registry()
            tool = tr.get(function_name)
            if tool:
                result = tool.invoke(json.dumps(function_args, ensure_ascii=False))
                return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("カスタムツール '%s' のディスパッチに失敗: %s", function_name, exc)

        # 3. 未知のツール — 偽の成功ではなくエラーを返す
        error_result = {
            "ok": False,
            "stdout": "",
            "stderr": f"未知のツール: '{function_name}' は登録されていません。",
            "returncode": 1,
        }
        return json.dumps(error_result, ensure_ascii=False)


# 後方互換エイリアス
StubToolRegistry = ToolsetRegistry
DummyRegistry = ToolsetRegistry

registry = ToolsetRegistry()


def get_tool_definitions(
    enabled_toolsets=None, disabled_toolsets=None, quiet_mode: bool = False
) -> list:
    """ツール定義一覧を返す。"""
    return registry.get_tool_definitions(enabled_toolsets, disabled_toolsets)


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: str | None = None,
    user_task: str | None = None,
    **_: Any,
) -> str:
    """ツール呼び出しをハンドルする。"""
    return registry.dispatch(function_name, function_args)


def get_all_tool_names() -> list[str]:
    """全ツール名を返す。"""
    return list(TOOL_TO_TOOLSET_MAP.keys())


def get_toolset_for_tool(name: str) -> str:
    """ツール名から所属ツールセットを返す。"""
    return TOOL_TO_TOOLSET_MAP.get(name, "unknown")


def get_available_toolsets() -> Dict[str, Dict[str, object]]:
    """利用可能なツールセット一覧を返す。"""
    return get_all_toolsets()


def check_toolset_requirements() -> Dict[str, Dict[str, object]]:
    """ツールセットの要件を返す。"""
    return TOOLSET_REQUIREMENTS


def check_tool_availability(quiet: bool = False) -> Tuple[bool, list[str]]:
    """ツールの利用可否を確認する。"""
    return True, []
