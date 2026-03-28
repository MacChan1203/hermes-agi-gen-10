"""旧 model_tools.py を簡略化した tool registry。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .toolsets import get_all_toolsets, resolve_toolset, validate_toolset

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


class DummyRegistry:
    def get_tool_definitions(self, enabled_toolsets: List[str] | None = None, disabled_toolsets: List[str] | None = None) -> List[Dict[str, Any]]:
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
        deduped = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return [{"type": "function", "function": {"name": name, "description": f"{name} ツール", "parameters": {"type": "object", "properties": {}}}} for name in deduped]

    def dispatch(self, function_name: str, function_args: Dict[str, Any]) -> str:
        return json.dumps({"ok": True, "tool": function_name, "args": function_args, "message": f"{function_name} はダミー実装です。"}, ensure_ascii=False)


registry = DummyRegistry()


def get_tool_definitions(enabled_toolsets=None, disabled_toolsets=None, quiet_mode: bool = False) -> list:
    return registry.get_tool_definitions(enabled_toolsets, disabled_toolsets)


def handle_function_call(function_name: str, function_args: Dict[str, Any], task_id: str | None = None, user_task: str | None = None, **_: Any) -> str:
    return registry.dispatch(function_name, function_args)


def get_all_tool_names() -> list[str]:
    return list(TOOL_TO_TOOLSET_MAP.keys())


def get_toolset_for_tool(name: str) -> str:
    return TOOL_TO_TOOLSET_MAP.get(name, "unknown")


def get_available_toolsets() -> Dict[str, Dict[str, object]]:
    return get_all_toolsets()


def check_toolset_requirements() -> Dict[str, Dict[str, object]]:
    return TOOLSET_REQUIREMENTS


def check_tool_availability(quiet: bool = False) -> Tuple[bool, list[str]]:
    return True, []
