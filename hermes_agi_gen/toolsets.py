"""Hermes AGI Gen の toolset 定義。旧 Hermes の考え方を保ちつつ簡潔化。"""
from __future__ import annotations

from typing import Dict, List, Set

_HERMES_CORE_TOOLS = [
    "web_search", "web_extract",
    "terminal", "process",
    "read_file", "write_file", "patch", "search_files",
    "vision_analyze", "image_generate",
    "mixture_of_agents",
    "skills_list", "skill_view", "skill_manage",
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "text_to_speech", "todo", "memory", "session_search",
    "clarify", "execute_code", "delegate_task", "cronjob", "send_message",
]

TOOLSETS: Dict[str, Dict[str, object]] = {
    "web": {"description": "Web 検索と抽出", "tools": ["web_search", "web_extract"], "includes": []},
    "search": {"description": "Web 検索のみ", "tools": ["web_search"], "includes": []},
    "vision": {"description": "画像解析", "tools": ["vision_analyze"], "includes": []},
    "image_gen": {"description": "画像生成", "tools": ["image_generate"], "includes": []},
    "terminal": {"description": "端末実行", "tools": ["terminal", "process"], "includes": []},
    "file": {"description": "ファイル操作", "tools": ["read_file", "write_file", "patch", "search_files"], "includes": []},
    "browser": {"description": "ブラウザ自動化", "tools": ["browser_navigate", "browser_snapshot", "browser_click", "browser_type", "browser_scroll", "browser_back", "web_search"], "includes": []},
    "moa": {"description": "高度推論", "tools": ["mixture_of_agents"], "includes": []},
    "skills": {"description": "スキル管理", "tools": ["skills_list", "skill_view", "skill_manage"], "includes": []},
    "development": {"description": "開発向け", "tools": [], "includes": ["terminal", "file", "web"]},
    "research": {"description": "調査向け", "tools": [], "includes": ["web", "browser", "vision"]},
    "all": {"description": "主要 tool を全部", "tools": _HERMES_CORE_TOOLS, "includes": []},
}


def _resolve(name: str, seen: Set[str] | None = None) -> List[str]:
    if name not in TOOLSETS:
        raise ValueError(f"未知の toolset: {name}")
    seen = seen or set()
    if name in seen:
        return []
    seen.add(name)
    entry = TOOLSETS[name]
    tools = list(entry.get("tools", []))
    for child in entry.get("includes", []):
        tools.extend(_resolve(str(child), seen))
    deduped: list[str] = []
    for tool in tools:
        if tool not in deduped:
            deduped.append(tool)
    return deduped


def resolve_toolset(name: str) -> List[str]:
    return _resolve(name)


def validate_toolset(name: str) -> bool:
    return name in TOOLSETS


def get_all_toolsets() -> Dict[str, Dict[str, object]]:
    return TOOLSETS


def get_toolset_info(name: str) -> Dict[str, object]:
    if name not in TOOLSETS:
        raise ValueError(f"未知の toolset: {name}")
    return TOOLSETS[name]
