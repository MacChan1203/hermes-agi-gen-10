"""バッチ実行向け toolset 分布。"""
from __future__ import annotations

import random
from typing import Dict, Optional

from .toolsets import validate_toolset

DISTRIBUTIONS = {
    "default": {
        "description": "主要 toolset を常時有効",
        "toolsets": {"web": 100, "vision": 100, "image_gen": 100, "terminal": 100, "file": 100, "moa": 100, "browser": 100},
    },
    "development": {
        "description": "開発寄り",
        "toolsets": {"terminal": 90, "file": 90, "web": 45, "browser": 20, "vision": 10},
    },
    "research": {
        "description": "調査寄り",
        "toolsets": {"web": 95, "browser": 75, "vision": 45, "terminal": 15},
    },
    "safe": {
        "description": "端末なしの安全モード",
        "toolsets": {"web": 80, "browser": 70, "vision": 60, "image_gen": 60},
    },
}


def get_distribution(name: str) -> Optional[Dict[str, object]]:
    return DISTRIBUTIONS.get(name)


def list_distributions() -> Dict[str, Dict[str, object]]:
    return DISTRIBUTIONS


def validate_distribution(name: str) -> bool:
    return name in DISTRIBUTIONS


def sample_toolsets_from_distribution(name: str) -> list[str]:
    dist = get_distribution(name)
    if dist is None:
        raise ValueError(f"未知の distribution: {name}")
    chosen: list[str] = []
    for toolset, pct in dist["toolsets"].items():
        if not validate_toolset(toolset):
            continue
        if random.random() * 100 <= pct:
            chosen.append(toolset)
    if not chosen:
        chosen.append("web")
    return chosen
