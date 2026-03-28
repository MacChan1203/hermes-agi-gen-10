from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .world_model import WorldModel


@dataclass
class AgentState:
    user_goal: str
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "read-only"
    iteration_count: int = 0
    max_iterations: int = 8
    is_done: bool = False
    last_step: Optional[str] = None
    last_status: Optional[str] = None
    session_id: Optional[str] = None
    # マルチエージェント用
    agent_role: str = "worker"                  # orchestrator / researcher / developer / critic / worker
    parent_session_id: Optional[str] = None     # オーケストレーターのセッション ID
    # AGI 拡張
    suggested_next_goal: Optional[str] = None   # メタ認知が提案する次のゴール
    domain: str = "general"                     # タスクドメイン: general / coding / research / writing / data / ops
    context: str = ""                           # 追加コンテキスト (ユーザーが自由に渡せる背景情報)
    # 世界モデル
    world_model: Optional[WorldModel] = None    # 環境の内部表現（因果追跡）

    def summary(self) -> str:
        return (
            f"goal={self.user_goal!r}, iterations={self.iteration_count}/{self.max_iterations}, "
            f"completed={len(self.completed_steps)}, failed={len(self.failed_steps)}, done={self.is_done}"
        )
