from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentMessage:
    """エージェント間の通信メッセージ。"""

    sender: str                              # "orchestrator" or agent role
    receiver: str                            # "researcher" | "developer" | "critic"
    task: str                                # 委任するタスク内容
    context: Any = field(default_factory=dict)  # Dict or str (依存ゴールの結果テキスト)
    result: Optional[str] = None
    status: str = "pending"                  # pending / success / partial / failed
    session_id: Optional[str] = None        # 親セッション ID
