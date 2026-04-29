"""Gen 10.2 — エージェント間 / ロール間の薄い通信バス。

サマリ ② 2体協調 / ③ 通信 に対応。AgentMessage を運搬単位とし、
中身 (`context.tokens`) に離散トークン id 列を載せて受け取る側が
TokenCodebook で解釈する設計。

#3 (CodeGenerator ↔ CodeReviewer) で使うが、API は宛先文字列で
ルーティングするだけなので、#2 (1 エージェント内の CognitiveRole 同士の
内部対話) でもそのまま流用できる。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from .agent_message import AgentMessage
from .config import PEER_CHANNEL_MAX_INBOX


class PeerChannel:
    """役割名 (receiver) で受信箱を分けるインメモリバス。

    - send: メッセージを receiver の受信箱に積む
    - receive: 受信箱を取り出す (FIFO, デフォルトでは全件取得して空にする)
    - peek: 受信箱を見るが消費しない
    """

    def __init__(self, max_inbox: int = PEER_CHANNEL_MAX_INBOX) -> None:
        self._inboxes: Dict[str, Deque[AgentMessage]] = {}
        self._max_inbox = max_inbox
        self._history: List[AgentMessage] = []

    def _box(self, role: str) -> Deque[AgentMessage]:
        if role not in self._inboxes:
            self._inboxes[role] = deque(maxlen=self._max_inbox)
        return self._inboxes[role]

    def send(
        self,
        sender: str,
        receiver: str,
        task: str,
        tokens: Optional[List[str]] = None,
        extra: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> AgentMessage:
        """トークン列付きメッセージを送る。"""
        ctx: dict = {"tokens": list(tokens or [])}
        if extra:
            ctx.update(extra)
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            task=task,
            context=ctx,
            session_id=session_id,
        )
        self._box(receiver).append(msg)
        self._history.append(msg)
        return msg

    def receive(self, role: str) -> List[AgentMessage]:
        """role 宛の受信箱を全部取り出す (空にする)。"""
        box = self._box(role)
        out = list(box)
        box.clear()
        return out

    def peek(self, role: str) -> List[AgentMessage]:
        return list(self._box(role))

    def history(self) -> List[AgentMessage]:
        return list(self._history)

    @staticmethod
    def tokens_of(message: AgentMessage) -> List[str]:
        ctx = message.context if isinstance(message.context, dict) else {}
        toks = ctx.get("tokens", [])
        return [str(t) for t in toks] if isinstance(toks, list) else []
