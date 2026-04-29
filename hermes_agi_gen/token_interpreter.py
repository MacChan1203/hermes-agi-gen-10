"""Gen 10.2 — 離散トークン列の人間可読化。

サマリ ⑥ 解釈 (トークン → 人間言語) に対応。
TokenCodebook の label / examples を使い、最小コストで
「この通信が何を意味していたか」を文字列に展開する。
"""
from __future__ import annotations

from typing import Iterable, List

from .token_codebook import TokenCodebook


class TokenInterpreter:
    """TokenCodebook を参照して、トークン列を自然言語に翻訳する。"""

    def __init__(self, codebook: TokenCodebook) -> None:
        self._codebook = codebook

    def interpret(self, token_ids: Iterable[str]) -> str:
        parts: List[str] = []
        for tid in token_ids:
            stats = self._codebook.get(tid)
            if stats is None:
                parts.append(f"{tid}=?")
                continue
            example = stats.examples[-1] if stats.examples else ""
            head = f"{tid}[{stats.label}]"
            if example:
                parts.append(f"{head} (例: {example[:40]})")
            else:
                parts.append(head)
        return " / ".join(parts) if parts else "(empty)"

    def explain_one(self, token_id: str) -> str:
        stats = self._codebook.get(token_id)
        if stats is None:
            return f"{token_id} は未知トークンです"
        bonus = self._codebook.bonus_for(token_id)
        return (
            f"{token_id}={stats.label} (使用 {stats.n_used} 回, "
            f"平均報酬 {stats.avg_reward:.2f}, ボーナス {bonus:+.2f})"
        )
