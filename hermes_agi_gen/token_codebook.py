"""Gen 10.2 — 離散トークン語彙とその強化学習統計。

階層的・予測的AIサマリ ④離散トークン / ⑤RL に対応するモジュール。
固定サイズの離散語彙を持ち、各トークンの「使われ方の良さ」を
EMA で更新し、emit() スコアにも avg_reward を反映することで
RL ループを閉じる (= ⑤ 良いトークンを強化)。

設計方針:
- 副作用なしで純粋に in-memory で動く。永続化は LongTermMemory に任せる。
- 既存の BellmanEvaluator から `bonus_for(token_id)` を将来 reward
  ソースとして引けるよう、軽量な数値 API を持たせる。
- 解釈層 (token_interpreter.py) からは label / examples を読めるよう保持。
- 受信側の予測誤差計算用 expected_patterns を語彙定義に統合 (一元管理)。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .config import (
    TOKEN_CODEBOOK_DEFAULT_VOCAB,
    TOKEN_CODEBOOK_EMIT_REWARD_WEIGHT,
    TOKEN_CODEBOOK_REWARD_EMA_ALPHA,
    TOKEN_CODEBOOK_REWARD_INIT,
    TOKEN_CODEBOOK_BONUS_MAX,
    TOKEN_CODEBOOK_BONUS_MIN,
    TOKEN_CODEBOOK_EXAMPLE_CAP,
)


@dataclass
class TokenStats:
    token_id: str
    label: str                                  # 人間可読の意味ラベル
    keywords: Tuple[str, ...]                   # 自然言語 → トークンマッチング語
    expected_patterns: Tuple[str, ...] = ()     # 受信側が予測する特徴語 (意味の核)
    n_used: int = 0
    avg_reward: float = TOKEN_CODEBOOK_REWARD_INIT
    examples: List[str] = field(default_factory=list)
    last_used: float = 0.0


def _normalize_vocab_entry(entry: tuple) -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]:
    """vocab タプルを (id, label, keywords, expected) の 4-tuple に正規化。

    後方互換のため 3-tuple (id, label, keywords) も受け付け、expected は () とする。
    """
    if len(entry) == 4:
        tid, label, kws, exp = entry
    elif len(entry) == 3:
        tid, label, kws = entry
        exp = ()
    else:
        raise ValueError(f"vocab entry は 3 または 4 要素のタプル: {entry!r}")
    return (
        str(tid),
        str(label),
        tuple(str(k).lower() for k in kws),
        tuple(str(p).lower() for p in exp),
    )


class TokenCodebook:
    """離散トークン辞書 + 各トークンの強化統計。

    使い方:
        codebook = TokenCodebook()
        tok = codebook.emit("Pythonでクイックソートを実装して")  # → "T_ALGO"
        # ... 相手が tok を使って予測 ...
        codebook.record_reward(tok, reward=0.8)
        bonus = codebook.bonus_for(tok)   # BellmanEvaluator 等に流す

    Note: vocab の最初の要素は「分類不能フォールバック」として扱う。
    別のフォールバックを使いたい場合は fallback_id を明示的に渡すこと。
    """

    def __init__(
        self,
        vocab: Optional[Iterable[tuple]] = None,
        fallback_id: Optional[str] = None,
    ) -> None:
        if vocab is None:
            vocab = TOKEN_CODEBOOK_DEFAULT_VOCAB
        self._tokens: Dict[str, TokenStats] = {}
        for entry in vocab:
            tid, label, kws, exp = _normalize_vocab_entry(entry)
            self._tokens[tid] = TokenStats(
                token_id=tid,
                label=label,
                keywords=kws,
                expected_patterns=exp,
            )
        if not self._tokens:
            raise ValueError("vocab が空です")
        if fallback_id is None:
            fallback_id = next(iter(self._tokens))
        elif fallback_id not in self._tokens:
            raise ValueError(f"fallback_id={fallback_id!r} は vocab に存在しません")
        self._fallback_id: str = fallback_id

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def fallback_id(self) -> str:
        return self._fallback_id

    def ids(self) -> List[str]:
        return list(self._tokens.keys())

    def get(self, token_id: str) -> Optional[TokenStats]:
        return self._tokens.get(token_id)

    def label_of(self, token_id: str) -> str:
        stats = self._tokens.get(token_id)
        return stats.label if stats else "UNKNOWN"

    def expected_patterns_of(self, token_id: str) -> Tuple[str, ...]:
        stats = self._tokens.get(token_id)
        return stats.expected_patterns if stats else ()

    # ------------------------------------------------------------------
    # Emit (自然言語 → 離散トークン)
    # ------------------------------------------------------------------
    def lookup(self, intent_text: str) -> str:
        """副作用なしで intent_text に最も適合するトークン id を返す。

        スコアリングは emit() と同一だが n_used / examples を更新しない。
        BellmanEvaluator の peer_reward_hook など、候補評価で頻繁に呼ばれる
        側ではこちらを使うこと (使用統計の汚染を防ぐ)。
        """
        text_l = (intent_text or "").lower()
        best_id = self._fallback_id
        best_score = float("-inf")
        for tid, stats in self._tokens.items():
            hits = sum(1 for k in stats.keywords if k and k in text_l)
            if hits == 0:
                continue
            reward_term = (stats.avg_reward - 0.5) * 2.0  # ∈ [-1, 1]
            score = float(hits) + TOKEN_CODEBOOK_EMIT_REWARD_WEIGHT * reward_term
            if score > best_score:
                best_id = tid
                best_score = score
        return best_id

    def emit(self, intent_text: str) -> str:
        """発話意図テキストから最も適合する離散トークン id を返し、使用統計を更新する。

        スコア = hits + λ * (avg_reward - 0.5) * 2
          - hits: keyword サブストリング一致数
          - λ: TOKEN_CODEBOOK_EMIT_REWARD_WEIGHT
          - 第二項は [-λ, +λ] に正規化された強化学習補正

        ヒット 0 で reward 補正だけが正でも fallback を選ぶ
        (語彙外発話を勝手に既存トークンに割り当てない)。
        """
        best_id = self.lookup(intent_text)
        stats = self._tokens[best_id]
        stats.n_used += 1
        stats.last_used = time.time()
        if intent_text and len(stats.examples) < TOKEN_CODEBOOK_EXAMPLE_CAP:
            stats.examples.append(intent_text[:80])
        return best_id

    # ------------------------------------------------------------------
    # Reinforcement (⑤ RL)
    # ------------------------------------------------------------------
    def record_reward(self, token_id: str, reward: float) -> float:
        """EMA でトークンの平均報酬を更新する。新しい avg_reward を返す。"""
        stats = self._tokens.get(token_id)
        if stats is None:
            return 0.0
        a = TOKEN_CODEBOOK_REWARD_EMA_ALPHA
        stats.avg_reward = (1.0 - a) * stats.avg_reward + a * float(reward)
        return stats.avg_reward

    def bonus_for(self, token_id: str) -> float:
        """BellmanEvaluator 等が即時報酬に上乗せするボーナス。

        avg_reward を中立 0.5 を中心に [-1, +1] へ正規化し、
        [BONUS_MIN, BONUS_MAX] にクリップして返す。
        未使用トークンは avg_reward=REWARD_INIT(=0.5) のため 0 付近。
        """
        stats = self._tokens.get(token_id)
        if stats is None:
            return 0.0
        centered = (stats.avg_reward - 0.5) * 2.0
        return max(TOKEN_CODEBOOK_BONUS_MIN, min(TOKEN_CODEBOOK_BONUS_MAX, centered))

    # ------------------------------------------------------------------
    # Snapshot (永続化用) — examples / expected_patterns も保持
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, object]]:
        return {
            tid: {
                "label": s.label,
                "keywords": list(s.keywords),
                "expected_patterns": list(s.expected_patterns),
                "n_used": s.n_used,
                "avg_reward": s.avg_reward,
                "examples": list(s.examples),
                "last_used": s.last_used,
            }
            for tid, s in self._tokens.items()
        }

    def load_snapshot(self, data: Dict[str, Dict[str, object]]) -> None:
        for tid, d in (data or {}).items():
            stats = self._tokens.get(tid)
            if stats is None:
                continue
            stats.n_used = int(d.get("n_used", 0))
            stats.avg_reward = float(d.get("avg_reward", TOKEN_CODEBOOK_REWARD_INIT))
            stats.last_used = float(d.get("last_used", 0.0))
            ex = d.get("examples", [])
            if isinstance(ex, list):
                stats.examples = [str(x)[:80] for x in ex[:TOKEN_CODEBOOK_EXAMPLE_CAP]]
