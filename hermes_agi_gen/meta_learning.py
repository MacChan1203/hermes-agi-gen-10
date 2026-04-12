"""メタ学習層: 学習方法自体を学習する。

AGIの本質的な能力の一つは「どう学ぶかを学ぶ」こと。
本モジュールは以下を実装:

1. 戦略レジストリ: 利用可能な戦略とその成功実績をDB管理
2. UCB1ベース戦略選択: 探索と活用のバランスを取る多腕バンディット
3. 転移学習: あるドメインの成功戦略を別ドメインに適用
4. 適応的学習率: 最近の改善率から学習パラメータを動的調整
"""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_META_LEARNING_DB = Path.home() / ".hermes" / "meta_learning.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    domain TEXT NOT NULL,
    description TEXT DEFAULT '',
    total_uses INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    total_reward REAL DEFAULT 0.0,
    avg_reward REAL DEFAULT 0.0,
    created_at REAL NOT NULL,
    last_used_at REAL
);

CREATE TABLE IF NOT EXISTS strategy_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    goal TEXT NOT NULL,
    reward REAL NOT NULL,
    context TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_params (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_name_domain
    ON strategies(name, domain);
"""

# 既定の戦略セット
_DEFAULT_STRATEGIES = [
    ("divide_and_conquer", "ゴールを小さなサブゴールに分解して順次解決"),
    ("depth_first", "一つのアプローチを深く追求し、行き詰まったら方向転換"),
    ("breadth_first", "複数のアプローチを浅く試し、最も有望なものを深掘り"),
    ("analogy", "過去の類似タスクの解法を適用"),
    ("simplify_first", "問題を最も単純な形に還元してから解く"),
    ("observe_then_act", "まず環境を十分に観察・理解してから行動"),
    ("iterative_refinement", "初期解を素早く出し、段階的に改善"),
    ("ask_and_verify", "仮説を立て、検証実験で確認しながら進める"),
]


@dataclass
class StrategyRecord:
    """戦略の使用実績レコード。"""
    name: str
    domain: str
    description: str
    total_uses: int
    successes: int
    avg_reward: float
    ucb_score: float = 0.0


@dataclass
class TransferCandidate:
    """転移学習の候補。"""
    strategy_name: str
    source_domain: str
    target_domain: str
    source_reward: float
    transfer_confidence: float  # 転移の確信度 (0〜1)


class MetaLearner:
    """メタ学習エンジン: 戦略の選択・評価・転移を管理する。

    使い方:
        ml = MetaLearner()
        strategy = ml.select_strategy("coding")
        # ... タスク実行 ...
        ml.record_outcome("coding", strategy.name, goal, reward=0.8)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or _META_LEARNING_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ensure_default_strategies()

        # 適応的学習パラメータ
        self._exploration_rate = self._load_param("exploration_rate", 1.5)
        self._transfer_threshold = self._load_param("transfer_threshold", 0.6)
        self._learning_rate = self._load_param("learning_rate", 0.1)

    # ------------------------------------------------------------------
    # 戦略登録
    # ------------------------------------------------------------------

    def register_strategy(self, name: str, domain: str, description: str = "") -> None:
        """新しい戦略を登録する。"""
        now = time.time()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO strategies (name, domain, description, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, domain, description, now),
        )
        self._conn.commit()

    def _ensure_default_strategies(self) -> None:
        """既定の戦略セットを登録する。"""
        now = time.time()
        for name, desc in _DEFAULT_STRATEGIES:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO strategies (name, domain, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, "general", desc, now),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # UCB1ベース戦略選択
    # ------------------------------------------------------------------

    def select_strategy(self, domain: str) -> StrategyRecord:
        """UCB1アルゴリズムで最適な戦略を選択する。

        UCB1 = avg_reward + C * sqrt(ln(N) / n_i)
        C: 探索率パラメータ
        N: 全戦略の総使用回数
        n_i: 当該戦略の使用回数
        """
        # ドメイン固有戦略 + 汎用戦略を候補に
        rows = self._conn.execute(
            """
            SELECT name, domain, description, total_uses, successes, avg_reward
            FROM strategies
            WHERE domain = ? OR domain = 'general'
            ORDER BY avg_reward DESC
            """,
            (domain,),
        ).fetchall()

        if not rows:
            # フォールバック: デフォルト戦略
            return StrategyRecord(
                name="observe_then_act",
                domain="general",
                description="まず環境を十分に観察・理解してから行動",
                total_uses=0, successes=0, avg_reward=0.5,
            )

        total_uses = sum(r["total_uses"] for r in rows)
        C = self._exploration_rate

        best = None
        best_ucb = -1.0

        for r in rows:
            n = max(r["total_uses"], 1)
            avg = r["avg_reward"]

            if total_uses == 0:
                ucb = avg + C  # 未使用戦略にボーナス
            else:
                ucb = avg + C * math.sqrt(math.log(max(total_uses, 1)) / n)

            rec = StrategyRecord(
                name=r["name"],
                domain=r["domain"],
                description=r["description"],
                total_uses=r["total_uses"],
                successes=r["successes"],
                avg_reward=avg,
                ucb_score=ucb,
            )

            if ucb > best_ucb:
                best_ucb = ucb
                best = rec

        return best  # type: ignore

    # ------------------------------------------------------------------
    # 結果記録と学習
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        domain: str,
        strategy_name: str,
        goal: str,
        reward: float,
        context: str = "",
    ) -> None:
        """戦略の実行結果を記録し、統計を更新する。"""
        now = time.time()

        # エピソード記録
        self._conn.execute(
            """
            INSERT INTO strategy_episodes (strategy_name, domain, goal, reward, context, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strategy_name, domain, goal, reward, context, now),
        )

        # 戦略統計更新
        row = self._conn.execute(
            "SELECT total_uses, total_reward, successes FROM strategies WHERE name = ? AND (domain = ? OR domain = 'general') LIMIT 1",
            (strategy_name, domain),
        ).fetchone()

        if row:
            new_uses = row["total_uses"] + 1
            new_reward = row["total_reward"] + reward
            new_successes = row["successes"] + (1 if reward >= 0.5 else 0)
            new_avg = new_reward / new_uses

            self._conn.execute(
                """
                UPDATE strategies
                SET total_uses = ?, total_reward = ?, successes = ?, avg_reward = ?, last_used_at = ?
                WHERE name = ? AND (domain = ? OR domain = 'general')
                """,
                (new_uses, new_reward, new_successes, new_avg, now, strategy_name, domain),
            )
        else:
            # 新規ドメイン用に戦略を複製登録
            self.register_strategy(strategy_name, domain)
            self._conn.execute(
                """
                UPDATE strategies
                SET total_uses = 1, total_reward = ?, successes = ?, avg_reward = ?, last_used_at = ?
                WHERE name = ? AND domain = ?
                """,
                (reward, 1 if reward >= 0.5 else 0, reward, now, strategy_name, domain),
            )

        self._conn.commit()

        # 学習率を適応的に調整
        self._adapt_learning_params()

    # ------------------------------------------------------------------
    # 転移学習
    # ------------------------------------------------------------------

    def find_transfer_candidates(self, target_domain: str) -> List[TransferCandidate]:
        """他ドメインの成功戦略を転移候補として返す。"""
        # ターゲットドメインにまだ十分なデータがない戦略を探す
        target_strategies = set()
        rows = self._conn.execute(
            "SELECT name FROM strategies WHERE domain = ? AND total_uses >= 3",
            (target_domain,),
        ).fetchall()
        target_strategies = {r["name"] for r in rows}

        # 他ドメインの高成功率戦略を検索
        candidates = []
        rows = self._conn.execute(
            """
            SELECT name, domain, avg_reward, total_uses
            FROM strategies
            WHERE domain != ? AND domain != 'general'
              AND total_uses >= 3 AND avg_reward >= ?
            ORDER BY avg_reward DESC
            LIMIT 10
            """,
            (target_domain, self._transfer_threshold),
        ).fetchall()

        for r in rows:
            if r["name"] not in target_strategies:
                # ドメイン類似度を簡易推定（同じ戦略が汎用で使えるか）
                general_row = self._conn.execute(
                    "SELECT avg_reward FROM strategies WHERE name = ? AND domain = 'general'",
                    (r["name"],),
                ).fetchone()
                general_score = general_row["avg_reward"] if general_row else 0.5

                # 転移確信度: ソース成功率 × 汎用での成功率
                confidence = r["avg_reward"] * general_score

                candidates.append(TransferCandidate(
                    strategy_name=r["name"],
                    source_domain=r["domain"],
                    target_domain=target_domain,
                    source_reward=r["avg_reward"],
                    transfer_confidence=min(1.0, confidence),
                ))

        candidates.sort(key=lambda c: c.transfer_confidence, reverse=True)
        return candidates[:5]

    def apply_transfer(self, candidate: TransferCandidate) -> None:
        """転移候補をターゲットドメインに登録する。"""
        self.register_strategy(
            candidate.strategy_name,
            candidate.target_domain,
            f"[転移] {candidate.source_domain}から (確信度={candidate.transfer_confidence:.0%})",
        )

    # ------------------------------------------------------------------
    # 適応的学習パラメータ
    # ------------------------------------------------------------------

    def _adapt_learning_params(self) -> None:
        """最近の改善率から学習パラメータを動的調整する。"""
        # 最近20エピソードの報酬推移を取得
        rows = self._conn.execute(
            "SELECT reward FROM strategy_episodes ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

        if len(rows) < 5:
            return

        rewards = [r["reward"] for r in rows]
        recent_half = rewards[:len(rewards)//2]
        older_half = rewards[len(rewards)//2:]

        recent_avg = sum(recent_half) / len(recent_half)
        older_avg = sum(older_half) / len(older_half)

        improvement = recent_avg - older_avg

        if improvement > 0.1:
            # 改善中: 活用を強化（探索率を下げる）
            self._exploration_rate = max(0.5, self._exploration_rate * 0.95)
        elif improvement < -0.1:
            # 悪化中: 探索を強化
            self._exploration_rate = min(3.0, self._exploration_rate * 1.1)

        self._save_param("exploration_rate", self._exploration_rate)

    def _load_param(self, key: str, default: float) -> float:
        row = self._conn.execute(
            "SELECT value FROM learning_params WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def _save_param(self, key: str, value: float) -> None:
        self._conn.execute(
            """
            INSERT INTO learning_params (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, time.time()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # レポート
    # ------------------------------------------------------------------

    def get_top_strategies(self, domain: str = "general", limit: int = 5) -> List[StrategyRecord]:
        """ドメインの上位戦略を返す。"""
        rows = self._conn.execute(
            """
            SELECT name, domain, description, total_uses, successes, avg_reward
            FROM strategies
            WHERE domain = ? OR domain = 'general'
            ORDER BY avg_reward DESC
            LIMIT ?
            """,
            (domain, limit),
        ).fetchall()

        return [
            StrategyRecord(
                name=r["name"], domain=r["domain"], description=r["description"],
                total_uses=r["total_uses"], successes=r["successes"], avg_reward=r["avg_reward"],
            )
            for r in rows
        ]

    def summary(self) -> str:
        total_row = self._conn.execute(
            "SELECT COUNT(*) as cnt, SUM(total_uses) as uses FROM strategies"
        ).fetchone()
        episode_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM strategy_episodes"
        ).fetchone()
        return (
            f"[MetaLearner] 戦略数={total_row['cnt']} "
            f"総使用={total_row['uses'] or 0} "
            f"エピソード={episode_row['cnt']} "
            f"探索率={self._exploration_rate:.2f}"
        )
