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

from .config import (
    UCB1_EXPLORATION_CONSTANT,
    META_LEARNING_EPISODE_WINDOW,
    META_EXPLORATION_RATE_MIN,
    META_EXPLORATION_RATE_MAX,
    META_TRANSFER_BASE_CONFIDENCE,
    META_MIN_TRIALS_FOR_TRANSFER,
    META_MAX_TRANSFER_CANDIDATES,
    META_TRANSFER_HISTORY_WINDOW,
    META_MIN_SAMPLES_FOR_ADJUSTMENT,
    META_TRANSFER_DECAY_FACTOR,
    UCB1_DECAY_RATE,
    UCB1_DECAY_MIN,
    DOMAIN_SEMANTIC_VECTORS,
    DOMAIN_VECTOR_MIN_USES,
    DOMAIN_VECTOR_STRATEGY_NAMES,
)


from .hermes_constants import get_hermes_home

_META_LEARNING_DB = get_hermes_home() / "meta_learning.db"

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
        self._exploration_rate = self._load_param("exploration_rate", UCB1_EXPLORATION_CONSTANT)
        self._transfer_threshold = self._load_param("transfer_threshold", META_TRANSFER_BASE_CONFIDENCE)
        self._learning_rate = self._load_param("learning_rate", 0.1)
        self._transfer_success_history: List[bool] = []  # 転移成功/失敗の履歴
        self._learned_vectors: Dict[str, List[float]] = {}  # ドメイン → 学習済みベクトル

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

        # UCB1 探索定数の自然減衰: 経験が蓄積されるにつれ活用を重視する
        if total_uses > 0:
            self._exploration_rate = max(
                UCB1_DECAY_MIN,
                self._exploration_rate * (UCB1_DECAY_RATE ** total_uses),
            )
            C = self._exploration_rate

        best = None
        best_ucb = -1.0

        for r in rows:
            avg = r["avg_reward"]

            if r["total_uses"] == 0:
                # 未試行の戦略には無限大の探索ボーナスを与える
                ucb = float('inf')
            elif total_uses == 0:
                ucb = avg + C  # 全体未使用時のフォールバック
            else:
                ucb = avg + C * math.sqrt(math.log(max(total_uses, 1)) / r["total_uses"])

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

        # 戦略統計更新: ドメイン固有レコードを優先し、なければ作成する
        # 1. まずドメイン固有レコードを探す
        row = self._conn.execute(
            "SELECT total_uses, total_reward, successes FROM strategies WHERE name = ? AND domain = ?",
            (strategy_name, domain),
        ).fetchone()

        if row is None and domain != "general":
            # ドメイン固有レコードがない場合、general から複製して作成
            self.register_strategy(strategy_name, domain)
            row = self._conn.execute(
                "SELECT total_uses, total_reward, successes FROM strategies WHERE name = ? AND domain = ?",
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
                WHERE name = ? AND domain = ?
                """,
                (new_uses, new_reward, new_successes, new_avg, now, strategy_name, domain),
            )

        # general の統計も同時に更新 (全体統計として)
        if domain != "general":
            gen_row = self._conn.execute(
                "SELECT total_uses, total_reward, successes FROM strategies WHERE name = ? AND domain = 'general'",
                (strategy_name,),
            ).fetchone()
            if gen_row:
                gu = gen_row["total_uses"] + 1
                gr = gen_row["total_reward"] + reward
                gs = gen_row["successes"] + (1 if reward >= 0.5 else 0)
                self._conn.execute(
                    "UPDATE strategies SET total_uses=?, total_reward=?, successes=?, avg_reward=?, last_used_at=? WHERE name=? AND domain='general'",
                    (gu, gr, gs, gr / gu, now, strategy_name),
                )

        self._conn.commit()

        # 学習率を適応的に調整
        self._adapt_learning_params()

        # ドメインベクトルを定期的に再学習 (10エピソードごと)
        episode_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM strategy_episodes"
        ).fetchone()["cnt"]
        if episode_count % 10 == 0:
            self.learn_domain_vectors()

    # ------------------------------------------------------------------
    # 転移学習
    # ------------------------------------------------------------------

    def find_transfer_candidates(self, target_domain: str) -> List[TransferCandidate]:
        """他ドメインの成功戦略を転移候補として返す。"""
        # ターゲットドメインにまだ十分なデータがない戦略を探す
        target_strategies = set()
        rows = self._conn.execute(
            "SELECT name FROM strategies WHERE domain = ? AND total_uses >= ?",
            (target_domain, META_MIN_TRIALS_FOR_TRANSFER),
        ).fetchall()
        target_strategies = {r["name"] for r in rows}

        # 他ドメインの高成功率戦略を検索
        candidates = []
        rows = self._conn.execute(
            """
            SELECT name, domain, avg_reward, total_uses
            FROM strategies
            WHERE domain != ? AND domain != 'general'
              AND total_uses >= ? AND avg_reward >= ?
            ORDER BY avg_reward DESC
            LIMIT 10
            """,
            (target_domain, META_MIN_TRIALS_FOR_TRANSFER, self._transfer_threshold),
        ).fetchall()

        for r in rows:
            if r["name"] not in target_strategies:
                # ドメイン類似度を簡易推定（同じ戦略が汎用で使えるか）
                general_row = self._conn.execute(
                    "SELECT avg_reward FROM strategies WHERE name = ? AND domain = 'general'",
                    (r["name"],),
                ).fetchone()
                general_score = general_row["avg_reward"] if general_row else META_TRANSFER_BASE_CONFIDENCE

                # ドメイン類似度: ソースとターゲットで共通の成功戦略の割合
                domain_similarity = self._compute_domain_similarity(r["domain"], target_domain)

                # 転移確信度: ソース成功率 × 汎用成功率 × ドメイン類似度
                confidence = r["avg_reward"] * general_score * domain_similarity

                candidates.append(TransferCandidate(
                    strategy_name=r["name"],
                    source_domain=r["domain"],
                    target_domain=target_domain,
                    source_reward=r["avg_reward"],
                    transfer_confidence=min(1.0, confidence),
                ))

        candidates.sort(key=lambda c: c.transfer_confidence, reverse=True)
        return candidates[:META_MAX_TRANSFER_CANDIDATES]

    def apply_transfer(self, candidate: TransferCandidate) -> None:
        """転移候補をターゲットドメインに登録する。"""
        self.register_strategy(
            candidate.strategy_name,
            candidate.target_domain,
            f"[転移] {candidate.source_domain}から (確信度={candidate.transfer_confidence:.0%})",
        )

    def _compute_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """ソースとターゲットドメインの類似度を算出する。

        1. 学習済みベクトル (実績データから自動生成) を最優先
        2. config の手動ベクトルにフォールバック
        3. ベクトルが存在しない場合は Jaccard 類似度にフォールバック
        """
        # 学習済みベクトル → config フォールバックの順で取得
        src_vec = self.get_domain_vector(source_domain)
        tgt_vec = self.get_domain_vector(target_domain)
        if src_vec and tgt_vec:
            dot = sum(a * b for a, b in zip(src_vec, tgt_vec))
            norm_s = math.sqrt(sum(a * a for a in src_vec))
            norm_t = math.sqrt(sum(b * b for b in tgt_vec))
            if norm_s > 0 and norm_t > 0:
                cosine = dot / (norm_s * norm_t)
                return max(0.3, cosine)

        # フォールバック: Jaccard 類似度 (共通成功戦略)
        source_rows = self._conn.execute(
            "SELECT name FROM strategies WHERE domain = ? AND successes > 0",
            (source_domain,),
        ).fetchall()
        target_rows = self._conn.execute(
            "SELECT name FROM strategies WHERE domain = ? AND successes > 0",
            (target_domain,),
        ).fetchall()

        source_strategies = {r["name"] for r in source_rows}
        target_strategies = {r["name"] for r in target_rows}

        union = source_strategies | target_strategies
        if not union:
            return 0.5

        intersection = source_strategies & target_strategies
        return max(0.3, len(intersection) / len(union))

    def record_transfer_outcome(self, success: bool) -> None:
        """転移学習の成否を記録し、閾値を適応的に調整する。"""
        self._transfer_success_history.append(success)

        # 履歴を上限(WINDOW * 2)に制限し、古いものを刈り込む
        max_history = META_TRANSFER_HISTORY_WINDOW * 2
        if len(self._transfer_success_history) > max_history:
            self._transfer_success_history = self._transfer_success_history[-max_history:]

        # 直近WINDOW件で判断
        recent = self._transfer_success_history[-META_TRANSFER_HISTORY_WINDOW:]
        if len(recent) < META_MIN_SAMPLES_FOR_ADJUSTMENT:
            return

        # 指数減衰重み付き成功率: 最新エントリほど高い重み
        weighted_successes = sum(
            META_TRANSFER_DECAY_FACTOR ** i * (1.0 if s else 0.0)
            for i, s in enumerate(reversed(recent))
        )
        total_weight = sum(
            META_TRANSFER_DECAY_FACTOR ** i
            for i in range(len(recent))
        )
        success_rate = weighted_successes / total_weight

        if success_rate > 0.6:
            # 成功率が高い: 閾値を下げて転移を促進
            self._transfer_threshold = max(0.3, self._transfer_threshold - 0.05)
        elif success_rate < 0.4:
            # 失敗率が高い: 閾値を上げて転移を慎重に
            self._transfer_threshold = min(0.8, self._transfer_threshold + 0.05)
        self._save_param("transfer_threshold", self._transfer_threshold)

    # ------------------------------------------------------------------
    # 適応的学習パラメータ
    # ------------------------------------------------------------------

    def _adapt_learning_params(self) -> None:
        """最近の改善率から学習パラメータを動的調整する。"""
        # 最近のエピソードの報酬推移を取得
        rows = self._conn.execute(
            "SELECT reward FROM strategy_episodes ORDER BY created_at DESC LIMIT ?",
            (META_LEARNING_EPISODE_WINDOW,),
        ).fetchall()

        if len(rows) < 5:
            return

        rewards = [r["reward"] for r in rows]
        recent_half = rewards[:len(rewards)//2]
        older_half = rewards[len(rewards)//2:]

        if not recent_half or not older_half:
            return
        recent_avg = sum(recent_half) / len(recent_half)
        older_avg = sum(older_half) / len(older_half)

        improvement = recent_avg - older_avg

        if improvement > 0.1:
            # 改善中: 活用を強化（探索率を下げる）
            self._exploration_rate = max(META_EXPLORATION_RATE_MIN, self._exploration_rate * 0.95)
        elif improvement < -0.1:
            # 悪化中: 探索を強化
            self._exploration_rate = min(META_EXPLORATION_RATE_MAX, self._exploration_rate * 1.1)

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
        learned = len(self._learned_vectors)
        return (
            f"[MetaLearner] 戦略数={total_row['cnt']} "
            f"総使用={total_row['uses'] or 0} "
            f"エピソード={episode_row['cnt']} "
            f"探索率={self._exploration_rate:.2f} "
            f"学習済みドメインベクトル={learned}"
        )

    # ------------------------------------------------------------------
    # ドメインベクトル自動学習
    # ------------------------------------------------------------------

    def learn_domain_vectors(self) -> Dict[str, List[float]]:
        """実績データからドメイン意味ベクトルを自動生成する。

        各ドメインについて、既定の戦略 (DOMAIN_VECTOR_STRATEGY_NAMES) ごとの
        avg_reward を要素としたベクトルを構築する。

        データが十分なドメイン (total_uses >= DOMAIN_VECTOR_MIN_USES) のみ生成し、
        不十分なドメインは config のフォールバックベクトルを維持する。

        Returns:
            学習済み + フォールバックのマージ済みベクトル辞書
        """
        strategy_names = DOMAIN_VECTOR_STRATEGY_NAMES
        dim = len(strategy_names)

        # 全ドメインの一覧を取得
        domain_rows = self._conn.execute(
            "SELECT DISTINCT domain FROM strategies WHERE domain != 'general'"
        ).fetchall()
        domains = [r["domain"] for r in domain_rows]

        learned: Dict[str, List[float]] = {}
        for domain in domains:
            # ドメインの総使用回数を確認
            usage_row = self._conn.execute(
                "SELECT SUM(total_uses) as total FROM strategies WHERE domain = ?",
                (domain,),
            ).fetchone()
            total_uses = usage_row["total"] or 0

            if total_uses < DOMAIN_VECTOR_MIN_USES:
                continue  # データ不足 → フォールバック

            # 各戦略の avg_reward をベクトルの各次元に
            vec: List[float] = []
            for strat_name in strategy_names:
                row = self._conn.execute(
                    "SELECT avg_reward FROM strategies WHERE name = ? AND domain = ?",
                    (strat_name, domain),
                ).fetchone()
                if row:
                    vec.append(max(0.0, min(1.0, row["avg_reward"])))
                else:
                    # ドメインにこの戦略がない場合、general の値を使用
                    gen_row = self._conn.execute(
                        "SELECT avg_reward FROM strategies WHERE name = ? AND domain = 'general'",
                        (strat_name,),
                    ).fetchone()
                    vec.append(gen_row["avg_reward"] if gen_row else 0.5)

            learned[domain] = vec

        # general ドメインのベクトルも生成
        gen_vec: List[float] = []
        for strat_name in strategy_names:
            row = self._conn.execute(
                "SELECT avg_reward FROM strategies WHERE name = ? AND domain = 'general'",
                (strat_name,),
            ).fetchone()
            gen_vec.append(row["avg_reward"] if row else 0.5)
        learned["general"] = gen_vec

        # キャッシュ更新
        self._learned_vectors = learned
        return learned

    def get_domain_vector(self, domain: str) -> Optional[List[float]]:
        """ドメインのベクトルを返す (学習済み優先、フォールバック付き)。"""
        # 学習済みベクトルを優先
        if domain in self._learned_vectors:
            return self._learned_vectors[domain]
        # config のフォールバック
        return DOMAIN_SEMANTIC_VECTORS.get(domain)
