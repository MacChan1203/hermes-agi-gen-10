"""長期記憶エンジン。セッションをまたいで知識・戦略・失敗パターンを蓄積する。

セマンティック検索: Ollamaの埋め込みAPIを使用し、意味的に類似した記憶を想起する。
Ollamaが利用不可の場合はTF-IDFコサイン類似度にフォールバック。
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import requests

from .config import LTM_EMBEDDING_TIMEOUT, LTM_EMBEDDING_DETECT_TIMEOUT, LTM_MAX_FACTS, LTM_CLEANUP_BATCH

logger = logging.getLogger(__name__)

from .hermes_constants import get_hermes_home

DEFAULT_LTM_PATH = get_hermes_home() / "long_term_memory.db"

# Ollama埋め込みモデル (nomic-embed-textが最良、qwen3はフォールバック)
_EMBED_MODEL_CANDIDATES = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
_OLLAMA_BASE = "http://127.0.0.1:11434"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    session_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    embedding TEXT          -- JSON array of floats (セマンティック検索用)
);

CREATE TABLE IF NOT EXISTS strategy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_hash TEXT NOT NULL,
    goal TEXT NOT NULL,
    strategy TEXT NOT NULL,
    outcome TEXT NOT NULL,
    session_id TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_goal ON strategy_log(goal_hash, outcome);
CREATE INDEX IF NOT EXISTS idx_strategy_recent ON strategy_log(created_at DESC);

CREATE TABLE IF NOT EXISTS failure_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_pattern TEXT NOT NULL,
    error_type TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    last_session_id TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    UNIQUE(command_pattern, error_type)
);
"""


# ------------------------------------------------------------------
# セマンティックインデクサー
# ------------------------------------------------------------------

class SemanticIndexer:
    """Ollamaの埋め込みAPIを使ってテキストをベクトル化する。
    Ollamaが利用不可の場合はTF-IDFコサイン類似度にフォールバック。
    """

    def __init__(self, ollama_base: str = _OLLAMA_BASE) -> None:
        self.ollama_base = ollama_base
        self._embed_model: Optional[str] = None
        self._use_ollama = True  # フォールバック前はTrueとする

    def _detect_embed_model(self) -> Optional[str]:
        """利用可能な埋め込みモデルを自動検出する。"""
        try:
            resp = requests.get(f"{self.ollama_base}/api/tags", timeout=LTM_EMBEDDING_DETECT_TIMEOUT)
            if resp.status_code == 200:
                models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
                for candidate in _EMBED_MODEL_CANDIDATES:
                    if any(candidate in m for m in models):
                        return candidate
        except Exception:
            pass
        return None

    def embed(self, text: str) -> Optional[List[float]]:
        """テキストを埋め込みベクトルに変換する。"""
        if not self._use_ollama:
            return None

        if self._embed_model is None:
            self._embed_model = self._detect_embed_model()
            if self._embed_model is None:
                self._use_ollama = False
                logger.info("Ollama embedding unavailable, falling back to TF-IDF")
                return None

        try:
            resp = requests.post(
                f"{self.ollama_base}/api/embeddings",
                json={"model": self._embed_model, "prompt": text},
                timeout=LTM_EMBEDDING_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("embedding")
        except Exception:
            self._use_ollama = False
            logger.info("Ollama embedding unavailable, falling back to TF-IDF")
        return None

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """コサイン類似度を計算する。"""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def tfidf_similarity(query: str, text: str) -> float:
        """TF-IDFベースのコサイン類似度フォールバック。"""
        def tokenize(s: str) -> List[str]:
            # 簡易トークナイザー: 日本語・英語対応
            import re
            return re.findall(r'[\w\u3040-\u9fff]+', s.lower())

        q_tokens = tokenize(query)
        t_tokens = tokenize(text)
        if not q_tokens or not t_tokens:
            return 0.0

        q_count = Counter(q_tokens)
        t_count = Counter(t_tokens)
        all_terms = set(q_count) | set(t_count)

        dot = sum(q_count.get(t, 0) * t_count.get(t, 0) for t in all_terms)
        norm_q = math.sqrt(sum(v * v for v in q_count.values()))
        norm_t = math.sqrt(sum(v * v for v in t_count.values()))
        if norm_q == 0 or norm_t == 0:
            return 0.0
        return dot / (norm_q * norm_t)


# ------------------------------------------------------------------
# LongTermMemory
# ------------------------------------------------------------------

class LongTermMemory:
    """セッションをまたいで知識を永続化する記憶エンジン。

    セマンティック検索 (recall_similar) でOllamaの埋め込みを利用する。
    埋め込みモデルが利用不可の場合はTF-IDFフォールバック。
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_LTM_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()
        self._indexer = SemanticIndexer()
        self._learn_count: int = 0
        self._lock = threading.Lock()  # 全DB操作を保護するスレッドロック

    def _migrate(self) -> None:
        """既存DBへのカラム追加マイグレーション。"""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(knowledge)")}
        if "embedding" not in cols:
            self._conn.execute("ALTER TABLE knowledge ADD COLUMN embedding TEXT")

    # ------------------------------------------------------------------
    # Knowledge store
    # ------------------------------------------------------------------

    def learn(
        self,
        key: str,
        value: str,
        *,
        confidence: float = 1.0,
        session_id: Optional[str] = None,
    ) -> None:
        """知識を記憶する。既存のキーは上書き。埋め込みも保存する。

        スレッドセーフ: _lock で書き込みを排他制御する。
        """
        now = time.time()
        # セマンティック埋め込みを計算 (I/O なのでロック外で実行)
        embedding_json: Optional[str] = None
        vec = self._indexer.embed(f"{key}: {value}")
        if vec:
            embedding_json = json.dumps(vec)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO knowledge(key, value, confidence, session_id, created_at, updated_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    session_id = excluded.session_id,
                    updated_at = excluded.updated_at,
                    embedding = COALESCE(excluded.embedding, knowledge.embedding)
                """,
                (key, value, confidence, session_id, now, now, embedding_json),
            )
            self._conn.commit()

        # Periodic cleanup every 100 learn() calls
        self._learn_count += 1
        if self._learn_count % 100 == 0:
            self.cleanup_old_facts()

    def cleanup_old_facts(self, max_facts: int = LTM_MAX_FACTS) -> int:
        """最大事実数を超えた古い事実を削除する。

        Args:
            max_facts: 保持する最大事実数

        Returns:
            削除された事実の数
        """
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) as cnt FROM knowledge").fetchone()
            total = row["cnt"] if row else 0
            if total <= max_facts:
                return 0

            to_delete = total - max_facts
            deleted = 0
            while deleted < to_delete:
                batch = min(LTM_CLEANUP_BATCH, to_delete - deleted)
                self._conn.execute(
                    """
                    DELETE FROM knowledge WHERE key IN (
                        SELECT key FROM knowledge ORDER BY updated_at ASC LIMIT ?
                    )
                    """,
                    (batch,),
                )
                deleted += batch
            self._conn.commit()
        logger.info("LTM cleanup: deleted %d old facts (total was %d, max %d)", deleted, total, max_facts)
        return deleted

    def recall(self, key: str) -> Optional[str]:
        """キーで記憶を取り出す。"""
        row = self._conn.execute(
            "SELECT value FROM knowledge WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def recall_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の記憶を新しい順に取り出す。"""
        rows = self._conn.execute(
            "SELECT key, value, confidence, updated_at FROM knowledge ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def recall_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """クエリに意味的に類似した知識を検索する。

        Ollamaの埋め込みが利用可能な場合はコサイン類似度、
        それ以外はTF-IDFコサイン類似度を使用する。
        """
        # まずベクトル検索を試みる
        query_vec = self._indexer.embed(query)
        rows = self._conn.execute(
            "SELECT key, value, confidence, embedding FROM knowledge ORDER BY updated_at DESC LIMIT 200"
        ).fetchall()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            r = dict(row)
            embedding_json = r.pop("embedding", None)

            if query_vec and embedding_json:
                try:
                    vec = json.loads(embedding_json)
                    score = SemanticIndexer.cosine_similarity(query_vec, vec)
                except Exception:
                    score = SemanticIndexer.tfidf_similarity(query, f"{r['key']}: {r['value']}")
            else:
                score = SemanticIndexer.tfidf_similarity(query, f"{r['key']}: {r['value']}")

            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    # ------------------------------------------------------------------
    # Strategy log
    # ------------------------------------------------------------------

    def log_strategy(
        self,
        goal: str,
        strategy: str,
        outcome: str,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """ゴールに対する戦略と結果を記録する。"""
        goal_hash = hashlib.md5(goal.encode()).hexdigest()[:8]
        with self._lock:
            self._conn.execute(
                "INSERT INTO strategy_log(goal_hash, goal, strategy, outcome, session_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (goal_hash, goal, strategy, outcome, session_id, time.time()),
            )
            self._conn.commit()

    def recall_strategies(self, goal: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        """類似ゴールでの過去の戦略を取り出す。

        セマンティック類似度でゴールテキストをランキングする。
        """
        rows = self._conn.execute(
            "SELECT goal, strategy, outcome, created_at FROM strategy_log ORDER BY created_at DESC LIMIT 100"
        ).fetchall()

        query_vec = self._indexer.embed(goal)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        goal_hash = hashlib.md5(goal.encode()).hexdigest()[:8]

        for row in rows:
            r = dict(row)
            # 同一ハッシュは最高スコア
            r_hash = hashlib.md5(r["goal"].encode()).hexdigest()[:8]
            if r_hash == goal_hash:
                score = 1.0
            elif query_vec:
                row_vec = self._indexer.embed(r["goal"])
                if row_vec:
                    score = SemanticIndexer.cosine_similarity(query_vec, row_vec)
                else:
                    score = SemanticIndexer.tfidf_similarity(goal, r["goal"])
            else:
                score = SemanticIndexer.tfidf_similarity(goal, r["goal"])
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def get_successful_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """成功した戦略を新しい順に取り出す。"""
        rows = self._conn.execute(
            "SELECT goal, strategy, created_at FROM strategy_log WHERE outcome = 'success' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Failure log
    # ------------------------------------------------------------------

    def log_failure(
        self,
        command: str,
        error_type: str,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """失敗パターンを記録する。同じパターンはカウントアップ。"""
        now = time.time()
        pattern = command[:100]
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO failure_log(command_pattern, error_type, count, last_session_id, first_seen, last_seen)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(command_pattern, error_type) DO UPDATE SET
                    count = count + 1,
                    last_session_id = excluded.last_session_id,
                    last_seen = excluded.last_seen
                """,
                (pattern, error_type, session_id, now, now),
            )
            self._conn.commit()

    def get_known_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """既知の失敗パターンを頻度順に取り出す。"""
        rows = self._conn.execute(
            "SELECT command_pattern, error_type, count FROM failure_log ORDER BY count DESC, last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def is_known_failure(self, command: str, error_type: str) -> bool:
        """このコマンド＋エラーが2回以上失敗しているか。"""
        pattern = command[:100]
        row = self._conn.execute(
            "SELECT count FROM failure_log WHERE command_pattern = ? AND error_type = ?",
            (pattern, error_type),
        ).fetchone()
        return row is not None and row["count"] >= 2
