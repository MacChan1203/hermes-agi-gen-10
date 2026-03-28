"""自己改善エンジン。過去の軌跡を分析してfew-shot例を生成し、プランナーを動的に改善する。

セッション終了後に:
1. 成功した軌跡からfew-shot例を抽出
2. 失敗パターンから「避けるべき行動」を学習
3. プランナープロンプトのfew-shot例を動的に更新
4. A/Bテストで改善の有効性を検証
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_state import AgentState
    from .mistral_client import MistralClient

_IMPROVEMENT_DB_PATH = Path.home() / ".hermes" / "self_improvement.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS few_shot_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    goal_pattern TEXT NOT NULL,    -- ゴールの要約/パターン
    good_action TEXT NOT NULL,     -- 成功したアクション
    context TEXT,                  -- 実行時のコンテキスト
    outcome TEXT NOT NULL,         -- 結果の要約
    quality_score REAL DEFAULT 0.5, -- 例の品質スコア (0〜1)
    use_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    session_id TEXT
);

CREATE TABLE IF NOT EXISTS anti_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    bad_action TEXT NOT NULL,      -- 避けるべきアクション
    error_type TEXT NOT NULL,
    lesson TEXT NOT NULL,          -- 学んだ教訓
    frequency INTEGER DEFAULT 1,
    last_seen REAL NOT NULL,
    session_id TEXT,
    UNIQUE(bad_action, error_type)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    few_shot_text TEXT NOT NULL,   -- 生成したfew-shot例のテキスト
    performance_score REAL,        -- このバージョンの平均パフォーマンス
    created_at REAL NOT NULL,
    is_active INTEGER DEFAULT 1
);
"""

_EXTRACT_FEWSHOT_PROMPT = """\
以下の成功した実行軌跡から、future sessionsで使えるfew-shot例を抽出してください。

ゴール: {goal}
ドメイン: {domain}
成功したステップ:
{steps}
観測メモ:
{observations}

良い例として再利用できる「ゴールパターン → アクション → 結果」の組み合わせを
2〜3個抽出してください。

JSON配列のみで返答:
[
  {{
    "goal_pattern": "このようなゴールに対して",
    "good_action": "CMD: xxx / SEARCH: xxx / PYTHON: xxx など",
    "context": "なぜこのアクションが良かったか",
    "outcome": "結果の要約",
    "quality_score": 0.8
  }}
]\
"""

_GENERATE_FEWSHOT_TEXT_PROMPT = """\
以下のfew-shot例をプランナーに注入するための簡潔なテキストにまとめてください。

例:
{examples}

フォーマット:
「[goal_pattern]の場合は[good_action]が有効。[context]」
という形式で各例を1〜2行にまとめてください。
日本語で、合計200字以内で。\
"""


class SelfImprovementEngine:
    """過去の軌跡を分析してエージェントを自己改善させるエンジン。"""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        llm: Optional[MistralClient] = None,
    ) -> None:
        self.db_path = db_path or _IMPROVEMENT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.llm = llm

    # ------------------------------------------------------------------
    # 軌跡分析 (セッション終了後に呼び出す)
    # ------------------------------------------------------------------

    def analyze_session(self, state: AgentState) -> None:
        """セッションの軌跡を分析して学習する。"""
        domain = getattr(state, "domain", "general")

        # 成功した軌跡からfew-shot例を抽出
        if state.completed_steps:
            self._extract_few_shot_examples(state, domain)

        # 失敗パターンからanti-patternを記録
        if state.failed_steps:
            self._record_anti_patterns(state, domain)

        # プロンプトのfew-shot例を更新
        self._update_prompt_version(domain, state.session_id)

    def _extract_few_shot_examples(self, state: AgentState, domain: str) -> None:
        """成功ステップからfew-shot例を抽出する。"""
        if self.llm is None:
            # LLMなし: ルールベースで抽出
            self._rule_based_extract(state, domain)
            return

        steps_text = "\n".join(f"- {s}" for s in state.completed_steps[-10:])
        obs_text = "\n".join(f"- {o}" for o in state.observations[-5:])

        try:
            data = self.llm.chat_json(
                [{"role": "user", "content": _EXTRACT_FEWSHOT_PROMPT.format(
                    goal=state.user_goal,
                    domain=domain,
                    steps=steps_text,
                    observations=obs_text,
                )}],
                temperature=0.3,
                max_tokens=1024,
            )
            if isinstance(data, list):
                now = time.time()
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    self._conn.execute(
                        """
                        INSERT INTO few_shot_examples
                        (domain, goal_pattern, good_action, context, outcome, quality_score, created_at, session_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            domain,
                            item.get("goal_pattern", ""),
                            item.get("good_action", ""),
                            item.get("context", ""),
                            item.get("outcome", ""),
                            float(item.get("quality_score", 0.5)),
                            now,
                            state.session_id,
                        ),
                    )
                self._conn.commit()
        except Exception:
            self._rule_based_extract(state, domain)

    def _rule_based_extract(self, state: AgentState, domain: str) -> None:
        """ルールベースのfew-shot抽出 (LLMフォールバック)。"""
        now = time.time()
        for step in state.completed_steps[-5:]:
            if step.upper().startswith(("CMD:", "SEARCH:", "PYTHON:")):
                obs = state.observations[-1] if state.observations else ""
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO few_shot_examples
                    (domain, goal_pattern, good_action, context, outcome, quality_score, created_at, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (domain, state.user_goal[:50], step, "", obs[:100], 0.6, now, state.session_id),
                )
        self._conn.commit()

    def _record_anti_patterns(self, state: AgentState, domain: str) -> None:
        """失敗ステップからanti-patternを記録する。"""
        error_history = state.working_memory.get("error_history", [])
        now = time.time()

        for i, step in enumerate(state.failed_steps[-5:]):
            error_type = error_history[i] if i < len(error_history) else "unknown_error"
            lesson = self._derive_lesson(step, error_type)

            self._conn.execute(
                """
                INSERT INTO anti_patterns(domain, bad_action, error_type, lesson, last_seen, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(bad_action, error_type) DO UPDATE SET
                    frequency = frequency + 1,
                    last_seen = excluded.last_seen,
                    session_id = excluded.session_id
                """,
                (domain, step[:200], error_type, lesson, now, state.session_id),
            )
        self._conn.commit()

    def _derive_lesson(self, failed_action: str, error_type: str) -> str:
        """失敗アクションとエラータイプから教訓を導く。"""
        lessons = {
            "missing_command": "このコマンドはインストールされていない。which/aptを先に確認する。",
            "permission_error": "権限が不足している。ls -la でパーミッションを確認する。",
            "missing_python_module": "Pythonモジュールが不足。pip listで確認してからimportする。",
            "connection_error": "接続エラー。サービスが起動しているか確認する。",
            "missing_file": "ファイルが存在しない。findコマンドで事前に確認する。",
            "syntax_error": "構文エラー。コードを実行前に確認する。",
        }
        return lessons.get(error_type, f"{error_type}が発生した。事前確認を強化する。")

    # ------------------------------------------------------------------
    # プロンプト更新
    # ------------------------------------------------------------------

    def _update_prompt_version(self, domain: str, session_id: Optional[str]) -> None:
        """最新のfew-shot例でプロンプトバージョンを更新する。"""
        examples = self.get_best_examples(domain=domain, limit=5)
        if not examples:
            return

        few_shot_text = self._generate_few_shot_text(examples, domain)
        if not few_shot_text:
            return

        # 旧バージョンを非アクティブ化
        self._conn.execute(
            "UPDATE prompt_versions SET is_active = 0 WHERE domain = ?",
            (domain,),
        )
        self._conn.execute(
            "INSERT INTO prompt_versions(domain, few_shot_text, created_at) VALUES (?, ?, ?)",
            (domain, few_shot_text, time.time()),
        )
        self._conn.commit()

    def _generate_few_shot_text(self, examples: List[Dict[str, Any]], domain: str) -> str:
        """few-shot例をプロンプト注入用テキストに変換する。"""
        if self.llm is not None:
            try:
                examples_text = "\n".join(
                    f"- {e['goal_pattern']} → {e['good_action']}: {e['context']}"
                    for e in examples[:5]
                )
                return self.llm.chat(
                    [{"role": "user", "content": _GENERATE_FEWSHOT_TEXT_PROMPT.format(
                        examples=examples_text
                    )}],
                    temperature=0.2,
                    max_tokens=256,
                ) or self._rule_based_few_shot_text(examples)
            except Exception:
                pass
        return self._rule_based_few_shot_text(examples)

    def _rule_based_few_shot_text(self, examples: List[Dict[str, Any]]) -> str:
        """ルールベースのfew-shot例テキスト生成。"""
        lines = []
        for e in examples[:3]:
            line = f"- {e.get('goal_pattern', '')}の場合: {e.get('good_action', '')}"
            if e.get("context"):
                line += f" ({e['context'][:50]})"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # クエリ
    # ------------------------------------------------------------------

    def get_best_examples(
        self, domain: str = "general", limit: int = 5
    ) -> List[Dict[str, Any]]:
        """品質スコア・使用回数でランキングしたfew-shot例を返す。"""
        rows = self._conn.execute(
            """
            SELECT goal_pattern, good_action, context, outcome, quality_score, use_count
            FROM few_shot_examples
            WHERE domain = ? OR domain = 'general'
            ORDER BY quality_score DESC, use_count DESC
            LIMIT ?
            """,
            (domain, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_anti_patterns(self, domain: str = "general", limit: int = 5) -> List[Dict[str, Any]]:
        """頻度順のanti-patternを返す。"""
        rows = self._conn.execute(
            """
            SELECT bad_action, error_type, lesson, frequency
            FROM anti_patterns
            WHERE domain = ? OR domain = 'general'
            ORDER BY frequency DESC
            LIMIT ?
            """,
            (domain, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active_few_shot_prompt(self, domain: str = "general") -> str:
        """現在アクティブなfew-shot例テキストを返す。"""
        row = self._conn.execute(
            "SELECT few_shot_text FROM prompt_versions WHERE domain = ? AND is_active = 1 ORDER BY created_at DESC LIMIT 1",
            (domain,),
        ).fetchone()
        return row["few_shot_text"] if row else ""

    def inject_into_state(self, state: AgentState) -> None:
        """ワーキングメモリにfew-shot例とanti-patternを注入する。"""
        domain = getattr(state, "domain", "general")

        few_shot = self.get_active_few_shot_prompt(domain)
        if few_shot:
            state.working_memory["few_shot_examples"] = few_shot

        anti = self.get_anti_patterns(domain, limit=3)
        if anti:
            anti_text = "\n".join(
                f"- 避けること: {a['bad_action'][:50]} → {a['lesson']}"
                for a in anti
            )
            state.working_memory["anti_patterns"] = anti_text

    def update_example_quality(self, example_id: int, score: float) -> None:
        """few-shot例の品質スコアを更新する (A/Bテスト用)。"""
        self._conn.execute(
            "UPDATE few_shot_examples SET quality_score = ?, use_count = use_count + 1 WHERE id = ?",
            (score, example_id),
        )
        self._conn.commit()
