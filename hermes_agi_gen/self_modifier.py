"""自己改善エンジン: ソースコードの自律的修正と検証。

propose → apply → test → accept/rollback の安全なサイクルで
エージェント自身のコードを改善する。

安全対策:
- 許可されたファイルのみ修正可能 (_SAFE_MODIFY_TARGETS)
- 修正前にgit statusがクリーンであることを確認
- pytest実行で動作確認後にコミット
- 失敗時は自動ロールバック
- 全修正履歴をSQLiteに保存
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import sqlite3

if TYPE_CHECKING:
    from .mistral_client import MistralClient

from .config import SELF_MODIFIER_MAX_PENDING_HIGH_RISK

logger = logging.getLogger(__name__)

# 自己修正が許可されるファイルのホワイトリスト (Gen 9: 拡張)
_SAFE_MODIFY_TARGETS = {
    "hermes_agi_gen/planner.py",
    "hermes_agi_gen/reviewer.py",
    "hermes_agi_gen/executor.py",
    "hermes_agi_gen/meta_cognition.py",
    "hermes_agi_gen/long_term_memory.py",
    "hermes_agi_gen/world_model.py",
    "hermes_agi_gen/self_improvement.py",
    # Gen 9: 認知モジュールへの自己修正を許可
    "hermes_agi_gen/cognitive_roles.py",
    "hermes_agi_gen/consciousness.py",
    "hermes_agi_gen/predictive_engine.py",
    "hermes_agi_gen/reflection_engine.py",
    "hermes_agi_gen/intrinsic_motivation.py",
    "hermes_agi_gen/meta_learning.py",
    "hermes_agi_gen/inner_dialogue.py",
}

from .hermes_constants import get_hermes_home

_PATCH_DB_PATH = get_hermes_home() / "self_modifier.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_patches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    rationale TEXT NOT NULL,
    original_hash TEXT,
    new_hash TEXT,
    test_passed INTEGER NOT NULL DEFAULT 0,
    performance_before REAL,
    performance_after REAL,
    performance_delta REAL,
    created_at REAL NOT NULL,
    session_id TEXT
);

CREATE TABLE IF NOT EXISTS learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_category TEXT NOT NULL,
    insight_keywords TEXT NOT NULL,
    target_file TEXT NOT NULL,
    patch_template TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS high_risk_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    rationale TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    reviewed INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
"""

_PROPOSE_PATCH_PROMPT = """\
あなたは Hermes AGI のソースコードを改善するエンジニアです。
以下のファイルを分析し、具体的な改善を提案してください。

ファイル: {file_path}
分析内容:
{analysis}

=== 現在のコード ===
{current_code}

=== 指示 ===
以下の形式でJSONを返してください:
{{
  "rationale": "改善の理由 (日本語)",
  "changes": [
    {{
      "description": "変更の説明",
      "old_code": "変更前のコード (完全一致する文字列)",
      "new_code": "変更後のコード"
    }}
  ],
  "risk_level": "low|medium|high",
  "expected_benefit": "期待される改善効果"
}}

重要な制約:
- old_code は現在のコードに完全に一致する文字列のみ使用する
- 既存の動作を壊さない保守的な変更のみ
- risk_level が high の変更は提案しない
- インポート文の変更は避ける
"""


@dataclass
class PatchChange:
    """1つのコード変更。"""
    description: str
    old_code: str
    new_code: str


@dataclass
class Patch:
    """ソースファイルへのパッチ。"""
    file_path: str
    rationale: str
    changes: list[PatchChange]
    original_content: str
    risk_level: str = "low"
    expected_benefit: str = ""
    created_at: float = field(default_factory=time.time)
    session_id: Optional[str] = None


@dataclass
class TestResult:
    """テスト実行結果。"""
    passed: bool
    output: str
    duration: float
    return_code: int


class SelfModifier:
    """ソースコードを安全に自己修正するエンジン。

    使い方:
        modifier = SelfModifier(llm=llm, repo_root=Path("."))
        patch = modifier.propose_change("hermes_agi_gen/planner.py", analysis)
        if patch:
            success = modifier.validate_and_commit(patch)
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        repo_root: Path = Path("."),
        db_path: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.db_path = db_path or _PATCH_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # パッチ提案
    # ------------------------------------------------------------------

    def propose_change(self, file_path: str, analysis: str) -> Optional[Patch]:
        """LLMにコード改善を提案させる。

        Args:
            file_path: 修正対象のファイルパス (リポジトリルートからの相対パス)
            analysis: 分析内容・改善方針

        Returns:
            Patch オブジェクト、または None (提案なし・エラー)
        """
        if self.llm is None:
            return None

        # ホワイトリストチェック
        if file_path not in _SAFE_MODIFY_TARGETS:
            return None

        abs_path = self.repo_root / file_path
        if not abs_path.exists():
            return None

        current_code = abs_path.read_text(encoding="utf-8")
        if len(current_code) > 8000:
            # 長すぎるファイルは先頭8000文字のみ渡す
            current_code_for_prompt = current_code[:8000] + "\n... (省略)"
        else:
            current_code_for_prompt = current_code

        prompt = _PROPOSE_PATCH_PROMPT.format(
            file_path=file_path,
            analysis=analysis,
            current_code=current_code_for_prompt,
        )

        try:
            data = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        # Gen 9: リスク段階制 (low→自動, medium→テスト必須, high→ユーザー確認要求)
        risk = data.get("risk_level", "low")
        if risk == "high":
            # high リスクはログに記録して保留（ユーザー確認を推奨）
            logger.warning("[SelfModifier] 高リスク変更を保留: %s", data.get("rationale", "")[:60])
            self._record_high_risk_proposal(file_path, data)
            return None

        changes_raw = data.get("changes", [])
        if not changes_raw:
            return None

        changes = []
        for c in changes_raw:
            if not isinstance(c, dict):
                continue
            old = c.get("old_code", "")
            new = c.get("new_code", "")
            if old and new and old in current_code:
                changes.append(PatchChange(
                    description=c.get("description", ""),
                    old_code=old,
                    new_code=new,
                ))

        if not changes:
            return None

        return Patch(
            file_path=file_path,
            rationale=data.get("rationale", ""),
            changes=changes,
            original_content=current_code,
            risk_level=data.get("risk_level", "low"),
            expected_benefit=data.get("expected_benefit", ""),
        )

    # ------------------------------------------------------------------
    # パッチ適用・ロールバック
    # ------------------------------------------------------------------

    def _check_git_clean(self, file_path: str) -> bool:
        """対象ファイルにコミットされていない変更がないか確認する。

        Returns:
            クリーンなら True、未コミットの変更があれば False。
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", file_path],
                capture_output=True, text=True, timeout=10,
                cwd=self.repo_root,
            )
            if result.stdout.strip():
                logger.warning(
                    "[SelfModifier] %s に未コミットの変更があります。修正を拒否します。",
                    file_path,
                )
                return False
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # git が使えない環境では警告のみでスキップ
            logger.warning("[SelfModifier] git status を実行できません。チェックをスキップします。")
            return True

    def _backup_file(self, patch: Patch) -> Optional[Path]:
        """パッチ適用前にオリジナルファイルのバックアップを保存する。

        Returns:
            バックアップファイルのパス、または None。
        """
        abs_path = self.repo_root / patch.file_path
        if not abs_path.exists():
            return None
        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        safe_name = patch.file_path.replace("/", "_").replace("\\", "_")
        backup_path = backup_dir / f"{safe_name}.{timestamp}.bak"
        backup_path.write_text(patch.original_content, encoding="utf-8")
        logger.info("[SelfModifier] バックアップ作成: %s", backup_path)
        return backup_path

    def apply_patch(self, patch: Patch) -> bool:
        """パッチをファイルに適用する (アトミック書き込み)。

        適用前に git status チェックとファイルバックアップを行う。
        一時ファイルに書き込み→検証→リネームでアトミック性を保証する。
        途中失敗時にファイルが破損しない。

        Returns:
            成功したら True
        """
        import tempfile

        # git clean チェック
        if not self._check_git_clean(patch.file_path):
            return False

        # バックアップ作成
        self._backup_file(patch)

        abs_path = self.repo_root / patch.file_path
        content = patch.original_content

        for change in patch.changes:
            if change.old_code not in content:
                return False
            content = content.replace(change.old_code, change.new_code, 1)

        # アトミック書き込み: 一時ファイル→検証→リネーム
        try:
            # 同じディレクトリに一時ファイルを作成 (同一ファイルシステムでの rename を保証)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(abs_path.parent),
                prefix=f".{abs_path.name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                # 元ファイルのパーミッションを保持
                if abs_path.exists():
                    os.chmod(tmp_path, abs_path.stat().st_mode)
                # アトミックリネーム
                os.replace(tmp_path, str(abs_path))
            except Exception:
                # 失敗時は一時ファイルを削除
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as exc:
            logger.error("[SelfModifier] アトミック書き込み失敗: %s — ロールバック", exc)
            self.rollback(patch)
            return False

        return True

    def rollback(self, patch: Patch) -> None:
        """パッチを元に戻す。"""
        abs_path = self.repo_root / patch.file_path
        abs_path.write_text(patch.original_content, encoding="utf-8")

    # ------------------------------------------------------------------
    # テスト実行
    # ------------------------------------------------------------------

    def run_tests(self) -> TestResult:
        """pytest を実行してテスト結果を返す。

        テスト副作用を防ぐため、一時ディレクトリを --rootdir として使用する。
        """
        start = time.time()
        try:
            with tempfile.TemporaryDirectory(prefix="hermes_test_") as tmp_dir:
                result = subprocess.run(
                    [
                        "python3", "-m", "pytest", "tests/",
                        "-x", "-q", "--tb=short",
                        f"--rootdir={tmp_dir}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.repo_root,
                )
            duration = time.time() - start
            return TestResult(
                passed=result.returncode == 0,
                output=(result.stdout + result.stderr)[:2000],
                duration=duration,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                output="テストタイムアウト (120秒)",
                duration=120.0,
                return_code=-1,
            )
        except FileNotFoundError:
            # pytestがない場合はインポートテストだけ行う
            return self._run_import_test()

    def _run_import_test(self) -> TestResult:
        """pytestがない場合のフォールバック: インポートテスト。"""
        start = time.time()
        result = subprocess.run(
            ["python3", "-c", "import hermes_agi_gen; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.repo_root,
        )
        duration = time.time() - start
        return TestResult(
            passed=result.returncode == 0,
            output=result.stdout + result.stderr,
            duration=duration,
            return_code=result.returncode,
        )

    # ------------------------------------------------------------------
    # 安全な適用サイクル
    # ------------------------------------------------------------------

    def validate_and_commit(self, patch: Patch) -> bool:
        """パッチを適用 → テスト → 成功なら確定、失敗ならロールバック。

        Returns:
            True: パッチが受け入れられた
            False: テスト失敗でロールバックされた
        """
        # パッチを適用
        if not self.apply_patch(patch):
            return False

        # テスト実行
        test_result = self.run_tests()

        if test_result.passed:
            # 成功: 記録して確定
            self._record_patch(patch, test_result, accepted=True)
            return True
        else:
            # 失敗: ロールバック
            self.rollback(patch)
            self._record_patch(patch, test_result, accepted=False)
            return False

    # ------------------------------------------------------------------
    # 履歴記録
    # ------------------------------------------------------------------

    def _record_patch(self, patch: Patch, test_result: TestResult, accepted: bool) -> None:
        """パッチ試行を DB に記録する。"""
        import hashlib
        original_hash = hashlib.md5(patch.original_content.encode()).hexdigest()[:8]

        self._conn.execute(
            """
            INSERT INTO code_patches
            (file_path, rationale, original_hash, test_passed, created_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patch.file_path,
                patch.rationale,
                original_hash,
                1 if (accepted and test_result.passed) else 0,
                patch.created_at,
                patch.session_id,
            ),
        )
        self._conn.commit()

    def get_patch_history(self, limit: int = 10) -> list[dict]:
        """最近のパッチ試行履歴を返す。"""
        rows = self._conn.execute(
            """
            SELECT file_path, rationale, test_passed, created_at
            FROM code_patches
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_success_rate(self) -> float:
        """パッチ受け入れ率を返す。"""
        row = self._conn.execute(
            "SELECT COUNT(*) as total, SUM(test_passed) as passed FROM code_patches"
        ).fetchone()
        if row and row["total"] > 0:
            return (row["passed"] or 0) / row["total"]
        return 0.0

    # ------------------------------------------------------------------
    # Gen 9: 学習済み修正パターン
    # ------------------------------------------------------------------

    def learn_pattern(self, insight_category: str, keywords: str, target_file: str, patch_template: str) -> None:
        """成功したパッチパターンを学習する。"""
        self._conn.execute(
            """
            INSERT INTO learned_patterns (insight_category, insight_keywords, target_file, patch_template, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (insight_category, keywords, target_file, patch_template, time.time()),
        )
        self._conn.commit()

    def find_similar_pattern(self, insight_content: str) -> Optional[dict]:
        """類似の成功パターンを検索する。"""
        rows = self._conn.execute(
            """
            SELECT * FROM learned_patterns
            WHERE success_count > failure_count
            ORDER BY success_count DESC
            LIMIT 10
            """
        ).fetchall()
        # キーワードマッチング
        content_lower = insight_content.lower()
        for row in rows:
            keywords = row["insight_keywords"].lower().split(",")
            if any(kw.strip() in content_lower for kw in keywords):
                return dict(row)
        return None

    def record_pattern_outcome(self, pattern_id: int, success: bool) -> None:
        """パターンの適用結果を記録する。"""
        col = "success_count" if success else "failure_count"
        self._conn.execute(
            f"UPDATE learned_patterns SET {col} = {col} + 1 WHERE id = ?",
            (pattern_id,),
        )
        self._conn.commit()

    def _record_high_risk_proposal(self, file_path: str, data: dict) -> None:
        """高リスク提案を記録する（後でユーザーが確認可能）。

        保留中の提案が上限に達している場合、最も古い提案を削除する。
        """
        # 保留中の提案数を確認し、上限に達していれば古いものを削除
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM high_risk_proposals WHERE reviewed = 0"
        ).fetchone()
        if row and row["cnt"] >= SELF_MODIFIER_MAX_PENDING_HIGH_RISK:
            self._conn.execute(
                """
                DELETE FROM high_risk_proposals WHERE id IN (
                    SELECT id FROM high_risk_proposals
                    WHERE reviewed = 0
                    ORDER BY created_at ASC
                    LIMIT 1
                )
                """
            )
            logger.info(
                "[SelfModifier] 保留提案が上限 (%d) に達したため最古の提案を削除しました",
                SELF_MODIFIER_MAX_PENDING_HIGH_RISK,
            )

        self._conn.execute(
            """
            INSERT INTO high_risk_proposals (file_path, rationale, risk_level, proposal_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_path, data.get("rationale", ""), data.get("risk_level", "high"),
             json.dumps(data, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def get_pending_high_risk(self) -> list[dict]:
        """未レビューの高リスク提案を返す。"""
        rows = self._conn.execute(
            "SELECT * FROM high_risk_proposals WHERE reviewed = 0 ORDER BY created_at DESC LIMIT ?",
            (SELF_MODIFIER_MAX_PENDING_HIGH_RISK,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_proposals_summary(self) -> list[dict]:
        """保留中の高リスク提案の人間が読みやすいサマリーを返す。

        CLIから呼び出して保留中の提案を確認するために使用する。
        """
        rows = self._conn.execute(
            """
            SELECT id, file_path, rationale, risk_level, created_at
            FROM high_risk_proposals
            WHERE reviewed = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (SELF_MODIFIER_MAX_PENDING_HIGH_RISK,),
        ).fetchall()
        summaries = []
        for row in rows:
            from datetime import datetime
            created_dt = datetime.fromtimestamp(row["created_at"])
            summaries.append({
                "id": row["id"],
                "file": row["file_path"],
                "rationale": row["rationale"][:120],
                "risk": row["risk_level"],
                "created": created_dt.strftime("%Y-%m-%d %H:%M"),
            })
        return summaries

    def approve_high_risk(self, proposal_id: int) -> bool:
        """高リスク提案を承認して適用する。"""
        row = self._conn.execute(
            "SELECT * FROM high_risk_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        if not row:
            return False

        data = json.loads(row["proposal_json"])
        file_path = row["file_path"]

        # 通常のパッチ生成・適用フローを再実行（risk_levelチェックをバイパス）
        abs_path = self.repo_root / file_path
        if not abs_path.exists():
            return False

        current_code = abs_path.read_text(encoding="utf-8")
        changes_raw = data.get("changes", [])
        changes = []
        for c in changes_raw:
            if isinstance(c, dict) and c.get("old_code") and c.get("new_code"):
                if c["old_code"] in current_code:
                    changes.append(PatchChange(
                        description=c.get("description", ""),
                        old_code=c["old_code"],
                        new_code=c["new_code"],
                    ))
        if not changes:
            return False

        patch = Patch(
            file_path=file_path, rationale=data.get("rationale", ""),
            changes=changes, original_content=current_code,
            risk_level="high", expected_benefit=data.get("expected_benefit", ""),
        )
        accepted = self.validate_and_commit(patch)
        self._conn.execute(
            "UPDATE high_risk_proposals SET reviewed = 1 WHERE id = ?", (proposal_id,)
        )
        self._conn.commit()
        return accepted
