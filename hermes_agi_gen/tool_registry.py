"""ツールレジストリ。エージェントが動的にツールを作成・登録・実行できる。

エージェントは失敗パターンを検出し、必要なツールを自ら生成して登録する。
登録されたツールはセッションをまたいで永続化される。
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import sqlite3
import threading
import time
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_REGISTRY_PATH = get_hermes_home() / "tool_registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    code TEXT NOT NULL,
    invocation_prefix TEXT NOT NULL,   -- 例: "SCREENSHOT:" "PDF_READ:" など
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    use_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    session_id TEXT
);
"""


def _compute_code_hash(code: str) -> str:
    """コードのSHA-256ハッシュを計算する。"""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# 禁止モジュール・関数 (AST解析用)
_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "sys", "shutil", "ctypes", "socket",
    "http", "urllib", "requests", "signal", "multiprocessing",
    "threading", "importlib", "code", "codeop", "compileall",
    "runpy", "webbrowser", "tempfile", "glob", "pathlib",
    "pickle", "marshal", "shelve",
})
_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "open", "breakpoint",
})
_FORBIDDEN_ATTRS = frozenset({
    "__subclasses__", "__class__", "__bases__", "__mro__",
    "__globals__", "__code__", "__builtins__",
})


def _is_tool_code_safe(code: str) -> Tuple[bool, str]:
    """ASTを解析してツールコードの安全性を検証する。

    Returns:
        (is_safe, reason) — 安全ならば (True, ""), 危険なら (False, 理由)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"構文エラー: {e}"

    for node in ast.walk(tree):
        # import X / import X as Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    return False, f"禁止モジュールのインポート: {alias.name}"

        # from X import Y — 相対インポートも検出
        if isinstance(node, ast.ImportFrom):
            # 相対インポート (from . import X) は全面禁止
            if node.module is None:
                return False, "相対インポートは禁止されています"
            top = node.module.split(".")[0]
            if top in _FORBIDDEN_MODULES:
                return False, f"禁止モジュールのインポート: {node.module}"
            # from X import * は禁止
            if any(alias.name == "*" for alias in node.names):
                return False, "ワイルドカードインポートは禁止されています"

        # 危険な関数呼び出し: eval(), exec(), __import__(), etc.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                return False, f"禁止関数の呼び出し: {func.id}()"
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_NAMES:
                return False, f"禁止関数の呼び出し: .{func.attr}()"

        # 危険な属性アクセス: __subclasses__, __class__, etc.
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS:
                return False, f"禁止属性へのアクセス: .{node.attr}"

        # 危険な名前参照
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            # 呼び出し以外でも参照自体を禁止 (変数への代入経由のバイパス防止)
            if isinstance(node.ctx, ast.Load):
                return False, f"禁止名への参照: {node.id}"

    return True, ""


class DynamicTool:
    """動的に登録されたツール。"""

    def __init__(
        self,
        name: str,
        description: str,
        code: str,
        invocation_prefix: str,
    ) -> None:
        self.name = name
        self.description = description
        self.code = code
        self.invocation_prefix = invocation_prefix.upper().rstrip(":")
        self._fn: Optional[Callable] = None
        self._code_hash: str = _compute_code_hash(code)
        self._lock = threading.Lock()

    def compile(self) -> bool:
        """ツールコードをコンパイルして実行可能にする。

        スレッドセーフ: ロックで保護し、exec 直前にハッシュを再検証する。
        """
        with self._lock:
            # セキュリティ: コード整合性を検証 (ハッシュチェック)
            current_hash = _compute_code_hash(self.code)
            if current_hash != self._code_hash:
                return False

            # セキュリティ: AST解析による危険パターン検出
            is_safe, reason = _is_tool_code_safe(self.code)
            if not is_safe:
                return False

            # exec 直前にハッシュを再検証 (TOCTOU 防御)
            final_hash = _compute_code_hash(self.code)
            if final_hash != self._code_hash:
                return False

            try:
                ns: Dict[str, Any] = {"__builtins__": {}}
                exec(self.code, ns)  # noqa: S102
                fn = ns.get("main") or ns.get(self.name) or ns.get("run") or ns.get("execute")
                if callable(fn):
                    self._fn = fn
                    return True
            except Exception:
                logger.debug("ツール '%s' のコンパイルに失敗", self.name, exc_info=True)
            return False

    def invoke(self, args: str) -> Dict[str, Any]:
        """ツールを実行する。"""
        if self._fn is None and not self.compile():
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"ツール '{self.name}' のコンパイルに失敗しました。セキュリティ制限または構文エラーの可能性があります。",
                "returncode": 1,
                "command": f"{self.invocation_prefix}: {args}",
            }
        try:
            assert self._fn is not None
            result = self._fn(args)
            if isinstance(result, str):
                return {"ok": True, "stdout": result, "stderr": "", "returncode": 0, "command": f"{self.invocation_prefix}: {args}"}
            if isinstance(result, dict):
                return result
            return {"ok": True, "stdout": str(result), "stderr": "", "returncode": 0, "command": f"{self.invocation_prefix}: {args}"}
        except Exception as e:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"ツール実行エラー: {e}",
                "returncode": 1,
                "command": f"{self.invocation_prefix}: {args}",
            }


class ToolRegistry:
    """動的ツールの登録・管理・実行を行うレジストリ。

    SQLiteで永続化し、セッション間でツールを共有する。
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or _REGISTRY_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._cache: Dict[str, DynamicTool] = {}
        self._load_all()

    def _load_all(self) -> None:
        """DBからすべてのツールをロードしてキャッシュする。"""
        rows = self._conn.execute("SELECT * FROM tools").fetchall()
        for row in rows:
            tool = DynamicTool(
                name=row["name"],
                description=row["description"],
                code=row["code"],
                invocation_prefix=row["invocation_prefix"],
            )
            self._cache[tool.invocation_prefix.upper()] = tool

    def register(
        self,
        name: str,
        description: str,
        code: str,
        invocation_prefix: str,
        *,
        session_id: Optional[str] = None,
    ) -> bool:
        """新しいツールを登録する。安全性検証またはコンパイル失敗時はFalseを返す。"""
        # 登録時にAST安全性検証 (実行前に拒否)
        is_safe, reason = _is_tool_code_safe(code)
        if not is_safe:
            return False

        tool = DynamicTool(name=name, description=description, code=code, invocation_prefix=invocation_prefix)

        # コンパイルテスト
        if not tool.compile():
            return False

        now = time.time()
        self._conn.execute(
            """
            INSERT INTO tools(name, description, code, invocation_prefix, created_at, updated_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                code = excluded.code,
                invocation_prefix = excluded.invocation_prefix,
                updated_at = excluded.updated_at,
                session_id = excluded.session_id
            """,
            (name, description, code, invocation_prefix.upper(), now, now, session_id),
        )
        self._conn.commit()
        self._cache[invocation_prefix.upper().rstrip(":")] = tool
        return True

    def get(self, invocation_prefix: str) -> Optional[DynamicTool]:
        """プレフィックスでツールを検索する。"""
        return self._cache.get(invocation_prefix.upper().rstrip(":"))

    def dispatch(self, step: str) -> Optional[Dict[str, Any]]:
        """ステップ文字列がカスタムツールにマッチする場合に実行する。"""
        step_upper = step.upper()
        for prefix, tool in self._cache.items():
            if step_upper.startswith(f"{prefix}:"):
                args = step[len(prefix) + 1:].strip()
                result = tool.invoke(args)
                # 使用カウントを更新
                self._conn.execute(
                    "UPDATE tools SET use_count = use_count + 1 WHERE name = ?",
                    (tool.name,),
                )
                self._conn.commit()
                return result
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """登録済みツールの一覧を返す。"""
        rows = self._conn.execute(
            "SELECT name, description, invocation_prefix, use_count, success_rate FROM tools ORDER BY use_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tool_descriptions_for_prompt(self) -> str:
        """プランナープロンプト用のツール説明文を生成する。"""
        tools = self.list_tools()
        if not tools:
            return ""
        lines = ["【カスタムツール (自己拡張)】"]
        for t in tools:
            lines.append(f"  {t['invocation_prefix']}: <args>  — {t['description']}")
        return "\n".join(lines)

    def generate_tool_from_need(
        self,
        need_description: str,
        llm,
        *,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """ニーズの説明から新しいツールを自動生成して登録する。

        Returns: 登録されたツールの呼び出しプレフィックス (成功時) or None
        """
        if llm is None:
            return None

        _TOOL_GEN_PROMPT = f"""\
以下のニーズを満たすPythonツール関数を作成してください。

ニーズ: {need_description}

要件:
1. 関数名は `main(args: str) -> str` とする
2. argsは文字列の引数
3. 返り値は文字列 (結果テキスト)
4. エラーは例外を上げずに文字列で返す
5. 外部ライブラリは標準ライブラリのみ使用

以下のJSON形式のみで返答:
{{
  "name": "tool_xxx (英数字とアンダースコアのみ)",
  "description": "日本語でツールの説明 (1行)",
  "prefix": "TOOLNAME (大文字、コロンなし)",
  "code": "def main(args: str) -> str:\\n    ..."
}}
"""
        data = llm.chat_json(
            [{"role": "user", "content": _TOOL_GEN_PROMPT}],
            temperature=0.3,
            max_tokens=1024,
        )

        if not isinstance(data, dict):
            return None

        name = data.get("name", "")
        description = data.get("description", "")
        prefix = data.get("prefix", "")
        code = data.get("code", "")

        if not all([name, description, prefix, code]):
            return None

        success = self.register(
            name=name,
            description=description,
            code=code,
            invocation_prefix=prefix,
            session_id=session_id,
        )
        return prefix if success else None
