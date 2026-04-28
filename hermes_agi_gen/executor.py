"""汎用ツールディスパッチャー。

対応ツール:
  CMD: <bash>      シェルコマンド実行
  READ: <path>     ファイル読み込み
  WRITE: <path>    ファイル書き込み (次行から EOF まで)
  PYTHON: <code>   Python コード実行
  DONE: <summary>  タスク完了宣言
"""
from __future__ import annotations

import ast
import math
import operator
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ExecutorResult(TypedDict, total=False):
    """Executor の全メソッドが返す統一結果型。"""
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    command: Optional[str]
    _cost: Dict[str, Any]  # tool_type, time

from .config import (
    EXECUTOR_MAX_OUTPUT,
    EXECUTOR_SHELL_TIMEOUT,
    EXECUTOR_PYTHON_TIMEOUT,
    EXECUTOR_MAX_WRITE_SIZE,
)
from .memory import remember_successful_command, set_environment_info
from .agent_state import AgentState
from .world_model import WorldModel
from .tool_registry import ToolRegistry

# --- Module-level compiled regex patterns ---
_RE_TILDE = re.compile(r'(?<![a-zA-Z0-9_])~(?=/|$| )')
_RE_CMD_SUBSTITUTION = re.compile(r'\$\(')

# Python sandbox: 許可リスト方式
# インポート可能なモジュール (安全と判断されたもののみ)
_ALLOWED_IMPORT_MODULES = frozenset({
    "json", "math", "re", "string", "textwrap", "collections",
    "itertools", "functools", "operator", "copy", "pprint",
    "datetime", "time", "calendar", "decimal", "fractions",
    "statistics", "random", "hashlib", "hmac", "base64",
    "csv", "io", "enum", "dataclasses", "typing", "abc",
    "unicodedata", "difflib", "bisect", "heapq",
    # ネットワーク取得 (HTTP/HTTPS のみ)
    "urllib", "http", "requests", "httpx",
    # パス操作
    "os", "pathlib",
    # HN 等の軽量パース用
    "html", "xml",
})

# os モジュールで禁止する関数 (os.xxx 形式で検査)
# パス操作 (os.path.*, os.getenv, os.makedirs 等) は許可、破壊的/シェル系のみ拒否
_OS_DENY_FUNCS = frozenset({
    "system", "popen",
    "remove", "unlink", "rmdir", "removedirs", "truncate",
    "execv", "execve", "execl", "execle", "execlp", "execlpe",
    "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "fork", "forkpty", "kill", "killpg",
    "chmod", "chown", "lchmod", "lchown", "chroot",
    "setuid", "setgid", "seteuid", "setegid", "setreuid", "setregid",
    "mknod", "mkfifo", "symlink", "link", "rename", "replace",
    "umask", "nice",
    "exec", "_exec", "abort", "_exit",
})

# pathlib.Path.write_text()/unlink()/chmod() などは import 自体が安全でも
# メソッド呼び出しでファイルシステムを書き換えられるため、AST段階で拒否する。
_PATHLIB_SIDE_EFFECT_METHODS = frozenset({
    "write_text", "write_bytes", "unlink", "chmod", "lchmod",
    "rename", "replace", "rmdir", "mkdir", "touch",
    "symlink_to", "hardlink_to", "link_to", "open",
})

# open()/WRITE: で書き込みを禁止するパス (literal 引数でのみ検査可能)
#
# 設計注記:
#   本 denylist は PYTHON: コード内の open(..., 'w') に対する AST 解析ゲートと、
#   WRITE: ツールの正規化後パス検査で共用する。
_OPEN_WRITE_DENYLIST_PREFIXES = (
    "/etc", "/usr", "/bin", "/sbin", "/boot", "/dev",
    "/System", "/Library/LaunchDaemons", "/Library/LaunchAgents",
    "/private/etc", "/private/var", "/var",
)
_OPEN_WRITE_DENYLIST_COMPONENTS = frozenset({
    ".ssh", ".aws", ".gnupg", ".config",
    ".bashrc", ".zshrc", ".profile", ".bash_profile", ".zshenv",
    ".env", ".netrc", ".pypirc", "authorized_keys", "id_rsa", "id_ed25519",
})


def _is_safe_open_write_path(path: Any) -> bool:
    """open() 書き込みモードで literal 指定されたパスが安全かを判定する。

    - システムディレクトリ (/etc, /usr, ...) や秘密ファイル (~/.ssh/*, ~/.bashrc) を拒否
    - それ以外 (リポジトリ配下、明示 allowlist 配下など) は呼び出し側で追加判定する
    """
    if not isinstance(path, str):
        return False
    # ~ を展開 (ホーム直下の秘密ファイルを検出するため)
    expanded = path.replace("~", "/HOME", 1) if path.startswith("~") else path
    lower = expanded
    for prefix in _OPEN_WRITE_DENYLIST_PREFIXES:
        if lower == prefix or lower.startswith(prefix + "/"):
            return False
    parts = [p for p in expanded.replace("\\", "/").split("/") if p]
    for part in parts:
        if part in _OPEN_WRITE_DENYLIST_COMPONENTS:
            return False
    return True


def _has_protected_path_component(path: Path | str) -> bool:
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    return any(part in _OPEN_WRITE_DENYLIST_COMPONENTS for part in parts)


def _iter_write_allow_dirs() -> list[Path]:
    """Return explicitly allowed external WRITE directories.

    HERMES_WRITE_ALLOW_DIRS is a path-list (os.pathsep separated). The repo
    root is always allowed by _write_file(); this only covers intentional
    external output directories such as a user-chosen Desktop folder.
    """
    raw = os.getenv("HERMES_WRITE_ALLOW_DIRS", "")
    dirs: list[Path] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        dirs.append(Path(item).expanduser().resolve())
    return dirs


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

# 許可する組み込み関数呼び出し
_ALLOWED_BUILTINS = frozenset({
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "list", "dict", "set", "tuple", "frozenset",
    "str", "int", "float", "bool", "complex", "bytes", "bytearray",
    "type", "isinstance", "issubclass", "hasattr",
    "abs", "round", "min", "max", "sum", "pow", "divmod",
    "any", "all", "hex", "oct", "bin", "ord", "chr", "repr",
    "format", "hash", "id", "input", "iter", "next",
    "open",  # open は別途モード検査で制御
})

# 禁止する dunder 属性アクセス
_DANGEROUS_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__code__", "__import__",
    "__qualname__", "__module__", "__dict__",
})

# CMD: は shell=True を使わないが、単体コマンドでも破壊的操作はあり得る。
# 実行前にコマンド名で拒否する。
_BLOCKED_SHELL_COMMANDS = frozenset({
    "rm", "rmdir", "unlink", "shred", "wipefs",
    "chmod", "chown", "chgrp", "chroot",
    "dd", "mkfs", "mount", "umount", "diskutil",
    "sudo", "su", "doas",
    "kill", "pkill", "killall",
    "shutdown", "reboot", "halt", "poweroff",
    # Interpreter entrypoints bypass the PYTHON: sandbox and shell policy.
    "python", "python3", "python2", "pypy", "pypy3",
    "bash", "sh", "zsh", "fish", "perl", "ruby", "node", "deno", "php",
})


# --- Safe math expression evaluator (replaces eval for CALC) ---
class _SafeMathEvaluator(ast.NodeVisitor):
    """AST-based evaluator that only allows numeric literals, basic operators,
    and a whitelist of math functions."""

    _ALLOWED_FUNCS: dict = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "atan2": math.atan2,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "log10": math.log10, "exp": math.exp,
        "ceil": math.ceil, "floor": math.floor,
        "degrees": math.degrees, "radians": math.radians,
        "factorial": math.factorial, "gcd": math.gcd,
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "int": int, "float": float,
        "len": len, "range": range, "list": list, "divmod": divmod,
    }

    _ALLOWED_CONSTANTS: dict = {
        "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
        "True": True, "False": False,
    }

    _BIN_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_, ast.BitXor: operator.xor,
        ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    }

    _UNARY_OPS = {
        ast.UAdd: operator.pos, ast.USub: operator.neg,
        ast.Invert: operator.invert,
    }

    _CMP_OPS = {
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
    }

    def evaluate(self, expr: str) -> Any:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError(f"許可されていないリテラル: {node.value!r}")

    def visit_Num(self, node: ast.Num) -> Any:  # Python 3.7 compat
        return node.n

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self._ALLOWED_CONSTANTS:
            return self._ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"許可されていない名前: {node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = self._BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"許可されていない演算子: {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op = self._UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"許可されていない単項演算子: {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = self._CMP_OPS.get(type(op_node))
            if op is None:
                raise ValueError(f"許可されていない比較演算子: {type(op_node).__name__}")
            right = self.visit(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for val in node.values:
                result = self.visit(val)
                if not result:
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            result = False
            for val in node.values:
                result = self.visit(val)
                if result:
                    return result
            return result
        raise ValueError(f"許可されていないブール演算子: {type(node.op).__name__}")

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        if self.visit(node.test):
            return self.visit(node.body)
        return self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("メソッド呼び出しは許可されていません")
        func_name = node.func.id
        if func_name not in self._ALLOWED_FUNCS:
            raise ValueError(f"許可されていない関数: {func_name}")
        func = self._ALLOWED_FUNCS[func_name]
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"許可されていない構文: {type(node).__name__}")


_safe_math_evaluator = _SafeMathEvaluator()


# --- AST-based Python safety checker (allowlist approach) ---
def _is_python_safe(code: str) -> tuple[bool, str]:
    """許可リスト方式で Python コードの安全性を検証する。

    - インポートは _ALLOWED_IMPORT_MODULES のみ許可
    - 関数呼び出しは _ALLOWED_BUILTINS + ユーザー定義関数のみ許可
    - dunder 属性アクセスは全面禁止
    - open() は読み取りモードのみ許可

    Returns (True, "") if safe, or (False, reason) if dangerous.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""

    # ユーザー定義の関数・クラス名を収集 (呼び出し許可用)
    _user_defined: set[str] = set()
    _import_aliases: dict[str, str] = {}
    _pathlike_vars: set[str] = set()

    def _is_pathlib_constructor(expr: ast.AST) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func
        if isinstance(func, ast.Name):
            return _import_aliases.get(func.id) == "pathlib"
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return _import_aliases.get(func.value.id, func.value.id) == "pathlib" and func.attr in {"Path", "PurePath"}
        return False

    def _expr_comes_from_pathlib(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in _pathlike_vars
        if _is_pathlib_constructor(expr):
            return True
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Attribute) and _expr_comes_from_pathlib(expr.func.value):
                return True
            return _expr_comes_from_pathlib(expr.func)
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name) and _import_aliases.get(expr.value.id, expr.value.id) == "pathlib":
                return expr.attr in {"Path", "PurePath"}
            return _expr_comes_from_pathlib(expr.value)
        if isinstance(expr, ast.BinOp):
            return _expr_comes_from_pathlib(expr.left) or _expr_comes_from_pathlib(expr.right)
        if isinstance(expr, ast.Subscript):
            return _expr_comes_from_pathlib(expr.value)
        if isinstance(expr, ast.IfExp):
            return _expr_comes_from_pathlib(expr.body) or _expr_comes_from_pathlib(expr.orelse)
        if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
            return any(_expr_comes_from_pathlib(elt) for elt in expr.elts)
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _user_defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            _user_defined.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            # from X import Y, Z — Y/Z をローカル名として呼び出し許可
            for alias in node.names:
                local = alias.asname or alias.name
                if local != "*":
                    _user_defined.add(local)
                    if node.module:
                        _import_aliases[local] = node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            # import X as Y / import X — 束縛名 (X または Y) を許可
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                _user_defined.add(local)
                _import_aliases[local] = alias.name.split(".")[0]

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _expr_comes_from_pathlib(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in _pathlike_vars:
                        _pathlike_vars.add(target.id)
                        changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and _expr_comes_from_pathlib(node.value)
                and isinstance(node.target, ast.Name)
                and node.target.id not in _pathlike_vars
            ):
                _pathlike_vars.add(node.target.id)
                changed = True
            elif (
                isinstance(node, ast.NamedExpr)
                and _expr_comes_from_pathlib(node.value)
                and isinstance(node.target, ast.Name)
                and node.target.id not in _pathlike_vars
            ):
                _pathlike_vars.add(node.target.id)
                changed = True

    for node in ast.walk(tree):
        # --- インポート: 許可リストのみ通す ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORT_MODULES:
                    return False, f"import '{alias.name}' は許可されていません"

        elif isinstance(node, ast.ImportFrom):
            # 相対インポート (from . import X) は全面禁止
            if node.module is None:
                return False, "相対 import は禁止されています"
            top = node.module.split(".")[0]
            if top not in _ALLOWED_IMPORT_MODULES:
                return False, f"from '{node.module}' import ... は許可されていません"
            # from X import * は禁止
            if any(alias.name == "*" for alias in node.names):
                return False, f"from {node.module} import * は禁止されています"
            # os モジュールから禁止関数の直接 import を拒否 (os.system バイパス防止)
            if top == "os":
                for alias in node.names:
                    if alias.name in _OS_DENY_FUNCS:
                        return False, f"from os import {alias.name} は禁止されています"

        # --- 危険 callable の別名化を拒否 ---
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            if isinstance(value, ast.Name) and value.id in {"open", "__import__", "eval", "exec", "compile"}:
                return False, f"危険な callable '{value.id}' の別名化は禁止されています"
            if isinstance(value, ast.Attribute):
                root = value.value
                if (
                    isinstance(root, ast.Name)
                    and _import_aliases.get(root.id, root.id) == "os"
                    and value.attr in _OS_DENY_FUNCS
                ):
                    return False, f"os.{value.attr} の別名化は禁止されています"

        # --- 関数呼び出し: 許可リスト + ユーザー定義のみ ---
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
                if name == "open":
                    # open() モード解析: 省略時 'r'、明示時は literal のみ許可
                    mode_value: Optional[str] = "r"  # default
                    mode_is_literal = True
                    if len(node.args) >= 2:
                        if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                            mode_value = node.args[1].value
                        else:
                            mode_is_literal = False
                    for kw in node.keywords:
                        if kw.arg == "mode":
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                mode_value = kw.value.value
                            else:
                                mode_is_literal = False
                    if not mode_is_literal:
                        return False, "open() の mode 引数は literal 文字列のみ許可されます"
                    # 有効モード文字のみ許可
                    if mode_value is None or not set(mode_value) <= {"r", "w", "a", "x", "b", "t", "+"}:
                        return False, f"open() の mode '{mode_value}' は無効です"
                    is_write = any(ch in mode_value for ch in ("w", "a", "x", "+"))
                    if is_write:
                        return False, "PYTHON: での open() 書き込みは禁止されています。WRITE: を使ってください"
                elif name not in _ALLOWED_BUILTINS and name not in _user_defined:
                    return False, f"関数 '{name}()' は許可されていません"
            # メソッド呼び出し (.method()) — 属性部分を検査
            elif isinstance(func, ast.Attribute):
                if func.attr.startswith("__") and func.attr.endswith("__"):
                    return False, f"メソッド '.{func.attr}()' は禁止されています (dunder)"
                if func.attr in _PATHLIB_SIDE_EFFECT_METHODS and _expr_comes_from_pathlib(func.value):
                    return False, f"pathlib の副作用メソッド '.{func.attr}()' は禁止されています"
                # os.<危険関数>() を拒否。os.path.xxx は value が Attribute なので通過。
                if (
                    isinstance(func.value, ast.Name)
                    and _import_aliases.get(func.value.id, func.value.id) == "os"
                    and func.attr in _OS_DENY_FUNCS
                ):
                    return False, f"os.{func.attr}() は禁止されています (破壊的/シェル実行系)"

        # --- dunder 属性アクセス: 全面禁止 ---
        elif isinstance(node, ast.Attribute):
            if node.attr in _DANGEROUS_ATTRS:
                return False, f"属性 '.{node.attr}' は禁止されています"
            # 未知の dunder も禁止 (安全側に倒す)
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, f"dunder 属性 '.{node.attr}' は禁止されています"

    return True, ""


class Executor:
    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.tool_registry = ToolRegistry()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self, step: str, state: AgentState) -> ExecutorResult:
        """ステップ文字列を解析して対応ツールを呼び出す。"""
        import time as _time
        _exec_start = _time.time()
        step = step.strip()

        # 世界モデルを初期化 (なければ)
        if state.world_model is None:
            state.world_model = WorldModel()

        # 実行前に世界モデルで結果を予測 (ログ用)
        prediction = state.world_model.predict_outcome(step)
        if prediction:
            state.working_memory["last_world_prediction"] = prediction

        # Gen 10: 実行結果にコスト記録を付加するヘルパー
        def _record_cost(result: ExecutorResult, tool_type: str) -> ExecutorResult:
            elapsed = _time.time() - _exec_start
            output_size = len(result.get("stdout", ""))
            state.world_model.record_resource_cost(
                tool_type=tool_type, execution_time=elapsed,
                output_size=output_size, success=result.get("ok", False),
            )
            result["_cost"] = {"time": round(elapsed, 3), "tool": tool_type}
            return result

        if step.upper().startswith("PLAN:"):
            raw = step[5:].strip()
            sub_steps = [s.strip() for s in raw.split("||") if s.strip()]
            # 最後のステップが空の ANSWER: プレースホルダーなら DONE: に置き換えて
            # リアクティブなまとめ生成に委ねる
            if sub_steps and sub_steps[-1].upper().startswith("ANSWER:"):
                last = sub_steps[-1][7:].strip()
                if len(last) < 20:  # 短い = プレースホルダー
                    sub_steps[-1] = "DONE: まとめて結論を出す"
            state.current_plan = sub_steps + state.current_plan
            state.working_memory["decomposed_plan"] = sub_steps
            summary = f"{len(sub_steps)} ステップに分解しました: " + " → ".join(
                s[:30] for s in sub_steps
            )
            return {"ok": True, "stdout": summary, "stderr": "", "returncode": 0, "command": None}

        if step.upper().startswith("ANSWER:"):
            answer = step[7:].strip()
            state.working_memory["completion_summary"] = answer
            return {"ok": True, "stdout": answer, "stderr": "", "returncode": 0, "command": None}

        if step.upper().startswith("CALC:"):
            expr = step[5:].strip()
            return _record_cost(self._run_calc(expr, state), "CALC")

        if step.upper().startswith("FETCH:"):
            url = step[6:].strip().splitlines()[0].strip()
            return _record_cost(self._run_fetch(url, state), "FETCH")

        if step.upper().startswith("SEARCH:"):
            query = step[7:].strip()
            return _record_cost(self._run_search(query, state), "SEARCH")

        if step.upper().startswith("CMD:"):
            cmd = step[4:].strip().splitlines()[0].strip()
            return _record_cost(self._run_shell(cmd, state), "CMD")

        if step.upper().startswith("READ:"):
            filepath = step[5:].strip().splitlines()[0].strip()
            return _record_cost(self._read_file(filepath, state), "READ")

        if step.upper().startswith("WRITE:"):
            return _record_cost(self._write_file(step[6:], state), "WRITE")

        if step.upper().startswith("PYTHON:"):
            code = textwrap.dedent(step[7:]).strip()
            return _record_cost(self._run_python(code, state), "PYTHON")

        if step.upper().startswith("DONE:"):
            summary = step[5:].strip()
            state.working_memory["completion_summary"] = summary
            state.is_done = True  # DONE宣言でエージェントを即座に終了させる
            return {"ok": True, "stdout": summary, "stderr": "", "returncode": 0, "command": None}

        if step.upper().startswith("SCHEDULE:"):
            return self._schedule_goal(step[9:], state)

        if step.upper().startswith("SCHEDULE_AT:"):
            return self._schedule_at(step[12:], state)

        # カスタムツール (ToolRegistry) を確認
        custom_result = self.tool_registry.dispatch(step)
        if custom_result is not None:
            if custom_result.get("ok"):
                remember_successful_command(state, step[:60])
            return custom_result

        # レガシー静的ステップ（後方互換）
        return self._legacy_execute(step, state)

    # ------------------------------------------------------------------
    # CMD: シェル実行
    # ------------------------------------------------------------------

    # パイプ後に許可する安全な読み取り専用コマンド
    # awk / sed は除外 — 任意コード実行が可能なため
    _SAFE_PIPE_CMDS = frozenset([
        "grep", "head", "tail", "wc", "sort", "uniq", "tr", "cut", "tee", "cat",
    ])

    def _run_shell(self, cmd: str, state: AgentState) -> ExecutorResult:
        # ~ をホームディレクトリに展開 (shlex.split はチルダを展開しないため)
        import os as _os
        _home = _os.path.expanduser("~")
        cmd = _RE_TILDE.sub(_home, cmd)

        # セキュリティ: 危険なコマンドチェインを禁止 (VULN-001)
        # `;` `&&` `||` バックティック `$()` は完全禁止
        # `|` はパイプとして許可 (後続コマンドが安全リストにある場合)
        _dangerous = [";", "&&", "||", "`"]
        for pat in _dangerous:
            if pat in cmd:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": f"セキュリティ制限: シェル連結演算子 '{pat}' は禁止されています",
                    "returncode": 1,
                    "command": None
                }
        # $() コマンド置換を禁止
        if _RE_CMD_SUBSTITUTION.search(cmd):
            return {
                "ok": False,
                "stdout": "",
                "stderr": "セキュリティ制限: $(...) コマンド置換は禁止されています",
                "returncode": 1,
                "command": None
            }
        try:
            _first_tokens = shlex.split(cmd)
        except ValueError as e:
            return {"ok": False, "stdout": "", "stderr": f"コマンド解析エラー: {e}", "returncode": -1, "command": None}
        if not _first_tokens:
            return {"ok": False, "stdout": "", "stderr": "コマンドが空です", "returncode": 1, "command": None}
        _cmd_name = os.path.basename(_first_tokens[0])
        if _cmd_name in _BLOCKED_SHELL_COMMANDS:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"セキュリティ制限: コマンド '{_cmd_name}' は禁止されています",
                "returncode": 1,
                "command": None
            }
        # パイプを含む場合: shell=True を使わず、subprocess でパイプチェーンを構築
        _has_pipe = "|" in cmd
        if _has_pipe:
            _pipe_segments = [s.strip() for s in cmd.split("|")]
            # 先頭以外のコマンドが安全リストにあるか確認
            for seg in _pipe_segments[1:]:
                if not seg:
                    continue
                try:
                    tokens = shlex.split(seg)
                except ValueError:
                    tokens = seg.split()
                if not tokens:
                    continue
                # コマンド名のみ (パスは basename で判定)
                import os as _os_inner
                cmd_name = _os_inner.path.basename(tokens[0])
                if cmd_name not in self._SAFE_PIPE_CMDS:
                    return {
                        "ok": False,
                        "stdout": "",
                        "stderr": f"セキュリティ制限: パイプ後続コマンド '{cmd_name}' は許可リストにありません",
                        "returncode": 1,
                        "command": None
                    }

        try:
            if _has_pipe:
                # shell=True を使わないパイプ実装
                _pipe_segments = [s.strip() for s in cmd.split("|")]
                prev_proc = None
                procs = []
                for i, seg in enumerate(_pipe_segments):
                    try:
                        args = shlex.split(seg)
                    except ValueError:
                        args = seg.split()
                    stdin = prev_proc.stdout if prev_proc else None
                    p = subprocess.Popen(
                        args,
                        stdin=stdin,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=str(self.repo_root),
                    )
                    if prev_proc and prev_proc.stdout:
                        prev_proc.stdout.close()
                    procs.append(p)
                    prev_proc = p
                # 最終プロセスの出力を取得
                last = procs[-1]
                try:
                    stdout_data, stderr_data = last.communicate(timeout=EXECUTOR_SHELL_TIMEOUT)
                except subprocess.TimeoutExpired:
                    for p in procs:
                        p.kill()
                    last.communicate()
                    return {"ok": False, "stdout": "", "stderr": "タイムアウト", "returncode": -1, "command": None}
                # 先行プロセスを待機
                for p in procs[:-1]:
                    p.wait()
                proc_returncode = last.returncode
                proc_stdout = stdout_data or ""
                proc_stderr = stderr_data or ""
                # proc 互換オブジェクトを作成
                class _PipeResult:
                    pass
                proc = _PipeResult()
                proc.returncode = proc_returncode
                proc.stdout = proc_stdout
                proc.stderr = proc_stderr
            else:
                proc = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                    cwd=str(self.repo_root),
                    timeout=EXECUTOR_SHELL_TIMEOUT,
                )
        except ValueError as e:
            return {"ok": False, "stdout": "", "stderr": f"コマンド解析エラー: {e}", "returncode": -1, "command": None}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": f"実行エラー: {e}", "returncode": -1, "command": None}

        stdout = (proc.stdout or "")[:EXECUTOR_MAX_OUTPUT]
        stderr = (proc.stderr or "")[:EXECUTOR_MAX_OUTPUT]

        if proc.returncode == 0:
            remember_successful_command(state, cmd)
            # 環境情報を抽出
            lines = stdout.splitlines()
            cwd = None
            pyver = None
            for line in lines[:5]:
                if line.startswith("/"):
                    cwd = line.strip()
                if line.lower().startswith("python "):
                    pyver = line.strip()
            set_environment_info(state, cwd=cwd, python_version=pyver, python_executable=sys.executable)

            # 構造情報を保存
            if "find ." in cmd and "sort" in cmd:
                state.working_memory["project_structure_text"] = stdout

            # 世界モデルを更新
            if state.world_model is not None:
                state.world_model.update_from_cmd(cmd, stdout)

        return {"ok": proc.returncode == 0, "stdout": stdout, "stderr": stderr, "returncode": proc.returncode, "command": cmd}

    # ------------------------------------------------------------------
    # CALC: 数式計算 (AST-based safe evaluator, no eval())
    # ------------------------------------------------------------------

    def _run_calc(self, expr: str, state: AgentState) -> ExecutorResult:
        # First try ast.literal_eval for simple numeric expressions
        try:
            result = ast.literal_eval(expr)
            if isinstance(result, (int, float, complex, bool, list, tuple)):
                text = f"{expr} = {result}"
                remember_successful_command(state, f"CALC: {expr[:60]}")
                return {"ok": True, "stdout": text, "stderr": "", "returncode": 0, "command": f"CALC: {expr}"}
        except (ValueError, SyntaxError):
            pass

        # Fall back to safe AST-based math evaluator
        try:
            result = _safe_math_evaluator.evaluate(expr)
            text = f"{expr} = {result}"
            remember_successful_command(state, f"CALC: {expr[:60]}")
            return {"ok": True, "stdout": text, "stderr": "", "returncode": 0, "command": f"CALC: {expr}"}
        except Exception as exc:
            msg = f"計算エラー: {exc}"
            return {"ok": False, "stdout": "", "stderr": msg, "returncode": 1, "command": f"CALC: {expr}"}

    # ------------------------------------------------------------------
    # SEARCH: ウェブ検索
    # ------------------------------------------------------------------

    def _run_search(self, query: str, state: AgentState) -> ExecutorResult:
        from .web_search import format_results, search

        results = search(query, max_results=5)
        text = format_results(results)
        ok = bool(results) and results[0].get("title") != "検索エラー"

        if ok:
            remember_successful_command(state, f"SEARCH: {query[:60]}")
            state.working_memory["last_search_query"] = query
            state.working_memory["last_search_results"] = results

        return {
            "ok": ok,
            "stdout": text,
            "stderr": "" if ok else text,
            "returncode": 0 if ok else 1,
            "command": f"SEARCH: {query}",
        }

    # ------------------------------------------------------------------
    # FETCH: URL コンテンツ取得
    # ------------------------------------------------------------------

    def _run_fetch(self, url: str, state: AgentState) -> ExecutorResult:
        from .web_search import fetch_url

        result = fetch_url(url)
        if result.get("type") == "error":
            err = result.get("error", "不明なエラー")
            return {"ok": False, "stdout": "", "stderr": f"FETCH エラー ({url}): {err}", "returncode": 1, "command": f"FETCH: {url}"}

        content = result.get("content", "")
        remember_successful_command(state, f"FETCH: {url[:60]}")
        state.working_memory["last_fetch_url"] = url
        state.working_memory["last_fetch_content"] = content[:500]
        return {
            "ok": True,
            "stdout": content,
            "stderr": "",
            "returncode": 0,
            "command": f"FETCH: {url}",
        }

    # ------------------------------------------------------------------
    # READ: ファイル読み込み
    # ------------------------------------------------------------------

    def _read_file(self, filepath: str, state: AgentState) -> ExecutorResult:
        path = (self.repo_root / filepath).resolve()
        # セキュリティ: repo_root の外は読まない (symlink を追跡した後に検証)
        resolved_root = self.repo_root.resolve()
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError:
            return {"ok": False, "stdout": "", "stderr": f"アクセス拒否: READ: はリポジトリ外のパス '{filepath}' を読めません", "returncode": 1, "command": None}

        if not path.exists():
            return {"ok": False, "stdout": "", "stderr": f"ファイルが見つかりません: {filepath}", "returncode": 1, "command": f"READ: {filepath}"}

        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:EXECUTOR_MAX_OUTPUT]
            remember_successful_command(state, f"READ: {filepath}")
            return {"ok": True, "stdout": content, "stderr": "", "returncode": 0, "command": f"READ: {filepath}"}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "returncode": 1, "command": f"READ: {filepath}"}

    # ------------------------------------------------------------------
    # WRITE: ファイル書き込み
    # ------------------------------------------------------------------

    def _write_file(self, spec: str, state: AgentState) -> ExecutorResult:
        """フォーマット: <filepath>\n<content>\n[EOF]"""
        lines = spec.strip().splitlines()
        if not lines:
            return {"ok": False, "stdout": "", "stderr": "WRITE: ファイルパスが指定されていません", "returncode": 1, "command": None}

        filepath = lines[0].strip()
        # EOF マーカーを除去
        content_lines = lines[1:]
        if content_lines and content_lines[-1].strip() == "EOF":
            content_lines = content_lines[:-1]
        content = "\n".join(content_lines)

        # サイズ制限チェック
        if len(content.encode("utf-8")) > EXECUTOR_MAX_WRITE_SIZE:
            return {
                "ok": False, "stdout": "",
                "stderr": f"セキュリティ制限: 書き込みサイズが上限 ({EXECUTOR_MAX_WRITE_SIZE} bytes) を超えています",
                "returncode": 1, "command": None,
            }

        # パス解決: ~/... は絶対パスに展開、それ以外は repo_root 基準
        home = Path.home()
        expanded = filepath.replace("~", str(home), 1) if filepath.startswith("~") else filepath
        path = Path(expanded).resolve() if Path(expanded).is_absolute() else (self.repo_root / expanded).resolve()

        # セキュリティ: symlink を追跡した後の正規パスで検証
        resolved_path = path.resolve()
        resolved_root = self.repo_root.resolve()
        if _has_protected_path_component(resolved_path):
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"アクセス拒否: WRITE: パス '{filepath}' は保護ゾーンにあります",
                "returncode": 1,
                "command": None,
            }

        allowed = _path_under(resolved_path, resolved_root)
        if not allowed:
            allowed = any(_path_under(resolved_path, allow_dir) for allow_dir in _iter_write_allow_dirs())
        if not allowed:
            return {
                "ok": False,
                "stdout": "",
                "stderr": (
                    f"アクセス拒否: WRITE: はリポジトリ外のパス '{filepath}' に書けません。"
                    "外部出力先は HERMES_WRITE_ALLOW_DIRS で明示してください。"
                ),
                "returncode": 1,
                "command": None,
            }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # アトミック書込: 同一ディレクトリの .tmp に書いてから os.replace で置換
            # (POSIX/Windows いずれもアトミック保証。途中クラッシュでもターゲットは破損しない)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                tmp_path.write_text(content, encoding="utf-8")
                os.replace(tmp_path, path)
            finally:
                # 失敗時の残留 tmp を掃除
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
            remember_successful_command(state, f"WRITE: {filepath}")
            return {"ok": True, "stdout": f"{filepath} に {len(content)} 文字を書き込みました", "stderr": "", "returncode": 0, "command": f"WRITE: {filepath}"}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "returncode": 1, "command": f"WRITE: {filepath}"}

    # ------------------------------------------------------------------
    # PYTHON: Python コード実行
    # ------------------------------------------------------------------

    def _run_python(self, code: str, state: AgentState) -> ExecutorResult:
        # Primary gate: AST-based analysis
        is_safe, reason = _is_python_safe(code)
        if not is_safe:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"セキュリティ制限: {reason}",
                "returncode": 1,
                "command": "PYTHON:"
            }

        # Note: AST-based allowlist (_is_python_safe) が主ゲート。
        # 旧ブロックリスト方式 (_PY_DANGER_PATTERNS) は許可リスト方式に置換済みのため削除。

        env = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"):
            value = os.getenv(key)
            if value:
                env[key] = value

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                env=env,
                timeout=EXECUTOR_PYTHON_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"Python タイムアウト ({EXECUTOR_PYTHON_TIMEOUT}秒)", "returncode": -1, "command": "PYTHON:"}

        stdout = (proc.stdout or "")[:EXECUTOR_MAX_OUTPUT]
        stderr = (proc.stderr or "")[:EXECUTOR_MAX_OUTPUT]

        if proc.returncode == 0:
            remember_successful_command(state, f"PYTHON: {code[:60]}")

        return {"ok": proc.returncode == 0, "stdout": stdout, "stderr": stderr, "returncode": proc.returncode, "command": "PYTHON:"}

    # ------------------------------------------------------------------
    # SCHEDULE: GoalQueueにゴールを追加
    # ------------------------------------------------------------------

    def _schedule_goal(self, args: str, state: AgentState) -> ExecutorResult:
        """SCHEDULE: <goal> でGoalQueueに目標を追加する。デーモンが自動処理する。

        書式:
          SCHEDULE: <ゴールテキスト>
          SCHEDULE: priority=0.8 <ゴールテキスト>
        """
        args = args.strip()
        priority = 0.6
        if args.startswith("priority="):
            parts = args.split(None, 1)
            try:
                priority = float(parts[0].split("=")[1])
                args = parts[1] if len(parts) > 1 else ""
            except (ValueError, IndexError):
                pass

        if not args:
            return {"ok": False, "stdout": "", "stderr": "SCHEDULE: ゴールテキストが空です", "returncode": 1}

        # LTM経由でGoalQueueに永続保存
        try:
            from .long_term_memory import LongTermMemory
            from .meta_cognition import GoalQueue, QueuedGoal
            import time as _time
            ltm = LongTermMemory()
            q = GoalQueue()
            q.load_from_ltm(ltm)
            q.add(QueuedGoal(
                goal=args,
                priority_score=priority,
                source="scheduled",
                rationale=f"SCHEDULE: ツールで追加 (セッション {state.session_id or '?'})",
                domain=state.domain or "general",
            ))
            q.save_to_ltm(ltm)
            msg = f"ゴールをキューに追加しました: {args[:60]} (priority={priority:.1f}, queue={q.size()}件)"
            state.working_memory.setdefault("scheduled_goals", []).append(args)
            return {"ok": True, "stdout": msg, "stderr": "", "returncode": 0, "command": f"SCHEDULE: {args}"}
        except Exception as exc:
            return {"ok": False, "stdout": "", "stderr": f"スケジュール登録エラー: {exc}", "returncode": 1}

    # ------------------------------------------------------------------
    # SCHEDULE_AT: 時刻指定スケジュール
    # ------------------------------------------------------------------

    def _schedule_at(self, args: str, state: AgentState) -> ExecutorResult:
        """SCHEDULE_AT: <trigger> <goal> で時刻指定スケジュールを登録する。

        書式:
          SCHEDULE_AT: daily:09:00 毎朝ニュースを要約する
          SCHEDULE_AT: every:30m  システム状態を確認する
          SCHEDULE_AT: 2026-03-31T09:00 レポートを作成する
          SCHEDULE_AT: weekly:mon:09:00 週次レポートを作成する
          SCHEDULE_AT: priority=0.8 daily:09:00 重要タスク
        """
        args = args.strip()
        priority = 0.6

        # priority= オプション
        if args.lower().startswith("priority="):
            parts = args.split(None, 1)
            try:
                priority = float(parts[0].split("=")[1])
                args = parts[1] if len(parts) > 1 else ""
            except (ValueError, IndexError):
                pass

        # trigger と goal を分離 (最初のトークンがトリガー)
        parts = args.split(None, 1)
        if len(parts) < 2:
            return {
                "ok": False, "stdout": "", "returncode": 1,
                "stderr": "SCHEDULE_AT: <trigger> <goal> の形式で指定してください。\n"
                          "例: SCHEDULE_AT: daily:09:00 毎朝ニュースを要約する",
            }

        trigger_raw, goal_text = parts[0].strip(), parts[1].strip()

        from .scheduler import JobScheduler, parse_trigger_spec
        trigger = parse_trigger_spec(trigger_raw)
        if trigger is None:
            return {
                "ok": False, "stdout": "", "returncode": 1,
                "stderr": (
                    f"トリガー形式が不明です: {trigger_raw}\n"
                    "サポート形式: once:<ISO8601> | every:<N>m/h | daily:<HH:MM> | weekly:<day>:<HH:MM>"
                ),
            }

        try:
            scheduler = JobScheduler()
            job = scheduler.add_job(
                goal=goal_text,
                trigger=trigger,
                domain=state.domain or "general",
                priority=priority,
            )
            next_str = scheduler.format_next_run(job)
            msg = (
                f"スケジュール登録完了: [{job.id}] {goal_text[:60]}\n"
                f"  トリガー: {trigger} | 次回実行: {next_str}"
            )
            state.working_memory.setdefault("scheduled_jobs", []).append(job.id)
            state.working_memory["completion_summary"] = msg
            state.is_done = True  # 登録完了 = タスク完了。別途実行は不要
            return {"ok": True, "stdout": msg, "stderr": "", "returncode": 0,
                    "command": f"SCHEDULE_AT: {trigger} {goal_text}"}
        except Exception as exc:
            return {"ok": False, "stdout": "", "stderr": f"スケジュール登録エラー: {exc}", "returncode": 1}

    # レガシー静的ステップ（後方互換）
    # ------------------------------------------------------------------

    def _legacy_execute(self, step: str, state: AgentState) -> ExecutorResult:
        python_bin = sys.executable
        legacy_map = {
            "Inspect project structure": f'pwd && {python_bin} --version && ls -la && find . -maxdepth 2 -not -path "*/__pycache__/*" | sort | head -100',
            "Read README": 'if [ -f README.md ]; then cat README.md; else echo "README.md not found"; fi',
            "Read requirements": 'if [ -f requirements.txt ]; then cat requirements.txt; else echo "requirements.txt not found"; fi',
            "Read pyproject config": 'if [ -f pyproject.toml ]; then cat pyproject.toml; else echo "not found"; fi',
            "Read main entry point": 'for f in main.py run_agent.py cli.py; do [ -f "$f" ] && echo "=== $f ===" && head -80 "$f"; done',
            "Inspect CLI entry point": 'if [ -f cli.py ]; then cat cli.py; fi',
            "Inspect tests": 'if [ -d tests ]; then find tests -maxdepth 2 | sort; else echo "tests/ not found"; fi',
            "Inspect state store": 'if [ -f hermes_agi_gen/state_store.py ]; then cat hermes_agi_gen/state_store.py; fi',
            "Check installed commands and PATH": 'echo "$PATH" && which python3 || true && python3 --version || true',
            "Check Python environment and pip packages": f'{python_bin} --version && {python_bin} -m pip list | head -40',
            "Summarize findings and propose next upgrade": None,
        }

        if step in legacy_map:
            cmd = legacy_map[step]
            if cmd is None:
                return {"ok": True, "stdout": "（論理ステップ: シェル実行なし）", "stderr": "", "returncode": 0, "command": None}
            return self._run_shell(cmd, state)

        return {"ok": False, "stdout": "", "stderr": f"未知のステップ: {step}", "returncode": 1, "command": None}
