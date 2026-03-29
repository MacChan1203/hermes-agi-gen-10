"""汎用ツールディスパッチャー。

対応ツール:
  CMD: <bash>      シェルコマンド実行
  READ: <path>     ファイル読み込み
  WRITE: <path>    ファイル書き込み (次行から EOF まで)
  PYTHON: <code>   Python コード実行
  DONE: <summary>  タスク完了宣言
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict

from .memory import remember_successful_command, set_environment_info
from .agent_state import AgentState
from .world_model import WorldModel
from .tool_registry import ToolRegistry

_MAX_OUTPUT = 8000   # stdout/stderr の最大文字数
_PY_TIMEOUT = 30     # Python 実行タイムアウト (秒)
_SH_TIMEOUT = 30     # シェル実行タイムアウト (秒)


class Executor:
    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.tool_registry = ToolRegistry()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self, step: str, state: AgentState) -> Dict[str, Any]:
        """ステップ文字列を解析して対応ツールを呼び出す。"""
        step = step.strip()

        # 世界モデルを初期化 (なければ)
        if state.world_model is None:
            state.world_model = WorldModel()

        # 実行前に世界モデルで結果を予測 (ログ用)
        prediction = state.world_model.predict_outcome(step)
        if prediction:
            state.working_memory["last_world_prediction"] = prediction

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
            return self._run_calc(expr, state)

        if step.upper().startswith("SEARCH:"):
            query = step[7:].strip()
            return self._run_search(query, state)

        if step.upper().startswith("CMD:"):
            cmd = step[4:].strip().splitlines()[0].strip()
            return self._run_shell(cmd, state)

        if step.upper().startswith("READ:"):
            filepath = step[5:].strip().splitlines()[0].strip()
            return self._read_file(filepath, state)

        if step.upper().startswith("WRITE:"):
            return self._write_file(step[6:], state)

        if step.upper().startswith("PYTHON:"):
            code = textwrap.dedent(step[7:]).strip()
            return self._run_python(code, state)

        if step.upper().startswith("DONE:"):
            summary = step[5:].strip()
            state.working_memory["completion_summary"] = summary
            return {"ok": True, "stdout": summary, "stderr": "", "returncode": 0, "command": None}

        if step.upper().startswith("SCHEDULE:"):
            return self._schedule_goal(step[9:], state)

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

    def _run_shell(self, cmd: str, state: AgentState) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=_SH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"タイムアウト ({_SH_TIMEOUT}秒)", "returncode": -1, "command": cmd}

        stdout = (proc.stdout or "")[:_MAX_OUTPUT]
        stderr = (proc.stderr or "")[:_MAX_OUTPUT]

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
    # CALC: 数式計算
    # ------------------------------------------------------------------

    _CALC_NS: dict = {}  # クラス変数として遅延初期化

    @staticmethod
    def _build_calc_ns() -> dict:
        import math
        return {
            "__builtins__": {},
            # 組み込み
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "divmod": divmod, "len": len,
            "range": range, "list": list, "int": int, "float": float,
            # math
            "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
            "log": math.log, "log2": math.log2, "log10": math.log10,
            "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "atan2": math.atan2,
            "degrees": math.degrees, "radians": math.radians,
            "factorial": math.factorial, "gcd": math.gcd,
            "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
        }

    def _run_calc(self, expr: str, state: AgentState) -> Dict[str, Any]:
        if not Executor._CALC_NS:
            Executor._CALC_NS = self._build_calc_ns()
        try:
            result = eval(expr, Executor._CALC_NS)  # noqa: S307
            text = f"{expr} = {result}"
            remember_successful_command(state, f"CALC: {expr[:60]}")
            return {"ok": True, "stdout": text, "stderr": "", "returncode": 0, "command": f"CALC: {expr}"}
        except Exception as exc:
            msg = f"計算エラー: {exc}"
            return {"ok": False, "stdout": "", "stderr": msg, "returncode": 1, "command": f"CALC: {expr}"}

    # ------------------------------------------------------------------
    # SEARCH: ウェブ検索
    # ------------------------------------------------------------------

    def _run_search(self, query: str, state: AgentState) -> Dict[str, Any]:
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
    # READ: ファイル読み込み
    # ------------------------------------------------------------------

    def _read_file(self, filepath: str, state: AgentState) -> Dict[str, Any]:
        path = (self.repo_root / filepath).resolve()
        # セキュリティ: repo_root の外は読まない
        try:
            path.relative_to(self.repo_root)
        except ValueError:
            return {"ok": False, "stdout": "", "stderr": f"アクセス拒否: {filepath} はリポジトリ外です", "returncode": 1, "command": f"READ: {filepath}"}

        if not path.exists():
            return {"ok": False, "stdout": "", "stderr": f"ファイルが見つかりません: {filepath}", "returncode": 1, "command": f"READ: {filepath}"}

        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:_MAX_OUTPUT]
            remember_successful_command(state, f"READ: {filepath}")
            return {"ok": True, "stdout": content, "stderr": "", "returncode": 0, "command": f"READ: {filepath}"}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "returncode": 1, "command": f"READ: {filepath}"}

    # ------------------------------------------------------------------
    # WRITE: ファイル書き込み
    # ------------------------------------------------------------------

    def _write_file(self, spec: str, state: AgentState) -> Dict[str, Any]:
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

        path = (self.repo_root / filepath).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError:
            return {"ok": False, "stdout": "", "stderr": f"アクセス拒否: {filepath} はリポジトリ外です", "returncode": 1, "command": f"WRITE: {filepath}"}

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            remember_successful_command(state, f"WRITE: {filepath}")
            return {"ok": True, "stdout": f"{filepath} に {len(content)} 文字を書き込みました", "stderr": "", "returncode": 0, "command": f"WRITE: {filepath}"}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "returncode": 1, "command": f"WRITE: {filepath}"}

    # ------------------------------------------------------------------
    # PYTHON: Python コード実行
    # ------------------------------------------------------------------

    def _run_python(self, code: str, state: AgentState) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=_PY_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"Python タイムアウト ({_PY_TIMEOUT}秒)", "returncode": -1, "command": "PYTHON:"}

        stdout = (proc.stdout or "")[:_MAX_OUTPUT]
        stderr = (proc.stderr or "")[:_MAX_OUTPUT]

        if proc.returncode == 0:
            remember_successful_command(state, f"PYTHON: {code[:60]}")

        return {"ok": proc.returncode == 0, "stdout": stdout, "stderr": stderr, "returncode": proc.returncode, "command": "PYTHON:"}

    # ------------------------------------------------------------------
    # SCHEDULE: GoalQueueにゴールを追加
    # ------------------------------------------------------------------

    def _schedule_goal(self, args: str, state: AgentState) -> Dict[str, Any]:
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

    # レガシー静的ステップ（後方互換）
    # ------------------------------------------------------------------

    def _legacy_execute(self, step: str, state: AgentState) -> Dict[str, Any]:
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
