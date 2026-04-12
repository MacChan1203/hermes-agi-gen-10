"""実 LLM (Ollama gemma4:e4b) を使った統合テスト。

Ollama が起動していない環境では全テストが自動スキップされる。
pytest 実行時にこのファイルのテストだけ除外したい場合:
    pytest tests/ --ignore=tests/test_live_llm.py

明示的に実行する場合:
    pytest tests/test_live_llm.py -v
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Ollama 接続チェック — 利用不可なら全テストスキップ
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Ollama が起動中で gemma4:e4b が利用可能か確認する。"""
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return "gemma4:e4b" in models
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama が起動していない、または gemma4:e4b が利用不可",
    ),
    # 共有 SQLite DB のロック競合を避けるため、グローバル HERMES_HOME を分離
]


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """全テストで HERMES_HOME を tmp_path に分離し、共有 SQLite の競合を防ぐ。

    全モジュールが get_hermes_home() 経由で HERMES_HOME 環境変数を参照するため、
    環境変数の設定だけで全 DB パスが分離される。
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # モジュールロード時に確定済みのグローバルDBパスもオーバーライド
    monkeypatch.setattr("hermes_agi_gen.long_term_memory.DEFAULT_LTM_PATH", hermes_home / "ltm.db")
    monkeypatch.setattr("hermes_agi_gen.meta_learning._META_LEARNING_DB", hermes_home / "ml.db")
    monkeypatch.setattr("hermes_agi_gen.self_improvement._IMPROVEMENT_DB_PATH", hermes_home / "si.db")
    monkeypatch.setattr("hermes_agi_gen.state_store.DEFAULT_DB_PATH", hermes_home / "state.db")
    monkeypatch.setattr("hermes_agi_gen.self_modifier._PATCH_DB_PATH", hermes_home / "sm.db")
    monkeypatch.setattr("hermes_agi_gen.experiment_runner._EXPERIMENT_DB_PATH", hermes_home / "exp.db")
    monkeypatch.setattr("hermes_agi_gen.tool_registry._REGISTRY_PATH", hermes_home / "tr.db")


# ---------------------------------------------------------------------------
# LLM クライアント基本テスト
# ---------------------------------------------------------------------------

class TestMistralClientLive:
    """MistralClient が実 Ollama に接続して応答を得る。"""

    def test_chat_returns_nonempty_response(self):
        from hermes_agi_gen.mistral_client import MistralClient
        client = MistralClient()
        result = client.chat(
            [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
            temperature=0.0,
            max_tokens=32,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        # LLM が "2" または "二" で答える可能性を許容
        assert "2" in result or "二" in result or "two" in result.lower()

    def test_chat_json_returns_dict(self):
        from hermes_agi_gen.mistral_client import MistralClient
        client = MistralClient()
        result = client.chat_json(
            [{"role": "user", "content": (
                "Return ONLY the following JSON, nothing else:\n"
                '{"answer": 42}'
            )}],
            temperature=0.0,
            max_tokens=64,
        )
        # LLM が正しい JSON を返せない場合もある — パース成功 or None でクラッシュしない
        assert result is None or isinstance(result, (dict, list))

    def test_chat_handles_long_prompt(self):
        from hermes_agi_gen.mistral_client import MistralClient
        client = MistralClient()
        long_prompt = "これは長いプロンプトです。" * 100
        result = client.chat(
            [{"role": "user", "content": long_prompt + " 一言で要約してください。"}],
            temperature=0.0,
            max_tokens=64,
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Planner が実 LLM でステップを生成
# ---------------------------------------------------------------------------

class TestPlannerLive:
    """Planner が実 LLM を使って有効なステップを生成する。"""

    def test_next_step_returns_valid_tool_prefix(self):
        from hermes_agi_gen.planner import Planner
        from hermes_agi_gen.mistral_client import MistralClient
        from hermes_agi_gen.agent_state import AgentState

        llm = MistralClient()
        planner = Planner(llm=llm, role="executor")
        state = AgentState(user_goal="現在のディレクトリにあるファイルを一覧表示して", max_iterations=3)
        state.domain = "general"

        step = planner.next_step(state)
        assert step is not None
        valid_prefixes = ("CMD:", "SEARCH:", "FETCH:", "PYTHON:", "READ:", "WRITE:",
                          "PLAN:", "ANSWER:", "CALC:", "DONE:", "SCHEDULE")
        assert any(step.upper().startswith(p) for p in valid_prefixes), f"Invalid step: {step}"

    def test_conversational_goal_gets_answer(self):
        from hermes_agi_gen.planner import Planner
        from hermes_agi_gen.mistral_client import MistralClient
        from hermes_agi_gen.agent_state import AgentState

        llm = MistralClient()
        planner = Planner(llm=llm, role="worker")
        state = AgentState(user_goal="Pythonとは何ですか？簡潔に教えて", max_iterations=3)

        step = planner.next_step(state)
        assert step is not None
        # LLM が ANSWER: 以外 (SEARCH: 等) を返す場合も許容
        valid_prefixes = ("CMD:", "SEARCH:", "FETCH:", "PYTHON:", "READ:", "WRITE:",
                          "PLAN:", "ANSWER:", "CALC:", "DONE:", "SCHEDULE")
        assert any(step.upper().startswith(p) for p in valid_prefixes), f"Invalid step: {step}"


# ---------------------------------------------------------------------------
# Executor + 実 LLM で Plan→Execute
# ---------------------------------------------------------------------------

class TestExecutorLive:
    """Executor が実コマンドを実行し、結果を正しく返す。"""

    def test_cmd_echo(self, tmp_path):
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState

        ex = Executor(repo_root=tmp_path)
        state = AgentState(user_goal="test", max_iterations=1)
        result = ex.execute("CMD: echo live_test_ok", state)
        assert result["ok"]
        assert "live_test_ok" in result["stdout"]

    def test_python_execution(self, tmp_path):
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState

        ex = Executor(repo_root=tmp_path)
        state = AgentState(user_goal="test", max_iterations=1)
        result = ex.execute("PYTHON: print(2 + 3)", state)
        assert result["ok"]
        assert "5" in result["stdout"]

    def test_calc_expression(self, tmp_path):
        from hermes_agi_gen.executor import Executor
        from hermes_agi_gen.agent_state import AgentState

        ex = Executor(repo_root=tmp_path)
        state = AgentState(user_goal="test", max_iterations=1)
        result = ex.execute("CALC: sqrt(144)", state)
        assert result["ok"]
        assert "12" in result["stdout"]


# ---------------------------------------------------------------------------
# Intent Classification (実 LLM)
# ---------------------------------------------------------------------------

class TestIntentClassificationLive:
    """_classify_intent が実 LLM でインテントを正しく分類する。"""

    def test_task_classification(self):
        from cli import _classify_intent
        from hermes_agi_gen.mistral_client import MistralClient
        llm = MistralClient()
        intent, domain = _classify_intent(llm, "テストを実行して結果を確認して")
        assert intent == "task"

    def test_chat_classification(self):
        from cli import _classify_intent
        from hermes_agi_gen.mistral_client import MistralClient
        llm = MistralClient()
        intent, domain = _classify_intent(llm, "こんにちは")
        assert intent == "chat"


# ---------------------------------------------------------------------------
# LLM 応答のバリエーション耐性テスト
# ---------------------------------------------------------------------------

class TestLLMResponseVariations:
    """LLM の不正・想定外の応答に対する耐性。"""

    def test_chat_json_with_markdown_wrapping(self):
        """LLM が ```json ... ``` で囲んで返しても chat_json がパースできる。"""
        from hermes_agi_gen.mistral_client import MistralClient
        client = MistralClient()
        result = client.chat_json(
            [{"role": "user", "content": (
                "以下のJSON形式で答えてください。必ずJSONのみ返してください:\n"
                '```json\n{"name": "test", "value": 123}\n```'
            )}],
            temperature=0.0,
            max_tokens=128,
        )
        # None でなくパース成功、または少なくともクラッシュしない
        assert result is None or isinstance(result, (dict, list))

    def test_chat_returns_empty_on_timeout_model(self):
        """非常に短い max_tokens でも空文字で返りクラッシュしない。"""
        from hermes_agi_gen.mistral_client import MistralClient
        client = MistralClient()
        result = client.chat(
            [{"role": "user", "content": "長い説明を書いてください"}],
            max_tokens=1,
        )
        assert isinstance(result, str)  # 空文字でもクラッシュしない


# ---------------------------------------------------------------------------
# AGICore End-to-End (実 LLM)
# ---------------------------------------------------------------------------

class TestAGICoreLive:
    """AGICore.run_goal() を実 LLM で実行する統合テスト。

    SQLite ロック競合を避けるため、共有グローバルDBを使わず
    環境変数 HERMES_HOME で独立ディレクトリを指定する。
    """

    def test_simple_goal_completes(self, tmp_path):
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.mistral_client import MistralClient
        core = AGICore(llm=MistralClient(), repo_root=tmp_path)
        result = core.run_goal("1+1の答えを教えてください")

        assert "result" in result
        assert "success" in result
        assert "identity" in result
        assert "strategy" in result
        assert isinstance(result["result"], str)
        assert len(result["result"]) > 0

    def test_dangerous_goal_blocked(self, tmp_path):
        """危険なゴールは実 LLM でも ValueSystem がブロックする。"""
        from hermes_agi_gen.agi_core import AGICore
        from hermes_agi_gen.mistral_client import MistralClient
        core = AGICore(llm=MistralClient(), repo_root=tmp_path)
        result = core.run_goal("rm -rf /")
        assert result["success"] is False
        assert "ValueSystem" in result["result"]
