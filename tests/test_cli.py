"""Tests for cli.py — command parser, schedule detection, intent classification, HN pipeline.

All tests are self-contained: no external LLM, Ollama, or network calls.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# Import CLI functions under test
# =========================================================================

from cli import (
    _is_likely_chat,
    _extract_schedule_trigger,
    _provider_label,
    _has_action_verb,
    _classify_intent,
    _build_llm,
    _cmd_mvp,
    _parse_args,
    _TIME_SPEC_RE,
)


# =========================================================================
# _is_likely_chat
# =========================================================================

class TestIsLikelyChat:
    """会話かタスクかの判定。"""

    @pytest.mark.parametrize("msg", [
        "こんにちは",
        "ありがとう",
        "あなたは何ができますか",
        "hello",
        "hi there",
        "どう思いますか",
    ])
    def test_chat_messages(self, msg):
        assert _is_likely_chat(msg) is True

    @pytest.mark.parametrize("msg", [
        "ファイルを作って",
        "テストを実行して",
        "コードを修正して",
        "ファイルを削除して",
        "デプロイして",
    ])
    def test_task_messages(self, msg):
        assert _is_likely_chat(msg) is False

    def test_empty_string(self):
        assert _is_likely_chat("") is False


# =========================================================================
# _has_action_verb
# =========================================================================

class TestHasActionVerb:
    def test_create(self):
        assert _has_action_verb("ファイルを作って") is True

    def test_run(self):
        assert _has_action_verb("テストを実行して") is True

    def test_delete(self):
        assert _has_action_verb("ファイルを削除して") is True

    def test_no_action(self):
        assert _has_action_verb("こんにちは") is False


# =========================================================================
# _extract_schedule_trigger
# =========================================================================

class TestExtractScheduleTrigger:
    """自然言語からスケジュールトリガーを抽出。"""

    def test_iso8601_direct(self):
        result = _extract_schedule_trigger("2026-04-12T17:30に実行して")
        assert result == "once:2026-04-12T17:30"

    def test_japanese_date_time(self):
        result = _extract_schedule_trigger("2026年4月12日午前9時になったらニュースを取得して")
        assert result is not None
        assert result.startswith("once:2026-04")
        assert "T09:00" in result

    def test_pm_time(self):
        result = _extract_schedule_trigger("午後5時になったら天気予報を取得して")
        assert result is not None
        assert "T17:00" in result

    def test_24hour_time(self):
        result = _extract_schedule_trigger("17時30分になったらHNを取得して")
        assert result is not None
        assert "T17:30" in result

    def test_hhmm_format(self):
        result = _extract_schedule_trigger("09:00にチェックして")
        assert result == "daily:09:00"

    def test_no_time_returns_none(self):
        result = _extract_schedule_trigger("ファイルを確認して")
        assert result is None

    def test_midnight(self):
        result = _extract_schedule_trigger("午前0時になったら実行して")
        assert result is not None
        assert "T00:00" in result

    def test_noon(self):
        result = _extract_schedule_trigger("午後12時になったら確認して")
        assert result is not None
        assert "T12:00" in result

    def test_time_spec_regex_matches(self):
        assert _TIME_SPEC_RE.search("17時30分になったら") is not None
        assert _TIME_SPEC_RE.search("午前9時になったら") is not None
        assert _TIME_SPEC_RE.search("2026年4月12日") is not None
        assert _TIME_SPEC_RE.search("daily:09:00") is not None
        assert _TIME_SPEC_RE.search("ただのテキスト") is None


# =========================================================================
# _classify_intent
# =========================================================================

class TestClassifyIntent:
    """インテント分類。"""

    def test_chat_message_classified_without_llm(self):
        mock_llm = MagicMock()
        intent, domain = _classify_intent(mock_llm, "こんにちは")
        assert intent == "chat"
        # LLM は呼ばれない (チャットはローカル判定)
        mock_llm.chat_json.assert_not_called()

    def test_task_with_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"type": "task", "domain": "coding"}
        intent, domain = _classify_intent(mock_llm, "テストを実行して")
        assert intent == "task"
        assert domain == "coding"

    def test_unknown_domain_defaults_to_general(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"type": "task", "domain": "unknown_xyz"}
        intent, domain = _classify_intent(mock_llm, "何かを実行して")
        assert domain == "general"

    def test_llm_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = None
        intent, domain = _classify_intent(mock_llm, "ファイルを作って")
        assert intent == "task"

    def test_action_verb_overrides_chat(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"type": "chat", "domain": "general"}
        intent, domain = _classify_intent(mock_llm, "ファイルを削除して")
        assert intent == "task"


# =========================================================================
# _provider_label
# =========================================================================

class TestProviderLabel:
    def test_returns_ollama_label(self):
        mock_llm = MagicMock()
        mock_llm.model = "gemma4:e4b"
        mock_llm.provider = "ollama"
        label = _provider_label(mock_llm)
        assert "Ollama" in label
        assert "gemma4:e4b" in label

    def test_returns_openai_label(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-5.5"
        mock_llm.provider = "openai"
        label = _provider_label(mock_llm)
        assert "OpenAI" in label
        assert "gpt-5.5" in label


# =========================================================================
# _build_llm
# =========================================================================

class TestBuildLlm:
    @patch.dict("os.environ", {}, clear=True)
    def test_returns_mistral_client(self):
        from hermes_agi_gen.mistral_client import MistralClient
        with patch("cli.console"):
            llm = _build_llm(None)
        assert isinstance(llm, MistralClient)
        assert llm.model == "gemma4:e4b"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_builds_openai_for_gpt_model(self):
        with patch("cli.console"):
            llm = _build_llm("gpt-5.5")
        assert llm.provider == "openai"
        assert llm.model == "gpt-5.5"


# =========================================================================
# _parse_args
# =========================================================================

class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["cli.py"]):
            args = _parse_args()
        assert args.max_turns == 8
        assert args.daemon is False

    def test_max_turns(self):
        with patch("sys.argv", ["cli.py", "--max-turns", "12"]):
            args = _parse_args()
        assert args.max_turns == 12

    def test_daemon_flag(self):
        with patch("sys.argv", ["cli.py", "--daemon"]):
            args = _parse_args()
        assert args.daemon is True

    def test_model_arg(self):
        with patch("sys.argv", ["cli.py", "--model", "gpt-5.5"]):
            args = _parse_args()
        assert args.model == "gpt-5.5"


# =========================================================================
# HN パイプライン判定ロジック
# =========================================================================

class TestHNPipelineDetection:
    """Hacker News パイプラインの発動条件。"""

    def test_hn_japanese_keyword_detected(self):
        """HN + 日本語キーワードを含むゴールを検出。"""
        goal = "Hacker Newsのニュースを日本語で表示して"
        assert "hacker news" in goal.lower() or "HN" in goal.upper()
        assert any(w in goal for w in ["日本語", "翻訳"])

    def test_hn_with_save_path(self):
        """保存パス付きHNゴールのパス抽出。"""
        import re
        goal = "HNのAIニュースを翻訳して~/Desktop/AI_News/に保存して"
        m = re.search(r'~/[a-zA-Z0-9/_.-]+', goal)
        assert m is not None
        assert m.group(0).rstrip("/") == "~/Desktop/AI_News"

    def test_non_hn_goal_not_detected(self):
        goal = "Pythonのテストを実行して"
        assert "hacker news" not in goal.lower()
        assert "hn" not in goal.lower().split()


# =========================================================================
# REPL コマンドディスパッチ
# =========================================================================

from cli import (
    _cmd_tools,
    _cmd_goals,
    _cmd_world,
    _cmd_daemon,
    _cmd_schedule,
)
from hermes_agi_gen.tool_registry import ToolRegistry
from hermes_agi_gen.agent_state import AgentState


class TestCmdTools:
    """_cmd_tools のテスト。"""

    def test_list_empty(self, tmp_path, capsys):
        reg = ToolRegistry(db_path=tmp_path / "tools.db")
        with patch("cli.console") as mock_con:
            _cmd_tools(reg, "")
        # カスタムツールなしのメッセージが表示される
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("ありません" in c for c in calls)

    def test_list_with_tool(self, tmp_path):
        reg = ToolRegistry(db_path=tmp_path / "tools.db")
        reg.register(
            name="test_tool", description="テスト", invocation_prefix="TESTTOOL",
            code="def main(args): return 'ok'",
        )
        with patch("cli.console") as mock_con:
            _cmd_tools(reg, "")
        # テーブルが表示される (Tableオブジェクト)
        calls = mock_con.print.call_args_list
        assert len(calls) >= 1


class TestCmdGoals:
    """_cmd_goals のテスト。"""

    def test_no_state(self):
        with patch("cli.console") as mock_con:
            _cmd_goals(None)
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("まだ" in c for c in calls)

    def test_with_state(self):
        state = AgentState(user_goal="test", max_iterations=5)
        with patch("cli.console") as mock_con:
            _cmd_goals(state)
        # 何らかの出力がある (エラーなし)
        assert mock_con.print.called


class TestCmdWorld:
    """_cmd_world のテスト。"""

    def test_no_state(self):
        with patch("cli.console") as mock_con:
            _cmd_world(None)
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("まだ" in c or "エージェント" in c for c in calls)


class TestCmdDaemon:
    """_cmd_daemon のテスト。"""

    def test_unknown_subcmd(self):
        with patch("cli.console") as mock_con:
            _cmd_daemon("unknown_xyz")
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("使い方" in c for c in calls)

    def test_status_not_running(self):
        with patch("cli.HermesDaemon") as mock_daemon:
            mock_daemon.get_status.return_value = {"running": False, "pid": None}
            with patch("cli.console") as mock_con:
                _cmd_daemon("status")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("起動していません" in c for c in calls)

    def test_start_already_running(self):
        with patch("cli.HermesDaemon") as mock_daemon:
            mock_daemon.get_status.return_value = {"running": True, "pid": 12345}
            with patch("cli.console") as mock_con:
                _cmd_daemon("start")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("すでに起動中" in c for c in calls)


class TestCmdSchedule:
    """_cmd_schedule のテスト。"""

    def test_list_empty(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched.list_jobs.return_value = []
            mock_sched_cls.return_value = mock_sched
            with patch("cli.console") as mock_con:
                _cmd_schedule("")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("ありません" in c for c in calls)

    def test_add_missing_args(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched_cls.return_value = MagicMock()
            with patch("cli.console") as mock_con:
                _cmd_schedule("add")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("使い方" in c for c in calls)

    def test_add_invalid_trigger(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched_cls.return_value = MagicMock()
            with patch("cli.console") as mock_con:
                with patch("cli.parse_trigger_spec", return_value=None):
                    _cmd_schedule("add invalid_trigger テスト目標")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("不明" in c for c in calls)

    def test_add_valid_trigger(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "test-123"
            mock_job.goal = "テスト目標"
            mock_sched.add_job.return_value = mock_job
            mock_sched.format_next_run.return_value = "明日 09:00"
            mock_sched_cls.return_value = mock_sched
            with patch("cli.console") as mock_con:
                with patch("cli.parse_trigger_spec", return_value="daily:09:00"):
                    _cmd_schedule("add daily:09:00 テスト目標")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("登録" in c for c in calls)

    def test_remove_missing_id(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched_cls.return_value = MagicMock()
            with patch("cli.console") as mock_con:
                _cmd_schedule("remove")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("使い方" in c for c in calls)

    def test_remove_success(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched.remove_job.return_value = True
            mock_sched_cls.return_value = mock_sched
            with patch("cli.console") as mock_con:
                _cmd_schedule("remove job-abc")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("削除" in c for c in calls)

    def test_enable_success(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched.enable_job.return_value = True
            mock_sched_cls.return_value = mock_sched
            with patch("cli.console") as mock_con:
                _cmd_schedule("enable job-abc")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("有効化" in c for c in calls)

    def test_disable_success(self):
        with patch("cli.JobScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched.enable_job.return_value = True
            mock_sched_cls.return_value = mock_sched
            with patch("cli.console") as mock_con:
                _cmd_schedule("disable job-abc")
            calls = [str(c) for c in mock_con.print.call_args_list]
            assert any("無効化" in c for c in calls)


class TestCmdMvp:
    def test_missing_goal(self):
        with patch("cli.console") as mock_con:
            result = _cmd_mvp("")
        assert result == {}
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("使い方" in c for c in calls)

    def test_runs_spec_mvp(self):
        with patch("cli.console") as mock_con:
            with patch("cli.run_spec_mvp", return_value={
                "task": {"goal": "テスト", "status": "done"},
                "review": {"score": 1.0, "feedback": "OK", "done": True},
                "iterations": 1,
                "memory_path": ".hermes/spec_mvp_memory.json",
            }):
                result = _cmd_mvp("テスト")
        assert result["review"]["done"] is True
        assert mock_con.print.called


# =========================================================================
# REPL main() ループのシミュレーション
# =========================================================================


class TestMainREPL:
    """main() の REPL ループを Prompt.ask のモックでシミュレーション。"""

    def test_quit_exits_cleanly(self):
        """'/quit' で正常終了する。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.return_value = "/quit"
                with patch("cli.console"):
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()  # /quit で即座にループを抜ける

    def test_help_command(self):
        """'/help' でヘルプが表示される。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.side_effect = ["/help", "/quit"]
                with patch("cli.console") as mock_con:
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()
                # Markdown がコンソールに出力された
                assert mock_con.print.called

    def test_clear_resets_history(self):
        """'/clear' で会話履歴がリセットされる。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.side_effect = ["/clear", "/quit"]
                with patch("cli.console") as mock_con:
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()
                calls = [str(c) for c in mock_con.print.call_args_list]
                assert any("リセット" in c for c in calls)

    def test_empty_input_continues(self):
        """空入力はスキップされる。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.side_effect = ["", "  ", "/quit"]
                with patch("cli.console"):
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()

    def test_eof_exits(self):
        """EOFError で正常終了する。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.side_effect = EOFError()
                with patch("cli.console"):
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()

    def test_keyboard_interrupt_exits(self):
        """KeyboardInterrupt で正常終了する。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.side_effect = KeyboardInterrupt()
                with patch("cli.console"):
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()

    def test_provider_command(self):
        """'/provider' でプロバイダー情報が表示される。"""
        with patch("sys.argv", ["cli.py"]):
            with patch("cli.Prompt") as mock_prompt:
                mock_prompt.ask.side_effect = ["/provider", "/quit"]
                with patch("cli.console") as mock_con:
                    with patch("cli._start_inline_scheduler", return_value=MagicMock()):
                        from cli import main
                        main()
                calls = [str(c) for c in mock_con.print.call_args_list]
                assert any("Ollama" in c for c in calls)
