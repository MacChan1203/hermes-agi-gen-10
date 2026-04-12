"""Tests for mistral_client, code_agents, utils, toolsets, toolset_distributions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# MistralClient
# =========================================================================
from hermes_agi_gen.mistral_client import (
    MistralClient,
    _extract_first_json,
    _sanitize_json_strings,
)


class TestMistralClientInit:
    def test_defaults_to_gemma4(self):
        c = MistralClient()
        assert c.model == "gemma4:e4b"
        assert "11434" in c.base_url

    def test_fast_returns_same_model(self):
        c = MistralClient.fast()
        assert c.model == "gemma4:e4b"


class TestMistralClientChat:
    def test_chat_returns_text(self):
        c = MistralClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello world"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("hermes_agi_gen.mistral_client.requests.post", return_value=mock_resp):
            result = c.chat([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    def test_chat_handles_connection_error(self):
        c = MistralClient()
        with patch("hermes_agi_gen.mistral_client.requests.post", side_effect=Exception("conn error")):
            result = c.chat([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_chat_strips_think_tags(self):
        c = MistralClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "<think>thinking...</think>actual answer"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("hermes_agi_gen.mistral_client.requests.post", return_value=mock_resp):
            result = c.chat([{"role": "user", "content": "test"}])
        assert "think" not in result
        assert result == "actual answer"


class TestMistralClientChatJson:
    def test_parses_json(self):
        c = MistralClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("hermes_agi_gen.mistral_client.requests.post", return_value=mock_resp):
            result = c.chat_json([{"role": "user", "content": "give json"}])
        assert result == {"key": "value"}

    def test_parses_json_in_code_block(self):
        c = MistralClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("hermes_agi_gen.mistral_client.requests.post", return_value=mock_resp):
            result = c.chat_json([{"role": "user", "content": "test"}])
        assert result == {"a": 1}

    def test_returns_none_on_invalid_json(self):
        c = MistralClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("hermes_agi_gen.mistral_client.requests.post", return_value=mock_resp):
            result = c.chat_json([{"role": "user", "content": "test"}])
        assert result is None


class TestExtractFirstJson:
    def test_extracts_from_mixed_text(self):
        text = 'Here is the result: {"name": "test", "value": 42} done.'
        result = _extract_first_json(text)
        assert result == {"name": "test", "value": 42}

    def test_extracts_array(self):
        text = 'Results: [1, 2, 3] end'
        result = _extract_first_json(text)
        assert result == [1, 2, 3]

    def test_returns_none_for_no_json(self):
        assert _extract_first_json("no json here") is None

    def test_handles_nested_objects(self):
        text = '{"outer": {"inner": true}}'
        result = _extract_first_json(text)
        assert result == {"outer": {"inner": True}}


class TestSanitizeJsonStrings:
    def test_fixes_invalid_escape(self):
        text = r'{"path": "C:\Users\test"}'
        result = _sanitize_json_strings(text)
        parsed = json.loads(result)
        assert "Users" in parsed["path"]

    def test_preserves_valid_escapes(self):
        text = r'{"msg": "hello\nworld"}'
        result = _sanitize_json_strings(text)
        parsed = json.loads(result)
        assert "\n" in parsed["msg"]


# =========================================================================
# CodeAgents
# =========================================================================
from hermes_agi_gen.code_agents import CodeGeneratorAgent, CodeReviewerAgent


class TestCodeAgents:
    def test_generator_init(self):
        mock_llm = MagicMock()
        agent = CodeGeneratorAgent(llm=mock_llm)
        assert agent._llm is mock_llm

    def test_reviewer_init(self):
        mock_llm = MagicMock()
        agent = CodeReviewerAgent(llm=mock_llm)
        assert agent._llm is mock_llm

    def test_generator_has_generate_method(self):
        mock_llm = MagicMock()
        agent = CodeGeneratorAgent(llm=mock_llm)
        assert hasattr(agent, "generate")

    def test_reviewer_has_review_method(self):
        mock_llm = MagicMock()
        agent = CodeReviewerAgent(llm=mock_llm)
        assert hasattr(agent, "review")


# =========================================================================
# Utils
# =========================================================================
from hermes_agi_gen.utils import atomic_json_write


class TestAtomicJsonWrite:
    def test_writes_valid_json(self, tmp_path):
        filepath = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        atomic_json_write(filepath, data)
        loaded = json.loads(filepath.read_text(encoding="utf-8"))
        assert loaded == data

    def test_preserves_permissions(self, tmp_path):
        filepath = tmp_path / "test.json"
        filepath.write_text("{}", encoding="utf-8")
        os.chmod(filepath, 0o644)
        original_mode = filepath.stat().st_mode
        atomic_json_write(filepath, {"updated": True})
        assert filepath.stat().st_mode == original_mode

    def test_no_temp_file_left(self, tmp_path):
        filepath = tmp_path / "test.json"
        atomic_json_write(filepath, {"clean": True})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_content_correct(self, tmp_path):
        filepath = tmp_path / "test.json"
        data = {"items": [1, 2, 3], "nested": {"a": "b"}}
        atomic_json_write(filepath, data)
        loaded = json.loads(filepath.read_text(encoding="utf-8"))
        assert loaded["items"] == [1, 2, 3]
        assert loaded["nested"]["a"] == "b"


# =========================================================================
# Toolsets
# =========================================================================
from hermes_agi_gen.toolsets import (
    resolve_toolset,
    validate_toolset,
    get_all_toolsets,
    get_toolset_info,
)


class TestToolsets:
    def test_resolve_web(self):
        tools = resolve_toolset("web")
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_resolve_all(self):
        tools = resolve_toolset("all")
        assert len(tools) > 10
        assert "terminal" in tools

    def test_validate_known(self):
        assert validate_toolset("web") is True
        assert validate_toolset("terminal") is True

    def test_validate_unknown(self):
        assert validate_toolset("nonexistent_xyz") is False

    def test_circular_include(self):
        # Should not crash even with included toolsets
        tools = resolve_toolset("development")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_toolset_info(self):
        info = get_toolset_info("web")
        assert "description" in info
        assert "tools" in info

    def test_get_all_toolsets(self):
        all_ts = get_all_toolsets()
        assert len(all_ts) > 5
        assert "all" in all_ts


# =========================================================================
# Toolset Distributions
# =========================================================================
from hermes_agi_gen.toolset_distributions import (
    get_distribution,
    list_distributions,
    sample_toolsets_from_distribution,
)


class TestToolsetDistributions:
    def test_get_distribution(self):
        dist = get_distribution("default")
        assert isinstance(dist, dict)
        assert "toolsets" in dist

    def test_list_distributions(self):
        dists = list_distributions()
        assert isinstance(dists, dict)
        assert len(dists) > 0
        assert "default" in dists

    def test_sample_returns_list(self):
        result = sample_toolsets_from_distribution("default")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_unknown_distribution_returns_none(self):
        dist = get_distribution("nonexistent_xyz")
        assert dist is None
