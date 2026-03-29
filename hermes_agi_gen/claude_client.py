"""Anthropic Claude API クライアント。

MistralClient と同じ公開インターフェース (.chat / .chat_json / .fast) を提供し、
ANTHROPIC_API_KEY が設定されている場合に最優先で使用される。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .hermes_constants import CLAUDE_DEFAULT_MODEL, CLAUDE_FAST_MODEL

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120


def _sanitize_json_strings(text: str) -> str:
    """JSON文字列値内の不正なエスケープ・裸の改行などを修正する。"""
    _VALID = set('"\\\/bfnrtu')
    result: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            if in_string and ch not in _VALID:
                result.append(ch)
            else:
                result.append("\\")
                result.append(ch)
            continue
        if ch == "\\":
            if in_string:
                escape = True
            else:
                result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == "\n":
                result.append("\\n")
                continue
            if ch == "\r":
                continue
            if ch == "\t":
                result.append("\\t")
                continue
        result.append(ch)
    return "".join(result)


def _extract_first_json(text: str) -> Optional[Any]:
    """括弧の深さを追跡して最初の完全な JSON オブジェクト/配列を抽出してパースする。"""
    sanitized = _sanitize_json_strings(text)
    for opener, closer in [('{', '}'), ('[', ']')]:
        start = sanitized.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(sanitized[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(sanitized[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class ClaudeClient:
    """Anthropic Claude API クライアント。

    MistralClient と同じ .chat() / .chat_json() / .fast() インターフェースを提供。
    Anthropic SDK を使用し、OpenAI形式のメッセージリストを自動変換する。
    """

    def __init__(self, model: str = CLAUDE_DEFAULT_MODEL) -> None:
        self.model = model
        self.base_url = "https://api.anthropic.com"
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._client = self._build_client()

    def _build_client(self):
        """Anthropic クライアントを構築する。SDKがない場合はNone。"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self._api_key)
        except ImportError:
            logger.warning(
                "anthropic パッケージが見つかりません。"
                "pip install anthropic でインストールしてください。"
            )
            return None

    @classmethod
    def is_available(cls) -> bool:
        """ANTHROPIC_API_KEY が設定されているか確認する。"""
        return bool(os.getenv("ANTHROPIC_API_KEY", ""))

    @classmethod
    def fast(cls) -> "ClaudeClient":
        """軽量タスク用の高速モデル (Haiku) を返す。"""
        return cls(model=CLAUDE_FAST_MODEL)

    # ------------------------------------------------------------------
    # メッセージ変換
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> tuple[str, list]:
        """OpenAI形式のメッセージリストをAnthropic形式に変換する。

        Returns:
            (system_prompt, anthropic_messages)
        """
        system_parts: list[str] = []
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                # 連続する同じロールをマージ
                if anthropic_messages and anthropic_messages[-1]["role"] == role:
                    prev = anthropic_messages[-1]["content"]
                    if isinstance(prev, str):
                        anthropic_messages[-1]["content"] = prev + "\n\n" + content
                    else:
                        anthropic_messages[-1]["content"] = str(prev) + "\n\n" + content
                else:
                    anthropic_messages.append({"role": role, "content": content})

        # 最初のメッセージがassistantの場合、userダミーを前置
        if anthropic_messages and anthropic_messages[0]["role"] == "assistant":
            anthropic_messages.insert(0, {"role": "user", "content": "続けてください。"})

        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        return system_prompt, anthropic_messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """メッセージリストを送信してテキスト応答を返す。エラー時は空文字。"""
        if self._client is None:
            logger.error("ClaudeClient: anthropic SDK が利用できません。")
            return ""

        system_prompt, anthropic_messages = self._convert_messages(messages)

        if not anthropic_messages:
            logger.error("ClaudeClient: 有効なメッセージがありません。")
            return ""

        try:
            kwargs: dict = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": anthropic_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self._client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as exc:
            logger.error("ClaudeClient.chat エラー: %s", exc)
            return ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Optional[Any]:
        """JSON を期待する呼び出し。マークダウンコードブロックを除去してパースする。"""
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if not raw:
            return None
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            result = _extract_first_json(cleaned)
            if result is not None:
                return result
        logger.warning("JSON parse 失敗 (先頭200文字): %s", raw[:200])
        return None
