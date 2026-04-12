"""Ollama LLM クライアント (gemma4:e4b 専用)。

ローカル Ollama (http://127.0.0.1:11434) の gemma4:e4b モデルのみを使用する。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .hermes_constants import DEFAULT_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 180


_VALID_JSON_ESCAPES = set('"\\bfnrtu/')


def _sanitize_json_strings(text: str) -> str:
    """JSON文字列値内の不正なエスケープ・裸の改行などを修正する。"""
    result: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            if in_string and ch not in _VALID_JSON_ESCAPES:
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


class MistralClient:
    """Ollama (gemma4:e4b) 専用 LLM クライアント。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = "ollama"
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = DEFAULT_MODEL
        self._claude_client = None

    @classmethod
    def fast(cls) -> "MistralClient":
        """軽量タスク用 — 同じ gemma4:e4b を返す。"""
        return cls()

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """メッセージリストを送信してテキスト応答を返す。エラー時は空文字。"""
        headers = {
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # thinking モデルの <think>...</think> ブロックを除去
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as exc:
            logger.error("MistralClient.chat エラー: %s", exc)
            return ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Optional[Any]:
        """JSON を期待する呼び出し。マークダウンコードブロックを除去して parse する。"""
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if not raw:
            return None
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            result = _extract_first_json(cleaned)
            if result is not None:
                return result
        logger.warning("JSON parse 失敗 (先頭200文字): %s", raw[:200])
        return None
