"""LLM クライアント。

既定ではローカル Ollama (http://127.0.0.1:11434) の gemma4:e4b を使用する。
HERMES_LLM_PROVIDER=openai または GPT 系モデル指定時は OpenAI Responses API を使用する。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .hermes_constants import DEFAULT_MODEL, DEFAULT_OPENAI_MODEL, OLLAMA_BASE_URL, OPENAI_BASE_URL

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
    """Hermes 用 LLM クライアント。

    クラス名は後方互換のため維持する。provider="openai" では Responses API、
    provider="ollama" では OpenAI 互換の Ollama chat completions を優先し、
    古い Ollama などで 404 の場合は native /api/chat にフォールバックする。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        requested_model = model or os.getenv("HERMES_MODEL") or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        requested_provider = (provider or os.getenv("HERMES_LLM_PROVIDER") or "").strip().lower()
        if not requested_provider:
            requested_provider = "openai" if requested_model.startswith("gpt-") else "ollama"
        if requested_provider not in {"ollama", "openai"}:
            raise ValueError(f"未対応の LLM provider: {requested_provider}")

        self.provider = requested_provider
        if self.provider == "openai":
            self.model = requested_model if requested_model != DEFAULT_MODEL else DEFAULT_OPENAI_MODEL
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
        else:
            self.model = requested_model
            self.api_key = api_key or "ollama"
            self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or OLLAMA_BASE_URL).rstrip("/")
        self.reasoning_effort = reasoning_effort or os.getenv("HERMES_REASONING_EFFORT")
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
        if self.provider == "openai":
            return self._chat_openai_responses(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._chat_ollama(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
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
            return self._clean_model_text(resp.json()["choices"][0]["message"]["content"])
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 404:
                logger.info(
                    "Ollama OpenAI互換APIが404のため native /api/chat にフォールバックします"
                )
                return self._chat_ollama_native(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            detail = response.text[:1000] if response is not None else str(exc)
            logger.error("MistralClient.chat エラー: %s | %s", exc, detail)
            return ""
        except Exception as exc:
            logger.error("MistralClient.chat エラー: %s", exc)
            return ""

    def _chat_ollama_native(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self._ollama_native_base_url()}/api/chat",
                headers=headers,
                json=payload,
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return self._clean_model_text(message["content"])
            if isinstance(data.get("response"), str):
                return self._clean_model_text(data["response"])
            return ""
        except requests.HTTPError as exc:
            response = exc.response
            detail = response.text[:1000] if response is not None else str(exc)
            if "model" in detail.lower() and "not found" in detail.lower():
                logger.error(
                    "Ollamaモデル %s が見つかりません。先に `ollama pull %s` を実行してください。",
                    self.model,
                    self.model,
                )
            logger.error("Ollama native chat エラー: %s | %s", exc, detail)
            return ""
        except Exception as exc:
            logger.error("Ollama native chat エラー: %s", exc)
            return ""

    def _ollama_native_base_url(self) -> str:
        """Return Ollama server root, converting .../v1 to the native API root."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    @staticmethod
    def _clean_model_text(text: str) -> str:
        content = str(text).strip()
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    def _chat_openai_responses(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.api_key:
            logger.error("OpenAI provider requires OPENAI_API_KEY")
            return ""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": self._to_responses_input(messages),
            "max_output_tokens": max_tokens,
        }
        # GPT-5系の一部モデルは既定temperatureのみを受け付けるため、明示送信しない。
        if not self.model.startswith("gpt-5"):
            payload["temperature"] = temperature
        if self.model.startswith("gpt-5") and self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        try:
            resp = requests.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            return self._extract_responses_text(resp.json())
        except requests.HTTPError as exc:
            response = exc.response
            detail = response.text[:1000] if response is not None else str(exc)
            logger.error("OpenAI Responses chat エラー: %s | %s", exc, detail)
            return ""
        except Exception as exc:
            logger.error("OpenAI Responses chat エラー: %s", exc)
            return ""

    @staticmethod
    def _to_responses_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        role_map = {"system": "system", "developer": "developer", "user": "user", "assistant": "assistant"}
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = role_map.get(message.get("role", "user"), "user")
            converted.append({
                "type": "message",
                "role": role,
                "content": message.get("content", ""),
            })
        return converted

    @staticmethod
    def _extract_responses_text(data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"].strip()
        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                refusal = content.get("refusal")
                if isinstance(refusal, str):
                    chunks.append(refusal)
        return "\n".join(chunks).strip()

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
