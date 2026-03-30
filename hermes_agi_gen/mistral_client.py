"""Mistral / Ollama LLM クライアント。

環境変数 MISTRAL_API_KEY が設定されていれば Mistral API (api.mistral.ai) を使用し、
未設定の場合はローカル Ollama (http://127.0.0.1:11434) にフォールバックする。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from .hermes_constants import (
    CLAUDE_DEFAULT_MODEL, CLAUDE_FAST_MODEL,
    DEFAULT_MISTRAL_MODEL, GROQ_BASE_URL, GROQ_DEFAULT_MODEL, GROQ_FAST_MODEL,
    MISTRAL_API_BASE_URL, OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60


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
                # \; \+ など JSON 非対応のエスケープ → バックスラッシュなしで文字のみ出力
                result.append(ch)
            else:
                result.append("\\")
                result.append(ch)
            continue
        if ch == "\\":
            if in_string:
                escape = True  # バックスラッシュはバッファリング (次の文字を確認してから出力)
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
    """OpenAI 互換エンドポイント経由で Mistral/Ollama を呼び出す。"""

    # バックエンド選択キーワード → 実際のモデル名
    _BACKEND_ALIASES: dict[str, tuple[str, str]] = {
        # keyword: (base_url_key, default_model)
        "groq":    ("groq",    GROQ_DEFAULT_MODEL),
        "mistral": ("mistral", "mistral-small-latest"),
        "ollama":  ("ollama",  DEFAULT_MISTRAL_MODEL),
    }

    def __init__(
        self,
        model: str = DEFAULT_MISTRAL_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        mistral_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

        model_lower = model.lower()
        # デフォルトモデル名 "mistral" は自動選択扱い (バックエンドキーワードと区別)
        is_auto = (model == DEFAULT_MISTRAL_MODEL)

        # ━━━ バックエンド優先順位 ━━━
        # 1. Claude (ANTHROPIC_API_KEY) — groq/ollama が明示指定された場合はスキップ
        if anthropic_key and model_lower not in ("groq", "ollama") and not base_url:
            from .claude_client import ClaudeClient
            self._claude_client: Optional[ClaudeClient] = ClaudeClient(
                model=os.getenv("CLAUDE_MODEL", CLAUDE_DEFAULT_MODEL)
            )
            self.api_key = anthropic_key
            self.base_url = "https://api.anthropic.com"
            self.model = self._claude_client.model
            return
        else:
            self._claude_client = None

        # 2. Groq (明示的キーワード指定 or 自動)
        if model_lower == "groq":
            if not groq_key:
                raise ValueError(
                    "モデルに 'groq' を指定しましたが GROQ_API_KEY が設定されていません。\n"
                    "  export GROQ_API_KEY=your_key  を実行してから再試行してください。\n"
                    "  Groq APIキーは https://console.groq.com で無料取得できます。\n"
                    "  ローカルOllamaを使う場合は --model qwen3 など実際のモデル名を指定してください。"
                )
            self.api_key = groq_key
            self.base_url = base_url or GROQ_BASE_URL
            self.model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
        # 3. Mistral (明示的キーワード指定 + MISTRAL_API_KEY あり)
        elif model_lower == "mistral" and not is_auto and mistral_key:
            self.api_key = mistral_key
            self.base_url = base_url or MISTRAL_API_BASE_URL
            self.model = "mistral-small-latest"
        elif model_lower == "mistral" and not is_auto and not mistral_key:
            raise ValueError(
                "モデルに 'mistral' を指定しましたが MISTRAL_API_KEY が設定されていません。\n"
                "  export MISTRAL_API_KEY=your_key  を実行してから再試行してください。"
            )
        # 4. 自動選択: MISTRAL_API_KEY → GROQ_API_KEY → Ollama
        elif mistral_key:
            self.api_key = mistral_key
            self.base_url = base_url or MISTRAL_API_BASE_URL
            self.model = model if not is_auto else "mistral-small-latest"
        elif groq_key:
            self.api_key = groq_key
            self.base_url = base_url or GROQ_BASE_URL
            self.model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL) if is_auto else model
        else:
            # Ollama は認証不要。OLLAMA_MODEL 環境変数でモデルを上書き可能
            self.api_key = "ollama"
            self.base_url = base_url or OLLAMA_BASE_URL
            self.model = os.getenv("OLLAMA_MODEL", model)

    # ------------------------------------------------------------------
    # ファクトリ
    # ------------------------------------------------------------------

    @classmethod
    def fast(cls) -> "MistralClient":
        """インテント分類など軽量タスク用の高速モデルを返す。

        Claude使用時は Haiku、Groq使用時は GROQ_FAST_MODEL を使用。
        """
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            # Claude Haiku を使用
            inst = cls.__new__(cls)
            from .claude_client import ClaudeClient
            inst._claude_client = ClaudeClient(model=CLAUDE_FAST_MODEL)
            inst.api_key = anthropic_key
            inst.base_url = "https://api.anthropic.com"
            inst.model = CLAUDE_FAST_MODEL
            return inst

        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            fast_model = os.getenv("GROQ_FAST_MODEL", GROQ_FAST_MODEL)
            return cls(model=fast_model)
        # Groq なし → 通常モデルをそのまま使用
        return cls()

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
        # Claude クライアントに委譲
        if getattr(self, "_claude_client", None) is not None:
            return self._claude_client.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
            if resp.status_code == 401:
                logger.error(
                    "MistralClient.chat 認証エラー (401): APIキーが無効または期限切れです。"
                    " .env の GROQ_API_KEY / MISTRAL_API_KEY を確認してください。"
                    " キーを削除するとローカル Ollama にフォールバックします。"
                )
                return ""
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("MistralClient.chat エラー: %s", exc)
            return ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Optional[Any]:
        """JSON を期待する呼び出し。マークダウンコードブロックを除去して parse する。"""
        # Claude クライアントに委譲
        if getattr(self, "_claude_client", None) is not None:
            return self._claude_client.chat_json(
                messages, temperature=temperature, max_tokens=max_tokens
            )

        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if not raw:
            return None
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            # 括弧の深さを追跡して最初の完全な JSON オブジェクト/配列を抽出
            result = _extract_first_json(cleaned)
            if result is not None:
                return result
        logger.warning("JSON parse 失敗 (先頭200文字): %s", raw[:200])
        return None
