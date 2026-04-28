"""Hermes AGI Gen 共有定数。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _is_writable_dir(path: Path) -> bool:
    """Return True when Hermes can create and replace files in path."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".hermes_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def get_hermes_home() -> Path:
    """HERMES_HOME ディレクトリを返す。

    環境変数 HERMES_HOME が設定されていればそれを使い、
    未設定なら ~/.hermes にフォールバックする。書き込み不可の環境では
    カレントディレクトリ配下の .hermes、最後に tempdir へ退避する。
    """
    configured = os.getenv("HERMES_HOME")
    candidates = [
        Path(configured).expanduser() if configured else Path.home() / ".hermes",
        Path.cwd() / ".hermes",
        Path(tempfile.gettempdir()) / "hermes-agi-gen",
    ]
    for home in candidates:
        if _is_writable_dir(home):
            return home
    raise OSError("Hermes home directory is not writable")


def get_hermes_path(filename: str) -> Path:
    """Return a file path under the current Hermes home.

    Keeping this lookup dynamic lets tests and embedding applications set
    HERMES_HOME after modules have been imported.
    """
    return get_hermes_home() / filename


# LLM providers
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_OPENAI_MODEL = "gpt-5.5"

# ドメイン別エージェント設定
DOMAIN_CONFIG: dict[str, dict] = {
    "general": {
        "success_criteria": ["目標を達成した", "結果を日本語で説明できる"],
        "constraints": ["まず現状を把握する"],
    },
    "coding": {
        "success_criteria": ["コードが動作する", "テストがパスする", "結果を日本語で説明できる"],
        "constraints": ["破壊的操作はしない", "まず現状を把握する"],
    },
    "research": {
        "success_criteria": ["情報を収集・整理できた", "信頼性を確認した", "結果を日本語でまとめられる"],
        "constraints": ["情報源を明記する", "推測と事実を区別する"],
    },
    "writing": {
        "success_criteria": ["文章を完成させた", "目的に合った表現になっている", "結果を日本語で説明できる"],
        "constraints": ["ユーザーの意図を尊重する", "簡潔かつ明確に書く"],
    },
    "data": {
        "success_criteria": ["データを分析できた", "洞察を抽出した", "結果を日本語で説明できる"],
        "constraints": ["データの整合性を保つ", "まず現状を把握する"],
    },
    "ops": {
        "success_criteria": ["タスクを実行できた", "結果を確認した", "結果を日本語で説明できる"],
        "constraints": ["破壊的操作はしない", "まず現状を確認する"],
    },
}
