"""Hermes AGI Gen 共有定数。"""
from __future__ import annotations

import os
from pathlib import Path


def get_hermes_home() -> Path:
    """HERMES_HOME ディレクトリを返す。

    環境変数 HERMES_HOME が設定されていればそれを使い、
    未設定なら ~/.hermes にフォールバックする。
    ディレクトリが存在しなければ自動作成する。
    """
    home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    home.mkdir(parents=True, exist_ok=True)
    return home


# Ollama (ローカル gemma4:e4b 専用)
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "gemma4:e4b"

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
