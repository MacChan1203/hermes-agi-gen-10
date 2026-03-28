"""コード生成・レビュー専用エージェント。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .mistral_client import MistralClient
from .state_store import SessionDB

# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------

_GENERATOR_SYSTEM = """\
あなたはコード生成の専門家です。ユーザーの要求に基づいて、動作する高品質なコードを生成してください。

ルール:
- コードは必ずフェンスコードブロック (```言語名 ... ```) で囲む
- 非自明なロジックには簡潔なコメントを付ける
- エッジケースと入力バリデーションを考慮する
- 要求が曖昧な場合は、仮定を最初に一行で述べてからコードを出力する
- コードブロックの後に「使い方」セクションを追加する (2〜4行)

出力形式:
1. 何を生成したかを一文で説明
2. フェンスコードブロック
3. 「## 使い方」セクション
"""

_REVIEWER_SYSTEM = """\
あなたはシニアコードレビュアーです。提供されたコードをレビューして、構造化されたフィードバックを返してください。

以下の観点でレビューしてください:
1. **正確性** - ロジックは意図通りか？バグはないか？
2. **セキュリティ** - SQLインジェクション・XSS・パストラバーサル・ハードコードされた秘密情報など
3. **パフォーマンス** - 不必要なループ・非効率な処理はないか？
4. **保守性** - 命名の明確さ・関数の長さ・マジックナンバーはないか？
5. **テスト** - 追加すべきテストケースは何か？

必ずこの形式で出力してください:

## 概要
一文での総評。

## 発見された問題
箇条書き。各項目: [重要度: 致命的/高/中/低] 説明

## 修正案
致命的/高の問題のみ、修正後のコードスニペットを示す。

## テストケース
具体的なテストケースを2〜3個説明する。
"""


class CodeGeneratorAgent:
    """自然言語からコードを生成するエージェント。"""

    def __init__(
        self,
        llm: MistralClient,
        session_db: Optional[SessionDB] = None,
    ) -> None:
        self._llm = llm
        self._db = session_db or SessionDB()

    def generate(self, request: str) -> str:
        """自然言語の要求からコードを生成して返す。"""
        result = self._llm.chat(
            [
                {"role": "system", "content": _GENERATOR_SYSTEM},
                {"role": "user", "content": request},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        if not result:
            return "（コードを生成できませんでした。モデルや接続を確認してください）"
        return result


class CodeReviewerAgent:
    """コードをレビューしてフィードバックを返すエージェント。"""

    def __init__(
        self,
        llm: MistralClient,
        session_db: Optional[SessionDB] = None,
    ) -> None:
        self._llm = llm
        self._db = session_db or SessionDB()

    def review(self, code: str) -> str:
        """コードをレビューして構造化フィードバックを返す。"""
        result = self._llm.chat(
            [
                {"role": "system", "content": _REVIEWER_SYSTEM},
                {"role": "user", "content": f"以下のコードをレビューしてください:\n\n```\n{code}\n```"},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        if not result:
            return "（レビューを生成できませんでした。モデルや接続を確認してください）"
        return result
