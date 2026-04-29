"""コード生成・レビュー専用エージェント。

Gen 10.2: 離散トークンによる Generator ↔ Reviewer 通信。
peer_channel と codebook を渡すと、両者は離散トークンで意図を共有し、
Reviewer は受信トークンから「期待されるコードの特徴」を **予測**してから
実コードを観測する → 予測誤差を計算 → TokenCodebook に reward を返す
(サマリ ② 2体協調 + ③ 通信 + ④ 離散トークン + ⑤ RL + ⑥ 解釈 + ⑧ 自己評価)。

期待パターンは TokenCodebook.expected_patterns_of() から取得 (config の
語彙定義に一元化されている)。重複定義は持たない。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .mistral_client import MistralClient
from .peer_channel import PeerChannel
from .state_store import SessionDB
from .token_codebook import TokenCodebook
from .token_interpreter import TokenInterpreter

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


def _prediction_error(code: str, expected: Tuple[str, ...]) -> float:
    """期待語の出現率から予測誤差 (0=完全一致, 1=完全不一致) を返す。

    expected が空の時は 0.5 (中立) を返す — 「予測なし」と「完全予測」を
    区別する必要があるため。
    """
    if not expected:
        return 0.5
    body = code.lower()
    hits = sum(1 for token in expected if token.lower() in body)
    return 1.0 - hits / float(len(expected))


class CodeGeneratorAgent:
    """自然言語からコードを生成するエージェント。"""

    def __init__(
        self,
        llm: MistralClient,
        session_db: Optional[SessionDB] = None,
        peer_channel: Optional[PeerChannel] = None,
        codebook: Optional[TokenCodebook] = None,
    ) -> None:
        self._llm = llm
        self._db = session_db or SessionDB()
        self._channel = peer_channel
        self._codebook = codebook

    def generate(self, request: str) -> str:
        """自然言語の要求からコードを生成して返す。

        peer_channel と codebook が両方ある場合、離散トークンを
        "reviewer" 宛に送って意図を共有する (サマリ ③ 通信)。
        """
        if self._channel is not None and self._codebook is not None:
            token_id = self._codebook.emit(request)
            self._channel.send(
                sender="generator",
                receiver="reviewer",
                task=request,
                tokens=[token_id],
                extra={"intent_excerpt": request[:80]},
            )

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


class CodePrediction:
    """Reviewer がコードを観測する前に立てる予測。

    expected_patterns: 受信トークン由来の「コード中に出現するはずの特徴語」の和集合。
    このオブジェクトを保持しておき、後で実コードと照合して prediction_error を出す。
    """

    __slots__ = ("token_ids", "expected_patterns")

    def __init__(self, token_ids: Tuple[str, ...], expected_patterns: Tuple[str, ...]):
        self.token_ids = token_ids
        self.expected_patterns = expected_patterns

    def evaluate(self, code: str) -> float:
        """実コードに対する予測誤差を返す (0=的中, 1=完全外れ)。"""
        return _prediction_error(code, self.expected_patterns)


class CodeReviewerAgent:
    """コードをレビューしてフィードバックを返すエージェント。

    Gen 10.2: 受信トークンから **コード観測前に**期待パターンを予測し、
    実コード観測後に予測誤差を測って codebook に reward として戻す。
    """

    def __init__(
        self,
        llm: MistralClient,
        session_db: Optional[SessionDB] = None,
        peer_channel: Optional[PeerChannel] = None,
        codebook: Optional[TokenCodebook] = None,
    ) -> None:
        self._llm = llm
        self._db = session_db or SessionDB()
        self._channel = peer_channel
        self._codebook = codebook
        self._interpreter = TokenInterpreter(codebook) if codebook is not None else None

    # ------------------------------------------------------------------
    # サマリ「予測 → 観測 → 誤差 → 学習」の核
    # ------------------------------------------------------------------
    def predict(self) -> Optional[CodePrediction]:
        """受信箱からトークンを取り出し、期待パターン集合を **予測** する。

        コードを見る前に呼ぶこと。codebook/channel のどちらかが無ければ None。
        受信箱はここで消費される (1 回の予測 = 1 回の通信に対応)。
        """
        if self._channel is None or self._codebook is None:
            return None
        token_ids: list[str] = []
        for msg in self._channel.receive("reviewer"):
            token_ids.extend(PeerChannel.tokens_of(msg))
        if not token_ids:
            return None
        # 各トークンの expected_patterns の和集合を予測とする
        expected: list[str] = []
        seen: set[str] = set()
        for tid in token_ids:
            for p in self._codebook.expected_patterns_of(tid):
                if p not in seen:
                    seen.add(p)
                    expected.append(p)
        return CodePrediction(
            token_ids=tuple(token_ids),
            expected_patterns=tuple(expected),
        )

    def review(self, code: str) -> str:
        """コードをレビューして構造化フィードバックを返す。

        フロー (codebook + channel あり時):
          1. predict() — 受信トークンから期待パターンを立てる (観測前)
          2. LLM でレビュー生成 (token のラベルをプロンプトヒントに混ぜる)
          3. evaluate — 実コードと予測の誤差を計算
          4. record_reward — 誤差小なら正の reward (サマリ ⑤ + ⑧)
          5. レビュー末尾に解釈層の説明を付与 (サマリ ⑥ ライブパス)
        """
        prediction = self.predict()

        token_hint = ""
        if prediction and self._codebook is not None:
            labels = [
                f"{t}({self._codebook.label_of(t)})" for t in prediction.token_ids
            ]
            token_hint = (
                f"\n[Generator からの意図トークン: {', '.join(labels)}]"
            )

        result = self._llm.chat(
            [
                {"role": "system", "content": _REVIEWER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"以下のコードをレビューしてください:{token_hint}\n\n"
                        f"```\n{code}\n```"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        # 予測誤差 → トークン強化 (予測があった場合のみ)
        if prediction and self._codebook is not None:
            err = prediction.evaluate(code)
            reward = 1.0 - err
            for tid in prediction.token_ids:
                self._codebook.record_reward(tid, reward=reward)

        if not result:
            return "（レビューを生成できませんでした。モデルや接続を確認してください）"

        # ライブ解釈層: トークン情報を末尾に添える (人間可読化)
        if prediction and self._interpreter is not None:
            interpreted = self._interpreter.interpret(prediction.token_ids)
            err_repr = f"{prediction.evaluate(code):.2f}"
            result = (
                f"{result}\n\n"
                f"---\n"
                f"[内部通信] {interpreted} / 予測誤差={err_repr}"
            )
        return result
