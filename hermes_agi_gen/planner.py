"""汎用プランナー。ドメインを問わずあらゆるタスクに対応する。

Chain-of-Thought (CoT) 推論: 行動を決定する前に多段階の推論チェーンを生成し、
より深い理解と正確な計画立案を実現する。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .agent_state import AgentState
from .tools import TOOL_CONSTRAINTS, TOOL_DESCRIPTIONS, TOOL_EXAMPLES

if TYPE_CHECKING:
    from .mistral_client import MistralClient

# ------------------------------------------------------------------
# ドメイン別ヒント（LLM プロンプトに注入）
# ------------------------------------------------------------------
_DOMAIN_HINTS: dict[str, str] = {
    "general": "質問・相談には ANSWER: で直接回答する。ファイル操作が必要なら CMD:/READ:、外部情報が必要なら SEARCH: を使う。",
    "coding": "コードを読んで理解してから変更を提案する。テストを確認する。外部ライブラリ情報は SEARCH: で調べる。",
    "research": (
        "URLが含まれる場合は SEARCH: でなく FETCH: でそのURLに直接アクセスする。"
        "「URLからデータ取得 → 処理 → ファイル保存」のような複数ステップのタスクは、"
        "requests/json/os/datetime などを使った単一の PYTHON: スクリプトにまとめて処理できる。"
        "スクリプト内でのLLM呼び出しは不要 — テキスト処理・フィルタはPythonで行う。"
        "ただし PYTHON: でファイル書き込みはしない。ファイル保存は WRITE: を使い、原則リポジトリ配下へ保存する。外部保存先は HERMES_WRITE_ALLOW_DIRS で許可された場合のみ使う。"
    ),
    "writing": "既存の文章・構造を確認してから追記・編集する。参考資料が必要なら SEARCH: を活用する。URLから直接コンテンツ取得するなら FETCH: を使う。時刻指定タスクは SCHEDULE_AT: で登録する。",
    "data": "データファイルの形式・サイズを確認してから処理する。簡単な集計は CALC: で行う。手法に迷ったら SEARCH: で調べる。",
    "ops": "現在の状態を確認してから操作する。破壊的操作は避ける。エラー解決策は SEARCH: で調べる。",
}

# ------------------------------------------------------------------
# CoT推論プロンプト
# ------------------------------------------------------------------
_COT_TEMPLATE = """\
あなたはローカルマシン上で動作する汎用 AGI エージェントのプランナーです。
まず段階的に推論してから、次のアクションを1行で決定してください。

役割: {role}
ドメイン: {domain}
目標: {goal}
{context_section}
ドメインヒント: {domain_hint}

{tool_descriptions}

{tool_examples}

{tool_constraints}

【長期記憶 (過去のセッションで学んだこと)】
{long_term_memories}

【意味的に類似した過去の記憶】
{semantic_memories}

【既知の失敗パターン (これらは避ける)】
{known_failures}

現在の状態:
- 完了済み: {completed}
- 失敗済み: {failed}
- 観測メモ: {observations}
- 世界モデル: {world_model_summary}
- ユーザー制約: {constraints}

=== Chain-of-Thought 推論手順 ===

以下の形式で推論してから最終アクションを決定してください:

<thinking>
1. 現状把握: [今何が分かっているか]
2. 時間指定チェック: [目標に「〜時になったら」「〜日に」など未来時刻の指定があるか? → あれば即 SCHEDULE_AT: を使う]
3. 目標分析: [何を達成する必要があるか]
4. 選択肢検討: [取りうるアクションは何か]
5. リスク評価: [各選択肢の危険性・副作用は何か]
6. 最良の選択: [なぜこのアクションが最善か]
</thinking>
<action>
[ツール形式1行のみ: ANSWER: / SEARCH: / FETCH: / CALC: / CMD: / READ: / WRITE: / PYTHON: / PLAN: / SCHEDULE_AT: / DONE:]
</action>

- DONE: は目標が完全に達成されたときだけ使う
- 複雑なタスク: PLAN: step1 || step2 || step3 で全ステップを一括計画する
- 「〜時になったら」「〜日に」など未来の時刻が指定されている場合は **SCHEDULE_AT:** で即座にスケジュール登録し、次のステップで DONE: を返す。実行は試みない\
"""

# 会話的なクエリを検出するキーワード
_CONVERSATIONAL_KEYWORDS: frozenset[str] = frozenset({
    "答えられますか", "できますか", "何ができ", "何を知", "教えて", "説明して",
    "どう思", "どう考え", "とは何", "とはなん", "とは？", "とは?",
    "あなたは", "あなたに", "あなたの", "できること", "機能", "使い方",
    "こんにちは", "はじめまして", "よろしく", "ありがとう", "お願い",
    "can you", "what can", "are you", "what is", "how do", "help me",
    "hello", "hi ", "please", "thank",
})

_STATIC_ANSWER_GENERAL = (
    "ANSWER: はい、幅広いトピックに対応できます。"
    "コーディング、調査・リサーチ、文章作成、データ分析、一般的な質問など"
    "さまざまなタスクをサポートします。"
    "具体的に何を手伝いましょうか？"
)


def _is_conversational(goal: str) -> bool:
    """ファイル操作やコマンド実行を必要としない会話的クエリかどうかを判定する。"""
    g = goal.lower()
    return any(kw in g for kw in _CONVERSATIONAL_KEYWORDS)


# ------------------------------------------------------------------
# 静的フォールバック: ドメイン別初期プラン
# ------------------------------------------------------------------
_STATIC_BOOTSTRAP: dict[str, list[str]] = {
    "general": [
        "CMD: pwd",
        "CMD: find . -maxdepth 2 -not -path '*/__pycache__/*' | sort | head -60",
        "DONE: 現状を把握しました。",
    ],
    "coding": [
        "CMD: pwd",
        "CMD: find . -maxdepth 2 -not -path '*/__pycache__/*' | sort | head -60",
        "READ: README.md",
        "READ: requirements.txt",
        "DONE: プロジェクト構造と依存関係を確認しました。",
    ],
    "research": [
        "SEARCH: {goal}",
        "DONE: 調査結果をまとめました。",
    ],
    "writing": [
        "CMD: pwd",
        "CMD: find . -name '*.md' -o -name '*.txt' -o -name '*.rst' | head -20",
        "DONE: 文書ファイルを確認しました。",
    ],
    "data": [
        "CMD: pwd",
        "CMD: find . -name '*.csv' -o -name '*.json' -o -name '*.parquet' | head -20",
        "DONE: データファイルを確認しました。",
    ],
    "ops": [
        "CMD: pwd",
        "CMD: env | grep -v SECRET | head -20",
        "DONE: 環境を確認しました。",
    ],
}


def _extract_action_from_cot(response: str) -> Optional[str]:
    """CoT出力から<action>タグ内のアクションを抽出する。"""
    # <action>タグを探す
    action_match = re.search(r'<action>\s*(.+?)\s*</action>', response, re.DOTALL | re.IGNORECASE)
    if action_match:
        line = action_match.group(1).strip().splitlines()[0].strip()
        return line if line else None

    # フォールバック: タグなしで最初の有意な行を探す
    for line in response.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        prefixes = (
            "ANSWER:", "SEARCH:", "FETCH:", "CALC:", "CMD:", "READ:",
            "WRITE:", "PYTHON:", "PLAN:", "SCHEDULE_AT:", "SCHEDULE:", "DONE:",
        )
        if any(line.upper().startswith(p) for p in prefixes):
            return line

    return None


def _extract_thinking(response: str) -> Optional[str]:
    """CoT出力から<thinking>タグ内の推論を抽出する。"""
    match = re.search(r'<thinking>\s*(.+?)\s*</thinking>', response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


class Planner:
    def __init__(
        self,
        llm: Optional[MistralClient] = None,
        role: str = "worker",
    ) -> None:
        self.llm = llm
        self.role = role

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def next_step(self, state: AgentState, repo_root: Path | str | None = None) -> str | None:
        if self.llm:
            return self._llm_next_step(state)
        return self._static_next_step(state)

    # ------------------------------------------------------------------
    # サマリー生成プロンプト
    # ------------------------------------------------------------------

    _SUMMARIZE_PROMPT = """\
以下の情報をもとに、ユーザーの目標に対する最終的なまとめを日本語で作成してください。

目標: {goal}

収集した情報:
{evidence}

簡潔かつ網羅的にまとめてください（200〜400字程度）。
"""

    def _generate_summary_answer(self, state: AgentState) -> str:
        """ワーキングメモリの検索結果や観測を使って最終まとめを ANSWER: 形式で生成する。"""
        assert self.llm is not None
        parts: list[str] = []
        results = state.working_memory.get("last_search_results", [])
        for r in results[:5]:
            if r.get("snippet"):
                parts.append(f"- {r['title']}: {r['snippet']}")
        if state.observations:
            parts.extend(f"- {o}" for o in state.observations[-3:])
        evidence = "\n".join(parts) or "（情報なし）"

        prompt = self._SUMMARIZE_PROMPT.format(goal=state.user_goal, evidence=evidence)
        summary = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return f"ANSWER: {summary}" if summary else "DONE: まとめを生成できませんでした"

    def _llm_next_step(self, state: AgentState) -> str | None:
        if state.is_done:
            return None

        # current_plan にステップが積まれている場合はそれを優先
        if state.current_plan:
            next_step = state.current_plan.pop(0)
            if next_step.upper().startswith("DONE:") and any(
                kw in next_step for kw in ("まとめ", "結論", "要約", "summarize", "summary")
            ):
                return self._generate_summary_answer(state)
            return next_step

        # 長期記憶をプロンプトに注入
        ltm_strategies = state.working_memory.get("ltm_strategies", [])
        ltm_failures = state.working_memory.get("ltm_known_failures", [])
        semantic_memories = state.working_memory.get("ltm_semantic", [])

        ltm_mem_text = "\n".join(
            f"- [{s['outcome']}] {s['strategy']}" for s in ltm_strategies[:3]
        ) or "なし"

        semantic_text = "\n".join(
            f"- {m['key']}: {m['value'][:80]}" for m in semantic_memories[:3]
        ) or "なし"

        ltm_fail_text = "\n".join(
            f"- {f['command_pattern'][:60]} ({f['error_type']}, {f['count']}回失敗)"
            for f in ltm_failures[:5]
        ) or "なし"

        domain = getattr(state, "domain", "general")
        context = getattr(state, "context", "")
        context_section = f"コンテキスト: {context}\n" if context else ""
        domain_hint = _DOMAIN_HINTS.get(domain, _DOMAIN_HINTS["general"])

        # 世界モデルのサマリー
        world_model = getattr(state, "world_model", None)
        if world_model:
            wm_summary = world_model.summary()
        else:
            wm_summary = "なし"

        # few-shot例 (自己改善ループから注入される場合)
        few_shot = state.working_memory.get("few_shot_examples", "")

        prompt = _COT_TEMPLATE.format(
            role=self.role,
            domain=domain,
            goal=state.user_goal,
            context_section=context_section,
            domain_hint=domain_hint,
            tool_descriptions=TOOL_DESCRIPTIONS,
            tool_examples=TOOL_EXAMPLES + ("\n\n【学習済み成功例】\n" + few_shot if few_shot else ""),
            tool_constraints=TOOL_CONSTRAINTS,
            completed=", ".join(state.completed_steps[-5:]) or "なし",
            failed=", ".join(state.failed_steps[-3:]) or "なし",
            observations="; ".join(state.observations[-3:]) or "なし",
            constraints=", ".join(state.constraints) or "なし",
            long_term_memories=ltm_mem_text,
            semantic_memories=semantic_text,
            known_failures=ltm_fail_text,
            world_model_summary=wm_summary,
        )

        assert self.llm is not None
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )

        # CoT推論を観測に記録 (デバッグ・自己改善用)
        thinking = _extract_thinking(response)
        if thinking:
            state.working_memory["last_cot_reasoning"] = thinking[:500]

        # <action>タグからアクションを抽出
        action = _extract_action_from_cot(response)
        if not action:
            return None

        if action.upper().startswith("DONE"):
            state.working_memory["completion_summary"] = action[5:].strip() if ":" in action else ""
            return None

        if action.upper().startswith("PLAN:"):
            return action

        return action

    # ------------------------------------------------------------------
    # 静的プランニング (LLM なし fallback)
    # ------------------------------------------------------------------

    def _static_next_step(self, state: AgentState) -> str | None:
        if not state.current_plan:
            domain = getattr(state, "domain", "general")
            if domain == "general" and _is_conversational(state.user_goal):
                state.current_plan = [_STATIC_ANSWER_GENERAL, "DONE: 回答しました。"]
            elif domain == "research":
                state.current_plan = [
                    f"SEARCH: {state.user_goal}",
                    "DONE: 調査結果をまとめました。",
                ]
            else:
                plan = _STATIC_BOOTSTRAP.get(domain, _STATIC_BOOTSTRAP["general"])
                state.current_plan = [s for s in plan if s not in state.completed_steps]

        if state.current_plan:
            return state.current_plan.pop(0)

        return None
