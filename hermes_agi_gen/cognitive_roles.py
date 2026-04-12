"""8つの専門認知ロール定義。

AGIに必要な認知機能を8つの専門ロールに分割:

1. perceiver   - 入力理解・解釈・コンテキスト把握
2. strategist  - 戦略的計画・ゴール分解
3. executor    - 安全な行動実行・ツール操作
4. critic      - 品質評価・安全確認・改善提案
5. memorist    - 知識管理・記憶統合・検索
6. goal_manager - ゴール優先付け・スケジュール管理
7. innovator   - 創造的解決策・代替案生成
8. ethicist    - 価値整合・倫理的判断・安全確認

各ロールは独立した認知プロセスを持ち、
GlobalWorkspaceを通じて統合された判断を生成する。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import (
    COGNITIVE_ROLE_COMPLEX_PATTERNS,
    COGNITIVE_ROLE_SIMPLE_PATTERNS,
    COGNITIVE_ROLE_SUCCESS_EMA_ALPHA,
    SIMPLE_GOAL_MAX_LEN,
)

# ------------------------------------------------------------------
# ロール別システムプロンプト
# ------------------------------------------------------------------

ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "perceiver": (
        "あなたは入力理解・解釈専門エージェントです。\n"
        "ユーザーの目標・意図・コンテキストを深く理解し、\n"
        "曖昧さを解消して明確なタスク定義を生成します。\n"
        "入力の本質的な意味と背後にある真のニーズを見抜いてください。\n"
        "結果は日本語で簡潔に報告してください。"
    ),
    "strategist": (
        "あなたは戦略的計画専門エージェントです。\n"
        "複雑な目標を達成可能なサブタスクに分解し、\n"
        "依存関係を考慮した実行順序を設計します。\n"
        "複数の戦略オプションを評価し、最善のアプローチを選択します。\n"
        "計画を日本語で体系的に説明してください。"
    ),
    "executor": (
        "あなたはコード実行・ファイル操作専門エージェントです。\n"
        "シェルコマンドの実行、ファイルの読み書き、コードの実行を担当します。\n"
        "インターネットやブラウザは使えません。\n"
        "安全を最優先に、破壊的操作は避けてください。\n"
        "実行結果を日本語で説明してください。"
    ),
    "critic": (
        "あなたは品質評価・改善提案専門エージェントです。\n"
        "他のエージェントの出力を批判的に評価し、\n"
        "品質・正確性・安全性・改善点を分析します。\n"
        "建設的なフィードバックと具体的な改善策を提示してください。\n"
        "評価を日本語で報告してください。"
    ),
    "memorist": (
        "あなたは知識管理・記憶統合専門エージェントです。\n"
        "ローカルファイル・コードを調査して情報を収集し、\n"
        "過去のセッションで得た知識と統合します。\n"
        "関連する情報を検索・整理・要約してください。\n"
        "得られた知識を日本語で簡潔にまとめてください。"
    ),
    "goal_manager": (
        "あなたはゴール優先付け・管理専門エージェントです。\n"
        "複数の目標間の優先順位を評価し、\n"
        "リソース制約と価値基準に基づいて最適な実行順序を決定します。\n"
        "長期的な目標とその達成に向けたロードマップを管理します。\n"
        "分析結果を日本語で報告してください。"
    ),
    "innovator": (
        "あなたは創造的解決策生成専門エージェントです。\n"
        "既存のアプローチにとらわれず、新しい視点から問題を分析します。\n"
        "複数の代替解決策を提案し、それぞれのトレードオフを説明します。\n"
        "既存のコードやシステムを活用した革新的な方法を探してください。\n"
        "アイデアを日本語で提案してください。"
    ),
    "ethicist": (
        "あなたは価値整合・倫理的判断専門エージェントです。\n"
        "すべての行動を安全性・誠実さ・有益性の観点で評価します。\n"
        "潜在的な危険・副作用・倫理的問題を特定し、\n"
        "安全な代替手段を提案します。\n"
        "評価結果を日本語で報告してください。"
    ),

    # 後方互換性のため gen-5 のロールも維持
    "researcher": (
        "あなたはローカルファイル調査専門エージェントです。"
        "シェルコマンド (ls, find, cat, grep など) でファイル・コードを調べ、情報をまとめます。"
        "インターネットやブラウザは使えません。ローカルのファイルのみを対象にしてください。"
        "得られた情報を日本語で簡潔にまとめてください。"
    ),
    "developer": (
        "あなたはコード実行・ファイル操作専門エージェントです。"
        "シェルコマンドの実行、ファイルの読み書き、コードの実行を担当します。"
        "インターネットやブラウザは使えません。"
        "実行結果を日本語で説明してください。"
    ),
}


@dataclass
class CognitiveRole:
    """認知ロールのメタデータ。"""
    name: str
    system_prompt: str
    primary_function: str      # 主要機能の説明
    input_types: List[str]     # 入力として期待するデータ型
    output_types: List[str]    # 出力として生成するデータ型
    preferred_tools: List[str] = field(default_factory=list)  # 優先ツール
    max_iterations: int = 4    # デフォルト最大反復数


# 8つの認知ロールの定義
COGNITIVE_ROLES: Dict[str, CognitiveRole] = {
    "perceiver": CognitiveRole(
        name="perceiver",
        system_prompt=ROLE_SYSTEM_PROMPTS["perceiver"],
        primary_function="入力理解・意図解釈",
        input_types=["user_goal", "context", "constraints"],
        output_types=["clarified_goal", "task_definition", "intent_analysis"],
        preferred_tools=["ANSWER", "READ"],
        max_iterations=2,
    ),
    "strategist": CognitiveRole(
        name="strategist",
        system_prompt=ROLE_SYSTEM_PROMPTS["strategist"],
        primary_function="戦略的計画・ゴール分解",
        input_types=["clarified_goal", "constraints", "past_strategies"],
        output_types=["execution_plan", "subtask_list", "strategy_options"],
        preferred_tools=["PLAN", "ANSWER"],
        max_iterations=3,
    ),
    "executor": CognitiveRole(
        name="executor",
        system_prompt=ROLE_SYSTEM_PROMPTS["executor"],
        primary_function="安全な行動実行",
        input_types=["task", "execution_plan", "tool_list"],
        output_types=["execution_result", "output", "state_change"],
        preferred_tools=["CMD", "READ", "WRITE", "PYTHON"],
        max_iterations=6,
    ),
    "critic": CognitiveRole(
        name="critic",
        system_prompt=ROLE_SYSTEM_PROMPTS["critic"],
        primary_function="品質評価・改善提案",
        input_types=["execution_result", "goal", "criteria"],
        output_types=["quality_report", "improvements", "pass_fail"],
        preferred_tools=["READ", "ANSWER"],
        max_iterations=2,
    ),
    "memorist": CognitiveRole(
        name="memorist",
        system_prompt=ROLE_SYSTEM_PROMPTS["memorist"],
        primary_function="知識収集・記憶統合",
        input_types=["query", "context", "past_knowledge"],
        output_types=["knowledge_summary", "relevant_facts", "retrieved_memories"],
        preferred_tools=["CMD", "READ", "SEARCH"],
        max_iterations=4,
    ),
    "goal_manager": CognitiveRole(
        name="goal_manager",
        system_prompt=ROLE_SYSTEM_PROMPTS["goal_manager"],
        primary_function="ゴール優先付け・管理",
        input_types=["goal_list", "constraints", "value_system"],
        output_types=["prioritized_goals", "roadmap", "next_action"],
        preferred_tools=["ANSWER"],
        max_iterations=2,
    ),
    "innovator": CognitiveRole(
        name="innovator",
        system_prompt=ROLE_SYSTEM_PROMPTS["innovator"],
        primary_function="創造的解決策生成",
        input_types=["problem", "constraints", "failed_approaches"],
        output_types=["alternative_solutions", "creative_approach", "tradeoffs"],
        preferred_tools=["ANSWER", "SEARCH", "PYTHON"],
        max_iterations=3,
    ),
    "ethicist": CognitiveRole(
        name="ethicist",
        system_prompt=ROLE_SYSTEM_PROMPTS["ethicist"],
        primary_function="価値整合・倫理的判断",
        input_types=["action_plan", "value_system", "context"],
        output_types=["ethics_report", "risk_assessment", "safe_alternatives"],
        preferred_tools=["ANSWER"],
        max_iterations=2,
    ),
}


# ------------------------------------------------------------------
# ロール依存関係: どのロールがどのロールの前に来るべきか
# ------------------------------------------------------------------
ROLE_DEPENDENCIES: Dict[str, List[str]] = {
    "strategist": ["perceiver"],       # perceiver が先に意図を明確化
    "executor": ["strategist"],         # strategist が計画を立ててから実行
    "critic": ["executor"],             # executor の結果を評価
    "innovator": ["perceiver"],         # perceiver の理解を踏まえて提案
    "ethicist": ["perceiver"],          # perceiver の意図理解が前提
    "goal_manager": ["perceiver"],      # perceiver の意図理解が前提
}


def _keyword_match(keyword: str, text: str) -> bool:
    """キーワードがテキストにマッチするか判定する。

    英語キーワード（ASCII のみ）は単語境界マッチ、
    日本語キーワードは部分文字列マッチを使う。
    """
    if keyword.isascii():
        return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
    else:
        return keyword in text


# ------------------------------------------------------------------
# ロール成功率フィードバック (EMA)
# ------------------------------------------------------------------

_role_success_rates: Dict[str, float] = {}


def record_role_outcome(roles: List[str], success: bool) -> None:
    """使用されたロールの成功率をEMAで更新する。

    Args:
        roles: 使用されたロール名のリスト
        success: ゴールが成功したかどうか
    """
    alpha = COGNITIVE_ROLE_SUCCESS_EMA_ALPHA
    signal = 1.0 if success else 0.0
    for role in roles:
        old_rate = _role_success_rates.get(role, 0.5)
        _role_success_rates[role] = (1 - alpha) * old_rate + alpha * signal


def get_role_performance() -> Dict[str, float]:
    """各ロールの成功率を返す（イントロスペクション用）。"""
    return dict(_role_success_rates)


def get_role(name: str) -> CognitiveRole:
    """ロール名から CognitiveRole を取得する。存在しない場合は executor を返す。"""
    return COGNITIVE_ROLES.get(name, COGNITIVE_ROLES["executor"])


def select_roles_for_goal(goal: str, context: str = "") -> List[str]:
    """目標の内容から適切なロールの組み合わせを選択する。

    AGIのコア機能: 目標に応じて最適な認知ロール編成を自動決定する。
    単純なクエリには最小限のロール、複雑なタスクには多くのロールを割り当てる。

    Args:
        goal: ユーザーの目標
        context: 追加コンテキスト

    Returns:
        実行順序付きのロール名リスト
    """
    goal_lower = (goal + " " + context).lower()

    # --- 単純クエリの早期判定 ---
    # "ls", "ファイル一覧", "確認して" のような単純な情報取得は executor のみで十分
    simple_patterns = COGNITIVE_ROLE_SIMPLE_PATTERNS
    complex_patterns = COGNITIVE_ROLE_COMPLEX_PATTERNS

    is_simple = (
        len(goal) < SIMPLE_GOAL_MAX_LEN
        or any(p in goal_lower for p in simple_patterns)
    ) and not any(p in goal_lower for p in complex_patterns)

    if is_simple:
        # 単純クエリ: executor のみ
        return ["executor"]

    roles: List[str] = []

    # 複雑なタスクは perceiver から始める
    roles.append("perceiver")

    # 調査系には memorist を追加
    if any(_keyword_match(k, goal_lower) for k in ["調査", "調べ", "find", "search", "探"]):
        roles.append("memorist")

    # 実装・修正・作成系
    if any(_keyword_match(k, goal_lower) for k in ["作成", "実装", "修正", "変更", "create", "implement", "fix", "write", "追加"]):
        roles.append("strategist")
        roles.append("executor")
        roles.append("critic")
    # 分析・評価系
    elif any(_keyword_match(k, goal_lower) for k in ["分析", "評価", "analyze", "evaluate", "レビュー", "review"]):
        roles.append("memorist")
        roles.append("executor")
        roles.append("critic")
    else:
        # デフォルト: 戦略 → 実行
        roles.append("strategist")
        roles.append("executor")

    # 改善・最適化には innovator を追加
    if any(_keyword_match(k, goal_lower) for k in ["改善", "最適化", "optimize", "improve", "革新"]):
        # executor の直前に挿入
        try:
            exec_idx = roles.index("executor")
            roles.insert(exec_idx, "innovator")
        except ValueError:
            roles.append("innovator")

    # 危険な操作には ethicist を追加
    if any(_keyword_match(k, goal_lower) for k in ["削除", "delete", "remove", "書き換え", "overwrite"]):
        # perceiver の直後に挿入
        roles.insert(1, "ethicist")

    # --- 成功率フィードバックによるロール調整 ---
    # 成功率が高いロールを前方へ移動（必須ロールの順序は維持）
    mandatory_roles = {"perceiver", "executor"}
    if _role_success_rates:
        # 高成功率ロールをブースト: 非必須ロールの中で前方に移動
        boosted: List[str] = []
        normal: List[str] = []
        for r in roles:
            rate = _role_success_rates.get(r)
            if rate is not None and rate > 0.7 and r not in mandatory_roles:
                boosted.append(r)
            else:
                normal.append(r)
        if boosted:
            # perceiver の直後にブーストされたロールを挿入
            insert_pos = 1 if normal and normal[0] == "perceiver" else 0
            roles = normal[:insert_pos] + boosted + normal[insert_pos:]

        # 低成功率ロールを除外（必須ロール以外）
        roles = [
            r for r in roles
            if r in mandatory_roles
            or _role_success_rates.get(r, 0.5) >= 0.2
        ]

    # 重複除去しながら順序維持
    seen = set()
    unique_roles = []
    for r in roles:
        if r not in seen:
            seen.add(r)
            unique_roles.append(r)

    return unique_roles


def decompose_into_roles(
    goal: str,
    context: str = "",
    available_roles: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """目標を役割分担されたサブタスクリストに変換する。

    Args:
        goal: ユーザーの目標
        context: 追加コンテキスト
        available_roles: 使用可能なロール（省略時は自動選択）

    Returns:
        [{"role": "...", "task": "..."}, ...] 形式のリスト
    """
    roles = available_roles or select_roles_for_goal(goal, context)

    # ロール依存関係を検証し、必要な前提ロールを挿入する
    validated_roles: List[str] = []
    for role in roles:
        deps = ROLE_DEPENDENCIES.get(role, [])
        for dep in deps:
            if dep not in validated_roles and dep not in roles[:roles.index(role)]:
                validated_roles.append(dep)
        if role not in validated_roles:
            validated_roles.append(role)

    # 重複除去しながら順序維持
    seen: set = set()
    final_roles: List[str] = []
    for r in validated_roles:
        if r not in seen:
            seen.add(r)
            final_roles.append(r)
    roles = final_roles

    subtasks = []

    role_task_map = {
        "perceiver": f"目標「{goal}」の意図・要件を明確化してください",
        "memorist": f"「{goal}」に関連する既存のファイル・コード・知識を調査してください",
        "ethicist": f"「{goal}」の実行計画の安全性・倫理的問題を評価してください",
        "strategist": f"「{goal}」を達成するための実行計画を立案してください",
        "innovator": f"「{goal}」に対する創造的・代替的アプローチを提案してください",
        "executor": f"「{goal}」を実際に実行してください",
        "critic": f"「{goal}」の実行結果を評価し、改善点を報告してください",
        "goal_manager": f"「{goal}」に関連する追加ゴールを特定し、優先順位を付けてください",
    }

    for role in roles:
        if role in role_task_map:
            subtasks.append({
                "role": role,
                "task": role_task_map[role],
            })

    return subtasks
