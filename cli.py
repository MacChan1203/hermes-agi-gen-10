#!/usr/bin/env python3
"""Hermes AI - 汎用インタラクティブ CLI。チャットとエージェントを統合。

新AGI機能:
  /orch    - 階層的マルチエージェントオーケストレーター
  /tools   - カスタムツールの一覧表示・登録
  /goals   - 自律ゴールキューの表示
  /world   - 世界モデルの状態表示
  /status  - AGIシステム全体のステータス
  /improve - 自己改善レポートの表示

使い方:
  python3 cli.py                    # 自動検出 (Groq→Mistral→Ollama)
  python3 cli.py --model groq       # Groq を明示指定
  python3 cli.py --model qwen3      # Ollama のモデルを指定
  python3 cli.py --model mistral    # Mistral API を指定
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from hermes_agi_gen import AgentOrchestrator, AgentState, HermesAgentV9
from hermes_agi_gen.code_agents import CodeGeneratorAgent, CodeReviewerAgent
from hermes_agi_gen.daemon import HermesDaemon
from hermes_agi_gen.hermes_constants import DOMAIN_CONFIG
from hermes_agi_gen.mistral_client import MistralClient
from hermes_agi_gen.self_improvement import SelfImprovementEngine
from hermes_agi_gen.self_modifier import SelfModifier, _SAFE_MODIFY_TARGETS
from hermes_agi_gen.state_store import SessionDB
from hermes_agi_gen.tool_registry import ToolRegistry
from hermes_agi_gen.world_model import WorldModel

console = Console()

# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------

_GENERAL_SYSTEM = """\
あなたは Hermes AGI — 自律型AIエージェントシステムです。
あなたは **できる** ことに焦点を当てて回答してください。

## 実際の能力 (Hermes AGI として)

### ✅ 能動的な行動
- CMD: でシェルコマンドを実行できます
- PYTHON: でPythonコードを直接実行できます
- WRITE: でファイルを作成・編集できます
- READ: でファイルを読み込めます
- PLAN: で複雑なタスクを複数ステップに分解できます

### ✅ 外部システムへのアクセス
- SEARCH: でウェブ検索（DuckDuckGo）ができます
- CMD: curl/wget で任意のAPIを呼び出せます
- ローカルファイルシステムへの読み書きができます

### ✅ 継続的な実行 (AGI機能)
- GoalQueueがLTM (SQLite) に永続化されており、再起動後も目標を継続できます
- /daemon start でバックグラウンドAGIが自律的にGoalQueueを処理します
- セッションをまたいで経験・失敗・戦略を記憶します

### ✅ スケジュール実行
- /daemon start でバックグラウンドデーモンが起動し自律稼働します
- launchd (macOS) でシステム起動時に自動起動できます
- GoalQueueに目標を積めばデーモンが自動的に処理します

## 動作モード
- **チャット**: 直接回答・相談・説明
- **エージェント (/run)**: ローカル環境でタスクを実行
- **オーケストレーター (/orch)**: 複数エージェントで並列実行
- **デーモン (/daemon start)**: 24時間自律稼働

ユーザーの質問・依頼に対して、上記の能力を前提に「できる」として回答してください。
「できません」という回答は避け、どの機能を使えば実現できるか説明してください。
"""

_INTENT_SYSTEM = """\
ユーザーのメッセージが「タスク実行」か「直接回答」かを判定してください。

タスク実行 (type=task):
  ファイル操作・コード実行・ディレクトリ調査・データ処理・システム操作など、
  ローカル環境で実際に何かを行う必要があるもの。

直接回答 (type=chat):
  挨拶・雑談・自己紹介・能力質問・説明・意見・アドバイスなど、
  テキストのみで応答できるもの。
  「何ができますか」「あなたは〜ですか」「〜とは何ですか」は必ず chat。

ドメイン (domain): general / coding / research / writing / data / ops

以下の JSON のみを返してください（説明不要）:
{"type": "chat", "domain": "general"}
"""

# ---------------------------------------------------------------------------
# ヘルプテキスト
# ---------------------------------------------------------------------------

_HELP = """\
## 使えるコマンド

| コマンド | 説明 |
|---|---|
| `<メッセージ>` | 自由に話しかける（チャット or 自動でエージェント起動） |
| `/run <目標>` | エージェントモードで明示的にタスクを実行 |
| `/orch <目標>` | 階層的マルチエージェントで複雑なタスクを並列実行 |
| `/reflect [対象]` | 自己診断: ソースを読んで改善提案を生成 |
| `/apply` | 最後の `/reflect` 提案をコードに適用 |
| `/self-modify [対象]` | 自律的コード修正: 提案→テスト→適用のサイクル |
| `/daemon start` | AGI自律デーモンをバックグラウンドで起動 |
| `/daemon stop` | デーモンを停止 |
| `/daemon status` | デーモンの状態を確認 |
| `/daemon log` | デーモンのログを表示 |
| `/generate <説明>` | 自然言語からコードを生成 |
| `/review` | コードのレビュー（貼り付けモード） |
| `/tools [list/add]` | カスタムツールの一覧表示・登録 |
| `/goals` | 自律ゴールキューを表示 |
| `/world` | 世界モデルの状態を表示 |
| `/status` | AGIシステム全体のステータスを表示 |
| `/improve` | 自己改善レポートを表示 |
| `/perf` | パフォーマンス履歴を表示 |
| `/clear` | 会話履歴をリセット |
| `/provider` | 現在の LLM プロバイダーを表示 |
| `/help` | このヘルプを表示 |
| `/quit` | 終了 |

**ヒント**: `/reflect` の対象は `planner` `executor` `reviewer` `runner` `memory`
`meta` `world` `registry` `improver` `cli` など。省略するとコアファイル全体を診断します。

**AGIデーモン**: `/daemon start` でバックグラウンドAGIが起動し、自律的にGoalQueueを処理します。
GoalQueueはセッションをまたいでLTMに永続化されます。
"""

# 短縮名 → 実ファイルパスのマッピング
_REFLECT_TARGETS: dict[str, str] = {
    "planner":   "hermes_agi_gen/planner.py",
    "executor":  "hermes_agi_gen/executor.py",
    "reviewer":  "hermes_agi_gen/reviewer.py",
    "runner":    "hermes_agi_gen/agent_runner.py",
    "memory":    "hermes_agi_gen/long_term_memory.py",
    "meta":      "hermes_agi_gen/meta_cognition.py",
    "tools":     "hermes_agi_gen/tools.py",
    "search":    "hermes_agi_gen/web_search.py",
    "state":     "hermes_agi_gen/agent_state.py",
    "world":     "hermes_agi_gen/world_model.py",
    "registry":  "hermes_agi_gen/tool_registry.py",
    "improver":  "hermes_agi_gen/self_improvement.py",
    "planner_h": "hermes_agi_gen/hierarchical_planner.py",
    "orch":      "hermes_agi_gen/orchestrator.py",
    "cli":       "cli.py",
}

_REFLECT_CORE_FILES = [
    "hermes_agi_gen/planner.py",
    "hermes_agi_gen/executor.py",
    "hermes_agi_gen/reviewer.py",
    "hermes_agi_gen/agent_runner.py",
]

_SELF_REFLECT_CONTEXT = """\
あなたは Hermes AI 自身です。これはあなた自身のローカルのソースコードです。
ファイルは READ: ツールで読み込んでください（SEARCH: は不要）。
以下の観点で自己診断を行い、具体的な改善提案をしてください:
1. バグ・エラーになりうる箇所 (コード行を具体的に指摘)
2. パフォーマンス・効率の改善機会
3. 設計・保守性の改善案
4. 未実装・TODO・将来拡張のアイデア
各項目に修正案のコードスニペットを含めてください。
"""

_SELF_APPLY_CONTEXT = """\
あなたは Hermes AI 自身のコードを改善するエンジニアです。
提示された改善提案を実際のコードに適用してください。
手順:
1. READ: で対象ファイルを読む
2. 改善内容を特定する
3. WRITE: で修正済みファイルを書き込む
破壊的な変更は避け、既存の動作を維持しながら改善してください。
"""

_TASK_ACTION_VERBS: frozenset[str] = frozenset({
    "実行して", "起動して", "インストールして", "ビルドして", "デプロイして",
    "テストして", "チェックして", "確認して", "調べて", "列挙して", "探して",
    "読んで", "開いて", "作って", "書いて", "修正して", "変更して", "削除して",
    "移動して", "コピーして", "検索して", "分析して", "比較して",
    "run ", "execute ", "check ", "find ", "search ", "read ",
})

_CHAT_KEYWORDS: frozenset[str] = frozenset({
    "あなたは何", "あなたに何", "何ができますか", "できますか", "できること",
    "何者ですか", "どんなことができ",
    "こんにちは", "こんばんは", "おはようございます", "はじめまして", "よろしく",
    "ありがとう", "お疲れ様", "お疲れさま",
    "とは何ですか", "とはなんですか", "とは？", "とは?",
    "どう思いますか", "どう考えますか", "ご意見",
    "what can you do", "can you help", "are you", "what are you",
    "hello", "hi there",
})


def _is_likely_chat(message: str) -> bool:
    m = message.lower()
    if any(v in m for v in _TASK_ACTION_VERBS):
        return False
    return any(kw in m for kw in _CHAT_KEYWORDS)


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _provider_label(llm: MistralClient) -> str:
    url = llm.base_url
    if "anthropic" in url:
        return f"[bold green]Claude[/bold green] ({llm.model})"
    if "groq" in url:
        return f"[bold yellow]Groq[/bold yellow] ({llm.model})"
    if "mistral" in url:
        return f"[bold blue]Mistral[/bold blue] ({llm.model})"
    if "openrouter" in url:
        return f"[bold magenta]OpenRouter[/bold magenta] ({llm.model})"
    return f"[bold white]Ollama[/bold white] ({llm.model})"


def _collect_code() -> str:
    console.print("[dim]コードを貼り付けてください。終わったら新しい行に [bold]END[/bold] と入力してください。[/dim]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _display(result: str, title: str, border: str = "green") -> None:
    console.print(Panel(Markdown(result), title=title, border_style=border))


def _has_action_verb(message: str) -> bool:
    return any(v in message.lower() for v in _TASK_ACTION_VERBS)


def _classify_intent(llm: MistralClient, message: str) -> Tuple[str, str]:
    if _is_likely_chat(message):
        return "chat", "general"

    result = llm.chat_json(
        [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    if isinstance(result, dict):
        intent_type = result.get("type", "chat")
        domain = result.get("domain", "general")
        if domain not in set(DOMAIN_CONFIG.keys()):
            domain = "general"
        if intent_type == "chat" and _has_action_verb(message):
            intent_type = "task"
        if intent_type in {"task", "chat"}:
            return intent_type, domain

    return ("task", "general") if _has_action_verb(message) else ("chat", "general")


def _run_agent(
    llm: MistralClient,
    goal: str,
    domain: str,
    context: str = "",
    max_iterations: int = 8,
    world_model: Optional[WorldModel] = None,
) -> Tuple[str, AgentState]:
    """HermesAgentV9 を起動してタスクを実行し、(サマリー, 最終state) を返す。"""
    console.print(Rule(
        f"[bold yellow]エージェントモード[/bold yellow]  domain=[cyan]{domain}[/cyan]",
        style="yellow",
    ))

    agent = HermesAgentV9(
        repo_root=Path("."),
        model=llm.model,
        max_iterations=max_iterations,
        llm=llm,
    )

    cfg = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG["general"])
    state = AgentState(
        user_goal=goal,
        domain=domain,
        context=context,
        success_criteria=cfg["success_criteria"],
        constraints=cfg["constraints"],
        max_iterations=max_iterations,
        world_model=world_model,  # 世界モデルを引き継ぐ
    )

    final_state = agent.run(state)

    console.print(Rule(style="yellow"))

    # 信頼度警告を表示
    warnings = final_state.working_memory.get("confidence_warnings", [])
    for w in warnings:
        console.print(f"[bold red]{w}[/bold red]")

    summary = final_state.working_memory.get("completion_summary", "")
    if not summary and final_state.observations:
        summary = final_state.observations[-1]

    if summary:
        _display(summary, title="エージェント完了", border="yellow")

    # 世界モデルの更新を表示
    if final_state.world_model:
        wm_summary = final_state.world_model.summary()
        if wm_summary != "未初期化":
            console.print(f"[dim]🌍 世界モデル: {wm_summary}[/dim]")

    # CoT推論を表示 (デバッグ用)
    cot = final_state.working_memory.get("last_cot_reasoning")
    if cot:
        console.print(f"[dim]💭 最後の推論: {cot[:100]}...[/dim]")

    # 自律ゴール提案を表示
    goal_queue = final_state.working_memory.get("goal_queue", [])
    if goal_queue:
        console.print(f"[dim]🎯 自律ゴールキュー: {len(goal_queue)}件待機中[/dim]")
        best = goal_queue[0]
        console.print(f"[dim]   最優先: [{best['source']}] {best['goal']}[/dim]")
    elif final_state.suggested_next_goal:
        console.print(f"[dim]🎯 次の推奨ゴール: {final_state.suggested_next_goal}[/dim]")

    return summary, final_state


# ---------------------------------------------------------------------------
# AGI機能: ツールレジストリ表示・登録
# ---------------------------------------------------------------------------

def _cmd_tools(registry: ToolRegistry, args: str) -> None:
    """カスタムツールの一覧表示または登録。"""
    subcmd = args.strip().lower()

    if subcmd == "add":
        console.print("[cyan]カスタムツールを登録します。[/cyan]")
        name = Prompt.ask("ツール名 (英数字・アンダースコア)")
        prefix = Prompt.ask("呼び出しプレフィックス (大文字英字、例: SCREENSHOT)").upper()
        description = Prompt.ask("ツールの説明 (日本語)")
        console.print("[dim]Pythonコードを入力してください。`def main(args: str) -> str:` の形式。END で終了。[/dim]")
        code = _collect_code()

        if not all([name, prefix, description, code]):
            console.print("[red]入力が不完全です。[/red]")
            return

        ok = registry.register(name=name, description=description, code=code, invocation_prefix=prefix)
        if ok:
            console.print(f"[green]ツール '{name}' を登録しました。{prefix}: <args> で呼び出せます。[/green]")
        else:
            console.print("[red]登録に失敗しました。コードに構文エラーがある可能性があります。[/red]")
        return

    # デフォルト: 一覧表示
    tools = registry.list_tools()
    if not tools:
        console.print("[dim]登録済みのカスタムツールはありません。[/dim]")
        console.print("[dim]/tools add で新しいツールを登録できます。[/dim]")
        return

    table = Table(title="カスタムツール一覧", border_style="cyan")
    table.add_column("プレフィックス", style="cyan bold")
    table.add_column("名前")
    table.add_column("説明")
    table.add_column("使用回数", justify="right")
    table.add_column("成功率", justify="right")

    for t in tools:
        table.add_row(
            f"{t['invocation_prefix']}:",
            t["name"],
            t["description"],
            str(t["use_count"]),
            f"{t['success_rate']:.0%}",
        )
    console.print(table)
    console.print(f"[dim]/tools add でツールを追加できます。[/dim]")


# ---------------------------------------------------------------------------
# AGI機能: ゴールキュー表示
# ---------------------------------------------------------------------------

def _cmd_goals(last_state: Optional[AgentState]) -> None:
    """自律ゴールキューを表示する。"""
    if last_state is None:
        console.print("[dim]まだエージェントを実行していません。/run または /orch でタスクを実行してください。[/dim]")
        return

    goal_queue = last_state.working_memory.get("goal_queue", [])
    if not goal_queue:
        console.print("[dim]ゴールキューは空です。[/dim]")
        return

    table = Table(title="自律ゴールキュー", border_style="yellow")
    table.add_column("優先度", justify="right", style="yellow bold")
    table.add_column("ソース", style="dim")
    table.add_column("ゴール")

    for g in goal_queue:
        table.add_row(f"{g['score']:.2f}", g["source"], g["goal"])
    console.print(table)


# ---------------------------------------------------------------------------
# AGI機能: 世界モデル表示
# ---------------------------------------------------------------------------

def _cmd_world(last_state: Optional[AgentState]) -> None:
    """世界モデルの状態を表示する。"""
    if last_state is None or last_state.world_model is None:
        console.print("[dim]世界モデルはまだ初期化されていません。/run でタスクを実行してください。[/dim]")
        return

    wm = last_state.world_model
    console.print(Panel(
        f"[bold]サマリー:[/bold] {wm.summary()}\n"
        f"[bold]インストール済みパッケージ:[/bold] {len(wm.installed_packages)}個確認済み\n"
        f"[bold]因果関係:[/bold] {len(wm.causal_graph)}件記録済み",
        title="🌍 世界モデル",
        border_style="blue",
    ))

    effects = wm.get_recent_effects(limit=5)
    if effects:
        table = Table(title="最近の因果関係", border_style="blue")
        table.add_column("アクション", style="cyan")
        table.add_column("結果")
        for action, effect in effects:
            table.add_row(action[:60], effect[:60])
        console.print(table)

    if wm.installed_packages:
        console.print(f"[dim]パッケージ (先頭10): {', '.join(wm.installed_packages[:10])}[/dim]")


# ---------------------------------------------------------------------------
# AGI機能: 自己改善レポート
# ---------------------------------------------------------------------------

def _cmd_improve(improver: SelfImprovementEngine, domain: str = "general") -> None:
    """自己改善レポートを表示する。"""
    examples = improver.get_best_examples(domain=domain, limit=5)
    anti = improver.get_anti_patterns(domain=domain, limit=5)

    if not examples and not anti:
        console.print("[dim]まだ自己改善データがありません。/run でタスクを実行すると蓄積されます。[/dim]")
        return

    if examples:
        table = Table(title="学習済みfew-shot例 (良いパターン)", border_style="green")
        table.add_column("ゴールパターン", style="cyan")
        table.add_column("有効なアクション")
        table.add_column("品質", justify="right")

        for e in examples:
            table.add_row(
                e["goal_pattern"][:40],
                e["good_action"][:50],
                f"{e['quality_score']:.0%}",
            )
        console.print(table)

    if anti:
        table = Table(title="学習済みanti-pattern (避けるべきパターン)", border_style="red")
        table.add_column("避けるべきアクション", style="red")
        table.add_column("エラー種別")
        table.add_column("教訓")
        table.add_column("頻度", justify="right")

        for a in anti:
            table.add_row(
                a["bad_action"][:40],
                a["error_type"],
                a["lesson"][:50],
                str(a["frequency"]),
            )
        console.print(table)

    active_prompt = improver.get_active_few_shot_prompt(domain)
    if active_prompt:
        console.print(Panel(active_prompt, title="現在のfew-shotプロンプト", border_style="dim"))


# ---------------------------------------------------------------------------
# AGI機能: AGIシステム全体のステータス
# ---------------------------------------------------------------------------

def _cmd_daemon(subcmd: str) -> None:
    """デーモンの起動・停止・状態確認・ログ表示。"""
    subcmd = subcmd.strip().lower()

    if subcmd == "start":
        status = HermesDaemon.get_status()
        if status["running"]:
            console.print(f"[yellow]デーモンはすでに起動中です (PID={status['pid']})。[/yellow]")
            return
        import subprocess as _sp
        import sys as _sys
        proc = _sp.Popen(
            [_sys.executable, "-m", "hermes_agi_gen.daemon"],
            cwd=str(Path(".")),
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            start_new_session=True,
        )
        console.print(f"[green]AGI自律デーモンを起動しました (PID={proc.pid})。[/green]")
        console.print(f"[dim]ログ: ~/.hermes/daemon.log | /daemon status で確認[/dim]")

    elif subcmd == "stop":
        if HermesDaemon.stop_daemon():
            console.print("[yellow]デーモンに停止シグナルを送りました。[/yellow]")
        else:
            console.print("[dim]デーモンは起動していません。[/dim]")

    elif subcmd == "status":
        status = HermesDaemon.get_status()
        if not status["running"]:
            console.print("[dim]デーモンは起動していません。[/dim]")
            return
        hb = status.get("heartbeat") or {}
        import time as _time
        age = _time.time() - hb.get("timestamp", 0)
        console.print(Panel(
            f"[bold green]稼働中[/bold green] (PID={status['pid']})\n"
            f"最終ハートビート: {age:.0f}秒前\n"
            f"GoalQueue: {hb.get('queue_size', '?')}件\n"
            f"本日の使用: {hb.get('daily_used', '?')}/{hb.get('daily_max', '?')}ゴール",
            title="🤖 AGIデーモン状態",
            border_style="green",
        ))

    elif subcmd == "log":
        log = HermesDaemon.get_log(lines=30)
        console.print(Panel(log, title="AGIデーモンログ (直近30行)", border_style="dim"))

    else:
        console.print("[dim]使い方: /daemon start|stop|status|log[/dim]")


def _cmd_self_modify(
    llm: MistralClient,
    target: str,
    last_reflection: Dict[str, str],
) -> None:
    """自律的コード修正サイクル: 提案 → テスト → 適用/ロールバック。"""
    modifier = SelfModifier(llm=llm, repo_root=Path("."))

    # 対象ファイルを決定
    if target in _REFLECT_TARGETS:
        file_path = _REFLECT_TARGETS[target]
    elif target in _SAFE_MODIFY_TARGETS:
        file_path = target
    elif last_reflection.get("file"):
        file_path = last_reflection["file"]
        if "," in file_path:
            # 複数ファイルの場合は最初の1つ
            file_path = file_path.split(",")[0].strip()
    else:
        console.print(
            "[yellow]対象を指定してください。例: /self-modify planner[/yellow]\n"
            f"[dim]修正可能ファイル: {', '.join(_SAFE_MODIFY_TARGETS)}[/dim]"
        )
        return

    if file_path not in _SAFE_MODIFY_TARGETS:
        console.print(f"[red]{file_path} は修正が許可されていません。[/red]")
        console.print(f"[dim]許可されているファイル: {', '.join(sorted(_SAFE_MODIFY_TARGETS))}[/dim]")
        return

    analysis = last_reflection.get("suggestion", "コードを改善してください。")

    console.print(f"[magenta]自律コード修正モード[/magenta] — 対象: [bold]{file_path}[/bold]")
    console.print("[dim]LLMに改善案を生成させています...[/dim]")

    with console.status("[cyan]改善案を生成中...[/cyan]"):
        patch = modifier.propose_change(file_path, analysis)

    if patch is None:
        console.print("[yellow]改善案が見つかりませんでした。/reflect で先に分析してください。[/yellow]")
        return

    # パッチ内容を表示して確認
    console.print(Panel(
        f"[bold]理由:[/bold] {patch.rationale}\n"
        f"[bold]変更数:[/bold] {len(patch.changes)}件\n"
        f"[bold]リスク:[/bold] {patch.risk_level}\n"
        f"[bold]期待効果:[/bold] {patch.expected_benefit}",
        title="提案されたパッチ",
        border_style="cyan",
    ))

    for i, change in enumerate(patch.changes, 1):
        console.print(f"[dim]変更 {i}: {change.description}[/dim]")

    confirm = Prompt.ask("このパッチを適用してテストしますか？", choices=["y", "n"], default="n")
    if confirm != "y":
        console.print("[dim]キャンセルしました。[/dim]")
        return

    with console.status("[yellow]パッチを適用してテストを実行中...[/yellow]"):
        success = modifier.validate_and_commit(patch)

    if success:
        console.print(f"[green]✓ パッチが適用されました！テストが通過しました。[/green]")
        console.print(f"[dim]変更ファイル: {file_path}[/dim]")
    else:
        console.print("[red]✗ テストが失敗しました。変更はロールバックされました。[/red]")

    # 履歴を表示
    history_data = modifier.get_patch_history(limit=5)
    if history_data:
        success_rate = modifier.get_success_rate()
        console.print(f"[dim]修正成功率: {success_rate:.0%} ({len(history_data)}件の履歴)[/dim]")


def _cmd_perf(improver: SelfImprovementEngine) -> None:
    """パフォーマンス履歴を表示する。"""
    history = improver.get_performance_history(limit=10)
    if not history:
        console.print("[dim]パフォーマンス履歴がありません。/run でタスクを実行すると記録されます。[/dim]")
        return

    table = Table(title="セッションパフォーマンス履歴", border_style="cyan")
    table.add_column("ドメイン", style="cyan")
    table.add_column("ゴール")
    table.add_column("スコア", justify="right", style="green")

    import time as _time
    from datetime import datetime
    for h in history:
        score = h["score"]
        color = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")
        table.add_row(
            h["domain"],
            h["goal"][:50],
            f"[{color}]{score:.0%}[/{color}]",
        )
    console.print(table)

    # 傾向を表示
    for domain in ("general", "coding", "research"):
        trend = improver.get_performance_trend(domain=domain, window=5)
        color = "green" if trend >= 0.7 else ("yellow" if trend >= 0.4 else "red")
        console.print(f"[dim]{domain} 直近5回の平均: [{color}]{trend:.0%}[/{color}][/dim]")


def _cmd_status(
    llm: MistralClient,
    registry: ToolRegistry,
    improver: SelfImprovementEngine,
    last_state: Optional[AgentState],
) -> None:
    """AGIシステム全体のステータスを表示する。"""
    tools = registry.list_tools()
    examples = improver.get_best_examples(limit=3)
    anti = improver.get_anti_patterns(limit=3)

    wm_summary = "未初期化"
    goals_count = 0
    if last_state:
        if last_state.world_model:
            wm_summary = last_state.world_model.summary()
        goals_count = len(last_state.working_memory.get("goal_queue", []))

    status_text = (
        f"[bold cyan]LLM[/bold cyan]: {_provider_label(llm)}\n"
        f"[bold cyan]カスタムツール[/bold cyan]: {len(tools)}個登録済み\n"
        f"[bold cyan]学習済みパターン[/bold cyan]: 良い例 {len(examples)}件 / 悪い例 {len(anti)}件\n"
        f"[bold cyan]世界モデル[/bold cyan]: {wm_summary}\n"
        f"[bold cyan]自律ゴールキュー[/bold cyan]: {goals_count}件待機中\n"
    )

    if last_state:
        status_text += (
            f"[bold cyan]最後のゴール[/bold cyan]: {last_state.user_goal[:60]}\n"
            f"[bold cyan]完了ステップ[/bold cyan]: {len(last_state.completed_steps)}件 / "
            f"失敗: {len(last_state.failed_steps)}件\n"
        )

    console.print(Panel(status_text, title="🤖 AGIシステムステータス", border_style="cyan"))

    if tools:
        console.print(f"[dim]ツール: {', '.join(t['invocation_prefix'] + ':' for t in tools[:5])}[/dim]")


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hermes AI インタラクティブ CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python3 cli.py                  # 自動検出 (Groq→Mistral→Ollama)\n"
            "  python3 cli.py --model groq     # Groq API を使用 (GROQ_API_KEY 必須)\n"
            "  python3 cli.py --model qwen3    # Ollama の qwen3 を使用\n"
            "  python3 cli.py --model mistral  # Mistral API を使用 (MISTRAL_API_KEY 必須)\n"
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="使用するモデル。'groq'/'mistral' はバックエンド指定、それ以外はOllamaモデル名",
    )
    parser.add_argument(
        "--max-turns", "-n",
        type=int,
        default=8,
        help="エージェントの最大イテレーション数 (デフォルト: 8)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="AGI自律デーモンモードで起動 (フォアグラウンドで継続実行)",
    )
    return parser.parse_args()


def _build_llm(model: Optional[str]) -> Optional[MistralClient]:
    """モデル指定からMistralClientを構築する。エラー時はNoneを返す。"""
    try:
        if model is None:
            return MistralClient()
        return MistralClient(model=model)
    except ValueError as e:
        console.print(f"\n[bold red][設定エラー][/bold red] {e}")
        return None


def main() -> None:
    args = _parse_args()

    llm = _build_llm(args.model)
    if llm is None:
        sys.exit(1)

    # デーモンモード
    if args.daemon:
        console.print(Panel(
            f"[bold cyan]Hermes AGI デーモンモード[/bold cyan]\n"
            f"LLM: {_provider_label(llm)}\n"
            "[dim]GoalQueueを継続処理します。Ctrl+C で停止。[/dim]",
            border_style="cyan",
        ))
        daemon = HermesDaemon(llm=llm)
        daemon.run_forever()
        return

    # fast_llm: Groq使用時は軽量モデル、それ以外は同じモデル
    fast_llm = MistralClient.fast() if args.model in (None, "groq") else llm

    db = SessionDB()
    generator = CodeGeneratorAgent(llm=llm, session_db=db)
    reviewer_agent = CodeReviewerAgent(llm=llm, session_db=db)

    # AGIコンポーネント
    tool_registry = ToolRegistry()
    self_improver = SelfImprovementEngine(llm=llm)
    shared_world_model = WorldModel()  # セッション間で世界モデルを引き継ぐ

    history: List[Dict[str, str]] = []
    last_reflection: Dict[str, str] = {}
    last_state: Optional[AgentState] = None

    fast_label = (
        f" / fast=[dim]{fast_llm.model}[/dim]"
        if fast_llm.model != llm.model else ""
    )
    console.print(Panel(
        f"[bold cyan]Hermes AI[/bold cyan]  [dim]AGI Edition[/dim]\n"
        f"LLM: {_provider_label(llm)}{fast_label}\n"
        f"[dim]ツール: {len(tool_registry.list_tools())}個 | "
        f"学習パターン: {len(self_improver.get_best_examples())}件 | "
        f"最大イテレーション: {args.max_turns}[/dim]\n"
        f"[dim]メッセージを入力 → チャット or 自動エージェント / /help でコマンド一覧 / /quit で終了[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            raw = Prompt.ask("[bold green]hermes[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]終了します。[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue

        # --- 終了 ---
        if raw in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]終了します。[/dim]")
            break

        # --- ヘルプ ---
        elif raw == "/help":
            console.print(Markdown(_HELP))

        # --- 会話履歴リセット ---
        elif raw == "/clear":
            history.clear()
            console.print("[dim]会話履歴をリセットしました。[/dim]")

        # --- プロバイダー表示 ---
        elif raw == "/provider":
            console.print(f"スマート: {_provider_label(llm)}")
            console.print(f"高速:     {_provider_label(fast_llm)}")
            console.print(f"[dim]DB  : {db.db_path}[/dim]")

        # --- AGIステータス ---
        elif raw == "/status":
            _cmd_status(llm, tool_registry, self_improver, last_state)

        # --- カスタムツール ---
        elif raw.startswith("/tools"):
            args = raw[6:].strip()
            _cmd_tools(tool_registry, args)

        # --- デーモン制御 ---
        elif raw.startswith("/daemon"):
            subcmd = raw[7:].strip()
            _cmd_daemon(subcmd)

        # --- 自律コード修正 ---
        elif raw.startswith("/self-modify"):
            target = raw[12:].strip().lower()
            _cmd_self_modify(llm, target, last_reflection)

        # --- パフォーマンス履歴 ---
        elif raw == "/perf":
            _cmd_perf(self_improver)

        # --- 自律ゴールキュー ---
        elif raw == "/goals":
            _cmd_goals(last_state)

        # --- 世界モデル ---
        elif raw == "/world":
            _cmd_world(last_state)

        # --- 自己改善レポート ---
        elif raw == "/improve":
            domain = last_state.domain if last_state else "general"
            _cmd_improve(self_improver, domain)

        # --- 明示的なエージェント起動 ---
        elif raw.startswith("/run"):
            goal = raw[4:].strip()
            if not goal:
                console.print("[red]使い方: /run <実行したい目標・タスク>[/red]")
                continue
            with console.status("[yellow]ドメインを判定中...[/yellow]"):
                _, domain = _classify_intent(fast_llm, goal)
            _, last_state = _run_agent(llm, goal, domain, world_model=shared_world_model,
                                       max_iterations=args.max_turns)
            if last_state.world_model:
                shared_world_model = last_state.world_model  # 世界モデルを更新

        # --- 階層的マルチエージェント ---
        elif raw.startswith("/orch"):
            goal = raw[5:].strip()
            if not goal:
                console.print("[red]使い方: /orch <複雑な目標・タスク>[/red]")
                continue
            console.print(Rule(
                "[bold magenta]オーケストレーターモード (階層的並列実行)[/bold magenta]",
                style="magenta",
            ))
            with console.status("[magenta]階層的ゴールツリーを生成・実行中...[/magenta]"):
                orch = AgentOrchestrator(llm=llm, use_hierarchical=True)
                result = orch.run(goal)
            console.print(Rule(style="magenta"))
            _display(result, title="オーケストレーター完了", border="magenta")

        # --- 自己診断・改善提案 ---
        elif raw.startswith("/reflect"):
            target = raw[8:].strip().lower()
            if target in _REFLECT_TARGETS:
                file_path = _REFLECT_TARGETS[target]
                files_desc = file_path
                goal = (
                    f"READ: {file_path} でファイルを読み込み、自己診断してください。"
                    f"バグ・パフォーマンス問題・設計改善・未実装機能の観点で "
                    f"具体的なコードスニペット付きの改善提案を日本語でまとめてください。"
                    f"SEARCH: は使わず、READ: でローカルファイルを直接読んでください。"
                )
            else:
                if target and target not in _REFLECT_TARGETS:
                    console.print(
                        f"[yellow]対象 '{target}' は不明です。コアファイル全体を診断します。[/yellow]"
                    )
                files_desc = ", ".join(_REFLECT_CORE_FILES)
                read_steps = " || ".join(f"READ: {f}" for f in _REFLECT_CORE_FILES)
                goal = (
                    f"PLAN: {read_steps} || ANSWER: まとめ を使って "
                    f"コアファイルを順に読んで自己診断してください。\n"
                    f"バグ・パフォーマンス問題・設計改善・未実装機能の観点で "
                    f"優先度付きの改善提案を日本語でまとめてください。"
                    f"SEARCH: は使わず READ: でローカルファイルを直接読んでください。"
                )
            console.print(f"[magenta]自己診断モード[/magenta] — 対象: [bold]{files_desc}[/bold]")
            suggestion, last_state = _run_agent(
                llm, goal, "coding",
                context=_SELF_REFLECT_CONTEXT,
                max_iterations=12,
                world_model=shared_world_model,
            )
            if suggestion:
                last_reflection["file"] = files_desc
                last_reflection["suggestion"] = suggestion
            if last_state.world_model:
                shared_world_model = last_state.world_model

        # --- 改善提案を適用 ---
        elif raw.startswith("/apply"):
            if not last_reflection:
                console.print("[yellow]まず /reflect を実行してください。[/yellow]")
                continue
            detail = raw[6:].strip()
            files = last_reflection.get("file", "（不明）")
            suggestion = last_reflection.get("suggestion", "")
            apply_goal = (
                f"以下の改善提案を {files} に適用してください。\n\n"
                f"改善提案:\n{suggestion[:800]}\n\n"
                + (f"特に適用したい改善: {detail}\n" if detail else "")
                + "既存の動作を維持しながら改善を適用し、WRITE: でファイルを更新してください。"
            )
            console.print(f"[magenta]自己改善モード[/magenta] — 対象: [bold]{files}[/bold]")
            _, last_state = _run_agent(
                llm, apply_goal, "coding",
                context=_SELF_APPLY_CONTEXT,
                max_iterations=12,
                world_model=shared_world_model,
            )
            if last_state.world_model:
                shared_world_model = last_state.world_model

        # --- コード生成 ---
        elif raw.startswith("/generate"):
            description = raw[len("/generate"):].strip()
            if not description:
                console.print("[red]使い方: /generate <コードの説明>[/red]")
                continue
            with console.status("[cyan]コードを生成中...[/cyan]"):
                result = generator.generate(description)
            _display(result, title="生成されたコード")

        # --- コードレビュー ---
        elif raw == "/review":
            code = _collect_code()
            if not code.strip():
                console.print("[yellow]コードが入力されていません。[/yellow]")
                continue
            with console.status("[cyan]レビュー中...[/cyan]"):
                result = reviewer_agent.review(code)
            _display(result, title="コードレビュー")

        # --- フリーテキスト: チャット or エージェントを自動判断 ---
        else:
            with console.status("[cyan]判断中...[/cyan]"):
                intent_type, domain = _classify_intent(fast_llm, raw)

            if intent_type == "task":
                console.print(
                    f"[yellow]タスクを検出しました（domain=[cyan]{domain}[/cyan]）。"
                    f"エージェントを起動します...[/yellow]"
                )
                _, last_state = _run_agent(llm, raw, domain, world_model=shared_world_model)
                if last_state.world_model:
                    shared_world_model = last_state.world_model
            else:
                history.append({"role": "user", "content": raw})
                messages = [{"role": "system", "content": _GENERAL_SYSTEM}] + history
                with console.status("[cyan]考え中...[/cyan]"):
                    reply = llm.chat(messages, temperature=0.7, max_tokens=2048)
                if not reply:
                    console.print("[red]応答を取得できませんでした。[/red]")
                    history.pop()
                else:
                    history.append({"role": "assistant", "content": reply})
                    _display(reply, title="Hermes AI")


if __name__ == "__main__":
    main()
