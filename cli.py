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
import re
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
from hermes_agi_gen.agi_core import AGICore
from hermes_agi_gen.claude_client import ClaudeClient
from hermes_agi_gen.code_agents import CodeGeneratorAgent, CodeReviewerAgent
from hermes_agi_gen.daemon import HermesDaemon
from hermes_agi_gen.hermes_constants import DOMAIN_CONFIG
from hermes_agi_gen.mistral_client import MistralClient
from hermes_agi_gen.scheduler import JobScheduler, ScheduledJob, parse_trigger_spec
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
| `/experiment` | AutoResearch方式: 洞察→コード実験→メトリクス比較→受容/ロールバック |
| `/daemon start` | AGI自律デーモンをバックグラウンドで起動 |
| `/daemon stop` | デーモンを停止 |
| `/daemon status` | デーモンの状態を確認 |
| `/daemon log` | デーモンのログを表示 |
| `/schedule` | スケジュール済みジョブ一覧 |
| `/schedule add <trigger> <goal>` | ジョブ追加 (例: `daily:09:00` `every:30m` `weekly:mon:09:00`) |
| `/schedule remove <id>` | ジョブ削除 |
| `/schedule enable/disable <id>` | ジョブ有効・無効化 |
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


# 時間指定パターン (「〜時になったら」「〜日の〜時に」など)
_TIME_SPEC_PATTERNS = [
    r'\d{4}年\d{1,2}月\d{1,2}日',   # 2026年4月6日
    r'午前\d{1,2}時\d{0,2}分?',     # 午前3時32分
    r'午後\d{1,2}時\d{0,2}分?',
    r'\d{1,2}時\d{0,2}分?になったら',  # 17時31分になったら (24時間表記)
    r'\d{1,2}時\d{0,2}分?になれば',
    r'\d{1,2}:\d{2}に',              # 09:00に
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}',  # ISO8601
    r'になったら',
    r'になれば',
    r'の時に',
    r'every:\d',
    r'daily:',
    r'weekly:',
]

_TIME_SPEC_RE = re.compile("|".join(_TIME_SPEC_PATTERNS))

def _extract_schedule_trigger(message: str) -> Optional[str]:
    """時間指定リクエストからISO8601トリガー文字列を抽出する。なければNone。"""
    import re
    import datetime

    # ISO8601 直接指定
    m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', message)
    if m:
        return f"once:{m.group()}"

    # 「YYYY年MM月DD日 午前/午後HH時MM分」形式
    date_m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', message)

    hour, minute = None, None

    # 午前/午後 付き: 午前3時32分, 午後5時
    time_m = re.search(r'午前(\d{1,2})時(\d{1,2})?分?', message)
    if time_m:
        hour = int(time_m.group(1))
        if hour == 12:
            hour = 0
        minute = int(time_m.group(2) or 0)
    else:
        time_m = re.search(r'午後(\d{1,2})時(\d{1,2})?分?', message)
        if time_m:
            hour = int(time_m.group(1))
            if hour != 12:
                hour += 12
            minute = int(time_m.group(2) or 0)

    # 24時間表記: 17時31分, 9時, 23時05分
    if hour is None:
        time_m = re.search(r'(\d{1,2})時(\d{1,2})?分?', message)
        if time_m:
            hour = int(time_m.group(1))
            minute = int(time_m.group(2) or 0)

    if hour is not None:
        if date_m:
            y, mo, d = date_m.group(1), date_m.group(2).zfill(2), date_m.group(3).zfill(2)
            return f"once:{y}-{mo}-{d}T{str(hour).zfill(2)}:{str(minute).zfill(2)}"
        else:
            # 日付なし → 今日の日付を補完
            today = datetime.date.today()
            return f"once:{today.isoformat()}T{str(hour).zfill(2)}:{str(minute).zfill(2)}"

    # 「HH:MM に」形式
    hhmm_m = re.search(r'(\d{1,2}):(\d{2})に', message)
    if hhmm_m:
        return f"daily:{hhmm_m.group(1).zfill(2)}:{hhmm_m.group(2)}"

    return None


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _provider_label(llm) -> str:
    if isinstance(llm, ClaudeClient):
        return f"[bold green]Claude (Anthropic)[/bold green] ({llm.model})"
    url = getattr(llm, "base_url", "")
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
    # ゴール内の URL を抽出 (ASCII文字のみ = 日本語で誤って伸びない)
    _urls = re.findall(r'https?://[a-zA-Z0-9./_?=&#%+~-]+', goal)
    _wants_save = any(w in goal for w in ["保存", "ファイル", "txt", "Desktop", "save", "write"])
    _wants_translate = any(w in goal for w in ["翻訳", "要約", "日本語", "translate", "summarize"])

    # URL + 保存 + 要約タスク: LLMに生成させず動作確認済みスクリプトを直接プランに注入
    _prebuilt_plan: List[str] = []
    if _urls and _wants_save and _wants_translate:
        _target_url = _urls[0]
        _out_dir_kw = ""
        _m = re.search(r'~/[a-zA-Z0-9/_.-]+', goal)  # ASCII文字のみ
        if _m:
            _out_dir_kw = _m.group(0).rstrip("/")
        if not _out_dir_kw:
            _out_dir_kw = "~/Desktop/AI_News"
        _groq_model = llm.model if "groq" in getattr(llm, "base_url", "") else "llama-3.1-8b-instant"
        # ゴールから長さ・詳細指示を抽出してスクリプトに注入
        # 数値(字/文字)があればそれを使い、なければユーザーの表現をそのまま使う
        _num_m = re.search(r'(\d+)\s*[字文]', goal)
        _target_chars = int(_num_m.group(1)) if _num_m else None
        # 翻訳/要約の指示部分を抽出 (例: "翻訳して1500字程度にまとめて" → "1500字程度にまとめること")
        _inst_m = re.search(r'(?:翻訳して|日本語[にで])(.+?)(?:[、,。]|~/|ファイル|txt)', goal)
        if _inst_m:
            _length_instruction = re.sub(r'(?:してください|して|ください)$', '', _inst_m.group(1).strip())
        elif _target_chars:
            _length_instruction = f"{_target_chars}字程度にまとめること"
        else:
            _length_instruction = "詳しくまとめること"
        _max_tokens = max(800, _target_chars * 3) if _target_chars else 2000
        _body_limit = max(4000, _target_chars * 10) if _target_chars else 8000
        _target_count_m = re.search(r'(\d+)\s*[件つ個]', goal)
        _target_count = int(_target_count_m.group(1)) if _target_count_m else 3
        _script = f"""\
import requests, os, re, datetime, json
# 1. HNからAI記事を取得
_hn = requests.get({_target_url!r}, timeout=15, headers={{"User-Agent": "Mozilla/5.0"}})
_items = re.findall(r'class=[\\x22\\x27]titleline[\\x22\\x27]><a href=[\\x22\\x27]([^\\x22\\x27]+)[\\x22\\x27]>([^<]+)', _hn.text)
_kw = ["ai", "llm", "gpt", "machine learning", "neural", "model", "agent", "openai", "anthropic",
       "deepmind", "claude", "gemini", "mistral", "ml", "nlp", "robot", "coding", "software"]
_ai = [(u, t) for u, t in _items if any(k in t.lower() for k in _kw)]
# 目標件数に満たなければ一般記事で補完
_targets = _ai[:{_target_count}]
if len(_targets) < {_target_count}:
    _non_ai = [(u, t) for u, t in _items if (u, t) not in _ai]
    _targets += _non_ai[:{_target_count} - len(_targets)]
# 2. 各記事の本文を取得
def _fetch_body(url, limit={_body_limit}):
    if url.startswith("item?") or url.startswith("//"):
        return ""
    try:
        _r = requests.get(url, timeout=15, headers={{"User-Agent": "Mozilla/5.0"}})
        _h = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", _r.text, flags=re.DOTALL|re.IGNORECASE)
        _t = re.sub(r"<[^>]+>", " ", _h)
        return re.sub(r"\\s+", " ", _t).strip()[:limit]
    except Exception:
        return ""
_articles = []
for _u, _tit in _targets:
    _tit_clean = re.sub(r"&#x27;", "'", _tit).strip()
    _body = _fetch_body(_u)
    _articles.append({{"title": _tit_clean, "url": _u, "body": _body}})
# 3. Anthropic/Groq APIで翻訳・要約
_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")
_sections = []
for _i, _a in enumerate(_articles):
    _body_snip = _a["body"][:2000] if _a["body"] else "(本文取得不可)"
    _prompt = (
        f"以下の英語記事を日本語に翻訳して{_length_instruction}で要約してください。\\n\\n"
        f"タイトル: {{_a['title']}}\\n\\n本文抜粋:\\n{{_body_snip}}\\n\\n"
        "以下のJSON形式のみで返してください:\\n"
        '{{"title_ja":"翻訳タイトル","summary_ja":"日本語要約"}}'
    )
    _title_ja, _summary_ja = _a["title"], "(翻訳なし)"
    if _anthropic_key:
        try:
            _resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={{"x-api-key": _anthropic_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}},
                json={{"model": "claude-haiku-4-5-20251001", "max_tokens": {_max_tokens},
                      "messages": [{{"role": "user", "content": _prompt}}]}},
                timeout=60,
            )
            _resp.raise_for_status()
            _raw = _resp.json()["content"][0]["text"]
            _m = re.search(r"\\{{.*?\\}}", _raw, re.DOTALL)
            if _m:
                _d = json.loads(_m.group())
                _title_ja = _d.get("title_ja", _title_ja)
                _summary_ja = _d.get("summary_ja", _summary_ja)
        except Exception as _e:
            _summary_ja = f"(翻訳エラー: {{_e}})"
    elif _groq_key:
        try:
            _gr = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={{"Authorization": f"Bearer {{_groq_key}}", "Content-Type": "application/json"}},
                json={{"model": {_groq_model!r}, "messages": [{{"role": "user", "content": _prompt}}], "max_tokens": {_max_tokens}}},
                timeout=60,
            )
            _gr.raise_for_status()
            _raw = _gr.json()["choices"][0]["message"]["content"]
            _m = re.search(r"\\{{.*?\\}}", _raw, re.DOTALL)
            if _m:
                _d = json.loads(_m.group())
                _title_ja = _d.get("title_ja", _title_ja)
                _summary_ja = _d.get("summary_ja", _summary_ja)
        except Exception as _e:
            _summary_ja = f"(翻訳エラー: {{_e}})"
    _sections.append(f"## 記事{{_i+1}}: {{_title_ja}}\\n原題: {{_a['title']}}\\n出典: {{_a['url']}}\\n\\n{{_summary_ja}}")
# 4. ファイル保存
_header = f"=== Hacker News AI ニュース {{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}} ===\\n"
_content = _header + "\\n\\n" + "\\n\\n---\\n\\n".join(_sections) + f"\\n\\n(Hacker Newsより取得 {{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}})"
_out_dir = os.path.expanduser({_out_dir_kw!r})
os.makedirs(_out_dir, exist_ok=True)
_fname = os.path.join(_out_dir, f"AI_News_{{datetime.datetime.now().strftime('%m-%d_%H%M')}}.txt")
with open(_fname, "w", encoding="utf-8") as _f:
    _f.write(_content)
print(f"保存完了: {{_fname}}")
print(_content)
"""
        _prebuilt_plan = [f"PYTHON:\n{_script}", "DONE: ファイルを保存しました"]
    elif not _prebuilt_plan and _wants_translate and any(kw in goal.lower() for kw in ["hacker news", "hn", "hackernews"]):
        # HN + 日本語表示 (保存なし): ニュース内容を取得して翻訳・表示するスクリプト
        _target_count_m = re.search(r'(\d+)\s*[件つ個]', goal)
        _hn_count = int(_target_count_m.group(1)) if _target_count_m else 1
        _hn_script = f"""\
import requests, re, json, os
# 1. HN APIでトップストーリーのIDを取得
_ids = requests.get('https://hacker-news.firebaseio.com/v0/topstories.json', timeout=15).json()
_articles = []
for _sid in _ids[:{_hn_count}]:
    _item = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{{_sid}}.json', timeout=10).json()
    _title = _item.get('title', '')
    _url = _item.get('url', '')
    # 記事本文を取得
    _body = ''
    if _url:
        try:
            _r = requests.get(_url, timeout=15, headers={{"User-Agent": "Mozilla/5.0"}})
            _h = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", _r.text, flags=re.DOTALL|re.IGNORECASE)
            _body = re.sub(r"<[^>]+>", " ", _h)
            _body = re.sub(r"\\s+", " ", _body).strip()[:3000]
        except Exception:
            pass
    _articles.append({{"title": _title, "url": _url, "body": _body}})
# 2. 翻訳 (Anthropic > Groq > Ollama)
_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")
for _a in _articles:
    _body_snip = _a["body"][:2000] if _a["body"] else "(本文なし)"
    _prompt = (
        "以下の英語記事を日本語に翻訳してください。タイトルと内容の要約(300字以上)を含めてください。\\n\\n"
        f"タイトル: {{_a['title']}}\\n出典: {{_a['url']}}\\n\\n本文抜粋:\\n{{_body_snip}}\\n\\n"
        "以下のJSON形式のみで返してください:\\n"
        '{{"title_ja":"日本語タイトル","summary_ja":"日本語の詳しい要約(300字以上)"}}'
    )
    _title_ja, _summary_ja = _a["title"], _a["body"][:500] if _a["body"] else "(内容取得不可)"
    if _anthropic_key:
        try:
            _resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={{"x-api-key": _anthropic_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}},
                json={{"model": "claude-haiku-4-5-20251001", "max_tokens": 1500,
                      "messages": [{{"role": "user", "content": _prompt}}]}},
                timeout=60,
            )
            _resp.raise_for_status()
            _raw = _resp.json()["content"][0]["text"]
            _m = re.search(r"\\{{.*?\\}}", _raw, re.DOTALL)
            if _m:
                _d = json.loads(_m.group())
                _title_ja = _d.get("title_ja", _title_ja)
                _summary_ja = _d.get("summary_ja", _summary_ja)
        except Exception as _e:
            _summary_ja = f"(翻訳エラー: {{_e}})"
    elif _groq_key:
        try:
            _gr = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={{"Authorization": f"Bearer {{_groq_key}}", "Content-Type": "application/json"}},
                json={{"model": "llama-3.1-8b-instant", "messages": [{{"role": "user", "content": _prompt}}], "max_tokens": 1500}},
                timeout=60,
            )
            _gr.raise_for_status()
            _raw = _gr.json()["choices"][0]["message"]["content"]
            _m = re.search(r"\\{{.*?\\}}", _raw, re.DOTALL)
            if _m:
                _d = json.loads(_m.group())
                _title_ja = _d.get("title_ja", _title_ja)
                _summary_ja = _d.get("summary_ja", _summary_ja)
        except Exception as _e:
            _summary_ja = f"(翻訳エラー: {{_e}})"
    print(f"\\n===== {{_title_ja}} =====")
    print(f"原題: {{_a['title']}}")
    print(f"出典: {{_a['url']}}")
    print(f"\\n{{_summary_ja}}")
    print()
"""
        _prebuilt_plan = [f"PYTHON:\n{_hn_script}", "DONE: Hacker Newsのニュースを日本語で表示しました"]
    elif _urls:
        context_hint = (
            "【重要】ゴールに以下のURLが含まれています。"
            "SEARCH: ではなく FETCH: でこのURLに直接アクセスしてください: "
            + ", ".join(_urls)
        )
        context = (context_hint + "\n" + context) if context else context_hint

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
    # 動作確認済みプランがあれば直接注入 (LLM計画をスキップ)
    if _prebuilt_plan:
        state.current_plan = _prebuilt_plan

    final_state = agent.run(state)

    console.print(Rule(style="yellow"))

    # 信頼度警告を表示
    warnings = final_state.working_memory.get("confidence_warnings", [])
    for w in warnings:
        console.print(f"[bold red]{w}[/bold red]")

    summary = final_state.working_memory.get("completion_summary", "")
    if not summary and final_state.observations:
        summary = final_state.observations[-1]

    # ツール実行の実際の出力を収集 (FETCH/PYTHON/CMD の stdout)
    tool_outputs = final_state.working_memory.get("tool_outputs", [])
    if tool_outputs:
        combined = "\n".join(tool_outputs)
        if combined.strip():
            _display(combined[:3000], title="実行結果", border="green")

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


def _post_run_learning(
    agi_core: AGICore,
    goal: str,
    domain: str,
    final_state,
) -> None:
    """エージェント実行後の自己改善フェーズ。

    _run_agent() のたびに呼び出し、認知サイクルの学習・省察部分を実行する。
    AGICore.run_goal() の後半部分を切り出したもの。
    """
    success = final_state.is_done and len(getattr(final_state, "failed_steps", [])) == 0
    perf_score = 1.0 if success else (0.4 if getattr(final_state, "completed_steps", []) else 0.1)

    # --- 学習: few-shot 例と anti-pattern を抽出 ---
    agi_core.self_improver.analyze_session(final_state)
    agi_core.self_improver.record_session_performance(
        session_id=getattr(final_state, "session_id", str(id(final_state))),
        goal=goal,
        domain=domain,
        score=perf_score,
    )
    if success:
        agi_core.identity.successful_goals += 1
    agi_core.identity.total_goals_processed += 1

    # --- 省察フェーズ (適応的インターバル) ---
    recent_trend = agi_core.self_improver.get_performance_trend(window=10)
    if agi_core.reflection_engine.should_reflect(recent_success_rate=recent_trend):
        console.print("[dim][ReflectionEngine] 自己省察中...[/dim]", flush=True)
        insights = agi_core.reflection_engine.reflect(agi_core.ltm)

        if insights:
            ins_strs = [f"[{i.category}] {i.content[:50]}" for i in insights[:3]]
            console.print(
                "[dim][省察] " + " / ".join(ins_strs) + "[/dim]"
            )

        # 戦略的ゴールを MetaCognition に追加
        strategic_goals = agi_core.reflection_engine.generate_strategic_goals(insights, agi_core.ltm)
        for sg in strategic_goals:
            agi_core.meta.goal_queue.add(sg)

        # 自己同一性を更新
        metrics = agi_core.reflection_engine.compute_growth_metrics(agi_core.ltm)
        agi_core.identity.update_from_metrics(metrics)

        # 3省察ごとに実験ループ
        agi_core._reflection_count += 1
        if agi_core._reflection_count % 3 == 0 and agi_core.llm is not None:
            console.print("[dim][ExperimentRunner] AutoResearch方式実験を開始...[/dim]")
            exp_results = agi_core.experiment_runner.run_experiments_from_insights(
                insights, max_experiments=2
            )
            accepted = sum(1 for r in exp_results if r.accepted)
            if exp_results:
                console.print(
                    f"[dim][ExperimentRunner] {len(exp_results)}件実験 / "
                    f"[green]{accepted}件採用[/green][/dim]"
                )
            elif agi_core.llm is not None:
                agi_core._attempt_self_modification(insights)


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


def _cmd_schedule(args: str) -> None:
    """スケジュールジョブの管理。

    /schedule               → 一覧表示
    /schedule list          → 一覧表示
    /schedule add <trigger> <goal>
                            → ジョブ追加
                              例: /schedule add daily:09:00 毎朝ニュースを要約
                                  /schedule add every:30m  システム状態を確認
                                  /schedule add 2026-04-01T09:00 四半期レポートを作成
                                  /schedule add weekly:mon:09:00 週次レポートを作成
    /schedule remove <id>   → ジョブ削除
    /schedule enable <id>   → ジョブ有効化
    /schedule disable <id>  → ジョブ無効化
    """
    scheduler = JobScheduler()
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if subcmd in ("", "list"):
        jobs = scheduler.list_jobs()
        if not jobs:
            console.print("[dim]スケジュール済みジョブはありません。[/dim]")
            console.print("[dim]/schedule add <trigger> <goal> で追加できます。[/dim]")
            return

        table = Table(title="スケジュールジョブ一覧", border_style="cyan")
        table.add_column("ID", style="cyan bold", width=10)
        table.add_column("状態", width=6)
        table.add_column("トリガー", style="dim")
        table.add_column("次回実行")
        table.add_column("ゴール")

        for job in jobs:
            status_str = "[green]有効[/green]" if job.enabled else "[dim]無効[/dim]"
            next_str = scheduler.format_next_run(job)
            table.add_row(job.id, status_str, job.trigger, next_str, job.goal[:50])
        console.print(table)
        return

    if subcmd == "add":
        add_parts = rest.split(None, 1)
        if len(add_parts) < 2:
            console.print("[red]使い方: /schedule add <trigger> <goal>[/red]")
            console.print("[dim]例: /schedule add daily:09:00 毎朝ニュースを要約する[/dim]")
            return
        trigger_raw, goal_text = add_parts[0], add_parts[1]
        trigger = parse_trigger_spec(trigger_raw)
        if trigger is None:
            console.print(f"[red]トリガー形式が不明です: {trigger_raw}[/red]")
            console.print(
                "[dim]サポート形式:\n"
                "  daily:09:00          毎日09:00\n"
                "  weekly:mon:09:00     毎週月曜09:00\n"
                "  every:30m            30分ごと\n"
                "  every:2h             2時間ごと\n"
                "  2026-04-01T09:00     指定日時に1回[/dim]"
            )
            return
        job = scheduler.add_job(goal=goal_text, trigger=trigger)
        next_str = scheduler.format_next_run(job)
        console.print(
            f"[green]ジョブを登録しました:[/green] [{job.id}] {goal_text[:60]}\n"
            f"[dim]トリガー: {trigger} | 次回実行: {next_str}[/dim]"
        )
        console.print("[dim]デーモン起動中の場合は自動実行されます。/daemon start で起動。[/dim]")
        return

    if subcmd == "remove":
        job_id = rest.strip()
        if not job_id:
            console.print("[red]使い方: /schedule remove <id>[/red]")
            return
        if scheduler.remove_job(job_id):
            console.print(f"[yellow]ジョブ [{job_id}] を削除しました。[/yellow]")
        else:
            console.print(f"[red]ジョブ [{job_id}] が見つかりません。[/red]")
        return

    if subcmd == "enable":
        job_id = rest.strip()
        if scheduler.enable_job(job_id, enabled=True):
            console.print(f"[green]ジョブ [{job_id}] を有効化しました。[/green]")
        else:
            console.print(f"[red]ジョブ [{job_id}] が見つかりません。[/red]")
        return

    if subcmd == "disable":
        job_id = rest.strip()
        if scheduler.enable_job(job_id, enabled=False):
            console.print(f"[yellow]ジョブ [{job_id}] を無効化しました。[/yellow]")
        else:
            console.print(f"[red]ジョブ [{job_id}] が見つかりません。[/red]")
        return

    console.print("[dim]使い方: /schedule [list|add|remove|enable|disable] ...[/dim]")


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


def _cmd_experiment(llm: "MistralClient", last_reflection: Dict[str, str]) -> None:
    """AutoResearch方式の実験ループを手動トリガーする。

    ReflectionEngine の洞察を使って、コード改変→テスト→メトリクス比較→
    受容/ロールバック の実験サイクルを実行する。
    """
    from hermes_agi_gen.agi_core import AGICore
    from hermes_agi_gen.reflection_engine import Insight

    console.print(Rule("[bold cyan]AutoResearch 方式 実験ループ[/bold cyan]", style="cyan"))
    console.print("[dim]洞察 → コード改変 → メトリクス比較 → 受容/ロールバック[/dim]\n")

    core = AGICore(llm=llm)

    # ReflectionEngine で洞察を生成
    with console.status("[yellow]自己省察中...[/yellow]"):
        insights = core.reflection_engine.reflect(core.ltm)

    if not insights:
        console.print("[yellow]洞察がありません。まず /run でタスクを実行してください。[/yellow]")
        return

    # actionable な洞察を表示
    actionable = [i for i in insights if i.actionable]
    if not actionable:
        console.print("[yellow]行動可能な洞察がありません。[/yellow]")
        for i in insights:
            console.print(f"  [{i.category}] {i.content[:80]} (確信度: {i.confidence:.0%})")
        return

    table = Table(title="行動可能な洞察 (実験対象)", border_style="cyan")
    table.add_column("カテゴリ", style="cyan", width=12)
    table.add_column("内容")
    table.add_column("確信度", justify="right", width=8)

    for i in actionable[:5]:
        cat_color = {"weakness": "red", "gap": "yellow", "opportunity": "green"}.get(i.category, "white")
        table.add_row(
            f"[{cat_color}]{i.category}[/{cat_color}]",
            i.content[:70],
            f"{i.confidence:.0%}",
        )
    console.print(table)

    from rich.prompt import Prompt
    confirm = Prompt.ask(
        f"\n{len(actionable)}件の洞察で実験を実行しますか？",
        choices=["y", "n"],
        default="n",
    )
    if confirm != "y":
        console.print("[dim]キャンセルしました。[/dim]")
        return

    console.print("\n[cyan]実験を開始します...[/cyan]")
    results = core.experiment_runner.run_experiments_from_insights(
        actionable, max_experiments=3
    )

    if not results:
        console.print("[yellow]実験結果なし。[/yellow]")
        return

    # 結果テーブル
    result_table = Table(title="実験結果", border_style="green")
    result_table.add_column("洞察", width=40)
    result_table.add_column("対象ファイル", width=30)
    result_table.add_column("テスト", width=8)
    result_table.add_column("改善", justify="right", width=8)
    result_table.add_column("判定", width=10)

    for r in results:
        test_str = "[green]✓[/green]" if r.test_passed else "[red]✗[/red]"
        imp_color = "green" if r.improvement > 0 else ("yellow" if r.improvement >= -0.01 else "red")
        status = "[green]採用[/green]" if r.accepted else "[red]ロールバック[/red]"
        result_table.add_row(
            r.insight.content[:38],
            r.patch.file_path if r.patch else "—",
            test_str,
            f"[{imp_color}]{r.improvement:+.3f}[/{imp_color}]",
            status,
        )
    console.print(result_table)

    accepted = sum(1 for r in results if r.accepted)
    console.print(
        f"\n[bold]実験サマリー:[/bold] {len(results)}件実行 / "
        f"[green]{accepted}件採用[/green] / "
        f"[red]{len(results) - accepted}件ロールバック[/red]"
    )
    console.print(f"[dim]累計実験履歴: {core.experiment_runner.summary()}[/dim]")


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

    from hermes_agi_gen.agi_core import AGICore
    agi_core = AGICore(llm=llm)
    agi_status = agi_core.get_status()

    wm_summary = agi_status.get("world_model_age", "未初期化")
    goals_count = agi_status.get("goal_queue_size", 0)
    if last_state:
        goals_count = max(goals_count, len(last_state.working_memory.get("goal_queue", [])))

    status_text = (
        f"[bold cyan]AGI Identity[/bold cyan]: {agi_status['identity']}\n"
        f"[bold cyan]LLM[/bold cyan]: {_provider_label(llm)}\n"
        f"[bold cyan]カスタムツール[/bold cyan]: {len(tools)}個登録済み\n"
        f"[bold cyan]学習済みパターン[/bold cyan]: 良い例 {len(examples)}件 / 悪い例 {len(anti)}件\n"
        f"[bold cyan]世界モデル[/bold cyan]: {wm_summary}\n"
        f"[bold cyan]自律ゴールキュー[/bold cyan]: {goals_count}件待機中\n"
        f"[bold cyan]成長指標[/bold cyan]: {agi_status.get('growth_metrics', 'N/A')}\n"
        f"[bold cyan]予測精度[/bold cyan]: {agi_status.get('prediction_accuracy', 0):.0%}\n"
        f"[bold cyan]省察エンジン[/bold cyan]: {agi_status.get('reflection', 'N/A')}\n"
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
# インプロセス スケジューラ (デーモン不要でCLI内から定時実行)
# ---------------------------------------------------------------------------

def _start_inline_scheduler(llm: MistralClient, con: Console) -> "threading.Event":
    """バックグラウンドスレッドでスケジューラを監視し、期限ジョブをその場で実行する。

    戻り値の Event を set() すると停止する。
    """
    import threading
    from hermes_agi_gen.meta_cognition import GoalQueue

    stop_event = threading.Event()

    def _loop() -> None:
        scheduler = JobScheduler()
        goal_queue = GoalQueue()  # ダミー — tick() の戻り値ジョブだけ使う
        while not stop_event.wait(timeout=20):  # 20秒ごとにチェック
            try:
                triggered = scheduler.tick(goal_queue)
                for job in triggered:
                    con.print(f"\n[bold cyan][スケジューラ][/bold cyan] ジョブ発火: [{job.id}] {job.goal[:60]}")
                    _run_scheduled_job(llm, job, con)
            except Exception as exc:
                con.print(f"[dim][スケジューラ] エラー: {exc}[/dim]")

    t = threading.Thread(target=_loop, daemon=True, name="hermes-inline-scheduler")
    t.start()
    return stop_event


def _run_scheduled_job(llm: MistralClient, job: "ScheduledJob", con: Console) -> None:
    """スケジュールされたゴールをエージェントで実行する。"""
    import threading
    import time as _time

    def _execute() -> None:
        try:
            con.print(f"[cyan][スケジューラ] 実行開始: {job.goal[:80]}[/cyan]")
            _time.sleep(5)  # Groqレートリミット回避: 直前のAPI呼び出しから間隔を空ける
            summary, _ = _run_agent(
                llm=llm,
                goal=job.goal,
                domain=job.domain or "general",
                max_iterations=6,  # スケジュール実行はイテレーションを控えめに
            )
            if not summary:
                summary = f"[{job.id}] 完了"
            con.print(f"[green][スケジューラ] 完了: {summary[:120]}[/green]")
        except Exception as exc:
            con.print(f"[red][スケジューラ] 実行エラー [{job.id}]: {exc}[/red]")
        finally:
            # バックグラウンド出力後、Prompt.ask() の表示が崩れるのを修復
            import sys
            sys.stdout.write("\n\033[32mhermes\033[0m: ")
            sys.stdout.flush()

    t = threading.Thread(target=_execute, daemon=True, name=f"hermes-job-{job.id}")
    t.start()


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hermes AI インタラクティブ CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python3 cli.py                  # 自動検出 (Groq→Claude→Ollama)\n"
            "  python3 cli.py --model claude   # Claude API を使用 (ANTHROPIC_API_KEY 必須)\n"
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


def _build_llm(model: Optional[str]):
    """モデル指定からLLMクライアントを構築する。

    優先順位 (--model 未指定の自動検出):
      1. GROQ_API_KEY が設定されていれば Groq
      2. ANTHROPIC_API_KEY が設定されていれば Claude
      3. それ以外は Ollama

    明示指定:
      --model groq    → Groq (GROQ_API_KEY 必須)
      --model claude  → Claude (ANTHROPIC_API_KEY 必須)
      --model qwen3   → Ollama の qwen3
    """
    import os as _os

    # --- 明示指定: claude ---
    if model == "claude":
        if not ClaudeClient.is_available():
            console.print(
                "[bold red]ANTHROPIC_API_KEY が設定されていません。[/bold red]\n"
                "[dim].env に ANTHROPIC_API_KEY=sk-ant-... を追加してください。[/dim]"
            )
            return None
        console.print("[green]Anthropic Claude API を使用します。[/green]")
        return ClaudeClient()

    # --- 自動検出: Groq を最優先 ---
    if model is None and _os.getenv("GROQ_API_KEY"):
        console.print("[green]GROQ_API_KEY を検出 — Groq API を使用します。[/green]")
        return MistralClient(model="groq")

    # --- 自動検出: Anthropic Claude (Groq がない場合) ---
    if model is None and ClaudeClient.is_available():
        console.print("[green]ANTHROPIC_API_KEY を検出 — Claude API を使用します。[/green]")
        return ClaudeClient()

    # --- 明示指定 or Ollama フォールバック ---
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

    # fast_llm: Claude使用時はそのまま、Groq使用時は軽量モデル、それ以外は同じ
    if isinstance(llm, ClaudeClient):
        fast_llm = ClaudeClient.fast()
    elif args.model in (None, "groq"):
        fast_llm = MistralClient.fast()
    else:
        fast_llm = llm

    db = SessionDB()
    generator = CodeGeneratorAgent(llm=llm, session_db=db)
    reviewer_agent = CodeReviewerAgent(llm=llm, session_db=db)

    # AGIコアコンポーネント (統合認知サイクル + 自己改善)
    tool_registry = ToolRegistry()
    agi_core = AGICore(llm=llm, repo_root=Path("."))
    self_improver = agi_core.self_improver   # AGICoreのものを共有
    shared_world_model = agi_core.world_model  # AGICoreの世界モデルを共有

    history: List[Dict[str, str]] = []
    last_reflection: Dict[str, str] = {}
    last_state: Optional[AgentState] = None

    # --- インプロセス スケジューラ (デーモン不要) ---
    _sched_stop = _start_inline_scheduler(llm, console)

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
            _sched_stop.set()
            console.print("\n[dim]終了します。[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue

        # --- 終了 ---
        if raw in {"/quit", "/exit", "quit", "exit"}:
            _sched_stop.set()
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
            try:
                _cmd_status(llm, tool_registry, self_improver, last_state)
            except Exception as _exc:
                console.print(f"[bold red]ステータス取得エラー:[/bold red] {_exc}")

        # --- カスタムツール ---
        elif raw.startswith("/tools"):
            tools_args = raw[6:].strip()
            _cmd_tools(tool_registry, tools_args)

        # --- デーモン制御 ---
        elif raw.startswith("/daemon"):
            subcmd = raw[7:].strip()
            _cmd_daemon(subcmd)

        # --- スケジュール管理 ---
        elif raw.startswith("/schedule"):
            _cmd_schedule(raw[9:].strip())

        # --- 自律コード修正 ---
        elif raw.startswith("/self-modify"):
            target = raw[12:].strip().lower()
            _cmd_self_modify(llm, target, last_reflection)

        # --- AutoResearch方式の実験ループ ---
        elif raw == "/experiment":
            _cmd_experiment(llm, last_reflection)

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
            try:
                with console.status("[yellow]ドメインを判定中...[/yellow]"):
                    _, domain = _classify_intent(fast_llm, goal)
                _, last_state = _run_agent(llm, goal, domain, world_model=shared_world_model,
                                           max_iterations=args.max_turns)
                if last_state.world_model:
                    shared_world_model = last_state.world_model  # 世界モデルを更新
                    agi_core.world_model = last_state.world_model
                # 学習・省察フェーズ
                try:
                    _post_run_learning(agi_core, goal, domain, last_state)
                except Exception as _learn_exc:
                    console.print(f"[dim][自己改善] エラー: {_learn_exc}[/dim]")
            except Exception as _exc:
                console.print(f"[bold red]エラーが発生しました:[/bold red] {_exc}")

        # --- 階層的マルチエージェント ---
        elif raw.startswith("/orch"):
            goal = raw[5:].strip()
            if not goal:
                console.print("[red]使い方: /orch <複雑な目標・タスク>[/red]")
                continue
            try:
                console.print(Rule(
                    "[bold magenta]オーケストレーターモード (階層的並列実行)[/bold magenta]",
                    style="magenta",
                ))
                with console.status("[magenta]階層的ゴールツリーを生成・実行中...[/magenta]"):
                    orch = AgentOrchestrator(llm=llm, use_hierarchical=True)
                    result = orch.run(goal)
                console.print(Rule(style="magenta"))
                _display(result, title="オーケストレーター完了", border="magenta")
            except Exception as _exc:
                console.print(f"[bold red]エラーが発生しました:[/bold red] {_exc}")

        # --- 自己診断・改善提案 ---
        elif raw.startswith("/reflect"):
            try:
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
            except Exception as _exc:
                console.print(f"[bold red]エラーが発生しました:[/bold red] {_exc}")

        # --- 改善提案を適用 ---
        elif raw.startswith("/apply"):
            if not last_reflection:
                console.print("[yellow]まず /reflect を実行してください。[/yellow]")
                continue
            try:
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
            except Exception as _exc:
                console.print(f"[bold red]エラーが発生しました:[/bold red] {_exc}")

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

        # --- 未知の /コマンド: タイポ警告 ---
        elif raw.startswith("/"):
            _KNOWN_CMDS = [
                "/quit", "/exit", "/help", "/clear", "/provider", "/status",
                "/tools", "/daemon", "/schedule", "/self-modify", "/experiment", "/run",
                "/orch", "/reflect", "/apply", "/generate", "/review",
            ]
            import difflib
            cmd_word = raw.split()[0]
            close = difflib.get_close_matches(cmd_word, _KNOWN_CMDS, n=1, cutoff=0.6)
            if close:
                console.print(
                    f"[red]未知のコマンド: {cmd_word}[/red]  "
                    f"もしかして: [bold]{close[0]}[/bold] ?"
                )
            else:
                console.print(f"[red]未知のコマンド: {cmd_word}[/red]  /help でコマンド一覧を確認してください。")

        # --- フリーテキスト: チャット or エージェントを自動判断 ---
        else:
            try:
                # 時間指定リクエストを事前検出して直接スケジュール登録
                trigger = _extract_schedule_trigger(raw)
                if trigger and _TIME_SPEC_RE.search(raw):
                    from hermes_agi_gen.scheduler import JobScheduler, parse_trigger_spec
                    parsed = parse_trigger_spec(trigger.split(":", 1)[1] if ":" in trigger else trigger)
                    if parsed is None:
                        parsed = trigger
                    job_scheduler = JobScheduler()
                    # ゴールテキスト: 「になったら」「になれば」以降を抽出、なければ日時部分だけ除去
                    _split = re.split(r'になったら[、,]?\s*|になれば[、,]?\s*', raw, maxsplit=1)
                    if len(_split) > 1:
                        goal_text = _split[1].strip()
                    else:
                        goal_text = re.sub(
                            r'\d{4}年\d{1,2}月\d{1,2}日\s*(?:午前|午後)?\d{1,2}時\d{0,2}分?'
                            r'|\d{1,2}:\d{2}に',
                            '', raw
                        ).strip()
                    if not goal_text:
                        goal_text = raw  # fallback: 原文をそのまま使う
                    job = job_scheduler.add_job(goal=goal_text, trigger=parsed, domain="research")
                    next_str = job_scheduler.format_next_run(job)
                    # デーモン稼働中の警告
                    daemon_status = HermesDaemon.get_status()
                    if daemon_status["running"]:
                        daemon_note = (
                            f"\n[bold yellow]⚠️  デーモン (PID={daemon_status['pid']}) が起動中です。[/bold yellow]\n"
                            "[yellow]ジョブはデーモンが実行するため、このターミナルに出力されません。[/yellow]\n"
                            "[yellow]/daemon stop してから再登録すると、このターミナルで実行されます。[/yellow]"
                        )
                    else:
                        daemon_note = "[dim]このターミナルのインラインスケジューラが自動実行します。[/dim]"
                    console.print(Panel(
                        f"[bold green]スケジュール登録完了[/bold green]\n"
                        f"ゴール: {goal_text[:80]}\n"
                        f"トリガー: {parsed} | 次回実行: {next_str}\n"
                        + daemon_note,
                        title="🕐 スケジュール登録",
                        border_style="green" if not daemon_status["running"] else "yellow",
                    ))
                    continue

                with console.status("[cyan]判断中...[/cyan]"):
                    intent_type, domain = _classify_intent(fast_llm, raw)

                if intent_type == "task":
                    console.print(
                        f"[yellow]タスクを検出しました（domain=[cyan]{domain}[/cyan]）。"
                        f"エージェントを起動します...[/yellow]"
                    )
                    # few-shot例をステートに注入してから実行
                    agi_core.self_improver.inject_into_state(
                        type("_S", (), {"working_memory": {}, "domain": domain, "user_goal": raw})()
                    )
                    _, last_state = _run_agent(llm, raw, domain, world_model=shared_world_model)
                    if last_state.world_model:
                        shared_world_model = last_state.world_model
                        agi_core.world_model = last_state.world_model
                    # 学習・省察フェーズ (自己改善の核心)
                    try:
                        _post_run_learning(agi_core, raw, domain, last_state)
                    except Exception as _learn_exc:
                        console.print(f"[dim][自己改善] エラー: {_learn_exc}[/dim]")
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
            except Exception as _exc:
                console.print(f"[bold red]エラーが発生しました:[/bold red] {_exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _top_exc:
        import logging as _logging
        _logging.basicConfig()
        _logging.getLogger(__name__).exception("CLI top-level error")
        print(f"\n[エラー] 予期しない例外が発生しました: {_top_exc}")
        print("cli.py を再起動してください。")
        sys.exit(1)
