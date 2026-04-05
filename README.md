# Hermes AGI Gen 7

Gen 6 の統合認知アーキテクチャに、**能動的自己省察ループ**と**統合AGI認知コア**を追加した自律型エージェントフレームワーク。

> Gen 6: タスク反応型AGI（入力→処理→出力）  
> Gen 7: 自律認知型AGI（知覚→省察→注意→計画→行動→学習 の継続的ループ）

---

## Gen 7 で新たに追加・強化したもの

### 5. 自己省察エンジン (`reflection_engine.py`)

AGIの「考える時間」を実装。Nゴールごとに能動的な反省フェーズを実行し、経験から戦略を自律更新する。

- **`ReflectionEngine`**: 適応的インターバルで省察サイクルを実行
  - 成功率 < 0.3（苦戦中）→ 3ゴールごとに省察（頻度を高めて課題を早期発見）
  - 成功率 > 0.75（好調）→ 8ゴールごとに省察（余裕を持った振り返り）
  - それ以外 → デフォルト（5ゴールごと）
- **`Insight`**: 省察で抽出された洞察（`strength` / `weakness` / `gap` / `pattern` / `opportunity`）
- **`GrowthMetrics`**: 成長指標（成功率・知識量・反省回数・予測精度）
- **持続的課題の検出**: 過去3回の省察で同一の洞察が繰り返された場合、`[持続的課題]` として確信度を昇格させ行動を促す

省察フロー:
```
LTM (成功/失敗パターン・学習事実)
  ↓
ルールベース + LLM深層省察
  ↓
持続的課題の検出 (前回省察との重複チェック)
  ↓
Insight リスト (強み・弱み・知識ギャップ・改善機会)
  ↓
戦略的ゴール生成 → GoalQueueに追加
```

### 6. 統合AGI認知コア (`agi_core.py`)

すべての認知モジュールを統合する単一の認知ループ。

- **`AGICore`**: 統一認知サイクルのエントリポイント
- **`AGIIdentity`**: 永続的自己同一性（能力プロファイル・自己評価・価値観）。経験とともに自己評価が自動更新される
- **SelfModifier 統合**: 3回の省察ごとに洞察ベースのコード自己修正を自律的に試みる
- **完全な自己改善ループ**: `inject_into_state` → 実行 → `analyze_session` → 省察 が1サイクルとして繋がった

認知サイクル:
```
知覚 (Perceive)   — 世界モデルをファイルシステムにグラウンド
  ↓
倫理評価           — ValueSystem でブロックチェック
  ↓
注意選択 (Attend) — GlobalWorkspace が認知リソースを配分
  ↓
予測 (Predict)    — PredictiveEngine が成功確率を算出
  ↓
few-shot 注入     — SelfImprovementEngine が過去の成功例をワーキングメモリへ
  ↓
実行 (Act)        — HermesAgentV9 が認知ロールで実行
  ↓
学習 (Learn)      — 予測誤差を記録・LTMに経験蓄積・軌跡からfew-shot例を抽出
  ↓
省察 (Reflect)    — 適応的インターバルで洞察生成 → 戦略的ゴールを自律生成
  ↓                  (3回に1回) 洞察に基づくコード自己修正を試みる
```

### Gen 7 の改善点

| モジュール | 変更内容 |
|---|---|
| `agi_core.py` | `analyze_session()` / `inject_into_state()` を統合 — 自己改善ループが実際に機能するよう接続 |
| `agi_core.py` | `SelfModifier` を統合 — 3省察ごとに洞察ベースの自律コード修正 |
| `agi_core.py` | `GrowthMetrics.prediction_accuracy` を `PredictiveEngine` と接続 — 常に0だった値が正確に反映される |
| `reflection_engine.py` | 適応的インターバル — 成功率に応じて省察頻度を動的調整 |
| `reflection_engine.py` | 持続的課題の検出 — 繰り返し現れる洞察を昇格させ行動を促す |
| `self_improvement.py` | クロスドメイン few-shot 転用 — ドメイン固有例不足時に他ドメインの知識を0.7倍割引で補完 |
| `executor.py` | `cmd.split()` → `shlex.split()` に修正（クォート・複雑引数を正しく処理） |
| `world_model.py` | `initialize_from_filesystem()` 追加 — ファイルシステム・Git状態に接地（グラウンディング） |
| `daemon.py` | `ReflectionEngine` を統合 — 5ゴールごとに省察し、戦略的ゴールを自律生成 |
| `cli.py` | 起動時に世界モデルをグラウンディング、`/status` に AGI Identity 表示を追加 |

---

## Gen 6 モジュール (継続)

### 1. グローバル・ワークスペース (`consciousness.py`)
Baars (1988) の **Global Workspace Theory (GWT)** を実装。

- 8つの認知モジュールが `WorkspaceSignal` を送信
- `AttentionMechanism` が関連度・緊急度・確信度で注意競争を実施
- 勝者のコンテンツが全モジュールにブロードキャストされる

### 2. 価値体系 (`value_system.py`)
明示的な**価値整合フレームワーク**による倫理的意思決定。

| 価値 | 重み | 内容 |
|---|---|---|
| 安全性 | 1.0 | 破壊的操作を自動ブロック |
| 誠実さ | 0.95 | 正確・透明な情報提供 |
| 有益性 | 0.85 | 真に役立つ行動を選択 |
| 自律尊重 | 0.80 | ユーザーの判断を優先 |
| 継続学習 | 0.75 | 経験から学び続ける |

### 3. 8つの専門認知ロール (`cognitive_roles.py`)

| ロール | 担当 |
|---|---|
| `perceiver` | 入力理解・意図解釈・要件明確化 |
| `memorist` | ローカルファイル調査・知識収集 |
| `ethicist` | 安全性・倫理的問題の評価 |
| `strategist` | 戦略的計画・ゴール分解 |
| `innovator` | 創造的・代替的アプローチの提案 |
| `executor` | コード実行・ファイル操作 |
| `critic` | 品質評価・改善提案 |
| `goal_manager` | ゴール優先付け・ロードマップ管理 |

### 4. 予測的処理エンジン (`predictive_engine.py`)
Clark & Friston の**予測符号化理論**に基づく事前予測と学習。

---

## アーキテクチャ概要

```
入力 (ユーザーの目標)
  │
  ▼
[AGICore]
  │
  ├─ [WorldModel.initialize_from_filesystem]  環境を実態にグラウンド
  ├─ [ValueSystem]  倫理評価・危険な行動をブロック
  ├─ [GlobalWorkspace]  注意競争・認知リソース配分
  ├─ [PredictiveEngine]  成功確率を事前予測
  ├─ [SelfImprovementEngine.inject_into_state]  few-shot例を注入
  │
  ▼
[認知パイプライン]
  perceiver → memorist → ethicist → strategist → innovator → executor → critic
  各ロールの結果は次のロールのコンテキストに引き継がれる
  │
  ├─ MetaCognition が行き詰まりを検出・戦略転換
  └─ LongTermMemory が成功/失敗を記録・次回に活用
  │
  ▼
[学習フェーズ]
  ├─ PredictiveEngine  予測誤差を記録
  ├─ SelfImprovementEngine.analyze_session  few-shot例・anti-patternを抽出
  └─ record_session_performance  ドメイン別パフォーマンストレンドを更新
  │
  ▼
[ReflectionEngine]  (適応的インターバル: 成功率に応じて3〜8ゴールごと)
  LTM分析 → 持続的課題検出 → Insight生成 → 戦略的ゴール → GoalQueue
  │
  ├─ AGIIdentity.self_assessment を更新 (prediction_accuracy 含む)
  └─ (3省察ごと) SelfModifier で洞察ベースのコード自己修正
  │
  ▼
出力 (統合された最終回答)
```

---

## すぐ試す

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# インタラクティブ CLI (推奨)
python cli.py

# シングルエージェント
python run_agent.py --query "このプロジェクトの構造を調べてください"
```

---

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `hermes_agi_gen/agi_core.py` | **[Gen 7]** 統合AGI認知コア・AGIIdentity・SelfModifier統合 |
| `hermes_agi_gen/reflection_engine.py` | **[Gen 7]** 能動的自己省察エンジン（適応的インターバル・持続的課題検出） |
| `hermes_agi_gen/self_improvement.py` | **[Gen 7]** few-shot学習・anti-pattern記録・クロスドメイン転用 |
| `hermes_agi_gen/self_modifier.py` | **[Gen 7]** ソースコード自己修正エンジン（AGICoreに統合済み） |
| `hermes_agi_gen/consciousness.py` | **[Gen 6]** グローバル・ワークスペース (GWT) |
| `hermes_agi_gen/value_system.py` | **[Gen 6]** 価値整合・倫理的意思決定 |
| `hermes_agi_gen/cognitive_roles.py` | **[Gen 6]** 8つの専門認知ロール定義 |
| `hermes_agi_gen/predictive_engine.py` | **[Gen 6]** 予測的処理エンジン |
| `hermes_agi_gen/agent_runner.py` | コアループ (Plan → Act → Review) |
| `hermes_agi_gen/orchestrator.py` | 8ロール対応オーケストレーター |
| `hermes_agi_gen/meta_cognition.py` | 行き詰まり検出・GoalQueue・自律ゴール生成 |
| `hermes_agi_gen/long_term_memory.py` | SQLite + セマンティック検索 (LTM) |
| `hermes_agi_gen/world_model.py` | 環境状態・因果グラフ追跡・FS グラウンディング |
| `hermes_agi_gen/daemon.py` | 24/7 自律デーモン (省察ループ付き) |
| `hermes_agi_gen/planner.py` | Chain-of-Thought プランナー |
| `hermes_agi_gen/executor.py` | ツール実行ディスパッチャー |
| `hermes_agi_gen/reviewer.py` | 結果評価・リカバリ提案 |
| `hermes_agi_gen/web_search.py` | DuckDuckGo 検索 (ddgs ライブラリ + HTML スクレイピング fallback) |
| `hermes_agi_gen/mistral_client.py` | LLM クライアント (Claude / Groq / Mistral / Ollama) |
| `hermes_agi_gen/state_store.py` | SQLite セッション永続化 |
| `cli.py` | インタラクティブ TUI・自然言語スケジューラ・URL取得＆翻訳パイプライン |

---

## API リファレンス

### `AGICore` (Gen 7)

統合AGI認知コア。単一のエントリポイントですべての認知モジュールを協調させる。

```python
from hermes_agi_gen import AGICore
from hermes_agi_gen.mistral_client import MistralClient

llm = MistralClient()
core = AGICore(llm=llm, reflection_interval=5)

# ゴールを認知サイクル全体で処理
result = core.run_goal("このプロジェクトの構造を調べて改善案を提案してください")
print(result["result"])
print(result["identity"])        # AGI Identity サマリー
print(result["insights"])        # 省察で生成された洞察
print(result["new_goals"])       # 戦略的ゴール追加数

# AGI全体状態の確認
core.print_status()
```

### `ReflectionEngine` (Gen 7)

```python
from hermes_agi_gen import ReflectionEngine, LongTermMemory

ltm = LongTermMemory()
engine = ReflectionEngine(llm=llm, reflection_interval=5)

# 省察の実行 (持続的課題の自動検出付き)
insights = engine.reflect(ltm)
for insight in insights:
    print(f"[{insight.category}] {insight.content} (確信={insight.confidence:.0%})")

# 洞察から戦略的ゴールを生成
goals = engine.generate_strategic_goals(insights, ltm)

# 成長指標の取得 (prediction_accuracy 含む)
metrics = engine.compute_growth_metrics(ltm)
print(metrics.summary())

# 適応的インターバルによる省察トリガー
if engine.should_reflect(recent_success_rate=0.4):
    insights = engine.reflect(ltm)
```

### `SelfImprovementEngine` (Gen 7)

```python
from hermes_agi_gen.self_improvement import SelfImprovementEngine

improver = SelfImprovementEngine(llm=llm)

# 実行前: few-shot例とanti-patternをワーキングメモリへ注入
improver.inject_into_state(state)

# 実行後: 軌跡から学習
improver.analyze_session(state)
improver.record_session_performance(session_id, goal, domain, score=1.0)

# クロスドメイン転用: 該当ドメインの例が少ない場合は他ドメインから補完
examples = improver.get_best_examples(domain="analysis")

# パフォーマンストレンドを取得 (適応的インターバルに活用)
trend = improver.get_performance_trend(window=10)
```

### `AgentOrchestrator` (Gen 6)

```python
from hermes_agi_gen import AgentOrchestrator, MistralClient

llm = MistralClient()
orch = AgentOrchestrator(llm=llm)

# 通常実行
result = orch.run("このプロジェクトの構造を調べて改善案をまとめてください")

# 予測情報付き実行
result = orch.run_with_prediction("テストカバレッジを改善してください")
print(f"予測精度: {result['prediction_accuracy']:.1%}")

# システム状態確認
print(orch.get_system_status())
```

### `GlobalWorkspace` (Gen 6)

```python
from hermes_agi_gen import GlobalWorkspace, WorkspaceSignal, SignalSource

ws = GlobalWorkspace()
ws.receive(WorkspaceSignal(
    source=SignalSource.PERCEIVER,
    content="目標を解析しました",
    relevance=0.9, urgency=0.7, confidence=0.85,
))
event = ws.broadcast()
print(event.winner.source.value)  # 勝者モジュール
```

### `ValueSystem` (Gen 6)

```python
from hermes_agi_gen import ValueSystem

vs = ValueSystem()
assessment = vs.assess("CMD: rm -rf /")
print(assessment.is_blocked)        # True
print(assessment.recommendation)    # ブロック理由
```

### `select_roles_for_goal` (Gen 6)

```python
from hermes_agi_gen import select_roles_for_goal

roles = select_roles_for_goal("バグを修正して実装してください")
# → ['perceiver', 'memorist', 'strategist', 'executor', 'critic']
```

---

## インタラクティブ CLI

```bash
python cli.py
```

| コマンド | 説明 |
|---|---|
| `/run <目標>` | エージェントモードで実行 |
| `/orch <目標>` | マルチ認知ロールで実行 |
| `/status` | AGI Identity・成長指標・世界モデル状態を表示 |
| `/goals` | 自律ゴールキューを表示 |
| `/world` | 世界モデルの状態を表示 |
| `/improve` | 自己改善レポートを表示 |
| `/schedule` | スケジュール済みジョブ一覧を表示 |
| `/daemon start` | 24/7 自律デーモンをバックグラウンドで起動 |
| `/daemon stop` | デーモンを停止 (インラインスケジューラ使用時は先に停止) |
| `/daemon status` | デーモンの稼働状態を確認 |
| `/daemon log` | デーモンのログを表示 |
| `/help` | コマンド一覧 |
| `/quit` | 終了 |

### 自然言語スケジュール

フリーテキスト入力で時刻付きタスクを自動スケジュール登録できます。

```
# 一回限り (絶対時刻)
2026年4月7日午前9時になったら、天気予報を取得して要約してください

# 毎日
毎日午前8時になったら、HNのAIニュースを翻訳して~/Desktop/AI_News/に保存して

# インターバル
30分ごとにシステム状態を確認して
```

`〜になったら` より前の時刻表現を自動検出し `SCHEDULE_AT:` トリガーに変換します。  
登録後はターミナルを開いたままにしておくことで **インラインスケジューラ** (20秒ごとにポーリング) が自動実行します。

> **注意**: `/daemon start` で起動した外部デーモンとインラインスケジューラが同時に動くと  
> ジョブが二重消費される場合があります。スケジュール機能を使う際は先に `/daemon stop` してください。

### URL + 翻訳 + ファイル保存パイプライン

URL・保存先・翻訳の3要素が揃ったゴールは、LLMを経由せず動作確認済みの Python スクリプトを直接実行します。

```
https://news.ycombinator.com/ からAI関連のニュースを一つ選んで
日本語に翻訳して800字程度にまとめて、~/Desktop/AI_News/にtxtファイルにして入れてください
```

処理フロー:
1. HN ページからAIキーワードで記事を選択
2. 記事本文を取得 (`<script>`/`<style>` 除去後)
3. Groq API で日本語タイトル＋要約を生成 (文字数はゴールから自動抽出)
4. `~/Desktop/AI_News/AI_News_MM-DD_HHMM.txt` に保存

出力フォーマット:
```
【AI News】{日本語タイトル}
原題: {English title}
出典: {URL}

{日本語要約 (指定文字数)}

(Hacker Newsより取得 YYYY-MM-DD HH:MM)
```

---

## LLM プロバイダー

`MistralClient` は以下の優先順で自動選択します：

| 優先順 | 環境変数 | バックエンド | デフォルトモデル |
|---|---|---|---|
| 1 | `GROQ_API_KEY` | Groq | `llama-3.3-70b-versatile` |
| 2 | `MISTRAL_API_KEY` | Mistral API | `mistral-small-latest` |
| 3 | なし | Ollama (ローカル) | `mistral` |

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
# または
echo "GROQ_API_KEY=gsk_..." > .env
```

---

## 不具合修正ログ

### CLI・スケジューラ・検索改善 (2026-04)

| # | ファイル | 問題 | 修正 |
|---|---|---|---|
| 11 | `cli.py` | 入力時にターミナルがクラッシュする | 各コマンドハンドラに `try/except` を追加、トップレベルにも保護を追加 |
| 12 | `cli.py` | `/tools` コマンドで `args` 変数が argparse Namespace を上書きしてクラッシュ | 変数名を `tools_args` に変更 |
| 13 | `cli.py` | 「〜時になったら〜してください」を即時実行しようとする | 時刻表現を検出して `SCHEDULE_AT:` に自動変換するプリプロセッサを追加 |
| 14 | `planner.py` | `SCHEDULE_AT:` アクションが `_extract_action_from_cot()` で無視される | `SCHEDULE_AT:` / `FETCH:` を認識プレフィックスに追加 |
| 15 | `web_search.py` | DuckDuckGo HTML スクレイピングが CAPTCHA でブロックされる | `ddgs` ライブラリを優先使用し、失敗時のみ HTML スクレイピングにフォールバック |
| 16 | `cli.py` | スケジュール実行時にファイルが作成されない / 内容が空 | LLM生成スクリプトをバイパスし、動作確認済みの Python スクリプトを `state.current_plan` に直接注入 |
| 17 | `cli.py` | 保存先パスに日本語が混入する (`~/Desktop/AI_News/にtxt...`) | URL・パス抽出の正規表現を ASCII 文字限定 (`[a-zA-Z0-9/_.-]+`) に変更 |
| 18 | `executor.py` | Python スクリプト実行が 30 秒でタイムアウト (HTTP 3リクエスト合計で超過) | `_PY_TIMEOUT` を 30 → 90 秒に延長 |
| 19 | `agent_runner.py` | PYTHON ステップのエラーが無音で飲み込まれる | `stderr` / `stdout` を実行直後にターミナルへ表示 |
| 20 | `cli.py` | 生成スクリプト内の `f'{\"key\"...}'` が `SyntaxError` になる | プロンプト構築を f-string から文字列連結に変更してクォート衝突を回避 |
| 21 | `cli.py` | 要約文字数指示が「100字」固定でゴールの指定を無視する | ゴールから `(\d+)[字文]` を抽出し `max_tokens`・本文取得量・プロンプト指示に動的反映 |
| 22 | `cli.py` | LLM が文字数指示に従わず短い要約を返す | プロンプトを「短い要約は不可。詳細・背景・技術的内容をすべて含めること」と強制化 |
| 23 | `cli.py` | 記事本文に JavaScript が大量混入して翻訳品質が低下 | `<script>`/`<style>` タグをタグ除去前に削除するステップを追加 |
| 24 | `cli.py` | スケジュール完了後にプロンプトが戻らない | バックグラウンドスレッドの `finally` ブロックで `hermes: ` プロンプトを再描画 |

### Gen 7

| # | ファイル | 問題 | 修正 |
|---|---|---|---|
| 7 | `executor.py` | `cmd.split()` がスペースを含む引数・クォートを正しく処理しない | `shlex.split(cmd)` に変更 |
| 8 | `agi_core.py` | `analyze_session()` が呼ばれず few-shot 学習が機能していなかった | `run_goal()` に `inject_into_state` → `analyze_session` → `record_session_performance` を追加 |
| 9 | `agi_core.py` | `SelfModifier` が孤立しており自律コード修正が機能しなかった | `AGICore` に統合し、3省察ごとに自動呼び出し |
| 10 | `agi_core.py` | `GrowthMetrics.prediction_accuracy` が常に 0.0 だった | `PredictiveEngine.get_accuracy()` と接続 |

### Gen 6

| # | ファイル | 問題 | 修正 |
|---|---|---|---|
| 1 | `orchestrator.py` | `tree.execute_tree()` — `GoalTree` にそのメソッドは存在せずクラッシュ | `self.hierarchical_planner.execute_tree()` に修正 |
| 2 | `hierarchical_planner.py` | ゴール分解プロンプトが旧3ロール（Gen 6の8ロールを知らない） | プロンプトを8認知ロール対応に更新 |
| 3 | `orchestrator.py` | 階層型がデフォルトのため `_run_cognitive_pipeline`（8ロール）が使われない | ロール数で実行パスを自動振り分け |
| 4 | `orchestrator.py` | パイプラインのコンテキスト引き継ぎで `sender`（常に"orchestrator"）を使っていた | `receiver`（ロール名）を使うよう修正 |
| 5 | `cognitive_roles.py` | 単純な「ls」「一覧を見せて」でも4ロール全部起動して低速 | 短い/単純クエリを検出して `executor` のみに絞る早期判定を追加 |
| 6 | `mistral_client.py` | qwen3の `<think>...</think>` ブロックがJSONパースと CoT解析を妨害 | `chat()` / `chat_json()` 両方で `<think>` タグを除去 |

### ロールに応じた実行パスの切り替え

```
1ロール    → 単一ロール直接実行（高速・単純クエリ向け）
2〜3ロール → 認知パイプライン（順次実行・コンテキスト引き継ぎ）
4ロール以上 → 階層型ゴールツリー（並列実行・依存関係管理）
```

---

## 設計思想

Hermes AGI は以下の認知科学・AI安全の理論を実装の指針としています：

- **Global Workspace Theory** (Baars, 1988) — 統合的意識の計算モデル
- **Predictive Coding** (Clark & Friston, 2013) — 予測誤差からの学習
- **Active Inference** (Friston, 2010) — 能動的な世界モデル更新
- **Value Alignment** — 明示的な価値体系による安全な自律行動
- **Cognitive Role Specialization** — 専門化した認知モジュールの協調
- **Metacognition** — 自己の認知プロセスへの省察と更新
