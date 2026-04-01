# Hermes AGI Gen 6

Gen 5 の Plan → Act → Review ループを土台に、**4つのAGIコアモジュール**を追加した自律型エージェントフレームワーク。

## Gen 6 で新たに追加したもの

### 1. グローバル・ワークスペース (`consciousness.py`)
Baars (1988) の **Global Workspace Theory (GWT)** を実装。

- 8つの認知モジュールが `WorkspaceSignal` を送信
- `AttentionMechanism` が関連度・緊急度・確信度で注意競争を実施
- 勝者のコンテンツが全モジュールにブロードキャストされる
- 断片的な処理が**統合的・一貫した認知**になる

### 2. 価値体系 (`value_system.py`)
明示的な**価値整合フレームワーク**による倫理的意思決定。

| 価値 | 重み | 内容 |
|---|---|---|
| 安全性 | 1.0 | 破壊的操作を自動ブロック |
| 誠実さ | 0.95 | 正確・透明な情報提供 |
| 有益性 | 0.85 | 真に役立つ行動を選択 |
| 自律尊重 | 0.80 | ユーザーの判断を優先 |
| 継続学習 | 0.75 | 経験から学び続ける |

`rm -rf`、`drop table` 等の危険パターンは即時ブロック。行動ごとに効用スコアを算出し最善手を選択。

### 3. 8つの専門認知ロール (`cognitive_roles.py`)
Gen 5 の 3 ロールを **8 つの専門認知ロール**に拡張。目標の内容に応じて動的に編成する。

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

`select_roles_for_goal(goal)` が目標を解析し、必要なロールのみを自動選択する。

### 4. 予測的処理エンジン (`predictive_engine.py`)
Clark & Friston の**予測符号化理論**に基づく事前予測と学習。

- 行動実行前に LTM の失敗パターン・成功戦略から成功確率を予測
- 実行後に予測誤差を記録
- セッションを重ねるほど予測精度が自動向上

---

## アーキテクチャ概要

```
入力 (ユーザーの目標)
  │
  ▼
[GlobalWorkspace]  ← 全モジュールの注意競争・統合
  │
  ├─ [ValueSystem]  評価して危険な行動をブロック
  ├─ [select_roles_for_goal]  最適なロール編成を決定
  │
  ▼
[認知パイプライン]
  perceiver → memorist → ethicist → strategist → innovator → executor → critic
  各ロールの結果は次のロールのコンテキストに引き継がれる
  │
  ├─ 各ステップで PredictiveEngine が事前予測
  ├─ 各ステップで ValueSystem が倫理評価
  ├─ MetaCognition が行き詰まりを検出・戦略転換
  └─ LongTermMemory が成功/失敗を記録・次回に活用

  ▼
[GlobalWorkspace.broadcast()]  全結果を統合
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

# シングルエージェント
python run_agent.py --query "このプロジェクトの構造を調べてください"

# マルチ認知ロール (Gen 6 フル機能)
python cli.py
```

---

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `hermes_agi_gen/consciousness.py` | **[Gen 6]** グローバル・ワークスペース (GWT) |
| `hermes_agi_gen/value_system.py` | **[Gen 6]** 価値整合・倫理的意思決定 |
| `hermes_agi_gen/cognitive_roles.py` | **[Gen 6]** 8つの専門認知ロール定義 |
| `hermes_agi_gen/predictive_engine.py` | **[Gen 6]** 予測的処理エンジン |
| `hermes_agi_gen/agent_runner.py` | コアループ (Plan → Act → Review) |
| `hermes_agi_gen/orchestrator.py` | 8ロール対応オーケストレーター |
| `hermes_agi_gen/meta_cognition.py` | 行き詰まり検出・GoalQueue・自律ゴール生成 |
| `hermes_agi_gen/long_term_memory.py` | SQLite + セマンティック検索 (LTM) |
| `hermes_agi_gen/world_model.py` | 環境状態・因果グラフ追跡 |
| `hermes_agi_gen/self_improvement.py` | few-shot 学習・anti-pattern 記録 |
| `hermes_agi_gen/daemon.py` | 24/7 自律デーモン |
| `hermes_agi_gen/planner.py` | Chain-of-Thought プランナー |
| `hermes_agi_gen/executor.py` | ツール実行ディスパッチャー |
| `hermes_agi_gen/reviewer.py` | 結果評価・リカバリ提案 |
| `hermes_agi_gen/mistral_client.py` | LLM クライアント (Claude / Groq / Mistral / Ollama) |
| `hermes_agi_gen/state_store.py` | SQLite セッション永続化 |
| `cli.py` | インタラクティブ TUI |

---

## マルチ認知ロール実行

```python
from hermes_agi_gen import AgentOrchestrator, MistralClient

llm = MistralClient()  # 環境変数で自動選択
orch = AgentOrchestrator(llm=llm)

# 通常実行 (目標に応じてロールを自動選択)
result = orch.run("このプロジェクトの構造を調べて改善案をまとめてください")
print(result)

# 予測情報付き実行
result = orch.run_with_prediction("テストカバレッジを改善してください")
print(f"結果: {result['result']}")
print(f"予測精度: {result['prediction_accuracy']:.1%}")

# システム状態確認
print(orch.get_system_status())
```

---

## Gen 6 API リファレンス

### `GlobalWorkspace`

```python
from hermes_agi_gen import GlobalWorkspace, WorkspaceSignal, SignalSource

ws = GlobalWorkspace()
ws.receive(WorkspaceSignal(
    source=SignalSource.PERCEIVER,
    content="目標を解析しました",
    relevance=0.9,
    urgency=0.7,
    confidence=0.85,
))
event = ws.broadcast()   # 注意競争を実施
print(event.winner.source.value)  # 勝者モジュール
```

### `ValueSystem`

```python
from hermes_agi_gen import ValueSystem

vs = ValueSystem()
assessment = vs.assess("CMD: rm -rf /")
print(assessment.is_blocked)        # True
print(assessment.recommendation)    # ブロック理由

score = vs.utility_score("CMD: ls -la", goal_relevance=0.8)
print(score)  # 効用スコア (0.0〜1.0)
```

### `PredictiveEngine`

```python
from hermes_agi_gen import PredictiveEngine

pe = PredictiveEngine(ltm=ltm)  # LTMと連携
pred = pe.predict("CMD: python3 -m pytest", goal="テストを実行")
print(pred.success_probability)  # 成功確率
print(pred.should_proceed)       # 実行推奨か

pe.record_outcome(pred, actual_outcome="...", actual_success=True)
print(pe.get_accuracy())         # 予測精度
```

### `select_roles_for_goal`

```python
from hermes_agi_gen import select_roles_for_goal, decompose_into_roles

roles = select_roles_for_goal("バグを修正して実装してください")
# → ['perceiver', 'memorist', 'strategist', 'executor', 'critic']

subtasks = decompose_into_roles("セキュリティ脆弱性を調査してください")
# → [{"role": "perceiver", "task": "..."}, {"role": "ethicist", "task": "..."}, ...]
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

## インタラクティブ CLI

```bash
python cli.py
```

| コマンド | 説明 |
|---|---|
| `/generate <説明>` | 自然言語からコードを生成 |
| `/review` | コードを貼り付けてレビュー |
| `/provider` | 現在の LLM プロバイダーを表示 |
| `/help` | コマンド一覧 |
| `/quit` | 終了 |

---

## 不具合修正ログ

実運用で発見・修正したバグと改善の記録。

### バグ修正

| # | ファイル | 問題 | 修正 |
|---|---|---|---|
| 1 | `orchestrator.py` | `tree.execute_tree()` — `GoalTree` にそのメソッドは存在せずクラッシュ | `self.hierarchical_planner.execute_tree()` に修正 |
| 2 | `hierarchical_planner.py` | ゴール分解プロンプトが旧3ロール（Gen 6の8ロールを知らない） | プロンプトを8認知ロール対応に更新 |
| 3 | `orchestrator.py` | 階層型がデフォルトのため `_run_cognitive_pipeline`（8ロール）が使われない | ロール数で実行パスを自動振り分け |
| 4 | `orchestrator.py` | パイプラインのコンテキスト引き継ぎで `sender`（常に"orchestrator"）を使っていた | `receiver`（ロール名）を使うよう修正 |
| 5 | `cognitive_roles.py` | 単純な「ls」「一覧を見せて」でも4ロール全部起動して低速 | 短い/単純クエリを検出して `executor` のみに絞る早期判定を追加 |
| 6 | `mistral_client.py` | qwen3の `<think>...</think>` ブロックがJSONパースと CoT解析を妨害 | `chat()` / `chat_json()` 両方で `<think>` タグを除去 |

### ロールに応じた実行パスの切り替え

単純なクエリで複数ロールが無駄に起動する問題を解消。目標の複雑さに応じて自動的に実行パスを選択する。

```
1ロール  → 単一ロール直接実行（高速・単純クエリ向け）
2〜3ロール → 認知パイプライン（順次実行・コンテキスト引き継ぎ）
4ロール以上 → 階層型ゴールツリー（並列実行・依存関係管理）
```

**ロール自動選択の例:**

| クエリ | 選択ロール |
|---|---|
| `ls` / `一覧を教えて` | `executor` のみ |
| `バグを修正して` | `perceiver → strategist → executor → critic` |
| `削除してください` | `perceiver → ethicist → strategist → executor` |
| `コードを調査して改善案を実装` | `perceiver → memorist → strategist → innovator → executor → critic` |

### Groq レートリミット対応

Groq API が `429 Too Many Requests` を返した場合の挙動：

```
retry-after ≤ 60秒  → 指定秒数待ってリトライ（最大3回）
retry-after > 60秒  → 待たずに即座に Ollama へフォールバック
リトライ3回失敗    → Ollama へフォールバック
```

CLIでの表示例：
```
[Groq] レートリミット — 10秒後にリトライ (1/3)...
[フォールバック] Ollama (mistral) を使用します
```

長時間（1520秒など）のフリーズは発生しない。

---

## Gen 5 からの移行

Gen 5 の既存コードはそのまま動作します。`AgentOrchestrator` に渡す `llm` が同じなら、挙動は変わりません。

Gen 6 の新機能を使いたい場合は `orchestrator.run_with_prediction()` または `get_system_status()` を呼び出してください。`select_roles_for_goal()` はオーケストレーター内部で自動的に使われます。

---

## 設計思想

Gen 6 は以下の認知科学・AI安全の理論を実装の指針としています：

- **Global Workspace Theory** (Baars, 1988) — 統合的意識の計算モデル
- **Predictive Coding** (Clark & Friston, 2013) — 予測誤差からの学習
- **Value Alignment** — 明示的な価値体系による安全な自律行動
- **Cognitive Role Specialization** — 専門化した認知モジュールの協調
