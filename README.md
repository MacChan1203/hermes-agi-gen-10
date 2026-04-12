# Hermes AGI Gen 9

Gen 7 の統合認知アーキテクチャを大幅に拡張し、**自律的内発動機**・**メタ学習**・**内部対話**・**永続的自己同一性**を備えた自律型AGIフレームワーク。

> Gen 6: タスク反応型AGI（入力→処理→出力）  
> Gen 7: 自律認知型AGI（知覚→省察→注意→計画→行動→学習 の継続的ループ）  
> Gen 8: 実験駆動型AGI（洞察→コード自己修正→メトリクス検証 の AutoResearch ループ）  
> **Gen 9: 自律進化型AGI（内発動機→メタ学習→内部対話→適応的行動→夢 の完全自律サイクル）**

---

## Gen 9 で追加した7つの変革

### 1. 永続的アイデンティティ (`agi_core.py`)

AGIIdentity がセッションをまたいで永続化。経験・能力・自己評価が蓄積され続ける。

- SQLite (LTM) に Identity を自動保存・復元
- セッション数・処理ゴール数・成功率が生涯を通じて追跡される
- 新能力の動的発見 (`discover_capability()`)
- 8次元の自己評価: reasoning / planning / execution / learning / reflection / autonomy / meta_learning / creativity

```python
core = AGICore(llm=llm)
print(core.identity.profile_summary())
# → Hermes AGI Gen 9 | 稼働: 142.3h | セッション: 47 | 処理ゴール: 312 | 成功率: 78%
```

### 2. 内発的動機エンジン (`intrinsic_motivation.py`)

外部ゴールなしでも自律的に行動を生成する。5つの動機源:

| 動機源 | 重み | 説明 |
|--------|------|------|
| 好奇心 (Curiosity) | 30% | 知識ギャップを検出し、探索ゴールを生成 |
| 達成動機 (Competence) | 25% | 自己評価の低い能力を鍛えるゴールを生成 |
| エントロピー低減 (Entropy) | 20% | WorldModel の不確実領域を優先探索 |
| 社会性 (Social) | 15% | ユーザーへの有益性を最大化 |
| 恒常性 (Homeostasis) | 10% | 長期間未使用のモジュールを活性化 |

LLM不要のルールベース版を基本とし、LLMがあれば高品質化する。

### 3. メタ学習層 (`meta_learning.py`)

「どう学ぶかを学ぶ」メタ認知エンジン。

- **戦略レジストリ**: 8つの既定戦略 (divide_and_conquer, depth_first, analogy, etc.) + ユーザー定義戦略をSQLite管理
- **UCB1アルゴリズム**: 探索と活用のバランスを取る多腕バンディットで最適戦略を選択
- **転移学習**: ドメインAで成功した戦略をドメインBに自動適用（類似度閾値付き）
- **適応的探索率**: 最近の改善率から探索パラメータを動的調整

```python
ml = MetaLearner()
strategy = ml.select_strategy("coding")
# → StrategyRecord(name='observe_then_act', ucb_score=1.42)
ml.record_outcome("coding", strategy.name, "テスト作成", reward=0.9)
```

### 4. 内部対話システム (`inner_dialogue.py`)

高リスク・高不確実タスクに対して、実行前に多角的検討を行う。

- **批判者 (Critic)**: リスク・欠陥・見落としを指摘
- **革新者 (Innovator)**: 創造的代替アプローチを提案
- **倫理家 (Ethicist)**: 安全性・公正性・透明性を評価
- **戦略家 (Strategist)**: 全意見を統合し、最適戦略をまとめる

発動条件: 予測確信度 < 40% / 倫理スコア > 50% / 自己修正タスク / 複雑なゴール

```python
dialogue = InnerDialogue(llm=llm)
result = dialogue.deliberate("本番DBのスキーマを変更する")
print(result.consensus_level)   # 0.35 (低い合意度 → 慎重に)
print(result.key_concerns)      # ["データ損失リスク", "ロールバック計画が不明"]
print(result.should_proceed)    # False
```

### 5. 自己修正の拡張 (`self_modifier.py`)

- **ホワイトリスト拡張**: 7 → 14ファイル (+cognitive_roles, consciousness, predictive_engine, reflection_engine, intrinsic_motivation, meta_learning, inner_dialogue)
- **学習済み修正パターンDB**: 過去の成功パッチを類似洞察に再適用
- **リスク段階制**: low→自動適用 / medium→テスト必須 / high→ログ記録+ユーザー確認要求
- **高リスク提案の管理**: `get_pending_high_risk()` / `approve_high_risk(id)` でユーザーが後から確認・承認

### 6. 資源認識型プランニング (`world_model.py`, `executor.py`)

- 全ツール実行の所要時間・出力サイズ・成功/失敗を自動記録
- `estimate_tool_cost()`: 過去の実績からツールの予想コストを算出
- `estimate_goal_complexity()`: ゴールの複雑度を推定し、推奨 max_iterations (3〜12) を算出
- 不確実性マップ: 領域ごとの不確実性スコアを追跡、内発動機エンジンと連携

### 7. 認知サイクルの高度化 (`agi_core.py`)

- **夢フェーズ (Dream)**: GoalQueue空時にLTMの知識を再構成・統合、転移学習候補を探索
- **適応的実行深度**: タスク複雑度に応じて max_iterations を動的設定 (3〜12)
- **autonomous_loop()**: GoalQueueからゴールを自動消化する連続認知モード
- **GlobalWorkspace拡張**: 3つの新SignalSource (MOTIVATOR, META_LEARNER, DELIBERATOR)

---

## アーキテクチャ概要

```
入力 (ユーザーの目標 / 内発動機 / GoalQueue)
  │
  ▼
[1. 知覚] WorldModel.initialize_from_filesystem — 環境を実態にグラウンド
  │
  ▼
[2. 内部対話] InnerDialogue.deliberate — 高リスク時のみ多角的検討
  │         批判者 → 革新者 → 倫理家 → 戦略家 → 合意形成
  │         ゴールの洗練 / 実行中止判断
  │
  ▼
[3. 倫理評価] ValueSystem.assess — 危険な行動をブロック
  │
  ▼
[4. 注意選択] GlobalWorkspace.broadcast — 11シグナルソースが注意競争
  │           PERCEIVER, STRATEGIST, EXECUTOR, CRITIC, MEMORIST,
  │           GOAL_MANAGER, INNOVATOR, ETHICIST,
  │           MOTIVATOR, META_LEARNER, DELIBERATOR
  │
  ▼
[5. メタ学習] MetaLearner.select_strategy — UCB1で最適戦略を選択
  │
  ▼
[6. 予測] PredictiveEngine.predict — 成功確率を算出
  │
  ▼
[7. 行動] HermesAgentV9 — 適応的実行深度 (3〜12イテレーション)
  │       認知パイプライン: perceiver → memorist → ethicist →
  │       strategist → innovator → executor → critic
  │       全ツール実行にコスト計測を注入
  │
  ▼
[8. 学習] 予測誤差記録 + メタ学習戦略更新 + few-shot抽出 + LTM蓄積
  │
  ▼
[9. 内発動機] IntrinsicMotivationEngine — 好奇心・達成動機からゴール自動生成
  │
  ▼
[10. 省察] ReflectionEngine — 適応的インターバルで洞察生成
  │        (3省察ごと) ExperimentRunner + SelfModifier で自律コード修正
  │        (10ゴールごと) MetaLearner.find_transfer_candidates で転移学習
  │
  ▼
[11. 夢] Dream Phase — GoalQueue空時にLTM知識統合・転移学習探索
  │
  ▼
Identity永続化 → [ループ / autonomous_loop]
```

---

## すぐ試す

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# インタラクティブ CLI (推奨)
python cli.py

# モデル指定
python cli.py --model qwen3      # Ollama
python cli.py --model groq       # Groq API

# デーモンモード (24/7 自律実行)
python cli.py --daemon

# Python から直接
python3 -c "
from hermes_agi_gen.agi_core import AGICore
from hermes_agi_gen.mistral_client import MistralClient

llm = MistralClient()
core = AGICore(llm=llm)
result = core.run_goal('このプロジェクトの構造を分析して')
print(result['success'], result['strategy'])

# 自律ループ (GoalQueue消化 + 内発動機 + 夢フェーズ)
results = core.autonomous_loop(max_cycles=10)
"
```

---

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `hermes_agi_gen/agi_core.py` | **[Gen 9]** 統合AGI認知コア・永続Identity・autonomous_loop・夢フェーズ |
| `hermes_agi_gen/intrinsic_motivation.py` | **[Gen 9]** 内発的動機エンジン（好奇心・達成・恒常性・エントロピー・社会性） |
| `hermes_agi_gen/meta_learning.py` | **[Gen 9]** メタ学習層（UCB1戦略選択・転移学習・適応的探索率） |
| `hermes_agi_gen/inner_dialogue.py` | **[Gen 9]** 内部対話システム（批判者・革新者・倫理家・戦略家の合意形成） |
| `hermes_agi_gen/self_modifier.py` | **[Gen 9]** ソースコード自己修正（14ファイル対応・学習パターン・リスク段階制） |
| `hermes_agi_gen/world_model.py` | **[Gen 9]** 世界モデル（資源コスト追跡・複雑度推定・不確実性マップ） |
| `hermes_agi_gen/experiment_runner.py` | **[Gen 8]** AutoResearch方式の実験ループ |
| `hermes_agi_gen/reflection_engine.py` | **[Gen 7]** 能動的自己省察エンジン（適応的インターバル・持続的課題検出） |
| `hermes_agi_gen/self_improvement.py` | **[Gen 7]** few-shot学習・anti-pattern記録・クロスドメイン転用 |
| `hermes_agi_gen/consciousness.py` | **[Gen 6+9]** グローバル・ワークスペース (11シグナルソース) |
| `hermes_agi_gen/value_system.py` | **[Gen 6]** 価値整合・倫理的意思決定 |
| `hermes_agi_gen/cognitive_roles.py` | **[Gen 6]** 8つの専門認知ロール定義 |
| `hermes_agi_gen/predictive_engine.py` | **[Gen 6]** 予測的処理エンジン |
| `hermes_agi_gen/agent_runner.py` | コアループ (Plan → Act → Review) |
| `hermes_agi_gen/orchestrator.py` | 8ロール対応オーケストレーター |
| `hermes_agi_gen/meta_cognition.py` | 行き詰まり検出・GoalQueue・自律ゴール生成 |
| `hermes_agi_gen/long_term_memory.py` | SQLite + セマンティック検索 (LTM) |
| `hermes_agi_gen/executor.py` | ツール実行ディスパッチャー（全ツールにコスト計測付き） |
| `hermes_agi_gen/daemon.py` | 24/7 自律デーモン (省察ループ付き) |
| `hermes_agi_gen/scheduler.py` | cron-like スケジューラ |
| `hermes_agi_gen/web_search.py` | DuckDuckGo 検索 |
| `hermes_agi_gen/mistral_client.py` | LLM クライアント (Claude / Groq / Mistral / Ollama) |
| `cli.py` | インタラクティブ TUI・自然言語スケジューラ |

---

## API リファレンス

### `AGICore` (Gen 9)

統合AGI認知コア。10段階の認知ループですべてのモジュールを協調させる。

```python
from hermes_agi_gen import AGICore
from hermes_agi_gen.mistral_client import MistralClient

llm = MistralClient()
core = AGICore(llm=llm, reflection_interval=5)

# ゴールを認知サイクル全体で処理
result = core.run_goal("このプロジェクトの構造を調べて改善案を提案してください")
print(result["result"])          # 実行結果
print(result["success"])         # 成功/失敗
print(result["identity"])        # AGI Identity サマリー
print(result["strategy"])        # 選択された戦略名
print(result["complexity"])      # ゴール複雑度
print(result["deliberation"])    # 内部対話の合意度 (実行された場合)
print(result["insights"])        # 省察で生成された洞察
print(result["new_goals"])       # 自動生成されたゴール数

# 自律ループ (GoalQueue消化 + 内発動機 + 夢フェーズ)
results = core.autonomous_loop(max_cycles=10, idle_dream=True)

# AGI全体状態の確認
core.print_status()

# Identity の手動保存
core.save_identity()
```

### `IntrinsicMotivationEngine` (Gen 9)

```python
from hermes_agi_gen import IntrinsicMotivationEngine

engine = IntrinsicMotivationEngine()
signals = engine.generate_intrinsic_goals(
    identity_assessment={"planning": 0.3, "execution": 0.8},
    knowledge_gaps=["security", "testing"],
    module_last_used={"reflection_engine": time.time() - 7200},
)
for s in signals:
    print(f"[{s.source}] {s.goal_text} (強度={s.drive_strength:.0%})")
```

### `MetaLearner` (Gen 9)

```python
from hermes_agi_gen import MetaLearner

ml = MetaLearner()
strategy = ml.select_strategy("coding")     # UCB1 で最適戦略を選択
ml.record_outcome("coding", strategy.name, "テスト作成", reward=0.9)

# 転移学習
candidates = ml.find_transfer_candidates("data_analysis")
for c in candidates:
    print(f"{c.strategy_name}: {c.source_domain}→{c.target_domain} (確信度={c.transfer_confidence:.0%})")
    ml.apply_transfer(c)
```

### `InnerDialogue` (Gen 9)

```python
from hermes_agi_gen import InnerDialogue

dialogue = InnerDialogue(llm=llm)
if dialogue.should_deliberate(goal, prediction_confidence=0.3):
    result = dialogue.deliberate(goal, context)
    print(result.refined_goal)      # 議論後の洗練されたゴール
    print(result.consensus_level)   # 合意度 (0〜1)
    print(result.key_concerns)      # 主要な懸念事項
    print(result.should_proceed)    # 実行すべきか
```

### `ReflectionEngine` (Gen 7)

```python
from hermes_agi_gen import ReflectionEngine, LongTermMemory

ltm = LongTermMemory()
engine = ReflectionEngine(llm=llm, reflection_interval=5)

insights = engine.reflect(ltm)
for insight in insights:
    print(f"[{insight.category}] {insight.content} (確信={insight.confidence:.0%})")

goals = engine.generate_strategic_goals(insights, ltm)
metrics = engine.compute_growth_metrics(ltm)
print(metrics.summary())
```

### `GlobalWorkspace` (Gen 6+9)

```python
from hermes_agi_gen import GlobalWorkspace, WorkspaceSignal, SignalSource

ws = GlobalWorkspace()
ws.receive(WorkspaceSignal(
    source=SignalSource.MOTIVATOR,  # Gen 9 の新シグナルソース
    content="好奇心: セキュリティ分野の知識ギャップ",
    relevance=0.8, urgency=0.5, confidence=0.7,
))
event = ws.broadcast()
print(event.winner.source.value)
```

### `ValueSystem` (Gen 6)

```python
from hermes_agi_gen import ValueSystem

vs = ValueSystem()
assessment = vs.assess("CMD: rm -rf /")
print(assessment.is_blocked)        # True
print(assessment.recommendation)    # ブロック理由
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
| `/status` | AGI Identity・成長指標・メタ学習・内発動機・世界モデル状態を表示 |
| `/goals` | 自律ゴールキューを表示 |
| `/world` | 世界モデルの状態を表示 |
| `/improve` | 自己改善レポートを表示 |
| `/reflect` | 手動省察を実行 |
| `/experiment` | AutoResearch実験を実行 |
| `/self-modify` | 自己修正を手動トリガー |
| `/schedule` | スケジュール済みジョブ一覧 |
| `/daemon start` | 24/7 自律デーモンを起動 |
| `/help` | コマンド一覧 |
| `/quit` | 終了 |

### 自然言語スケジュール

```
2026年4月12日午前9時になったら、天気予報を取得して要約してください
毎日午前8時になったら、HNのAIニュースを翻訳して~/Desktop/AI_News/に保存して
30分ごとにシステム状態を確認して
```

---

## LLM プロバイダー

`MistralClient` は以下の優先順で自動選択:

| 優先順 | 環境変数 | バックエンド | デフォルトモデル |
|---|---|---|---|
| 1 | `GROQ_API_KEY` | Groq | `llama-3.3-70b-versatile` |
| 2 | `MISTRAL_API_KEY` | Mistral API | `mistral-small-latest` |
| 3 | なし | Ollama (ローカル) | `mistral` |

```bash
echo "GROQ_API_KEY=gsk_..." > .env
# または
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

---

## 世代間の進化

| 世代 | コア思想 | 認知ループ |
|------|---------|-----------|
| Gen 6 | タスク反応型 | 入力→GWT注意→8ロール実行→出力 |
| Gen 7 | 自律認知型 | +省察→洞察→戦略的ゴール→few-shot学習 |
| Gen 8 | 実験駆動型 | +AutoResearch実験→コード自己修正→メトリクス検証 |
| **Gen 9** | **自律進化型** | **+内発動機→メタ学習→内部対話→適応的行動→夢→永続Identity** |

### Gen 9 の新モジュール

| モジュール | ファイル | 行数 | 永続化 |
|-----------|---------|------|--------|
| AGIIdentity永続化 | `agi_core.py` | 530 | SQLite (LTM) |
| IntrinsicMotivationEngine | `intrinsic_motivation.py` | 270 | メモリ |
| MetaLearner | `meta_learning.py` | 330 | SQLite |
| InnerDialogue | `inner_dialogue.py` | 290 | メモリ |
| ResourceCost追跡 | `world_model.py` | +120 | メモリ |
| 学習済みパターンDB | `self_modifier.py` | +100 | SQLite |

---

## 設計思想

Hermes AGI は以下の認知科学・AI安全の理論を実装の指針としています:

- **Global Workspace Theory** (Baars, 1988) — 統合的意識の計算モデル
- **Predictive Coding** (Clark & Friston, 2013) — 予測誤差からの学習
- **Active Inference** (Friston, 2010) — 能動的な世界モデル更新
- **Intrinsic Motivation** (Oudeyer & Kaplan, 2007) — 好奇心駆動の探索
- **UCB1 Multi-Armed Bandit** (Auer et al., 2002) — 探索と活用のバランス
- **Transfer Learning** — ドメイン間の知識転移
- **Value Alignment** — 明示的な価値体系による安全な自律行動
- **Cognitive Role Specialization** — 専門化した認知モジュールの協調
- **Metacognition** — 自己の認知プロセスへの省察と更新
- **Deliberative Alignment** — 実行前の多角的検討による安全な意思決定
