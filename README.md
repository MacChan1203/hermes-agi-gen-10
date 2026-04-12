# Hermes AGI Gen 9

**自律進化型AGIフレームワーク** — 認知科学に基づく10段階認知ループ、6つの自己適応フィードバックループ、許可リスト方式セキュリティ、563件のテストを備えた自律型AGI。

> Gen 6: タスク反応型AGI (入力→処理→出力)
> Gen 7: 自律認知型AGI (知覚→省察→注意→計画→行動→学習)
> Gen 8: 実験駆動型AGI (洞察→コード自己修正→メトリクス検証)
> **Gen 9: 自律進化型AGI (内発動機→メタ学習→内部対話→適応的行動→夢→永続Identity)**

| 指標 | 値 |
|------|-----|
| モジュール数 | 41 |
| コード行数 | 13,730 |
| テスト件数 | **563 (全合格)** |
| テストカバレッジ | 41/41 モジュール (100%) |
| config 定数 | ~160 |

---

## すぐ試す

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama で gemma4:e4b を起動しておく
ollama pull gemma4:e4b
ollama serve

# インタラクティブ CLI
python cli.py

# デーモンモード (24/7 自律実行)
python cli.py --daemon

# テスト実行
python3 -m pytest tests/ -q                          # モックテスト (549件)
python3 -m pytest tests/test_live_llm.py -v           # 実LLMテスト (14件, Ollama必須)
python3 -m pytest tests/ --ignore=tests/test_live_llm.py  # Ollama不要で全テスト

# Python から直接
python3 -c "
from hermes_agi_gen import AGICore
from hermes_agi_gen.mistral_client import MistralClient

llm = MistralClient()
core = AGICore(llm=llm)
result = core.run_goal('このプロジェクトの構造を分析して')
print(result['success'], result['strategy'])
"
```

### LLM プロバイダー

ローカル Ollama の **gemma4:e4b** を使用します。

```bash
# .env (オプション)
OLLAMA_MODEL=gemma4:e4b
```

### 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HERMES_HOME` | `~/.hermes` | 全 DB・設定ファイルの格納先 |
| `OLLAMA_MODEL` | `gemma4:e4b` | 使用する Ollama モデル |

---

## アーキテクチャ: 10段階認知ループ

```
入力 (ユーザーの目標 / 内発動機 / GoalQueue)
  |
  v
[1. 知覚] WorldModel — 環境をファイルシステムにグラウンド
  |
  v
[2. 内部対話] InnerDialogue — 高リスク時のみ多角的検討
  |         批判者 -> 革新者 -> 倫理家 -> 戦略家 -> 合意形成
  |
  v
[3. 倫理評価] ValueSystem — Unicode NFKC 正規化 + 危険行動ブロック
  |
  v
[4. 注意選択] GlobalWorkspace — 11シグナルソースが3次元注意競争
  |           (relevance/urgency/confidence に勝者抑制付き)
  |
  v
[5. メタ学習] MetaLearner — UCB1 自然減衰 + ドメインベクトル自動学習
  |
  v
[6. 予測] PredictiveEngine — アクション別事前分布 + ベイズ時間減衰
  |
  v
[7. 行動] HermesAgentV9 — 許可リスト方式 AST sandbox + アトミック書込
  |       認知パイプライン: perceiver -> strategist -> executor -> critic
  |
  v
[8. 学習] 予測誤差記録 + メタ学習更新 + few-shot抽出 + LTM蓄積
  |
  v
[9. 内発動機] IntrinsicMotivationEngine — 5動機源 (収束保証+ヒステリシス)
  |
  v
[10. 省察] ReflectionEngine — 適応的インターバル + 解決済みTTL管理
  |        自律コード修正 / 転移学習 / ドメインベクトル再学習
  |
  v
[夢] Dream Phase — GoalQueue空時にLTM知識統合
  |
  v
Identity永続化 -> [ループ]
```

---

## Gen 9 の核心機能

### 1. 永続的アイデンティティ (`agi_core.py`)

セッションをまたいで経験・能力・自己評価が蓄積され続ける。

- SQLite (LTM) に Identity を自動保存・復元
- 8次元の自己評価 (EMA 学習率 `config.IDENTITY_EMA_ALPHA` で更新)
- 全10認知フェーズに個別 try-except (単一モジュール障害で全体が停止しない)
- `RunGoalResult` TypedDict で型安全な結果返却

### 2. 内発的動機エンジン (`intrinsic_motivation.py`)

5動機源に **収束保証** [0.05, 0.50] + **ヒステリシス帯域** 付き。

| 動機源 | 初期重み | 説明 |
|--------|---------|------|
| 好奇心 | 30% | 知識ギャップを検出し探索ゴールを生成 |
| 達成動機 | 25% | 自己評価の低い能力を鍛えるゴールを生成 |
| エントロピー低減 | 20% | WorldModel の不確実領域を優先探索 |
| 社会性 | 15% | ユーザーへの有益性を最大化 |
| 恒常性 | 10% | 長期間未使用のモジュールを活性化 |

`record_goal_outcome()` で重みが自動適応。ヒステリシス帯域内の報酬では重みが変化しない。

### 3. メタ学習層 (`meta_learning.py`)

- **UCB1 自然減衰**: 経験蓄積に応じて探索定数が `0.995^total_uses` で自動減衰 (下限 0.3)
- **ドメインベクトル自動学習**: 実績データから戦略別 avg_reward をベクトル化、10エピソードごとに再学習
- **転移学習**: コサイン意味ベクトル類似度 (10ドメイン対応) + Jaccard フォールバック
- **転移閾値の指数減衰調整**: 成功率に応じて閾値を自動調整

### 4. 内部対話システム (`inner_dialogue.py`)

- 批判者/革新者/倫理家/戦略家の4ロール合意形成
- **対話品質 EMA**: `record_outcome()` で品質を追跡、高品質時はより積極的に発動
- 発動条件: 予測確信度 < 40% / 倫理スコア > 50% / 自己修正タスク

### 5. セキュリティ

| レイヤー | 手法 |
|---------|------|
| Python sandbox | **許可リスト方式** AST 解析 (30モジュール + 40関数のみ許可、全dunder禁止) |
| シェル実行 | `shell=True` 完全廃止、`subprocess.Popen` チェーンでパイプ実装 |
| CALC | `eval()` 廃止、`ast.literal_eval` + 安全な再帰降下パーサー |
| DynamicTool | SHA-256 ハッシュ + `threading.Lock` + AST 検査 |
| 価値体系 | Unicode NFKC 正規化 + ダッシュ/スペース類統一 + 柔軟マッチ |
| LLM プロンプト | NFKC 正規化 + ロール偽装検出 + 長文切り詰め |
| FTS5 クエリ | 演算子エスケープ + パラメータバインド |
| ファイル書込 | `os.replace` アトミック書き込み + バックアップ |

### 6. 6つの自己適応フィードバックループ

| モジュール | フィードバック機構 |
|-----------|------------------|
| IntrinsicMotivation | ドライブ重み調整 (収束保証 + ヒステリシス) |
| MetaLearner | UCB1 減衰 + 転移閾値の指数減衰 + ドメインベクトル自動学習 |
| InnerDialogue | 対話品質 EMA -> 発動閾値の動的調整 |
| CognitiveRoles | ロール別成功率 EMA -> 選択優先度の自動調整 |
| PredictiveEngine | ベイズ精度の時間減衰 + 古エントリ剪定 |
| ReflectionEngine | 解決済み課題の TTL 管理 + 定期クリーンアップ |

---

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `hermes_agi_gen/agi_core.py` | 統合AGI認知コア・永続Identity・autonomous_loop・夢フェーズ |
| `hermes_agi_gen/config.py` | ~160定数の一元管理 |
| `hermes_agi_gen/intrinsic_motivation.py` | 内発的動機エンジン (5動機源・収束保証) |
| `hermes_agi_gen/meta_learning.py` | メタ学習層 (UCB1減衰・ドメインベクトル自動学習・転移学習) |
| `hermes_agi_gen/inner_dialogue.py` | 内部対話システム (4ロール合意・品質EMA) |
| `hermes_agi_gen/consciousness.py` | Global Workspace (11シグナル・3次元勝者抑制) |
| `hermes_agi_gen/predictive_engine.py` | 予測エンジン (アクション別事前分布・ベイズ時間減衰) |
| `hermes_agi_gen/value_system.py` | 価値整合・倫理的意思決定 (Unicode NFKC 正規化) |
| `hermes_agi_gen/executor.py` | ツール実行 (許可リスト AST sandbox・`ExecutorResult` TypedDict) |
| `hermes_agi_gen/self_modifier.py` | 自己修正 (git clean検査・アトミック書込・リスク段階制) |
| `hermes_agi_gen/world_model.py` | 世界モデル (資源コスト追跡・複雑度推定・不確実性マップ) |
| `hermes_agi_gen/reflection_engine.py` | 自己省察 (適応的インターバル・解決済みTTL) |
| `hermes_agi_gen/self_improvement.py` | few-shot学習・anti-pattern記録 |
| `hermes_agi_gen/experiment_runner.py` | AutoResearch方式の実験ループ |
| `hermes_agi_gen/cognitive_roles.py` | 8認知ロール定義 (成功率EMAフィードバック) |
| `hermes_agi_gen/agent_runner.py` | コアループ (Plan -> Act -> Review) |
| `hermes_agi_gen/orchestrator.py` | 8ロール対応オーケストレーター |
| `hermes_agi_gen/long_term_memory.py` | SQLite + セマンティック検索 (スレッドロック保護) |
| `hermes_agi_gen/meta_cognition.py` | GoalQueue・自律ゴール生成 (プロンプトインジェクション防御) |
| `hermes_agi_gen/daemon.py` | 24/7 自律デーモン (アトミック予算カウンタ) |
| `hermes_agi_gen/scheduler.py` | cron-like スケジューラ (ファイルロック) |
| `hermes_agi_gen/mistral_client.py` | Ollama gemma4:e4b 専用 LLM クライアント |
| `hermes_agi_gen/hermes_constants.py` | 共有定数 + `get_hermes_home()` (HERMES_HOME 統一) |
| `cli.py` | インタラクティブ TUI・自然言語スケジューラ・HNパイプライン |

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
| `/perf` | パフォーマンス履歴を表示 |
| `/tools` | カスタムツール一覧 |
| `/help` | コマンド一覧 |
| `/quit` | 終了 |

### 自然言語スケジュール

```
17時30分になったら、Hacker Newsの最初のニュースを日本語で表示して
午後5時になったら、天気予報を取得して要約してください
2026年4月12日午前9時になったら、HNのAIニュースを翻訳して~/Desktop/AI_News/に保存して
```

---

## テスト

```bash
# 全テスト (Ollama 不要: 549件)
python3 -m pytest tests/ --ignore=tests/test_live_llm.py

# 実 LLM テスト (Ollama + gemma4:e4b 必須: 14件)
python3 -m pytest tests/test_live_llm.py -v

# 全テスト (Ollama 起動時: 563件)
python3 -m pytest tests/
```

| テストファイル | 件数 | カバー内容 |
|---------------|------|-----------|
| `test_security.py` | 129 | AST許可リスト、Unicode回避、パス遍歴、FTSサニタイズ |
| `test_infrastructure.py` | 111 | 並行安全、DAG循環、アトミック書込、エラーハンドリング |
| `test_cognitive.py` | 92 | GWT 3次元抑制、フィードバック収束、ドメインベクトル学習 |
| `test_cli.py` | 61 | REPL、コマンド、スケジュール検出、インテント分類 |
| `test_planner_orchestrator.py` | 60 | プランナー、オーケストレーター |
| `test_integration.py` | 37 | Plan->Execute->Review、AGICore E2E、セキュリティ横断 |
| `test_clients_utils.py` | 33 | MistralClient、utils、toolsets |
| `test_experiment_improvement.py` | 21 | 実験メトリクス、自己改善 |
| `test_live_llm.py` | 14 | 実 Ollama 接続、応答バリエーション耐性 |
| `test_errors.py` | 4 | エラー分類、リトライ判定 |
| `test_agent.py` | 1 | エージェント基本動作 |

---

## 設計思想

以下の認知科学・AI安全の理論を実装の指針としています:

- **Global Workspace Theory** (Baars, 1988) — 11シグナルソースの3次元注意競争
- **Predictive Coding** (Clark & Friston, 2013) — アクションタイプ別事前分布 + ベイズ時間減衰
- **Active Inference** (Friston, 2010) — 不確実性マップの動的更新
- **Intrinsic Motivation** (Oudeyer & Kaplan, 2007) — 5動機源 (収束保証+ヒステリシス)
- **UCB1 Multi-Armed Bandit** (Auer et al., 2002) — 経験蓄積に応じた自然減衰スケジュール
- **Transfer Learning** — コサイン意味ベクトル類似度 + ドメインベクトル自動学習
- **Value Alignment** — Unicode NFKC 正規化 + 許可リスト方式セキュリティ
- **Cognitive Role Specialization** — 8ロール + 成功率 EMA フィードバック
- **Metacognition** — 省察 + 解決済みTTL管理 + 適応的インターバル
- **Deliberative Alignment** — 4ロール内部対話 + 品質 EMA 適応

---

## 世代間の進化

| 世代 | コア思想 | 認知ループ |
|------|---------|-----------|
| Gen 6 | タスク反応型 | 入力->GWT注意->8ロール実行->出力 |
| Gen 7 | 自律認知型 | +省察->洞察->戦略的ゴール->few-shot学習 |
| Gen 8 | 実験駆動型 | +AutoResearch実験->コード自己修正->メトリクス検証 |
| **Gen 9** | **自律進化型** | **+内発動機->メタ学習->内部対話->適応的行動->夢->永続Identity** |
