# Hermes AIエージェント Gen 10

**自律進化型AIエージェントフレームワーク** — 認知科学に基づく10段階認知ループ、6つの自己適応フィードバックループ、許可リスト方式セキュリティ、モックテストと条件付き実LLMテストを備えた自律型AIエージェント。

> Gen 6: タスク反応型AIエージェント (入力→処理→出力)
> Gen 7: 自律認知型AIエージェント (知覚→省察→注意→計画→行動→学習)
> Gen 8: 実験駆動型AIエージェント (洞察→コード自己修正→メトリクス検証)
> Gen 10: 自律進化型AIエージェント (内発動機→メタ学習→内部対話→適応的行動→夢→永続Identity)
> Gen 10.1: + ベルマン最適方程式 (Q\*(s,a) = r + γ·max Q\*(s',a')) によるモデル/表ハイブリッド行動選択
> **Gen 10.2: + 離散トークン通信レイヤー (エージェント間/ロール間で内部言語を発話・予測・強化学習)**

| 指標 | 値 |
|------|-----|
| モジュール数 | 48 |
| コード行数 | 16,800 |
| テスト件数 | **モック 715 + 実LLM 14 (Ollama 起動時のみ実行)** |
| テストカバレッジ | 48/48 モジュール (100%) |
| config 定数 | 170 |

---

## すぐ試す

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# テストや開発も行う場合
pip install -r requirements-dev.txt

# Ollama で gemma4:e4b を起動しておく
ollama pull gemma4:e4b
ollama serve

# インタラクティブ CLI
python cli.py

# デーモンモード (24/7 自律実行)
python cli.py --daemon

# テスト実行
pip install -r requirements-dev.txt
python3 -m pytest tests/ --ignore=tests/test_live_llm.py  # Ollama不要で全テスト (715件)
python3 -m pytest tests/test_live_llm.py -v              # 実LLMテスト (14件, Ollama必須。未起動ならskip)
python3 -m pytest tests/                                  # Ollama起動時は実LLMテストも実行

# Python から直接
python3 -c "
from hermes_agi_gen import AGICore
from hermes_agi_gen.mistral_client import MistralClient

llm = MistralClient()
core = AGICore(llm=llm)
result = core.run_goal('このプロジェクトの構造を分析して')
print(result['success'], result['strategy'])
"

# 仕様書準拠の最小 Planner -> Executor -> Critic ループ (MVP)
python3 -c "
from hermes_agi_gen import run_spec_mvp
print(run_spec_mvp('READMEを確認して改善点をまとめる', '.hermes/spec_mvp_memory.json'))
"

# 正式版 spec ループ (反復上限解除・SQLite履歴・LLM対応・価値整合・プラトー検出)
python3 -c "
from hermes_agi_gen import run_spec_full, FullConfig
out = run_spec_full(
    'READMEを確認して改善点をまとめる',
    '.hermes/spec_full.sqlite',
    config=FullConfig(max_iterations=10, patience=2),
)
print('done=', out['review']['done'], 'iters=', out['iterations'], 'metrics=', out['metrics'])
"
```

### LLM プロバイダー

既定ではローカル Ollama の **gemma4:e4b** を使用します。

```bash
# .env (オプション)
OLLAMA_MODEL=gemma4:e4b
```

OpenAI の GPT-5.5 型モデルを使う場合:

```bash
export OPENAI_API_KEY=sk-...
export HERMES_LLM_PROVIDER=openai
export HERMES_MODEL=gpt-5.5
export HERMES_REASONING_EFFORT=medium  # 任意: none/minimal/low/medium/high/xhigh

python cli.py --model gpt-5.5
```

### 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HERMES_HOME` | `~/.hermes` | 全 DB・設定ファイルの格納先 |
| `OLLAMA_MODEL` | `gemma4:e4b` | 使用する Ollama モデル |
| `HERMES_LLM_PROVIDER` | 自動判定 | `ollama` または `openai` |
| `HERMES_MODEL` | `gemma4:e4b` | 使用するモデル名 (`gpt-*` は OpenAI として扱う) |
| `OPENAI_API_KEY` | 未設定 | OpenAI provider 使用時の API キー |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI互換エンドポイント |
| `HERMES_REASONING_EFFORT` | 未設定 | GPT-5系の reasoning effort |

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
[7. 行動] HermesAgentV10 — 許可リスト方式 AST sandbox + アトミック書込
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

## Gen 10 の核心機能

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

### 4.5. ベルマン最適方程式プランナー (`bellman_planner.py`)

認知ループに **MDP (Markov Decision Process)** 化レイヤーを追加。`Q*(s,a) = r(s,a) + γ · max_{a'} Q*(s',a')` に従って候補行動を評価・選択する。

- **状態 s**: ドメイン + 直近行動種別 + last_status から SHA1 シグネチャ
- **行動 a**: planner 出力 + LTM 成功戦略 + Reviewer 改善ヒントの候補プール
- **報酬 r**: `ValueSystem.utility_score` × 目標トークン重複度 + 終端ボーナス (DONE/ANSWER/WRITE)
- **遷移 P**: `PredictiveEngine.predict(a).success_probability` で近似

二段階構成:

| Phase | 役割 | 学習方式 |
|-------|------|----------|
| **A: BellmanEvaluator** | r + γ·V_model(s') によるモデルベース評価 | 学習なし (即時評価) |
| **B: QTable** | 表形式 Q-learning を LTM (`qtable:*` キー) に永続化 | TD 更新: `Q ← Q + α[r + γ max Q(s',a') − Q]` |

混合スコア `Q_total = (1−β)·Q_model + β·Q_table` で β は訪問回数に応じて 0.0 → 0.6 へ成長。経験ゼロではモデル評価、経験が貯まると表 Q を信頼する。状態あたり 64 行動で LRU 削減。

**有効化** (デフォルト無効):
```python
agent = HermesAgentV10(repo_root=".", llm=llm, use_bellman=True)
```

### 4.6. 仕様準拠ループ 正式版 (`spec_full.py`)

`spec_core.HermesAGIMVP` の MVP 制約を取り払った本番運用向けバリアント。

| 項目 | MVP (`spec_core.py`) | **正式版 (`spec_full.py`)** |
|------|---------------------|---------------------------|
| 反復上限 | 3 で固定クランプ | 任意 (`FullConfig.max_iterations`) |
| プラン | 3 ステップ固定テンプレ | LLM 駆動 + ドメイン認識テンプレ (general/coding/research/writing/data) |
| ステップ数 | 3 固定 | `plan_min_steps`〜`plan_max_steps` の範囲で可変 |
| Executor | 副作用なしダミー | リトライ (指数バックオフ) + 価値整合ゲート + 任意の実ツール (`make_real_tool_runner`) |
| Critic | 成功率のみ | 完了率 × 価値整合 × 目標被覆度 の重み付きスコア |
| 停止条件 | 反復数のみ | done / プラトー / 価値違反 (`halt_on_value_violation`) |
| メモリ | フラット JSON | `SqliteMemory` (タスク横断クエリ・FK・タイムスタンプ・試行回数・実行時間) |
| 反復間学習 | critic_feedback を制約に追記 | 同上 + 前回フィードバックを次プランの冒頭に反映 |
| 出力 | task / review / iterations | 同上 + `history` / `halted_reason` / `metrics` (best_score, failure_rate) |

```python
from hermes_agi_gen import HermesAGIFull, FullConfig, make_real_tool_runner, FullExecutor
# 実ツールを使う場合
runner = make_real_tool_runner(repo_root=".")
app = HermesAGIFull(
    "/tmp/full.sqlite",
    config=FullConfig(max_iterations=10, patience=2, retry_max=3),
    executor=FullExecutor(runner=runner),  # CMD:/READ:/WRITE: 等を実行
)
print(app.run("テストを実行して結果をまとめる"))
```

### 4.7. 離散トークン通信レイヤー (Gen 10.2)

階層的・予測的AIサマリの **② 2体協調 / ③ 通信 / ④ 離散トークン / ⑤ RL / ⑥ 解釈 / ⑧ 自己評価** に対応する内部言語レイヤー。エージェント間 (#3 CodeGenerator ↔ CodeReviewer) およびロール間 (#2 critic/innovator/ethicist ↔ strategist) が固定サイズの離散語彙でメッセージを交換し、予測誤差で語彙を強化学習する。

| モジュール | 役割 |
|---|---|
| `token_codebook.py` | 離散語彙 (id, label, keywords, expected_patterns) + EMA 強化統計。`emit()` / `lookup()` (副作用なし) / `record_reward()` / `bonus_for()` / snapshot 永続化 |
| `peer_channel.py` | `AgentMessage` を運搬する受信箱バス。受信者ロール名でルーティング |
| `token_interpreter.py` | トークン列 → 人間可読な解釈文字列 (label + 使用例) |

**フロー (#3 コード生成 ↔ レビュー)**:

```
[Generator]  request → codebook.emit() → token_id → channel.send("reviewer")
                                                            ↓
[Reviewer]   channel.receive("reviewer") → predict() ──→ CodePrediction
                                              ↓ (コード観測前に期待パターンを立てる)
                  LLM レビュー (token_hint をプロンプトへ注入)
                                              ↓
                  prediction.evaluate(code) → 予測誤差
                                              ↓
                  codebook.record_reward(token_id, 1.0 - error)  ←  サマリ⑤⑧
                                              ↓
                  TokenInterpreter.interpret() → レビュー末尾に [内部通信] 行を付与  ← サマリ⑥
```

**フロー (#2 内部対話)**: `InnerDialogue.deliberate()` で critic/innovator/ethicist の発話を `(stance × concern type)` で離散化 → strategist の context に `[内部トークン要約: T_RISK(リスク警告), T_EXTEND(拡張提案), ...]` として注入。`record_outcome(success=...)` 時に **stance 整合性で credit assignment** (support/extend は success 時に強化、oppose/qualify は failure 時に強化)。

**RL ループの閉**:
- `emit()` のスコア = `hits + λ·(avg_reward − 0.5)·2` (キーワード一致だけでなく強化済みかも考慮)
- 強化されたトークンが同点時に選ばれるため、繰り返し成功した「内部言語」が固定化される

**Bellman プランナーへの配線**: `HermesAgentV10(use_bellman=True, codebook=cb)` で `BellmanEvaluator.peer_reward_hook` に `lookup() → bonus_for()` が自動配線され、強化済みトークンに対応する候補行動の即時報酬が `[BONUS_MIN, BONUS_MAX]` の範囲で押し上げられる。

```python
from hermes_agi_gen import HermesAgentV10, TokenCodebook, PeerChannel
from hermes_agi_gen.code_agents import CodeGeneratorAgent, CodeReviewerAgent

# #3: エージェント間トークン通信
ch, cb = PeerChannel(), TokenCodebook()
gen = CodeGeneratorAgent(llm=llm, peer_channel=ch, codebook=cb)
rev = CodeReviewerAgent(llm=llm, peer_channel=ch, codebook=cb)

gen.generate("クイックソートを実装")          # → channel に T_ALGO が emit
print(rev.review("def quicksort(a): ..."))    # 予測誤差で T_ALGO を強化、解釈付き出力

# Bellman プランナーへ強化済み語彙のボーナスを流す
agent = HermesAgentV10(use_bellman=True, codebook=cb)
```

### 5. セキュリティ

| レイヤー | 手法 |
|---------|------|
| Python sandbox | **許可リスト方式** AST 解析 (33モジュール許可、全dunder禁止) |
| os モジュール | パス操作系は許可、破壊的/シェル系 (`system`, `popen`, `remove`, `exec*`, `fork`, `chmod` 等) を denylist 方式で拒否 |
| open() 書込 | literal パス解析でシステム領域 (`/etc`, `/usr`, `/System`, ...) と秘密ファイル (`~/.ssh/*`, `~/.bashrc` 等) を拒否 |
| シェル実行 | `shell=True` 完全廃止、`subprocess.Popen` チェーンでパイプ実装 |
| CALC | `eval()` 廃止、`ast.literal_eval` + 安全な再帰降下パーサー |
| DynamicTool | SHA-256 ハッシュ + `threading.Lock` + AST 検査 |
| 価値体系 | Unicode NFKC 正規化 + ダッシュ/スペース類統一 + 柔軟マッチ |
| LLM プロンプト | NFKC 正規化 + ロール偽装検出 + 長文切り詰め |
| FTS5 クエリ | 演算子エスケープ + パラメータバインド |
| ファイル書込 | `os.replace` アトミック書き込み + バックアップ (WRITE: ツール・self_modifier・daemon 予算カウンタで統一) |
| 並行安全 | Daemon 予算カウンタ fcntl + フォールバック `threading.Lock` / タイムゾーンキャッシュ DCL / LTM スレッドロック / DynamicTool lock |
| LTM 例外粒度 | Ollama 埋め込み呼び出しを `Timeout` / `ConnectionError` / `JSONDecodeError` / 汎用 の4段で分別、破損埋め込みはキー付き WARNING |
| メタ認知 pivot / reviewer recovery | executor 認識 prefix (`PYTHON:/CMD:/FETCH:/...`) 検証、非実行形式は破棄 |

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
| `hermes_agi_gen/agi_core.py` | 統合AIエージェント認知コア・永続Identity・autonomous_loop・夢フェーズ |
| `hermes_agi_gen/config.py` | ~160定数の一元管理 |
| `hermes_agi_gen/intrinsic_motivation.py` | 内発的動機エンジン (5動機源・収束保証) |
| `hermes_agi_gen/meta_learning.py` | メタ学習層 (UCB1減衰・ドメインベクトル自動学習・転移学習) |
| `hermes_agi_gen/inner_dialogue.py` | 内部対話システム (4ロール合意・品質EMA) |
| `hermes_agi_gen/consciousness.py` | Global Workspace (11シグナル・3次元勝者抑制) |
| `hermes_agi_gen/predictive_engine.py` | 予測エンジン (アクション別事前分布・ベイズ時間減衰) |
| `hermes_agi_gen/bellman_planner.py` | ベルマン最適方程式プランナー (モデル評価 + 表形式Q-learning + LTM永続化 + peer_reward_hook) |
| `hermes_agi_gen/token_codebook.py` | 離散トークン語彙 + EMA 強化統計 (Gen 10.2) |
| `hermes_agi_gen/peer_channel.py` | エージェント/ロール間メッセージバス (Gen 10.2) |
| `hermes_agi_gen/token_interpreter.py` | トークン列 → 人間可読化 (Gen 10.2) |
| `hermes_agi_gen/value_system.py` | 価値整合・倫理的意思決定 (Unicode NFKC 正規化) |
| `hermes_agi_gen/executor.py` | ツール実行 (許可リスト AST sandbox・`ExecutorResult` TypedDict) |
| `hermes_agi_gen/self_modifier.py` | 自己修正 (git clean検査・アトミック書込・リスク段階制) |
| `hermes_agi_gen/world_model.py` | 世界モデル (資源コスト追跡・複雑度推定・不確実性マップ) |
| `hermes_agi_gen/reflection_engine.py` | 自己省察 (適応的インターバル・解決済みTTL) |
| `hermes_agi_gen/self_improvement.py` | few-shot学習・anti-pattern記録 |
| `hermes_agi_gen/experiment_runner.py` | AutoResearch方式の実験ループ |
| `hermes_agi_gen/cognitive_roles.py` | 8認知ロール定義 (成功率EMAフィードバック) |
| `hermes_agi_gen/agent_runner.py` | コアループ (Plan -> Act -> Review) |
| `hermes_agi_gen/spec_core.py` | 仕様書準拠MVP (Task/Plan/Result/CriticOutput + JSONメモリ + 最大3ループ) |
| `hermes_agi_gen/spec_full.py` | 仕様準拠ループ 正式版 (LLM駆動プラン・SQLite履歴・重み付きCritic・プラトー検出・実ツール対応) |
| `hermes_agi_gen/orchestrator.py` | 8ロール対応オーケストレーター |
| `hermes_agi_gen/long_term_memory.py` | SQLite + セマンティック検索 (スレッドロック保護) |
| `hermes_agi_gen/meta_cognition.py` | GoalQueue・自律ゴール生成 (プロンプトインジェクション防御) |
| `hermes_agi_gen/daemon.py` | 24/7 自律デーモン (アトミック予算カウンタ) |
| `hermes_agi_gen/scheduler.py` | cron-like スケジューラ (ファイルロック) |
| `hermes_agi_gen/mistral_client.py` | Ollama / OpenAI 対応 LLM クライアント |
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
| `/status` | AIエージェント Identity・成長指標・メタ学習・内発動機・世界モデル状態を表示 |
| `/goals` | 自律ゴールキューを表示 |
| `/world` | 世界モデルの状態を表示 |
| `/improve` | 自己改善レポートを表示 |
| `/mvp <目標>` | 仕様書準拠の最小 Planner→Executor→Critic ループを実行 |
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

#### HNニュースパイプライン

- ゴールに `HN` / `Hacker News` / `hackernews` いずれかを含むと HN API からトップストーリーを取得
- URL 直接指定時 (`https://news.ycombinator.com/` など) はそのページをスクレイプ
- `保存` / `Desktop` / `ファイル` / `txt` のいずれかを含むと `~/Desktop/AI_News/AI_News_MM-DD_HHMM.txt` に保存 (出力先は goal 内の `~/path/...` で上書き可)
- 件数指定 (`3件` / `2つ`) と文字数指定 (`1500字` 等) も goal から自動抽出

---

## テスト

```bash
# 全テスト (Ollama 不要: 715件)
python3 -m pytest tests/ --ignore=tests/test_live_llm.py

# 実 LLM テスト (Ollama + gemma4:e4b 必須: 14件。未起動ならskip)
python3 -m pytest tests/test_live_llm.py -v

# 全テスト (Ollama 起動時は実LLMテストも実行)
python3 -m pytest tests/
```

| テストファイル | 件数 | カバー内容 |
|---------------|------|-----------|
| `test_security.py` | 132 | AST許可リスト、os denylist、open書込パス、Unicode回避、パス遍歴、FTSサニタイズ |
| `test_infrastructure.py` | 111 | 並行安全、DAG循環、アトミック書込、エラーハンドリング |
| `test_cognitive.py` | 92 | GWT 3次元抑制、フィードバック収束、ドメインベクトル学習 |
| `test_cli.py` | 66 | REPL、コマンド、スケジュール検出、インテント分類、HNパイプライン |
| `test_planner_orchestrator.py` | 60 | プランナー、オーケストレーター |
| `test_bellman_planner.py` | 20 | 状態/行動シグネチャ、Bellman評価、QTable TD更新、LTM永続化、AgentRunner統合 |
| `test_token_layer.py` | 34 | 離散トークン語彙 (RL ループ閉)、PeerChannel、TokenInterpreter、Generator↔Reviewer 通信、Reviewer.predict、peer_reward_hook 配線 |
| `test_inner_dialogue_tokens.py` | 8 | InnerDialogue × トークン通信、stance 別 credit assignment、strategist プロンプト注入 |
| `test_integration.py` | 37 | Plan->Execute->Review、AGICore E2E、セキュリティ横断 |
| `test_clients_utils.py` | 36 | MistralClient、utils、toolsets |
| `test_spec_core.py` | 3 | 仕様書準拠MVP、JSONメモリ、最大3ループ |
| `test_spec_full.py` | 37 | 正式版: SQLiteメモリ、LLMプランナー、重み付きCritic、プラトー/価値違反停止、実ツールアダプタ |
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
- **Bellman Optimality** (Bellman, 1957) — `Q*(s,a) = r + γ·max Q*(s',a')` のモデル評価 + 表形式 Q-learning ハイブリッド
- **Intrinsic Motivation** (Oudeyer & Kaplan, 2007) — 5動機源 (収束保証+ヒステリシス)
- **UCB1 Multi-Armed Bandit** (Auer et al., 2002) — 経験蓄積に応じた自然減衰スケジュール
- **Transfer Learning** — コサイン意味ベクトル類似度 + ドメインベクトル自動学習
- **Value Alignment** — Unicode NFKC 正規化 + 許可リスト方式セキュリティ
- **Cognitive Role Specialization** — 8ロール + 成功率 EMA フィードバック
- **Metacognition** — 省察 + 解決済みTTL管理 + 適応的インターバル
- **Deliberative Alignment** — 4ロール内部対話 + 品質 EMA 適応
- **Emergent Communication** (Foerster et al., 2016 ほか) — 離散トークン通信 + 予測誤差による語彙強化 (Gen 10.2)

---

## 世代間の進化

| 世代 | コア思想 | 認知ループ |
|------|---------|-----------|
| Gen 6 | タスク反応型 | 入力->GWT注意->8ロール実行->出力 |
| Gen 7 | 自律認知型 | +省察->洞察->戦略的ゴール->few-shot学習 |
| Gen 8 | 実験駆動型 | +AutoResearch実験->コード自己修正->メトリクス検証 |
| Gen 10 | 自律進化型 | +内発動機->メタ学習->内部対話->適応的行動->夢->永続Identity |
| Gen 10.1 | + ベルマン最適化 | +Q\*(s,a)=r+γ·max Q\*(s',a') (モデル/表ハイブリッド行動選択) |
| **Gen 10.2** | **+ 内部言語 (現行)** | **+離散トークン通信->予測->誤差->語彙強化->Bellman 報酬連動** |
