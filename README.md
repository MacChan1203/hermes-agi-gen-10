# hermes-agi-gen

Hermes Agent の既存設計を踏まえつつ、v9 の **Plan → Act → Review** を中核にして再構成した軽量版です。

## この版で残したもの

- `hermes_time.py` のタイムゾーン解決と安全なフォールバック
- `minisweagent_path.py` の worktree / submodule 探索
- `utils.py` の atomic write
- `hermes_constants.py` の API エンドポイント定数
- `hermes_state.py` の方向性を受けた SQLite + FTS5 の `SessionDB`
- `toolsets.py`, `toolset_distributions.py` の toolset 発想
- `run_agent.py`, `cli.py`, `batch_runner.py`, `mini_swe_runner.py`, `trajectory_compressor.py` の入口

## この版で強化したもの

- `AgentState` による状態管理
- `Planner` による小さな計画生成
- `Executor` による観測中心の実行
- `Reviewer` による失敗分類と回復アクション
- セッション保存を `SessionDB` に統合
- 出力メッセージを極力日本語化

## すぐ試す

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_agent.py --query "Hermes AGI Gen の状態を見てください"
```

## CLI

```bash
python cli.py
python cli.py --query "このプロジェクトの次の改善案を出してください"
```

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `hermes_agi_gen/agent_runner.py` | v9 コアループ (Plan → Act → Review) |
| `hermes_agi_gen/orchestrator.py` | マルチエージェント オーケストレーター |
| `hermes_agi_gen/mistral_client.py` | LLM クライアント (Mistral / Groq / Ollama) |
| `hermes_agi_gen/code_agents.py` | コード生成・レビュー専用エージェント |
| `hermes_agi_gen/agent_message.py` | エージェント間メッセージ型 |
| `hermes_agi_gen/state_store.py` | SQLite セッション保存 |
| `hermes_agi_gen/planner.py` | 動的 plan |
| `hermes_agi_gen/executor.py` | shell ベース観測 |
| `hermes_agi_gen/reviewer.py` | review と recovery |
| `cli.py` | インタラクティブ TUI |

## LLM プロバイダー

`MistralClient` は以下の優先順で自動選択します：

| 優先順 | 環境変数 | バックエンド | デフォルトモデル |
|---|---|---|---|
| 1 | `MISTRAL_API_KEY` | Mistral API | `mistral-small-latest` |
| 2 | `GROQ_API_KEY` | Groq | `llama-3.3-70b-versatile` |
| 3 | なし | Ollama (ローカル) | `mistral` |

`.env` ファイルにキーを書いておくか、環境変数を設定するだけで切り替わります。

```bash
# Mistral API を使う場合
echo "MISTRAL_API_KEY=sk-..." > .env

# Groq を使う場合
# echo "GROQ_API_KEY=gsk_..." > .env
```

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

## マルチエージェント オーケストレーション

`AgentOrchestrator` は目標を複数のサブタスクに分解し、ロール別のワーカーエージェントに委任して結果を統合します。

```python
from hermes_agi_gen import AgentOrchestrator, MistralClient

llm = MistralClient()  # 環境変数で自動選択
orch = AgentOrchestrator(llm=llm)
result = orch.run("このプロジェクトの構造を調べて改善案をまとめてください")
print(result)
```

**ワーカーロール**

| ロール | 担当 |
|---|---|
| `researcher` | ローカルファイル・コードの調査 |
| `developer` | コード実行・ファイル操作 |
| `critic` | 成果物の評価・改善提案 |

## 設計上の意見

この版では、旧 Hermes の巨大な機能面をそのまま全部移すより、
**自律的に前進する骨格** を先に固める方が正しいと判断しました。

つまり、まずは

1. 現状把握
2. 小さく実行
3. 結果検証
4. 失敗時に立て直し

を確実に回せることを優先しています。

## 次にやると強いこと

- 既存 `tools/registry` と本格接続する
- `run_agent.py` に OpenAI/OpenRouter 呼び出しを戻す
- `cli.py` に旧 TUI を段階的に戻す
- `toolset` の requirement チェックを本格化する
- `SessionDB` にセッションタイトル自動生成と親子チェーンを入れる

