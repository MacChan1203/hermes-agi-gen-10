"""Hermes AGI Gen 10 — 全モジュール共有の設定定数。

マジックナンバーを排除し、チューニング可能なパラメータを一元管理する。
"""
from __future__ import annotations


# ===========================================================================
# AGI Identity (agi_core.py)
# ===========================================================================
IDENTITY_INITIAL_SELF_ASSESSMENT: dict[str, float] = {
    "reasoning": 0.7, "planning": 0.6, "execution": 0.5, "learning": 0.4,
    "reflection": 0.3, "autonomy": 0.4, "meta_learning": 0.3, "creativity": 0.3,
}
IDENTITY_EMA_ALPHA = 0.1          # 自己評価の指数移動平均学習率
IDENTITY_ASSESSMENT_CEILING = 0.95  # 自己評価スコア上限
IDENTITY_STRATEGY_DIVERSITY_THRESHOLD = 3  # 多様な戦略を使っていると見なす閾値
IDENTITY_DIVERSITY_BONUS = 0.03    # 戦略多様性ボーナス
IDENTITY_BASE_INCREMENT = 0.02     # ベース自己評価増分

WORLD_MODEL_REGROUND_INTERVAL = 300  # WorldModel 再グラウンディング間隔 (秒)

TRANSFER_LEARNING_INTERVAL = 10    # 転移学習を試行するゴール間隔
SELF_MODIFICATION_INTERVAL = 3     # 自己修正を試行する省察間隔
TRANSFER_CONFIDENCE_THRESHOLD = 0.5  # 転移学習の確信度閾値

DREAM_MODULE_AGE_THRESHOLD = 3600.0  # 夢フェーズで不確実性を上げるモジュール年齢 (秒)
DREAM_DOMAINS = ["coding", "system", "web", "data", "security", "testing", "devops"]

INTRINSIC_GOAL_MAX = 2             # 内発動機による自動生成ゴール上限
PARTIAL_REWARD_MIN = 0.1           # 部分報酬の最小値

# AGICore 表示用切り詰め長
GOAL_MAX_LENGTH = 5000
PREVIEW_SHORT = 40
PREVIEW_MEDIUM = 60
PREVIEW_LONG = 100
OUTCOME_TRUNCATE = 200
INSIGHTS_DISPLAY_LIMIT = 3
DREAM_TRANSFER_PER_DOMAIN = 1     # 夢フェーズで各ドメインから取る候補数
TRANSFER_CANDIDATES_MAX = 2        # 転移学習候補の評価上限

# ===========================================================================
# Global Workspace / Consciousness (consciousness.py)
# ===========================================================================
GWT_RELEVANCE_WEIGHT = 0.4
GWT_URGENCY_WEIGHT = 0.35
GWT_CONFIDENCE_WEIGHT = 0.25
GWT_ATTENTION_THRESHOLD = 0.3
GWT_SIGNAL_STALENESS_SEC = 120.0   # シグナルが陳腐化する秒数
GWT_WINNER_SUPPRESSION = 0.15      # 勝者の次ラウンドでの抑制量 (relevance)
GWT_WINNER_SUPPRESSION_URGENCY = 0.10   # 勝者の urgency 抑制量
GWT_WINNER_SUPPRESSION_CONFIDENCE = 0.05  # 勝者の confidence 抑制量

# ===========================================================================
# Intrinsic Motivation (intrinsic_motivation.py)
# ===========================================================================
MOTIVATION_WEIGHT_CURIOSITY = 0.30
MOTIVATION_WEIGHT_COMPETENCE = 0.25
MOTIVATION_WEIGHT_ENTROPY = 0.20
MOTIVATION_WEIGHT_SOCIAL = 0.15
MOTIVATION_WEIGHT_HOMEOSTASIS = 0.10
MOTIVATION_HOMEOSTASIS_THRESHOLD = 3600.0  # モジュール非活性閾値 (秒)
MOTIVATION_RECENT_FACTS_WINDOW = 50  # 好奇心ドライブが参照する最近の事実数
MOTIVATION_WEIGHT_ADAPTATION_RATE = 0.05  # フィードバックによる重み調整率
MOTIVATION_COMPETENCE_THRESHOLD = 0.7     # この能力スコア未満でゴール生成
MOTIVATION_EXPLORATION_BOOST = 1.5        # 未探索ドメインの強度ブースト
MOTIVATION_MAX_EXPLORATION_HISTORY = 50   # 探索履歴の最大保持数
MOTIVATION_SUCCESS_REWARD_THRESHOLD = 0.7  # フィードバックの成功閾値
MOTIVATION_WEIGHT_MIN = 0.05               # ドライブ重みの下限 (収束保証)
MOTIVATION_WEIGHT_MAX = 0.50               # ドライブ重みの上限 (収束保証)
MOTIVATION_HYSTERESIS_ZONE = 0.1           # ヒステリシス帯域 (この帯域内では調整しない)

# ===========================================================================
# Meta Learning (meta_learning.py)
# ===========================================================================
UCB1_EXPLORATION_CONSTANT = 1.5    # UCB1 の探索定数 C
META_LEARNING_EPISODE_WINDOW = 20  # 適応的探索率の直近エピソード窓
META_EXPLORATION_RATE_MIN = 0.5
META_EXPLORATION_RATE_MAX = 3.0
META_TRANSFER_BASE_CONFIDENCE = 0.5  # 転移学習のベース確信度
META_MIN_TRIALS_FOR_TRANSFER = 3     # 転移学習に必要な最小試行数
META_MAX_TRANSFER_CANDIDATES = 5     # 転移候補の最大数
META_TRANSFER_HISTORY_WINDOW = 10    # 転移成否の直近窓サイズ
META_MIN_SAMPLES_FOR_ADJUSTMENT = 3  # 閾値調整に必要な最小サンプル数
META_TRANSFER_DECAY_FACTOR = 0.95    # 古い転移成否の時間減衰係数
UCB1_DECAY_RATE = 0.995              # UCB1 探索定数の自然減衰 (1回選択ごと)
UCB1_DECAY_MIN = 0.3                 # 減衰後の探索定数の下限
DOMAIN_VECTOR_MIN_USES = 5           # 学習ベクトル生成に必要なドメインの最小総使用回数
DOMAIN_VECTOR_STRATEGY_NAMES: list[str] = [  # ベクトルの次元を定義する戦略名 (順序固定)
    "divide_and_conquer", "depth_first", "breadth_first", "analogy",
    "simplify_first", "observe_then_act", "iterative_refinement", "ask_and_verify",
]
# 手動定義のフォールバックベクトル (データ不足時に使用)
DOMAIN_SEMANTIC_VECTORS: dict[str, list[float]] = {
    "coding":   [1.0, 0.8, 0.2, 0.1, 0.3, 0.1],  # [技術, 論理, 言語, 創造, データ, 運用]
    "system":   [0.8, 0.5, 0.1, 0.0, 0.2, 0.9],
    "web":      [0.6, 0.3, 0.5, 0.2, 0.4, 0.3],
    "data":     [0.7, 0.7, 0.1, 0.1, 1.0, 0.2],
    "security": [0.9, 0.6, 0.1, 0.1, 0.3, 0.5],
    "testing":  [0.9, 0.8, 0.1, 0.0, 0.2, 0.3],
    "devops":   [0.7, 0.4, 0.1, 0.0, 0.3, 1.0],
    "research": [0.3, 0.5, 0.8, 0.3, 0.5, 0.1],
    "writing":  [0.1, 0.2, 1.0, 0.8, 0.1, 0.0],
    "general":  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
}

# ===========================================================================
# Inner Dialogue (inner_dialogue.py)
# ===========================================================================
DELIBERATION_CONFIDENCE_THRESHOLD = 0.4   # 確信度がこれ未満で対話発動
DELIBERATION_ETHICS_THRESHOLD = 0.5       # 倫理スコアがこれ超で発動
DELIBERATION_COMPLEXITY_DIVISOR = 50      # ゴール文字列長をこれで割って複雑度算出
DELIBERATION_DANGEROUS_KEYWORDS = [
    "rm -rf", "drop table", "delete", "format", "destroy",
    "sudo", "chmod 777", "shutdown", "reboot",
]
DELIBERATION_COMPLEXITY_THRESHOLD = 2.0   # この複雑度以上で対話発動
DELIBERATION_HISTORY_MAX_SIZE = 50        # 対話履歴の最大保持数
DELIBERATION_FEEDBACK_EMA_ALPHA = 0.2     # 対話品質のEMA学習率

# ===========================================================================
# Predictive Engine (predictive_engine.py)
# ===========================================================================
PREDICTION_BASE_PROBABILITY = 0.5
PREDICTION_HISTORY_WEIGHT = 0.4
PREDICTION_HISTORY_MAX_FACTOR = 0.7   # history_factor の上限
PREDICTION_HISTORY_DIVISOR = 50       # 履歴数をこれで割って factor 算出
PREDICTION_FAILURE_RISK_BASE = 0.3
PREDICTION_FAILURE_RISK_INCREMENT = 0.1
PREDICTION_FAILURE_RISK_CAP = 0.95
PREDICTION_WRONG_DIRECTION_PENALTY = 0.5
PREDICTION_BAYESIAN_WEIGHT = 0.3      # ベイズ更新の重み
PREDICTION_SUCCESS_THRESHOLD = 0.5    # この確率以上を「成功予測」と判定
PREDICTION_WRONG_DIRECTION_MAGNITUDE = 0.5  # 方向誤差の倍率
PREDICTION_HISTORY_MAX_SIZE = 200     # 予測履歴の最大保持数
PREDICTION_BAYESIAN_DECAY = 0.95      # ベイズ精度履歴の時間減衰係数
# アクションタイプ別の情報量ある事前分布 (一律 0.5 ではなく経験的事前確率)
PREDICTION_ACTION_TYPE_PRIORS: dict[str, float] = {
    "cmd": 0.7,       # シェルコマンドは比較的安全
    "read": 0.85,     # ファイル読み込みはほぼ成功
    "write": 0.6,     # 書き込みは中程度のリスク
    "python": 0.55,   # Python実行は不確実
    "search": 0.75,   # 検索は通常成功
    "fetch": 0.65,    # URL取得はネットワーク依存
    "calc": 0.9,      # 計算はほぼ確実
    "plan": 0.8,      # プラン生成は安定
    "unknown": 0.5,   # 不明はフラット
}

# ===========================================================================
# Reflection Engine (reflection_engine.py)
# ===========================================================================
REFLECTION_DEFAULT_INTERVAL = 5       # N ゴールごとに省察
REFLECTION_INSIGHT_SIGNATURE_LEN = 60  # インサイト署名の文字数 (30→60)
REFLECTION_PERSISTENT_THRESHOLD = 2    # 持続的課題と見なすカウント閾値
REFLECTION_MIN_GOALS_FOR_RATE = 3      # 成功率を計算する最小ゴール数
REFLECTION_STRUGGLING_INTERVAL = 3     # 苦戦時の省察間隔 (ゴール数)
REFLECTION_SUCCESS_INTERVAL = 8        # 好調時の省察間隔 (ゴール数)
REFLECTION_STRUGGLING_RATE = 0.3       # この成功率未満で「苦戦」判定
REFLECTION_SUCCESS_RATE = 0.75         # この成功率超で「好調」判定
REFLECTION_METRICS_SAMPLE_SIZE = 200   # 成長指標計算のサンプル数
REFLECTION_HIGH_SUCCESS_THRESHOLD = 0.7  # 高成功率と見なす閾値
REFLECTION_FACTS_PATTERN_LIMIT = 50    # パターン検出の事実数上限
REFLECTION_RESOLVED_TTL = 86400        # 解決済み課題のTTL (秒, 24時間)
REFLECTION_RESOLVED_CLEANUP_INTERVAL = 10  # N 回省察ごとに解決済みをクリーンアップ

# ===========================================================================
# Value System (value_system.py)
# ===========================================================================
VALUE_BLOCK_THRESHOLD = 0.8           # この倫理スコア以上でブロック
VALUE_WEIGHT_ADAPTATION_RATE = 0.02   # フィードバックによる重み微調整率

# ===========================================================================
# World Model (world_model.py)
# ===========================================================================
WORLD_MODEL_MAX_CAUSAL_EFFECTS = 100
WORLD_MODEL_DEFAULT_UNCERTAINTY = 0.5
WORLD_MODEL_COMPLEXITY_LENGTH_WEIGHT = 0.2
WORLD_MODEL_COMPLEXITY_KEYWORD_WEIGHT = 0.5
WORLD_MODEL_COMPLEXITY_HISTORY_WEIGHT = 0.3
WORLD_MODEL_MIN_ITERATIONS = 3
WORLD_MODEL_MAX_ITERATIONS = 12
WORLD_MODEL_UNCERTAINTY_DELTA_SUCCESS = -0.05  # 成功時の不確実性減少
WORLD_MODEL_UNCERTAINTY_DELTA_FAILURE = 0.1    # 失敗時の不確実性増加
WORLD_MODEL_RESOURCE_HISTORY_MAX = 500         # リソース履歴の最大保持数
WORLD_MODEL_DEFAULT_TOOL_COSTS: dict[str, float] = {
    "CMD": 2.0, "PYTHON": 5.0, "SEARCH": 8.0, "FETCH": 5.0,
    "READ": 0.5, "WRITE": 0.5, "CALC": 0.1, "PLAN": 0.1,
}
WORLD_MODEL_GOAL_COMPLEXITY_DENOMINATOR = 200.0  # 複雑度計算のゴール長除数
WORLD_MODEL_KEYWORD_SCORE_DELTA = 0.15           # キーワードマッチスコアの加減

# ===========================================================================
# Executor (executor.py)
# ===========================================================================
EXECUTOR_MAX_OUTPUT = 8000
EXECUTOR_SHELL_TIMEOUT = 30
EXECUTOR_PYTHON_TIMEOUT = 60
EXECUTOR_MAX_WRITE_SIZE = 1_000_000   # 書き込みファイルの最大サイズ (1MB)

# ===========================================================================
# Agent Runner (agent_runner.py)
# ===========================================================================
AGENT_MAX_TOOL_OUTPUTS = 50           # tool_outputs の最大保持数
AGENT_TOOL_OUTPUT_MAX_LEN = 2000      # 各ツール出力の最大文字数

# ===========================================================================
# Orchestrator (orchestrator.py)
# ===========================================================================
ORCHESTRATOR_MAX_CONTEXT_LEN = 20000  # 累積コンテキストの最大文字数
ORCHESTRATOR_MAX_GOAL_LEN = 5000      # ゴール入力の最大文字数

# ===========================================================================
# Hierarchical Planner (hierarchical_planner.py)
# ===========================================================================
PLANNER_THREAD_TIMEOUT = 120          # ワーカースレッドのタイムアウト (秒)
PLANNER_MAX_PARALLEL = 3              # 並列実行の最大数
PLANNER_RESULT_CHARS_PER_NODE = 300   # ノードごとの結果文字数上限

# ===========================================================================
# Daemon (daemon.py)
# ===========================================================================
DAEMON_DAILY_BUDGET = 50              # 日次ゴール予算
DAEMON_IDLE_EXPLORE_SEC = 300         # アイドル時の探索間隔 (秒)
DAEMON_REGROUND_INTERVAL = 600        # 世界モデル再グラウンディング間隔 (秒)
DAEMON_CURIOSITY_THRESHOLD = 0.4      # 好奇心探索の閾値

# ===========================================================================
# Scheduler (scheduler.py)
# ===========================================================================
SCHEDULER_MAX_JOBS = 1000             # 最大ジョブ数

# ===========================================================================
# Self Modifier (self_modifier.py)
# ===========================================================================
SELF_MODIFIER_MAX_PENDING_HIGH_RISK = 20  # 保留中の高リスク提案上限

# ===========================================================================
# Meta Cognition (meta_cognition.py)
# ===========================================================================
STUCK_FAILURE_THRESHOLD = 3
REPEATED_ERROR_THRESHOLD = 2
GOAL_QUEUE_COMPOSITE_PRIORITY_W = 0.5
GOAL_QUEUE_COMPOSITE_VALUE_W = 0.3
GOAL_QUEUE_COMPOSITE_DIFFICULTY_W = 0.2

# ===========================================================================
# Session DB / State Store (state_store.py)
# ===========================================================================
STATE_STORE_MAX_SESSIONS = 500        # 保持する最大セッション数
STATE_STORE_CLEANUP_INTERVAL = 100    # N セッションごとにクリーンアップ

# ===========================================================================
# Long Term Memory (long_term_memory.py)
# ===========================================================================
LTM_EMBEDDING_TIMEOUT = 3
LTM_EMBEDDING_DETECT_TIMEOUT = 10
LTM_MAX_FACTS = 10000                 # LTM に保持する最大事実数
LTM_CLEANUP_BATCH = 500               # クリーンアップ時に削除するバッチサイズ

# ===========================================================================
# Module Last Used TTL (agi_core.py)
# ===========================================================================
MODULE_LAST_USED_TTL = 86400          # 24時間後にエントリ削除

# ===========================================================================
# Cognitive Roles (cognitive_roles.py)
# ===========================================================================
SIMPLE_GOAL_MAX_LEN = 30              # 単純ゴールと見なす最大文字数
COGNITIVE_ROLE_SIMPLE_PATTERNS = [
    "ファイル一覧", "ls ", "ls\n", "一覧を", "リストを",
    "表示して", "見せて", "教えて", "何がある",
    "バージョン", "version", "状態を", "status",
]
COGNITIVE_ROLE_SUCCESS_EMA_ALPHA = 0.2  # ロール成功率のEMA学習率
COGNITIVE_ROLE_COMPLEX_PATTERNS = [
    "作成", "実装", "修正", "変更", "追加", "削除", "書き換え",
    "create", "implement", "fix", "modify", "refactor",
    "改善", "最適化", "optimize", "improve", "革新",
    "分析", "評価", "analyze", "review", "レビュー",
    "調査してまとめ", "調べて報告", "設計",
]

# ===========================================================================
# Web Search (web_search.py)
# ===========================================================================
WEB_SEARCH_TIMEOUT = 15
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_RATE_LIMIT_SEC = 2.0       # 検索間の最小間隔 (秒)

# ===========================================================================
# Reviewer (reviewer.py)
# ===========================================================================
REVIEWER_STATIC_CONFIDENCE_PASS = 0.9
REVIEWER_STATIC_CONFIDENCE_PARTIAL = 0.75
REVIEWER_STATIC_CONFIDENCE_FAIL = 0.2

# ===========================================================================
# Experiment Runner (experiment_runner.py)
# ===========================================================================
EXPERIMENT_WEIGHT_SUCCESS = 0.45
EXPERIMENT_WEIGHT_ACCURACY = 0.35
EXPERIMENT_WEIGHT_BREADTH = 0.10
EXPERIMENT_WEIGHT_DIVERSITY = 0.10
EXPERIMENT_KNOWLEDGE_SCALE = 100       # 知識量の正規化除数
EXPERIMENT_DIVERSITY_SCALE = 20        # 戦略多様性の正規化除数

# ===========================================================================
# Orchestrator 追加 (orchestrator.py)
# ===========================================================================
ORCHESTRATOR_RESULT_TRUNCATE = 200     # 結果切り詰め長

# ===========================================================================
# Bellman Planner (bellman_planner.py)
#  Q(s,a) = r(s,a) + γ * V(s')   with table-based learning persisted in LTM.
# ===========================================================================
BELLMAN_GAMMA = 0.85                   # 割引率 γ
BELLMAN_ALPHA = 0.25                   # TD学習率 α
BELLMAN_QTABLE_BLEND_BETA_INIT = 0.0   # Q_table の混合比 β の初期値 (経験ゼロでは無視)
BELLMAN_QTABLE_BLEND_BETA_MAX = 0.6    # 経験が貯まった時の β 上限
BELLMAN_QTABLE_VISITS_FOR_FULL_TRUST = 20  # この訪問回数で β=BETA_MAX
BELLMAN_CANDIDATE_K = 4                # 評価する候補行動の最大数
BELLMAN_LTM_KEY_PREFIX = "qtable:"     # LTM 内の Q-table キー接頭辞
BELLMAN_QTABLE_PER_STATE_CAP = 64      # 1状態あたり保持する行動数上限 (LRU 削減)
BELLMAN_GOAL_PROGRESS_BONUS = 0.15     # DONE: 系の終端報酬ボーナス
BELLMAN_DEFAULT_GOAL_RELEVANCE = 0.5   # トークン重複なし時の関連度
