"""AGI コアループ: すべての認知モジュールを統合する統一認知サイクル。

Gen 7 の中枢。以下の認知ループを実装する:

  知覚 (Perceive)
    ↓
  省察 (Reflect)
    ↓
  注意選択 (Attend)
    ↓
  計画 (Plan)
    ↓
  行動 (Act)
    ↓
  学習 (Learn)
    ↓
  (繰り返し)

AGI的観点:
- 受動的な「タスク実行機」ではなく、自律的な認知サイクルを持つ
- 各サイクルで世界モデル・予測エンジン・価値体系が協調して動作
- GlobalWorkspace が注意競争を調停し、認知リソースを最適に配分
- ReflectionEngine が定期的に自己を振り返り、戦略を更新

使い方:
    from hermes_agi_gen.agi_core import AGICore
    from hermes_agi_gen.mistral_client import MistralClient

    llm = MistralClient()
    core = AGICore(llm=llm)
    core.run_goal("プロジェクトの構造を分析して改善案を提案してください")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_runner import HermesAgentV9
from .agent_state import AgentState
from .consciousness import GlobalWorkspace, SignalSource, WorkspaceSignal
from .long_term_memory import LongTermMemory
from .meta_cognition import MetaCognition, QueuedGoal
from .mistral_client import MistralClient
from .predictive_engine import PredictiveEngine
from .reflection_engine import GrowthMetrics, Insight, ReflectionEngine
from .self_improvement import SelfImprovementEngine
from .state_store import SessionDB
from .value_system import ValueSystem
from .world_model import WorldModel


# ------------------------------------------------------------------
# AGI Identity: 永続的自己同一性
# ------------------------------------------------------------------

@dataclass
class AGIIdentity:
    """AGIの永続的な自己モデル。セッションをまたいで進化する。

    知能の本質の一つは「自分が何者で、何ができるか」を把握していること。
    このクラスは AGI の自己認識を表現する。
    """
    name: str = "Hermes AGI"
    version: str = "Gen 7"
    birth_time: float = field(default_factory=time.time)

    # 能力プロファイル
    capabilities: List[str] = field(default_factory=lambda: [
        "ファイル操作・コード実行",
        "ローカル環境の調査・分析",
        "複数認知ロールによる協調推論",
        "長期記憶による経験蓄積",
        "自己省察と戦略更新",
    ])

    # 自己評価 (経験から更新)
    self_assessment: Dict[str, float] = field(default_factory=lambda: {
        "reasoning": 0.7,
        "planning": 0.7,
        "execution": 0.6,
        "learning": 0.7,
        "reflection": 0.5,
    })

    # 価値観
    core_values: List[str] = field(default_factory=lambda: [
        "安全性 — 有害な行動を取らない",
        "誠実性 — 事実に基づいて行動する",
        "有益性 — ユーザーと世界に貢献する",
        "自律性 — 目標に向けて自律的に行動する",
        "成長 — 経験から学び続ける",
    ])

    total_goals_processed: int = 0
    successful_goals: int = 0

    @property
    def age_hours(self) -> float:
        return (time.time() - self.birth_time) / 3600

    @property
    def success_rate(self) -> float:
        if self.total_goals_processed == 0:
            return 0.0
        return self.successful_goals / self.total_goals_processed

    def update_from_metrics(self, metrics: GrowthMetrics) -> None:
        """成長指標から自己評価を更新する。"""
        if metrics.success_rate > 0:
            # 成功率が高いほど execution/planning スコアを上げる
            alpha = 0.1  # 学習率
            self.self_assessment["execution"] = (
                self.self_assessment["execution"] * (1 - alpha)
                + metrics.success_rate * alpha
            )
        if metrics.reflection_count > 0:
            self.self_assessment["reflection"] = min(
                0.95,
                self.self_assessment["reflection"] + 0.02,
            )

    def profile_summary(self) -> str:
        return (
            f"{self.name} {self.version} | "
            f"稼働: {self.age_hours:.1f}h | "
            f"処理ゴール: {self.total_goals_processed} | "
            f"成功率: {self.success_rate:.0%}"
        )


# ------------------------------------------------------------------
# AGICognitiveLoop
# ------------------------------------------------------------------

class AGICore:
    """統合AGI認知コア。

    知覚→省察→注意→計画→行動→学習 のループを実装する。
    すべての認知モジュールがここで協調する。

    Args:
        llm: MistralClient インスタンス
        repo_root: 作業ディレクトリ
        reflection_interval: 何ゴールごとに省察するか
    """

    def __init__(
        self,
        llm: Optional[MistralClient] = None,
        repo_root: str | Path = ".",
        reflection_interval: int = 5,
    ) -> None:
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()

        # 認知モジュール
        self.identity = AGIIdentity()
        self.world_model = WorldModel()
        self.workspace = GlobalWorkspace()
        self.value_system = ValueSystem()
        self.predictor = PredictiveEngine()
        self.ltm = LongTermMemory()
        self.meta = MetaCognition(llm=llm)
        self.reflection_engine = ReflectionEngine(llm=llm, reflection_interval=reflection_interval)
        self.self_improver = SelfImprovementEngine(llm=llm)
        self.session_db = SessionDB()

        # 初期グラウンディング
        self._ground_world_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_goal(self, goal: str, context: str = "") -> Dict[str, Any]:
        """1つのゴールを認知サイクル全体で処理する。

        Returns:
            {
                "result": str,
                "success": bool,
                "identity": str,
                "insights": List[str],
                "new_goals": int,
                "metrics": str,
            }
        """
        self.identity.total_goals_processed += 1

        # --- 知覚フェーズ: 世界モデルを最新化 ---
        if self.world_model.needs_regrounding(max_age_seconds=300):
            self._ground_world_model()

        # --- 倫理評価フェーズ ---
        ethics = self.value_system.assess(goal)
        if ethics.is_blocked:
            return {
                "result": f"[ValueSystem] {ethics.recommendation}",
                "success": False,
                "identity": self.identity.profile_summary(),
                "insights": [],
                "new_goals": 0,
                "metrics": "",
            }

        # --- 注意選択フェーズ: GlobalWorkspace ---
        self.workspace.build_signals_from_state(
            goal=goal,
            context=context,
            is_stuck=False,
            value_risk=ethics.total_score,
        )
        broadcast = self.workspace.broadcast()
        if broadcast:
            print(
                f"[GlobalWorkspace] 注意焦点: [{broadcast.winner.source.value}] "
                f"{broadcast.winner.content[:60]}",
                flush=True,
            )

        # --- 予測フェーズ ---
        prediction = self.predictor.predict(
            action=f"GOAL: {goal}",
            goal=goal,
            context=context,
        )
        print(
            f"[PredictiveEngine] 予測: 成功={prediction.success_probability:.0%}",
            flush=True,
        )

        # --- 実行フェーズ ---
        agent = HermesAgentV9(
            repo_root=self.repo_root,
            model=getattr(self.llm, "model", "unknown"),
            llm=self.llm,
            ltm=self.ltm,
            session_db=self.session_db,
        )

        state = AgentState(
            user_goal=goal,
            success_criteria=["タスクを完了できた", "結果を日本語で説明できる"],
            constraints=["破壊的操作はしない", "まず読んで把握する"],
            max_iterations=6,
            world_model=self.world_model,
        )

        final_state = agent.run(state)
        result_text = "\n".join(final_state.observations) if final_state.observations else "（観測なし）"
        success = final_state.is_done and len(final_state.failed_steps) == 0

        # --- 学習フェーズ ---
        actual_success = success
        self.predictor.record_outcome(
            prediction=prediction,
            actual_outcome=result_text[:200],
            actual_success=actual_success,
        )

        if success:
            self.identity.successful_goals += 1

        # --- 省察フェーズ (定期的) ---
        insights_summary: List[str] = []
        new_goals_count = 0

        if self.reflection_engine.should_reflect():
            insights = self.reflection_engine.reflect(self.ltm)
            insights_summary = [f"[{i.category}] {i.content[:60]}" for i in insights[:3]]

            # 戦略的ゴールを MetaCognition に追加
            strategic_goals = self.reflection_engine.generate_strategic_goals(insights, self.ltm)
            for sg in strategic_goals:
                self.meta.goal_queue.add(sg)
                new_goals_count += 1

            # 自己同一性を更新
            metrics = self.reflection_engine.compute_growth_metrics(self.ltm)
            self.identity.update_from_metrics(metrics)

        return {
            "result": result_text,
            "success": success,
            "identity": self.identity.profile_summary(),
            "insights": insights_summary,
            "new_goals": new_goals_count,
            "metrics": self.predictor.summary(),
        }

    def get_status(self) -> Dict[str, Any]:
        """AGIコアの全体状態を返す。"""
        metrics = self.reflection_engine.compute_growth_metrics(self.ltm)
        grounding_age = self.world_model.grounding_age()

        return {
            "identity": self.identity.profile_summary(),
            "self_assessment": self.identity.self_assessment,
            "world_model_age": f"{grounding_age:.0f}秒前" if grounding_age else "未グラウンディング",
            "goal_queue_size": self.meta.goal_queue.size(),
            "growth_metrics": metrics.summary(),
            "reflection": self.reflection_engine.summary(),
            "workspace": self.workspace.summary(),
            "prediction_accuracy": self.predictor.get_accuracy(),
        }

    def print_status(self) -> None:
        """AGIコアの状態を表示する。"""
        status = self.get_status()
        print("=" * 60)
        print(f"[AGI Core Status] {status['identity']}")
        print("-" * 60)
        for key, val in status.items():
            if key == "identity":
                continue
            if isinstance(val, dict):
                print(f"  {key}:")
                for k, v in val.items():
                    print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                print(f"  {key}: {val}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ground_world_model(self) -> None:
        """世界モデルをファイルシステムの実態にグラウンドする。"""
        self.world_model.initialize_from_filesystem(str(self.repo_root))
        age = self.world_model.grounding_age()
        if age is not None and age < 1.0:
            print("[WorldModel] グラウンディング完了", flush=True)
