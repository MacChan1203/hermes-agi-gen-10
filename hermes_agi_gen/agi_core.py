"""AGI コアループ: すべての認知モジュールを統合する統一認知サイクル。

Gen 9 の中枢。以下の認知ループを実装する:

  知覚 (Perceive)
    ↓
  内部対話 (Deliberate) — 高リスク時のみ
    ↓
  倫理評価 (Ethics)
    ↓
  注意選択 (Attend)
    ↓
  メタ学習戦略選択 (Meta-Learn)
    ↓
  予測 (Predict)
    ↓
  行動 (Act) — 適応的実行深度
    ↓
  学習 (Learn) — メタ学習更新
    ↓
  内発動機 (Motivate) — 自律的ゴール生成
    ↓
  省察 (Reflect)
    ↓
  夢 (Dream) — アイドル時のみ
    ↓
  (繰り返し / autonomous_loop)

AGI的観点:
- 受動的な「タスク実行機」ではなく、自律的な認知サイクルを持つ
- 各サイクルで世界モデル・予測エンジン・価値体系が協調して動作
- GlobalWorkspace が注意競争を調停し、認知リソースを最適に配分
- ReflectionEngine が定期的に自己を振り返り、戦略を更新
- IntrinsicMotivation が外部ゴールなしでも自律的に行動を生成
- MetaLearner が「学習方法自体を学習」する
- InnerDialogue が高リスク決定前に多角的検討を行う
- AGIIdentity がセッションをまたいで永続化する

使い方:
    from hermes_agi_gen.agi_core import AGICore
    from hermes_agi_gen.mistral_client import MistralClient

    llm = MistralClient()
    core = AGICore(llm=llm)
    core.run_goal("プロジェクトの構造を分析して改善案を提案してください")

    # 自律モード: GoalQueueのゴールを自動消化
    core.autonomous_loop(max_cycles=10)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_runner import HermesAgentV9
from .agent_state import AgentState
from .consciousness import GlobalWorkspace, SignalSource, WorkspaceSignal
from .experiment_runner import ExperimentRunner
from .inner_dialogue import InnerDialogue
from .intrinsic_motivation import IntrinsicMotivationEngine
from .long_term_memory import LongTermMemory
from .meta_cognition import MetaCognition, QueuedGoal
from .meta_learning import MetaLearner
from .mistral_client import MistralClient
from .predictive_engine import PredictiveEngine
from .reflection_engine import GrowthMetrics, Insight, ReflectionEngine
from .self_improvement import SelfImprovementEngine
from .self_modifier import SelfModifier
from .state_store import SessionDB
from .value_system import ValueSystem
from .world_model import WorldModel


# ------------------------------------------------------------------
# AGI Identity: 永続的自己同一性
# ------------------------------------------------------------------

_IDENTITY_LTM_KEY = "agi_identity_v9"


@dataclass
class AGIIdentity:
    """AGIの永続的な自己モデル。セッションをまたいで進化する。

    知能の本質の一つは「自分が何者で、何ができるか」を把握していること。
    このクラスは AGI の自己認識を表現する。
    """
    name: str = "Hermes AGI"
    version: str = "Gen 9"
    birth_time: float = field(default_factory=time.time)

    # 能力プロファイル
    capabilities: List[str] = field(default_factory=lambda: [
        "ファイル操作・コード実行",
        "ローカル環境の調査・分析",
        "複数認知ロールによる協調推論",
        "長期記憶による経験蓄積",
        "自己省察と戦略更新",
        "内発的動機による自律行動",       # Gen 9
        "メタ学習による戦略最適化",       # Gen 9
        "内部対話による多角的判断",       # Gen 9
        "資源認識型の適応的計画",         # Gen 9
    ])

    # 自己評価 (経験から更新)
    self_assessment: Dict[str, float] = field(default_factory=lambda: {
        "reasoning": 0.7,
        "planning": 0.7,
        "execution": 0.6,
        "learning": 0.7,
        "reflection": 0.5,
        "autonomy": 0.4,      # Gen 9: 自律性
        "meta_learning": 0.3,  # Gen 9: メタ学習
        "creativity": 0.4,     # Gen 9: 創造性
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
    total_sessions: int = 0          # Gen 9: セッション数
    discovered_capabilities: List[str] = field(default_factory=list)  # Gen 9

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
        alpha = 0.1  # 学習率
        if metrics.success_rate > 0:
            self.self_assessment["execution"] = (
                self.self_assessment["execution"] * (1 - alpha)
                + metrics.success_rate * alpha
            )
        if metrics.reflection_count > 0:
            self.self_assessment["reflection"] = min(
                0.95,
                self.self_assessment["reflection"] + 0.02,
            )
        if metrics.strategy_diversity > 3:
            self.self_assessment["meta_learning"] = min(
                0.95,
                self.self_assessment["meta_learning"] + 0.03,
            )

    def discover_capability(self, capability: str) -> None:
        """新しい能力を発見・記録する。"""
        if capability not in self.discovered_capabilities:
            self.discovered_capabilities.append(capability)
            if capability not in self.capabilities:
                self.capabilities.append(capability)

    def profile_summary(self) -> str:
        return (
            f"{self.name} {self.version} | "
            f"稼働: {self.age_hours:.1f}h | "
            f"セッション: {self.total_sessions} | "
            f"処理ゴール: {self.total_goals_processed} | "
            f"成功率: {self.success_rate:.0%}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """シリアライズ用辞書に変換。"""
        return {
            "name": self.name,
            "version": self.version,
            "birth_time": self.birth_time,
            "capabilities": self.capabilities,
            "self_assessment": self.self_assessment,
            "core_values": self.core_values,
            "total_goals_processed": self.total_goals_processed,
            "successful_goals": self.successful_goals,
            "total_sessions": self.total_sessions,
            "discovered_capabilities": self.discovered_capabilities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AGIIdentity":
        """辞書から復元。"""
        identity = cls()
        for key in ("name", "version", "birth_time", "total_goals_processed",
                     "successful_goals", "total_sessions"):
            if key in data:
                setattr(identity, key, data[key])
        if "capabilities" in data and isinstance(data["capabilities"], list):
            identity.capabilities = data["capabilities"]
        if "self_assessment" in data and isinstance(data["self_assessment"], dict):
            identity.self_assessment.update(data["self_assessment"])
        if "core_values" in data and isinstance(data["core_values"], list):
            identity.core_values = data["core_values"]
        if "discovered_capabilities" in data and isinstance(data["discovered_capabilities"], list):
            identity.discovered_capabilities = data["discovered_capabilities"]
        return identity


# ------------------------------------------------------------------
# AGICognitiveLoop
# ------------------------------------------------------------------

class AGICore:
    """統合AGI認知コア — Gen 9。

    知覚→内部対話→倫理→注意→メタ学習→予測→行動→学習→内発動機→省察→夢
    のフル認知ループを実装する。すべての認知モジュールがここで協調する。

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

        # 認知モジュール (Gen 7 基盤)
        self.world_model = WorldModel()
        self.workspace = GlobalWorkspace()
        self.value_system = ValueSystem()
        self.predictor = PredictiveEngine()
        self.ltm = LongTermMemory()
        self.meta = MetaCognition(llm=llm)
        self.reflection_engine = ReflectionEngine(llm=llm, reflection_interval=reflection_interval)
        self.self_improver = SelfImprovementEngine(llm=llm)
        self.self_modifier = SelfModifier(llm=llm, repo_root=self.repo_root)
        self.session_db = SessionDB()

        # Gen 9: 新認知モジュール
        self.meta_learner = MetaLearner()
        self.motivation = IntrinsicMotivationEngine(llm=llm)
        self.inner_dialogue = InnerDialogue(llm=llm)

        # AutoResearch方式の実験ループ
        self.experiment_runner = ExperimentRunner(agi_core=self)

        # 省察サイクルカウンタ
        self._reflection_count: int = 0

        # Gen 9: モジュール活性化時刻追跡
        self._module_last_used: Dict[str, float] = {}

        # Gen 9: 永続Identityの復元
        self.identity = self._load_identity()
        self.identity.total_sessions += 1

        # 初期グラウンディング
        self._ground_world_model()

    # ------------------------------------------------------------------
    # 永続Identity管理
    # ------------------------------------------------------------------

    def _load_identity(self) -> AGIIdentity:
        """LTMから永続Identityを復元する。なければ新規作成。"""
        try:
            data = self.ltm.recall(_IDENTITY_LTM_KEY)
            if data and isinstance(data, str):
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    print("[AGICore] 永続Identityを復元しました", flush=True)
                    return AGIIdentity.from_dict(parsed)
        except Exception:
            pass
        return AGIIdentity()

    def save_identity(self) -> None:
        """Identityを永続化する。"""
        try:
            self.ltm.learn(
                _IDENTITY_LTM_KEY,
                json.dumps(self.identity.to_dict(), ensure_ascii=False),
                confidence=1.0,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_goal(self, goal: str, context: str = "") -> Dict[str, Any]:
        """1つのゴールを認知サイクル全体で処理する。

        Gen 9 認知ループ:
        1. 知覚 (Perceive) — 世界モデル更新
        2. 内部対話 (Deliberate) — 高リスク時のみ多角的検討
        3. 倫理評価 (Ethics) — 価値体系チェック
        4. 注意選択 (Attend) — GlobalWorkspace競争
        5. メタ学習戦略選択 (Meta-Learn) — UCB1で最適戦略
        6. 予測 (Predict) — 成功確率予測
        7. 行動 (Act) — 適応的実行深度で実行
        8. 学習 (Learn) — メタ学習更新 + few-shot抽出
        9. 内発動機 (Motivate) — 自律的ゴール生成
        10. 省察 (Reflect) — 定期的自己省察
        """
        self.identity.total_goals_processed += 1

        # --- 1. 知覚フェーズ: 世界モデルを最新化 ---
        if self.world_model.needs_regrounding(max_age_seconds=300):
            self._ground_world_model()
        self._record_module_use("world_model")

        # --- 2. 内部対話フェーズ (高リスク・高不確実時のみ) ---
        deliberation = None
        preliminary_prediction = self.predictor.predict(
            action=f"GOAL: {goal}", goal=goal, context=context,
        )
        preliminary_ethics = self.value_system.assess(goal)

        if self.inner_dialogue.should_deliberate(
            goal,
            prediction_confidence=preliminary_prediction.success_probability,
            ethics_score=preliminary_ethics.total_score,
        ):
            deliberation = self.inner_dialogue.deliberate(goal, context)
            self._record_module_use("inner_dialogue")
            if deliberation.refined_goal != goal:
                print(
                    f"[InnerDialogue] ゴール洗練: {goal[:40]}... → {deliberation.refined_goal[:40]}...",
                    flush=True,
                )
                goal = deliberation.refined_goal
            if not deliberation.should_proceed:
                return {
                    "result": f"[InnerDialogue] 対話の結果、実行を見送り: {deliberation.key_concerns}",
                    "success": False,
                    "identity": self.identity.profile_summary(),
                    "insights": [],
                    "new_goals": 0,
                    "metrics": "",
                    "strategy": "",
                    "deliberation": deliberation.consensus_level,
                }

        # --- 3. 倫理評価フェーズ ---
        ethics = self.value_system.assess(goal)
        if ethics.is_blocked:
            return {
                "result": f"[ValueSystem] {ethics.recommendation}",
                "success": False,
                "identity": self.identity.profile_summary(),
                "insights": [],
                "new_goals": 0,
                "metrics": "",
                "strategy": "",
            }
        self._record_module_use("value_system")

        # --- 4. 注意選択フェーズ: GlobalWorkspace ---
        self._build_gen9_signals(goal, context, ethics, deliberation)
        broadcast = self.workspace.broadcast()
        if broadcast:
            print(
                f"[GlobalWorkspace] 注意焦点: [{broadcast.winner.source.value}] "
                f"{broadcast.winner.content[:60]}",
                flush=True,
            )
        self._record_module_use("workspace")

        # --- 5. メタ学習戦略選択フェーズ ---
        domain = getattr(deliberation, "original_goal", goal).split()[0] if deliberation else "general"
        strategy = self.meta_learner.select_strategy(domain)
        print(f"[MetaLearner] 選択戦略: {strategy.name} (UCB={strategy.ucb_score:.2f})", flush=True)
        self._record_module_use("meta_learner")

        # --- 6. 予測フェーズ ---
        prediction = self.predictor.predict(
            action=f"GOAL: {goal}",
            goal=goal,
            context=f"{context} | 戦略: {strategy.name}",
        )
        print(
            f"[PredictiveEngine] 予測: 成功={prediction.success_probability:.0%}",
            flush=True,
        )
        self._record_module_use("predictor")

        # --- 7. 行動フェーズ (適応的実行深度) ---
        complexity = self.world_model.estimate_goal_complexity(goal)
        max_iter = complexity["recommended_iterations"]

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
            max_iterations=max_iter,
            world_model=self.world_model,
        )

        # 戦略をワーキングメモリに注入
        state.working_memory["selected_strategy"] = strategy.name
        state.working_memory["strategy_description"] = strategy.description
        state.working_memory["goal_complexity"] = complexity

        # few-shot例とanti-patternをワーキングメモリに注入
        self.self_improver.inject_into_state(state)

        # 内部対話の結果を注入
        if deliberation:
            state.working_memory["deliberation_concerns"] = deliberation.key_concerns
            state.working_memory["deliberation_approach"] = deliberation.suggested_approach

        final_state = agent.run(state)
        result_text = "\n".join(final_state.observations) if final_state.observations else "（観測なし）"
        success = final_state.is_done and len(final_state.failed_steps) == 0

        # --- 8. 学習フェーズ ---
        actual_success = success
        self.predictor.record_outcome(
            prediction=prediction,
            actual_outcome=result_text[:200],
            actual_success=actual_success,
        )

        if success:
            self.identity.successful_goals += 1

        # メタ学習: 戦略の成否を記録
        reward = 1.0 if success else (0.3 if final_state.completed_steps else 0.0)
        self.meta_learner.record_outcome(
            domain=domain,
            strategy_name=strategy.name,
            goal=goal[:100],
            reward=reward,
        )

        # 軌跡から few-shot 例・anti-pattern を学習
        self.self_improver.analyze_session(final_state)
        perf_score = 1.0 if success else (0.4 if final_state.completed_steps else 0.1)
        self.self_improver.record_session_performance(
            session_id=final_state.session_id,
            goal=goal,
            domain=domain,
            score=perf_score,
        )
        self._record_module_use("self_improver")

        # --- 9. 内発動機フェーズ: 自律的ゴール生成 ---
        new_goals_count = self._generate_intrinsic_goals()
        self._record_module_use("motivation")

        # --- 10. 省察フェーズ (定期的・適応的インターバル) ---
        recent_trend = self.self_improver.get_performance_trend(window=10)
        insights_summary: List[str] = []

        if self.reflection_engine.should_reflect(recent_success_rate=recent_trend):
            insights = self.reflection_engine.reflect(self.ltm)
            insights_summary = [f"[{i.category}] {i.content[:60]}" for i in insights[:3]]

            # 戦略的ゴールを MetaCognition に追加
            strategic_goals = self.reflection_engine.generate_strategic_goals(insights, self.ltm)
            for sg in strategic_goals:
                self.meta.goal_queue.add(sg)
                new_goals_count += 1

            # 自己同一性を更新 (予測精度も含む)
            metrics = self.reflection_engine.compute_growth_metrics(self.ltm)
            metrics.prediction_accuracy = self.predictor.get_accuracy()
            self.identity.update_from_metrics(metrics)

            # 省察サイクルカウンタを更新
            self._reflection_count += 1
            self._record_module_use("reflection_engine")

            if self._reflection_count % 3 == 0:
                # AutoResearch方式: 洞察 → 実験 → メトリクス検証
                if self.llm is not None:
                    exp_results = self.experiment_runner.run_experiments_from_insights(
                        insights, max_experiments=2
                    )
                    accepted_count = sum(1 for r in exp_results if r.accepted)
                    if accepted_count == 0 and exp_results:
                        self._attempt_self_modification(insights)

        # 転移学習チェック (10ゴールごと)
        if self.identity.total_goals_processed % 10 == 0:
            self._attempt_transfer_learning(domain)

        # Identityを永続化
        self.save_identity()

        return {
            "result": result_text,
            "success": success,
            "identity": self.identity.profile_summary(),
            "insights": insights_summary,
            "new_goals": new_goals_count,
            "metrics": self.predictor.summary(),
            "strategy": strategy.name,
            "complexity": complexity["complexity"],
            "deliberation": deliberation.consensus_level if deliberation else None,
        }

    # ------------------------------------------------------------------
    # 自律ループ: GoalQueueを自動消化
    # ------------------------------------------------------------------

    def autonomous_loop(self, max_cycles: int = 10, idle_dream: bool = True) -> List[Dict[str, Any]]:
        """GoalQueueからゴールを取り出して連続実行する。

        Args:
            max_cycles: 最大実行サイクル数
            idle_dream: キュー空時にDreamフェーズを実行するか

        Returns:
            各サイクルの実行結果リスト
        """
        results = []
        print(f"[AGICore] 自律ループ開始 (最大{max_cycles}サイクル)", flush=True)

        for cycle in range(max_cycles):
            # GoalQueueからゴールを取得
            queued = self.meta.goal_queue.pop()
            if queued is None:
                if idle_dream:
                    # キューが空の場合: Dreamフェーズ
                    print(f"[AGICore] サイクル {cycle+1}: GoalQueue空 → 夢フェーズ", flush=True)
                    self._dream_phase()

                    # 内発動機で新しいゴールを生成
                    new_count = self._generate_intrinsic_goals()
                    if new_count == 0:
                        print("[AGICore] 自律ループ完了: 内発動機なし", flush=True)
                        break
                    continue
                else:
                    print("[AGICore] 自律ループ完了: GoalQueue空", flush=True)
                    break

            print(
                f"[AGICore] サイクル {cycle+1}/{max_cycles}: "
                f"{queued.goal[:50]} (優先度={queued.composite_score:.2f})",
                flush=True,
            )
            result = self.run_goal(queued.goal, context=queued.rationale)
            results.append(result)

        self.save_identity()
        return results

    # ------------------------------------------------------------------
    # Dreamフェーズ: オフライン知識統合
    # ------------------------------------------------------------------

    def _dream_phase(self) -> None:
        """アイドル時にLTMの知識を再構成・統合する。

        - 古い知識の重要度を再評価
        - 関連知識のクラスタリング
        - 世界モデルの不確実性マップを更新
        """
        print("[AGICore] 夢フェーズ: 知識の統合・再構成...", flush=True)

        # 世界モデルの不確実性を更新
        # 長期間アクセスされていない領域の不確実性を上げる
        for module, last_used in self._module_last_used.items():
            age_hours = (time.time() - last_used) / 3600
            if age_hours > 1.0:
                self.world_model.update_uncertainty(module, 0.1)

        # 戦略の転移候補を探索
        for domain in ["coding", "system", "web", "data"]:
            candidates = self.meta_learner.find_transfer_candidates(domain)
            for c in candidates[:1]:  # 各ドメイン最大1件
                self.meta_learner.apply_transfer(c)
                print(f"[Dream] 転移学習: {c.strategy_name} ({c.source_domain}→{c.target_domain})", flush=True)

        self._record_module_use("dream")

    # ------------------------------------------------------------------
    # 内発動機によるゴール生成
    # ------------------------------------------------------------------

    def _generate_intrinsic_goals(self) -> int:
        """内発的動機からゴールを生成してGoalQueueに注入する。"""
        signals = self.motivation.generate_intrinsic_goals(
            identity_assessment=self.identity.self_assessment,
            knowledge_gaps=None,
            module_last_used=self._module_last_used,
            world_model_uncertainties=self.world_model.get_uncertainty_areas(),
            ltm=self.ltm,
            max_goals=2,
        )

        if not signals:
            return 0

        queued_goals = self.motivation.to_queued_goals(signals)
        count = 0
        for qg in queued_goals:
            self.meta.goal_queue.add(qg)
            count += 1

        if count > 0:
            print(f"[IntrinsicMotivation] {count}件の内発ゴールを生成", flush=True)

        return count

    # ------------------------------------------------------------------
    # 転移学習
    # ------------------------------------------------------------------

    def _attempt_transfer_learning(self, current_domain: str) -> None:
        """メタ学習の転移候補を探して適用する。"""
        candidates = self.meta_learner.find_transfer_candidates(current_domain)
        for c in candidates[:2]:
            if c.transfer_confidence > 0.5:
                self.meta_learner.apply_transfer(c)
                print(
                    f"[MetaLearner] 転移: {c.strategy_name} "
                    f"({c.source_domain}→{c.target_domain}, 確信度={c.transfer_confidence:.0%})",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # GlobalWorkspace信号構築 (Gen 9拡張)
    # ------------------------------------------------------------------

    def _build_gen9_signals(
        self,
        goal: str,
        context: str,
        ethics: Any,
        deliberation: Any,
    ) -> None:
        """Gen 9の全認知モジュールからGlobalWorkspace信号を構築する。"""
        # 基本信号 (Gen 7)
        self.workspace.build_signals_from_state(
            goal=goal,
            context=context,
            is_stuck=False,
            value_risk=ethics.total_score,
        )

        # Gen 9: 内発動機信号
        self.workspace.receive(WorkspaceSignal(
            source=SignalSource.MOTIVATOR,
            content=f"内発動機: 自律性={self.identity.self_assessment.get('autonomy', 0.4):.0%}",
            relevance=0.6,
            urgency=0.3,
            confidence=0.7,
            tags=["motivation", "intrinsic"],
        ))

        # Gen 9: メタ学習信号
        self.workspace.receive(WorkspaceSignal(
            source=SignalSource.META_LEARNER,
            content=f"メタ学習: {self.meta_learner.summary()}",
            relevance=0.5,
            urgency=0.2,
            confidence=0.8,
            tags=["meta_learning", "strategy"],
        ))

        # Gen 9: 内部対話信号 (対話が行われた場合)
        if deliberation:
            self.workspace.receive(WorkspaceSignal(
                source=SignalSource.DELIBERATOR,
                content=f"対話合意度={deliberation.consensus_level:.0%}: {deliberation.suggested_approach[:60]}",
                relevance=0.8,
                urgency=0.5 if deliberation.key_concerns else 0.2,
                confidence=deliberation.consensus_level,
                tags=["deliberation", "consensus"],
            ))

    # ------------------------------------------------------------------
    # 状態表示
    # ------------------------------------------------------------------

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
            "experiment_loop": self.experiment_runner.summary(),
            # Gen 9
            "meta_learner": self.meta_learner.summary(),
            "motivation": self.motivation.summary(),
            "inner_dialogue": self.inner_dialogue.summary(),
            "resource_usage": self.world_model.resource_summary(),
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

    def _record_module_use(self, module_name: str) -> None:
        """モジュール使用を記録する。"""
        self._module_last_used[module_name] = time.time()
        self.motivation.record_module_activation(module_name)

    def _attempt_self_modification(self, insights: List[Insight]) -> None:
        """洞察に基づいてソースコードの自己修正を試みる。"""
        if self.llm is None:
            return

        candidates = sorted(
            [i for i in insights if i.actionable and i.category in ("weakness", "gap")],
            key=lambda i: i.confidence,
            reverse=True,
        )
        if not candidates:
            return

        insight = candidates[0]

        # 学習済みパターンを先にチェック
        pattern = self.self_modifier.find_similar_pattern(insight.content)
        if pattern:
            print(f"[SelfModifier] 学習済みパターンを適用: {pattern['insight_keywords']}", flush=True)

        keyword_to_target = {
            "計画": "hermes_agi_gen/planner.py",
            "plan": "hermes_agi_gen/planner.py",
            "実行": "hermes_agi_gen/executor.py",
            "executor": "hermes_agi_gen/executor.py",
            "記憶": "hermes_agi_gen/long_term_memory.py",
            "memory": "hermes_agi_gen/long_term_memory.py",
            "自己改善": "hermes_agi_gen/self_improvement.py",
            "few-shot": "hermes_agi_gen/self_improvement.py",
            "world": "hermes_agi_gen/world_model.py",
            "世界モデル": "hermes_agi_gen/world_model.py",
            "レビュー": "hermes_agi_gen/reviewer.py",
            "review": "hermes_agi_gen/reviewer.py",
            "動機": "hermes_agi_gen/intrinsic_motivation.py",
            "motivation": "hermes_agi_gen/intrinsic_motivation.py",
            "戦略": "hermes_agi_gen/meta_learning.py",
            "strategy": "hermes_agi_gen/meta_learning.py",
            "対話": "hermes_agi_gen/inner_dialogue.py",
            "dialogue": "hermes_agi_gen/inner_dialogue.py",
            "認知": "hermes_agi_gen/cognitive_roles.py",
            "意識": "hermes_agi_gen/consciousness.py",
            "予測": "hermes_agi_gen/predictive_engine.py",
            "省察": "hermes_agi_gen/reflection_engine.py",
        }
        target = "hermes_agi_gen/self_improvement.py"
        content_lower = insight.content.lower()
        for keyword, filepath in keyword_to_target.items():
            if keyword in content_lower:
                target = filepath
                break

        analysis = (
            f"洞察カテゴリ: {insight.category}\n"
            f"洞察内容: {insight.content}\n"
            f"確信度: {insight.confidence:.2f}\n"
            f"根拠: {insight.source}\n"
            f"最近の成功率: {self.identity.success_rate:.0%}"
        )
        print(f"[SelfModifier] 自己修正を試みる: {target}", flush=True)
        patch = self.self_modifier.propose_change(target, analysis)
        if patch:
            accepted = self.self_modifier.validate_and_commit(patch)
            status = "受け入れ ✓" if accepted else "ロールバック ✗"
            print(f"[SelfModifier] {target} → {status}: {patch.rationale[:60]}", flush=True)

            # 学習済みパターンに記録
            if accepted:
                self.self_modifier.learn_pattern(
                    insight.category,
                    ",".join(content_lower.split()[:5]),
                    target,
                    patch.rationale,
                )
        else:
            print("[SelfModifier] 適切なパッチ提案なし", flush=True)

    def _ground_world_model(self) -> None:
        """世界モデルをファイルシステムの実態にグラウンドする。"""
        self.world_model.initialize_from_filesystem(str(self.repo_root))
        age = self.world_model.grounding_age()
        if age is not None and age < 1.0:
            print("[WorldModel] グラウンディング完了", flush=True)
