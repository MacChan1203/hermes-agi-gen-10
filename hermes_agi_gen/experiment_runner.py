"""AutoResearch方式の実験ループ: 洞察→コード改変→検証→受容/ロールバック。

karpathy/autoresearch (https://github.com/karpathy/autoresearch) を参考にした
固定時間実験サイクル。AutoResearchの「train.py改変→5分実験→val_bpb測定」を
Hermesの認知アーキテクチャに翻訳する。

対応関係:
  AutoResearch           → Hermes-AGI-Gen-8
  ─────────────────────────────────────────────
  train.py               → hermes_agi_gen/*.py (ホワイトリスト対象)
  5分固定時間実験         → experiment_timeout (デフォルト300秒)
  val_bpb (低いほど良)    → ExperimentMetrics.score() (高いほど良)
  overnight 100実験       → run_experiments_from_insights()
  program.md の指示       → Insight オブジェクト (ReflectionEngine から)

安全設計:
  - SelfModifier のホワイトリストと pytest 検証を再利用
  - メトリクス改善がなければ自動ロールバック
  - 全実験を SQLite に記録 (experiments.db)
  - タイムアウト超過時はロールバック

使い方:
    runner = ExperimentRunner(agi_core=core)
    results = runner.run_experiments_from_insights(insights, max_experiments=3)
    print(runner.summary())
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agi_core import AGICore
    from .reflection_engine import Insight
    from .self_modifier import Patch

from .config import (
    EXPERIMENT_DIVERSITY_SCALE,
    EXPERIMENT_KNOWLEDGE_SCALE,
    EXPERIMENT_WEIGHT_ACCURACY,
    EXPERIMENT_WEIGHT_BREADTH,
    EXPERIMENT_WEIGHT_DIVERSITY,
    EXPERIMENT_WEIGHT_SUCCESS,
)

logger = logging.getLogger(__name__)

from .hermes_constants import get_hermes_path

_EXPERIMENT_DB_NAME = "experiments.db"
_EXPERIMENT_DB_PATH = get_hermes_path(_EXPERIMENT_DB_NAME)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_category TEXT,
    insight_content TEXT,
    target_file TEXT,
    patch_rationale TEXT,
    metrics_before TEXT,
    metrics_after TEXT,
    improvement REAL,
    accepted INTEGER NOT NULL DEFAULT 0,
    test_passed INTEGER NOT NULL DEFAULT 0,
    duration REAL,
    created_at REAL NOT NULL
);
"""

# 洞察キーワード → 修正対象ファイルのマッピング
# AutoResearch の「どのモジュールを改変するか」に相当
_INSIGHT_TO_TARGET: Dict[str, str] = {
    "計画": "hermes_agi_gen/planner.py",
    "plan": "hermes_agi_gen/planner.py",
    "プランナー": "hermes_agi_gen/planner.py",
    "実行": "hermes_agi_gen/executor.py",
    "executor": "hermes_agi_gen/executor.py",
    "エグゼキュータ": "hermes_agi_gen/executor.py",
    "記憶": "hermes_agi_gen/long_term_memory.py",
    "memory": "hermes_agi_gen/long_term_memory.py",
    "ltm": "hermes_agi_gen/long_term_memory.py",
    "自己改善": "hermes_agi_gen/self_improvement.py",
    "few-shot": "hermes_agi_gen/self_improvement.py",
    "anti-pattern": "hermes_agi_gen/self_improvement.py",
    "world": "hermes_agi_gen/world_model.py",
    "世界モデル": "hermes_agi_gen/world_model.py",
    "レビュー": "hermes_agi_gen/reviewer.py",
    "review": "hermes_agi_gen/reviewer.py",
    "meta": "hermes_agi_gen/meta_cognition.py",
    "ゴール": "hermes_agi_gen/meta_cognition.py",
    "goal": "hermes_agi_gen/meta_cognition.py",
    "deadlock": "hermes_agi_gen/meta_cognition.py",
    "デッドロック": "hermes_agi_gen/meta_cognition.py",
}


# ------------------------------------------------------------------
# データクラス
# ------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    """実験のパフォーマンス指標。

    AutoResearchの val_bpb に相当する総合スコアを提供する。
    Hermes では複数次元の指標を組み合わせる。
    """
    success_rate: float = 0.0         # セッション成功率
    prediction_accuracy: float = 0.0  # 予測エンジンの精度
    knowledge_breadth: int = 0        # LTM の知識量
    strategy_diversity: int = 0       # 戦略の多様性
    reflection_count: int = 0         # 省察回数

    def score(self) -> float:
        """総合スコア (0.0〜1.0、高いほど良い)。

        AutoResearchの val_bpb は「低いほど良い」が、
        Hermes では「高いほど良い」方向で統一する。
        """
        return (
            self.success_rate * EXPERIMENT_WEIGHT_SUCCESS
            + self.prediction_accuracy * EXPERIMENT_WEIGHT_ACCURACY
            + min(1.0, self.knowledge_breadth / EXPERIMENT_KNOWLEDGE_SCALE) * EXPERIMENT_WEIGHT_BREADTH
            + min(1.0, self.strategy_diversity / EXPERIMENT_DIVERSITY_SCALE) * EXPERIMENT_WEIGHT_DIVERSITY
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "prediction_accuracy": self.prediction_accuracy,
            "knowledge_breadth": self.knowledge_breadth,
            "strategy_diversity": self.strategy_diversity,
            "reflection_count": self.reflection_count,
            "score": self.score(),
        }

    def summary(self) -> str:
        return (
            f"成功率={self.success_rate:.0%} "
            f"予測精度={self.prediction_accuracy:.0%} "
            f"知識={self.knowledge_breadth}件 "
            f"スコア={self.score():.3f}"
        )


@dataclass
class ExperimentResult:
    """1回の実験結果。

    AutoResearchの実験1件に相当。
    洞察→パッチ→テスト→判定 のサイクルを記録する。
    """
    insight: Any                          # Insight オブジェクト
    patch: Optional[Any]                  # Patch オブジェクト、または None
    metrics_before: ExperimentMetrics
    metrics_after: ExperimentMetrics
    accepted: bool
    test_passed: bool
    duration: float
    improvement: float                    # after.score() - before.score()

    def summary(self) -> str:
        status = "✓ 採用" if self.accepted else "✗ ロールバック"
        test_str = "テスト✓" if self.test_passed else "テスト✗"
        return (
            f"[Experiment] {status} | {test_str} | "
            f"改善: {self.improvement:+.3f} | "
            f"所要: {self.duration:.0f}秒"
        )


# ------------------------------------------------------------------
# ExperimentRunner
# ------------------------------------------------------------------

class ExperimentRunner:
    """AutoResearch方式の自律実験ループ。

    ReflectionEngine が生成した Insight を受け取り、
    対応するコード改変を提案・適用し、メトリクスで改善を判定する。

    AutoResearch との対応:
      - Insight      = 改変の動機 (program.md の指示に相当)
      - SelfModifier = コード改変エンジン (train.py エディタに相当)
      - ExperimentMetrics.score() = 最適化対象スコア (val_bpb に相当)
      - experiment_timeout = 実験タイムアウト (AutoResearch は5分固定)

    Args:
        agi_core: AGICore インスタンス (SelfModifier, LTM, PredictiveEngine を共有)
        experiment_timeout: 1実験の最大秒数 (default: 300)
        db_path: 実験ログDB のパス
    """

    def __init__(
        self,
        agi_core: "AGICore",
        experiment_timeout: int = 300,
        db_path: Optional[Path] = None,
    ) -> None:
        self.agi_core = agi_core
        self.experiment_timeout = experiment_timeout
        self.db_path = Path(db_path) if db_path is not None else get_hermes_path(_EXPERIMENT_DB_NAME)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._experiment_count: int = 0

    # ------------------------------------------------------------------
    # Public: 実験実行
    # ------------------------------------------------------------------

    def run_experiment(self, insight: "Insight") -> ExperimentResult:
        """1つの洞察に基づいて実験を実行する。

        AutoResearchの1実験サイクルに相当:
          1. ベースラインメトリクスを計測 (実験前)
          2. 洞察に基づくパッチを提案・適用
          3. テスト実行で動作確認
          4. パッチ後メトリクスを計測
          5. 改善があれば確定、なければロールバック

        Returns:
            ExperimentResult
        """
        start_time = time.time()
        self._experiment_count += 1

        logger.info(
            f"[ExperimentRunner] 実験 #{self._experiment_count} 開始"
        )
        logger.info(
            f"  洞察: [{insight.category}] {insight.content[:80]}"
        )

        # ── フェーズ1: ベースラインメトリクス計測 ──
        metrics_before = self._compute_current_metrics()
        logger.info(f"  ベースライン: {metrics_before.summary()}")

        # タイムアウトチェック用ヘルパー
        def timed_out() -> bool:
            return (time.time() - start_time) >= self.experiment_timeout

        # ── フェーズ2: パッチ提案 ──
        patch = self._propose_patch_for_insight(insight)

        if patch is None:
            logger.info("  パッチ提案なし — 実験スキップ")
            return self._make_result(
                insight, None, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        logger.info(f"  パッチ提案: {patch.file_path} — {patch.rationale[:60]}")

        # ── フェーズ3: パッチ適用 ──
        if timed_out():
            logger.warning("  タイムアウト — パッチ適用をスキップ")
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        applied = self.agi_core.self_modifier.apply_patch(patch)
        if not applied:
            logger.warning("  パッチ適用失敗")
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        # Periodic timeout check after patch application
        if timed_out():
            self.agi_core.self_modifier.rollback(patch)
            logger.warning("  タイムアウト（パッチ適用後）— ロールバック")
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        # ── フェーズ4: テスト実行 ──
        if timed_out():
            self.agi_core.self_modifier.rollback(patch)
            logger.warning("  タイムアウト — ロールバック")
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        test_result = self.agi_core.self_modifier.run_tests()

        # Periodic timeout check after test execution
        if timed_out():
            self.agi_core.self_modifier.rollback(patch)
            logger.warning("  タイムアウト（テスト後）— ロールバック")
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        logger.info(
            f"  テスト: {'✓ パス' if test_result.passed else '✗ 失敗'} "
            f"({test_result.duration:.1f}秒)"
        )

        if not test_result.passed:
            # テスト失敗 → 即座にロールバック
            self.agi_core.self_modifier.rollback(patch)
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=False,
                start_time=start_time,
            )

        # ── フェーズ5: パッチ後メトリクス計測 ──
        if timed_out():
            self.agi_core.self_modifier.rollback(patch)
            logger.warning("  タイムアウト（メトリクス計測前）— ロールバック")
            return self._make_result(
                insight, patch, metrics_before, metrics_before,
                accepted=False, test_passed=True,
                start_time=start_time,
            )

        metrics_after = self._compute_current_metrics()
        improvement = metrics_after.score() - metrics_before.score()

        logger.info(f"  パッチ後:  {metrics_after.summary()}")
        logger.info(f"  改善度:    {improvement:+.3f}")

        # ── フェーズ6: 受容判定 ──
        # AutoResearchと同様: スコアが改善 (または横ばい) なら採用
        accepted = improvement >= -0.01  # わずかな誤差を許容

        if accepted:
            self._record_success_to_ltm(insight, patch, improvement)
            logger.info(f"  ✓ 採用 — {patch.file_path} を更新")
        else:
            self.agi_core.self_modifier.rollback(patch)
            logger.warning(f"  ✗ ロールバック — スコア改善なし ({improvement:+.3f})")

        result = self._make_result(
            insight, patch, metrics_before, metrics_after,
            accepted=accepted, test_passed=True,
            start_time=start_time,
        )
        logger.info(result.summary())
        return result

    def run_experiments_from_insights(
        self,
        insights: List["Insight"],
        max_experiments: int = 3,
    ) -> List[ExperimentResult]:
        """複数の洞察から実験を順番に実行する。

        AutoResearchの「overnight loop」に相当。
        actionable な weakness/gap/opportunity 洞察を確信度順に選び、
        max_experiments 回まで実験する。

        Args:
            insights: ReflectionEngine.reflect() の出力
            max_experiments: 最大実験回数

        Returns:
            ExperimentResult のリスト
        """
        # actionable な洞察を確信度順でフィルタ
        candidates = sorted(
            [
                i for i in insights
                if i.actionable and i.category in ("weakness", "gap", "opportunity")
            ],
            key=lambda i: i.confidence,
            reverse=True,
        )[:max_experiments]

        if not candidates:
            logger.info("[ExperimentRunner] 実験対象の洞察なし")
            return []

        logger.info(
            f"[ExperimentRunner] {len(candidates)} 件の洞察で実験開始 "
            f"(最大 {self.experiment_timeout}秒/実験)"
        )

        results: List[ExperimentResult] = []
        for insight in candidates:
            result = self.run_experiment(insight)
            results.append(result)

        accepted_count = sum(1 for r in results if r.accepted)
        logger.info(
            f"[ExperimentRunner] 実験完了: {len(results)}件実行 / {accepted_count}件採用"
        )
        return results

    # ------------------------------------------------------------------
    # 履歴・サマリー
    # ------------------------------------------------------------------

    def get_experiment_history(self, limit: int = 10) -> List[Dict]:
        """最近の実験履歴を返す。"""
        rows = self._conn.execute(
            """
            SELECT insight_category, insight_content, target_file,
                   improvement, accepted, test_passed, duration, created_at
            FROM experiments
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_acceptance_rate(self) -> float:
        """実験採用率を返す。"""
        row = self._conn.execute(
            "SELECT COUNT(*) as total, SUM(accepted) as acc FROM experiments"
        ).fetchone()
        if row and row["total"] > 0:
            return (row["acc"] or 0) / row["total"]
        return 0.0

    def summary(self) -> str:
        """実験ループのサマリー文字列。"""
        history = self.get_experiment_history(limit=20)
        if not history:
            return "実験履歴なし"
        accepted = sum(1 for h in history if h["accepted"])
        avg_improvement = (
            sum(h["improvement"] for h in history) / len(history)
            if history else 0.0
        )
        return (
            f"実験数={len(history)} 採用={accepted} "
            f"採用率={self.get_acceptance_rate():.0%} "
            f"平均改善={avg_improvement:+.3f}"
        )

    # ------------------------------------------------------------------
    # Private: メトリクス計測
    # ------------------------------------------------------------------

    def _compute_current_metrics(self) -> ExperimentMetrics:
        """現在のLTM・予測エンジン状態からメトリクスを計算する。

        AutoResearchが実際にモデルを動かして val_bpb を測るのに対し、
        Hermes ではLTM蓄積データと予測エンジン精度から推定する。
        副作用がなく安全。
        """
        ltm = self.agi_core.ltm
        predictor = self.agi_core.predictor
        reflection_engine = self.agi_core.reflection_engine

        # ReflectionEngineのcompute_growth_metricsを再利用
        growth = reflection_engine.compute_growth_metrics(ltm)

        return ExperimentMetrics(
            success_rate=growth.success_rate,
            prediction_accuracy=predictor.get_accuracy(),
            knowledge_breadth=growth.knowledge_breadth,
            strategy_diversity=growth.strategy_diversity,
            reflection_count=growth.reflection_count,
        )

    # ------------------------------------------------------------------
    # Private: パッチ提案
    # ------------------------------------------------------------------

    def _propose_patch_for_insight(self, insight: "Insight") -> Optional["Patch"]:
        """洞察のキーワードからターゲットファイルを決定しパッチを提案する。

        AutoResearchでいう「エージェントが train.py のどこを変えるか」の決定に相当。
        """
        # ターゲットファイルの決定 (word-boundary matching)
        target = "hermes_agi_gen/self_improvement.py"  # デフォルト
        content_normalized = insight.content.lower()
        for keyword, filepath in _INSIGHT_TO_TARGET.items():
            keyword_lower = keyword.lower()
            # Use word-boundary matching to avoid substring false positives
            # For CJK characters, simple 'in' is used since \b doesn't work for them
            if any(ord(c) > 0x2FFF for c in keyword_lower):
                # CJK keyword: substring match is appropriate
                if keyword_lower in content_normalized:
                    target = filepath
                    break
            else:
                # ASCII/Latin keyword: use word-boundary regex
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, content_normalized):
                    target = filepath
                    break

        # SelfModifierへの分析文を構築
        analysis = (
            f"=== AutoResearch方式の実験ループから呼び出し ===\n\n"
            f"実験動機となった洞察:\n"
            f"  カテゴリ: {insight.category}\n"
            f"  内容: {insight.content}\n"
            f"  確信度: {insight.confidence:.2f}\n"
            f"  根拠: {insight.source}\n\n"
            f"現在のシステム状態:\n"
            f"  成功率: {self.agi_core.identity.success_rate:.0%}\n"
            f"  処理済みゴール: {self.agi_core.identity.total_goals_processed}\n\n"
            f"目的: この洞察が示す問題を {target} の改善で解決してください。\n"
            f"保守的かつ具体的な変更のみ提案してください。"
        )

        logger.info(f"  ターゲット: {target}")
        return self.agi_core.self_modifier.propose_change(target, analysis)

    # ------------------------------------------------------------------
    # Private: 結果生成・記録
    # ------------------------------------------------------------------

    def _make_result(
        self,
        insight: "Insight",
        patch: Optional["Patch"],
        metrics_before: ExperimentMetrics,
        metrics_after: ExperimentMetrics,
        accepted: bool,
        test_passed: bool,
        start_time: float,
    ) -> ExperimentResult:
        """ExperimentResult を生成してDBに記録する。"""
        duration = time.time() - start_time
        improvement = metrics_after.score() - metrics_before.score()

        result = ExperimentResult(
            insight=insight,
            patch=patch,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            accepted=accepted,
            test_passed=test_passed,
            duration=duration,
            improvement=improvement,
        )
        self._record_to_db(result)
        return result

    def _record_to_db(self, result: ExperimentResult) -> None:
        """実験結果をSQLiteに記録する。"""
        self._conn.execute(
            """
            INSERT INTO experiments
            (insight_category, insight_content, target_file, patch_rationale,
             metrics_before, metrics_after, improvement, accepted, test_passed,
             duration, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.insight.category,
                result.insight.content[:200],
                result.patch.file_path if result.patch else None,
                result.patch.rationale[:200] if result.patch else None,
                json.dumps(result.metrics_before.to_dict()),
                json.dumps(result.metrics_after.to_dict()),
                result.improvement,
                1 if result.accepted else 0,
                1 if result.test_passed else 0,
                result.duration,
                time.time(),
            ),
        )
        self._conn.commit()

    def _record_success_to_ltm(
        self,
        insight: "Insight",
        patch: "Patch",
        improvement: float,
    ) -> None:
        """採用された実験をLTMに知識として保存する。"""
        try:
            key = f"experiment_accepted_{int(time.time())}"
            data = {
                "insight_category": insight.category,
                "insight_content": insight.content[:100],
                "target_file": patch.file_path,
                "rationale": patch.rationale[:100],
                "improvement": improvement,
                "timestamp": time.time(),
            }
            self.agi_core.ltm.learn(key, json.dumps(data, ensure_ascii=False))
        except Exception:
            logger.debug("実験受諾のLTM記録に失敗", exc_info=True)
