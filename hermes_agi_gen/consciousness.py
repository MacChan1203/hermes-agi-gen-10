"""グローバル・ワークスペース理論に基づく意識的情報統合。

Baars (1988) の Global Workspace Theory (GWT) を実装:
- 複数の専門認知モジュールが「意識」(注意) をめぐって競争する
- 勝者のコンテンツが全モジュールにブロードキャストされる
- これにより統合的・一貫した認知が実現する

各モジュールは WorkspaceSignal を生成し、GlobalWorkspace に送信する。
AttentionMechanism が重要度を評価し、最も重要なシグナルを選択する。
選択されたシグナルは全モジュールに共有され、次の処理に影響を与える。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalSource(str, Enum):
    """シグナルの発生源となる認知モジュール。"""
    PERCEIVER = "perceiver"          # 入力理解・解釈
    STRATEGIST = "strategist"        # 戦略的計画
    EXECUTOR = "executor"            # 行動実行
    CRITIC = "critic"                # 品質評価
    MEMORIST = "memorist"            # 記憶管理
    GOAL_MANAGER = "goal_manager"    # ゴール管理
    INNOVATOR = "innovator"          # 創造的解決
    ETHICIST = "ethicist"            # 価値整合


@dataclass
class WorkspaceSignal:
    """グローバル・ワークスペースに送信される認知シグナル。

    各専門モジュールが生成する情報のパケット。
    重要度スコアによって注意競争に参加する。
    """
    source: SignalSource             # 発生源モジュール
    content: str                     # シグナルの内容
    relevance: float                 # 目標への関連度 (0.0〜1.0)
    urgency: float                   # 緊急度 (0.0〜1.0)
    confidence: float                # 確信度 (0.0〜1.0)
    tags: List[str] = field(default_factory=list)   # 分類タグ
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def attention_score(self) -> float:
        """注意競争スコア: 関連度・緊急度・確信度の加重平均。"""
        return (self.relevance * 0.4) + (self.urgency * 0.35) + (self.confidence * 0.25)

    def __repr__(self) -> str:
        return (
            f"WorkspaceSignal({self.source.value}, "
            f"score={self.attention_score:.2f}, "
            f"{self.content[:50]}...)"
        )


@dataclass
class BroadcastEvent:
    """グローバル・ワークスペースのブロードキャストイベント。

    注意競争で勝ったシグナルが全モジュールに共有される際の情報。
    """
    winner: WorkspaceSignal
    all_signals: List[WorkspaceSignal]
    broadcast_time: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def runner_up(self) -> Optional[WorkspaceSignal]:
        """2位のシグナル（代替案として参照できる）。"""
        others = [s for s in self.all_signals if s is not self.winner]
        if not others:
            return None
        return max(others, key=lambda s: s.attention_score)


class AttentionMechanism:
    """注意メカニズム: 複数シグナルの中から最も重要なものを選択する。

    生物的な注意と同様に、関連度・緊急度・確信度を総合評価し、
    最も重要なシグナルを選択して意識的処理に渡す。
    """

    def __init__(self, threshold: float = 0.3) -> None:
        """
        Args:
            threshold: このスコア未満のシグナルは無視される
        """
        self.threshold = threshold
        self._history: List[BroadcastEvent] = []

    def compete(self, signals: List[WorkspaceSignal]) -> Optional[BroadcastEvent]:
        """注意競争を実行し、勝者を選択してブロードキャストイベントを返す。

        Args:
            signals: 競争に参加するシグナルのリスト

        Returns:
            勝者が存在する場合は BroadcastEvent、なければ None
        """
        if not signals:
            return None

        # 閾値以上のシグナルのみを候補にする
        candidates = [s for s in signals if s.attention_score >= self.threshold]
        if not candidates:
            # 全シグナルが閾値未満でも最高スコアを選ぶ
            candidates = signals

        # 注意スコアで勝者を決定
        winner = max(candidates, key=lambda s: s.attention_score)

        event = BroadcastEvent(winner=winner, all_signals=signals)
        self._history.append(event)

        # 履歴を最新100件に制限
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return event

    def get_history(self, limit: int = 10) -> List[BroadcastEvent]:
        """最近のブロードキャスト履歴を返す。"""
        return self._history[-limit:]

    def source_attention_stats(self) -> Dict[str, float]:
        """各モジュールが注意を獲得した割合を返す。"""
        if not self._history:
            return {}
        counts: Dict[str, int] = {}
        for event in self._history:
            src = event.winner.source.value
            counts[src] = counts.get(src, 0) + 1
        total = len(self._history)
        return {k: v / total for k, v in counts.items()}


class GlobalWorkspace:
    """グローバル・ワークスペース: AGIの統合的認知中枢。

    すべての専門認知モジュール (perceiver, strategist, etc.) が情報を送り、
    注意競争を通じて最重要情報が全モジュールに共有される。

    これにより:
    - 断片的な処理が統合的な認知になる
    - 全モジュールが共通の「意識」を持てる
    - 一貫した意思決定が可能になる
    """

    def __init__(self) -> None:
        self.attention = AttentionMechanism()
        self._current_signals: List[WorkspaceSignal] = []
        self._shared_context: Dict[str, Any] = {}
        self._last_broadcast: Optional[BroadcastEvent] = None

    def receive(self, signal: WorkspaceSignal) -> None:
        """モジュールからシグナルを受信する。"""
        self._current_signals.append(signal)

    def broadcast(self) -> Optional[BroadcastEvent]:
        """現在蓄積されたシグナルで注意競争を実施し、全モジュールに共有する。

        Returns:
            ブロードキャストイベント（勝者情報）
        """
        if not self._current_signals:
            return None

        event = self.attention.compete(self._current_signals)
        if event:
            # 共有コンテキストを更新
            self._shared_context["last_broadcast_source"] = event.winner.source.value
            self._shared_context["last_broadcast_content"] = event.winner.content
            self._shared_context["last_broadcast_time"] = event.broadcast_time
            self._last_broadcast = event

        # シグナルをクリア（次のサイクルへ）
        self._current_signals = []

        return event

    def get_context(self) -> Dict[str, Any]:
        """全モジュールが参照できる共有コンテキストを返す。"""
        return dict(self._shared_context)

    def get_last_broadcast(self) -> Optional[BroadcastEvent]:
        """最後のブロードキャストイベントを返す。"""
        return self._last_broadcast

    def inject_context(self, key: str, value: Any) -> None:
        """共有コンテキストに情報を注入する（外部からの強制共有）。"""
        self._shared_context[key] = value

    def summary(self) -> str:
        """ワークスペースの現状サマリを返す。"""
        stats = self.attention.source_attention_stats()
        stats_str = ", ".join(f"{k}: {v:.0%}" for k, v in stats.items()) if stats else "なし"
        last = self._last_broadcast
        last_str = (
            f"{last.winner.source.value}: {last.winner.content[:40]}..."
            if last
            else "なし"
        )
        return (
            f"[GlobalWorkspace] 注意統計={stats_str} | 最終ブロードキャスト={last_str}"
        )

    def build_signals_from_state(
        self,
        goal: str,
        context: str = "",
        observations: Optional[List[str]] = None,
        is_stuck: bool = False,
        value_risk: float = 0.0,
    ) -> None:
        """エージェント状態から基本シグナルセットを生成してワークスペースに送信する。

        Orchestratorがシグナルを手動生成しなくても済むようにするヘルパー。
        """
        obs = observations or []

        # Perceiver: 目標の解釈
        self.receive(WorkspaceSignal(
            source=SignalSource.PERCEIVER,
            content=f"目標: {goal}" + (f" | コンテキスト: {context[:80]}" if context else ""),
            relevance=0.9,
            urgency=0.7,
            confidence=0.85,
            tags=["goal", "input"],
        ))

        # Memorist: 過去の観測を注入
        if obs:
            self.receive(WorkspaceSignal(
                source=SignalSource.MEMORIST,
                content=f"過去の観測({len(obs)}件): {obs[-1][:80]}" if obs else "観測なし",
                relevance=0.7,
                urgency=0.4,
                confidence=0.75,
                tags=["memory", "context"],
            ))

        # Ethicist: リスク評価
        if value_risk > 0.5:
            self.receive(WorkspaceSignal(
                source=SignalSource.ETHICIST,
                content=f"倫理リスク検出: score={value_risk:.2f} — 慎重な評価が必要",
                relevance=0.8,
                urgency=value_risk,
                confidence=0.9,
                tags=["ethics", "risk"],
            ))

        # Goal Manager: 行き詰まり時は緊急シグナル
        if is_stuck:
            self.receive(WorkspaceSignal(
                source=SignalSource.GOAL_MANAGER,
                content="行き詰まり検出: 戦略の転換が必要",
                relevance=0.9,
                urgency=0.95,
                confidence=0.85,
                tags=["stuck", "pivot"],
            ))
