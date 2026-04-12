"""階層的プランナー。依存関係付きゴールツリーで複雑なタスクを管理する。

独立したサブゴールは並列実行、依存関係のあるゴールは順次実行する。
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mistral_client import MistralClient

from .config import PLANNER_THREAD_TIMEOUT, PLANNER_MAX_PARALLEL, PLANNER_RESULT_CHARS_PER_NODE

logger = logging.getLogger(__name__)


class GoalStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"   # 依存ゴールが未完了


@dataclass
class GoalNode:
    """ゴールツリーの1ノード。"""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    role: str = "worker"          # worker / researcher / developer / critic
    status: GoalStatus = GoalStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    parent_id: Optional[str] = None
    children: List[GoalNode] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)   # goal_idのリスト
    priority: int = 0             # 高いほど優先
    is_parallel: bool = True      # 並列実行可能かどうか

    def is_ready(self, completed_ids: set[str]) -> bool:
        """依存するゴールが全て完了しているか確認する。"""
        return all(dep in completed_ids for dep in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "role": self.role,
            "status": self.status,
            "result": self.result[:200] if self.result else None,
            "parent_id": self.parent_id,
            "depends_on": self.depends_on,
            "priority": self.priority,
        }


class GoalTree:
    """ゴールのツリー構造を管理する。"""

    def __init__(self, root_goal: str) -> None:
        self.root = GoalNode(goal=root_goal, role="orchestrator")
        self._nodes: Dict[str, GoalNode] = {self.root.goal_id: self.root}

    def add_child(
        self,
        parent_id: str,
        goal: str,
        role: str = "worker",
        depends_on: Optional[List[str]] = None,
        is_parallel: bool = True,
        priority: int = 0,
    ) -> GoalNode:
        """子ゴールを追加する。"""
        node = GoalNode(
            goal=goal,
            role=role,
            parent_id=parent_id,
            depends_on=depends_on or [],
            is_parallel=is_parallel,
            priority=priority,
        )
        parent = self._nodes.get(parent_id)
        if parent:
            parent.children.append(node)
        self._nodes[node.goal_id] = node
        return node

    def get_ready_nodes(self) -> List[GoalNode]:
        """実行準備ができたノード (依存完了済み、未実行) を返す。"""
        completed_ids = {
            nid for nid, n in self._nodes.items()
            if n.status == GoalStatus.COMPLETED
        }
        ready = [
            n for n in self._nodes.values()
            if n.status == GoalStatus.PENDING
            and n.is_ready(completed_ids)
            and n.goal_id != self.root.goal_id
        ]
        # 優先度順にソート
        ready.sort(key=lambda n: n.priority, reverse=True)
        return ready

    def get_parallel_batch(self) -> List[GoalNode]:
        """並列実行できるノードのバッチを返す。"""
        ready = self.get_ready_nodes()
        parallel = [n for n in ready if n.is_parallel]
        sequential = [n for n in ready if not n.is_parallel]

        if parallel:
            return parallel  # 並列ノードを全て返す
        if sequential:
            return [sequential[0]]  # 順次ノードは1つずつ
        return []

    def mark_completed(self, goal_id: str, result: str) -> None:
        node = self._nodes.get(goal_id)
        if node:
            node.status = GoalStatus.COMPLETED
            node.result = result

    def mark_failed(self, goal_id: str, error: str) -> None:
        node = self._nodes.get(goal_id)
        if node:
            node.status = GoalStatus.FAILED
            node.error = error

    def is_complete(self) -> bool:
        """全子ノードが完了または失敗しているか。"""
        children = [n for n in self._nodes.values() if n.goal_id != self.root.goal_id]
        if not children:
            return False
        return all(n.status in (GoalStatus.COMPLETED, GoalStatus.FAILED) for n in children)

    def summary(self) -> str:
        """ツリーの状態サマリーを返す。"""
        nodes = list(self._nodes.values())
        total = len(nodes) - 1  # rootを除く
        completed = sum(1 for n in nodes if n.status == GoalStatus.COMPLETED)
        failed = sum(1 for n in nodes if n.status == GoalStatus.FAILED)
        in_progress = sum(1 for n in nodes if n.status == GoalStatus.IN_PROGRESS)
        pending = sum(1 for n in nodes if n.status == GoalStatus.PENDING)
        return f"合計:{total} 完了:{completed} 失敗:{failed} 実行中:{in_progress} 待機:{pending}"

    @property
    def nodes(self) -> Dict[str, GoalNode]:
        """ノードマップへの読み取りアクセス。"""
        return self._nodes


_DECOMPOSE_HIERARCHICAL = """\
あなたはAGIエージェントのタスク分解専門家です。
以下のゴールを依存関係付きのサブゴールに分解してください。

ゴール: {goal}
コンテキスト: {context}

分解ルール:
1. 独立して実行できるタスクは並列 (is_parallel: true)
2. 前のタスクの結果が必要なタスクは依存関係を設定 (depends_on: [goal_id])
3. 各タスクにロールを割り当て:
   - perceiver:   入力の意図・要件を明確化する
   - memorist:    ローカルファイル・コードを調査して情報を収集する
   - ethicist:    実行計画の安全性・倫理的問題を評価する
   - strategist:  実行計画を立案する
   - innovator:   創造的・代替的アプローチを提案する
   - executor:    コード実行・ファイル操作・テスト実行を担当する
   - critic:      成果物の評価・確認・改善提案を行う
   - goal_manager: 追加ゴールの特定と優先付けを行う
4. タスクは2〜4個に収める

必ずJSON配列のみで返してください:
[
  {{"goal_id": "a1", "goal": "...", "role": "memorist", "depends_on": [], "is_parallel": true, "priority": 1}},
  {{"goal_id": "a2", "goal": "...", "role": "executor", "depends_on": ["a1"], "is_parallel": false, "priority": 0}}
]\
"""


class HierarchicalPlanner:
    """階層的ゴールツリーを生成・管理するプランナー。"""

    def __init__(self, llm: Optional[MistralClient] = None) -> None:
        self.llm = llm

    # ------------------------------------------------------------------
    # DAG サイクル検出
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_cycles(tree: GoalTree) -> list[str]:
        """DFS（白/灰/黒の彩色法）でゴールツリーのサイクルを検出する。

        Returns:
            検出されたバックエッジのリスト ("from_id -> to_id" 形式)。
            サイクルがなければ空リスト。
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in tree.nodes}
        back_edges: list[str] = []

        def dfs(node_id: str) -> None:
            color[node_id] = GRAY
            node = tree.nodes.get(node_id)
            if node is None:
                color[node_id] = BLACK
                return
            # depends_on は "この node が依存する先" なので、
            # グラフ的には node_id → dep_id の辺と見なす
            for dep_id in node.depends_on:
                if dep_id not in color:
                    continue
                if color[dep_id] == GRAY:
                    back_edges.append(f"{node_id} -> {dep_id}")
                elif color[dep_id] == WHITE:
                    dfs(dep_id)
            color[node_id] = BLACK

        for nid in tree.nodes:
            if color[nid] == WHITE:
                dfs(nid)

        return back_edges

    @staticmethod
    def _remove_back_edges(tree: GoalTree, back_edges: list[str]) -> None:
        """検出されたバックエッジの依存関係を除去する。"""
        for edge in back_edges:
            parts = edge.split(" -> ")
            if len(parts) != 2:
                continue
            from_id, to_id = parts
            node = tree.nodes.get(from_id)
            if node and to_id in node.depends_on:
                node.depends_on.remove(to_id)
                logger.warning(
                    "サイクル検出: %s → %s のバックエッジ依存を除去しました",
                    from_id, to_id,
                )

    def decompose(self, goal: str, context: str = "") -> GoalTree:
        """ゴールをGoalTreeに分解する。"""
        tree = GoalTree(goal)

        if self.llm is None:
            # LLMなし: デフォルト2ステップ分解
            research = tree.add_child(
                tree.root.goal_id, f"調査: {goal}", role="researcher", is_parallel=True, priority=1
            )
            tree.add_child(
                tree.root.goal_id, f"実行: {goal}", role="developer",
                depends_on=[research.goal_id], is_parallel=False, priority=0
            )
            return tree

        data = self.llm.chat_json(
            [{"role": "user", "content": _DECOMPOSE_HIERARCHICAL.format(goal=goal, context=context)}],
            temperature=0.3,
            max_tokens=1024,
        )

        if not isinstance(data, list) or not data:
            # フォールバック
            tree.add_child(tree.root.goal_id, goal, role="worker", is_parallel=True)
            return tree

        # 仮のgoal_idマッピング (LLMが返したIDを実際のIDに変換)
        id_map: Dict[str, str] = {}
        nodes_data = []

        for item in data:
            if not isinstance(item, dict):
                continue
            old_id = item.get("goal_id", "")
            node = tree.add_child(
                tree.root.goal_id,
                goal=item.get("goal", ""),
                role=item.get("role", "worker"),
                depends_on=[],  # 後で設定
                is_parallel=bool(item.get("is_parallel", True)),
                priority=int(item.get("priority", 0)),
            )
            if old_id:
                id_map[old_id] = node.goal_id
            nodes_data.append((node, item.get("depends_on", [])))

        # 依存関係をIDマッピングで解決
        for node, old_deps in nodes_data:
            node.depends_on = [id_map.get(d, d) for d in old_deps if d in id_map]

        # DAG サイクル検出・除去
        back_edges = self._detect_cycles(tree)
        if back_edges:
            self._remove_back_edges(tree, back_edges)

        return tree

    def execute_tree(
        self,
        tree: GoalTree,
        worker_fn: Callable[[GoalNode], str],
        max_parallel: int = PLANNER_MAX_PARALLEL,
    ) -> str:
        """GoalTreeを実行する。並列ノードはスレッドで並列実行する。

        Args:
            tree: 実行するGoalTree
            worker_fn: ノードを受け取り結果文字列を返す関数
            max_parallel: 最大並列数

        Returns:
            全ノードの実行結果をまとめた文字列
        """
        results: List[str] = []

        while not tree.is_complete():
            batch = tree.get_parallel_batch()
            if not batch:
                break

            # 並列バッチを制限
            batch = batch[:max_parallel]

            if len(batch) == 1 or not batch[0].is_parallel:
                # 単一実行
                node = batch[0]
                node.status = GoalStatus.IN_PROGRESS
                try:
                    result = worker_fn(node)
                    tree.mark_completed(node.goal_id, result)
                    results.append(
                        f"[{node.role}] {node.goal[:50]}: "
                        f"{result[:PLANNER_RESULT_CHARS_PER_NODE]}"
                    )
                except Exception as e:
                    tree.mark_failed(node.goal_id, str(e))
                    results.append(f"[{node.role}][失敗] {node.goal[:50]}: {e}")
            else:
                # 並列実行 — Lock で thread_results への書き込みを保護
                threads = []
                thread_results: Dict[str, Any] = {}
                results_lock = threading.Lock()

                for node in batch:
                    node.status = GoalStatus.IN_PROGRESS

                def run_node(n: GoalNode) -> None:
                    try:
                        r = worker_fn(n)
                        with results_lock:
                            thread_results[n.goal_id] = ("ok", r)
                    except Exception as e:
                        with results_lock:
                            thread_results[n.goal_id] = ("err", str(e))

                for node in batch:
                    t = threading.Thread(target=run_node, args=(node,), daemon=True)
                    threads.append((node, t))
                    t.start()

                for node, t in threads:
                    t.join(timeout=PLANNER_THREAD_TIMEOUT)

                for node in batch:
                    with results_lock:
                        status, val = thread_results.get(node.goal_id, ("err", "タイムアウト"))
                    if status == "ok":
                        tree.mark_completed(node.goal_id, val)
                        results.append(
                            f"[{node.role}][並列] {node.goal[:50]}: "
                            f"{val[:PLANNER_RESULT_CHARS_PER_NODE]}"
                        )
                    else:
                        tree.mark_failed(node.goal_id, val)
                        results.append(f"[{node.role}][並列失敗] {node.goal[:50]}: {val}")

        return "\n".join(results) or "（実行結果なし）"
