"""マルチエージェント オーケストレーター。

使い方:
    from hermes_agi_gen import AgentOrchestrator, MistralClient

    llm = MistralClient(model="mistral")          # Ollama ローカル
    # llm = MistralClient(model="mistral-small-latest")  # Mistral API (要 MISTRAL_API_KEY)

    orch = AgentOrchestrator(llm=llm)
    result = orch.run("このプロジェクトの構造を調べて改善案をまとめてください")
    print(result)
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .agent_message import AgentMessage
from .agent_runner import HermesAgentV9
from .agent_state import AgentState
from .hierarchical_planner import GoalNode, GoalTree, HierarchicalPlanner
from .mistral_client import MistralClient
from .state_store import SessionDB

# ロール別システムプロンプト
ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "researcher": (
        "あなたはローカルファイル調査専門エージェントです。"
        "シェルコマンド (ls, find, cat, grep など) でファイル・コードを調べ、情報をまとめます。"
        "インターネットやブラウザは使えません。ローカルのファイルのみを対象にしてください。"
        "得られた情報を日本語で簡潔にまとめてください。"
    ),
    "developer": (
        "あなたはコード実行・ファイル操作専門エージェントです。"
        "シェルコマンドの実行、ファイルの読み書き、コードの実行を担当します。"
        "インターネットやブラウザは使えません。"
        "実行結果を日本語で説明してください。"
    ),
    "critic": (
        "あなたは成果物を評価する批評エージェントです。"
        "他のエージェントの出力を確認し、品質・正確性・改善点を評価してください。"
        "評価結果を日本語で報告してください。"
    ),
}

_DECOMPOSE_SYSTEM = """\
あなたはローカルマシン上で動作するマルチエージェントシステムのオーケストレーターです。
ユーザーの目標を 1〜2 個のサブタスクに分解してください。
各サブタスクには以下のいずれかのロールを割り当ててください:
- researcher: ローカルファイル・コードの調査 (ls, find, cat, grep などを使用)
- developer: コード実行・ファイル操作・テスト実行
- critic: 成果物の評価・確認・改善提案

【重要】インターネット・Web検索は使えません。すべてローカル操作のみです。
必ず JSON 配列のみで返してください (説明不要):
[{"role": "researcher", "task": "..."}, ...]\
"""

_SYNTHESIZE_SYSTEM = """\
あなたはマルチエージェントシステムのオーケストレーターです。
複数のエージェントの実行結果を受け取り、最終的な成果を日本語で簡潔にまとめてください。\
"""


class AgentOrchestrator:
    """目標を分解してワーカーエージェントに委任し、結果を統合する。

    Args:
        llm: MistralClient インスタンス
        repo_root: 作業ディレクトリ
        session_db: セッション DB (省略時は新規作成)
        max_worker_iterations: 各ワーカーの最大イテレーション数
    """

    def __init__(
        self,
        llm: MistralClient,
        repo_root: str | Path = ".",
        session_db: Optional[SessionDB] = None,
        max_worker_iterations: int = 4,
        use_hierarchical: bool = True,
    ) -> None:
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.session_db = session_db or SessionDB()
        self.max_worker_iterations = max_worker_iterations
        self.orchestrator_session_id = str(uuid.uuid4())
        self.hierarchical_planner = HierarchicalPlanner(llm=llm) if use_hierarchical else None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, goal: str, context: str = "") -> str:
        """目標を受け取り、ワーカーに委任して最終結果を返す。

        use_hierarchical=True の場合、階層的プランニングで依存関係付きゴールツリーを生成し、
        独立したサブゴールを並列実行する。
        """
        self.session_db.create_session(
            self.orchestrator_session_id,
            source="orchestrator",
            model=self.llm.model,
            title=f"[Orchestrator] {goal}",
        )
        self.session_db.append_message(self.orchestrator_session_id, "user", goal)

        # 階層的プランニングを使用
        if self.hierarchical_planner is not None:
            return self._run_hierarchical(goal, context)

        return self._run_flat(goal)

    def _run_hierarchical(self, goal: str, context: str = "") -> str:
        """階層的GoalTreeを使って目標を実行する。"""
        print(f"[Orchestrator] 階層的ゴールツリーを生成中...", flush=True)
        assert self.hierarchical_planner is not None
        tree = self.hierarchical_planner.decompose(goal, context=context)

        tree_summary = tree.summary()
        self.session_db.append_message(
            self.orchestrator_session_id, "assistant",
            f"ゴールツリー: {tree_summary}"
        )
        print(f"[Orchestrator] {tree_summary}", flush=True)

        # GoalNodeをAgentMessageに変換してワーカーを実行
        def worker_fn(node: GoalNode) -> str:
            msg = AgentMessage(
                sender="orchestrator",
                receiver=node.role,
                task=node.goal,
                context="\n".join(
                    f"[{n.role}の結果] {n.result[:200]}"
                    for n in tree._nodes.values()
                    if n.goal_id in node.depends_on and n.result
                ),
                session_id=self.orchestrator_session_id,
            )
            print(f"  [GoalNode:{node.goal_id}] {node.role}: {node.goal[:60]}", flush=True)
            completed = self._run_worker(msg)
            self.session_db.append_message(
                self.orchestrator_session_id, "tool",
                f"[{node.role}:{node.goal_id}] {completed.result or '（結果なし）'}",
                tool_name="worker",
            )
            return completed.result or ""

        tree.execute_tree(tree, worker_fn, max_parallel=3)

        # 結果をAgentMessage形式に変換して統合
        messages = []
        for node in tree._nodes.values():
            if node.goal_id == tree.root.goal_id:
                continue
            msg = AgentMessage(
                sender=node.role, receiver="orchestrator",
                task=node.goal, result=node.result,
                status=node.status.value,
                session_id=self.orchestrator_session_id,
            )
            messages.append(msg)

        final = self._synthesize(goal, messages)
        self.session_db.append_message(self.orchestrator_session_id, "assistant", final)
        self.session_db.end_session(self.orchestrator_session_id, "completed")
        return final

    def _run_flat(self, goal: str) -> str:
        """従来のフラットな順次実行。"""
        print(f"[Orchestrator] 目標を分解中...", flush=True)
        sub_tasks = self._decompose(goal)
        if not sub_tasks:
            sub_tasks = [
                AgentMessage(
                    sender="orchestrator",
                    receiver="developer",
                    task=goal,
                    session_id=self.orchestrator_session_id,
                )
            ]

        self.session_db.append_message(
            self.orchestrator_session_id,
            "assistant",
            "サブタスク: " + " / ".join(f"[{m.receiver}] {m.task}" for m in sub_tasks),
        )

        results: List[AgentMessage] = []
        for i, msg in enumerate(sub_tasks, 1):
            print(f"[Worker {i}/{len(sub_tasks)}] {msg.receiver}: {msg.task[:60]}", flush=True)
            completed = self._run_worker(msg)
            results.append(completed)
            print(f"[Worker {i}/{len(sub_tasks)}] 完了 (status={completed.status})", flush=True)
            self.session_db.append_message(
                self.orchestrator_session_id, "tool",
                f"[{msg.receiver}] {completed.result or '（結果なし）'}",
                tool_name="worker",
            )

        final = self._synthesize(goal, results)
        self.session_db.append_message(self.orchestrator_session_id, "assistant", final)
        self.session_db.end_session(self.orchestrator_session_id, "completed")
        return final

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _decompose(self, goal: str) -> List[AgentMessage]:
        """Mistral を呼び出して目標をサブタスクの AgentMessage リストに変換する。"""
        data = self.llm.chat_json(
            [
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": f"目標: {goal}"},
            ]
        )
        if not isinstance(data, list):
            return []
        messages: List[AgentMessage] = []
        for item in data:
            if isinstance(item, dict) and "role" in item and "task" in item:
                messages.append(
                    AgentMessage(
                        sender="orchestrator",
                        receiver=str(item["role"]),
                        task=str(item["task"]),
                        session_id=self.orchestrator_session_id,
                    )
                )
        return messages

    def _run_worker(self, msg: AgentMessage) -> AgentMessage:
        """ワーカーエージェントを生成してサブタスクを実行する。"""
        role = msg.receiver
        system_prompt = ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS["developer"])

        agent = HermesAgentV9(
            repo_root=self.repo_root,
            model=self.llm.model,
            max_iterations=self.max_worker_iterations,
            session_db=self.session_db,
            source=f"worker/{role}",
            llm=self.llm,
            agent_role=role,
            system_prompt=system_prompt,
        )

        state = AgentState(
            user_goal=msg.task,
            success_criteria=["タスクを完了できた", "結果を日本語で説明できる"],
            constraints=["破壊的操作はしない", "まず読んで把握する"],
            max_iterations=self.max_worker_iterations,
            agent_role=role,
            parent_session_id=self.orchestrator_session_id,
        )

        final_state = agent.run(state)
        result_text = "\n".join(final_state.observations) if final_state.observations else "（観測なし）"

        msg.result = result_text
        msg.status = "success" if final_state.is_done else "partial"
        return msg

    def _synthesize(self, goal: str, results: List[AgentMessage]) -> str:
        """全ワーカーの結果を Mistral で統合して最終回答を生成する。"""
        results_text = "\n\n".join(
            f"=== {r.receiver} の結果 ===\n{r.result or '（なし）'}" for r in results
        )
        response = self.llm.chat(
            [
                {"role": "system", "content": _SYNTHESIZE_SYSTEM},
                {
                    "role": "user",
                    "content": f"目標: {goal}\n\n{results_text}\n\n上記の結果をまとめてください。",
                },
            ],
            max_tokens=1024,
        )
        return response or "（統合結果を生成できませんでした）"
