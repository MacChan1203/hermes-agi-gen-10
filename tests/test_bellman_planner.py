"""Bellman 最適方程式プランナーのテスト。"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_agi_gen.agent_state import AgentState
from hermes_agi_gen.bellman_planner import (
    BellmanEvaluator,
    BellmanPlanner,
    QTable,
    action_signature,
    goal_relevance,
    state_signature,
)
from hermes_agi_gen.config import (
    BELLMAN_GAMMA,
    BELLMAN_QTABLE_PER_STATE_CAP,
)
from hermes_agi_gen.long_term_memory import LongTermMemory
from hermes_agi_gen.predictive_engine import Prediction, PredictiveEngine
from hermes_agi_gen.value_system import ValueSystem


# ----------------------------------------------------------------------
# ヘルパー
# ----------------------------------------------------------------------
def _make_state(goal: str = "READMEを確認する", **kw) -> AgentState:
    s = AgentState(user_goal=goal)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ----------------------------------------------------------------------
# シグネチャ
# ----------------------------------------------------------------------
class TestSignatures:
    def test_state_signature_stable(self):
        s1 = _make_state(domain="coding", completed_steps=["READ: a.py"])
        s2 = _make_state(domain="coding", completed_steps=["READ: a.py"])
        assert state_signature(s1) == state_signature(s2)

    def test_state_signature_changes_with_progress(self):
        s = _make_state()
        sig0 = state_signature(s)
        s.completed_steps.append("CMD: ls")
        assert state_signature(s) != sig0

    def test_action_signature_groups_by_type(self):
        a = action_signature("READ: foo.py")
        b = action_signature("READ: foo.py")
        c = action_signature("WRITE: foo.py")
        assert a == b
        assert a.startswith("READ:")
        assert c.startswith("WRITE:")
        assert a != c

    def test_goal_relevance_overlap(self):
        # トークン重複あり
        rel = goal_relevance("READ: README.md", "READMEを確認する")
        assert rel >= 0.5
        # 完全に無関係でもデフォルト 0.5 を下回らない
        rel2 = goal_relevance("CMD: ls", "全く異なるトピック")
        assert rel2 == 0.5


# ----------------------------------------------------------------------
# BellmanEvaluator (Phase A)
# ----------------------------------------------------------------------
class TestBellmanEvaluator:
    def test_q_value_decomposes(self):
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        q, r, v = ev.q_value("READ: README.md", "READMEを読む")
        assert q == pytest.approx(r + BELLMAN_GAMMA * v)
        assert 0.0 <= v <= 1.0

    def test_done_is_terminal(self):
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        _q, _r, v = ev.q_value("DONE: 完了しました", "目標")
        assert v == 0.0  # 終端

    def test_blocked_action_gets_low_reward(self):
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        # ValueSystem は "rm -rf" をブロック
        r_safe = ev.reward("READ: foo.py", "fooを読む")
        r_unsafe = ev.reward("CMD: rm -rf /", "削除する")
        assert r_unsafe < r_safe
        # ブロックされた行動は utility=0、終端ボーナスもつかない
        assert r_unsafe == 0.0

    def test_uses_predictor_for_v_next(self):
        predictor = MagicMock(spec=PredictiveEngine)
        predictor.predict.return_value = Prediction(
            action="CMD: ls",
            predicted_outcome="ok",
            success_probability=0.9,
            confidence=0.8,
        )
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=predictor)
        _q, _r, v = ev.q_value("CMD: ls", "ファイル一覧")
        assert v > 0.0
        predictor.predict.assert_called_once()


# ----------------------------------------------------------------------
# QTable (Phase B)
# ----------------------------------------------------------------------
class TestQTable:
    def test_get_unknown_returns_zero(self):
        qt = QTable(ltm=None)
        q, n = qt.get("s1", "READ:abc")
        assert (q, n) == (0.0, 0)

    def test_td_update_moves_q_toward_target(self):
        qt = QTable(ltm=None, alpha=0.5, gamma=0.9)
        # 初回更新: Q ← 0 + 0.5 * (1.0 + 0.9*0 - 0) = 0.5
        new_q, n = qt.update("s1", "a1", reward=1.0, next_state_sig="s2", terminal=True)
        assert new_q == pytest.approx(0.5)
        assert n == 1
        # 2回目: target = 1.0, Q ← 0.5 + 0.5*(1.0 - 0.5) = 0.75
        new_q2, n2 = qt.update("s1", "a1", reward=1.0, next_state_sig="s2", terminal=True)
        assert new_q2 == pytest.approx(0.75)
        assert n2 == 2

    def test_best_q_returns_max(self):
        qt = QTable(ltm=None, alpha=1.0, gamma=0.9)
        qt.update("s1", "a1", reward=0.2, next_state_sig="s2", terminal=True)
        qt.update("s1", "a2", reward=0.8, next_state_sig="s2", terminal=True)
        assert qt.best_q("s1") == pytest.approx(0.8)

    def test_persists_to_ltm(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "ltm.sqlite"
            ltm1 = LongTermMemory(db_path=db_path)
            qt1 = QTable(ltm=ltm1, alpha=0.5, gamma=0.9)
            qt1.update("sX", "READ:abc", reward=1.0, next_state_sig="sY", terminal=True)

            # 別インスタンスで読み戻す
            ltm2 = LongTermMemory(db_path=db_path)
            qt2 = QTable(ltm=ltm2, alpha=0.5, gamma=0.9)
            q, n = qt2.get("sX", "READ:abc")
            assert q == pytest.approx(0.5)
            assert n == 1

    def test_per_state_cap_lru(self):
        qt = QTable(ltm=None)
        # キャップを超えた数の行動を登録しても保持上限を守る
        for i in range(BELLMAN_QTABLE_PER_STATE_CAP + 10):
            qt.update("s", f"A:{i:03d}", reward=0.1, next_state_sig="s2", terminal=True)
        bucket = qt._load("s")
        assert len(bucket) <= BELLMAN_QTABLE_PER_STATE_CAP


# ----------------------------------------------------------------------
# BellmanPlanner 統合
# ----------------------------------------------------------------------
class _StubPlanner:
    """next_step を制御できるテスト用プランナー。"""
    def __init__(self, action: str | None, role: str = "worker", llm=None) -> None:
        self.action = action
        self.role = role
        self.llm = llm

    def next_step(self, state, repo_root=None):
        return self.action


class TestBellmanPlannerIntegration:
    def test_passes_through_when_only_one_candidate(self):
        # working_memory に LTM 候補がない場合、planner の出力をそのまま返す
        planner = _StubPlanner("READ: README.md")
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        bp = BellmanPlanner(planner=planner, evaluator=ev, qtable=None)
        s = _make_state()
        out = bp.next_step(s)
        assert out == "READ: README.md"

    def test_picks_higher_q_alternative(self):
        planner = _StubPlanner("CMD: rm -rf /")  # 倫理ブロックでスコア最低
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        bp = BellmanPlanner(planner=planner, evaluator=ev, qtable=None)
        s = _make_state(goal="READMEを確認する")
        s.working_memory["ltm_strategies"] = [
            {"strategy": "READ: README.md", "outcome": "success"},
        ]
        out = bp.next_step(s)
        assert out == "READ: README.md"
        scores = s.working_memory.get("bellman_last_scores")
        assert scores and len(scores) >= 2

    def test_records_observation_when_overriding(self):
        planner = _StubPlanner("CMD: dangerous_thing")
        # dangerous_thing 自体は ValueSystem の違反パターンに無いので倫理は通る。
        # ここでは LTM 候補のほうが goal トークン重複で q_model が高くなる事を確認。
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        bp = BellmanPlanner(planner=planner, evaluator=ev, qtable=None)
        s = _make_state(goal="READMEを確認する")
        s.working_memory["ltm_strategies"] = [
            {"strategy": "READ: README.md", "outcome": "success"},
        ]
        bp.next_step(s)
        # 上書きが起きれば observation に Bellman ログが残る
        msgs = " ".join(s.observations)
        assert "[Bellman]" in msgs

    def test_update_after_step_persists_q(self):
        with tempfile.TemporaryDirectory() as td:
            ltm = LongTermMemory(db_path=Path(td) / "ltm.sqlite")
            planner = _StubPlanner("READ: README.md")
            ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
            qt = QTable(ltm=ltm)
            bp = BellmanPlanner(planner=planner, evaluator=ev, qtable=qt)

            s = _make_state(goal="READMEを確認する")
            s.working_memory["ltm_strategies"] = [
                {"strategy": "READ: docs.md", "outcome": "success"},
            ]
            picked = bp.next_step(s)
            # 状態を進めてから更新
            s.completed_steps.append(picked)
            s.last_status = "success"
            res = bp.update_after_step(
                state=s, action=picked, success=True, terminal=False
            )
            assert res is not None
            new_q, n = res
            assert n == 1
            assert new_q != 0.0

    def test_update_no_op_without_qtable(self):
        planner = _StubPlanner("READ: README.md")
        ev = BellmanEvaluator(value_system=ValueSystem(), predictor=None)
        bp = BellmanPlanner(planner=planner, evaluator=ev, qtable=None)
        s = _make_state()
        bp.next_step(s)
        assert bp.update_after_step(state=s, action="READ: README.md", success=True) is None


# ----------------------------------------------------------------------
# HermesAgentV10 への統合 (use_bellman フラグ)
# ----------------------------------------------------------------------
class TestAgentRunnerIntegration:
    def test_use_bellman_flag_wires_up(self, tmp_path):
        from hermes_agi_gen.agent_runner import HermesAgentV10
        from hermes_agi_gen.state_store import SessionDB

        db = SessionDB(db_path=tmp_path / "session.sqlite") if hasattr(SessionDB, "__init__") else None
        ltm = LongTermMemory(db_path=tmp_path / "ltm.sqlite")
        agent = HermesAgentV10(
            repo_root=tmp_path,
            ltm=ltm,
            session_db=db,
            use_bellman=True,
        )
        assert agent.bellman_planner is not None
        assert agent.use_bellman is True

    def test_default_does_not_enable_bellman(self, tmp_path):
        from hermes_agi_gen.agent_runner import HermesAgentV10

        ltm = LongTermMemory(db_path=tmp_path / "ltm.sqlite")
        agent = HermesAgentV10(repo_root=tmp_path, ltm=ltm)
        assert agent.bellman_planner is None
        assert agent.use_bellman is False
