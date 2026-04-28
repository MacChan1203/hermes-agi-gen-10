"""Bellman 最適方程式に基づく行動選択レイヤー。

V*(s) = max_a [ r(s,a) + γ * Σ_{s'} P(s'|s,a) V*(s') ]
Q*(s,a) = r(s,a) + γ * max_{a'} Q*(s', a')

このモジュールは hermes-agi-gen-10 の MDP 化を行う:
- 状態 s : ドメイン + 直近の行動種別履歴 + last_status からハッシュ
- 行動 a : Planner / LTM / Reviewer.recovery_action から得た候補プール
- 報酬 r : ValueSystem.utility_score(a) + 目標関連度 + DONE 終端ボーナス
- 遷移 P : PredictiveEngine.predict(a).success_probability で近似

二段階導入:
  Phase A (BellmanEvaluator): r + γ V_model(s') によるモデルベース評価。
                              既存ループに最小侵襲で planner の選択を補強。
  Phase B (QTable): TD 学習による表形式 Q を LongTermMemory に永続化。
                    経験が貯まるほど β で QTable を信頼し、Bellman 最適方程式に
                    収束させる。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .config import (
    BELLMAN_ALPHA,
    BELLMAN_CANDIDATE_K,
    BELLMAN_DEFAULT_GOAL_RELEVANCE,
    BELLMAN_GAMMA,
    BELLMAN_GOAL_PROGRESS_BONUS,
    BELLMAN_LTM_KEY_PREFIX,
    BELLMAN_QTABLE_BLEND_BETA_INIT,
    BELLMAN_QTABLE_BLEND_BETA_MAX,
    BELLMAN_QTABLE_PER_STATE_CAP,
    BELLMAN_QTABLE_VISITS_FOR_FULL_TRUST,
)

if TYPE_CHECKING:
    from .agent_state import AgentState
    from .long_term_memory import LongTermMemory
    from .planner import Planner
    from .predictive_engine import PredictiveEngine
    from .value_system import ValueSystem

logger = logging.getLogger(__name__)


# 英数字と CJK / かな は別トークンとして扱う (連結を防ぐ)。
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[぀-ヿ]+|[一-鿿]+")
_ACTION_PREFIXES = (
    "ANSWER", "SEARCH", "FETCH", "CALC", "CMD", "READ",
    "WRITE", "PYTHON", "PLAN", "SCHEDULE_AT", "SCHEDULE", "DONE",
)


def _action_type(action: str) -> str:
    head = action.strip().split(":", 1)[0].upper()
    return head if head in _ACTION_PREFIXES else "UNKNOWN"


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def goal_relevance(action: str, goal: str) -> float:
    """目標と行動のトークン重複から関連度を 0..1 で見積もる。"""
    a, g = _tokens(action), _tokens(goal)
    if not a or not g:
        return BELLMAN_DEFAULT_GOAL_RELEVANCE
    overlap = len(a & g) / max(1, len(g))
    return max(BELLMAN_DEFAULT_GOAL_RELEVANCE, min(1.0, 0.3 + overlap))


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:length]


def state_signature(state: "AgentState") -> str:
    """AgentState から離散状態シグネチャを作る。"""
    domain = getattr(state, "domain", "general") or "general"
    last_types = [_action_type(s) for s in state.completed_steps[-3:]]
    fail_types = [_action_type(s) for s in state.failed_steps[-2:]]
    last_status = getattr(state, "last_status", "") or ""
    plan_depth = len(state.current_plan or [])
    raw = f"{domain}|c={','.join(last_types)}|f={','.join(fail_types)}|s={last_status}|p={plan_depth}"
    return _short_hash(raw, 16)


def action_signature(action: str) -> str:
    """行動を行動種別 + 先頭 40 文字でハッシュ化する。"""
    norm = re.sub(r"\s+", " ", action.strip())[:120]
    head = norm.split(":", 1)
    prefix = head[0].upper() if len(head) == 2 else "UNKNOWN"
    body = head[1].strip() if len(head) == 2 else norm
    return f"{prefix}:{_short_hash(body[:40], 10)}"


@dataclass
class CandidateScore:
    action: str
    q_model: float       # r + γ V_model(s')
    q_table: float       # 表 Q (なければ q_model と同値)
    blended: float       # 混合後の最終スコア
    visits: int          # この状態-行動の訪問回数
    reward: float        # 即時報酬 r
    v_next: float        # 推定 V(s')
    source: str          # 候補の出所


# ----------------------------------------------------------------------
# Phase A: モデルベース評価
# ----------------------------------------------------------------------
class BellmanEvaluator:
    """Bellman 方程式の右辺をモデルから評価する。

    Q_model(s, a) = r(s, a) + γ * V_model(s')
    V_model(s') ≈ predictor.predict(a).success_probability
                  * goal_relevance(a) + 終端ボーナス(DONE: なら有効)
    """

    def __init__(
        self,
        value_system: "ValueSystem",
        predictor: Optional["PredictiveEngine"] = None,
        gamma: float = BELLMAN_GAMMA,
    ) -> None:
        self.value_system = value_system
        self.predictor = predictor
        self.gamma = gamma

    def reward(self, action: str, goal: str) -> float:
        rel = goal_relevance(action, goal)
        # ValueSystem.utility_score: ブロック時 0.0, 違反なし時 ≈ rel
        util = self.value_system.utility_score(action, goal_relevance=rel)
        a_type = _action_type(action)
        if a_type == "DONE":
            return util + BELLMAN_GOAL_PROGRESS_BONUS
        if a_type in ("ANSWER", "WRITE"):
            return util + BELLMAN_GOAL_PROGRESS_BONUS * 0.5
        return util

    def estimate_v_next(self, action: str, goal: str) -> float:
        """V(s') の推定値。DONE: は終端なので 0。"""
        a_type = _action_type(action)
        if a_type == "DONE":
            return 0.0
        if self.predictor is None:
            return BELLMAN_DEFAULT_GOAL_RELEVANCE
        try:
            pred = self.predictor.predict(action=action, goal=goal)
        except Exception:
            logger.debug("predict() 失敗、デフォルトで継続", exc_info=True)
            return BELLMAN_DEFAULT_GOAL_RELEVANCE
        # P(成功) を遷移期待値の代理として使う。
        # 将来の価値は「成功時に goal_relevance に到達する」と近似。
        rel = goal_relevance(action, goal)
        return float(pred.success_probability) * rel

    def q_value(self, action: str, goal: str) -> Tuple[float, float, float]:
        """Q_model(s,a) と (r, V(s')) を返す。"""
        r = self.reward(action, goal)
        v_next = self.estimate_v_next(action, goal)
        q = r + self.gamma * v_next
        return q, r, v_next


# ----------------------------------------------------------------------
# Phase B: 表形式 Q-learning (LTM 永続化)
# ----------------------------------------------------------------------
class QTable:
    """状態シグネチャごとに {action_signature: {q, n}} を持つ表形式 Q。

    LongTermMemory に 1 状態 = 1 レコード (JSON dict) で永続化する。
    LTM が無い場合はインメモリのみ。
    """

    def __init__(
        self,
        ltm: Optional["LongTermMemory"] = None,
        alpha: float = BELLMAN_ALPHA,
        gamma: float = BELLMAN_GAMMA,
    ) -> None:
        self.ltm = ltm
        self.alpha = alpha
        self.gamma = gamma
        self._cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _ltm_key(self, state_sig: str) -> str:
        return f"{BELLMAN_LTM_KEY_PREFIX}{state_sig}"

    def _load(self, state_sig: str) -> Dict[str, Dict[str, float]]:
        if state_sig in self._cache:
            return self._cache[state_sig]
        bucket: Dict[str, Dict[str, float]] = {}
        if self.ltm is not None:
            try:
                raw = self.ltm.recall(self._ltm_key(state_sig))
            except Exception:
                logger.debug("QTable load 失敗", exc_info=True)
                raw = None
            if raw:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, dict) and "q" in v and "n" in v:
                                bucket[str(k)] = {
                                    "q": float(v["q"]),
                                    "n": float(v["n"]),
                                    "ts": float(v.get("ts", 0.0)),
                                }
                except (json.JSONDecodeError, ValueError, TypeError):
                    logger.debug("QTable JSON 解析失敗 state=%s", state_sig)
        self._cache[state_sig] = bucket
        return bucket

    def _trim(self, state_sig: str) -> None:
        bucket = self._cache.get(state_sig)
        if not bucket or len(bucket) <= BELLMAN_QTABLE_PER_STATE_CAP:
            return
        ranked = sorted(
            bucket.items(),
            key=lambda kv: (kv[1].get("n", 0.0), kv[1].get("ts", 0.0)),
            reverse=True,
        )
        self._cache[state_sig] = dict(ranked[:BELLMAN_QTABLE_PER_STATE_CAP])

    def _persist(self, state_sig: str, session_id: Optional[str] = None) -> None:
        self._trim(state_sig)
        if self.ltm is None:
            return
        bucket = self._cache.get(state_sig)
        if bucket is None:
            return
        try:
            self.ltm.learn(
                self._ltm_key(state_sig),
                json.dumps(bucket, ensure_ascii=False),
                session_id=session_id,
            )
        except Exception:
            logger.debug("QTable persist 失敗", exc_info=True)

    def get(self, state_sig: str, action_sig: str) -> Tuple[float, int]:
        bucket = self._load(state_sig)
        entry = bucket.get(action_sig)
        if not entry:
            return 0.0, 0
        return float(entry["q"]), int(entry["n"])

    def best_q(self, state_sig: str) -> float:
        bucket = self._load(state_sig)
        if not bucket:
            return 0.0
        return max(float(e["q"]) for e in bucket.values())

    def update(
        self,
        state_sig: str,
        action_sig: str,
        reward: float,
        next_state_sig: str,
        terminal: bool,
        session_id: Optional[str] = None,
    ) -> Tuple[float, int]:
        """TD 更新: Q ← Q + α[ r + γ max_a' Q(s', a') (1-terminal) − Q ]"""
        bucket = self._load(state_sig)
        entry = bucket.setdefault(action_sig, {"q": 0.0, "n": 0.0, "ts": 0.0})
        target_future = 0.0 if terminal else self.best_q(next_state_sig)
        td_target = reward + self.gamma * target_future
        old_q = float(entry["q"])
        new_q = old_q + self.alpha * (td_target - old_q)
        entry["q"] = new_q
        entry["n"] = float(entry["n"]) + 1.0
        import time as _time
        entry["ts"] = _time.time()
        self._trim(state_sig)
        self._persist(state_sig, session_id=session_id)
        return new_q, int(entry["n"])


# ----------------------------------------------------------------------
# Phase A + B 統合プランナー
# ----------------------------------------------------------------------
class BellmanPlanner:
    """Bellman 最適方程式に従って候補行動を評価・選択する高位プランナー。

    既存の Planner.next_step() の出力を「デフォルト候補」として扱い、
    LTM の成功戦略・Reviewer.recovery_action・Q-table 上位行動などを
    追加候補としてプールする。Q_total = (1-β)*Q_model + β*Q_table で
    最良を選び、実行後に TD 更新で永続学習を行う。
    """

    def __init__(
        self,
        planner: "Planner",
        evaluator: BellmanEvaluator,
        qtable: Optional[QTable] = None,
        candidate_k: int = BELLMAN_CANDIDATE_K,
    ) -> None:
        self.planner = planner
        self.evaluator = evaluator
        self.qtable = qtable
        self.candidate_k = candidate_k
        # 直前のステップ (TD 更新で next_state が必要)
        self._last_state_sig: Optional[str] = None
        self._last_action_sig: Optional[str] = None

    @property
    def role(self) -> str:
        return self.planner.role

    @property
    def llm(self) -> Any:
        return self.planner.llm

    # ------------------------------------------------------------------
    # 候補プール生成
    # ------------------------------------------------------------------
    def _gather_candidates(
        self,
        state: "AgentState",
        primary: Optional[str],
    ) -> List[Tuple[str, str]]:
        """[(action, source), ...] を返す。重複は除去、最大 candidate_k 件。"""
        seen: set[str] = set()
        out: List[Tuple[str, str]] = []

        def push(action: Optional[str], source: str) -> None:
            if not action:
                return
            key = action.strip()
            if not key or key in seen:
                return
            seen.add(key)
            out.append((key, source))

        push(primary, "planner")

        # LTM の成功戦略から、現在の目標に意味的に近いものを候補化
        ltm_strategies = state.working_memory.get("ltm_strategies", []) or []
        for s in ltm_strategies[:3]:
            if s.get("outcome") == "success":
                push(s.get("strategy"), "ltm_success")

        # Reviewer が直前に提案した recovery_action
        hints = state.working_memory.get("last_improvement_hints", []) or []
        for h in hints[:1]:
            if isinstance(h, str):
                push(h, "reviewer_hint")

        # Q-table の上位行動 (経験のある状態のみ)
        if self.qtable is not None:
            sig = state_signature(state)
            bucket = self.qtable._load(sig)
            if bucket:
                top = sorted(bucket.items(), key=lambda kv: kv[1]["q"], reverse=True)
                for action_sig, _entry in top[:2]:
                    # action_sig はハッシュなので、bucket の登録時に保存した raw を
                    # 持っていない。Phase B 単独では文字列復元できないため、
                    # ここでは ltm_success 由来のものに任せ、未知ハッシュは無視する。
                    _ = action_sig

        return out[: max(1, self.candidate_k)]

    # ------------------------------------------------------------------
    # スコアリング
    # ------------------------------------------------------------------
    def _blend_beta(self, visits: int) -> float:
        if BELLMAN_QTABLE_VISITS_FOR_FULL_TRUST <= 0:
            return BELLMAN_QTABLE_BLEND_BETA_INIT
        ratio = min(1.0, visits / float(BELLMAN_QTABLE_VISITS_FOR_FULL_TRUST))
        return (
            BELLMAN_QTABLE_BLEND_BETA_INIT
            + (BELLMAN_QTABLE_BLEND_BETA_MAX - BELLMAN_QTABLE_BLEND_BETA_INIT) * ratio
        )

    def score_candidates(
        self,
        state: "AgentState",
        candidates: List[Tuple[str, str]],
    ) -> List[CandidateScore]:
        goal = state.user_goal
        sig = state_signature(state)
        scored: List[CandidateScore] = []
        for action, src in candidates:
            q_model, r, v_next = self.evaluator.q_value(action, goal)
            q_table = q_model
            visits = 0
            if self.qtable is not None:
                a_sig = action_signature(action)
                q_t, visits = self.qtable.get(sig, a_sig)
                if visits > 0:
                    q_table = q_t
            beta = self._blend_beta(visits)
            blended = (1.0 - beta) * q_model + beta * q_table
            scored.append(
                CandidateScore(
                    action=action,
                    q_model=q_model,
                    q_table=q_table,
                    blended=blended,
                    visits=visits,
                    reward=r,
                    v_next=v_next,
                    source=src,
                )
            )
        scored.sort(key=lambda c: c.blended, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Public API: 既存 Planner と互換
    # ------------------------------------------------------------------
    def next_step(self, state: "AgentState", repo_root: Any = None) -> Optional[str]:
        primary = self.planner.next_step(state, repo_root)

        # 候補が 1 つ以下しか得られない場面 (PLAN: 展開済みの中盤など) では
        # そのまま渡す。Bellman 評価は複数候補がある時だけ意味を持つ。
        candidates = self._gather_candidates(state, primary)
        if len(candidates) <= 1:
            self._record_choice(state, primary)
            return primary

        scored = self.score_candidates(state, candidates)
        best = scored[0]

        # 観測ログに残してトレーサビリティを確保
        debug_lines = [
            f"[Bellman] cand={len(scored)} pick={best.source} "
            f"Q={best.blended:.3f} (r={best.reward:.2f}, V'={best.v_next:.2f}, "
            f"n={best.visits})"
        ]
        for c in scored[1:3]:
            debug_lines.append(
                f"  alt[{c.source}] Q={c.blended:.3f} action={c.action[:40]}"
            )
        state.working_memory["bellman_last_scores"] = [
            {
                "action": c.action[:80],
                "q": round(c.blended, 4),
                "q_model": round(c.q_model, 4),
                "q_table": round(c.q_table, 4),
                "visits": c.visits,
                "source": c.source,
            }
            for c in scored
        ]
        if best.action != primary:
            state.observations.append(
                f"[Bellman] planner提案を上書き: {best.source} (Q={best.blended:.2f})"
            )
        logger.debug("\n".join(debug_lines))

        self._record_choice(state, best.action)
        return best.action

    def _record_choice(self, state: "AgentState", action: Optional[str]) -> None:
        if action is None:
            self._last_state_sig = None
            self._last_action_sig = None
            return
        self._last_state_sig = state_signature(state)
        self._last_action_sig = action_signature(action)

    # ------------------------------------------------------------------
    # 学習: 実行結果を受けて Q を更新
    # ------------------------------------------------------------------
    def update_after_step(
        self,
        state: "AgentState",
        action: str,
        success: bool,
        terminal: bool = False,
        session_id: Optional[str] = None,
    ) -> Optional[Tuple[float, int]]:
        """実行直後に呼ぶ。state は execute 後の最新状態であること。"""
        if self.qtable is None:
            return None
        if self._last_state_sig is None or self._last_action_sig is None:
            return None
        # 報酬: success/失敗 + 価値整合の即時 r を加算
        base_r = self.evaluator.reward(action, state.user_goal)
        outcome_r = 1.0 if success else -0.5
        reward = 0.5 * base_r + 0.5 * outcome_r
        next_sig = state_signature(state)
        result = self.qtable.update(
            state_sig=self._last_state_sig,
            action_sig=self._last_action_sig,
            reward=reward,
            next_state_sig=next_sig,
            terminal=terminal,
            session_id=session_id,
        )
        # 次回の next_state とのリンクを切る (1 ステップ更新)
        self._last_state_sig = None
        self._last_action_sig = None
        return result
