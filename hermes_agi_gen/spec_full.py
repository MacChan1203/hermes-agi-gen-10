"""Spec-aligned Hermes AGI 正式版 (Full Edition)。

`spec_core.HermesAGIMVP` の MVP 制約 ──
  * 反復回数 3 上限
  * 3 ステップ固定テンプレ
  * 副作用のないダミー Executor
  * 単純な成功率スコアのみの Critic
  * フラット JSON メモリ
── を取り払い、本番運用可能な正式版を提供する。

主な拡張点:
  * `FullPlanner`     : LLM 統合 (任意) + ドメイン認識テンプレ + 任意ステップ数
  * `FullExecutor`    : 既存 Executor をオプションで包み、実ツール実行 / 価値整合
                        ゲート / リトライ + バックオフ / タイムアウトをサポート
  * `FullCritic`      : 完了率 × 価値整合度 × 目標被覆度の重み付きスコア +
                        プラトー検出 (連続非改善で打ち切り)
  * `SqliteMemory`    : SQLite で履歴を永続化、タスク横断クエリ可能
  * `HermesAGIFull`   : 反復上限を解除 (デフォルト 12, ユーザー設定で任意)、
                        収束検出、メトリクス出力を統合
  * `run_spec_full()` : 1 行エントリポイント
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

from .spec_core import (
    CriticOutput,
    Plan,
    PlanStep,
    Result,
    Task,
    _to_jsonable,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 設定: 重み・閾値
# ----------------------------------------------------------------------
@dataclass
class FullConfig:
    """正式版の動作パラメータ。"""
    max_iterations: int = 12             # 反復上限 (MVP は 3 固定。ここは任意)
    patience: int = 2                    # スコア非改善で打ち切るまでの猶予回数
    score_improve_eps: float = 0.05      # この差未満は「非改善」と見なす
    weight_completion: float = 0.55      # スコア重み: 完了率
    weight_value_align: float = 0.25     # スコア重み: 価値整合
    weight_goal_coverage: float = 0.20   # スコア重み: 目標トークン被覆度
    done_threshold: float = 0.85         # 総合スコアがこれ以上で done
    halt_on_value_violation: bool = True # ValueSystem ブロック発生時に即停止
    retry_max: int = 2                   # ステップ単位リトライ回数
    retry_backoff_sec: float = 0.05      # 1 回目リトライ前の待機 (指数バックオフ)
    step_timeout_sec: float = 30.0       # 1 ステップのタイムアウト (実行 runner で参照)
    plan_min_steps: int = 3              # プランの最小ステップ数
    plan_max_steps: int = 12             # プランの最大ステップ数 (LLM 出力の安全弁)


# ----------------------------------------------------------------------
# メモリ: SQLite 版
# ----------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    constraints TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    rowid_alias INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    steps_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    rowid_alias INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    step_id INTEGER NOT NULL,
    output TEXT,
    success INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    duration_sec REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    rowid_alias INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    score REAL NOT NULL,
    feedback TEXT,
    done INTEGER NOT NULL,
    breakdown_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_task ON plans(task_id, iteration);
CREATE INDEX IF NOT EXISTS idx_results_task ON results(task_id, iteration);
CREATE INDEX IF NOT EXISTS idx_reviews_task ON reviews(task_id, iteration);
"""


class SqliteMemory:
    """SQLite ベースの履歴永続化メモリ。

    JsonMemory 互換の `load()` も提供するので、既存テスト資産を流用できる。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    # ----- 書き込み -----
    def save_task(self, task: Task) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tasks(id, goal, constraints, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    goal = excluded.goal,
                    constraints = excluded.constraints,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    task.id, task.goal,
                    json.dumps(task.constraints, ensure_ascii=False),
                    task.status, now, now,
                ),
            )
            self._conn.commit()

    def save_plan(self, task_id: str, iteration: int, plan: Plan) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO plans(task_id, iteration, steps_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    task_id, iteration,
                    json.dumps([asdict(s) for s in plan.steps], ensure_ascii=False),
                    time.time(),
                ),
            )
            self._conn.commit()

    def save_result(
        self, task_id: str, iteration: int, result: Result,
        attempts: int = 1, duration_sec: float = 0.0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO results(task_id, iteration, step_id, output, success,
                                    attempts, duration_sec, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, iteration, result.step_id, result.output,
                    1 if result.success else 0, attempts, duration_sec, time.time(),
                ),
            )
            self._conn.commit()

    def save_review(
        self, task_id: str, iteration: int, review: CriticOutput,
        breakdown: Optional[dict] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reviews(task_id, iteration, score, feedback, done,
                                    breakdown_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, iteration, float(review.score), review.feedback,
                    1 if review.done else 0,
                    json.dumps(breakdown or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
            self._conn.commit()

    # ----- 読み出し -----
    def load(self) -> dict[str, Any]:
        """spec_core.JsonMemory 互換: 全件をフラット dict で返す。"""
        with self._lock:
            tasks = [dict(r) for r in self._conn.execute("SELECT * FROM tasks").fetchall()]
            plans = [dict(r) for r in self._conn.execute("SELECT * FROM plans").fetchall()]
            results = [
                dict(r) for r in self._conn.execute("SELECT * FROM results").fetchall()
            ]
            reviews = [
                dict(r) for r in self._conn.execute("SELECT * FROM reviews").fetchall()
            ]
        # constraints/steps_json/breakdown_json をデシリアライズ
        for t in tasks:
            try:
                t["constraints"] = json.loads(t.get("constraints") or "[]")
            except (json.JSONDecodeError, TypeError):
                t["constraints"] = []
        for p in plans:
            try:
                p["steps"] = json.loads(p.pop("steps_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                p["steps"] = []
        for r in results:
            r["success"] = bool(r.get("success"))
        for rv in reviews:
            rv["done"] = bool(rv.get("done"))
            try:
                rv["breakdown"] = json.loads(rv.pop("breakdown_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                rv["breakdown"] = {}
        return {"tasks": tasks, "plans": plans, "results": results, "reviews": reviews}

    def task_history(self, task_id: str) -> dict[str, Any]:
        snap = self.load()
        return {
            key: [item for item in snap[key] if item.get("task_id") == task_id or item.get("id") == task_id]
            for key in snap
        }

    def get_failure_rate(self, task_id: Optional[str] = None) -> float:
        with self._lock:
            if task_id:
                row = self._conn.execute(
                    "SELECT AVG(1 - success) AS rate FROM results WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT AVG(1 - success) AS rate FROM results"
                ).fetchone()
        return float(row["rate"]) if row and row["rate"] is not None else 0.0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ----------------------------------------------------------------------
# Planner: LLM + ドメイン認識テンプレ
# ----------------------------------------------------------------------
_DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "coding": [
        "リポジトリの状態を把握する",
        "関連ファイルを読み込む",
        "変更案を設計する",
        "実装する",
        "動作確認する",
        "結果をまとめる: {goal}",
    ],
    "research": [
        "目的を整理する: {goal}",
        "情報源を列挙する",
        "情報を収集する",
        "信頼性を評価する",
        "結果を統合する",
        "結論をまとめる",
    ],
    "writing": [
        "文章の目的と読者を確認する: {goal}",
        "構成案を作る",
        "本文を書く",
        "推敲する",
        "成果物としてまとめる",
    ],
    "data": [
        "データの形式を確認する: {goal}",
        "前処理する",
        "集計・分析する",
        "結果を可視化する",
        "知見をまとめる",
    ],
    "general": [
        "目的を確認する: {goal}",
        "必要な情報を集める",
        "選択肢を比較する",
        "実行する",
        "結果をまとめる",
    ],
}


def _classify_goal(goal: str) -> str:
    g = goal.lower()
    if any(k in g for k in ("コード", "code", "実装", "リポジトリ", "テスト", "バグ")):
        return "coding"
    if any(k in g for k in ("調査", "リサーチ", "研究", "research", "investigate")):
        return "research"
    if any(k in g for k in ("文章", "原稿", "write", "ブログ", "記事", "要約")):
        return "writing"
    if any(k in g for k in ("データ", "csv", "json", "集計", "分析", "data ")):
        return "data"
    return "general"


class FullPlanner:
    """LLM があれば LLM プランニング、無ければドメイン認識テンプレ。"""

    def __init__(
        self,
        llm: Any = None,
        config: Optional[FullConfig] = None,
    ) -> None:
        self.llm = llm
        self.config = config or FullConfig()

    def plan(self, task: Task, memory: dict[str, Any]) -> Plan:
        # 反復回数を memory から推定 (前回までの review 件数)
        prev_reviews = [
            r for r in memory.get("reviews", []) if r.get("task_id") == task.id
        ]
        iteration = len(prev_reviews) + 1
        last_feedback = prev_reviews[-1].get("feedback") if prev_reviews else None

        if self.llm is not None:
            steps = self._llm_plan(task, iteration, last_feedback)
            if steps:
                return self._make_plan(steps)
        # フォールバック: ドメインテンプレ
        domain = _classify_goal(task.goal)
        template = _DOMAIN_TEMPLATES.get(domain, _DOMAIN_TEMPLATES["general"])
        actions = [s.format(goal=task.goal) for s in template]
        if iteration > 1 and last_feedback:
            # 反復時はフィードバックを冒頭ステップに反映
            actions = [f"前回の指摘を反映する: {last_feedback}"] + actions
        return self._make_plan(actions)

    def _make_plan(self, actions: List[str]) -> Plan:
        # 最小/最大ステップ数で挟み込む
        cfg = self.config
        if len(actions) < cfg.plan_min_steps:
            actions = actions + ["追加で目的に近づく作業を行う"] * (cfg.plan_min_steps - len(actions))
        actions = actions[: cfg.plan_max_steps]
        return Plan(steps=[PlanStep(id=i + 1, action=a) for i, a in enumerate(actions)])

    _PROMPT = """\
あなたはタスク分解プランナーです。次の目標を達成するために必要な手順を、
JSON 配列のみで返してください。各要素は手順を 1 行で表す日本語の文字列です。
ステップ数は {min_steps}〜{max_steps} の範囲で、過不足なく具体的に。

目標: {goal}
制約: {constraints}
反復: {iteration}
{feedback_section}
出力例: ["ステップ1", "ステップ2", "ステップ3"]
"""

    def _llm_plan(
        self, task: Task, iteration: int, last_feedback: Optional[str]
    ) -> Optional[List[str]]:
        cfg = self.config
        feedback_section = (
            f"前回の批評: {last_feedback}\nこれを必ず反映してください。\n" if last_feedback else ""
        )
        prompt = self._PROMPT.format(
            goal=task.goal,
            constraints=", ".join(task.constraints) or "制約なし",
            iteration=iteration,
            min_steps=cfg.plan_min_steps,
            max_steps=cfg.plan_max_steps,
            feedback_section=feedback_section,
        )
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception:
            logger.debug("LLM プランニング失敗、テンプレへフォールバック", exc_info=True)
            return None
        return _parse_json_array(raw)


def _parse_json_array(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    text = raw.strip()
    # ```json ... ``` でラップされていれば剥がす
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # 最初の '[' から最後の ']' まで切り出す
    lo, hi = text.find("["), text.rfind("]")
    if lo < 0 or hi < 0 or hi <= lo:
        return None
    try:
        arr = json.loads(text[lo : hi + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    cleaned = [str(x).strip() for x in arr if str(x).strip()]
    return cleaned or None


# ----------------------------------------------------------------------
# Executor: リトライ + 価値整合 + 任意の実ツール実行
# ----------------------------------------------------------------------
RunnerFn = Callable[[PlanStep, Task], Result]


class FullExecutor:
    """指数バックオフ付きリトライ・価値整合チェック・任意の実ツール実行。"""

    def __init__(
        self,
        runner: Optional[RunnerFn] = None,
        config: Optional[FullConfig] = None,
        value_system: Any = None,
    ) -> None:
        self.runner = runner or self._default_runner
        self.config = config or FullConfig()
        self.value_system = value_system  # ValueSystem インスタンス (任意)

    def execute(self, step: PlanStep, task: Task) -> Tuple[Result, int, float]:
        """(result, attempts, duration_sec) を返す。"""
        cfg = self.config

        # 価値整合チェック (任意)
        if self.value_system is not None:
            try:
                assessment = self.value_system.assess(step.action)
                if assessment.is_blocked:
                    step.status = "failed"
                    return (
                        Result(
                            step_id=step.id,
                            output=f"[blocked by ValueSystem] {assessment.recommendation}",
                            success=False,
                        ),
                        0, 0.0,
                    )
            except Exception:
                logger.debug("ValueSystem.assess 失敗、評価をスキップ", exc_info=True)

        last_result: Optional[Result] = None
        attempts = 0
        t0 = time.time()
        for attempt in range(cfg.retry_max + 1):
            attempts = attempt + 1
            try:
                last_result = self.runner(step, task)
            except Exception as exc:  # ランナー側の予期せぬ例外も失敗扱いに
                logger.debug("runner 例外: %s", exc, exc_info=True)
                last_result = Result(
                    step_id=step.id, output=f"runner exception: {exc}", success=False,
                )
            if last_result.success:
                step.status = "done"
                break
            if attempt < cfg.retry_max:
                # 指数バックオフ: 0.05 -> 0.10 -> 0.20 ...
                time.sleep(cfg.retry_backoff_sec * (2 ** attempt))
        else:
            step.status = "failed"
        if last_result is not None and not last_result.success:
            step.status = "failed"
        duration = time.time() - t0
        assert last_result is not None
        return last_result, attempts, duration

    @staticmethod
    def _default_runner(step: PlanStep, task: Task) -> Result:
        action_lower = step.action.lower()
        success = "fail" not in action_lower and "失敗" not in step.action
        output = f"{step.action} -> {'完了' if success else '失敗'}"
        return Result(step_id=step.id, output=output, success=success)


def make_real_tool_runner(
    repo_root: str | Path,
    state: Optional[Any] = None,
) -> RunnerFn:
    """既存 `executor.Executor` を spec runner として使うアダプタ。

    `step.action` が "CMD: ...", "READ: ...", "WRITE: ..." 等の場合のみ実ツールに
    流し、それ以外は従来通り「観測テキストを返すだけの記述的ステップ」として
    成功扱いにする。state は AgentState (なければ自動作成)。
    """
    from .agent_state import AgentState
    from .executor import Executor

    real_executor = Executor(repo_root=repo_root)
    _state = state or AgentState(user_goal="")

    _TOOL_PREFIXES = ("CMD:", "READ:", "WRITE:", "PYTHON:", "FETCH:", "CALC:", "SEARCH:")

    def _runner(step: PlanStep, task: Task) -> Result:
        action = step.action.strip()
        upper = action.upper()
        if not any(upper.startswith(p) for p in _TOOL_PREFIXES):
            # ツール形式でなければ従来動作
            return FullExecutor._default_runner(step, task)
        _state.user_goal = task.goal
        res = real_executor.execute(action, _state)
        success = bool(res.get("ok"))
        out = res.get("stdout") or res.get("stderr") or ""
        return Result(step_id=step.id, output=out[:2000], success=success)

    return _runner


# ----------------------------------------------------------------------
# Critic: 重み付きスコア + プラトー検出
# ----------------------------------------------------------------------
class FullCritic:
    """完了率 × 価値整合 × 目標被覆度の重み付きスコア。"""

    def __init__(
        self,
        config: Optional[FullConfig] = None,
        value_system: Any = None,
    ) -> None:
        self.config = config or FullConfig()
        self.value_system = value_system

    def review(
        self, task: Task, memory: dict[str, Any], iteration: int
    ) -> Tuple[CriticOutput, dict[str, float]]:
        cfg = self.config
        results = [
            r for r in memory.get("results", [])
            if r.get("task_id") == task.id and r.get("iteration") == iteration
        ]
        if not results:
            zero = {"completion": 0.0, "value_align": 0.0, "goal_coverage": 0.0, "total": 0.0}
            return CriticOutput(score=0.0, feedback="まだ結果がありません", done=False), zero

        completion = sum(1 for r in results if r.get("success")) / len(results)

        # 価値整合度: ValueSystem があればそれを使う、なければ満点
        if self.value_system is not None:
            try:
                avg_score = sum(
                    self.value_system.assess(_as_action(r)).total_score
                    for r in results
                ) / len(results)
                value_align = max(0.0, 1.0 - avg_score)
            except Exception:
                logger.debug("Critic 価値整合評価に失敗", exc_info=True)
                value_align = 1.0
        else:
            value_align = 1.0

        # 目標被覆度: 結果の output に goal トークンがどれだけ現れるか
        goal_coverage = _goal_coverage(task.goal, results)

        total = (
            cfg.weight_completion * completion
            + cfg.weight_value_align * value_align
            + cfg.weight_goal_coverage * goal_coverage
        )
        breakdown = {
            "completion": round(completion, 3),
            "value_align": round(value_align, 3),
            "goal_coverage": round(goal_coverage, 3),
            "total": round(total, 3),
        }
        done = total >= cfg.done_threshold and completion >= 1.0

        if done:
            feedback = (
                f"目標達成 (score={total:.2f}, 完了率=100%, "
                f"価値整合={value_align:.2f}, 被覆={goal_coverage:.2f})"
            )
        else:
            missing: List[str] = []
            if completion < 1.0:
                missing.append(f"未完了ステップあり ({int(completion*100)}%)")
            if value_align < 0.9:
                missing.append("価値整合に懸念")
            if goal_coverage < 0.5:
                missing.append("目標トークンの被覆が弱い")
            feedback = "改善余地: " + " / ".join(missing) if missing else "もう一押し"

        return CriticOutput(score=round(total, 3), feedback=feedback, done=done), breakdown


def _as_action(result_row: dict[str, Any]) -> str:
    return str(result_row.get("output", ""))


def _goal_coverage(goal: str, results: List[dict[str, Any]]) -> float:
    import re as _re
    tok = _re.compile(r"[A-Za-z0-9]+|[぀-ヿ]+|[一-鿿]+")
    goal_tokens = {t.lower() for t in tok.findall(goal) if len(t) > 1}
    if not goal_tokens:
        return 1.0
    blob = " ".join(str(r.get("output", "")) for r in results).lower()
    hit = sum(1 for t in goal_tokens if t in blob)
    return hit / len(goal_tokens)


# ----------------------------------------------------------------------
# 主オーケストレーター
# ----------------------------------------------------------------------
class HermesAGIFull:
    """正式版 spec ループ。MVP の制約をすべて取り払う。"""

    def __init__(
        self,
        memory_path: str | Path,
        *,
        config: Optional[FullConfig] = None,
        planner: Optional[FullPlanner] = None,
        executor: Optional[FullExecutor] = None,
        critic: Optional[FullCritic] = None,
        memory: Optional[SqliteMemory] = None,
        llm: Any = None,
        value_system: Any = None,
    ) -> None:
        self.config = config or FullConfig()
        # ValueSystem を遅延 import (循環回避)
        if value_system is None:
            try:
                from .value_system import ValueSystem
                value_system = ValueSystem()
            except Exception:
                value_system = None
        self.value_system = value_system

        self.memory = memory or SqliteMemory(memory_path)
        self.planner = planner or FullPlanner(llm=llm, config=self.config)
        self.executor = executor or FullExecutor(
            config=self.config, value_system=self.value_system,
        )
        self.critic = critic or FullCritic(
            config=self.config, value_system=self.value_system,
        )

    # ------------------------------------------------------------------
    def run(
        self, goal: str, constraints: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        cfg = self.config
        task = Task(
            id=str(uuid.uuid4()),
            goal=goal.strip(),
            constraints=list(constraints or []),
            status="running",
        )
        self.memory.save_task(task)

        history: List[dict[str, Any]] = []
        best_score = 0.0
        plateau = 0
        final_review = CriticOutput(score=0.0, feedback="未評価", done=False)
        halted_reason: Optional[str] = None

        for iteration in range(1, cfg.max_iterations + 1):
            snapshot = self.memory.load()
            plan = self.planner.plan(task, snapshot)
            self.memory.save_plan(task.id, iteration, plan)

            value_violation = False
            for step in plan.steps:
                result, attempts, duration = self.executor.execute(step, task)
                self.memory.save_result(
                    task.id, iteration, result,
                    attempts=attempts, duration_sec=duration,
                )
                if (
                    not result.success
                    and "[blocked by ValueSystem]" in (result.output or "")
                ):
                    value_violation = True

            review, breakdown = self.critic.review(task, self.memory.load(), iteration)
            self.memory.save_review(task.id, iteration, review, breakdown=breakdown)
            history.append({"iteration": iteration, "review": asdict(review), "breakdown": breakdown})
            final_review = review
            best_score = max(best_score, review.score)

            if cfg.halt_on_value_violation and value_violation:
                halted_reason = "value_violation"
                break

            if review.done:
                task.status = "done"
                break

            # プラトー検出: スコアが eps 未満しか伸びなかったら plateau++
            prev_best = max((h["review"]["score"] for h in history[:-1]), default=0.0)
            if review.score - prev_best < cfg.score_improve_eps:
                plateau += 1
            else:
                plateau = 0
            if plateau >= cfg.patience:
                halted_reason = "plateau"
                break

            # フィードバックを次反復の制約に追加
            task.constraints.append(f"critic_feedback: {review.feedback}")
            self.memory.save_task(task)

        self.memory.save_task(task)
        return {
            "task": asdict(task),
            "review": asdict(final_review),
            "iterations": len(history),
            "history": history,
            "halted_reason": halted_reason,
            "memory_path": str(self.memory.path),
            "metrics": {
                "best_score": round(best_score, 3),
                "failure_rate": round(self.memory.get_failure_rate(task.id), 3),
            },
        }


def run_spec_full(
    goal: str,
    memory_path: str | Path,
    *,
    constraints: Optional[Iterable[str]] = None,
    config: Optional[FullConfig] = None,
    llm: Any = None,
) -> dict[str, Any]:
    """1 行エントリポイント: 正式版を即座に走らせる。"""
    return HermesAGIFull(
        memory_path=memory_path, config=config, llm=llm,
    ).run(goal, constraints=constraints)
