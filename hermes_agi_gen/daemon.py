"""Hermes AGI 自律デーモン。

GoalQueueからゴールを継続的に処理し、空のときは好奇心探索を行う。
バックグラウンドプロセスとして24時間稼働するAGIの「魂」。

使い方:
    python3 -m hermes_agi_gen.daemon          # フォアグラウンドで起動
    python3 cli.py --daemon                   # デーモンモードで起動

シグナル:
    SIGTERM / SIGINT → クリーンシャットダウン (GoalQueueをLTMに保存)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mistral_client import MistralClient

from .agent_runner import HermesAgentV9
from .agent_state import AgentState
from .hermes_constants import DOMAIN_CONFIG
from .long_term_memory import LongTermMemory
from .meta_cognition import GoalQueue, MetaCognition, QueuedGoal
from .scheduler import JobScheduler
from .self_improvement import SelfImprovementEngine

logger = logging.getLogger(__name__)

_HERMES_DIR = Path.home() / ".hermes"
_PID_FILE = _HERMES_DIR / "daemon.pid"
_LOG_FILE = _HERMES_DIR / "daemon.log"
_HEARTBEAT_KEY = "daemon_heartbeat"
_BUDGET_KEY = "daemon_daily_budget"

# デフォルト設定
_DEFAULT_IDLE_SECONDS = 300        # キューが空のときのアイドル待機時間
_DEFAULT_MAX_DAILY_GOALS = 50      # 1日あたりの最大ゴール実行数
_DEFAULT_CURIOSITY_THRESHOLD = 0.4 # 好奇心ゴールを実行する最低スコア


class DailyBudgetGuard:
    """1日あたりのAPI使用量を制限する。コスト爆発を防ぐ。"""

    def __init__(self, ltm: LongTermMemory, max_daily: int = _DEFAULT_MAX_DAILY_GOALS) -> None:
        self.ltm = ltm
        self.max_daily = max_daily

    def _today_key(self) -> str:
        from datetime import date
        return f"{_BUDGET_KEY}_{date.today().isoformat()}"

    def get_used(self) -> int:
        key = self._today_key()
        facts = self.ltm.recall_recent(limit=100)
        for f in facts:
            if f["key"] == key:
                try:
                    return int(f["value"])
                except (ValueError, TypeError):
                    return 0
        return 0

    def increment(self) -> int:
        count = self.get_used() + 1
        self.ltm.learn(self._today_key(), str(count))
        return count

    def is_exhausted(self) -> bool:
        return self.get_used() >= self.max_daily

    def remaining(self) -> int:
        return max(0, self.max_daily - self.get_used())


class HermesDaemon:
    """Hermes AGI 自律デーモン。

    GoalQueueのゴールを継続的に処理し、空時は好奇心探索を行う。
    長期記憶を通じてGoalQueueと世界モデルを永続化し、
    再起動しても前回の状態から続行できる。
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        idle_seconds: int = _DEFAULT_IDLE_SECONDS,
        max_daily_goals: int = _DEFAULT_MAX_DAILY_GOALS,
        curiosity_threshold: float = _DEFAULT_CURIOSITY_THRESHOLD,
    ) -> None:
        _HERMES_DIR.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.idle_seconds = idle_seconds
        self.ltm = LongTermMemory()
        self.meta = MetaCognition(llm=llm)
        self.self_improver = SelfImprovementEngine(llm=llm)
        self.budget = DailyBudgetGuard(self.ltm, max_daily_goals)
        self.curiosity_threshold = curiosity_threshold
        self.scheduler = JobScheduler()
        self._stop_event = threading.Event()
        self._recently_completed: list[str] = []  # 重複防止LRU (最大20件)
        self._max_recent = 20

        # ファイルロガーを設定
        self._setup_file_logger()

    def _setup_file_logger(self) -> None:
        """デーモン専用のファイルロガーを設定する。"""
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 起動・停止
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """デーモンのメインループ。SIGTERMまたはSIGINTで停止する。"""
        self._register_signals()
        self._write_pid()
        self._load_state()

        logger.info("=== Hermes AGI デーモン起動 ===")
        logger.info("PIDファイル: %s", _PID_FILE)
        logger.info("ログファイル: %s", _LOG_FILE)
        logger.info("GoalQueue: %d件ロード済み", self.meta.goal_queue.size())

        print(f"[デーモン] 起動しました (PID={os.getpid()})")
        print(f"[デーモン] GoalQueue: {self.meta.goal_queue.size()}件")
        print(f"[デーモン] 本日の残り予算: {self.budget.remaining()}件")
        print("[デーモン] Ctrl+C で停止")

        while not self._stop_event.is_set():
            try:
                self._heartbeat()
                self._run_one_cycle()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("デーモンループでエラー: %s", exc, exc_info=True)
                time.sleep(10)  # エラー後に少し待機

        self._shutdown()

    def _run_one_cycle(self) -> None:
        """1サイクル: ゴールを1つ処理するか、アイドル探索を行う。"""
        # スケジューラーをティックして期限到達ジョブをGoalQueueに追加
        triggered = self.scheduler.tick(self.meta.goal_queue)
        if triggered:
            for job in triggered:
                logger.info("スケジューラー発火 → GoalQueue: [%s] %s", job.id, job.goal[:60])
                print(f"[スケジューラー] ジョブ発火: [{job.id}] {job.goal[:60]}")
            self._save_state()

        if self.budget.is_exhausted():
            logger.info("本日の予算を使い切りました。明日まで待機します。")
            self._stop_event.wait(self.idle_seconds)
            return

        goal = self.meta.goal_queue.pop_best()

        if goal is None:
            # キューが空 → 好奇心探索
            logger.info("GoalQueueが空です。好奇心探索を開始します。")
            self._idle_explore()
            self._stop_event.wait(self.idle_seconds)
            return

        # 最近完了したゴールと重複チェック
        if goal.goal in self._recently_completed:
            logger.info("重複ゴールをスキップ: %s", goal.goal[:60])
            return

        self._process_one_goal(goal)

    def _process_one_goal(self, goal: QueuedGoal) -> None:
        """GoalQueueからポップしたゴールを実行する。"""
        logger.info(
            "ゴール処理開始 [%s / score=%.2f]: %s",
            goal.source, goal.composite_score, goal.goal[:80],
        )
        print(f"\n[デーモン] ゴール実行: {goal.goal[:70]}")

        used = self.budget.increment()
        logger.info("本日 %d/%d ゴール消費", used, self.budget.max_daily)

        try:
            agent = HermesAgentV9(
                repo_root=Path("."),
                model=getattr(self.llm, "model", "daemon"),
                llm=self.llm,
                ltm=self.ltm,
                agent_role=goal.domain or "worker",
            )

            domain = goal.domain or "general"
            cfg = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG["general"])
            state = AgentState(
                user_goal=goal.goal,
                domain=domain,
                success_criteria=cfg["success_criteria"],
                constraints=cfg["constraints"],
                max_iterations=6,  # デーモンは控えめに
            )

            final_state = agent.run(state)

            # 完了記録
            self._recently_completed.append(goal.goal)
            if len(self._recently_completed) > self._max_recent:
                self._recently_completed.pop(0)

            # パフォーマンスを記録
            score = self.meta.performance_score(final_state)
            self.self_improver.record_session_performance(
                session_id=final_state.session_id or "unknown",
                goal=goal.goal,
                domain=domain,
                score=score,
            )

            logger.info(
                "ゴール完了 [score=%.0f%%]: %s",
                score * 100, goal.goal[:60],
            )
            print(f"[デーモン] 完了 (成功率: {score:.0%}): {goal.goal[:60]}")

            # GoalQueueを保存 (次ゴールが追加されている可能性)
            self._save_state()

        except Exception as exc:
            logger.error("ゴール処理エラー: %s → %s", goal.goal[:60], exc)

    def _idle_explore(self) -> None:
        """GoalQueueが空のとき、好奇心駆動のゴールを生成して追加する。"""
        if self.llm is None:
            return

        logger.info("好奇心探索: 新しいゴールを生成します。")

        # ダミーステートで好奇心ゴールを生成
        dummy_state = AgentState(
            user_goal="自律探索",
            is_done=True,
        )

        try:
            self.meta._generate_curiosity_goals(dummy_state, self.ltm)
        except Exception as exc:
            logger.warning("好奇心ゴール生成エラー: %s", exc)
            return

        # 閾値以上のゴールのみキューに残す
        best = self.meta.goal_queue.peek_best()
        if best and best.priority_score >= self.curiosity_threshold:
            logger.info(
                "好奇心ゴールを追加: [score=%.2f] %s",
                best.composite_score, best.goal[:60],
            )
            print(f"[デーモン] 好奇心ゴール: {best.goal[:60]}")
        else:
            # スコアが低いゴールは捨てる
            while self.meta.goal_queue.size() > 0:
                g = self.meta.goal_queue.pop_best()
                if g and g.priority_score < self.curiosity_threshold:
                    logger.debug("低スコアゴールを廃棄: %s", g.goal[:40])
                else:
                    # スコアが十分なら戻す
                    if g:
                        self.meta.goal_queue.add(g)
                    break

    # ------------------------------------------------------------------
    # 状態の永続化
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """GoalQueueと学習状態をLTMに保存する。"""
        try:
            self.meta.goal_queue.save_to_ltm(self.ltm)
        except Exception as exc:
            logger.warning("状態保存エラー: %s", exc)

    def _load_state(self) -> None:
        """LTMからGoalQueueを復元する。"""
        try:
            self.meta.goal_queue.load_from_ltm(self.ltm)
        except Exception as exc:
            logger.warning("状態ロードエラー: %s", exc)

    # ------------------------------------------------------------------
    # ハートビート
    # ------------------------------------------------------------------

    def _heartbeat(self) -> None:
        """LTMにハートビートを書き込む (daemon status確認用)。"""
        status = {
            "pid": os.getpid(),
            "timestamp": time.time(),
            "queue_size": self.meta.goal_queue.size(),
            "daily_used": self.budget.get_used(),
            "daily_max": self.budget.max_daily,
        }
        self.ltm.learn(_HEARTBEAT_KEY, json.dumps(status))

    # ------------------------------------------------------------------
    # シグナル処理
    # ------------------------------------------------------------------

    def _register_signals(self) -> None:
        """SIGTERMとSIGINTのハンドラを登録する。"""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """シグナル受信時にクリーンシャットダウンを開始する。"""
        logger.info("シグナル %d を受信。シャットダウンします。", signum)
        print(f"\n[デーモン] シグナル {signum} を受信。停止します...")
        self._stop_event.set()

    def _shutdown(self) -> None:
        """クリーンシャットダウン処理。"""
        logger.info("デーモンをシャットダウンします。状態を保存中...")
        print("[デーモン] 状態を保存中...")
        self._save_state()
        self._remove_pid()
        logger.info("=== Hermes AGI デーモン停止 ===")
        print("[デーモン] 停止しました。")

    # ------------------------------------------------------------------
    # PIDファイル
    # ------------------------------------------------------------------

    def _write_pid(self) -> None:
        """現在のPIDをファイルに書き込む。"""
        _PID_FILE.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """PIDファイルを削除する。"""
        try:
            _PID_FILE.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # 静的ユーティリティ (CLIから呼ぶ)
    # ------------------------------------------------------------------

    @staticmethod
    def get_status() -> dict:
        """デーモンの状態を返す。LTMのハートビートを使用。"""
        ltm = LongTermMemory()
        facts = ltm.recall_recent(limit=200)

        heartbeat = None
        for f in facts:
            if f["key"] == _HEARTBEAT_KEY:
                try:
                    heartbeat = json.loads(f["value"])
                    heartbeat["recorded_at"] = f.get("timestamp", 0)
                except (json.JSONDecodeError, TypeError):
                    pass
                break

        pid_running = False
        pid = None
        if _PID_FILE.exists():
            try:
                pid = int(_PID_FILE.read_text().strip())
                os.kill(pid, 0)  # シグナル0でプロセス存在確認
                pid_running = True
            except (ValueError, OSError):
                pid_running = False

        return {
            "running": pid_running,
            "pid": pid,
            "heartbeat": heartbeat,
        }

    @staticmethod
    def stop_daemon() -> bool:
        """PIDファイルを読んでデーモンにSIGTERMを送る。"""
        if not _PID_FILE.exists():
            return False
        try:
            pid = int(_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            return True
        except (ValueError, OSError):
            return False

    @staticmethod
    def get_log(lines: int = 50) -> str:
        """デーモンのログを返す。"""
        if not _LOG_FILE.exists():
            return "ログファイルが存在しません。"
        try:
            log_lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
            return "\n".join(log_lines[-lines:])
        except Exception as exc:
            return f"ログ読み込みエラー: {exc}"


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    from hermes_agi_gen.mistral_client import MistralClient
    try:
        llm = MistralClient()
        print(f"[デーモン] LLM: {llm.model}")
    except Exception as e:
        print(f"[デーモン] LLM初期化エラー: {e}")
        llm = None

    daemon = HermesDaemon(llm=llm)
    daemon.run_forever()
