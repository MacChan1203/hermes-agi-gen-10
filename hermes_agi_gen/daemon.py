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
from datetime import date
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mistral_client import MistralClient

from .agent_runner import HermesAgentV10
from .agent_state import AgentState
from .config import (
    DAEMON_DAILY_BUDGET,
    DAEMON_IDLE_EXPLORE_SEC,
    DAEMON_REGROUND_INTERVAL,
    DAEMON_CURIOSITY_THRESHOLD,
)
from .hermes_constants import DOMAIN_CONFIG
from .long_term_memory import LongTermMemory
from .meta_cognition import GoalQueue, MetaCognition, QueuedGoal
from .reflection_engine import ReflectionEngine
from .scheduler import JobScheduler
from .self_improvement import SelfImprovementEngine
from .world_model import WorldModel

logger = logging.getLogger(__name__)

from .hermes_constants import get_hermes_home

_HERMES_DIR = get_hermes_home()
_PID_FILE = _HERMES_DIR / "daemon.pid"
_LOG_FILE = _HERMES_DIR / "daemon.log"
_BUDGET_COUNTER_FILE = _HERMES_DIR / "daemon_budget.json"
_HEARTBEAT_KEY = "daemon_heartbeat"


class DailyBudgetGuard:
    """1日あたりのAPI使用量を制限する。コスト爆発を防ぐ。

    日付ごとのカウンタをJSONファイルで管理する (LTMスキャンを排除)。
    """

    # プロセス内の複数スレッドが同時に fallback 経路へ入っても
    # 予算カウンタが破壊されないよう、スレッドロックを併用する。
    _thread_lock: threading.Lock = threading.Lock()

    def __init__(self, max_daily: int = DAEMON_DAILY_BUDGET) -> None:
        self.max_daily = max_daily
        self._counter_file = _BUDGET_COUNTER_FILE

    def _today_key(self) -> str:
        return date.today().isoformat()

    def _load_counters(self) -> dict:
        """カウンタファイルを読み込む。"""
        if not self._counter_file.exists():
            return {}
        try:
            return json.loads(self._counter_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_counters(self, counters: dict) -> None:
        """カウンタファイルに書き出す。古い日付のエントリは削除する。

        tmp 書込 + os.replace でアトミック化し、途中クラッシュ時の
        JSON 破損を防ぐ。
        """
        today = self._today_key()
        cleaned = {k: v for k, v in counters.items() if k == today}
        tmp = self._counter_file.with_suffix(self._counter_file.suffix + ".tmp")
        try:
            self._counter_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._counter_file)
        except OSError as exc:
            logger.warning("予算カウンタ保存エラー: %s", exc)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def get_used(self) -> int:
        counters = self._load_counters()
        return counters.get(self._today_key(), 0)

    def increment(self) -> int:
        """カウンタをアトミックにインクリメントする。

        ファイルロックで読み→書きの競合を防止する。
        """
        import fcntl
        self._counter_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._counter_file, "a+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read()
                    counters = json.loads(raw) if raw.strip() else {}
                    key = self._today_key()
                    count = counters.get(key, 0) + 1
                    counters[key] = count
                    # 今日以外のエントリを削除
                    cleaned = {k: v for k, v in counters.items() if k == key}
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(cleaned, ensure_ascii=False))
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return count
        except Exception as exc:
            logger.warning("予算カウンタ更新エラー (fcntl): %s — スレッドロック版にフォールバック", exc)
            # フォールバック: fcntl が使えない環境 (Windows / 一部 FS) 用。
            # プロセス間安全性は失われるが、スレッドロック + アトミック置換で
            # 単一プロセス内の race と途中書き込み破損は防ぐ。
            with self.__class__._thread_lock:
                counters = self._load_counters()
                key = self._today_key()
                count = counters.get(key, 0) + 1
                counters[key] = count
                self._save_counters(counters)
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
        idle_seconds: int = DAEMON_IDLE_EXPLORE_SEC,
        max_daily_goals: int = DAEMON_DAILY_BUDGET,
        curiosity_threshold: float = DAEMON_CURIOSITY_THRESHOLD,
    ) -> None:
        _HERMES_DIR.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.idle_seconds = idle_seconds
        self.ltm = LongTermMemory()
        self.meta = MetaCognition(llm=llm)
        self.self_improver = SelfImprovementEngine(llm=llm)
        self.budget = DailyBudgetGuard(max_daily=max_daily_goals)
        self.curiosity_threshold = curiosity_threshold
        self.scheduler = JobScheduler()
        self.reflection_engine = ReflectionEngine(
            llm=llm,
            reflection_interval=5,  # 5ゴールごとに省察
        )
        self.world_model = WorldModel()
        self._stop_event = threading.Event()
        self._recently_completed: list[str] = []  # 重複防止LRU (最大20件)
        self._max_recent = 20

        # ファイルロガーを設定
        self._setup_file_logger()

    def _setup_file_logger(self) -> None:
        """デーモン専用のファイルロガーを設定する。"""
        self._file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        self._file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file_handler.setFormatter(formatter)
        logger.addHandler(self._file_handler)
        logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 起動・停止
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """デーモンのメインループ。SIGTERMまたはSIGINTで停止する。"""
        self._register_signals()
        self._write_pid()
        self._load_state()

        # 世界モデルのグラウンディング
        self.world_model.initialize_from_filesystem(".")
        logger.info("世界モデルをグラウンディング済み")

        logger.info("[デーモン] 起動しました (PID=%d)", os.getpid())
        logger.info("[デーモン] GoalQueue: %d件", self.meta.goal_queue.size())
        logger.info("[デーモン] 本日の残り予算: %d件", self.budget.remaining())
        logger.info("[デーモン] PID=%s, ログ=%s", _PID_FILE, _LOG_FILE)

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
                logger.info("[スケジューラー] ジョブ発火: [%s] %s", job.id, job.goal[:60])
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

        # 省察タイミングチェック: N ゴールごとに能動的省察を実行
        if hasattr(self.reflection_engine, 'should_reflect') and self.reflection_engine.should_reflect():
            self._run_reflection_cycle()

    def _process_one_goal(self, goal: QueuedGoal) -> None:
        """GoalQueueからポップしたゴールを実行する。"""
        logger.info(
            "[デーモン] ゴール実行 [%s / score=%.2f]: %s",
            goal.source, goal.composite_score, goal.goal[:70],
        )

        used = self.budget.increment()
        logger.info("[デーモン] 本日 %d/%d ゴール消費", used, self.budget.max_daily)

        try:
            agent = HermesAgentV10(
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

            logger.info("[デーモン] 完了 (成功率: %.0f%%): %s", score * 100, goal.goal[:60])

            # GoalQueueを保存 (次ゴールが追加されている可能性)
            self._save_state()

        except Exception as exc:
            logger.error("ゴール処理エラー: %s → %s", goal.goal[:60], exc)

    def _run_reflection_cycle(self) -> None:
        """能動的自己省察サイクル: LTMを分析して洞察と新ゴールを生成する。"""
        logger.info("[デーモン] 自己省察フェーズ...")

        try:
            # 世界モデルが古ければ再グラウンディング
            if self.world_model.needs_regrounding(max_age_seconds=DAEMON_REGROUND_INTERVAL):
                self.world_model.initialize_from_filesystem(".")
                logger.info("[ReflectionCycle] 世界モデルを再グラウンディング")

            # 省察を実行
            insights = self.reflection_engine.reflect(self.ltm)

            # 洞察から戦略的ゴールを生成してキューに追加
            strategic_goals = self.reflection_engine.generate_strategic_goals(insights, self.ltm)
            for goal in strategic_goals:
                self.meta.goal_queue.add(goal)
                logger.info("[デーモン] 省察→新ゴール: %s", goal.goal[:60])

            self._save_state()
            logger.info("[デーモン] 省察完了: %d洞察, %d新ゴール", len(insights), len(strategic_goals))
        except Exception as exc:
            logger.error("[ReflectionCycle] エラー: %s", exc, exc_info=True)

    def _idle_explore(self) -> None:
        """GoalQueueが空のとき、好奇心駆動のゴールを生成して追加する。"""
        if self.llm is None:
            return

        logger.info("[デーモン] 好奇心探索: 新しいゴールを生成します。")

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
            logger.info("[デーモン] 好奇心ゴール: %s (score=%.2f)", best.goal[:60], best.composite_score)
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
        logger.info("[デーモン] シグナル %d を受信。停止します...", signum)
        self._stop_event.set()

    def _shutdown(self) -> None:
        """クリーンシャットダウン処理。"""
        logger.info("[デーモン] 状態を保存中...")
        self._save_state()
        self._remove_pid()
        # ログファイルハンドルを確実に閉じる
        if hasattr(self, "_file_handler"):
            logger.removeHandler(self._file_handler)
            self._file_handler.close()
        logger.info("[デーモン] 停止しました。")

    # ------------------------------------------------------------------
    # PIDファイル
    # ------------------------------------------------------------------

    def _write_pid(self) -> None:
        """現在のPIDをファイルに書き込む。

        既存PIDファイルがあり、そのプロセスがまだ生きている場合は
        起動を拒否して重複デーモンを防ぐ。
        """
        if _PID_FILE.exists():
            try:
                old_pid = int(_PID_FILE.read_text().strip())
                if old_pid != os.getpid():
                    os.kill(old_pid, 0)  # プロセス存在確認
                    raise RuntimeError(
                        f"デーモンは既に起動中です (PID={old_pid})。"
                        " 停止してから再起動してください。"
                    )
            except (ValueError, OSError):
                # PIDファイルが壊れているか、プロセスが存在しない → 上書きOK
                pass
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
        logger.info("[デーモン] LLM: %s", llm.model)
    except Exception as e:
        logger.error("[デーモン] LLM初期化エラー: %s", e)
        llm = None

    daemon = HermesDaemon(llm=llm)
    daemon.run_forever()
