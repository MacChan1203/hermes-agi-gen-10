"""時刻ベースのジョブスケジューラー。

外部ライブラリ不要。~/.hermes/scheduler.json に永続化。

サポートするトリガー形式:
  once:<ISO8601>           例: once:2026-03-31T09:00
  every:<N>m               例: every:30m  (30分ごと)
  every:<N>h               例: every:2h   (2時間ごと)
  daily:<HH:MM>            例: daily:09:00
  weekly:<weekday>:<HH:MM> 例: weekly:mon:09:00
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .meta_cognition import GoalQueue

from .config import SCHEDULER_MAX_JOBS

logger = logging.getLogger(__name__)

from .hermes_constants import get_hermes_path

_HERMES_DIR: Optional[Path] = None
_SCHEDULE_FILE: Optional[Path] = None

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _parse_hh_mm(value: str) -> Optional[tuple[int, int]]:
    try:
        hh, mm = map(int, value.split(":"))
    except ValueError:
        return None
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return hh, mm


def _parse_positive_interval(spec: str) -> Optional[timedelta]:
    spec = spec.lower().strip()
    try:
        if spec.endswith("m"):
            amount = int(spec[:-1])
            delta = timedelta(minutes=amount)
        elif spec.endswith("h"):
            amount = int(spec[:-1])
            delta = timedelta(hours=amount)
        elif spec.endswith("s"):
            amount = int(spec[:-1])
            delta = timedelta(seconds=amount)
        else:
            logger.warning("every: の単位が不明です: %s", spec)
            return None
    except ValueError:
        logger.warning("every: の間隔が無効です: %s", spec)
        return None
    if amount <= 0:
        logger.warning("every: の間隔は正の値が必要です: %s", spec)
        return None
    return delta


@dataclass
class ScheduledJob:
    """スケジュール済みジョブ。"""
    id: str
    goal: str
    trigger: str           # "once:...", "every:30m", "daily:09:00", "weekly:mon:09:00"
    domain: str = "general"
    priority: float = 0.6
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_run: Optional[float] = None   # 最後に実行した時刻 (UNIX timestamp)
    next_run: Optional[float] = None   # 次の実行予定時刻

    def __post_init__(self) -> None:
        if self.next_run is None:
            self.next_run = _calc_next_run(self.trigger, last_run=None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledJob":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# トリガー計算
# ------------------------------------------------------------------

def _calc_next_run(trigger: str, last_run: Optional[float]) -> Optional[float]:
    """トリガー文字列から次の実行時刻 (UNIXタイムスタンプ) を計算する。"""
    now = datetime.now()
    trigger = trigger.strip()

    # once:<ISO8601>
    if trigger.lower().startswith("once:"):
        iso = trigger[5:].strip()
        try:
            dt = datetime.fromisoformat(iso)
            ts = dt.timestamp()
            # 過去の once ジョブは実行済みとして None を返す
            if last_run is not None and last_run >= ts:
                return None
            return ts
        except ValueError:
            logger.warning("once: の日時が無効です: %s", iso)
            return None

    # every:<N>m or every:<N>h
    if trigger.lower().startswith("every:"):
        spec = trigger[6:].lower().strip()
        delta = _parse_positive_interval(spec)
        if delta is None:
            return None

        base = datetime.fromtimestamp(last_run) if last_run else now
        return (base + delta).timestamp()

    # daily:<HH:MM>
    if trigger.lower().startswith("daily:"):
        t_str = trigger[6:].strip()
        parsed_time = _parse_hh_mm(t_str)
        if parsed_time is None:
            logger.warning("daily: の時刻が無効です: %s", t_str)
            return None
        hh, mm = parsed_time
        candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate.timestamp() <= time.time():
            candidate += timedelta(days=1)
        return candidate.timestamp()

    # weekly:<weekday>:<HH:MM>
    if trigger.lower().startswith("weekly:"):
        parts = trigger[7:].split(":", 1)
        if len(parts) != 2:
            logger.warning("weekly: の形式が無効です: %s", trigger)
            return None
        day_str, t_str = parts[0].lower(), parts[1].strip()
        weekday = _WEEKDAYS.get(day_str)
        if weekday is None:
            logger.warning("weekly: の曜日が不明です: %s", day_str)
            return None
        parsed_time = _parse_hh_mm(t_str)
        if parsed_time is None:
            logger.warning("weekly: の時刻が無効です: %s", t_str)
            return None
        hh, mm = parsed_time

        candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        days_ahead = (weekday - now.weekday()) % 7
        if days_ahead == 0 and candidate.timestamp() <= time.time():
            days_ahead = 7
        candidate += timedelta(days=days_ahead)
        return candidate.timestamp()

    logger.warning("不明なトリガー形式: %s", trigger)
    return None


def parse_trigger_spec(spec: str) -> Optional[str]:
    """ユーザー入力からトリガー文字列を正規化して返す。

    サポートする入力:
      2026-03-31T09:00          → once:2026-03-31T09:00
      +30m                      → every:30m
      +2h                       → every:2h
      every 30m / every 30min   → every:30m
      daily 09:00               → daily:09:00
      weekly mon 09:00          → weekly:mon:09:00
    """
    spec = spec.strip()

    # ISO datetime → once:
    try:
        datetime.fromisoformat(spec)
        return f"once:{spec}"
    except ValueError:
        pass

    # +30m / +2h
    if spec.startswith("+"):
        canonical = f"every:{spec[1:]}"
        return canonical if _calc_next_run(canonical, last_run=None) is not None else None

    lower = spec.lower()

    # every 30m / every 30min / every 2h
    if lower.startswith("every "):
        rest = lower[6:].strip().replace("min", "m").replace("hour", "h").replace("hours", "h")
        rest = rest.replace(" ", "")
        canonical = f"every:{rest}"
        return canonical if _calc_next_run(canonical, last_run=None) is not None else None

    # daily 09:00
    if lower.startswith("daily "):
        canonical = f"daily:{lower[6:].strip()}"
        return canonical if _calc_next_run(canonical, last_run=None) is not None else None

    # weekly mon 09:00
    if lower.startswith("weekly "):
        parts = lower[7:].strip().split()
        if len(parts) == 2:
            canonical = f"weekly:{parts[0]}:{parts[1]}"
            return canonical if _calc_next_run(canonical, last_run=None) is not None else None

    # already in canonical form
    if any(lower.startswith(p) for p in ("once:", "every:", "daily:", "weekly:")):
        return spec if _calc_next_run(spec, last_run=None) is not None else None

    return None


# ------------------------------------------------------------------
# JobScheduler
# ------------------------------------------------------------------

class JobScheduler:
    """ジョブのスケジュール管理と実行トリガー。"""

    def __init__(self) -> None:
        self._schedule_file = self._resolve_schedule_file()
        self._schedule_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._schedule_file.with_suffix(self._schedule_file.suffix + ".lock")
        self._jobs: List[ScheduledJob] = []
        self.load()

    @staticmethod
    def _resolve_schedule_file() -> Path:
        """Resolve schedule path at construction time.

        The module-level variables are kept only for older tests/callers that
        monkeypatch them directly.
        """
        if _SCHEDULE_FILE is not None:
            return Path(_SCHEDULE_FILE)
        if _HERMES_DIR is not None:
            return Path(_HERMES_DIR) / "scheduler.json"
        return get_hermes_path("scheduler.json")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_job(
        self,
        goal: str,
        trigger: str,
        domain: str = "general",
        priority: float = 0.6,
        job_id: Optional[str] = None,
    ) -> ScheduledJob:
        """ジョブを追加して保存する。

        Raises:
            RuntimeError: ジョブ数が SCHEDULER_MAX_JOBS に達している場合。
        """
        if len(self._jobs) >= SCHEDULER_MAX_JOBS:
            raise RuntimeError(
                f"ジョブ数が上限 ({SCHEDULER_MAX_JOBS}) に達しています。"
                " 不要なジョブを削除してから追加してください。"
            )
        next_run = _calc_next_run(trigger, last_run=None)
        if next_run is None:
            raise ValueError(f"無効なスケジュールトリガーです: {trigger}")
        job = ScheduledJob(
            id=job_id or str(uuid.uuid4())[:8],
            goal=goal,
            trigger=trigger,
            domain=domain,
            priority=priority,
            next_run=next_run,
        )
        self._jobs.append(job)
        self.save()
        logger.info("スケジュール追加: [%s] %s (trigger=%s)", job.id, goal[:60], trigger)
        return job

    def remove_job(self, job_id: str) -> bool:
        """ジョブIDでジョブを削除する。"""
        before = len(self._jobs)
        self._jobs = [j for j in self._jobs if j.id != job_id]
        if len(self._jobs) < before:
            self.save()
            return True
        return False

    def enable_job(self, job_id: str, enabled: bool = True) -> bool:
        for job in self._jobs:
            if job.id == job_id:
                job.enabled = enabled
                self.save()
                return True
        return False

    def list_jobs(self) -> List[ScheduledJob]:
        return list(self._jobs)

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    # ------------------------------------------------------------------
    # スケジューラーティック
    # ------------------------------------------------------------------

    def tick(self, goal_queue: "GoalQueue") -> List[ScheduledJob]:
        """期限を迎えたジョブをGoalQueueに追加する。変更があれば保存する。"""
        # デーモン起動後に追加されたジョブを拾うためにディスクから再読み込みする
        self._reload_if_changed()
        now = time.time()
        triggered: List[ScheduledJob] = []
        changed = False

        for job in self._jobs:
            if not job.enabled:
                continue
            if job.next_run is None:
                continue
            if job.next_run > now:
                continue

            # 期限到達 → GoalQueueに追加
            from .meta_cognition import QueuedGoal
            goal_queue.add(QueuedGoal(
                goal=job.goal,
                priority_score=job.priority,
                source="scheduler",
                rationale=f"スケジュール実行 [job_id={job.id}, trigger={job.trigger}]",
                domain=job.domain,
            ))
            logger.info(
                "スケジュール発火: [%s] %s (trigger=%s)",
                job.id, job.goal[:60], job.trigger,
            )
            triggered.append(job)

            # 次の実行時刻を更新
            job.last_run = now
            next_ts = _calc_next_run(job.trigger, last_run=now)
            job.next_run = next_ts

            # once: ジョブは次の実行がないので無効化
            if next_ts is None and job.trigger.lower().startswith("once:"):
                job.enabled = False

            changed = True

        if changed:
            self.save()

        return triggered

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def _reload_if_changed(self) -> None:
        """ファイルが更新されていれば再読み込みする (デーモン起動後の追加ジョブを拾う)。"""
        if not self._schedule_file.exists():
            return
        try:
            mtime = self._schedule_file.stat().st_mtime
            if not hasattr(self, "_last_mtime") or mtime != self._last_mtime:
                self.load()
                self._last_mtime = mtime
        except Exception as exc:
            logger.warning("スケジューラー再読み込みエラー: %s", exc)

    def save(self) -> None:
        """ジョブリストをJSONファイルに保存する (flock + atomic replace)。"""
        try:
            self._schedule_file.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file.parent.mkdir(parents=True, exist_ok=True)
            data = [j.to_dict() for j in self._jobs]
            content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            with open(self._lock_file, "a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                tmp_path = self._schedule_file.with_suffix(self._schedule_file.suffix + ".tmp")
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self._schedule_file)
                    self._last_mtime = self._schedule_file.stat().st_mtime
                finally:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            logger.warning("スケジューラー保存エラー: %s", exc)

    def load(self) -> None:
        """JSONファイルからジョブリストを読み込む (flock で共有ロック)。

        ロック保持中にファイル読み込み・JSON解析・ジョブ構築を全て行い、
        TOCTOU 競合を防止する。
        """
        if not self._schedule_file.exists():
            return
        try:
            with open(self._lock_file, "a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
                try:
                    raw = self._schedule_file.read_text(encoding="utf-8")
                    data = json.loads(raw) if raw.strip() else []
                    self._jobs = [ScheduledJob.from_dict(d) for d in data]
                    self._last_mtime = self._schedule_file.stat().st_mtime
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            logger.warning("スケジューラーロードエラー: %s", exc)

    # ------------------------------------------------------------------
    # 表示ユーティリティ
    # ------------------------------------------------------------------

    def format_next_run(self, job: ScheduledJob) -> str:
        """次の実行時刻を人間が読みやすい形式で返す。"""
        if job.next_run is None:
            return "完了済み / 無効"
        dt = datetime.fromtimestamp(job.next_run)
        delta = job.next_run - time.time()
        if delta < 0:
            return f"{dt.strftime('%m/%d %H:%M')} (実行待ち)"
        if delta < 3600:
            return f"{dt.strftime('%m/%d %H:%M')} (あと{delta/60:.0f}分)"
        if delta < 86400:
            return f"{dt.strftime('%m/%d %H:%M')} (あと{delta/3600:.1f}時間)"
        return f"{dt.strftime('%m/%d %H:%M')} (あと{delta/86400:.1f}日)"
