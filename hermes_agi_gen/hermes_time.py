"""Hermes AGI Gen 用のタイムゾーン対応時計。"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

# キャッシュ初期化はデーモンの複数スレッドから同時に呼ばれうる。
# ダブルチェックロッキングで不整合な中間状態が観測されるのを防ぐ。
_cache_lock = threading.Lock()
_cached_tz: Optional[ZoneInfo] = None
_cached_tz_name: Optional[str] = None
_cache_resolved: bool = False


def _resolve_timezone_name() -> str:
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    if tz_env:
        return tz_env

    try:
        import yaml
        from .hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            tz_cfg = cfg.get("timezone", "")
            if isinstance(tz_cfg, str) and tz_cfg.strip():
                return tz_cfg.strip()
    except Exception:
        logger.debug("config.yaml からタイムゾーン取得に失敗", exc_info=True)

    return ""


def _get_zoneinfo(name: str) -> Optional[ZoneInfo]:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception as exc:
        logger.warning("不正なタイムゾーン '%s': %s。ローカル時刻へフォールバックします。", name, exc)
        return None


def get_timezone() -> Optional[ZoneInfo]:
    global _cached_tz, _cached_tz_name, _cache_resolved
    # ダブルチェックロッキング: 既に解決済みならロックを取らず即 return。
    if _cache_resolved:
        return _cached_tz
    with _cache_lock:
        if not _cache_resolved:
            _cached_tz_name = _resolve_timezone_name()
            _cached_tz = _get_zoneinfo(_cached_tz_name)
            _cache_resolved = True
    return _cached_tz


def get_timezone_name() -> str:
    if not _cache_resolved:
        get_timezone()
    return _cached_tz_name or ""


def now() -> datetime:
    tz = get_timezone()
    if tz is not None:
        return datetime.now(tz)
    return datetime.now().astimezone()


def reset_cache() -> None:
    global _cached_tz, _cached_tz_name, _cache_resolved
    with _cache_lock:
        _cached_tz = None
        _cached_tz_name = None
        _cache_resolved = False
