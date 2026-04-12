"""共有ユーティリティ。"""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Union

import yaml


def atomic_json_write(path: Union[str, Path], data: Any, *, indent: int = 2, **dump_kwargs: Any) -> None:
    """JSONデータをアトミックにファイルへ書き込む。

    一時ファイルに書き込み後、os.replace でアトミックに置換する。
    元ファイルが存在する場合はパーミッションを引き継ぐ。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original file permissions if file exists
    original_mode = None
    if path.exists():
        try:
            original_mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            pass

    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            tmp_fd = None  # os.fdopen takes ownership of the fd
            json.dump(data, f, indent=indent, ensure_ascii=False, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
        tmp_path = None  # successfully replaced, no cleanup needed
    except BaseException:
        raise
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
) -> None:
    """YAMLデータをアトミックにファイルへ書き込む。

    一時ファイルに書き込み後、os.replace でアトミックに置換する。
    元ファイルが存在する場合はパーミッションを引き継ぐ。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original file permissions if file exists
    original_mode = None
    if path.exists():
        try:
            original_mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            pass

    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            tmp_fd = None  # os.fdopen takes ownership of the fd
            yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys, allow_unicode=True)
            if extra_content:
                f.write(extra_content)
            f.flush()
            os.fsync(f.fileno())
        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
        tmp_path = None  # successfully replaced, no cleanup needed
    except BaseException:
        raise
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
