#!/usr/bin/env python3
"""Hermes AGI デーモンの自動起動設定スクリプト。

macOS の launchd に登録してシステム起動時にデーモンを自動起動する。

使い方:
    python3 setup_autostart.py install    # launchd に登録
    python3 setup_autostart.py uninstall  # launchd から削除
    python3 setup_autostart.py status     # 登録状態を確認
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PLIST_NAME = "com.hermes.agi.daemon"
_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_NAME}.plist"
_HERMES_DIR = Path(__file__).parent.resolve()
_LOG_DIR = Path.home() / ".hermes"


def _find_python() -> str:
    """現在のPython実行ファイルパスを返す。"""
    return sys.executable


def _find_dotenv() -> str:
    """dotenvファイルのパスを返す。"""
    return str(_HERMES_DIR / ".env")


def _build_plist() -> str:
    """launchd plist XMLを生成する。"""
    python = _find_python()
    env_file = _find_dotenv()
    log_out = str(_LOG_DIR / "daemon.stdout.log")
    log_err = str(_LOG_DIR / "daemon.stderr.log")
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # .env から環境変数を読み込む
    env_vars = {}
    dotenv_path = Path(env_file)
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env_vars[key.strip()] = val.strip()

    env_dict_xml = "\n".join(
        f"            <key>{k}</key>\n            <string>{v}</string>"
        for k, v in env_vars.items()
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>hermes_agi_gen.daemon</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{_HERMES_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
{env_dict_xml}
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_out}</string>

    <key>StandardErrorPath</key>
    <string>{log_err}</string>

    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
"""


def install() -> None:
    """launchd にデーモンを登録する。"""
    plist_content = _build_plist()
    _PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PLIST_PATH.write_text(plist_content)
    print(f"✓ plist を作成しました: {_PLIST_PATH}")

    # 既存のエージェントをアンロード (エラーは無視)
    subprocess.run(
        ["launchctl", "unload", str(_PLIST_PATH)],
        capture_output=True,
    )

    # 新しいエージェントをロード
    result = subprocess.run(
        ["launchctl", "load", str(_PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✓ launchd に登録しました ({_PLIST_NAME})")
        print("  システム起動時・ユーザーログイン時に自動的にデーモンが起動します。")
        print(f"  ログ: ~/.hermes/daemon.stdout.log")
    else:
        print(f"✗ launchd 登録エラー: {result.stderr}")


def uninstall() -> None:
    """launchd からデーモンの登録を削除する。"""
    if not _PLIST_PATH.exists():
        print("デーモンは登録されていません。")
        return

    subprocess.run(
        ["launchctl", "unload", str(_PLIST_PATH)],
        capture_output=True,
    )
    _PLIST_PATH.unlink()
    print(f"✓ launchd から削除しました ({_PLIST_NAME})")


def status() -> None:
    """登録状態を確認する。"""
    if not _PLIST_PATH.exists():
        print("❌ launchd 未登録")
        print("  python3 setup_autostart.py install で登録できます。")
        return

    result = subprocess.run(
        ["launchctl", "list", _PLIST_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✓ launchd 登録済み")
        print(result.stdout)
    else:
        print(f"⚠ plist は存在しますが、launchd にロードされていません。")
        print(f"  python3 setup_autostart.py install で再登録してください。")

    print(f"plist: {_PLIST_PATH}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "install":
        install()
    elif cmd == "uninstall":
        uninstall()
    elif cmd == "status":
        status()
    else:
        print(__doc__)
        sys.exit(1)
