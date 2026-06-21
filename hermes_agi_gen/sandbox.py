"""OS レベルのプロセス隔離 (macOS seatbelt / sandbox-exec)。

設計意図:
    executor.py の PYTHON:/CMD: は、最終的に本物のインタプリタ/バイナリを
    subprocess で実行する。AST 許可リストやコマンド denylist は静的解析であり、
    本物のインタプリタ上では原理的に回避され得る (例: str.format 経由の dunder
    アクセス、`env python3 -c`、awk の system() など)。

    したがって信頼境界をカーネルに移す。実行対象のプロセスを sandbox-exec で
    包み、以下をカーネルレベルで強制する:
      - ネットワーク全面拒否 (情報持ち出しチャネルを塞ぐ)
      - 書き込みを repo_root と明示 allow-dir、一時ディレクトリのみに限定
      - 読み取りは許可 (stdlib ロード等に必要。ネットワーク拒否済みのため
        秘密を読めても持ち出せない)

    seatbelt は子プロセスに継承されるため、サンドボックス内から起動された
    別バイナリ (awk が呼ぶ sh、env が呼ぶ python3 等) も同じ制約を受ける。
    これが「列挙ではなく構造で」突破を防ぐ核心。

    静的チェック (_is_python_safe / コマンド denylist) は引き続き多層防御として
    機能させ、本モジュールはその下のカーネル境界を担う。

注記:
    SEARCH:/FETCH:/WRITE: は executor 本体 (メインプロセス) 内で各自の検証済み
    経路を通るため、本サンドボックス (ネットワーク拒否・書込制限) の影響を受けない。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def sandbox_available() -> bool:
    """このプラットフォームで sandbox-exec による隔離が使えるか。"""
    return sys.platform == "darwin" and os.path.exists(_SANDBOX_EXEC)


def _network_allowed() -> bool:
    """PYTHON:/CMD: サブプロセスにネットワークを許可するか (既定: 拒否)。

    既定では拒否。エージェントの正規のネットワークアクセスは SEARCH:/FETCH:
    (メインプロセス) を経由する。どうしても PYTHON: 内から HTTP したい場合のみ
    HERMES_SANDBOX_ALLOW_NETWORK=1 で明示的にオプトインする。
    """
    return os.getenv("HERMES_SANDBOX_ALLOW_NETWORK", "") == "1"


def _sb_escape(path: str) -> str:
    """seatbelt プロファイルの文字列リテラル用にエスケープする。"""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _resolve_variants(path: Path) -> List[str]:
    """パスを resolve し、macOS の /tmp→/private/tmp 等のエイリアスも含める。"""
    variants: List[str] = []
    try:
        rp = str(path.resolve())
    except OSError:
        rp = str(path)
    variants.append(rp)
    # macOS では /tmp, /var は /private/... の symlink。realpath 済みでも
    # 念のため両表記を許可しておく。
    if rp.startswith("/private/"):
        variants.append(rp[len("/private"):])
    else:
        variants.append("/private" + rp)
    return variants


def _write_subpaths(repo_root: Path, allow_dirs: Iterable[Path]) -> List[str]:
    """書き込みを許可する subpath 群を構築する。"""
    paths: set[str] = set()
    for p in [repo_root, *allow_dirs]:
        for v in _resolve_variants(Path(p)):
            paths.add(v)
    # インタプリタ/ツールが使う一時ディレクトリ
    for tmp in (tempfile.gettempdir(), "/var/folders", "/private/var/folders"):
        for v in _resolve_variants(Path(tmp)):
            paths.add(v)
    return sorted(paths)


def build_profile(repo_root: Path | str, allow_dirs: Iterable[Path] = ()) -> str:
    """repo_root と allow_dirs を書込可能とする seatbelt プロファイルを生成する。"""
    repo_root = Path(repo_root)
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow file-read*)",
        "(allow network*)" if _network_allowed() else "(deny network*)",
    ]
    write_rules = " ".join(
        f'(subpath "{_sb_escape(p)}")' for p in _write_subpaths(repo_root, allow_dirs)
    )
    lines.append(f'(allow file-write* {write_rules} (literal "/dev/null"))')
    return "\n".join(lines)


def wrap(
    argv: Sequence[str],
    repo_root: Path | str,
    allow_dirs: Iterable[Path] = (),
) -> Optional[List[str]]:
    """argv を sandbox-exec でラップした新しい argv を返す。

    sandbox-exec が利用できない環境では None を返す (呼び出し側がフォールバック)。
    """
    if not sandbox_available():
        return None
    profile = build_profile(repo_root, allow_dirs)
    return [_SANDBOX_EXEC, "-p", profile, *argv]
