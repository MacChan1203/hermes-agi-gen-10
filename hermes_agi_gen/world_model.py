"""世界モデル。エージェントが環境状態と因果関係を追跡する。

各アクションが世界に与えた変化を記録し、
計画前に環境の現在状態を考慮できるようにする。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CausalEffect:
    """アクションと結果の因果関係。"""
    action: str
    effect: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class WorldModel:
    """エージェントが保持する環境の内部表現。

    Attributes:
        filesystem: 既知のファイル/ディレクトリ情報
        environment: 環境変数・インストール済みパッケージ・設定
        causal_graph: アクション→結果の因果関係リスト
        known_services: 稼働中のサービス・プロセス情報
        installed_packages: インストール済みPythonパッケージ
        git_state: gitリポジトリの状態
    """
    filesystem: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    causal_graph: List[CausalEffect] = field(default_factory=list)
    known_services: Dict[str, str] = field(default_factory=dict)
    installed_packages: List[str] = field(default_factory=list)
    git_state: Dict[str, Any] = field(default_factory=dict)

    def add_causal_effect(self, action: str, effect: str, confidence: float = 1.0) -> None:
        """アクションと結果の因果関係を記録する。"""
        # 同じアクションの古い記録は上書き
        self.causal_graph = [c for c in self.causal_graph if c.action != action]
        self.causal_graph.append(CausalEffect(action=action, effect=effect, confidence=confidence))
        # 最大100件
        if len(self.causal_graph) > 100:
            self.causal_graph = self.causal_graph[-100:]

    def update_from_cmd(self, cmd: str, stdout: str) -> None:
        """シェルコマンドの出力から世界状態を更新する。"""
        lines = stdout.splitlines()

        # ファイルシステム情報を抽出
        if "ls -la" in cmd or ("find" in cmd and "sort" in cmd):
            self.filesystem["last_scan"] = stdout[:2000]
            self.filesystem["scan_time"] = time.time()

        # Python環境情報
        if "pip list" in cmd or "pip freeze" in cmd:
            packages = []
            for line in lines:
                parts = line.split()
                if parts and not line.startswith("-") and not line.startswith("Package"):
                    packages.append(parts[0].lower())
            if packages:
                self.installed_packages = packages
                self.environment["pip_packages"] = packages

        # gitの状態
        if "git status" in cmd:
            self.git_state["status"] = stdout[:500]
            self.git_state["updated_at"] = time.time()
        if "git log" in cmd:
            self.git_state["recent_commits"] = stdout[:500]

        # 環境変数
        if cmd.startswith("env") or "printenv" in cmd:
            for line in lines[:30]:
                if "=" in line and not any(s in line for s in ["SECRET", "PASSWORD", "TOKEN", "KEY"]):
                    k, _, v = line.partition("=")
                    self.environment[k.strip()] = v.strip()[:100]

        # ファイル書き込みの追跡
        if cmd.startswith("WRITE:") or "> " in cmd:
            self.add_causal_effect(cmd[:60], f"ファイルを書き込みました", confidence=0.9)

    def predict_outcome(self, action: str) -> Optional[str]:
        """過去の因果グラフから、このアクションの結果を予測する。"""
        action_lower = action.lower()
        for effect in reversed(self.causal_graph):
            if effect.action.lower() in action_lower or action_lower in effect.action.lower():
                return f"[予測: {effect.effect}] (信頼度: {effect.confidence:.0%})"
        return None

    def has_package(self, package_name: str) -> Optional[bool]:
        """パッケージがインストール済みかどうかを返す。未確認はNone。"""
        if not self.installed_packages:
            return None
        return package_name.lower() in self.installed_packages

    def summary(self) -> str:
        """世界モデルの現在状態を要約する。"""
        parts = []
        if self.filesystem.get("last_scan"):
            scan_time = self.filesystem.get("scan_time", 0)
            age = int(time.time() - scan_time)
            parts.append(f"FS: {age}秒前にスキャン済み")
        if self.installed_packages:
            parts.append(f"パッケージ: {len(self.installed_packages)}個確認済み")
        if self.git_state:
            parts.append("Git: 状態確認済み")
        if self.causal_graph:
            parts.append(f"因果関係: {len(self.causal_graph)}件記録済み")
        return " | ".join(parts) if parts else "未初期化"

    def get_recent_effects(self, limit: int = 5) -> List[Tuple[str, str]]:
        """最近の因果関係を返す。"""
        recent = self.causal_graph[-limit:]
        return [(e.action, e.effect) for e in reversed(recent)]
