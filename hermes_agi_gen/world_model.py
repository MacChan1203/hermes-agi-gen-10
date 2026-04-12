"""世界モデル。エージェントが環境状態と因果関係を追跡する。

各アクションが世界に与えた変化を記録し、
計画前に環境の現在状態を考慮できるようにする。

Gen 7追加: プロアクティブなグラウンディング (initialize_from_filesystem)
Gen 9追加: 資源認識型プランニング (コスト追跡・複雑度推定)
"""
from __future__ import annotations

import os
import subprocess
import sys
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
class ResourceCost:
    """ツール実行の資源コスト記録。"""
    tool_type: str           # "CMD" / "PYTHON" / "SEARCH" / etc.
    execution_time: float    # 実行時間 (秒)
    output_size: int         # 出力バイト数
    success: bool
    timestamp: float = field(default_factory=time.time)


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
        resource_history: ツール実行のコスト履歴 (Gen 9)
        uncertainty_map: 領域ごとの不確実性スコア (Gen 9)
    """
    filesystem: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    causal_graph: List[CausalEffect] = field(default_factory=list)
    known_services: Dict[str, str] = field(default_factory=dict)
    installed_packages: List[str] = field(default_factory=list)
    git_state: Dict[str, Any] = field(default_factory=dict)
    # Gen 9: 資源認識
    resource_history: List[ResourceCost] = field(default_factory=list)
    uncertainty_map: Dict[str, float] = field(default_factory=dict)

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

    # ------------------------------------------------------------------
    # Gen 7: プロアクティブなグラウンディング
    # ------------------------------------------------------------------

    def initialize_from_filesystem(self, path: str = ".") -> None:
        """実際のファイルシステムをスキャンして世界モデルを初期化する。

        デーモン起動時・セッション開始時に呼び出すことで、
        エージェントが実際の環境状態を把握した上で行動できるようにする。
        """
        try:
            # Python バージョン情報
            self.environment["python_version"] = sys.version
            self.environment["python_executable"] = sys.executable

            # カレントディレクトリ
            self.environment["cwd"] = os.path.abspath(path)

            # ファイルシステムの簡易スキャン (最大深度2)
            result = subprocess.run(
                ["find", path, "-maxdepth", "2", "-not", "-path", "*/.git/*",
                 "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                self.filesystem["shallow_scan"] = result.stdout[:3000]
                self.filesystem["scan_time"] = time.time()

            # Git状態
            git_result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, timeout=5, cwd=path,
            )
            if git_result.returncode == 0:
                self.git_state["status"] = git_result.stdout[:500]
                self.git_state["updated_at"] = time.time()

            self.environment["grounded_at"] = time.time()
        except Exception:
            pass  # グラウンディング失敗は致命的ではない

    def needs_regrounding(self, max_age_seconds: float = 300.0) -> bool:
        """世界モデルが古くなっていて再グラウンディングが必要か。"""
        grounded_at = self.environment.get("grounded_at")
        if grounded_at is None:
            return True
        return (time.time() - grounded_at) > max_age_seconds

    def grounding_age(self) -> Optional[float]:
        """グラウンディングからの経過秒数。未グラウンディングの場合はNone。"""
        grounded_at = self.environment.get("grounded_at")
        if grounded_at is None:
            return None
        return time.time() - grounded_at

    # ------------------------------------------------------------------
    # Gen 9: 資源認識型プランニング
    # ------------------------------------------------------------------

    def record_resource_cost(self, tool_type: str, execution_time: float, output_size: int, success: bool) -> None:
        """ツール実行のコストを記録する。"""
        self.resource_history.append(ResourceCost(
            tool_type=tool_type, execution_time=execution_time,
            output_size=output_size, success=success,
        ))
        if len(self.resource_history) > 500:
            self.resource_history = self.resource_history[-500:]

    def estimate_tool_cost(self, tool_type: str) -> Dict[str, float]:
        """過去の実績からツールの予想コストを返す。"""
        relevant = [r for r in self.resource_history if r.tool_type == tool_type]
        if not relevant:
            # デフォルト推定
            defaults = {
                "CMD": 2.0, "PYTHON": 5.0, "SEARCH": 8.0,
                "FETCH": 5.0, "READ": 0.5, "WRITE": 0.5,
                "CALC": 0.1, "PLAN": 0.1,
            }
            return {"avg_time": defaults.get(tool_type, 3.0), "success_rate": 0.7, "samples": 0}

        times = [r.execution_time for r in relevant]
        successes = sum(1 for r in relevant if r.success)
        return {
            "avg_time": sum(times) / len(times),
            "success_rate": successes / len(relevant),
            "samples": len(relevant),
        }

    def estimate_goal_complexity(self, goal: str) -> Dict[str, Any]:
        """ゴールの複雑度を推定する。"""
        length_score = min(1.0, len(goal) / 200.0)

        # キーワードベースの複雑度
        complex_keywords = ["分析", "リファクタ", "統合", "設計", "最適化", "デバッグ",
                           "analyze", "refactor", "integrate", "design", "optimize", "debug"]
        simple_keywords = ["表示", "読む", "確認", "計算", "print", "read", "check", "calc"]

        keyword_score = 0.5
        goal_lower = goal.lower()
        for kw in complex_keywords:
            if kw in goal_lower:
                keyword_score = min(1.0, keyword_score + 0.15)
        for kw in simple_keywords:
            if kw in goal_lower:
                keyword_score = max(0.1, keyword_score - 0.15)

        # 過去の類似タスク実績（因果グラフから）
        similar_effects = [e for e in self.causal_graph if any(w in e.action.lower() for w in goal_lower.split()[:3])]
        historical_score = 0.5
        if similar_effects:
            historical_score = sum(e.confidence for e in similar_effects) / len(similar_effects)

        complexity = (length_score * 0.2) + (keyword_score * 0.5) + ((1 - historical_score) * 0.3)

        # 推奨 max_iterations
        if complexity < 0.3:
            recommended_iterations = 3
        elif complexity < 0.6:
            recommended_iterations = 6
        elif complexity < 0.8:
            recommended_iterations = 9
        else:
            recommended_iterations = 12

        return {
            "complexity": round(complexity, 2),
            "length_score": round(length_score, 2),
            "keyword_score": round(keyword_score, 2),
            "historical_score": round(historical_score, 2),
            "recommended_iterations": recommended_iterations,
        }

    def get_uncertainty_areas(self) -> List[str]:
        """不確実性が高い領域のリストを返す。"""
        return [area for area, score in self.uncertainty_map.items() if score > 0.6]

    def update_uncertainty(self, area: str, delta: float) -> None:
        """領域の不確実性を更新する。"""
        current = self.uncertainty_map.get(area, 0.5)
        self.uncertainty_map[area] = max(0.0, min(1.0, current + delta))

    def resource_summary(self) -> str:
        """資源使用状況のサマリを返す。"""
        if not self.resource_history:
            return "資源記録なし"
        total_time = sum(r.execution_time for r in self.resource_history)
        total_ops = len(self.resource_history)
        success_rate = sum(1 for r in self.resource_history if r.success) / total_ops
        return f"総実行={total_ops}回 総時間={total_time:.1f}秒 成功率={success_rate:.0%}"
