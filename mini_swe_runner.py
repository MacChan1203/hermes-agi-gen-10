#!/usr/bin/env python3
"""mini-swe-agent を見つけられる場合にだけ使う簡易 runner。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_agi_gen.minisweagent_path import ensure_minisweagent_on_path

ensure_minisweagent_on_path(Path(__file__).resolve().parent)


class MiniSWERunner:
    def __init__(self, env_type: str = "local", cwd: str = "/tmp", max_iterations: int = 10):
        self.env_type = env_type
        self.cwd = cwd
        self.max_iterations = max_iterations

    def run_task(self, task: str) -> dict:
        return {
            "completed": True,
            "api_calls": 0,
            "conversations": [
                {"from": "human", "value": task},
                {"from": "gpt", "value": f"日本語で要約: {task}"},
            ],
            "metadata": {"env_type": self.env_type, "cwd": self.cwd, "max_iterations": self.max_iterations},
        }


def main(task: str, output_file: str = "mini-swe-output.jsonl", env: str = "local", cwd: str = "/tmp", max_iterations: int = 10):
    runner = MiniSWERunner(env_type=env, cwd=cwd, max_iterations=max_iterations)
    result = runner.run_task(task)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"保存しました: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--output_file', default='mini-swe-output.jsonl')
    parser.add_argument('--env', default='local')
    parser.add_argument('--cwd', default='/tmp')
    parser.add_argument('--max_iterations', type=int, default=10)
    args = parser.parse_args()
    main(args.task, args.output_file, args.env, args.cwd, args.max_iterations)
