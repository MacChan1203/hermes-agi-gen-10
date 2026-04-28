#!/usr/bin/env python3
"""Hermes AGI Gen の簡易バッチ実行。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_agent import AIAgent
from hermes_agi_gen.toolset_distributions import sample_toolsets_from_distribution, validate_distribution


def main(dataset_file: str, output_file: str = "batch_results.jsonl", run_name: str = "run", distribution: str = "development", max_iterations: int = 6):
    if not validate_distribution(distribution):
        raise ValueError(f"未知の distribution: {distribution}")
    prompts: list[str] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                prompts.append(item.get("prompt") or item.get("task") or line)
            except json.JSONDecodeError:
                prompts.append(line)
    agent = AIAgent(max_iterations=max_iterations)
    selected_toolsets = sample_toolsets_from_distribution(distribution)
    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, start=1):
            result = agent.run_conversation(prompt)
            entry = {
                "run_name": run_name,
                "index": i,
                "prompt": prompt,
                "toolsets_used": selected_toolsets,
                "timestamp": datetime.now().isoformat(),
                "result": result,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"{len(prompts)} 件を書き出しました: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True)
    parser.add_argument('--output_file', default='batch_results.jsonl')
    parser.add_argument('--run_name', default='run')
    parser.add_argument('--distribution', default='development')
    parser.add_argument('--max_iterations', type=int, default=6)
    args = parser.parse_args()
    main(args.dataset_file, args.output_file, args.run_name, args.distribution, args.max_iterations)
