#!/usr/bin/env python3
"""Hermes AGI Gen 汎用ランナー。あらゆるドメインのタスクを実行できる。"""
from __future__ import annotations

import argparse
from pathlib import Path

from hermes_agi_gen import AgentState, HermesAgentV9
from hermes_agi_gen.hermes_constants import DOMAIN_CONFIG
from hermes_agi_gen.mistral_client import MistralClient


def main(
    query: str,
    repo_root: str = ".",
    model: str = "local/mock-model",
    max_turns: int = 8,
    domain: str = "general",
    context: str = "",
) -> None:
    print(f"Hermes AGI Gen  |  domain={domain}  |  model={model}")
    print("=" * 60)
    print(f"目標: {query}")
    if context:
        print(f"コンテキスト: {context}")
    print("=" * 60)

    if model == "local/mock-model":
        llm = None
    else:
        try:
            llm = MistralClient(model=model)
            print(f"LLM: {llm.base_url}  model={llm.model}")
        except ValueError as e:
            print(f"\n[設定エラー] {e}")
            return
    agent = HermesAgentV9(
        repo_root=Path(repo_root),
        model=model,
        max_iterations=max_turns,
        llm=llm,
    )

    cfg = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG["general"])
    state = AgentState(
        user_goal=query,
        domain=domain,
        context=context,
        success_criteria=cfg["success_criteria"],
        constraints=cfg["constraints"],
        max_iterations=max_turns,
    )

    final_state = agent.run(state)
    print(agent.render_progress(final_state))

    if final_state.suggested_next_goal:
        print(f"\n[次の推奨ゴール] {final_state.suggested_next_goal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermes AGI Gen 汎用ランナー")
    parser.add_argument("--query", "-q", default="このディレクトリの構造を調べて概要を教えてください。", help="実行する目標・タスク")
    parser.add_argument("--domain", "-d", default="general", choices=["general", "coding", "research", "writing", "data", "ops"], help="タスクドメイン")
    parser.add_argument("--context", "-c", default="", help="追加コンテキスト (背景情報)")
    parser.add_argument("--repo_root", default=".", help="作業ディレクトリ")
    parser.add_argument("--model", default="local/mock-model", help="LLM モデル名")
    parser.add_argument("--max_turns", type=int, default=8, help="最大イテレーション数")
    args = parser.parse_args()
    main(args.query, args.repo_root, args.model, args.max_turns, args.domain, args.context)
