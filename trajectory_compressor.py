#!/usr/bin/env python3
"""軌跡 JSONL を単純圧縮する小さなツール。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compress_text(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[省略]"


def main(input: str, output: str | None = None, max_chars: int = 400):
    input_path = Path(input)
    output_path = Path(output) if output else input_path.with_name(input_path.stem + "_compressed.jsonl")
    count = 0
    with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "result" in obj and "final_response" in obj["result"]:
                obj["result"]["final_response"] = compress_text(obj["result"]["final_response"], max_chars=max_chars)
            dst.write(json.dumps(obj, ensure_ascii=False) + "
")
            count += 1
    print(f"{count} 件を圧縮して保存しました: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--max_chars', type=int, default=400)
    args = parser.parse_args()
    main(args.input, args.output, args.max_chars)
