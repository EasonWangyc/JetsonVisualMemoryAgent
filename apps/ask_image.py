from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.vlm import build_json_prompt, create_vlm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a VLM about one image and request structured JSON.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="描述风险并输出JSON")
    parser.add_argument("--query", default="")
    parser.add_argument("--model", default="mock", choices=["mock", "auto", "smolvlm", "smolvlm-500m"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vlm = create_vlm(args.model)
    prompt = build_json_prompt(args.prompt)
    response = vlm.analyze(args.image, prompt, query=args.query or args.prompt)
    print(json.dumps(response.parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
