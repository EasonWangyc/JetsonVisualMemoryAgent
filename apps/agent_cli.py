from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.encoders import create_encoder
from jetson_visual_memory.memory_index import MemoryIndex
from jetson_visual_memory.schema import validate_agent_result
from jetson_visual_memory.vlm import build_json_prompt, create_vlm


ALLOWED_ACTIONS = {"describe_scene", "rank_images", "find_risky_cases", "suggest_robot_action"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual memory agent: retrieve images and produce action JSON.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="artifacts/index")
    parser.add_argument("--encoder", default="auto", choices=["auto", "clip", "clip-vit-b32", "hash"])
    parser.add_argument("--model", default="mock", choices=["mock", "auto", "smolvlm", "smolvlm-500m"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--action", default="suggest_robot_action", choices=sorted(ALLOWED_ACTIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = create_encoder(args.encoder)
    index = MemoryIndex.load(args.index)
    matches = index.query(args.query, encoder, top_k=args.top_k)
    if not matches:
        print(json.dumps({"query": args.query, "matched_images": [], "risk_level": "unknown"}, ensure_ascii=False))
        return

    vlm = create_vlm(args.model)
    prompt = build_json_prompt(args.query, task="robot_action")
    response = vlm.analyze(matches[0]["path"], prompt, query=args.query, matched_images=matches)
    payload = validate_agent_result(response.parsed)
    payload["action_type"] = args.action
    payload["allowed_actions"] = sorted(ALLOWED_ACTIONS)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
