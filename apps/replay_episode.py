from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.encoders import create_encoder
from jetson_visual_memory.episodes import load_episodes
from jetson_visual_memory.memory_index import MemoryIndex
from jetson_visual_memory.schema import validate_agent_result
from jetson_visual_memory.vlm import build_json_prompt, create_vlm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay VLA-style episodes through visual memory and VLM action JSON.")
    parser.add_argument("--episodes", default="data/episodes/sample_episodes.jsonl")
    parser.add_argument("--index", default="artifacts/index")
    parser.add_argument("--encoder", default="hash", choices=["auto", "clip", "clip-vit-b32", "hash"])
    parser.add_argument("--model", default="mock", choices=["mock", "auto", "smolvlm", "smolvlm-500m"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = load_episodes(args.episodes)
    if args.limit > 0:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit(f"No episodes found in {args.episodes}")

    index = MemoryIndex.load(args.index)
    encoder = create_encoder(args.encoder)
    vlm = create_vlm(args.model)
    outputs = []
    for episode in episodes:
        query = episode.to_query()
        matches = index.query(query, encoder, top_k=args.top_k)
        image = matches[0]["path"] if matches else episode.image
        prompt = build_json_prompt(query, task="robot_action")
        response = vlm.analyze(image, prompt, query=query, matched_images=matches)
        payload = validate_agent_result(response.parsed)
        payload["episode_id"] = episode.episode_id
        payload["instruction"] = episode.instruction
        payload["candidate_actions"] = episode.candidate_actions
        payload["chosen_action"] = episode.chosen_action
        payload["safety_flags"] = episode.safety_flags
        outputs.append(payload)

    text = json.dumps(outputs, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
