from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.encoders import create_encoder
from jetson_visual_memory.memory_index import MemoryIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the visual memory index with natural language.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="artifacts/index")
    parser.add_argument("--encoder", default="auto", choices=["auto", "clip", "clip-vit-b32", "hash"])
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = create_encoder(args.encoder)
    index = MemoryIndex.load(args.index)
    matches = index.query(args.query, encoder, top_k=args.top_k)
    print(json.dumps({"query": args.query, "matched_images": matches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
