from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.encoders import create_encoder
from jetson_visual_memory.memory_index import MemoryIndex, iter_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual memory index from local images.")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--out", default="artifacts/index")
    parser.add_argument("--encoder", default="auto", choices=["auto", "clip", "clip-vit-b32", "hash"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = iter_image_paths(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {args.image_dir}")
    encoder = create_encoder(args.encoder)
    index = MemoryIndex.from_image_dir(args.image_dir, encoder)
    index.save(args.out)
    print(f"Indexed {len(index.records)} images -> {args.out}")


if __name__ == "__main__":
    main()
