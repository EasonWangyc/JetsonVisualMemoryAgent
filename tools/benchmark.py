from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.encoders import create_encoder
from jetson_visual_memory.memory_index import MemoryIndex, iter_image_paths
from jetson_visual_memory.metrics import (
    append_jsonl,
    current_memory_mb,
    load_jsonl,
    summarize_benchmark_rows,
    timed_ms,
    write_markdown_summary,
)
from jetson_visual_memory.schema import validate_agent_result
from jetson_visual_memory.vlm import build_json_prompt, create_vlm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark visual memory and VLM analysis.")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--index", default="artifacts/index")
    parser.add_argument("--encoder", default="auto", choices=["auto", "clip", "clip-vit-b32", "hash"])
    parser.add_argument("--model", default="mock", choices=["mock", "auto", "smolvlm", "smolvlm-500m"])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--query", default="哪些场景适合机器人操作？")
    parser.add_argument("--out", default="reports/benchmark.jsonl")
    parser.add_argument("--summary", default="reports/benchmark.md")
    parser.add_argument("--clip-only", action="store_true")
    return parser.parse_args()


def _ensure_index(args: argparse.Namespace, encoder) -> MemoryIndex:
    index_dir = Path(args.index)
    if (index_dir / "embeddings.npy").exists() and (index_dir / "metadata.jsonl").exists():
        return MemoryIndex.load(index_dir)
    index = MemoryIndex.from_image_dir(args.image_dir, encoder)
    index.save(index_dir)
    return index


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        out.unlink()

    encoder = create_encoder(args.encoder)
    index = _ensure_index(args, encoder)
    vlm = None if args.clip_only else create_vlm(args.model)
    image_paths = [str(path) for path in iter_image_paths(args.image_dir)]
    if not image_paths:
        raise SystemExit(f"No images found under {args.image_dir}")

    for run_id in range(args.runs):
        image = image_paths[run_id % len(image_paths)]
        row = {
            "run_id": run_id,
            "image": image,
            "model": "clip-only" if args.clip_only else getattr(vlm, "model_name", args.model),
            "vision_encode_ms": 0.0,
            "retrieval_ms": 0.0,
            "vlm_generate_ms": 0.0,
            "json_parse_ms": 0.0,
            "total_ms": 0.0,
            "peak_mem_mb": 0.0,
            "output_tokens": 0,
            "tokens_per_s": 0.0,
        }
        start_total = time.perf_counter()
        with timed_ms(row, "vision_encode_ms"):
            encoder.encode_images([image])
        with timed_ms(row, "retrieval_ms"):
            matches = index.query(args.query, encoder, top_k=args.top_k)
        if vlm is not None and matches:
            prompt = build_json_prompt(args.query)
            response = vlm.analyze(matches[0]["path"], prompt, query=args.query, matched_images=matches)
            row["vlm_generate_ms"] = response.elapsed_ms
            parse_start = time.perf_counter()
            validate_agent_result(response.parsed)
            row["json_parse_ms"] = (time.perf_counter() - parse_start) * 1000.0
            row["output_tokens"] = response.output_tokens
            if response.elapsed_ms > 0:
                row["tokens_per_s"] = response.output_tokens / (response.elapsed_ms / 1000.0)
        row["total_ms"] = (time.perf_counter() - start_total) * 1000.0
        row["peak_mem_mb"] = current_memory_mb()
        append_jsonl(out, row)

    rows = load_jsonl(out)
    summary = summarize_benchmark_rows(rows)
    write_markdown_summary(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
