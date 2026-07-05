from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.backends import probe_image_backend
from jetson_visual_memory.encoders import create_encoder
from jetson_visual_memory.episodes import load_episodes
from jetson_visual_memory.memory_index import MemoryIndex, iter_image_paths
from jetson_visual_memory.metrics import append_jsonl, current_memory_mb, load_jsonl, summarize_benchmark_rows
from jetson_visual_memory.vlm import build_json_prompt, create_vlm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix for retrieval, VLM reading, and episode replay.")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--index", default="artifacts/index")
    parser.add_argument("--episodes", default="data/episodes/sample_episodes.jsonl")
    parser.add_argument("--encoder", default="hash", choices=["auto", "clip", "clip-vit-b32", "hash"])
    parser.add_argument("--model", default="mock", choices=["mock", "auto", "smolvlm", "smolvlm-500m"])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--query", default="哪些场景适合机器人安全操作？")
    parser.add_argument("--out", default="reports/benchmark_matrix.jsonl")
    parser.add_argument("--summary", default="reports/benchmark_matrix.md")
    parser.add_argument("--onnx", default="artifacts/onnx/clip_image_encoder.onnx")
    parser.add_argument("--trt-engine", default="artifacts/trt/clip_image_encoder.engine")
    return parser.parse_args()


def tegrastats_snapshot() -> dict:
    if shutil.which("tegrastats") is None:
        return {"status": "not_available", "reason": "tegrastats not found"}
    command = ["tegrastats", "--interval", "100"]
    if shutil.which("timeout") is not None:
        command = ["timeout", "1s", *command]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = completed.stdout.strip()
    if "RAM" not in output and "CPU" not in output:
        return {"status": "not_available", "reason": output or f"tegrastats exited with {completed.returncode}"}
    return {"status": "ready", "sample": output.splitlines()[0]}


def ensure_index(index_dir: str, image_dir: str, encoder) -> MemoryIndex:
    root = Path(index_dir)
    if (root / "embeddings.npy").exists() and (root / "metadata.jsonl").exists():
        return MemoryIndex.load(root)
    index = MemoryIndex.from_image_dir(image_dir, encoder)
    index.save(root)
    return index


def make_row(scenario: str, run_id: int, model: str) -> dict:
    return {
        "scenario": scenario,
        "run_id": run_id,
        "model": model,
        "vision_encode_ms": 0.0,
        "retrieval_ms": 0.0,
        "vlm_generate_ms": 0.0,
        "json_parse_ms": 0.0,
        "total_ms": 0.0,
        "peak_mem_mb": 0.0,
        "output_tokens": 0,
        "tokens_per_s": 0.0,
        "status": "ready",
        "failure": "",
    }


def run_matrix(args: argparse.Namespace) -> list[dict]:
    out = Path(args.out)
    if out.exists():
        out.unlink()
    encoder = create_encoder(args.encoder)
    index = ensure_index(args.index, args.image_dir, encoder)
    vlm = create_vlm(args.model)
    image_paths = [str(path) for path in iter_image_paths(args.image_dir)]
    if not image_paths:
        raise SystemExit(f"No images found under {args.image_dir}")

    backend_reports = [
        probe_image_backend("pytorch_clip"),
        probe_image_backend("onnx_clip", args.onnx),
        probe_image_backend("tensorrt_clip", args.trt_engine),
    ]
    for backend_report in backend_reports:
        append_jsonl(out, {"scenario": "backend_status", **backend_report})

    scenarios = [("clip_only", 0), ("clip_vlm_top1", 1), ("clip_vlm_top3", 3)]
    episodes = []
    episode_path = Path(args.episodes)
    if episode_path.exists():
        episodes = load_episodes(episode_path)
        scenarios.append(("episode_replay", 3))

    for scenario, top_k in scenarios:
        for run_id in range(args.runs):
            row = make_row(scenario, run_id, "clip-only" if top_k == 0 else getattr(vlm, "model_name", args.model))
            start = time.perf_counter()
            image = image_paths[run_id % len(image_paths)]
            query = args.query
            if scenario == "episode_replay" and episodes:
                episode = episodes[run_id % len(episodes)]
                query = episode.to_query()
                row["episode_id"] = episode.episode_id
            try:
                t0 = time.perf_counter()
                encoder.encode_images([image])
                row["vision_encode_ms"] = (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                matches = index.query(query, encoder, top_k=max(1, top_k))
                row["retrieval_ms"] = (time.perf_counter() - t0) * 1000.0
                if top_k > 0 and matches:
                    prompt = build_json_prompt(query, task="robot_action")
                    response = vlm.analyze(matches[0]["path"], prompt, query=query, matched_images=matches)
                    row["vlm_generate_ms"] = response.elapsed_ms
                    row["output_tokens"] = response.output_tokens
                    if response.elapsed_ms > 0:
                        row["tokens_per_s"] = response.output_tokens / (response.elapsed_ms / 1000.0)
                row["total_ms"] = (time.perf_counter() - start) * 1000.0
                row["peak_mem_mb"] = current_memory_mb()
            except Exception as exc:  # benchmark should preserve failure rows for reports
                row["status"] = "failed"
                row["failure"] = str(exc)
            append_jsonl(out, row)
    return load_jsonl(out)


def write_matrix_summary(path: str | Path, rows: list[dict], tegra: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Benchmark Matrix", "", "## Tegrastats", "", f"```json\n{json.dumps(tegra, ensure_ascii=False, indent=2)}\n```", ""]
    backend_rows = [row for row in rows if row.get("scenario") == "backend_status"]
    lines.extend(["## Backend Status", ""])
    for row in backend_rows:
        lines.append(f"- {row.get('name')}: {row.get('status')} ({row.get('reason', '')})")
    lines.append("")
    for scenario in sorted({row.get("scenario") for row in rows if row.get("total_ms") is not None and row.get("scenario") != "backend_status"}):
        scenario_rows = [row for row in rows if row.get("scenario") == scenario and row.get("status") == "ready"]
        summary = summarize_benchmark_rows(scenario_rows)
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"- Count: {summary['count']}",
                f"- Total latency median: {summary['total_ms']['median']:.2f} ms",
                f"- Total latency p90: {summary['total_ms']['p90']:.2f} ms",
                f"- Total latency p99: {summary['total_ms']['p99']:.2f} ms",
                f"- Peak memory max: {summary['peak_mem_mb']['max']:.2f} MB",
                f"- Tokens/s median: {summary['tokens_per_s']['median']:.2f}",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = run_matrix(args)
    tegra = tegrastats_snapshot()
    write_matrix_summary(args.summary, rows, tegra)
    print(json.dumps({"rows": len(rows), "summary": args.summary, "tegrastats": tegra}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
