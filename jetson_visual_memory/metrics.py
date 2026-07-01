from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def timed_ms(target: dict, key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        target[key] = (time.perf_counter() - start) * 1000.0


def current_memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except ImportError:
        return 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[index])


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": float(statistics.median(values)),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_benchmark_rows(rows: list[dict]) -> dict:
    fields = ["total_ms", "vision_encode_ms", "retrieval_ms", "vlm_generate_ms", "json_parse_ms", "tokens_per_s"]
    summary = {"count": len(rows)}
    for field in fields:
        summary[field] = summarize_numeric([float(row.get(field, 0.0)) for row in rows])
    summary["peak_mem_mb"] = {
        "max": max([float(row.get("peak_mem_mb", 0.0)) for row in rows], default=0.0)
    }
    return summary


def append_jsonl(path: str | Path, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_markdown_summary(path: str | Path, summary: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Summary",
        "",
        f"- Count: {summary.get('count', 0)}",
        f"- Total latency median: {summary['total_ms']['median']:.2f} ms",
        f"- Total latency p90: {summary['total_ms']['p90']:.2f} ms",
        f"- Total latency p99: {summary['total_ms']['p99']:.2f} ms",
        f"- Tokens/s median: {summary['tokens_per_s']['median']:.2f}",
        f"- Peak memory max: {summary['peak_mem_mb']['max']:.2f} MB",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
