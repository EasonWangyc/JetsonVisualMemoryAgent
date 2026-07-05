from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX CLIP image encoder.")
    parser.add_argument("--onnx", default="artifacts/onnx/clip_image_encoder.onnx")
    parser.add_argument("--engine", default="artifacts/trt/clip_image_encoder.engine")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    parser.add_argument("--report", default="reports/build_trt_engine.json")
    return parser.parse_args()


def write_report(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = {
        "tool": "build_trt_engine",
        "onnx": args.onnx,
        "engine": args.engine,
        "status": "skipped",
        "reason": "",
        "command": [],
    }
    onnx = Path(args.onnx)
    if not onnx.exists():
        report["reason"] = "ONNX model artifact is missing"
        write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        report["reason"] = "trtexec is not available in PATH"
        write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    engine = Path(args.engine)
    engine.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--memPoolSize=workspace:{args.workspace_mb}",
    ]
    if args.fp16:
        command.append("--fp16")
    report["command"] = command
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    report["trtexec_output_tail"] = completed.stdout[-4000:]
    if completed.returncode == 0 and engine.exists():
        report.update({"status": "ready", "reason": "engine_built", "bytes": engine.stat().st_size})
    else:
        report.update({"status": "failed", "reason": f"trtexec exited with {completed.returncode}"})
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
