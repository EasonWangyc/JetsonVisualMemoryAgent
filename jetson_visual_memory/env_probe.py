from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or result.stderr).strip()


def collect_env() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "jetson_release": "",
        "cuda_nvcc": "",
        "tensorrt": "",
        "torch": {"installed": False, "version": "", "cuda_available": False},
        "disk_root": {},
        "memory": {},
    }
    release = Path("/etc/nv_tegra_release")
    if release.exists():
        info["jetson_release"] = release.read_text(encoding="utf-8", errors="ignore").strip()
    info["cuda_nvcc"] = _run(["nvcc", "--version"])
    info["tensorrt"] = _run(["dpkg-query", "-W", "tensorrt", "libnvinfer10", "python3-libnvinfer"])
    try:
        import torch

        info["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:
        info["torch"] = {
            "installed": False,
            "version": "",
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    disk = shutil.disk_usage("/")
    info["disk_root"] = {"total_gb": disk.total / 1e9, "used_gb": disk.used / 1e9, "free_gb": disk.free / 1e9}
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["memory"] = {"total_gb": mem.total / 1e9, "available_gb": mem.available / 1e9}
    except ImportError:
        info["memory"] = {"note": "psutil not installed"}
    return info


def format_env_json() -> str:
    return json.dumps(collect_env(), ensure_ascii=False, indent=2)
