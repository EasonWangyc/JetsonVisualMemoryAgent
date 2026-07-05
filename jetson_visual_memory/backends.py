from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .encoders import ClipEmbeddingEncoder, HashEmbeddingEncoder


class ImageEmbeddingBackend(Protocol):
    name: str

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        ...

    def status(self) -> dict:
        ...


@dataclass
class BackendStatus:
    name: str
    status: str
    reason: str = ""
    artifact: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "artifact": self.artifact,
        }


class HashImageBackend:
    name = "hash"

    def __init__(self):
        self.encoder = HashEmbeddingEncoder()

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        return self.encoder.encode_images(image_paths)

    def status(self) -> dict:
        return BackendStatus(self.name, "ready", "dependency-free smoke backend").to_dict()


class PytorchClipImageBackend:
    name = "pytorch_clip"

    def __init__(self):
        self.encoder = ClipEmbeddingEncoder()

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        return self.encoder.encode_images(image_paths)

    def status(self) -> dict:
        return BackendStatus(self.name, "ready", "Transformers CLIP image encoder").to_dict()


class SkippedImageBackend:
    def __init__(self, name: str, reason: str, artifact: str = ""):
        self.name = name
        self.reason = reason
        self.artifact = artifact

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        raise RuntimeError(f"{self.name} backend is skipped: {self.reason}")

    def status(self) -> dict:
        return BackendStatus(self.name, "skipped", self.reason, self.artifact).to_dict()


def create_image_backend(name: str, artifact: str | Path | None = None) -> ImageEmbeddingBackend:
    normalized = name.lower().replace("-", "_")
    artifact_path = Path(artifact) if artifact else None
    if normalized in {"hash", "mock"}:
        return HashImageBackend()
    if normalized in {"pytorch_clip", "clip", "clip_vit_b32"}:
        try:
            return PytorchClipImageBackend()
        except RuntimeError as exc:
            return SkippedImageBackend("pytorch_clip", str(exc))
    if normalized in {"onnx_clip", "onnx"}:
        if artifact_path is None or not artifact_path.exists():
            return SkippedImageBackend("onnx_clip", "ONNX model artifact is missing", str(artifact_path or ""))
        try:
            import onnxruntime as ort
        except ImportError:
            return SkippedImageBackend("onnx_clip", "onnxruntime is not installed", str(artifact_path))
        return OnnxClipImageBackend(artifact_path, ort)
    if normalized in {"tensorrt_clip", "trt_clip", "tensorrt"}:
        if artifact_path is None or not artifact_path.exists():
            return SkippedImageBackend("tensorrt_clip", "TensorRT engine artifact is missing", str(artifact_path or ""))
        if shutil.which("trtexec") is None:
            return SkippedImageBackend("tensorrt_clip", "trtexec is not available in PATH", str(artifact_path))
        return SkippedImageBackend(
            "tensorrt_clip",
            "TensorRT runtime bindings are not implemented; use build_trt_engine.py for engine generation reports",
            str(artifact_path),
        )
    raise ValueError(f"unknown image backend: {name}")


def probe_image_backend(name: str, artifact: str | Path | None = None) -> dict:
    normalized = name.lower().replace("-", "_")
    artifact_path = Path(artifact) if artifact else None
    if normalized in {"hash", "mock"}:
        return BackendStatus("hash", "ready", "dependency-free smoke backend").to_dict()
    if normalized in {"pytorch_clip", "clip", "clip_vit_b32"}:
        try:
            import torch  # noqa: F401
            import PIL  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            return BackendStatus("pytorch_clip", "skipped", f"dependency probe failed: {exc}").to_dict()
        return BackendStatus("pytorch_clip", "ready", "torch/Pillow/transformers imports available").to_dict()
    if normalized in {"onnx_clip", "onnx"}:
        if artifact_path is None or not artifact_path.exists():
            return BackendStatus("onnx_clip", "skipped", "ONNX model artifact is missing", str(artifact_path or "")).to_dict()
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return BackendStatus("onnx_clip", "skipped", f"dependency probe failed: {exc}", str(artifact_path)).to_dict()
        return BackendStatus("onnx_clip", "ready", "onnxruntime import and model artifact available", str(artifact_path)).to_dict()
    if normalized in {"tensorrt_clip", "trt_clip", "tensorrt"}:
        if artifact_path is None or not artifact_path.exists():
            return BackendStatus("tensorrt_clip", "skipped", "TensorRT engine artifact is missing", str(artifact_path or "")).to_dict()
        if shutil.which("trtexec") is None:
            return BackendStatus("tensorrt_clip", "skipped", "trtexec is not available in PATH", str(artifact_path)).to_dict()
        return BackendStatus("tensorrt_clip", "ready", "trtexec and engine artifact available", str(artifact_path)).to_dict()
    raise ValueError(f"unknown image backend: {name}")


class OnnxClipImageBackend:
    name = "onnx_clip"

    def __init__(self, model_path: Path, ort_module):
        self.model_path = model_path
        self.ort = ort_module
        self.session = ort_module.InferenceSession(str(model_path), providers=ort_module.get_available_providers())

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        raise RuntimeError(
            "onnx_clip image preprocessing is export-specific. Use tools/benchmark_matrix.py to report backend status, "
            "or compare exported engines with a prepared pixel_values tensor."
        )

    def status(self) -> dict:
        providers = ",".join(self.session.get_providers())
        return BackendStatus(self.name, "ready", f"providers={providers}", str(self.model_path)).to_dict()
