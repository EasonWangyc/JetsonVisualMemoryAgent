from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class HashEmbeddingEncoder:
    """Dependency-free encoder for smoke tests and offline development."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, value: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = value.lower().replace("\\", "/").replace("_", " ").replace("-", " ").split()
        if not tokens:
            tokens = [value.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset, byte in enumerate(digest[:16]):
                vector[(byte + offset * 17) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        return np.asarray([self._embed(Path(path).stem) for path in image_paths], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self._embed(text)


class ClipEmbeddingEncoder:
    """CLIP ViT encoder loaded lazily so the project can run without ML deps."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP backend requires torch, pillow, and transformers. "
                "Use --encoder hash for a dependency-free smoke test."
            ) from exc
        self.torch = torch
        self.image_cls = Image
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        images = [self.image_cls.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy()[0].astype(np.float32)


def create_encoder(name: str = "hash"):
    normalized = name.lower()
    if normalized == "hash":
        return HashEmbeddingEncoder()
    if normalized in {"clip", "clip-vit-b32"}:
        return ClipEmbeddingEncoder()
    if normalized == "auto":
        try:
            return ClipEmbeddingEncoder()
        except RuntimeError:
            return HashEmbeddingEncoder()
    raise ValueError(f"unknown encoder: {name}")
