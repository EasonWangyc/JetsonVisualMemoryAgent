from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EmbeddingEncoder(Protocol):
    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        ...

    def encode_text(self, text: str) -> np.ndarray:
        ...


@dataclass
class ImageRecord:
    path: str
    source: str = "local"
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ImageRecord":
        return cls(
            path=str(payload.get("path", "")),
            source=str(payload.get("source", "local")),
            tags=[str(item) for item in payload.get("tags", [])],
            created_at=str(payload.get("created_at", "")),
        )


def iter_image_paths(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@dataclass
class MemoryIndex:
    records: list[ImageRecord]
    embeddings: np.ndarray

    @classmethod
    def from_records(cls, records: list[ImageRecord], encoder: EmbeddingEncoder) -> "MemoryIndex":
        image_paths = [record.path for record in records]
        embeddings = encoder.encode_images(image_paths)
        return cls(records=records, embeddings=normalize_embeddings(embeddings))

    @classmethod
    def from_image_dir(cls, image_dir: str | Path, encoder: EmbeddingEncoder) -> "MemoryIndex":
        records = [ImageRecord(path=str(path.as_posix()), source="local") for path in iter_image_paths(image_dir)]
        return cls.from_records(records, encoder)

    def query(self, text: str, encoder: EmbeddingEncoder, top_k: int = 5) -> list[dict]:
        if not self.records:
            return []
        text_embedding = normalize_embeddings(encoder.encode_text(text))[0]
        scores = self.embeddings @ text_embedding
        top_indices = np.argsort(scores)[::-1][: max(1, top_k)]
        matches = []
        for idx in top_indices:
            record = self.records[int(idx)]
            matches.append(
                {
                    "path": record.path,
                    "score": float(scores[int(idx)]),
                    "reason": "embedding_similarity",
                    "source": record.source,
                    "tags": record.tags,
                }
            )
        return matches

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings)
        with (out / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: str | Path) -> "MemoryIndex":
        root = Path(index_dir)
        embeddings = np.load(root / "embeddings.npy")
        records = []
        with (root / "metadata.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(ImageRecord.from_dict(json.loads(line)))
        return cls(records=records, embeddings=normalize_embeddings(embeddings))
