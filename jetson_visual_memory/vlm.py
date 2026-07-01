from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import parse_agent_json, validate_agent_result


PROMPT_TEMPLATES = {
    "scene": "请描述这张图片中的主要场景，并输出 JSON。",
    "grasp": "判断这张图片是否适合机器人抓取，说明风险，并输出 JSON。",
    "anomaly": "找出这张图片中可能影响机器人操作的异常情况，并输出 JSON。",
    "compare": "比较候选图片和用户问题的相关性，并输出 JSON。",
    "robot_action": "基于图片内容给出机器人下一步动作建议，并输出 JSON。",
}


def build_json_prompt(query: str, task: str = "robot_action") -> str:
    template = PROMPT_TEMPLATES.get(task, PROMPT_TEMPLATES["robot_action"])
    return (
        f"{template}\n"
        f"用户问题: {query}\n"
        "输出必须是 JSON object，字段为 query, matched_images, scene_summary, "
        "risk_level, suggested_action, failure_modes, confidence。"
    )


@dataclass
class VLMResponse:
    raw_text: str
    parsed: dict[str, Any]
    output_tokens: int
    elapsed_ms: float


class MockVLMAnalyzer:
    """Safe fallback that preserves the final API without model downloads."""

    model_name = "mock-vlm"

    def analyze(
        self,
        image_path: str,
        prompt: str,
        query: str = "",
        matched_images: list[dict[str, Any]] | None = None,
    ) -> VLMResponse:
        start = time.perf_counter()
        stem = Path(image_path).stem.replace("_", " ")
        risk_level = "medium" if any(word in stem.lower() for word in ["wire", "crowd", "dark", "risk"]) else "unknown"
        payload = {
            "query": query,
            "matched_images": matched_images or [{"path": image_path, "score": 1.0, "reason": "direct_image"}],
            "scene_summary": f"Mock analysis for image '{stem}'. Install SmolVLM for real visual understanding.",
            "risk_level": risk_level,
            "suggested_action": "Use this mock result only for pipeline smoke tests.",
            "failure_modes": ["mock_backend"],
            "confidence": 0.1,
        }
        raw = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return VLMResponse(raw_text=raw, parsed=validate_agent_result(payload), output_tokens=len(raw.split()), elapsed_ms=elapsed_ms)


class SmolVLMAnalyzer:
    model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"

    def __init__(self, model_name: str | None = None, device: str | None = None, max_new_tokens: int = 256):
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("SmolVLM backend requires torch, pillow, and transformers.") from exc
        self.torch = torch
        self.image_cls = Image
        self.processor = AutoProcessor.from_pretrained(model_name or self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name or self.model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def analyze(
        self,
        image_path: str,
        prompt: str,
        query: str = "",
        matched_images: list[dict[str, Any]] | None = None,
    ) -> VLMResponse:
        start = time.perf_counter()
        image = self.image_cls.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        parsed = parse_agent_json(raw, query=query, matched_images=matched_images)
        return VLMResponse(raw_text=raw, parsed=parsed, output_tokens=len(raw.split()), elapsed_ms=elapsed_ms)


def create_vlm(name: str = "mock"):
    normalized = name.lower()
    if normalized == "mock":
        return MockVLMAnalyzer()
    if normalized in {"smolvlm", "smolvlm-500m"}:
        return SmolVLMAnalyzer()
    if normalized == "auto":
        try:
            return SmolVLMAnalyzer()
        except RuntimeError:
            return MockVLMAnalyzer()
    raise ValueError(f"unknown VLM backend: {name}")
