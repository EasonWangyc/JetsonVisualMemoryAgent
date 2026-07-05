from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


RISK_LEVELS = {"low", "medium", "high", "unknown"}


@dataclass
class AgentResult:
    query: str
    matched_images: list[dict[str, Any]]
    scene_summary: str
    risk_level: str = "unknown"
    suggested_action: str = ""
    failure_modes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    action_type: str = "suggest_robot_action"
    target_object: str = ""
    preconditions: list[str] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)
    reasoning_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matched_images": self.matched_images,
            "scene_summary": self.scene_summary,
            "risk_level": self.risk_level,
            "suggested_action": self.suggested_action,
            "failure_modes": self.failure_modes,
            "confidence": self.confidence,
            "action_type": self.action_type,
            "target_object": self.target_object,
            "preconditions": self.preconditions,
            "safety_constraints": self.safety_constraints,
            "reasoning_summary": self.reasoning_summary,
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def validate_agent_result(payload: dict[str, Any]) -> dict[str, Any]:
    risk_level = str(payload.get("risk_level", "unknown")).lower()
    if risk_level not in RISK_LEVELS:
        risk_level = "unknown"

    matched_images = payload.get("matched_images", [])
    if not isinstance(matched_images, list):
        matched_images = []

    normalized_matches = []
    for item in matched_images:
        if not isinstance(item, dict):
            continue
        normalized_matches.append(
            {
                "path": str(item.get("path", "")),
                "score": _clamp(_as_float(item.get("score", 0.0))),
                "reason": str(item.get("reason", "")),
            }
        )

    return {
        "query": str(payload.get("query", "")),
        "matched_images": normalized_matches,
        "scene_summary": str(payload.get("scene_summary", "")),
        "risk_level": risk_level,
        "suggested_action": str(payload.get("suggested_action", "")),
        "failure_modes": _as_string_list(payload.get("failure_modes", [])),
        "confidence": _clamp(_as_float(payload.get("confidence", 0.0))),
        "action_type": str(payload.get("action_type", "suggest_robot_action")),
        "target_object": str(payload.get("target_object", "")),
        "preconditions": _as_string_list(payload.get("preconditions", [])),
        "safety_constraints": _as_string_list(payload.get("safety_constraints", [])),
        "reasoning_summary": str(payload.get("reasoning_summary", "")),
    }


def fallback_agent_result(query: str, reason: str, matched_images: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return AgentResult(
        query=query,
        matched_images=matched_images or [],
        scene_summary="模型输出不可解析，已返回保守结果。",
        risk_level="unknown",
        suggested_action="请人工复核该场景。",
        failure_modes=[reason],
        confidence=0.0,
    ).to_dict()


def parse_agent_json(text: str, query: str = "", matched_images: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return fallback_agent_result(query=query, reason="invalid_json", matched_images=matched_images)
    if not isinstance(payload, dict):
        return fallback_agent_result(query=query, reason="non_object_json", matched_images=matched_images)
    payload.setdefault("query", query)
    if matched_images is not None:
        payload.setdefault("matched_images", matched_images)
    return validate_agent_result(payload)
