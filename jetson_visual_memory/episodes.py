from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class EpisodeRecord:
    episode_id: str
    image: str
    instruction: str
    state: dict[str, Any] = field(default_factory=dict)
    candidate_actions: list[str] = field(default_factory=list)
    chosen_action: str = ""
    safety_flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeRecord":
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            image=str(payload.get("image", "")),
            instruction=str(payload.get("instruction", "")),
            state=payload.get("state", {}) if isinstance(payload.get("state", {}), dict) else {},
            candidate_actions=_as_string_list(payload.get("candidate_actions", [])),
            chosen_action=str(payload.get("chosen_action", "")),
            safety_flags=_as_string_list(payload.get("safety_flags", [])),
        )

    def to_query(self) -> str:
        candidates = ", ".join(self.candidate_actions) if self.candidate_actions else "none"
        flags = ", ".join(self.safety_flags) if self.safety_flags else "none"
        return (
            f"instruction: {self.instruction}\n"
            f"state: {json.dumps(self.state, ensure_ascii=False, sort_keys=True)}\n"
            f"candidate_actions: {candidates}\n"
            f"safety_flags: {flags}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "image": self.image,
            "instruction": self.instruction,
            "state": self.state,
            "candidate_actions": self.candidate_actions,
            "chosen_action": self.chosen_action,
            "safety_flags": self.safety_flags,
        }


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def load_episodes(path: str | Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"episode line {line_no} is not a JSON object")
            record = EpisodeRecord.from_dict(payload)
            if not record.episode_id or not record.image or not record.instruction:
                raise ValueError(f"episode line {line_no} must include episode_id, image, and instruction")
            records.append(record)
    return records


def iter_episode_dicts(records: Iterable[EpisodeRecord]) -> Iterable[dict[str, Any]]:
    for record in records:
        yield record.to_dict()
