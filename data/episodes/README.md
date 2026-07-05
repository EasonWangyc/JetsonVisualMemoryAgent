# VLA-Style Episodes

`sample_episodes.jsonl` uses an offline observation-language-action format for replaying embodied inference cases without a real robot.

Each line is a JSON object:

```json
{
  "episode_id": "ep-001",
  "image": "data/images/table_scene.jpg",
  "instruction": "pick up the cup and avoid the cable",
  "state": {"gripper": "open", "battery": "normal"},
  "candidate_actions": ["describe_scene", "move_to_cup", "ask_human_review"],
  "chosen_action": "ask_human_review",
  "safety_flags": ["cable_near_target"]
}
```

The image path should point to a local image that is also present in the visual memory index. For smoke tests, the `hash` encoder and `mock` VLM can exercise the pipeline without model downloads.
