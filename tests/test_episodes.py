import json
import tempfile
import unittest
from pathlib import Path

from jetson_visual_memory.episodes import EpisodeRecord, load_episodes


class EpisodeRecordTest(unittest.TestCase):
    def test_episode_record_normalizes_lists_and_query_text(self):
        record = EpisodeRecord.from_dict(
            {
                "episode_id": "ep-001",
                "image": "data/images/table.jpg",
                "instruction": "pick up the cup",
                "state": {"gripper": "open"},
                "candidate_actions": "move_to_cup",
                "safety_flags": ["low_light"],
            }
        )

        self.assertEqual(record.candidate_actions, ["move_to_cup"])
        self.assertIn("pick up the cup", record.to_query())
        self.assertIn("low_light", record.to_query())

    def test_load_episodes_requires_core_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(json.dumps({"episode_id": "ep-001", "image": "x.jpg"}, ensure_ascii=False), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_episodes(path)


if __name__ == "__main__":
    unittest.main()
