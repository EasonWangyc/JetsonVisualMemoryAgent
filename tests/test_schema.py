import json
import unittest

from jetson_visual_memory.schema import (
    AgentResult,
    parse_agent_json,
    validate_agent_result,
)


class AgentResultSchemaTest(unittest.TestCase):
    def test_valid_result_round_trips_with_known_risk_level(self):
        result = AgentResult(
            query="哪些场景适合抓取？",
            matched_images=[{"path": "data/images/001.jpg", "score": 0.82, "reason": "桌面清晰"}],
            scene_summary="桌面上有一个杯子。",
            risk_level="low",
            suggested_action="可以尝试抓取杯身中部。",
            failure_modes=[],
            confidence=0.73,
        )

        payload = validate_agent_result(result.to_dict())

        self.assertEqual(payload["risk_level"], "low")
        self.assertEqual(payload["confidence"], 0.73)
        self.assertEqual(payload["matched_images"][0]["path"], "data/images/001.jpg")

    def test_invalid_json_falls_back_to_unknown_result(self):
        payload = parse_agent_json("not json", query="查询")

        self.assertEqual(payload["query"], "查询")
        self.assertEqual(payload["risk_level"], "unknown")
        self.assertIn("invalid_json", payload["failure_modes"])

    def test_json_with_bad_risk_is_normalized(self):
        raw = json.dumps(
            {
                "query": "q",
                "matched_images": [],
                "scene_summary": "summary",
                "risk_level": "danger",
                "suggested_action": "act",
                "failure_modes": ["遮挡"],
                "confidence": 2.0,
            },
            ensure_ascii=False,
        )

        payload = parse_agent_json(raw, query="q")

        self.assertEqual(payload["risk_level"], "unknown")
        self.assertEqual(payload["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
