import unittest

from jetson_visual_memory.metrics import summarize_benchmark_rows


class MetricsTest(unittest.TestCase):
    def test_summarize_benchmark_rows_reports_latency_and_tokens(self):
        rows = [
            {"total_ms": 10.0, "tokens_per_s": 5.0, "peak_mem_mb": 100.0},
            {"total_ms": 20.0, "tokens_per_s": 10.0, "peak_mem_mb": 120.0},
            {"total_ms": 30.0, "tokens_per_s": 15.0, "peak_mem_mb": 140.0},
        ]

        summary = summarize_benchmark_rows(rows)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["total_ms"]["median"], 20.0)
        self.assertEqual(summary["peak_mem_mb"]["max"], 140.0)
        self.assertEqual(summary["tokens_per_s"]["median"], 10.0)


if __name__ == "__main__":
    unittest.main()
