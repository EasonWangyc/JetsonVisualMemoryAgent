import unittest
from pathlib import Path
import shutil

import numpy as np

from jetson_visual_memory.memory_index import ImageRecord, MemoryIndex


class FakeEncoder:
    def encode_images(self, image_paths):
        rows = []
        for path in image_paths:
            name = Path(path).stem
            if name == "robot":
                rows.append([1.0, 0.0, 0.0])
            elif name == "desk":
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)

    def encode_text(self, text):
        if "robot" in text.lower() or "机器人" in text:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


class MemoryIndexTest(unittest.TestCase):
    def test_query_returns_most_similar_images(self):
        records = [
            ImageRecord(path="data/images/robot.jpg", source="test"),
            ImageRecord(path="data/images/desk.jpg", source="test"),
        ]
        index = MemoryIndex.from_records(records, FakeEncoder())

        matches = index.query("robot arm scene", FakeEncoder(), top_k=1)

        self.assertEqual(matches[0]["path"], "data/images/robot.jpg")
        self.assertGreater(matches[0]["score"], 0.99)

    def test_index_save_and_load_preserves_records(self):
        records = [ImageRecord(path="data/images/desk.jpg", source="test", tags=["desktop"])]
        index = MemoryIndex.from_records(records, FakeEncoder())

        tmp = Path("artifacts/test_tmp/index_case")
        if tmp.exists():
            shutil.rmtree(tmp)
        out = tmp / "index"
        index.save(out)
        loaded = MemoryIndex.load(out)
        shutil.rmtree(tmp)

        self.assertEqual(loaded.records[0].path, "data/images/desk.jpg")
        self.assertEqual(loaded.records[0].tags, ["desktop"])
        self.assertEqual(loaded.embeddings.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
