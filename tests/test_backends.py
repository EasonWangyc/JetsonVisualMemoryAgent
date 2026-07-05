import unittest

from jetson_visual_memory.backends import HashImageBackend, create_image_backend, probe_image_backend


class BackendSmokeTest(unittest.TestCase):
    def test_hash_backend_is_ready_and_encodes(self):
        backend = create_image_backend("hash")

        self.assertIsInstance(backend, HashImageBackend)
        self.assertEqual(backend.status()["status"], "ready")
        self.assertEqual(backend.encode_images(["data/images/cup.jpg"]).shape[0], 1)

    def test_missing_onnx_backend_is_skipped(self):
        backend = create_image_backend("onnx_clip", artifact="artifacts/missing.onnx")

        self.assertEqual(backend.status()["status"], "skipped")
        self.assertIn("missing", backend.status()["reason"])

    def test_missing_tensorrt_backend_is_skipped(self):
        backend = create_image_backend("tensorrt_clip", artifact="artifacts/missing.engine")

        self.assertEqual(backend.status()["status"], "skipped")
        self.assertIn("missing", backend.status()["reason"])

    def test_probe_does_not_require_loading_clip_weights(self):
        status = probe_image_backend("onnx_clip", artifact="artifacts/missing.onnx")

        self.assertEqual(status["name"], "onnx_clip")
        self.assertEqual(status["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
