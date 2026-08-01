import unittest
from pathlib import Path


SPACE = (
    Path(__file__).parent
    / "huggingface"
    / "spaces"
    / "clark-swin2sr-lightweight-x2-1.58bit"
)
G3_REVISION = "b357304af178a6e31a5482958f1adf702f99be82"


class HuggingFaceSpaceWorkerTest(unittest.TestCase):
    def test_inference_is_off_the_page_thread(self):
        app = (SPACE / "app.js").read_text()
        worker = (SPACE / "inference-worker.js").read_text()

        self.assertIn('new Worker(', app)
        self.assertIn(
            'new URL("./inference-worker.js?v=20260801-worker1", import.meta.url)',
            app,
        )
        self.assertNotIn("InferenceSession.create", app)
        self.assertIn("InferenceSession.create", worker)
        self.assertIn('executionProviders: ["webgpu"]', worker)
        self.assertIn('executionProviders: ["wasm"]', worker)

    def test_worker_is_pinned_and_transfers_tile_buffers(self):
        app = (SPACE / "app.js").read_text()
        worker = (SPACE / "inference-worker.js").read_text()

        self.assertIn(G3_REVISION, worker)
        self.assertIn('[pixels.buffer]', app)
        self.assertIn('[result.output]', worker)
        self.assertIn("controls remain available", app)
        self.assertIn("180_000", app)


if __name__ == "__main__":
    unittest.main()
