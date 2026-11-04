import asyncio
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from evolveai.system import EvolveAISystem
from evolveai.parallel_data_engine import SourceRecord


class TinyConnector:
    source_id = "tiny"

    async def stream(self):
        for i in range(8):
            yield SourceRecord(
                content=f"Repeated training example {i} describes a recurrent spiking network.",
                source_id=self.source_id,
                source_uri=f"https://example.test/{i}",
                domain="neuroscience",
            )


@unittest.skipIf(torch is None, "PyTorch is required")
class IntegratedSystemTests(unittest.TestCase):
    def test_registered_stream_reaches_snn_learning_path(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                system = EvolveAISystem(
                    state_dir=str(Path(tmp) / "state"),
                    device="cpu",
                    queue_size=3,
                    evaluation_workers=2,
                    batch_size=2,
                    train_every=1,
                    hidden_size=16,
                    embedding_dim=8,
                    block_size=32,
                )
                system.register(TinyConnector())
                status = await system.run()
                self.assertEqual(status["sources_completed"], 1)
                self.assertGreater(status["records_persisted"], 0)
                self.assertGreater(system.learner.updates, 0)
                system.save()
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
