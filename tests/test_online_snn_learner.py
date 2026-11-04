import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from evolveai.parallel_data_engine import SourceRecord

if torch is not None:
    from snn.online_learner import OnlineSNNLearner


@unittest.skipIf(torch is None, "PyTorch is required")
class OnlineSNNLearnerTests(unittest.TestCase):
    def test_selected_records_update_and_checkpoint_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "online.pt")
            learner = OnlineSNNLearner(
                path, device="cpu", hidden_size=16, embedding_dim=8,
                block_size=32, batch_size=2, train_every=1, save_every=100,
            )
            record = SourceRecord(
                content="A recurrent spiking neural network processes sequential observations.",
                source_id="test",
            )
            learner.add(record, {"decision": "learn_now"})
            learner.add(record, {"decision": "learn_now"})
            self.assertEqual(learner.updates, 1)
            self.assertIsNotNone(learner.last_loss)
            learner.save()
            restored = OnlineSNNLearner(
                path, device="cpu", hidden_size=16, embedding_dim=8,
                block_size=32, batch_size=2, train_every=1, save_every=100,
            )
            self.assertEqual(restored.updates, 1)
            self.assertEqual(restored.accepted_records, 2)


if __name__ == "__main__":
    unittest.main()
