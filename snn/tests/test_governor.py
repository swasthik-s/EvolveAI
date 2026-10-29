import hashlib
import tempfile
import unittest
from pathlib import Path

from governor import validate_dataset_manifest, load_state, save_state


class GovernorTests(unittest.TestCase):
    def test_allows_reviewed_tinystories_source(self):
        validate_dataset_manifest({
            "dataset_id": "roneneldan/TinyStories",
            "dataset_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        })

    def test_blocks_unknown_dataset(self):
        with self.assertRaises(ValueError):
            validate_dataset_manifest({
                "dataset_id": "someone/unreviewed-dataset",
                "dataset_url": "https://huggingface.co/datasets/someone/unreviewed-dataset",
            })

    def test_blocks_mismatched_source_url(self):
        with self.assertRaises(ValueError):
            validate_dataset_manifest({
                "dataset_id": "roneneldan/TinyStories",
                "dataset_url": "https://example.com/datasets/TinyStories",
            })

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            expected = {
                "state_version": 1,
                "cycles": [{"cycle": 1}],
                "best_validation_loss": 4.2,
            }
            save_state(expected, path)
            self.assertEqual(load_state(path), expected)

    def test_missing_state_starts_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            state = load_state(Path(directory) / "missing.json")
            self.assertEqual(state["cycles"], [])
            self.assertEqual(state["state_version"], 1)


if __name__ == "__main__":
    unittest.main()
