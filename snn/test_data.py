"""Tests for chronological corpus partitioning and holdout isolation."""
import unittest
import torch

from data import split_three_way


class ThreeWaySplitTests(unittest.TestCase):
    def test_split_is_chronological_and_disjoint(self):
        tokens = torch.arange(1000)
        train, validation, test = split_three_way(tokens)
        self.assertEqual(train.tolist(), list(range(800)))
        self.assertEqual(validation.tolist(), list(range(800, 900)))
        self.assertEqual(test.tolist(), list(range(900, 1000)))

    def test_split_preserves_all_tokens(self):
        tokens = torch.arange(1003)
        train, validation, test = split_three_way(tokens)
        self.assertEqual(train.numel() + validation.numel() + test.numel(), 1003)
        self.assertEqual(int(train[-1]) + 1, int(validation[0]))
        self.assertEqual(int(validation[-1]) + 1, int(test[0]))

    def test_invalid_fractions_are_rejected(self):
        with self.assertRaises(ValueError):
            split_three_way(torch.arange(100), train_fraction=0.9, validation_fraction=0.2)


if __name__ == "__main__":
    unittest.main()
