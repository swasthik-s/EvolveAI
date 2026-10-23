import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
from dataset_harvester import DATASETS, normalize_text


def test_only_reviewed_datasets_are_exposed():
    assert list(DATASETS) == ["TinyStories (research starter)"]
    spec = DATASETS["TinyStories (research starter)"]
    assert spec["id"] == "roneneldan/TinyStories"
    assert "license" in spec
    assert "url" in spec


def test_normalize_text_removes_nulls_and_extra_whitespace():
    assert normalize_text("  Hello\x00   world \n\n\n next ") == "Hello world\n\n next"


def test_normalize_text_does_not_change_words():
    assert normalize_text("The quick brown fox.") == "The quick brown fox."
