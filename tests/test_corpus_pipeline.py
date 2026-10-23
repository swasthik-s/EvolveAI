import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
from prepare_corpus import normalize


def test_normalize_removes_nulls_and_excess_whitespace():
    assert normalize("  Hello\x00   world \n\n\n next ") == "Hello world\n\n next"


def test_normalize_empty_text():
    assert normalize(" \n\t ") == ""
