"""Prepare a size-limited, provenance-tracked TinyStories corpus.

Run locally or in a hosted notebook, not in the Streamlit web process:
  pip install datasets
  python snn/prepare_corpus.py --max-mb 5 --out-dir data/tinystories

The script streams rows, deduplicates normalized stories, caps retained text bytes,
and writes deterministic train/validation/test text files plus a provenance manifest.
Review upstream dataset terms before any redistribution or commercial use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = "roneneldan/TinyStories"
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
LICENSE = "CDLA-Sharing-1.0 (verify current dataset card and applicable terms)"
SPLIT_NAMES = ("train", "validation", "test")


def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-mb", type=float, default=5.0, help="Maximum retained text size in MiB before split-file overhead")
    parser.add_argument("--out-dir", default="data/tinystories")
    parser.add_argument("--split", default="train", help="Upstream split to stream (default: train)")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap; 0 means stop at byte limit")
    args = parser.parse_args()
    if args.max_mb <= 0 or args.max_rows < 0:
        parser.error("--max-mb must be positive and --max-rows non-negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing optional dependency. Run: pip install datasets") from exc

    target_bytes = int(args.max_mb * 1024 * 1024)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    stories = []
    retained_bytes = 0
    rows_seen = 0
    duplicates = 0
    skipped_empty = 0

    stream = load_dataset(DATASET_ID, split=args.split, streaming=True)
    for row in stream:
        rows_seen += 1
        story = normalize(str(row.get("text", "")))
        if len(story) < 20:
            skipped_empty += 1
            continue
        key = hashlib.sha256(story.casefold().encode("utf-8")).hexdigest()
        if key in seen:
            duplicates += 1
            continue
        encoded_size = len((story + "\n\n").encode("utf-8"))
        if retained_bytes + encoded_size > target_bytes:
            break
        seen.add(key)
        stories.append(story)
        retained_bytes += encoded_size
        if args.max_rows and len(stories) >= args.max_rows:
            break

    if len(stories) < 10:
        raise SystemExit(f"Only retained {len(stories)} stories; check dataset access/split or raise limits.")

    # Split by whole story to prevent a story leaking across partitions.
    n = len(stories)
    train_end = max(1, int(n * 0.90))
    val_end = max(train_end + 1, int(n * 0.95))
    val_end = min(val_end, n - 1)
    partitions = {
        "train.txt": stories[:train_end],
        "validation.txt": stories[train_end:val_end],
        "test.txt": stories[val_end:],
    }
    for filename, items in partitions.items():
        (out_dir / filename).write_text("\n\n".join(items) + "\n", encoding="utf-8")

    corpus_bytes = sum(len((s + "\n\n").encode("utf-8")) for s in stories)
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "upstream_split": args.split,
        "license_declared_in_script": LICENSE,
        "license_warning": "Verify the live dataset card and terms before redistribution or commercial use.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_seen": rows_seen,
        "stories_retained": len(stories),
        "duplicates_skipped": duplicates,
        "empty_or_short_skipped": skipped_empty,
        "retained_text_bytes_before_split_overhead": corpus_bytes,
        "sha256_by_file": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in partitions
        },
        "split_counts": {name: len(items) for name, items in partitions.items()},
        "seed": None,
        "notes": "Chronological stream sample; splits are by whole story. Use the supplied train file for training and validation/test files only for evaluation.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "stories_retained": len(stories),
        "retained_bytes": corpus_bytes,
        "splits": manifest["split_counts"],
        "manifest": str(out_dir / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
