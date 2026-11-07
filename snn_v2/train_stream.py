"""Bounded multi-source continual training runner for SNN V2."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from .data import iter_public_records
from .learner import ContinualSNNLearner
from .model import AdaptiveLIFNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="tinystories,wikitext")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-record-chars", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="runs/snn-v2/latest.pt")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()
    if args.max_records < 1 or args.save_every < 1:
        parser.error("max-records and save-every must be positive")
    model = AdaptiveLIFNetwork(hidden_size=args.hidden_size)
    learner = ContinualSNNLearner(
        model, device=args.device, checkpoint_path=args.checkpoint,
        max_record_bytes=args.max_record_chars,
    )
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_file():
        learner.load()
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    started = time.time()
    for index, record in enumerate(iter_public_records(
        sources, max_records=args.max_records, max_record_chars=args.max_record_chars
    ), start=1):
        result = learner.learn_record(
            record.text, source_id=record.source_id, stream_type="public", quality_score=1.0
        )
        if result.get("accepted") and index % 10 == 0:
            print(json.dumps({
                "records": index, "source": record.source_id,
                "loss": result.get("loss"), "accuracy": result.get("accuracy"),
                "updates": learner.stats.updates,
            }))
        if index % args.save_every == 0:
            learner.save()
    learner.save()
    print(json.dumps({
        "elapsed_seconds": round(time.time() - started, 3),
        "stats": learner.stats.__dict__,
        "checkpoint": str(checkpoint),
        "sources": sources,
    }, indent=2))


if __name__ == "__main__":
    main()
