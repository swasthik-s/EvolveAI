"""Allowlisted, size-bounded dataset acquisition for the SNN research lab.

Only explicitly reviewed dataset IDs are available. This is not an open-ended web
scraper. Check the live dataset card and license before each use.
"""
from __future__ import annotations
import hashlib
import re
from datetime import datetime, timezone

DATASETS = {
    "TinyStories (research starter)": {
        "id": "roneneldan/TinyStories",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "split": "train",
        "text_field": "text",
        "license": "CDLA-Sharing-1.0 (verify current dataset card)",
        "commercial_use": "Do not assume commercial permission; review the current terms.",
        "purpose": "Simple stories for early language-pattern experiments.",
    },
}


def normalize_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def fetch_corpus(dataset_label: str, max_mb: float = 5.0, max_rows: int = 0,
                 progress=None) -> tuple[str, dict]:
    """Stream an allowlisted dataset and return deduplicated text plus provenance."""
    if dataset_label not in DATASETS:
        raise ValueError("Dataset is not in the reviewed allowlist.")
    if not (0.1 <= max_mb <= 50):
        raise ValueError("max_mb must be between 0.1 and 50 MiB for the Streamlit lab.")
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Dataset support is not installed. Deploy with snn/requirements.txt.") from exc

    spec = DATASETS[dataset_label]
    limit = int(max_mb * 1024 * 1024)
    stream = load_dataset(spec["id"], split=spec["split"], streaming=True)
    seen = set()
    rows_seen = 0
    duplicates = 0
    skipped = 0
    kept_bytes = 0
    parts = []

    for row in stream:
        rows_seen += 1
        raw = row.get(spec["text_field"], "")
        text = normalize_text(raw if isinstance(raw, str) else "")
        if len(text) < 20:
            skipped += 1
            continue
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        encoded = (text + "\n\n").encode("utf-8")
        if kept_bytes + len(encoded) > limit:
            break
        seen.add(digest)
        parts.append(text)
        kept_bytes += len(encoded)
        if progress and (len(parts) % 100 == 0):
            progress(len(parts), kept_bytes, rows_seen)
        if max_rows and len(parts) >= max_rows:
            break

    if len(parts) < 10:
        raise RuntimeError(
            f"Only {len(parts)} usable records were collected. Check dataset access or raise the cap."
        )
    corpus = "\n\n".join(parts) + "\n"
    manifest = {
        "dataset_id": spec["id"],
        "dataset_url": spec["url"],
        "upstream_split": spec["split"],
        "declared_license_reference": spec["license"],
        "commercial_use_note": spec["commercial_use"],
        "purpose": spec["purpose"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_seen": rows_seen,
        "records_kept": len(parts),
        "duplicates_skipped": duplicates,
        "short_or_empty_skipped": skipped,
        "corpus_bytes": len(corpus.encode("utf-8")),
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "streaming": True,
        "max_mib": max_mb,
        "note": "Review upstream dataset card/license. This manifest is provenance, not legal advice.",
    }
    return corpus, manifest
