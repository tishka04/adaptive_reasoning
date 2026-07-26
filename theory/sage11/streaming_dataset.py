"""Verified source-corpus loader for the shared streaming feature interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dataset import DatasetManifest
from .effect_pilot_runner import (
    DEFAULT_MANIFEST_PATH,
    iter_source_rows,
)
from .source_dataset_runner import verify_source_dataset
from .splits import SOURCE_TRAIN, SOURCE_VALIDATION
from .streaming_features import (
    EncodedStreamingDataset,
    encode_transition_rows,
)


def load_source_streaming_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> EncodedStreamingDataset:
    """Verify the manifest and encode every row through the shared tracker."""
    path = Path(manifest_path)
    verify_source_dataset(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = encode_transition_rows(
        lambda: iter_source_rows(path, payload),
        total_rows=int(payload["total_transitions"]),
        manifest_checksum=str(payload["manifest_checksum"]),
    )
    if set(dataset.games[dataset.train_mask]) != set(SOURCE_TRAIN):
        raise ValueError(
            "streaming source-training games do not match registry"
        )
    if set(dataset.games[~dataset.train_mask]) != set(SOURCE_VALIDATION):
        raise ValueError(
            "streaming source-validation games do not match registry"
        )
    return dataset


def load_source_train_streaming_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> EncodedStreamingDataset:
    """Load and verify only source-train shards for development audits."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.from_dict(payload)
    if manifest.checksum != str(payload.get("manifest_checksum", "")):
        raise ValueError("SAGE.11 manifest checksum mismatch")
    if manifest.legacy_weights_loaded:
        raise ValueError("source-train audit rejects legacy weights")
    selected = [
        dict(shard)
        for shard in payload["shards"]
        if set(dict(shard["split_counts"])) == {"source_train"}
    ]
    selected_games = {
        str(game)
        for shard in selected
        for game in shard["games"]
    }
    if selected_games != set(SOURCE_TRAIN):
        raise ValueError("source-train shard set does not match registry")
    total_rows = 0
    for shard in selected:
        shard_path = _resolve_manifest_path(str(shard["path"]), path)
        raw = shard_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(shard["sha256"]):
            raise ValueError(f"source-train shard checksum mismatch: {shard_path}")
        rows = sum(1 for line in raw.splitlines() if line.strip())
        if rows != int(shard["transitions"]):
            raise ValueError(f"source-train shard row mismatch: {shard_path}")
        total_rows += rows
    expected_rows = int(payload["split_counts"]["source_train"])
    if total_rows != expected_rows:
        raise ValueError("source-train rows do not match manifest")
    filtered_payload = {"shards": selected}
    dataset = encode_transition_rows(
        lambda: iter_source_rows(path, filtered_payload),
        total_rows=total_rows,
        manifest_checksum=str(payload["manifest_checksum"]),
    )
    if not dataset.train_mask.all():
        raise ValueError("source-train audit loaded a non-training row")
    return dataset


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in manifest_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve source shard {raw_path}")


__all__ = [
    "load_source_streaming_dataset",
    "load_source_train_streaming_dataset",
]
