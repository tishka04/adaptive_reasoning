"""Small source-train corpus that preserves pre-action object relations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from ..unified_cognitive_controller import UnifiedCognitiveController
from .dataset import NeuroTransition, Sage11ControllerCollector
from .relational_features import (
    RELATIONAL_FEATURE_SCHEMA,
    RelationalFeatureSchema,
    encode_relational_features,
)
from .splits import ArtifactPurpose, SAGE11_SPLITS, SOURCE_TRAIN


RELATIONAL_TRANSITION_FORMAT_VERSION = "sage11-relational-transition-v1"
RELATIONAL_MANIFEST_FORMAT_VERSION = "sage11-relational-pilot-manifest-v1"
RELATIONAL_PILOT_GAME_QUOTAS: Mapping[str, int] = {
    game: (27 if game == "lp85" else 1_000)
    for game in SOURCE_TRAIN
}
RELATIONAL_PILOT_TARGET_TRANSITIONS = sum(
    RELATIONAL_PILOT_GAME_QUOTAS.values()
)


@dataclass(frozen=True)
class RelationalPilotTransition:
    """A v2 base transition plus its unreconstructable live geometry."""

    base_transition: NeuroTransition
    relational_features_before: Tuple[float, ...]
    relational_schema_checksum: str = RELATIONAL_FEATURE_SCHEMA.checksum
    format_version: str = RELATIONAL_TRANSITION_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RELATIONAL_TRANSITION_FORMAT_VERSION:
            raise ValueError("unsupported relational transition version")
        if (
            self.relational_schema_checksum
            != RELATIONAL_FEATURE_SCHEMA.checksum
        ):
            raise ValueError("relational transition schema mismatch")
        if len(self.relational_features_before) != (
            RELATIONAL_FEATURE_SCHEMA.feature_count
        ):
            raise ValueError("relational transition feature width mismatch")
        if any(
            not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0
            for value in self.relational_features_before
        ):
            raise ValueError(
                "relational transition features must be finite in [0, 1]"
            )
        SAGE11_SPLITS.assert_authorized(
            [self.base_transition.game_id],
            purpose=ArtifactPurpose.TRAIN,
        )

    @property
    def transition_signature(self) -> str:
        return self.base_transition.transition_signature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "relational_schema_checksum": (
                self.relational_schema_checksum
            ),
            "relational_features_before": [
                float(value)
                for value in self.relational_features_before
            ],
            "base_transition": self.base_transition.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RelationalPilotTransition":
        return cls(
            base_transition=NeuroTransition.from_dict(
                dict(payload["base_transition"])
            ),
            relational_features_before=tuple(
                float(value)
                for value in payload["relational_features_before"]
            ),
            relational_schema_checksum=str(
                payload["relational_schema_checksum"]
            ),
            format_version=str(payload["format_version"]),
        )


class RelationalJsonlCapture:
    """Append accepted relations in lockstep with a resumable base shard."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_existing_rows: int,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[RelationalPilotTransition] = []
        if self.path.exists():
            self._records = list(iter_relational_shard(self.path))
        expected = max(0, int(expected_existing_rows))
        if len(self._records) < expected:
            raise ValueError(
                "relational sidecar trails its verified base checkpoint"
            )
        if len(self._records) > expected:
            self._recover_checkpoint_prefix(expected)
        if not self.path.exists():
            self.path.touch()

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> Tuple[RelationalPilotTransition, ...]:
        return tuple(self._records)

    def append(self, transition: RelationalPilotTransition) -> None:
        if any(
            existing.transition_signature
            == transition.transition_signature
            for existing in self._records
        ):
            raise ValueError("duplicate relational transition signature")
        line = json.dumps(
            transition.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(transition)

    def _recover_checkpoint_prefix(self, expected: int) -> None:
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        recovery = self.path.with_name(
            f"{self.path.stem}.uncheckpointed-{digest}{self.path.suffix}"
        )
        if recovery.exists():
            raise ValueError("relational recovery artifact already exists")
        self.path.replace(recovery)
        self._records = self._records[:expected]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for record in self._records
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class RelationalCapturingCollector:
    """Delegate policy accounting while preserving live pre-action geometry."""

    def __init__(
        self,
        delegate: Sage11ControllerCollector,
        capture: RelationalJsonlCapture,
        *,
        schema: RelationalFeatureSchema = RELATIONAL_FEATURE_SCHEMA,
    ) -> None:
        self.delegate = delegate
        self.capture = capture
        self.schema = schema

    def on_reset(self) -> None:
        self.delegate.on_reset()

    def record(
        self,
        update: Any,
        **kwargs: Any,
    ) -> bool:
        action_name = str(kwargs["action_name"])
        action_data = dict(kwargs.get("action_data", {}) or {})
        features = encode_relational_features(
            update.record.obs_before,
            action_name=action_name,
            action_data=action_data,
            schema=self.schema,
        )
        accepted = self.delegate.record(update, **kwargs)
        if not accepted:
            return False
        transition = self.delegate.builder.records[-1]
        relational = RelationalPilotTransition(
            base_transition=transition,
            relational_features_before=tuple(
                float(value) for value in features
            ),
            relational_schema_checksum=self.schema.checksum,
        )
        self.capture.append(relational)
        return True


def make_relational_controller(
    game_id: str,
    delegate: Sage11ControllerCollector,
    capture: RelationalJsonlCapture,
) -> UnifiedCognitiveController:
    """Create the normal controller with a capture-only collector wrapper."""
    return UnifiedCognitiveController(
        game_id,
        neuro_transition_collector=RelationalCapturingCollector(
            delegate,
            capture,
        ),
    )


def iter_relational_shard(
    path: str | Path,
) -> Iterable[RelationalPilotTransition]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield RelationalPilotTransition.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid relational row {path}:{line_number}"
                ) from exc


def relational_shard_metadata(
    path: str | Path,
    *,
    expected_game: str | None = None,
    expected_rows: int | None = None,
) -> Dict[str, Any]:
    source = Path(path)
    raw = source.read_bytes()
    records = tuple(iter_relational_shard(source))
    games = {
        record.base_transition.game_id
        for record in records
    }
    if expected_game is not None and games != {str(expected_game)}:
        raise ValueError("relational shard game mismatch")
    if expected_rows is not None and len(records) != int(expected_rows):
        raise ValueError("relational shard row mismatch")
    signatures = {
        record.transition_signature
        for record in records
    }
    if len(signatures) != len(records):
        raise ValueError("relational shard contains duplicate transitions")
    return {
        "path": _repo_path(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "transitions": len(records),
        "game_id": (
            next(iter(games))
            if len(games) == 1
            else None
        ),
        "source_split": "source_train",
    }


def build_relational_manifest(
    shards: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    items = tuple(
        dict(item)
        for item in shards
    )
    game_counts = {
        str(item["game_id"]): int(item["transitions"])
        for item in items
    }
    if game_counts != dict(RELATIONAL_PILOT_GAME_QUOTAS):
        raise ValueError("relational manifest game quotas do not match")
    payload: Dict[str, Any] = {
        "format_version": RELATIONAL_MANIFEST_FORMAT_VERSION,
        "split_registry_checksum": SAGE11_SPLITS.checksum,
        "artifact_purpose": ArtifactPurpose.TRAIN.value,
        "target_transitions": RELATIONAL_PILOT_TARGET_TRANSITIONS,
        "total_transitions": sum(game_counts.values()),
        "games": list(SOURCE_TRAIN),
        "game_quotas": dict(RELATIONAL_PILOT_GAME_QUOTAS),
        "game_counts": game_counts,
        "collection_policy": {
            "active_controller": 0.70,
            "uniform_legal": 0.20,
            "frontier_stall_probe": 0.10,
        },
        "relational_schema": RELATIONAL_FEATURE_SCHEMA.to_dict(),
        "raw_grids_archived": False,
        "raw_action_coordinates_archived_for_audit": True,
        "raw_action_coordinates_used_as_model_features": False,
        "source_validation_shards_opened": False,
        "historical_shards_opened": False,
        "holdout_shards_opened": False,
        "shards": list(items),
    }
    payload["manifest_checksum"] = _checksum_payload(payload)
    return payload


def verify_relational_manifest(
    path: str | Path,
) -> Dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format_version") != RELATIONAL_MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported relational manifest version")
    expected = str(payload.get("manifest_checksum", ""))
    canonical = dict(payload)
    canonical.pop("manifest_checksum", None)
    if _checksum_payload(canonical) != expected:
        raise ValueError("relational manifest checksum mismatch")
    if str(payload.get("split_registry_checksum")) != SAGE11_SPLITS.checksum:
        raise ValueError("relational manifest split checksum mismatch")
    schema = dict(payload.get("relational_schema", {}) or {})
    if schema.get("checksum") != RELATIONAL_FEATURE_SCHEMA.checksum:
        raise ValueError("relational manifest schema checksum mismatch")
    observed = []
    seen_signatures: set[str] = set()
    for shard in payload["shards"]:
        shard_path = _resolve_path(str(shard["path"]), source)
        metadata = relational_shard_metadata(
            shard_path,
            expected_game=str(shard["game_id"]),
            expected_rows=int(shard["transitions"]),
        )
        if metadata["sha256"] != str(shard["sha256"]):
            raise ValueError("relational shard checksum mismatch")
        for record in iter_relational_shard(shard_path):
            if record.transition_signature in seen_signatures:
                raise ValueError(
                    "relational manifest contains cross-shard duplicate"
                )
            seen_signatures.add(record.transition_signature)
        observed.append(metadata)
    rebuilt = build_relational_manifest(observed)
    if rebuilt["manifest_checksum"] != expected:
        raise ValueError("relational manifest contents do not reproduce")
    return payload


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _resolve_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in manifest_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve relational shard {raw_path}")


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


__all__ = [
    "RELATIONAL_MANIFEST_FORMAT_VERSION",
    "RELATIONAL_PILOT_GAME_QUOTAS",
    "RELATIONAL_PILOT_TARGET_TRANSITIONS",
    "RELATIONAL_TRANSITION_FORMAT_VERSION",
    "RelationalCapturingCollector",
    "RelationalJsonlCapture",
    "RelationalPilotTransition",
    "build_relational_manifest",
    "iter_relational_shard",
    "make_relational_controller",
    "relational_shard_metadata",
    "verify_relational_manifest",
]
