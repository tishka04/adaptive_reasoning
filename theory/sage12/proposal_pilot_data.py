"""Frozen data contract for the SAGE12 grounded-proposal pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .scene_graph import (
    GroundedEntity,
    GroundedRelation,
    SceneGraph,
)


PILOT_FORMAT_VERSION = "sage12-proposal-pilot-v1"
TRACE_FORMAT_VERSION = "sage12-proposal-trace-v1"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "proposal_pilot_v1"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"


@dataclass(frozen=True)
class ProposalPilotTrace:
    game_id: str
    source_split: str
    policy_seed: int
    reset_index: int
    step_index: int
    scene_graph: Mapping[str, Any]
    available_action_names: Tuple[str, ...]
    selected_action_name: str
    selected_action_data: Mapping[str, Any]
    observed_effects: Tuple[str, ...]
    changed: bool
    noop: bool
    player_moved: bool
    level_complete: bool
    game_over: bool
    productive: bool
    repeat_index: int
    format_version: str = TRACE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != TRACE_FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 proposal trace version")
        if self.source_split not in {
            "source_train",
            "source_validation",
        }:
            raise ValueError("proposal traces are source-only")
        if not self.available_action_names:
            raise ValueError("proposal trace requires legal action names")
        if self.selected_action_name not in self.available_action_names:
            raise ValueError("executed action must have been legal")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scene_graph"] = _json_safe(self.scene_graph)
        payload["selected_action_data"] = _json_safe(
            self.selected_action_data
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalPilotTrace":
        return cls(
            game_id=str(payload["game_id"]),
            source_split=str(payload["source_split"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            step_index=int(payload["step_index"]),
            scene_graph=dict(payload["scene_graph"]),
            available_action_names=tuple(payload["available_action_names"]),
            selected_action_name=str(payload["selected_action_name"]),
            selected_action_data=dict(
                payload.get("selected_action_data", {})
            ),
            observed_effects=tuple(payload["observed_effects"]),
            changed=bool(payload["changed"]),
            noop=bool(payload["noop"]),
            player_moved=bool(payload["player_moved"]),
            level_complete=bool(payload["level_complete"]),
            game_over=bool(payload["game_over"]),
            productive=bool(payload["productive"]),
            repeat_index=int(payload["repeat_index"]),
            format_version=str(payload["format_version"]),
        )


def graph_to_mapping(graph: SceneGraph) -> dict[str, Any]:
    """Serialize only the model-facing structural scene."""
    return {
        "entities": [
            {
                "entity_id": entity.entity_id,
                "roles": list(entity.roles),
                "area_bucket": entity.area_bucket,
                "aspect_bucket": entity.aspect_bucket,
            }
            for entity in graph.entities
        ],
        "relations": [
            {
                "kind": relation.kind,
                "subject_id": relation.subject_id,
                "object_id": relation.object_id,
            }
            for relation in graph.relations
        ],
        "state_predicates": sorted(graph.state_predicates),
        "signature": graph.signature,
    }


def graph_from_mapping(payload: Mapping[str, Any]) -> SceneGraph:
    entities = tuple(
        GroundedEntity(
            entity_id=str(item["entity_id"]),
            roles=tuple(str(role) for role in item["roles"]),
            center=(0.0, 0.0),
            area_bucket=str(item["area_bucket"]),
            aspect_bucket=str(item["aspect_bucket"]),
            value_token="excluded",
        )
        for item in payload.get("entities", ())
    )
    relations = tuple(
        GroundedRelation(
            kind=str(item["kind"]),
            subject_id=str(item["subject_id"]),
            object_id=str(item["object_id"]),
        )
        for item in payload.get("relations", ())
    )
    return SceneGraph(
        entities=entities,
        relations=relations,
        state_predicates=frozenset(
            str(item) for item in payload.get("state_predicates", ())
        ),
        signature=str(payload["signature"]),
    )


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != PILOT_FORMAT_VERSION:
        raise ValueError("unsupported SAGE12 proposal-pilot manifest")
    if payload.get("status") not in {
        "FROZEN_BEFORE_OUTCOMES",
        "AMENDED_BEFORE_OUTCOMES_AFTER_INFEASIBLE_PREFLIGHT",
    }:
        raise ValueError("SAGE12 proposal pilot was not frozen")
    expected = str(payload.get("manifest_checksum", ""))
    actual = manifest_checksum(payload)
    if expected != actual:
        raise ValueError(
            f"SAGE12 frozen-manifest checksum mismatch: {expected} != {actual}"
        )
    quotas = {
        str(game): int(value)
        for game, value in dict(payload.get("game_quotas", {})).items()
    }
    if sum(quotas.values()) != int(payload["target_transitions"]):
        raise ValueError("SAGE12 frozen quotas do not sum to target")
    return payload


def manifest_checksum(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_checksum", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def read_trace_shard(path: str | Path) -> Tuple[ProposalPilotTrace, ...]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(
                    ProposalPilotTrace.from_dict(json.loads(line))
                )
    return tuple(records)


def shard_metadata(
    path: str | Path,
    *,
    expected_game: str,
    expected_rows: int,
) -> dict[str, Any]:
    shard_path = Path(path)
    records = read_trace_shard(shard_path)
    if len(records) != int(expected_rows):
        raise ValueError(
            f"{expected_game} row mismatch: "
            f"{len(records)} != {expected_rows}"
        )
    if any(record.game_id != expected_game for record in records):
        raise ValueError(f"{expected_game} shard contains another game")
    return {
        "game_id": expected_game,
        "source_split": records[0].source_split,
        "rows": len(records),
        "path": shard_path.as_posix(),
        "sha256": _file_sha256(shard_path),
        "productive_rows": sum(record.productive for record in records),
        "action_counts": _counts(
            record.selected_action_name for record in records
        ),
    }


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


__all__ = [
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "PILOT_FORMAT_VERSION",
    "ProposalPilotTrace",
    "TRACE_FORMAT_VERSION",
    "graph_from_mapping",
    "graph_to_mapping",
    "load_frozen_manifest",
    "manifest_checksum",
    "read_trace_shard",
    "shard_metadata",
]
