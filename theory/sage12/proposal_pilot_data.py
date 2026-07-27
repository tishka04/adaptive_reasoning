"""Frozen data contract for the SAGE12 grounded-proposal pilot."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

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
    relation_shuffle_graph: Mapping[str, Any] = field(default_factory=dict)
    trace_digest: str = ""
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
        if not self.trace_digest:
            object.__setattr__(
                self,
                "trace_digest",
                _pre_action_digest(self),
            )

    @property
    def digest(self) -> str:
        return self.trace_digest

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scene_graph"] = _json_safe(self.scene_graph)
        payload["relation_shuffle_graph"] = _json_safe(
            self.relation_shuffle_graph
        )
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
            relation_shuffle_graph=dict(
                payload.get("relation_shuffle_graph", {})
            ),
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
            trace_digest=str(payload.get("trace_digest", "")),
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


def compact_trace_views(
    trace: ProposalPilotTrace,
    *,
    maximum_entities: int,
    maximum_relations: int,
) -> ProposalPilotTrace:
    """Materialize bounded original/shuffled views without using outcomes."""
    graph = graph_from_mapping(trace.scene_graph)
    compact = _compact_graph(
        graph,
        maximum_entities=maximum_entities,
        maximum_relations=maximum_relations,
    )
    shuffled = _relation_shuffle_graph(
        compact,
        salt=trace.digest,
        maximum_relations=maximum_relations,
    )
    return replace(
        trace,
        scene_graph=graph_to_mapping(compact),
        relation_shuffle_graph=graph_to_mapping(shuffled),
    )


def compact_existing_collection(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    """Rewrite the collected full graphs to exact bounded prompt views."""
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    collection_path = destination / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    original_shards = {
        str(item["game_id"]): str(item["sha256"])
        for item in collection["shards"]
    }
    limits = frozen["model"]["prompt_limits"]
    metadata_by_game = {}
    game_items = sorted(
        frozen["game_quotas"].items(),
        key=lambda item: (
            destination / "shards" / f"{item[0]}.jsonl"
        ).stat().st_size,
    )
    for game, quota in game_items:
        path = destination / "shards" / f"{game}.jsonl"
        if not _shard_is_compact(path):
            temporary = path.with_suffix(path.suffix + ".compact.tmp")
            written = 0
            with (
                path.open("r", encoding="utf-8") as source,
                temporary.open("w", encoding="utf-8", newline="\n") as target,
            ):
                for line in source:
                    if not line.strip():
                        continue
                    record = ProposalPilotTrace.from_dict(json.loads(line))
                    compacted = compact_trace_views(
                        record,
                        maximum_entities=int(limits["maximum_entities"]),
                        maximum_relations=int(limits["maximum_relations"]),
                    )
                    target.write(
                        json.dumps(
                            compacted.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    written += 1
            if written != int(quota):
                raise ValueError(
                    f"{game} projection row mismatch: {written} != {quota}"
                )
            os.replace(temporary, path)
        metadata_by_game[str(game)] = shard_metadata(
            path,
            expected_game=str(game),
            expected_rows=int(quota),
        )
    metadata = [
        metadata_by_game[str(game)]
        for game in frozen["game_quotas"]
    ]
    combined = hashlib.sha256(
        "".join(item["sha256"] for item in metadata).encode("ascii")
    ).hexdigest()
    collection.update(
        {
            "shards": metadata,
            "combined_shard_checksum": combined,
            "projection_amendment": {
                "outcome_fields_used": False,
                "maximum_entities": int(limits["maximum_entities"]),
                "maximum_relations": int(limits["maximum_relations"]),
                "sampling_digest": "pre_action_only_v1",
                "original_shard_sha256": original_shards,
            },
        }
    )
    temporary_manifest = collection_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, collection_path)
    return collection


def _shard_is_compact(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                return bool(
                    payload.get("trace_digest")
                    and payload.get("relation_shuffle_graph")
                )
    return False


def _compact_graph(
    graph: SceneGraph,
    *,
    maximum_entities: int,
    maximum_relations: int,
) -> SceneGraph:
    entities = tuple(
        sorted(graph.entities, key=_entity_priority)[
            : max(1, int(maximum_entities))
        ]
    )
    entity_ids = {entity.entity_id for entity in entities}
    candidates = [
        relation
        for relation in graph.relations
        if relation.subject_id in entity_ids
        and relation.object_id in entity_ids
    ]
    relations = _stratified_relations(candidates, maximum_relations)
    relation_kinds = {relation.kind for relation in graph.relations}
    state = {
        predicate
        for predicate in graph.state_predicates
        if predicate.split("|", 1)[0]
        not in relation_kinds
        and (
            predicate.split("|")[1] in entity_ids
            if len(predicate.split("|")) > 1
            and predicate.split("|")[1] not in {"", "-"}
            else True
        )
    }
    state.update(relation.key for relation in relations)
    return SceneGraph(
        entities=entities,
        relations=relations,
        state_predicates=frozenset(state),
        signature=_graph_signature(entities, state),
    )


def _relation_shuffle_graph(
    graph: SceneGraph,
    *,
    salt: str,
    maximum_relations: int,
) -> SceneGraph:
    entity_ids = sorted(entity.entity_id for entity in graph.entities)
    if len(entity_ids) < 2:
        return graph
    shift = int(salt[:8], 16) % (len(entity_ids) - 1) + 1
    mapped = {
        entity_id: entity_ids[(index + shift) % len(entity_ids)]
        for index, entity_id in enumerate(entity_ids)
    }
    candidates = [
        GroundedRelation(
            kind=relation.kind,
            subject_id=mapped[relation.subject_id],
            object_id=mapped[relation.object_id],
        )
        for relation in graph.relations
    ]
    relations = _stratified_relations(candidates, maximum_relations)
    relation_kinds = {relation.kind for relation in graph.relations}
    state = {
        predicate
        for predicate in graph.state_predicates
        if predicate.split("|", 1)[0]
        not in relation_kinds
    }
    state.update(relation.key for relation in relations)
    return SceneGraph(
        entities=graph.entities,
        relations=relations,
        state_predicates=frozenset(state),
        signature=_graph_signature(graph.entities, state),
    )


def _entity_priority(entity: GroundedEntity) -> tuple[int, str]:
    priorities = {
        "player": 0,
        "hazardous": 1,
        "collectible": 2,
        "movable": 3,
        "clickable": 4,
        "target": 5,
        "object": 6,
    }
    best = min(
        (priorities.get(role, 7) for role in entity.roles),
        default=7,
    )
    return best, entity.entity_id


def _stratified_relations(
    candidates: Sequence[GroundedRelation],
    maximum: int,
) -> Tuple[GroundedRelation, ...]:
    by_kind: dict[str, list[GroundedRelation]] = {}
    for relation in sorted(candidates, key=lambda item: item.key):
        by_kind.setdefault(relation.kind, []).append(relation)
    selected = []
    kinds = sorted(by_kind)
    while kinds and len(selected) < max(1, int(maximum)):
        remaining = []
        for kind in kinds:
            bucket = by_kind[kind]
            if bucket and len(selected) < int(maximum):
                selected.append(bucket.pop(0))
            if bucket:
                remaining.append(kind)
        kinds = remaining
    return tuple(selected)


def _graph_signature(
    entities: Sequence[GroundedEntity],
    state: Iterable[str],
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "entities": [
                    {
                        "id": entity.entity_id,
                        "roles": entity.roles,
                        "area": entity.area_bucket,
                        "aspect": entity.aspect_bucket,
                    }
                    for entity in entities
                ],
                "relations": sorted(state),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _pre_action_digest(trace: ProposalPilotTrace) -> str:
    payload = {
        "game_id": trace.game_id,
        "source_split": trace.source_split,
        "policy_seed": trace.policy_seed,
        "reset_index": trace.reset_index,
        "step_index": trace.step_index,
        "scene_signature": str(trace.scene_graph.get("signature", "")),
        "available_action_names": list(trace.available_action_names),
        "selected_action_name": trace.selected_action_name,
        "selected_action_data": _json_safe(trace.selected_action_data),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
        (
            "AMENDED_AFTER_PARTIAL_INVALID_OUTPUTS_FOR_STORAGE_"
            "AND_SAMPLING_CORRECTION"
        ),
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
    "compact_existing_collection",
    "compact_trace_views",
    "graph_from_mapping",
    "graph_to_mapping",
    "load_frozen_manifest",
    "manifest_checksum",
    "read_trace_shard",
    "shard_metadata",
]
