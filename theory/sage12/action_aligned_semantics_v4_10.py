"""SAGE12 V4.10 action-aligned semantic transfer.

V4.10 keeps the validated V4.9 post-transition teacher, but replaces compass
relations with intervention-relative axes and trains a game-balanced invariant
DeepSets student.  Fresh traces, when present, are source-only and are compiled
with the same frozen teacher.  All evaluation predictions remain outer
leave-one-game-out.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN

from .action_target_data import ActionTargetTrace
from .compiler import SLOT_EFFECTS
from .integration_pilot import load_complete_roots
from .integration_pilot_v4_7 import load_slot_examples
from .object_relative_student_v4_9 import (
    _action_only_probabilities,
    _batch_arrays,
    _brier_metrics,
    _completion_recall_at_8,
    _identity_probe,
    _masked_effect_loss,
    _pair_ranking_metrics,
    _per_game_brier,
    _select_device,
    _torch_model,
    tensorize_records,
)
from .semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
    PairLink,
    SemanticTeacherRecord,
    _checksum,
    _file_sha256,
    _json_safe,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    build_object_relative_graph,
    compile_semantics,
    validate_model_graph,
)
from .semantic_teacher_v4_9 import (
    load_pair_links as load_v49_pair_links,
)
from .semantic_teacher_v4_9 import (
    load_teacher_records as load_v49_records,
)

FORMAT_VERSION = "sage12-action-aligned-teacher-record-v4.10"
MANIFEST_VERSION = "sage12-action-aligned-manifest-v4.10"
RESULT_VERSION = "sage12-action-aligned-student-result-v4.10"
PREDICTION_VERSION = "sage12-action-aligned-logo-prediction-v4.10"
SLOT_EXPORT_VERSION = "sage12-action-aligned-slot-annotations-v4.10"
CAPACITY_AMENDMENT_VERSION = "sage12-action-aligned-capacity-amendment-v4.10"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "action_aligned_semantics_v4_10"
DEFAULT_V49_DIR = Path("training") / "sage12" / "object_relative_teacher_v4_9"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"

_DIRECTION_VECTOR = {
    "north": (-1.0, 0.0),
    "north_east": (-1.0, 1.0),
    "east": (0.0, 1.0),
    "south_east": (1.0, 1.0),
    "south": (1.0, 0.0),
    "south_west": (1.0, -1.0),
    "west": (0.0, -1.0),
    "north_west": (-1.0, -1.0),
}
_REQUEST_VECTOR = {
    "up": (-1.0, 0.0),
    "down": (1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
}
_AXIS_SHUFFLE = {
    "ahead": "lateral_right",
    "lateral_right": "behind",
    "behind": "lateral_left",
    "lateral_left": "ahead",
    "overlap": "overlap",
    "radial": "radial",
}


def _axis_vector(root: Mapping[str, Any]) -> tuple[float, float] | None:
    requested = str(root.get("requested_direction", "none"))
    if requested in _REQUEST_VECTOR:
        return _REQUEST_VECTOR[requested]
    actor_direction = str(root.get("actor_relative_direction", "unknown"))
    return _DIRECTION_VECTOR.get(actor_direction)


def _axis_relation(
    direction: str,
    axis: tuple[float, float] | None,
) -> str:
    if direction == "overlap":
        return "overlap"
    vector = _DIRECTION_VECTOR.get(direction)
    if vector is None or axis is None:
        return "radial"
    norm = math.hypot(*axis) * math.hypot(*vector)
    if norm <= 0.0:
        return "radial"
    forward = (axis[0] * vector[0] + axis[1] * vector[1]) / norm
    right_axis = (axis[1], -axis[0])
    right = (right_axis[0] * vector[0] + right_axis[1] * vector[1]) / norm
    if abs(forward) >= abs(right):
        return "ahead" if forward >= 0.0 else "behind"
    return "lateral_right" if right >= 0.0 else "lateral_left"


def action_aligned_graph(
    graph: ObjectRelativeGraph,
    *,
    relation_shuffle: bool = False,
) -> ObjectRelativeGraph:
    """Remove compass tokens and express topology in the intervention frame."""

    source_root = dict(graph.root)
    axis = _axis_vector(source_root)
    root = {
        key: value
        for key, value in source_root.items()
        if key
        not in {
            "requested_direction",
            "actor_relative_direction",
        }
    }
    root["axis_source"] = (
        "movement"
        if str(source_root.get("requested_direction")) in _REQUEST_VECTOR
        else "actor_to_target"
        if axis is not None
        else "none"
    )
    neighbors = []
    contact_count = 0
    adjacent_count = 0
    actor_present = False
    ahead_contact = False
    for raw in graph.neighbors:
        relation = _axis_relation(str(raw.get("direction", "none")), axis)
        if relation_shuffle:
            relation = _AXIS_SHUFFLE[relation]
        proximity = str(raw.get("proximity", "far"))
        contact_count += int(proximity == "contact")
        adjacent_count += int(proximity == "adjacent")
        actor_present = actor_present or bool(raw.get("is_actor", 0))
        ahead_contact = ahead_contact or bool(
            relation == "ahead" and proximity in {"contact", "adjacent"}
        )
        neighbors.append(
            {
                "axis_relation": relation,
                "topology": (
                    "root_contact"
                    if proximity == "contact"
                    else "root_adjacent"
                    if proximity == "adjacent"
                    else "near"
                    if proximity in {"near", "mid"}
                    else "distant"
                ),
                "relative_size": raw.get("relative_size", "unknown"),
                "area_bucket": raw.get("area_bucket", "none"),
                "aspect_bucket": raw.get("aspect_bucket", "none"),
                "is_actor": int(bool(raw.get("is_actor", 0))),
                "touches_boundary": int(bool(raw.get("touches_boundary", 0))),
            }
        )
    root.update(
        {
            "contact_degree": _count_bucket(contact_count),
            "adjacent_degree": _count_bucket(adjacent_count),
            "actor_neighbor_present": int(actor_present),
            "ahead_contact": int(ahead_contact),
        }
    )
    aligned = ObjectRelativeGraph(root=root, neighbors=tuple(neighbors))
    validate_action_aligned_graph(aligned)
    return aligned


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "few"
    return "many"


def validate_action_aligned_graph(graph: ObjectRelativeGraph) -> None:
    validate_model_graph(graph)
    rendered = json.dumps(graph.to_dict(), sort_keys=True).lower()
    for compass in (
        '"north"',
        '"north_east"',
        '"east"',
        '"south_east"',
        '"south"',
        '"south_west"',
        '"west"',
        '"north_west"',
    ):
        if compass in rendered:
            raise ValueError(f"compass relation leaked into V4.10 graph: {compass}")
    neighbor_fields = {key for neighbor in graph.neighbors for key in neighbor}
    if "direction" in neighbor_fields:
        raise ValueError("raw relation direction leaked into V4.10 graph")


def _aligned_record(
    record: SemanticTeacherRecord,
    *,
    relation_shuffle: bool = False,
) -> SemanticTeacherRecord:
    return replace(
        record,
        graph=action_aligned_graph(
            record.graph,
            relation_shuffle=relation_shuffle,
        ),
        format_version=FORMAT_VERSION,
    )


def _source_fingerprints(v49_dir: Path) -> dict[str, Any]:
    paths = (
        v49_dir / "teacher_corpus.jsonl",
        v49_dir / "same_prestate_pairs.jsonl",
        v49_dir / "teacher_qa.json",
    )
    return {
        path.name: {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    }


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v49_dir: str | Path = DEFAULT_V49_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    collection_quotas = {game: 64 if game == "lp85" else 160 for game in SOURCE_TRAIN}
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "source_games": list(SOURCE_TRAIN),
        "source_fingerprints": _source_fingerprints(Path(v49_dir)),
        "representation": {
            "kind": "action_aligned_contact_topology",
            "compass_relations_excluded": True,
            "axis_relations": [
                "ahead",
                "behind",
                "lateral_left",
                "lateral_right",
                "overlap",
                "radial",
            ],
            "absolute_coordinates_excluded": True,
            "raw_values_and_colors_excluded": True,
            "maximum_neighbors": 16,
        },
        "collection": {
            "source_only": True,
            "rows_per_game": collection_quotas,
            "target_rows": sum(collection_quotas.values()),
            "action_budget_per_reset": 96,
            "maximum_resets_per_game": 40,
            "policy_seeds": [211, 307, 401, 503, 601],
            "exploration_fraction": 0.30,
            "exact_repeat_cap_across_v49_and_v410": 1,
            "target_effects": [
                "local_change",
                "path_opened",
                "path_closed",
                "actor_approached_root",
                "contact_gained",
                "reachable_area_increased",
                "reachable_area_decreased",
                "target_created",
                "target_removed",
                "target_moved",
                "productive",
                "risk",
            ],
            "minimum_rows_ratio_per_game": 0.90,
        },
        "training": {
            "seed": 5_010,
            "hash_buckets": 2048,
            "embedding_width": 32,
            "hidden_width": 96,
            "epochs": 40,
            "samples_per_game_per_epoch": 512,
            "samples_per_game_per_step": 32,
            "learning_rate": 0.0015,
            "weight_decay": 0.0001,
            "identity_adversary_weight": 0.20,
            "latent_alignment_weight": 0.05,
            "output_alignment_weight": 0.10,
            "pairwise_ranking_weight": 0.15,
            "calibration": "game_balanced_prevalence_logit_shift",
        },
        "evaluation": {
            "outer_split": "leave_one_source_train_game_out",
            "confirmatory": False,
            "can_promote_live_authority": False,
            "decision_thresholds": {
                "collection_minimum_rows_ratio_per_game": 0.90,
                "macro_brier_gain_over_action_only_strictly_positive": True,
                "macro_brier_gain_over_root_only_strictly_positive": True,
                "macro_brier_gain_over_v49_strictly_positive": True,
                "productive_pair_accuracy_gain_over_root_only_strictly_positive": True,
                "relation_shuffle_brier_degradation_strictly_positive": True,
                "neighbor_permutation_max_probability_delta": 1e-6,
                "semantic_output_identity_accuracy_maximum": 0.60,
                "identity_accuracy_reduction_from_v49_minimum": 0.15,
                "completion_recall_at_8_minimum": 0.20,
                "nonnegative_games_over_action_only_minimum": 6,
            },
        },
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / "frozen_manifest.json")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.10 manifest")
    expected = str(manifest["manifest_checksum"])
    check = dict(manifest)
    check.pop("manifest_checksum")
    if _checksum(check) != expected:
        raise ValueError("V4.10 manifest checksum mismatch")
    if tuple(manifest["source_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.10 source split drift")
    return manifest


def write_capacity_amendment(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Authorize only the empirically exhausted su15 source shortfall.

    This amendment can be written after collection capacity is known but before
    teacher compilation or model fitting. It cannot change representation,
    training, evaluation, or validation access.
    """

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    collection = _read_json(destination / "collection_manifest.json")
    if collection.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ValueError("V4.10 collection/manifest checksum mismatch")
    reports = dict(collection["reports"])
    original = {
        game: int(manifest["collection"]["rows_per_game"][game])
        for game in SOURCE_TRAIN
    }
    for game in SOURCE_TRAIN:
        if game == "su15":
            continue
        if int(reports[game]["rows"]) < original[game]:
            raise RuntimeError(
                f"capacity amendment cannot cover non-su15 shortfall: {game}"
            )
    su15 = dict(reports["su15"])
    expected_resets = int(manifest["collection"]["maximum_resets_per_game"])
    if int(su15.get("resets_used", 0)) < expected_resets:
        raise RuntimeError("su15 capacity was not exhausted through the frozen resets")
    if int(su15.get("duplicate_rejections", 0)) < 1_000:
        raise RuntimeError("su15 shortfall lacks duplicate-saturation evidence")
    minimums = dict(original)
    minimums["su15"] = 80
    amendment: dict[str, Any] = {
        "format_version": CAPACITY_AMENDMENT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "collection_checksum": collection["collection_checksum"],
        "reason": (
            "su15 exposed only ACTION6 and saturated unique capacity at "
            f"{su15['rows']} rows after {su15['raw_steps']} steps with "
            f"{su15['duplicate_rejections']} exact-repeat rejections"
        ),
        "authorized_minimum_rows_per_game": minimums,
        "authorized_total_rows_minimum": sum(minimums.values()),
        "observed_rows_per_game": {
            row["game_id"]: int(row["rows"]) for row in collection["shards"]
        },
        "representation_changed": False,
        "training_changed": False,
        "evaluation_thresholds_changed": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "model_result_observed": False,
    }
    amendment["checks"] = {
        "su15_exhausted_frozen_budget": True,
        "su15_duplicate_saturation": True,
        "all_other_games_complete": True,
        "amended_minimums_met": all(
            amendment["observed_rows_per_game"][game] >= minimums[game]
            for game in SOURCE_TRAIN
        ),
    }
    amendment["collection_ready_under_amendment"] = all(amendment["checks"].values())
    amendment["amendment_checksum"] = _checksum(amendment)
    _write_json(destination / "capacity_amendment.json", amendment)
    return amendment


def _collection_authority(
    destination: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    collection = _read_json(destination / "collection_manifest.json")
    if collection.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ValueError("V4.10 collection/manifest checksum mismatch")
    if collection.get("collection_ready"):
        return collection, None
    amendment = _read_json(destination / "capacity_amendment.json")
    expected = str(amendment["amendment_checksum"])
    check = dict(amendment)
    check.pop("amendment_checksum")
    if _checksum(check) != expected:
        raise ValueError("V4.10 capacity amendment checksum mismatch")
    if (
        amendment.get("format_version") != CAPACITY_AMENDMENT_VERSION
        or amendment.get("manifest_checksum") != manifest["manifest_checksum"]
        or amendment.get("collection_checksum") != collection["collection_checksum"]
        or not amendment.get("collection_ready_under_amendment")
    ):
        raise RuntimeError("V4.10 capacity amendment is not authoritative")
    return collection, amendment


def _fresh_trace_paths(output_dir: Path) -> list[Path]:
    shard_dir = output_dir / "source_train_shards"
    if not shard_dir.exists():
        return []
    return [shard_dir / f"{game}.jsonl" for game in SOURCE_TRAIN]


def compile_teacher_corpus(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v49_dir: str | Path = DEFAULT_V49_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    collection, amendment = _collection_authority(destination, manifest)
    base_records = list(load_v49_records(v49_dir))
    by_digest = {record.trace_digest: record for record in base_records}
    repeat_keys = {record.exact_repeat_key for record in base_records}
    fresh_counts = {game: 0 for game in SOURCE_TRAIN}
    duplicates = 0
    for path in _fresh_trace_paths(destination):
        if not path.exists():
            continue
        for payload in _read_jsonl(path):
            trace = ActionTargetTrace.from_dict(payload)
            if (
                trace.source_split != "source_train"
                or trace.game_id not in SOURCE_TRAIN
            ):
                raise ValueError("V4.10 fresh trace violates source firewall")
            repeat_key = trace.exact_repeat_key()
            if repeat_key in repeat_keys:
                duplicates += 1
                continue
            labels, applicable, score, evidence = compile_semantics(trace)
            record = SemanticTeacherRecord(
                example_id="sem410_"
                + _checksum(
                    {
                        "trace": trace.trace_digest,
                        "version": FORMAT_VERSION,
                    }
                )[:20],
                game_id=trace.game_id,
                source_corpus="functional_intervention_v4_10",
                trace_digest=trace.trace_digest,
                exact_repeat_key=repeat_key,
                same_prestate_keys=(),
                graph=build_object_relative_graph(trace),
                labels=labels,
                applicable=applicable,
                productive_score=score,
                teacher_evidence=evidence,
            )
            repeat_keys.add(repeat_key)
            by_digest[trace.trace_digest] = record
            fresh_counts[trace.game_id] += 1
    records = [_aligned_record(record) for record in by_digest.values()]
    records.sort(key=lambda record: record.example_id)
    _write_jsonl(
        destination / "teacher_corpus.jsonl",
        (record.to_dict() for record in records),
    )
    pair_links = load_v49_pair_links(v49_dir)
    _write_jsonl(
        destination / "same_prestate_pairs.jsonl",
        (link.to_dict() for link in pair_links),
    )
    capacity = {}
    for effect in SEMANTIC_EFFECTS:
        eligible = [record for record in records if record.applicable[effect]]
        capacity[effect] = {
            "applicable": len(eligible),
            "positive": sum(record.labels[effect] for record in eligible),
            "games_with_positive": sum(
                any(
                    record.labels[effect]
                    for record in eligible
                    if record.game_id == game
                )
                for game in SOURCE_TRAIN
            ),
            "per_game_positive": {
                game: sum(
                    record.labels[effect]
                    for record in eligible
                    if record.game_id == game
                )
                for game in SOURCE_TRAIN
            },
        }
    corpus_path = destination / "teacher_corpus.jsonl"
    pair_path = destination / "same_prestate_pairs.jsonl"
    qa: dict[str, Any] = {
        "format_version": "sage12-action-aligned-teacher-qa-v4.10",
        "manifest_checksum": manifest["manifest_checksum"],
        "collection_checksum": collection["collection_checksum"],
        "capacity_amendment_checksum": (
            amendment["amendment_checksum"] if amendment is not None else None
        ),
        "base_records": len(base_records),
        "fresh_records": sum(fresh_counts.values()),
        "fresh_per_game": fresh_counts,
        "duplicates_against_v49": duplicates,
        "records": len(records),
        "pair_links": len(pair_links),
        "label_capacity": capacity,
        "all_graphs_action_aligned": all(
            _graph_is_action_aligned(record.graph) for record in records
        ),
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "teacher_corpus": _file_sha256(corpus_path),
            "same_prestate_pairs": _file_sha256(pair_path),
        },
    }
    qa["teacher_ready"] = bool(
        qa["all_graphs_action_aligned"]
        and all(record.game_id in SOURCE_TRAIN for record in records)
    )
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def _graph_is_action_aligned(graph: ObjectRelativeGraph) -> bool:
    try:
        validate_action_aligned_graph(graph)
    except ValueError:
        return False
    return True


def load_teacher_records(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[SemanticTeacherRecord, ...]:
    return tuple(
        SemanticTeacherRecord.from_dict(row)
        for row in _read_jsonl(Path(output_dir) / "teacher_corpus.jsonl")
    )


def load_pair_links(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[PairLink, ...]:
    return tuple(
        PairLink(**row)
        for row in _read_jsonl(Path(output_dir) / "same_prestate_pairs.jsonl")
    )


def _game_balanced_effect_loss(
    logits: Any,
    labels: Any,
    masks: Any,
    identities: Any,
) -> Any:
    import torch

    losses = []
    for game_index in torch.unique(identities):
        selected = identities == game_index
        losses.append(
            _masked_effect_loss(
                logits[selected],
                labels[selected],
                masks[selected],
            )
        )
    return torch.stack(losses).mean()


def _alignment_loss(
    latent: Any,
    logits: Any,
    masks: Any,
    identities: Any,
) -> tuple[Any, Any]:
    import torch

    latent_means = []
    output_means = []
    probabilities = torch.sigmoid(logits)
    for game_index in torch.unique(identities):
        selected = identities == game_index
        latent_means.append(latent[selected].mean(dim=0))
        effect_means = []
        for effect_index in range(logits.shape[1]):
            applicable = selected & (masks[:, effect_index] > 0)
            effect_means.append(
                probabilities[applicable, effect_index].mean()
                if applicable.any()
                else probabilities[:, effect_index].mean().detach()
            )
        output_means.append(torch.stack(effect_means))
    latent_matrix = torch.stack(latent_means)
    output_matrix = torch.stack(output_means)
    return (
        torch.mean((latent_matrix - latent_matrix.mean(dim=0)) ** 2),
        torch.mean((output_matrix - output_matrix.mean(dim=0)) ** 2),
    )


def _balanced_epoch_indices(
    records: Sequence[SemanticTeacherRecord],
    train_indices: np.ndarray,
    *,
    samples_per_game: int,
    per_game_step: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    games = sorted({records[index].game_id for index in train_indices})
    sampled = {}
    for game in games:
        indices = np.asarray(
            [index for index in train_indices if records[index].game_id == game],
            dtype=np.int64,
        )
        sampled[game] = rng.choice(
            indices,
            size=samples_per_game,
            replace=len(indices) < samples_per_game,
        )
    batches = []
    for start in range(0, samples_per_game, per_game_step):
        batch = np.concatenate(
            [sampled[game][start : start + per_game_step] for game in games]
        )
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def _fit_invariant_model(
    records: Sequence[SemanticTeacherRecord],
    tensors: Any,
    *,
    train_indices: np.ndarray,
    pair_links: Sequence[PairLink],
    parameters: Mapping[str, Any],
    device: str,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_games = sorted({records[index].game_id for index in train_indices})
    game_to_index = {game: index for index, game in enumerate(train_games)}
    model = _torch_model(
        hash_buckets=int(parameters["hash_buckets"]),
        embedding_width=int(parameters["embedding_width"]),
        hidden_width=int(parameters["hidden_width"]),
        effect_count=len(SEMANTIC_EFFECTS),
        identity_classes=len(train_games),
    ).to(device)
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("identity_head.")
    ]
    optimizer = torch.optim.AdamW(
        main_parameters,
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    adversary_optimizer = torch.optim.AdamW(
        model.identity_head.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    by_digest = {record.trace_digest: index for index, record in enumerate(records)}
    training_set = {int(index) for index in train_indices}
    ranking_pairs = []
    for link in pair_links:
        left = by_digest.get(link.left_trace_digest)
        right = by_digest.get(link.right_trace_digest)
        if (
            left is None
            or right is None
            or left not in training_set
            or right not in training_set
        ):
            continue
        delta = records[left].productive_score - records[right].productive_score
        if abs(delta) > 1e-9:
            ranking_pairs.append((left, right, 1.0 if delta > 0 else -1.0))
    productive_index = SEMANTIC_EFFECTS.index("productive")
    started = time.perf_counter()
    final_losses = {}
    for epoch in range(int(parameters["epochs"])):
        model.train()
        rows = defaultdict(list)
        for batch_indices in _balanced_epoch_indices(
            records,
            train_indices,
            samples_per_game=int(parameters["samples_per_game_per_epoch"]),
            per_game_step=int(parameters["samples_per_game_per_step"]),
            seed=seed + epoch,
        ):
            root, nodes, mask, labels, applicable = _batch_arrays(
                tensors, batch_indices
            )
            root = root.to(device)
            nodes = nodes.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            applicable = applicable.to(device)
            identities = torch.as_tensor(
                [game_to_index[records[index].game_id] for index in batch_indices],
                dtype=torch.long,
                device=device,
            )
            adversary_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, detached = model(root, nodes, mask)
            identity_logits = model.identity_head(detached.detach())
            identity_loss = torch.nn.functional.cross_entropy(
                identity_logits, identities
            )
            identity_loss.backward()
            adversary_optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            logits, latent = model(root, nodes, mask)
            semantic = _game_balanced_effect_loss(
                logits,
                labels,
                applicable,
                identities,
            )
            identity_probabilities = torch.softmax(model.identity_head(latent), dim=-1)
            uniform = torch.full_like(
                identity_probabilities,
                1.0 / identity_probabilities.shape[-1],
            )
            confusion = torch.nn.functional.kl_div(
                torch.log(identity_probabilities.clamp_min(1e-8)),
                uniform,
                reduction="batchmean",
            )
            latent_alignment, output_alignment = _alignment_loss(
                latent,
                logits,
                applicable,
                identities,
            )
            loss = (
                semantic
                + float(parameters["identity_adversary_weight"]) * confusion
                + float(parameters["latent_alignment_weight"]) * latent_alignment
                + float(parameters["output_alignment_weight"]) * output_alignment
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, 2.0)
            optimizer.step()
            rows["semantic"].append(float(semantic.detach().cpu()))
            rows["identity_confusion"].append(float(confusion.detach().cpu()))
            rows["latent_alignment"].append(float(latent_alignment.detach().cpu()))
            rows["output_alignment"].append(float(output_alignment.detach().cpu()))

        ranking_losses = []
        shuffled_pairs = list(ranking_pairs)
        random.Random(seed + 10_000 + epoch).shuffle(shuffled_pairs)
        batch_size = int(parameters["samples_per_game_per_step"]) * len(train_games)
        for start in range(0, len(shuffled_pairs), batch_size):
            chunk = shuffled_pairs[start : start + batch_size]
            left_indices = np.asarray([row[0] for row in chunk], dtype=np.int64)
            right_indices = np.asarray([row[1] for row in chunk], dtype=np.int64)
            signs = torch.as_tensor(
                [row[2] for row in chunk],
                dtype=torch.float32,
                device=device,
            )
            left = _batch_arrays(tensors, left_indices)
            right = _batch_arrays(tensors, right_indices)
            optimizer.zero_grad(set_to_none=True)
            left_logits, _ = model(
                left[0].to(device), left[1].to(device), left[2].to(device)
            )
            right_logits, _ = model(
                right[0].to(device), right[1].to(device), right[2].to(device)
            )
            ranking = torch.nn.functional.softplus(
                -signs
                * (left_logits[:, productive_index] - right_logits[:, productive_index])
            ).mean()
            (float(parameters["pairwise_ranking_weight"]) * ranking).backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, 2.0)
            optimizer.step()
            ranking_losses.append(float(ranking.detach().cpu()))
        final_losses = {key: float(np.mean(values)) for key, values in rows.items()}
        final_losses["pairwise_ranking"] = (
            float(np.mean(ranking_losses)) if ranking_losses else 0.0
        )

    train_logits = _predict_logits(
        model,
        tensors,
        train_indices,
        device=device,
        batch_size=512,
    )
    shifts = _game_balanced_calibration_shifts(
        records,
        train_indices,
        train_logits,
    )
    return (
        model,
        shifts,
        {
            "runtime_seconds": time.perf_counter() - started,
            "train_rows": len(train_indices),
            "train_games": train_games,
            "ranking_pairs": len(ranking_pairs),
            "calibration_shifts": {
                effect: float(shifts[index])
                for index, effect in enumerate(SEMANTIC_EFFECTS)
            },
            "final_losses": final_losses,
        },
    )


def _predict_logits(
    model: Any,
    tensors: Any,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            root, nodes, mask, _labels, _applicable = _batch_arrays(tensors, selected)
            logits, _ = model(root.to(device), nodes.to(device), mask.to(device))
            rows.append(logits.cpu().numpy())
    return np.concatenate(rows, axis=0)


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
    return math.log(value / (1.0 - value))


def _game_balanced_calibration_shifts(
    records: Sequence[SemanticTeacherRecord],
    train_indices: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    shifts = np.zeros(len(SEMANTIC_EFFECTS), dtype=np.float64)
    lookup = {
        int(global_index): local_index
        for local_index, global_index in enumerate(train_indices)
    }
    games = sorted({records[index].game_id for index in train_indices})
    for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
        target_rates = []
        predicted_rates = []
        for game in games:
            applicable = [
                index
                for index in train_indices
                if records[index].game_id == game and records[index].applicable[effect]
            ]
            if not applicable:
                continue
            target_rates.append(
                float(np.mean([records[index].labels[effect] for index in applicable]))
            )
            predicted_rates.append(
                float(
                    np.mean(
                        [
                            1.0
                            / (
                                1.0
                                + math.exp(-float(logits[lookup[index], effect_index]))
                            )
                            for index in applicable
                        ]
                    )
                )
            )
        target = float(np.mean(target_rates)) if target_rates else 0.5
        predicted = float(np.mean(predicted_rates)) if predicted_rates else 0.5
        shifts[effect_index] = _logit(target) - _logit(predicted)
    return shifts


def _calibrated_probabilities(logits: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    values = np.clip(logits + shifts.reshape(1, -1), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _prediction_rows(
    records: Sequence[SemanticTeacherRecord],
    original: np.ndarray,
    root_only: np.ndarray,
    action_only: np.ndarray,
    shuffled: np.ndarray,
) -> Sequence[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        probabilities = {}
        for name, matrix in (
            ("action_aligned", original),
            ("root_only", root_only),
            ("action_only", action_only),
            ("relation_shuffle", shuffled),
        ):
            probabilities[name] = {
                effect: float(matrix[index, effect_index])
                for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
            }
        rows.append(
            {
                "format_version": PREDICTION_VERSION,
                "example_id": record.example_id,
                "trace_digest": record.trace_digest,
                "game_id": record.game_id,
                "probabilities": probabilities,
            }
        )
    return rows


def evaluate_student(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v49_dir: str | Path = DEFAULT_V49_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa.get("teacher_ready"):
        raise RuntimeError("V4.10 teacher corpus is not ready")
    records = load_teacher_records(destination)
    pair_links = load_pair_links(destination)
    parameters = manifest["training"]
    selected_device = _select_device(device)
    maximum_neighbors = int(manifest["representation"]["maximum_neighbors"])
    tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
    )
    root_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="root_only",
    )
    shuffled_records = tuple(
        replace(
            record,
            graph=ObjectRelativeGraph(
                root=dict(record.graph.root),
                neighbors=tuple(
                    {
                        **dict(neighbor),
                        "axis_relation": _AXIS_SHUFFLE[str(neighbor["axis_relation"])],
                    }
                    for neighbor in record.graph.neighbors
                ),
            ),
        )
        for record in records
    )
    shuffled_tensors = tensorize_records(
        shuffled_records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
    )
    reversed_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
        reverse_neighbors=True,
    )
    shape = (len(records), len(SEMANTIC_EFFECTS))
    original = np.zeros(shape, dtype=np.float64)
    root_only = np.zeros(shape, dtype=np.float64)
    action_only = np.zeros(shape, dtype=np.float64)
    relation_shuffle = np.zeros(shape, dtype=np.float64)
    reversed_predictions = np.zeros(shape, dtype=np.float64)
    folds = []
    for fold_index, held_out_game in enumerate(SOURCE_TRAIN):
        train_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id != held_out_game
            ],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id == held_out_game
            ],
            dtype=np.int64,
        )
        full_model, full_shifts, full_runtime = _fit_invariant_model(
            records,
            tensors,
            train_indices=train_indices,
            pair_links=pair_links,
            parameters=parameters,
            device=selected_device,
            seed=int(parameters["seed"]) + fold_index * 100,
        )
        root_model, root_shifts, root_runtime = _fit_invariant_model(
            records,
            root_tensors,
            train_indices=train_indices,
            pair_links=pair_links,
            parameters=parameters,
            device=selected_device,
            seed=int(parameters["seed"]) + fold_index * 100 + 1,
        )
        original[test_indices] = _calibrated_probabilities(
            _predict_logits(
                full_model,
                tensors,
                test_indices,
                device=selected_device,
                batch_size=512,
            ),
            full_shifts,
        )
        relation_shuffle[test_indices] = _calibrated_probabilities(
            _predict_logits(
                full_model,
                shuffled_tensors,
                test_indices,
                device=selected_device,
                batch_size=512,
            ),
            full_shifts,
        )
        reversed_predictions[test_indices] = _calibrated_probabilities(
            _predict_logits(
                full_model,
                reversed_tensors,
                test_indices,
                device=selected_device,
                batch_size=512,
            ),
            full_shifts,
        )
        root_only[test_indices] = _calibrated_probabilities(
            _predict_logits(
                root_model,
                root_tensors,
                test_indices,
                device=selected_device,
                batch_size=512,
            ),
            root_shifts,
        )
        action_only[test_indices] = _action_only_probabilities(
            records,
            train_indices,
            test_indices,
        )
        folds.append(
            {
                "held_out_game": held_out_game,
                "training_rows": len(train_indices),
                "validation_rows": len(test_indices),
                "action_aligned": full_runtime,
                "root_only": root_runtime,
            }
        )
    prediction_path = destination / "logo_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        _prediction_rows(
            records,
            original,
            root_only,
            action_only,
            relation_shuffle,
        ),
    )
    metrics = {
        "action_aligned": _brier_metrics(records, original),
        "root_only": _brier_metrics(records, root_only),
        "action_only": _brier_metrics(records, action_only),
        "relation_shuffle": _brier_metrics(records, relation_shuffle),
    }
    pair_metrics = {
        "action_aligned": _pair_ranking_metrics(records, pair_links, original),
        "root_only": _pair_ranking_metrics(records, pair_links, root_only),
        "action_only": _pair_ranking_metrics(records, pair_links, action_only),
        "relation_shuffle": _pair_ranking_metrics(
            records, pair_links, relation_shuffle
        ),
    }
    identity = _identity_probe(records, original)
    completion = _completion_recall_at_8(records, original)
    permutation_delta = float(np.max(np.abs(original - reversed_predictions)))
    per_game = _per_game_brier(records, original)
    action_per_game = _per_game_brier(records, action_only)
    nonnegative_games = sum(
        per_game[game]["macro_brier"] <= action_per_game[game]["macro_brier"]
        for game in SOURCE_TRAIN
    )
    v49_result = _read_json(Path(v49_dir) / "student_result.json")
    thresholds = manifest["evaluation"]["decision_thresholds"]
    gains = {
        "over_action_only": (
            metrics["action_only"]["macro_brier"]
            - metrics["action_aligned"]["macro_brier"]
        ),
        "over_root_only": (
            metrics["root_only"]["macro_brier"]
            - metrics["action_aligned"]["macro_brier"]
        ),
        "over_v49": (
            float(v49_result["metrics"]["object_relative"]["macro_brier"])
            - metrics["action_aligned"]["macro_brier"]
        ),
        "relation_shuffle_degradation": (
            metrics["relation_shuffle"]["macro_brier"]
            - metrics["action_aligned"]["macro_brier"]
        ),
    }
    checks = {
        "teacher_ready": bool(qa["teacher_ready"]),
        "macro_brier_gain_over_action_only_strictly_positive": (
            gains["over_action_only"] > 0.0
        ),
        "macro_brier_gain_over_root_only_strictly_positive": (
            gains["over_root_only"] > 0.0
        ),
        "macro_brier_gain_over_v49_strictly_positive": (gains["over_v49"] > 0.0),
        "productive_pair_accuracy_gain_over_root_only_strictly_positive": (
            pair_metrics["action_aligned"]["accuracy"]
            > pair_metrics["root_only"]["accuracy"]
        ),
        "relation_shuffle_brier_degradation_strictly_positive": (
            gains["relation_shuffle_degradation"] > 0.0
        ),
        "neighbor_permutation_invariance": (
            permutation_delta
            <= float(thresholds["neighbor_permutation_max_probability_delta"])
        ),
        "semantic_output_identity_accuracy_maximum": (
            identity["accuracy"]
            <= float(thresholds["semantic_output_identity_accuracy_maximum"])
        ),
        "identity_accuracy_reduction_from_v49_minimum": (
            float(v49_result["semantic_output_game_identity_probe"]["accuracy"])
            - identity["accuracy"]
            >= float(thresholds["identity_accuracy_reduction_from_v49_minimum"])
        ),
        "completion_recall_at_8_minimum": (
            completion["recall_at_8"]
            >= float(thresholds["completion_recall_at_8_minimum"])
        ),
        "nonnegative_games_over_action_only_minimum": (
            nonnegative_games
            >= int(thresholds["nonnegative_games_over_action_only_minimum"])
        ),
    }
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "teacher_qa_checksum": qa["qa_checksum"],
        "verdict": (
            "ACTION_ALIGNED_INVARIANT_SEMANTICS_SUPPORTED"
            if all(checks.values())
            else "ACTION_ALIGNED_INVARIANT_SEMANTICS_NOT_YET_SUPPORTED"
        ),
        "checks": checks,
        "confirmatory": False,
        "authority_promoted": False,
        "device": selected_device,
        "records": len(records),
        "fresh_records": qa["fresh_records"],
        "metrics": metrics,
        "macro_brier_gain": gains,
        "productive_pair_ranking": pair_metrics,
        "semantic_output_game_identity_probe": identity,
        "v49_identity_accuracy": v49_result["semantic_output_game_identity_probe"][
            "accuracy"
        ],
        "completion_recall_at_8": completion,
        "neighbor_permutation_max_probability_delta": permutation_delta,
        "nonnegative_games_over_action_only": nonnegative_games,
        "per_game_transfer": per_game,
        "action_only_per_game": action_per_game,
        "folds": folds,
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "teacher_corpus": _file_sha256(destination / "teacher_corpus.jsonl"),
            "logo_predictions": _file_sha256(prediction_path),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "student_result.json", result)
    export_v47_annotations(output_dir=destination)
    return result


def export_v47_annotations(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    lookup = {
        str(row["trace_digest"]): dict(row["probabilities"])
        for row in _read_jsonl(destination / "logo_predictions.jsonl")
    }
    roots = load_complete_roots(v43_dir)
    examples = load_slot_examples(roots)
    by_position = {(item.root_key, item.path, item.side): item for item in examples}
    rows = []
    for root in roots:
        for path, pair in sorted(root.tree.items()):
            for side, arm in zip("LR", (pair.left, pair.right)):
                item = by_position[(root.root_key, path, side)]
                prediction = lookup.get(arm.trace.trace_digest)
                if prediction is None:
                    raise ValueError("V4.3 slot lacks V4.10 LOGO prediction")
                for variant, source in (
                    ("action_aligned", "action_aligned_invariant_logo_v4_10"),
                    (
                        "relation_shuffle",
                        "action_aligned_relation_shuffle_logo_v4_10",
                    ),
                ):
                    rows.append(
                        {
                            "format_version": SLOT_EXPORT_VERSION,
                            "slot_id": item.slot.slot_id,
                            "example_id": item.example_id,
                            "game_id": item.game_id,
                            "variant": variant,
                            "effect_probabilities": {
                                effect: float(prediction[variant][effect])
                                for effect in SLOT_EFFECTS
                            },
                            "source": source,
                            "support": 0,
                        }
                    )
    path = destination / "v4_7_slot_annotations.jsonl"
    _write_jsonl(path, rows)
    summary: dict[str, Any] = {
        "format_version": SLOT_EXPORT_VERSION,
        "slots": len(examples),
        "rows": len(rows),
        "variants": ["action_aligned", "relation_shuffle"],
        "missing": 0,
        "sha256": _file_sha256(path),
    }
    summary["checksum"] = _checksum(summary)
    _write_json(destination / "v4_7_slot_export.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--device", default="cuda:0")
    amend = subparsers.add_parser("amend-capacity")
    amend.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "amend-capacity":
        payload = write_capacity_amendment(output_dir=args.output_dir)
    elif args.command == "compile":
        payload = compile_teacher_corpus(output_dir=args.output_dir)
    else:
        payload = evaluate_student(
            output_dir=args.output_dir,
            device=args.device,
        )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "action_aligned_graph",
    "compile_teacher_corpus",
    "evaluate_student",
    "export_v47_annotations",
    "freeze_manifest",
    "load_manifest",
    "load_pair_links",
    "load_teacher_records",
    "validate_action_aligned_graph",
    "write_capacity_amendment",
]
