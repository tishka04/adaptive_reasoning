"""SAGE12 V4.15 demonstration-conditioned milestone policy.

Commands::

    python -m theory.sage12.demonstration_milestone_policy_v4_15 freeze
    python -m theory.sage12.demonstration_milestone_policy_v4_15 compile
    python -m theory.sage12.demonstration_milestone_policy_v4_15 train
    python -m theory.sage12.demonstration_milestone_policy_v4_15 evaluate
    python -m theory.sage12.demonstration_milestone_policy_v4_15 active
    python -m theory.sage12.demonstration_milestone_policy_v4_15 run-all

The policy learns candidate choice from complete human prefixes.  Milestone
and return-to-go targets are teacher-only during fitting; the deployable lane
predicts its own milestone and requests a high return.  Every diagnostic lane
continues into the global and active panels regardless of local results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from theory.live_transition_loop import build_observation
from theory.sage11.splits import NEURO_HOLDOUT_V1

from .action_target_data import (
    ActionTargetAnchor,
    build_action_target_trace,
    grid_sha256,
    resolve_action_target,
)
from .counterfactual_semantic_panels_v4_11 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V411_DIR,
)
from .counterfactual_semantic_panels_v4_11 import (
    load_teacher_panels,
)
from .human_temporal_semantics_v4_14 import (
    ACTIVE_VALIDATION_GAMES,
    HUMAN_TRAIN_GAMES,
    SEMANTIC_EFFECTS,
    TRANSFER_GAMES,
    TemporalBeliefState,
    _action_names,
    _action_sequence_tables,
    _candidate_action_plan,
    _episode_summaries,
    _file_fingerprint,
    _future_targets,
    _graph_for_action,
    _identity_probe,
    _json_safe,
    _live_action_signature,
    _live_candidate_graph,
    _load_active_ebm,
    _paired_bootstrap_rows,
    _predict_candidate_rollouts,
    _prediction_features,
    _read_jsonl,
    _step_rows,
    _summarize_decisions,
    _tensorize_graph_batch,
)
from .human_temporal_semantics_v4_14 import (
    _load_checkpoint as _load_temporal_checkpoint,
)
from .human_temporal_semantics_v4_14 import (
    load_teacher_records as load_temporal_records,
)
from .object_relative_student_v4_9 import _token_id, _tokens
from .semantic_teacher_v4_9 import (
    ObjectRelativeGraph,
    _area_bucket,
    _aspect_bucket,
    _bbox_gap_to_point,
    _boundary_bucket,
    _checksum,
    _direction,
    _distance_bucket,
    _file_sha256,
    _object_at,
    _read_json,
    _write_json,
    _write_jsonl,
    build_object_relative_graph,
    compile_semantics,
    validate_model_graph,
)

FORMAT_VERSION = "sage12-demonstration-milestone-policy-v4.15"
MANIFEST_VERSION = "sage12-demonstration-milestone-manifest-v4.15"
RECORD_VERSION = "sage12-demonstration-choice-record-v4.15"
CHECKPOINT_VERSION = "sage12-demonstration-policy-checkpoint-v4.15"
RESULT_VERSION = "sage12-demonstration-policy-result-v4.15"

DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage12" / "demonstration_milestone_policy_v4_15"
)
DEFAULT_TRACES_DIR = Path("human_traces")
DEFAULT_V414_DIR = Path("training") / "sage12" / "human_temporal_semantics_v4_14"
SEED = 5_150
MAXIMUM_CANDIDATES = 16
MAXIMUM_NEIGHBORS = 16
MAX_HISTORY = 32
MILESTONE_HORIZON = 64
EBM_COEFFICIENT = 0.5

MILESTONE_LABELS = (
    "level_complete",
    "path_opened",
    "target_removed",
    "target_created",
    "target_moved",
    "actor_approached_root",
    "productive",
    "none_within_64",
)
MILESTONE_PRIORITY = MILESTONE_LABELS[:-1]
FORBIDDEN_STUDENT_FIELDS = (
    "game_id",
    "episode_id",
    "source_file",
    "pre_state_sha256",
    "post_state_sha256",
    "frame_before",
    "frame_after",
    "row",
    "col",
    "x",
    "y",
    "object_id",
    "target_object_id",
    "intent",
    "hypothesis",
    "objective_guess",
    "game_type_guess",
    "color",
    "colour",
    "value",
)


def _trace_paths(trace_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(trace_dir.glob("*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no human trace files under {trace_dir}")
    return paths


def _source_fingerprints(
    trace_dir: Path,
    v414_dir: Path,
) -> dict[str, Any]:
    paths = list(_trace_paths(trace_dir))
    paths.extend(
        (
            v414_dir / "frozen_manifest.json",
            v414_dir / "teacher_qa.json",
            v414_dir / "semantic_result.json",
            v414_dir / "checkpoint_metadata.json",
            v414_dir / "transfer_predictions.jsonl",
            v414_dir / "transfer_decisions.jsonl",
            v414_dir / "active_validation.json",
            v414_dir / "active_runs.jsonl",
            v414_dir / "result.json",
        )
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(path.as_posix() for path in missing))
    return {path.as_posix(): _file_fingerprint(path) for path in paths}


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    traces_dir: str | Path = DEFAULT_TRACES_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "split": {
            "human_train": list(HUMAN_TRAIN_GAMES),
            "transfer_evaluation": list(TRANSFER_GAMES),
            "active_validation": list(ACTIVE_VALIDATION_GAMES),
            "final_confirmation_closed": list(NEURO_HOLDOUT_V1),
            "legacy_sage11_registry_unchanged": True,
        },
        "source_fingerprints": _source_fingerprints(
            Path(traces_dir),
            Path(v414_dir),
        ),
        "compiler": {
            "maximum_candidates": MAXIMUM_CANDIDATES,
            "negative_candidates": (
                "all recorded legal action names plus deterministic "
                "object-centre click anchors"
            ),
            "candidate_order": "identity-free graph checksum",
            "milestone_horizon": MILESTONE_HORIZON,
            "milestone_labels": list(MILESTONE_LABELS),
            "success_weight_cap": 5.0,
        },
        "model": {
            "hash_buckets": 2048,
            "embedding_width": 32,
            "graph_hidden_width": 96,
            "temporal_hidden_width": 128,
            "milestone_embedding_width": 16,
            "maximum_neighbors": MAXIMUM_NEIGHBORS,
            "maximum_history": MAX_HISTORY,
            "epochs": 30,
            "learning_rate": 0.0015,
            "weight_decay": 0.0001,
            "loss_weights": {
                "behavior_cloning": 1.0,
                "success_conditioned_cloning": 1.0,
                "milestone": 0.5,
                "distance": 0.25,
                "return": 0.25,
            },
            "seed": SEED,
            "outer_diagnostic": "leave_one_human_game_out",
            "final_fit": "all_six_human_games",
        },
        "evaluation": {
            "bootstrap_samples": 1_000,
            "bootstrap_seed": SEED + 1,
            "behavior_nonnegative_games_minimum": 4,
            "milestone_balanced_gain_minimum": 0.05,
            "identity_gain_maximum": 0.20,
            "global_nonnegative_games_minimum": 5,
            "completion_fraction_minimum": 0.5,
            "completion_absolute_minimum": 1,
            "ebm_coefficient": EBM_COEFFICIENT,
            "active_seeds": [0, 1, 2],
            "active_action_budget": 1_000,
            "active_maximum_resets": 14,
            "all_conditions_run_unconditionally": True,
        },
        "forbidden_student_fields": list(FORBIDDEN_STUDENT_FIELDS),
        "holdout_opened": False,
        "authority_promoted": False,
        "result_observed_at_freeze": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    payload = _read_json(Path(output_dir) / "frozen_manifest.json")
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.15 manifest")
    expected = str(payload["manifest_checksum"])
    clean = dict(payload)
    clean.pop("manifest_checksum")
    if _checksum(clean) != expected:
        raise ValueError("V4.15 manifest checksum mismatch")
    if tuple(payload["split"]["human_train"]) != HUMAN_TRAIN_GAMES:
        raise ValueError("V4.15 human split drift")
    if tuple(payload["split"]["transfer_evaluation"]) != TRANSFER_GAMES:
        raise ValueError("V4.15 transfer split drift")
    return payload


def _assert_student_view_safe(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).lower()
    for field_name in FORBIDDEN_STUDENT_FIELDS:
        if f'"{field_name}"' in encoded:
            raise ValueError(f"forbidden V4.15 student field: {field_name}")


@dataclass(frozen=True)
class DemonstrationChoiceRecord:
    example_id: str
    game_id: str
    episode_id: str
    sequence_index: int
    source_file: str
    pre_state_sha256: str
    post_state_sha256: str
    candidates: tuple[ObjectRelativeGraph, ...]
    selected_index: int
    observed_effects: Mapping[str, bool]
    milestone: str
    milestone_distance: int
    suffix_return: float
    success_weight: float
    within_16: bool
    productive: bool
    actual_action_name: str
    actual_action_data: Mapping[str, Any]
    format_version: str = RECORD_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RECORD_VERSION:
            raise ValueError("unsupported V4.15 record")
        if not self.candidates:
            raise ValueError("V4.15 choice record has no candidates")
        if not 0 <= self.selected_index < len(self.candidates):
            raise ValueError("V4.15 selected candidate is out of range")
        if self.milestone not in MILESTONE_LABELS:
            raise ValueError("unknown V4.15 milestone")
        _assert_student_view_safe(self.student_view())

    @property
    def sequence_key(self) -> str:
        return f"{self.game_id}:{self.episode_id}"

    @property
    def selected_graph(self) -> ObjectRelativeGraph:
        return self.candidates[self.selected_index]

    def student_view(self) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "root": dict(graph.root),
                    "neighbors": [dict(row) for row in graph.neighbors],
                }
                for graph in self.candidates
            ],
            "desired_return": float(self.suffix_return),
            "milestone_token": self.milestone,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "student_view": self.student_view(),
            "teacher": {
                "selected_index": self.selected_index,
                "observed_effects": dict(self.observed_effects),
                "milestone": self.milestone,
                "milestone_distance": self.milestone_distance,
                "suffix_return": self.suffix_return,
                "success_weight": self.success_weight,
                "within_16": self.within_16,
                "productive": self.productive,
            },
            "audit": {
                "example_id": self.example_id,
                "game_id": self.game_id,
                "episode_id": self.episode_id,
                "sequence_index": self.sequence_index,
                "source_file": self.source_file,
                "pre_state_sha256": self.pre_state_sha256,
                "post_state_sha256": self.post_state_sha256,
                "actual_action_name": self.actual_action_name,
                "actual_action_data": _json_safe(self.actual_action_data),
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> DemonstrationChoiceRecord:
        view = payload["student_view"]
        teacher = payload["teacher"]
        audit = payload["audit"]
        return cls(
            example_id=str(audit["example_id"]),
            game_id=str(audit["game_id"]),
            episode_id=str(audit["episode_id"]),
            sequence_index=int(audit["sequence_index"]),
            source_file=str(audit["source_file"]),
            pre_state_sha256=str(audit["pre_state_sha256"]),
            post_state_sha256=str(audit["post_state_sha256"]),
            candidates=tuple(
                ObjectRelativeGraph(
                    root=dict(row["root"]),
                    neighbors=tuple(dict(neighbor) for neighbor in row["neighbors"]),
                )
                for row in view["candidates"]
            ),
            selected_index=int(teacher["selected_index"]),
            observed_effects={
                effect: bool(teacher["observed_effects"][effect])
                for effect in SEMANTIC_EFFECTS
            },
            milestone=str(teacher["milestone"]),
            milestone_distance=int(teacher["milestone_distance"]),
            suffix_return=float(teacher["suffix_return"]),
            success_weight=float(teacher["success_weight"]),
            within_16=bool(teacher["within_16"]),
            productive=bool(teacher["productive"]),
            actual_action_name=str(audit["actual_action_name"]),
            actual_action_data=dict(audit.get("actual_action_data") or {}),
            format_version=str(payload["format_version"]),
        )


def _graph_payload(graph: ObjectRelativeGraph) -> dict[str, Any]:
    return {
        "root": dict(graph.root),
        "neighbors": [dict(row) for row in graph.neighbors],
    }


def _candidate_graph(
    *,
    game: str,
    sequence_index: int,
    available: Sequence[str],
    action_name: str,
    action_data: Mapping[str, Any],
    frame: Any,
    game_state: str,
    levels_completed: int,
) -> ObjectRelativeGraph:
    trace = build_action_target_trace(
        game_id=game,
        source_split="source_train",
        policy_seed=SEED,
        reset_index=0,
        step_index=sequence_index,
        collection_phase="human_candidate_v4_15",
        available_action_names=available,
        selected_action_name=action_name,
        selected_action_data=action_data,
        frame_before=frame,
        frame_after=frame,
        game_state_before=game_state,
        game_state_after=game_state,
        levels_completed_before=levels_completed,
        levels_completed_after=levels_completed,
    )
    return build_object_relative_graph(
        trace,
        maximum_neighbors=MAXIMUM_NEIGHBORS,
    )


def _candidate_graph_from_observation(
    *,
    game: str,
    observation: Any,
    anchor: ActionTargetAnchor,
    action_name: str,
) -> ObjectRelativeGraph:
    """Build the canonical graph without re-extracting the same observation."""

    root_object = _object_at(observation.objects, anchor.row, anchor.col)
    player = observation.best_player
    if root_object is not None:
        root_kind = "occupied_object"
        center = tuple(float(value) for value in root_object.center)
    elif anchor.in_bounds and anchor.row is not None and anchor.col is not None:
        root_kind = "virtual_cell"
        center = (float(anchor.row), float(anchor.col))
    elif player is not None:
        root_kind = "actor"
        center = tuple(float(value) for value in player.position)
    else:
        root_kind = "action_root"
        center = (
            (observation.raw_grid.shape[0] - 1) / 2.0,
            (observation.raw_grid.shape[1] - 1) / 2.0,
        )

    root: dict[str, Any] = {
        "action_name": action_name,
        "action_family": anchor.action_family,
        "requested_direction": anchor.requested_direction,
        "root_kind": root_kind,
        "root_occupied": int(root_object is not None),
        "root_area_bucket": (
            _area_bucket(root_object.area) if root_object is not None else "none"
        ),
        "root_aspect_bucket": (
            _aspect_bucket(root_object) if root_object is not None else "none"
        ),
        "root_affordance": anchor.target_affordance,
        "actor_relation": anchor.actor_relation,
        "actor_relative_direction": anchor.actor_relative_direction,
        "path_status": anchor.path_status,
        "boundary": _boundary_bucket(center, observation.raw_grid.shape),
        "player_available": int(player is not None),
    }

    candidates: list[tuple[float, tuple[Any, ...], dict[str, Any]]] = []
    for obj in observation.objects:
        if root_object is not None and obj.object_id == root_object.object_id:
            continue
        distance = math.dist(center, obj.center)
        gap = _bbox_gap_to_point(obj, center)
        is_actor = bool(player is not None and player.position in set(obj.cells))
        if root_object is None:
            relative_size = "unknown"
        elif obj.area < root_object.area:
            relative_size = "smaller"
        elif obj.area > root_object.area:
            relative_size = "larger"
        else:
            relative_size = "equal"
        descriptor = {
            "direction": _direction(center, obj.center),
            "proximity": _distance_bucket(distance, gap),
            "relative_size": relative_size,
            "area_bucket": _area_bucket(obj.area),
            "aspect_bucket": _aspect_bucket(obj),
            "is_actor": int(is_actor),
            "aligned_row": int(abs(center[0] - obj.center[0]) <= 0.5),
            "aligned_col": int(abs(center[1] - obj.center[1]) <= 0.5),
            "touches_boundary": int(
                obj.bbox[0] == 0
                or obj.bbox[1] == 0
                or obj.bbox[2] == observation.raw_grid.shape[0] - 1
                or obj.bbox[3] == observation.raw_grid.shape[1] - 1
            ),
        }
        tie_break = tuple(sorted(descriptor.items()))
        candidates.append((distance, tie_break, descriptor))
    candidates.sort(key=lambda item: (item[0], item[1]))
    neighbors = tuple(item[2] for item in candidates[: max(1, int(MAXIMUM_NEIGHBORS))])
    graph = ObjectRelativeGraph(root=root, neighbors=neighbors)
    validate_model_graph(graph, game_id=game)
    return graph


def _candidate_specs(
    row: Mapping[str, Any],
    *,
    observation: Any,
    available: Sequence[str],
    actual_action: str,
    actual_data: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = [(actual_action, dict(actual_data))]
    for action_name in available:
        if action_name != "ACTION6":
            specs.append((action_name, {}))
    if "ACTION6" in available:
        centers = {
            (
                round(float(obj.center[1])),
                round(float(obj.center[0])),
            )
            for obj in observation.objects
        }
        if actual_action == "ACTION6":
            x = actual_data.get("x")
            y = actual_data.get("y")
            if x is not None and y is not None:
                centers.add((int(x), int(y)))
        for x, y in sorted(
            centers,
            key=lambda point: hashlib.sha256(
                f"{point[0]}:{point[1]}".encode()
            ).hexdigest(),
        ):
            specs.append(("ACTION6", {"x": x, "y": y}))
    deduplicated = []
    seen = set()
    for action_name, action_data in specs:
        key = json.dumps(
            {"action": action_name, "data": action_data},
            sort_keys=True,
            separators=(",", ":"),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append((action_name, action_data))
    return deduplicated


def _choice_candidates(
    row: Mapping[str, Any],
    *,
    game: str,
    sequence_index: int,
    available: Sequence[str],
    actual_action: str,
    actual_data: Mapping[str, Any],
    game_state: str,
    levels_completed: int,
) -> tuple[tuple[ObjectRelativeGraph, ...], int, int]:
    observation = build_observation(
        row["frame_before"],
        available_actions=available,
        game_state=game_state,
        levels_completed=levels_completed,
        infer_players=True,
    )
    actual_anchor = resolve_action_target(
        observation,
        actual_action,
        actual_data,
    )
    actual_graph = _candidate_graph_from_observation(
        game=game,
        observation=observation,
        anchor=actual_anchor,
        action_name=actual_action,
    )
    actual_digest = _checksum(_graph_payload(actual_graph))
    by_digest: dict[str, ObjectRelativeGraph] = {actual_digest: actual_graph}
    aliases = 0
    for action_name, action_data in _candidate_specs(
        row,
        observation=observation,
        available=available,
        actual_action=actual_action,
        actual_data=actual_data,
    ):
        anchor = resolve_action_target(observation, action_name, action_data)
        graph = _candidate_graph_from_observation(
            game=game,
            observation=observation,
            anchor=anchor,
            action_name=action_name,
        )
        digest = _checksum(_graph_payload(graph))
        aliases += int(digest in by_digest)
        by_digest.setdefault(digest, graph)
    other = sorted(
        (
            (digest, graph)
            for digest, graph in by_digest.items()
            if digest != actual_digest
        ),
        key=lambda item: item[0],
    )[: MAXIMUM_CANDIDATES - 1]
    selected = [(actual_digest, actual_graph), *other]
    selected.sort(key=lambda item: item[0])
    graphs = tuple(graph for _digest, graph in selected)
    selected_index = next(
        index
        for index, (digest, _graph) in enumerate(selected)
        if digest == actual_digest
    )
    return graphs, selected_index, aliases


def _next_milestone(
    rows: Sequence[Mapping[str, Any]],
    *,
    index: int,
) -> tuple[str, int]:
    stop = min(len(rows), index + MILESTONE_HORIZON)
    for future_index in range(index, stop):
        labels = rows[future_index]["_labels"]
        for milestone in MILESTONE_PRIORITY:
            if milestone == "productive":
                active = bool(labels["productive"])
            else:
                active = bool(labels[milestone])
            if active:
                return milestone, future_index - index + 1
    return "none_within_64", MILESTONE_HORIZON


def _suffix_return(
    rows: Sequence[Mapping[str, Any]],
    *,
    index: int,
    discounted_progress: float,
    danger: bool,
) -> float:
    productive = 0.0
    for offset, row in enumerate(rows[index : index + MILESTONE_HORIZON]):
        productive += (0.97**offset) * float(row["_productive_score"])
    value = float(discounted_progress) + 0.1 * productive - 0.5 * float(danger)
    return float(np.clip(value, -1.0, 1.0))


def compile_demonstrations(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    traces_dir: str | Path = DEFAULT_TRACES_DIR,
) -> dict[str, Any]:
    started = time.perf_counter()
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    trace_dir = Path(traces_dir)
    summaries = _episode_summaries(trace_dir)
    grouped = _step_rows(trace_dir)
    records: list[DemonstrationChoiceRecord] = []
    continuity_links = 0
    continuity_matches = 0
    alias_count = 0
    orphan_sequences = 0
    for (game, episode_id), raw_rows in sorted(grouped.items()):
        if summaries.get((game, episode_id)) is None:
            orphan_sequences += 1
        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous_level = 0
        previous_state = "NOT_FINISHED"
        for raw in raw_rows:
            row = dict(raw)
            action = str(row.get("action", "")).upper()
            current_after = int(row.get("levels_completed_after", 0) or 0)
            if action == "RESET":
                if current:
                    segments.append(current)
                    current = []
                previous_level = current_after
                previous_state = str(row.get("game_state_after", "NOT_FINISHED"))
                continue
            row["_levels_before"] = previous_level
            row["_game_state_before"] = previous_state
            current.append(row)
            previous_level = current_after
            previous_state = str(row.get("game_state_after", previous_state))
        if current:
            segments.append(current)
        for segment_index, play_rows in enumerate(segments):
            sequence_id = (
                episode_id
                if len(segments) == 1
                else f"{episode_id}:segment{segment_index}"
            )
            levels_before = [int(row["_levels_before"]) for row in play_rows]
            for left, right in pairwise(play_rows):
                continuity_links += 1
                continuity_matches += int(
                    left.get("frame_after") == right.get("frame_before")
                )
            for index, row in enumerate(play_rows):
                available = _action_names(row.get("available_actions") or ())
                action = str(row["action"]).upper()
                if action not in available:
                    available = tuple(dict.fromkeys((*available, action)))
                trace = build_action_target_trace(
                    game_id=game,
                    source_split="source_train",
                    policy_seed=SEED,
                    reset_index=segment_index,
                    step_index=int(row.get("step", index)),
                    collection_phase="human_demonstration_v4_15",
                    available_action_names=available,
                    selected_action_name=action,
                    selected_action_data=dict(row.get("action_args") or {}),
                    frame_before=row["frame_before"],
                    frame_after=row["frame_after"],
                    game_state_before=str(row["_game_state_before"]),
                    game_state_after=str(row.get("game_state_after", "NOT_FINISHED")),
                    levels_completed_before=levels_before[index],
                    levels_completed_after=int(
                        row.get(
                            "levels_completed_after",
                            levels_before[index],
                        )
                    ),
                )
                labels, _applicable, score, _evidence = compile_semantics(trace)
                row["_labels"] = labels
                row["_productive_score"] = float(score)
            for index, row in enumerate(play_rows):
                progress, danger, _distance, _censored, discounted = _future_targets(
                    play_rows,
                    index=index,
                    level_before=levels_before,
                )
                milestone, milestone_distance = _next_milestone(
                    play_rows,
                    index=index,
                )
                suffix_return = _suffix_return(
                    play_rows,
                    index=index,
                    discounted_progress=discounted,
                    danger=danger,
                )
                action = str(row["action"]).upper()
                actual_data = dict(row.get("action_args") or {})
                available = _action_names(row.get("available_actions") or ())
                if action not in available:
                    available = tuple(dict.fromkeys((*available, action)))
                candidates, selected_index, aliases = _choice_candidates(
                    row,
                    game=game,
                    sequence_index=index,
                    available=available,
                    actual_action=action,
                    actual_data=actual_data,
                    game_state=str(row["_game_state_before"]),
                    levels_completed=levels_before[index],
                )
                alias_count += aliases
                pre_hash = grid_sha256(row["frame_before"])
                post_hash = grid_sha256(row["frame_after"])
                records.append(
                    DemonstrationChoiceRecord(
                        example_id=(
                            "dm15_"
                            + _checksum(
                                {
                                    "game": game,
                                    "episode": sequence_id,
                                    "index": index,
                                    "pre": pre_hash,
                                    "action": action,
                                    "data": actual_data,
                                }
                            )[:20]
                        ),
                        game_id=game,
                        episode_id=sequence_id,
                        sequence_index=index,
                        source_file=str(row["_source_file"]),
                        pre_state_sha256=pre_hash,
                        post_state_sha256=post_hash,
                        candidates=candidates,
                        selected_index=selected_index,
                        observed_effects={
                            effect: bool(row["_labels"][effect])
                            for effect in SEMANTIC_EFFECTS
                        },
                        milestone=milestone,
                        milestone_distance=milestone_distance,
                        suffix_return=suffix_return,
                        success_weight=float(
                            np.clip(
                                1.0 + 4.0 * max(0.0, suffix_return),
                                1.0,
                                5.0,
                            )
                        ),
                        within_16=bool(progress["within_16"]),
                        productive=bool(row["_labels"]["productive"]),
                        actual_action_name=action,
                        actual_action_data=actual_data,
                    )
                )
    corpus_path = destination / "demonstration_choices.jsonl"
    _write_jsonl(corpus_path, (record.to_dict() for record in records))
    by_game = {}
    for game in HUMAN_TRAIN_GAMES:
        selected = [record for record in records if record.game_id == game]
        by_game[game] = {
            "records": len(selected),
            "sequences": len({record.sequence_key for record in selected}),
            "mean_candidates": float(
                np.mean([len(record.candidates) for record in selected])
            ),
            "milestones": dict(Counter(record.milestone for record in selected)),
            "positive_return": sum(record.suffix_return > 0.0 for record in selected),
            "within_16": sum(record.within_16 for record in selected),
        }
    qa: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "records": len(records),
        "sequences": len({record.sequence_key for record in records}),
        "games": list(HUMAN_TRAIN_GAMES),
        "candidate_count": {
            "minimum": min(len(record.candidates) for record in records),
            "maximum": max(len(record.candidates) for record in records),
            "mean": float(np.mean([len(record.candidates) for record in records])),
            "semantic_aliases_removed": alias_count,
        },
        "continuity": {
            "links": continuity_links,
            "matches": continuity_matches,
            "fraction": (
                continuity_matches / continuity_links if continuity_links else 1.0
            ),
        },
        "orphan_sequences": orphan_sequences,
        "milestones": dict(Counter(record.milestone for record in records)),
        "positive_return": sum(record.suffix_return > 0.0 for record in records),
        "student_view_forbidden_fields_absent": True,
        "runtime_seconds": time.perf_counter() - started,
        "by_game": by_game,
        "artifact_sha256": _file_sha256(corpus_path),
    }
    qa["teacher_ready"] = bool(
        records
        and continuity_links == continuity_matches
        and set(by_game) == set(HUMAN_TRAIN_GAMES)
        and qa["candidate_count"]["maximum"] <= MAXIMUM_CANDIDATES
    )
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def load_demonstration_records(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[DemonstrationChoiceRecord, ...]:
    return tuple(
        DemonstrationChoiceRecord.from_dict(row)
        for row in _read_jsonl(Path(output_dir) / "demonstration_choices.jsonl")
    )


@dataclass(frozen=True)
class TensorizedChoices:
    candidate_root_ids: np.ndarray
    candidate_neighbor_ids: np.ndarray
    candidate_neighbor_mask: np.ndarray
    candidate_mask: np.ndarray
    selected_index: np.ndarray
    observed_effects: np.ndarray
    milestone: np.ndarray
    distance: np.ndarray
    suffix_return: np.ndarray
    success_weight: np.ndarray
    sequences: tuple[tuple[int, ...], ...]


def _record_sequences(
    records: Sequence[DemonstrationChoiceRecord],
) -> tuple[tuple[int, ...], ...]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.sequence_key].append(index)
    return tuple(
        tuple(
            sorted(
                indices,
                key=lambda index: records[index].sequence_index,
            )
        )
        for _key, indices in sorted(grouped.items())
    )


def tensorize_choices(
    records: Sequence[DemonstrationChoiceRecord],
    *,
    hash_buckets: int,
    maximum_neighbors: int,
) -> TensorizedChoices:
    candidate_count = max(
        max(len(record.candidates) for record in records),
        1,
    )
    root_rows = [
        [_tokens("root", dict(graph.root)) for graph in record.candidates]
        for record in records
    ]
    neighbor_rows = [
        [
            [
                _tokens("neighbor", dict(neighbor))
                for neighbor in graph.neighbors[:maximum_neighbors]
            ]
            for graph in record.candidates
        ]
        for record in records
    ]
    root_width = max(
        (len(tokens) for candidates in root_rows for tokens in candidates),
        default=1,
    )
    neighbor_width = max(
        (
            len(tokens)
            for candidates in neighbor_rows
            for neighbors in candidates
            for tokens in neighbors
        ),
        default=1,
    )
    count = len(records)
    roots = np.zeros(
        (count, candidate_count, root_width),
        dtype=np.int64,
    )
    neighbors = np.zeros(
        (
            count,
            candidate_count,
            maximum_neighbors,
            neighbor_width,
        ),
        dtype=np.int64,
    )
    neighbor_mask = np.zeros(
        (count, candidate_count, maximum_neighbors),
        dtype=np.float32,
    )
    candidate_mask = np.zeros(
        (count, candidate_count),
        dtype=np.float32,
    )
    for record_index, candidates in enumerate(root_rows):
        for candidate_index, tokens in enumerate(candidates):
            roots[record_index, candidate_index, : len(tokens)] = [
                _token_id(token, hash_buckets) for token in tokens
            ]
            candidate_mask[record_index, candidate_index] = 1.0
    for record_index, candidates in enumerate(neighbor_rows):
        for candidate_index, graph_neighbors in enumerate(candidates):
            for neighbor_index, tokens in enumerate(graph_neighbors):
                neighbors[
                    record_index,
                    candidate_index,
                    neighbor_index,
                    : len(tokens),
                ] = [_token_id(token, hash_buckets) for token in tokens]
                neighbor_mask[
                    record_index,
                    candidate_index,
                    neighbor_index,
                ] = 1.0
    return TensorizedChoices(
        candidate_root_ids=roots,
        candidate_neighbor_ids=neighbors,
        candidate_neighbor_mask=neighbor_mask,
        candidate_mask=candidate_mask,
        selected_index=np.asarray(
            [record.selected_index for record in records],
            dtype=np.int64,
        ),
        observed_effects=np.asarray(
            [
                [float(record.observed_effects[effect]) for effect in SEMANTIC_EFFECTS]
                for record in records
            ],
            dtype=np.float32,
        ),
        milestone=np.asarray(
            [MILESTONE_LABELS.index(record.milestone) for record in records],
            dtype=np.int64,
        ),
        distance=np.asarray(
            [record.milestone_distance / MILESTONE_HORIZON for record in records],
            dtype=np.float32,
        ),
        suffix_return=np.asarray(
            [record.suffix_return for record in records],
            dtype=np.float32,
        ),
        success_weight=np.asarray(
            [record.success_weight for record in records],
            dtype=np.float32,
        ),
        sequences=_record_sequences(records),
    )


def _torch_model(
    *,
    hash_buckets: int,
    embedding_width: int,
    graph_hidden_width: int,
    temporal_hidden_width: int,
    milestone_embedding_width: int,
) -> Any:
    import torch

    class DemonstrationMilestonePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(
                hash_buckets,
                embedding_width,
                padding_idx=0,
            )
            self.node_encoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_width, embedding_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(embedding_width),
            )
            self.graph_trunk = torch.nn.Sequential(
                torch.nn.Linear(
                    embedding_width * 3,
                    graph_hidden_width,
                ),
                torch.nn.GELU(),
                torch.nn.LayerNorm(graph_hidden_width),
                torch.nn.Linear(
                    graph_hidden_width,
                    graph_hidden_width,
                ),
                torch.nn.GELU(),
            )
            self.temporal = torch.nn.GRU(
                graph_hidden_width + len(SEMANTIC_EFFECTS),
                temporal_hidden_width,
                batch_first=True,
            )
            state_width = temporal_hidden_width + graph_hidden_width
            self.milestone_head = torch.nn.Sequential(
                torch.nn.Linear(state_width, 96),
                torch.nn.GELU(),
                torch.nn.Linear(96, len(MILESTONE_LABELS)),
            )
            self.distance_head = torch.nn.Sequential(
                torch.nn.Linear(state_width, 1),
                torch.nn.Sigmoid(),
            )
            self.return_head = torch.nn.Sequential(
                torch.nn.Linear(state_width, 1),
                torch.nn.Tanh(),
            )
            self.milestone_embedding = torch.nn.Embedding(
                len(MILESTONE_LABELS) + 1,
                milestone_embedding_width,
            )
            score_width = (
                temporal_hidden_width
                + graph_hidden_width
                + milestone_embedding_width
                + 1
            )
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(score_width, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 1),
            )

        @staticmethod
        def _mean_tokens(ids: Any, embeddings: Any) -> Any:
            mask = (ids != 0).to(embeddings.dtype)
            total = (embeddings * mask.unsqueeze(-1)).sum(dim=-2)
            denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return total / denominator

        def encode_graph(
            self,
            root_ids: Any,
            neighbor_ids: Any,
            neighbor_mask: Any,
        ) -> Any:
            root = self._mean_tokens(
                root_ids,
                self.embedding(root_ids),
            )
            nodes = self._mean_tokens(
                neighbor_ids,
                self.embedding(neighbor_ids),
            )
            nodes = self.node_encoder(nodes)
            mask = neighbor_mask.unsqueeze(-1)
            mean = (nodes * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
            masked = nodes.masked_fill(mask == 0, -1e4)
            maximum = masked.max(dim=-2).values
            empty = neighbor_mask.sum(dim=-1, keepdim=True) == 0
            maximum = torch.where(
                empty,
                torch.zeros_like(maximum),
                maximum,
            )
            return self.graph_trunk(torch.cat((root, mean, maximum), dim=-1))

        def score(
            self,
            contexts: Any,
            candidate_latent: Any,
            milestone_ids: Any,
            desired_return: Any,
        ) -> Any:
            candidate_count = candidate_latent.shape[-2]
            expanded_context = contexts.unsqueeze(-2).expand(
                *contexts.shape[:-1],
                candidate_count,
                contexts.shape[-1],
            )
            milestone = self.milestone_embedding(milestone_ids).unsqueeze(-2)
            milestone = milestone.expand(
                *milestone.shape[:-2],
                candidate_count,
                milestone.shape[-1],
            )
            conditioned_return = desired_return.unsqueeze(-1).unsqueeze(-1)
            conditioned_return = conditioned_return.expand(
                *conditioned_return.shape[:-2],
                candidate_count,
                1,
            )
            features = torch.cat(
                (
                    expanded_context,
                    candidate_latent,
                    milestone,
                    conditioned_return,
                ),
                dim=-1,
            )
            return self.scorer(features).squeeze(-1)

        def forward(
            self,
            candidate_root_ids: Any,
            candidate_neighbor_ids: Any,
            candidate_neighbor_mask: Any,
            candidate_mask: Any,
            selected_index: Any,
            observed_effects: Any,
        ) -> dict[str, Any]:
            batch, length, candidate_count = candidate_mask.shape
            flat_root = candidate_root_ids.reshape(
                batch * length * candidate_count,
                candidate_root_ids.shape[-1],
            )
            flat_neighbors = candidate_neighbor_ids.reshape(
                batch * length * candidate_count,
                candidate_neighbor_ids.shape[-2],
                candidate_neighbor_ids.shape[-1],
            )
            flat_neighbor_mask = candidate_neighbor_mask.reshape(
                batch * length * candidate_count,
                candidate_neighbor_mask.shape[-1],
            )
            candidate_latent = self.encode_graph(
                flat_root,
                flat_neighbors,
                flat_neighbor_mask,
            ).reshape(
                batch,
                length,
                candidate_count,
                -1,
            )
            selected = torch.gather(
                candidate_latent,
                2,
                selected_index[..., None, None].expand(
                    batch,
                    length,
                    1,
                    candidate_latent.shape[-1],
                ),
            ).squeeze(2)
            temporal_input = torch.cat(
                (selected, observed_effects),
                dim=-1,
            )
            temporal, hidden = self.temporal(temporal_input)
            contexts = torch.cat(
                (
                    torch.zeros_like(temporal[:, :1, :]),
                    temporal[:, :-1, :],
                ),
                dim=1,
            )
            mask = candidate_mask.unsqueeze(-1)
            candidate_mean = (candidate_latent * mask).sum(dim=2) / mask.sum(
                dim=2
            ).clamp_min(1.0)
            state = torch.cat((contexts, candidate_mean), dim=-1)
            return {
                "contexts": contexts,
                "candidate_latent": candidate_latent,
                "milestone_logits": self.milestone_head(state),
                "distance": self.distance_head(state).squeeze(-1),
                "suffix_return": self.return_head(state).squeeze(-1),
                "hidden": hidden,
            }

    return DemonstrationMilestonePolicy()


def _sequence_windows(
    tensors: TensorizedChoices,
    allowed: set[int],
    *,
    maximum_history: int,
) -> list[tuple[int, ...]]:
    windows = []
    for sequence in tensors.sequences:
        selected = [index for index in sequence if index in allowed]
        for offset in range(0, len(selected), maximum_history):
            window = tuple(selected[offset : offset + maximum_history])
            if window:
                windows.append(window)
    return windows


def _choice_batch(
    tensors: TensorizedChoices,
    windows: Sequence[Sequence[int]],
    *,
    device: str,
) -> tuple[dict[str, Any], Any]:
    import torch

    batch = len(windows)
    length = max(len(window) for window in windows)
    indices = np.zeros((batch, length), dtype=np.int64)
    row_mask = np.zeros((batch, length), dtype=np.float32)
    for row, window in enumerate(windows):
        indices[row, : len(window)] = np.asarray(window, dtype=np.int64)
        row_mask[row, : len(window)] = 1.0

    def take(array: np.ndarray, dtype: Any) -> Any:
        return torch.as_tensor(
            array[indices],
            dtype=dtype,
            device=device,
        )

    return (
        {
            "candidate_root_ids": take(
                tensors.candidate_root_ids,
                torch.long,
            ),
            "candidate_neighbor_ids": take(
                tensors.candidate_neighbor_ids,
                torch.long,
            ),
            "candidate_neighbor_mask": take(
                tensors.candidate_neighbor_mask,
                torch.float32,
            ),
            "candidate_mask": take(
                tensors.candidate_mask,
                torch.float32,
            ),
            "selected_index": take(
                tensors.selected_index,
                torch.long,
            ),
            "observed_effects": take(
                tensors.observed_effects,
                torch.float32,
            ),
            "milestone": take(
                tensors.milestone,
                torch.long,
            ),
            "distance": take(
                tensors.distance,
                torch.float32,
            ),
            "suffix_return": take(
                tensors.suffix_return,
                torch.float32,
            ),
            "success_weight": take(
                tensors.success_weight,
                torch.float32,
            ),
        },
        torch.as_tensor(
            row_mask,
            dtype=torch.float32,
            device=device,
        ),
    )


def _masked_mean(values: Any, mask: Any) -> Any:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _policy_losses(
    model: Any,
    batch: Mapping[str, Any],
    row_mask: Any,
    *,
    loss_weights: Mapping[str, float],
) -> tuple[Any, dict[str, float]]:
    import torch

    output = model(
        batch["candidate_root_ids"],
        batch["candidate_neighbor_ids"],
        batch["candidate_neighbor_mask"],
        batch["candidate_mask"],
        batch["selected_index"],
        batch["observed_effects"],
    )
    invalid = batch["candidate_mask"] == 0
    none_condition = torch.full_like(
        batch["milestone"],
        len(MILESTONE_LABELS),
    )
    behavior_scores = model.score(
        output["contexts"],
        output["candidate_latent"],
        none_condition,
        torch.zeros_like(batch["suffix_return"]),
    ).masked_fill(invalid, -1e4)
    conditioned_scores = model.score(
        output["contexts"],
        output["candidate_latent"],
        batch["milestone"],
        batch["suffix_return"],
    ).masked_fill(invalid, -1e4)
    behavior_loss = torch.nn.functional.cross_entropy(
        behavior_scores.flatten(0, 1),
        batch["selected_index"].flatten(),
        reduction="none",
    ).reshape_as(row_mask)
    conditioned_loss = torch.nn.functional.cross_entropy(
        conditioned_scores.flatten(0, 1),
        batch["selected_index"].flatten(),
        reduction="none",
    ).reshape_as(row_mask)
    milestone_loss = torch.nn.functional.cross_entropy(
        output["milestone_logits"].flatten(0, 1),
        batch["milestone"].flatten(),
        reduction="none",
    ).reshape_as(row_mask)
    distance_loss = (output["distance"] - batch["distance"]).square()
    return_loss = (output["suffix_return"] - batch["suffix_return"]).square()
    weighted_conditioned = conditioned_loss * batch["success_weight"]
    pieces = {
        "behavior_cloning": _masked_mean(behavior_loss, row_mask),
        "success_conditioned_cloning": _masked_mean(
            weighted_conditioned,
            row_mask,
        ),
        "milestone": _masked_mean(milestone_loss, row_mask),
        "distance": _masked_mean(distance_loss, row_mask),
        "return": _masked_mean(return_loss, row_mask),
    }
    total = sum(float(loss_weights[key]) * value for key, value in pieces.items())
    return total, {key: float(value.detach().cpu()) for key, value in pieces.items()}


def _fit_policy(
    records: Sequence[DemonstrationChoiceRecord],
    tensors: TensorizedChoices,
    *,
    train_indices: Sequence[int],
    parameters: Mapping[str, Any],
    device: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _torch_model(
        hash_buckets=int(parameters["hash_buckets"]),
        embedding_width=int(parameters["embedding_width"]),
        graph_hidden_width=int(parameters["graph_hidden_width"]),
        temporal_hidden_width=int(parameters["temporal_hidden_width"]),
        milestone_embedding_width=int(parameters["milestone_embedding_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    windows = _sequence_windows(
        tensors,
        {int(index) for index in train_indices},
        maximum_history=int(parameters["maximum_history"]),
    )
    rng = random.Random(seed)
    losses = []
    started = time.perf_counter()
    model.train()
    for _epoch in range(int(parameters["epochs"])):
        rng.shuffle(windows)
        epoch_pieces: dict[str, list[float]] = defaultdict(list)
        for offset in range(0, len(windows), 8):
            selected_windows = windows[offset : offset + 8]
            batch, row_mask = _choice_batch(
                tensors,
                selected_windows,
                device=device,
            )
            optimizer.zero_grad()
            total, pieces = _policy_losses(
                model,
                batch,
                row_mask,
                loss_weights=parameters["loss_weights"],
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(total.detach().cpu()))
            for key, value in pieces.items():
                epoch_pieces[key].append(value)
    model.eval()
    return model, {
        "device": device,
        "epochs": int(parameters["epochs"]),
        "windows": len(windows),
        "train_rows": len(train_indices),
        "train_games": sorted({records[int(index)].game_id for index in train_indices}),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "runtime_seconds": time.perf_counter() - started,
    }


def _benchmark_devices(
    tensors: TensorizedChoices,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    allowed = set(range(min(len(tensors.selected_index), 128)))
    windows = _sequence_windows(
        tensors,
        allowed,
        maximum_history=32,
    )[:4]

    def measure(device: str) -> float:
        model = _torch_model(
            hash_buckets=int(parameters["hash_buckets"]),
            embedding_width=int(parameters["embedding_width"]),
            graph_hidden_width=int(parameters["graph_hidden_width"]),
            temporal_hidden_width=int(parameters["temporal_hidden_width"]),
            milestone_embedding_width=int(parameters["milestone_embedding_width"]),
        ).to(device)
        batch, row_mask = _choice_batch(
            tensors,
            windows,
            device=device,
        )
        for _ in range(2):
            total, _pieces = _policy_losses(
                model,
                batch,
                row_mask,
                loss_weights=parameters["loss_weights"],
            )
            total.backward()
            model.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            total, _pieces = _policy_losses(
                model,
                batch,
                row_mask,
                loss_weights=parameters["loss_weights"],
            )
            total.backward()
            model.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter() - started

    cpu = measure("cpu")
    result: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "timings_seconds": {"cpu": cpu},
        "selected_device": "cpu",
        "cuda_speedup": 0.0,
    }
    if torch.cuda.is_available():
        cuda = measure("cuda:0")
        speedup = cpu / max(cuda, 1e-9)
        result.update(
            {
                "cuda_name": torch.cuda.get_device_name(0),
                "timings_seconds": {
                    "cpu": cpu,
                    "cuda:0": cuda,
                },
                "cuda_speedup": speedup,
                "selected_device": ("cuda:0" if speedup >= 1.2 else "cpu"),
            }
        )
    return result


def _load_policy_checkpoint(
    path: Path,
    *,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported V4.15 checkpoint")
    parameters = payload["parameters"]
    model = _torch_model(
        hash_buckets=int(parameters["hash_buckets"]),
        embedding_width=int(parameters["embedding_width"]),
        graph_hidden_width=int(parameters["graph_hidden_width"]),
        temporal_hidden_width=int(parameters["temporal_hidden_width"]),
        milestone_embedding_width=int(parameters["milestone_embedding_width"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _save_policy_checkpoint(
    path: Path,
    model: Any,
    *,
    parameters: Mapping[str, Any],
    manifest_checksum: str,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "parameters": dict(parameters),
            "games": list(HUMAN_TRAIN_GAMES),
            "manifest_checksum": manifest_checksum,
            "state_dict": model.state_dict(),
        },
        path,
    )


def _shuffled_records(
    records: Sequence[DemonstrationChoiceRecord],
) -> tuple[DemonstrationChoiceRecord, ...]:
    return tuple(
        DemonstrationChoiceRecord(
            **{
                **record.__dict__,
                "candidates": tuple(
                    graph.relation_shuffled() for graph in record.candidates
                ),
            }
        )
        for record in records
    )


def _masked_scores(scores: Any, candidate_mask: Any) -> Any:
    return scores.masked_fill(candidate_mask == 0, -1e4)


def _predict_policy(
    model: Any,
    tensors: TensorizedChoices,
    relation_tensors: TensorizedChoices,
    *,
    indices: Sequence[int],
    parameters: Mapping[str, Any],
    device: str,
    seed: int,
) -> dict[int, dict[str, Any]]:
    import torch

    windows = _sequence_windows(
        tensors,
        {int(index) for index in indices},
        maximum_history=int(parameters["maximum_history"]),
    )
    output_rows: dict[int, dict[str, Any]] = {}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(windows), 8):
            selected_windows = windows[offset : offset + 8]
            batch, _row_mask = _choice_batch(
                tensors,
                selected_windows,
                device=device,
            )
            relation_batch, _ = _choice_batch(
                relation_tensors,
                selected_windows,
                device=device,
            )
            output = model(
                batch["candidate_root_ids"],
                batch["candidate_neighbor_ids"],
                batch["candidate_neighbor_mask"],
                batch["candidate_mask"],
                batch["selected_index"],
                batch["observed_effects"],
            )
            relation_output = model(
                relation_batch["candidate_root_ids"],
                relation_batch["candidate_neighbor_ids"],
                relation_batch["candidate_neighbor_mask"],
                relation_batch["candidate_mask"],
                relation_batch["selected_index"],
                relation_batch["observed_effects"],
            )
            learned_milestone = output["milestone_logits"].argmax(dim=-1)
            relation_milestone = relation_output["milestone_logits"].argmax(dim=-1)
            none_condition = torch.full_like(
                learned_milestone,
                len(MILESTONE_LABELS),
            )
            one = torch.ones_like(batch["suffix_return"])
            behavior = _masked_scores(
                model.score(
                    output["contexts"],
                    output["candidate_latent"],
                    none_condition,
                    torch.zeros_like(one),
                ),
                batch["candidate_mask"],
            )
            learned = _masked_scores(
                model.score(
                    output["contexts"],
                    output["candidate_latent"],
                    learned_milestone,
                    one,
                ),
                batch["candidate_mask"],
            )
            oracle = _masked_scores(
                model.score(
                    output["contexts"],
                    output["candidate_latent"],
                    batch["milestone"],
                    one,
                ),
                batch["candidate_mask"],
            )
            relation = _masked_scores(
                model.score(
                    relation_output["contexts"],
                    relation_output["candidate_latent"],
                    relation_milestone,
                    one,
                ),
                relation_batch["candidate_mask"],
            )
            shuffled_context = torch.roll(
                output["contexts"],
                shifts=1,
                dims=1,
            )
            history_shuffle = _masked_scores(
                model.score(
                    shuffled_context,
                    output["candidate_latent"],
                    learned_milestone,
                    one,
                ),
                batch["candidate_mask"],
            )
            milestone_probabilities = torch.softmax(
                output["milestone_logits"],
                dim=-1,
            )
            modes = {
                "behavior": behavior,
                "learned_milestone": learned,
                "oracle_milestone": oracle,
                "relation_shuffle": relation,
                "history_shuffle": history_shuffle,
            }
            for batch_index, window in enumerate(selected_windows):
                for local_index, record_index in enumerate(window):
                    candidate_count = int(
                        batch["candidate_mask"][
                            batch_index,
                            local_index,
                        ].sum()
                    )
                    row_modes = {}
                    for name, scores in modes.items():
                        values = scores[
                            batch_index,
                            local_index,
                            :candidate_count,
                        ]
                        probabilities = torch.softmax(
                            values,
                            dim=-1,
                        )
                        row_modes[name] = {
                            "selected_index": int(probabilities.argmax().cpu()),
                            "probabilities": [
                                float(value) for value in probabilities.cpu().tolist()
                            ],
                        }
                    output_rows[int(record_index)] = {
                        "modes": row_modes,
                        "milestone_probabilities": [
                            float(value)
                            for value in milestone_probabilities[
                                batch_index,
                                local_index,
                            ]
                            .cpu()
                            .tolist()
                        ],
                        "predicted_milestone": int(
                            learned_milestone[
                                batch_index,
                                local_index,
                            ].cpu()
                        ),
                        "predicted_distance": float(
                            output["distance"][
                                batch_index,
                                local_index,
                            ].cpu()
                        ),
                        "predicted_return": float(
                            output["suffix_return"][
                                batch_index,
                                local_index,
                            ].cpu()
                        ),
                    }
    if set(output_rows) != {int(index) for index in indices}:
        raise RuntimeError("V4.15 prediction coverage mismatch")
    return output_rows


def _action_only_selection(
    record: DemonstrationChoiceRecord,
    action_frequency: Mapping[str, float],
) -> tuple[int, list[float]]:
    scores = np.asarray(
        [
            float(
                action_frequency.get(
                    str(graph.root.get("action_name", "unknown")),
                    0.0,
                )
            )
            for graph in record.candidates
        ],
        dtype=np.float64,
    )
    probabilities = np.exp(scores - np.max(scores))
    probabilities /= probabilities.sum()
    return int(np.argmax(probabilities)), probabilities.tolist()


def _template_selection(
    record: DemonstrationChoiceRecord,
) -> tuple[int, list[float]]:
    scores = np.asarray(
        [
            float(graph.root.get("root_occupied", 0))
            + 0.5 * float(graph.root.get("path_status") == "open")
            + 0.25 * float(graph.root.get("action_family") == "move")
            + 0.1 * float(graph.root.get("action_family") == "click")
            for graph in record.candidates
        ],
        dtype=np.float64,
    )
    probabilities = np.exp(scores - np.max(scores))
    probabilities /= probabilities.sum()
    return int(np.argmax(probabilities)), probabilities.tolist()


def _paired_bootstrap_values(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return {"mean_gain": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            float(np.mean(values[rng.integers(0, len(values), len(values))]))
            for _ in range(samples)
        ]
    )
    return {
        "mean_gain": float(np.mean(values)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _choice_metrics(
    records: Sequence[DemonstrationChoiceRecord],
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    correct = np.asarray(
        [
            int(row["predictions"][mode]["selected_index"]) == record.selected_index
            for record, row in zip(records, rows)
        ],
        dtype=np.float64,
    )
    nll = []
    for record, row in zip(records, rows):
        probability = float(
            row["predictions"][mode]["probabilities"][record.selected_index]
        )
        nll.append(-math.log(max(probability, 1e-9)))
    productive = np.asarray(
        [record.productive for record in records],
        dtype=bool,
    )
    within = np.asarray(
        [record.within_16 for record in records],
        dtype=bool,
    )
    by_game = {}
    for game in HUMAN_TRAIN_GAMES:
        selected = np.asarray(
            [record.game_id == game for record in records],
            dtype=bool,
        )
        by_game[game] = {
            "rows": int(selected.sum()),
            "top1_accuracy": float(correct[selected].mean()) if selected.any() else 0.0,
            "mean_nll": float(np.mean(np.asarray(nll)[selected]))
            if selected.any()
            else 0.0,
        }
    return {
        "rows": len(records),
        "top1_accuracy": float(correct.mean()),
        "mean_nll": float(np.mean(nll)),
        "productive_top1_accuracy": float(correct[productive].mean())
        if productive.any()
        else 0.0,
        "within_16_top1_accuracy": float(correct[within].mean())
        if within.any()
        else 0.0,
        "per_game": by_game,
    }


def _three_action_exact(
    records: Sequence[DemonstrationChoiceRecord],
    rows_by_example: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
) -> float:
    windows = []
    by_sequence: dict[str, list[DemonstrationChoiceRecord]] = defaultdict(list)
    for record in records:
        by_sequence[record.sequence_key].append(record)
    for sequence in by_sequence.values():
        sequence.sort(key=lambda record: record.sequence_index)
        for offset in range(max(0, len(sequence) - 2)):
            window = sequence[offset : offset + 3]
            windows.append(
                all(
                    int(
                        rows_by_example[record.example_id]["predictions"][mode][
                            "selected_index"
                        ]
                    )
                    == record.selected_index
                    for record in window
                )
            )
    return float(np.mean(windows)) if windows else 0.0


def _milestone_metrics(
    records: Sequence[DemonstrationChoiceRecord],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    truth = np.asarray(
        [MILESTONE_LABELS.index(record.milestone) for record in records],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [row["milestone_probabilities"] for row in rows],
        dtype=np.float64,
    )
    predicted = probabilities.argmax(axis=1)
    recalls = []
    per_label = {}
    for index, label in enumerate(MILESTONE_LABELS):
        selected = truth == index
        recall = (
            float(np.mean(predicted[selected] == index)) if selected.any() else None
        )
        if recall is not None:
            recalls.append(recall)
        per_label[label] = {
            "rows": int(selected.sum()),
            "recall": recall,
            "mean_probability": float(probabilities[:, index].mean()),
        }
    majority_index = Counter(truth).most_common(1)[0][0]
    majority_predicted = np.full_like(truth, majority_index)
    majority_recalls = [
        float(np.mean(majority_predicted[truth == index] == index))
        for index in sorted(set(truth.tolist()))
    ]
    one_hot = np.eye(len(MILESTONE_LABELS))[truth]
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "macro_balanced_accuracy": float(np.mean(recalls)),
        "macro_brier": float(np.mean((probabilities - one_hot) ** 2)),
        "majority_accuracy": float(np.mean(majority_predicted == truth)),
        "majority_macro_balanced_accuracy": float(np.mean(majority_recalls)),
        "per_label": per_label,
    }


def train_demonstration_policy(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa["teacher_ready"]:
        raise RuntimeError("V4.15 teacher QA failed")
    records = load_demonstration_records(destination)
    parameters = dict(manifest["model"])
    tensors = tensorize_choices(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=int(parameters["maximum_neighbors"]),
    )
    shuffled = _shuffled_records(records)
    relation_tensors = tensorize_choices(
        shuffled,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=int(parameters["maximum_neighbors"]),
    )
    benchmark = _benchmark_devices(tensors, parameters)
    device = (
        str(benchmark["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    prediction_by_index: dict[int, dict[str, Any]] = {}
    fold_rows = []
    for fold_index, held_game in enumerate(HUMAN_TRAIN_GAMES):
        train_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id != held_game
            ],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id == held_game
            ],
            dtype=np.int64,
        )
        model, summary = _fit_policy(
            records,
            tensors,
            train_indices=train_indices,
            parameters=parameters,
            device=device,
            seed=SEED + fold_index,
        )
        predicted = _predict_policy(
            model,
            tensors,
            relation_tensors,
            indices=test_indices,
            parameters=parameters,
            device=device,
            seed=SEED + 100 + fold_index,
        )
        action_counts = Counter(
            records[int(index)].actual_action_name for index in train_indices
        )
        action_total = sum(action_counts.values())
        action_frequency = {
            action: count / action_total for action, count in action_counts.items()
        }
        for index in test_indices:
            record = records[int(index)]
            action_index, action_probabilities = _action_only_selection(
                record, action_frequency
            )
            template_index, template_probabilities = _template_selection(record)
            row = predicted[int(index)]
            row["modes"]["action_only"] = {
                "selected_index": action_index,
                "probabilities": action_probabilities,
            }
            row["modes"]["template"] = {
                "selected_index": template_index,
                "probabilities": template_probabilities,
            }
            prediction_by_index[int(index)] = row
        fold_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "held_game": held_game,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "training": summary,
            }
        )
    prediction_rows = []
    for index, record in enumerate(records):
        row = prediction_by_index[index]
        prediction_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "example_id": record.example_id,
                "game_id": record.game_id,
                "selected_index": record.selected_index,
                "candidate_count": len(record.candidates),
                "predictions": row["modes"],
                "milestone": record.milestone,
                "milestone_probabilities": row["milestone_probabilities"],
                "predicted_distance": row["predicted_distance"],
                "predicted_return": row["predicted_return"],
            }
        )
    predictions_path = destination / "logo_predictions.jsonl"
    folds_path = destination / "folds.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(folds_path, fold_rows)
    modes = (
        "action_only",
        "template",
        "behavior",
        "learned_milestone",
        "oracle_milestone",
        "relation_shuffle",
        "history_shuffle",
    )
    metrics = {
        mode: _choice_metrics(records, prediction_rows, mode=mode) for mode in modes
    }
    by_example = {row["example_id"]: row for row in prediction_rows}
    three_action = {
        mode: _three_action_exact(
            records,
            by_example,
            mode=mode,
        )
        for mode in modes
    }
    milestone = _milestone_metrics(records, prediction_rows)
    samples = int(manifest["evaluation"]["bootstrap_samples"])
    learned_correct = np.asarray(
        [
            row["predictions"]["learned_milestone"]["selected_index"]
            == record.selected_index
            for record, row in zip(records, prediction_rows)
        ],
        dtype=np.float64,
    )
    action_correct = np.asarray(
        [
            row["predictions"]["action_only"]["selected_index"] == record.selected_index
            for record, row in zip(records, prediction_rows)
        ],
        dtype=np.float64,
    )
    relation_correct = np.asarray(
        [
            row["predictions"]["relation_shuffle"]["selected_index"]
            == record.selected_index
            for record, row in zip(records, prediction_rows)
        ],
        dtype=np.float64,
    )
    learned_over_action = _paired_bootstrap_values(
        learned_correct - action_correct,
        samples=samples,
        seed=SEED + 400,
    )
    relation_degradation = _paired_bootstrap_values(
        learned_correct - relation_correct,
        samples=samples,
        seed=SEED + 401,
    )
    nonnegative_games = sum(
        metrics["learned_milestone"]["per_game"][game]["top1_accuracy"]
        >= metrics["action_only"]["per_game"][game]["top1_accuracy"]
        for game in HUMAN_TRAIN_GAMES
    )
    identity_features = np.asarray(
        [
            [
                *row["milestone_probabilities"],
                float(row["predicted_distance"]),
                float(row["predicted_return"]),
            ]
            for row in prediction_rows
        ],
        dtype=np.float64,
    )
    identity = _identity_probe(records, identity_features)
    checks = {
        "learned_over_action_ci_lower_positive": (learned_over_action["ci_low"] > 0.0),
        "nonnegative_human_games": (
            nonnegative_games
            >= int(manifest["evaluation"]["behavior_nonnegative_games_minimum"])
        ),
        "relation_shuffle_degradation_ci_lower_positive": (
            relation_degradation["ci_low"] > 0.0
        ),
        "milestone_balanced_gain": (
            milestone["macro_balanced_accuracy"]
            - milestone["majority_macro_balanced_accuracy"]
            >= float(manifest["evaluation"]["milestone_balanced_gain_minimum"])
        ),
        "identity_gain_within_limit": (
            identity["gain_over_majority"]
            <= float(manifest["evaluation"]["identity_gain_maximum"])
        ),
    }
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "verdict": (
            "BEHAVIOR_PRIOR_SUPPORTED"
            if all(checks.values())
            else "BEHAVIOR_PRIOR_NOT_SUPPORTED"
        ),
        "behavior_prior_supported": all(checks.values()),
        "selected_device": device,
        "device_benchmark": benchmark,
        "records": len(records),
        "metrics": metrics,
        "three_action_exact": three_action,
        "milestone": milestone,
        "comparisons": {
            "learned_over_action": learned_over_action,
            "relation_shuffle_degradation": relation_degradation,
        },
        "nonnegative_human_games": nonnegative_games,
        "game_identity_probe": identity,
        "checks": checks,
        "folds": fold_rows,
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "demonstration_choices": _file_sha256(
                destination / "demonstration_choices.jsonl"
            ),
            "logo_predictions": _file_sha256(predictions_path),
            "folds": _file_sha256(folds_path),
        },
    }
    result["semantic_result_checksum"] = _checksum(result)
    _write_json(destination / "semantic_result.json", result)
    all_indices = np.arange(len(records), dtype=np.int64)
    final_model, final_summary = _fit_policy(
        records,
        tensors,
        train_indices=all_indices,
        parameters=parameters,
        device=device,
        seed=SEED + 900,
    )
    checkpoint_path = destination / "demonstration_policy.pt"
    _save_policy_checkpoint(
        checkpoint_path,
        final_model,
        parameters=parameters,
        manifest_checksum=manifest["manifest_checksum"],
    )
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "path": checkpoint_path.as_posix(),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _file_sha256(checkpoint_path),
        "device": device,
        "training": final_summary,
    }
    checkpoint["checkpoint_checksum"] = _checksum(checkpoint)
    _write_json(destination / "checkpoint_metadata.json", checkpoint)
    result["final_checkpoint"] = checkpoint
    result["semantic_result_checksum"] = _checksum(
        {
            key: value
            for key, value in result.items()
            if key != "semantic_result_checksum"
        }
    )
    _write_json(destination / "semantic_result.json", result)
    return result


@dataclass(frozen=True)
class PolicyBelief:
    hidden: tuple[float, ...] = ()
    step_count: int = 0


def _score_candidate_graphs(
    model: Any,
    graphs: Sequence[ObjectRelativeGraph],
    *,
    parameters: Mapping[str, Any],
    device: str,
    belief: PolicyBelief | None = None,
    relation_shuffle: bool = False,
    oracle_milestone: str | None = None,
) -> dict[str, Any]:
    import torch

    selected_graphs = [
        graph.relation_shuffled() if relation_shuffle else graph for graph in graphs
    ]
    roots, neighbors, masks = _tensorize_graph_batch(
        selected_graphs,
        parameters=parameters,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        candidate_latent = model.encode_graph(
            roots,
            neighbors,
            masks,
        ).unsqueeze(0)
        if belief is not None and belief.hidden:
            context = torch.as_tensor(
                belief.hidden,
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
        else:
            context = torch.zeros(
                (1, int(parameters["temporal_hidden_width"])),
                dtype=torch.float32,
                device=device,
            )
        state = torch.cat(
            (context, candidate_latent.mean(dim=1)),
            dim=-1,
        )
        milestone_logits = model.milestone_head(state)
        milestone_probabilities = torch.softmax(
            milestone_logits,
            dim=-1,
        )
        learned_milestone = milestone_logits.argmax(dim=-1)
        none_condition = torch.full_like(
            learned_milestone,
            len(MILESTONE_LABELS),
        )
        zero = torch.zeros((1,), dtype=torch.float32, device=device)
        one = torch.ones((1,), dtype=torch.float32, device=device)
        behavior = model.score(
            context,
            candidate_latent,
            none_condition,
            zero,
        )[0]
        learned = model.score(
            context,
            candidate_latent,
            learned_milestone,
            one,
        )[0]
        oracle = None
        if oracle_milestone is not None:
            oracle_id = torch.as_tensor(
                [MILESTONE_LABELS.index(oracle_milestone)],
                dtype=torch.long,
                device=device,
            )
            oracle = model.score(
                context,
                candidate_latent,
                oracle_id,
                one,
            )[0]
        return {
            "behavior_scores": [float(value) for value in behavior.cpu().tolist()],
            "learned_scores": [float(value) for value in learned.cpu().tolist()],
            "oracle_scores": (
                [float(value) for value in oracle.cpu().tolist()]
                if oracle is not None
                else None
            ),
            "milestone_probabilities": [
                float(value) for value in milestone_probabilities[0].cpu().tolist()
            ],
            "predicted_milestone": MILESTONE_LABELS[int(learned_milestone[0].cpu())],
            "predicted_distance": float(model.distance_head(state)[0, 0].cpu()),
            "predicted_return": float(model.return_head(state)[0, 0].cpu()),
        }


def _advance_policy_belief(
    model: Any,
    graph: ObjectRelativeGraph,
    observed_effects: Mapping[str, bool],
    *,
    parameters: Mapping[str, Any],
    device: str,
    belief: PolicyBelief,
) -> PolicyBelief:
    import torch

    roots, neighbors, masks = _tensorize_graph_batch(
        (graph,),
        parameters=parameters,
        device=device,
    )
    hidden = (
        torch.as_tensor(
            belief.hidden,
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, -1)
        if belief.hidden
        else None
    )
    feedback = torch.as_tensor(
        [[float(observed_effects.get(effect, False)) for effect in SEMANTIC_EFFECTS]],
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        latent = model.encode_graph(roots, neighbors, masks)
        _temporal, next_hidden = model.temporal(
            torch.cat((latent, feedback), dim=-1).unsqueeze(1),
            hidden,
        )
    return PolicyBelief(
        hidden=tuple(float(value) for value in next_hidden[:, 0, :].cpu().numpy()[0]),
        step_count=belief.step_count + 1,
    )


def _zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    standard = float(array.std())
    if standard <= 1e-9:
        return np.zeros_like(array)
    return (array - array.mean()) / standard


def _arm_milestone(arm: Any) -> str:
    for milestone in MILESTONE_PRIORITY:
        if milestone == "productive":
            active = bool(arm.labels["productive"])
        else:
            active = bool(arm.labels[milestone])
        if active:
            return milestone
    return "none_within_64"


def evaluate_transfer_reranking(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    semantic = _read_json(destination / "semantic_result.json")
    device = (
        str(semantic["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    model, checkpoint = _load_policy_checkpoint(
        destination / "demonstration_policy.pt",
        device=device,
    )
    if checkpoint["manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("V4.15 checkpoint/manifest mismatch")
    parameters = checkpoint["parameters"]
    panels = tuple(
        panel
        for panel in load_teacher_panels(v411_dir)
        if panel.game_id in TRANSFER_GAMES
    )
    if {panel.game_id for panel in panels} != set(TRANSFER_GAMES):
        raise ValueError("V4.15 transfer coverage is incomplete")
    v414_path = Path(v414_dir)
    v414_panels = {
        str(row["panel_id"]): row
        for row in _read_jsonl(v414_path / "transfer_predictions.jsonl")
    }
    old_decisions = _read_jsonl(v414_path / "transfer_decisions.jsonl")
    old_selection = {
        (str(row["panel_id"]), str(row["method"])): int(row["selected_arm"])
        for row in old_decisions
    }
    decisions = []
    prediction_rows = []
    baseline_methods = {
        "action_only": "action_only",
        "action_sequence_only": "action_sequence_only",
        "deterministic_template": "deterministic_template",
        "true_world_learned_ebm": "true_world_learned_ebm",
        "oracle_energy": "oracle_energy",
    }
    for panel in panels:
        graphs = tuple(arm.graph for arm in panel.arms)
        oracle_arm = max(
            panel.arms,
            key=lambda arm: (
                float(arm.horizon_return),
                -int(arm.arm_index),
            ),
        )
        oracle_milestone = _arm_milestone(oracle_arm)
        policy = _score_candidate_graphs(
            model,
            graphs,
            parameters=parameters,
            device=device,
            oracle_milestone=oracle_milestone,
        )
        relation = _score_candidate_graphs(
            model,
            graphs,
            parameters=parameters,
            device=device,
            relation_shuffle=True,
        )
        v414_row = v414_panels[panel.panel_id]
        v414_by_arm = {int(row["arm_index"]): row for row in v414_row["arms"]}
        temporal_energies = [
            float(v414_by_arm[int(arm.arm_index)]["temporal_energy"])
            for arm in panel.arms
        ]
        learned_combined = _zscore(policy["learned_scores"]) - float(
            manifest["evaluation"]["ebm_coefficient"]
        ) * _zscore(temporal_energies)
        oracle_combined = _zscore(policy["oracle_scores"]) - float(
            manifest["evaluation"]["ebm_coefficient"]
        ) * _zscore(temporal_energies)
        selected_indices = {
            "behavior_policy": int(np.argmax(policy["behavior_scores"])),
            "learned_milestone_policy": int(np.argmax(policy["learned_scores"])),
            "relation_shuffle_policy": int(np.argmax(relation["learned_scores"])),
            "policy_temporal_ebm": int(np.argmax(learned_combined)),
            "oracle_milestone_temporal_ebm": int(np.argmax(oracle_combined)),
        }
        arm_by_index = {int(arm.arm_index): arm for arm in panel.arms}
        for output_name, old_name in baseline_methods.items():
            selected_indices[output_name] = old_selection[(panel.panel_id, old_name)]
        for method, arm_index in selected_indices.items():
            selected = arm_by_index[int(arm_index)]
            decisions.append(
                {
                    "format_version": FORMAT_VERSION,
                    "panel_id": panel.panel_id,
                    "game_id": panel.game_id,
                    "method": method,
                    "selected_arm": int(selected.arm_index),
                    "oracle_arm": int(oracle_arm.arm_index),
                    "utility": float(selected.horizon_return),
                    "oracle_utility": float(oracle_arm.horizon_return),
                    "regret": float(oracle_arm.horizon_return)
                    - float(selected.horizon_return),
                    "oracle_action": (
                        int(selected.arm_index) == int(oracle_arm.arm_index)
                    ),
                    "completion_selected": bool(selected.labels["level_complete"]),
                    "completion_available": any(
                        bool(arm.labels["level_complete"]) for arm in panel.arms
                    ),
                }
            )
        prediction_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "panel_id": panel.panel_id,
                "game_id": panel.game_id,
                "oracle_milestone": oracle_milestone,
                "predicted_milestone": policy["predicted_milestone"],
                "milestone_probabilities": policy["milestone_probabilities"],
                "arms": [
                    {
                        "arm_index": int(arm.arm_index),
                        "action_name": arm.action_name,
                        "utility": float(arm.horizon_return),
                        "completion": bool(arm.labels["level_complete"]),
                        "behavior_score": float(policy["behavior_scores"][index]),
                        "learned_score": float(policy["learned_scores"][index]),
                        "oracle_milestone_score": float(policy["oracle_scores"][index]),
                        "relation_shuffle_score": float(
                            relation["learned_scores"][index]
                        ),
                        "temporal_energy": temporal_energies[index],
                        "learned_combined_score": float(learned_combined[index]),
                        "oracle_combined_score": float(oracle_combined[index]),
                    }
                    for index, arm in enumerate(panel.arms)
                ],
            }
        )
    decisions_path = destination / "transfer_decisions.jsonl"
    predictions_path = destination / "transfer_predictions.jsonl"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(predictions_path, prediction_rows)
    methods = sorted({str(row["method"]) for row in decisions})
    by_method = {
        method: [row for row in decisions if row["method"] == method]
        for method in methods
    }
    metrics = {method: _summarize_decisions(rows) for method, rows in by_method.items()}
    primary = by_method["learned_milestone_policy"]
    comparisons = {
        f"{method}_over_learned_policy": _paired_bootstrap_rows(
            rows,
            primary,
            samples=int(manifest["evaluation"]["bootstrap_samples"]),
            seed=SEED + index,
        )
        for index, (method, rows) in enumerate(sorted(by_method.items()))
        if method != "learned_milestone_policy"
    }
    primary_per_game = metrics["learned_milestone_policy"]["per_game"]
    nonnegative_games = {
        method: sum(
            metrics[method]["per_game"][game]["mean_utility"]
            >= primary_per_game[game]["mean_utility"]
            for game in TRANSFER_GAMES
        )
        for method in methods
    }
    completion = {
        method: sum(bool(row["completion_selected"]) for row in rows)
        for method, rows in by_method.items()
    }
    oracle_completion = completion["oracle_energy"]
    required_completion = max(
        int(manifest["evaluation"]["completion_absolute_minimum"]),
        math.ceil(
            float(manifest["evaluation"]["completion_fraction_minimum"])
            * oracle_completion
        ),
    )
    learned_method = "policy_temporal_ebm"
    comparison = comparisons[f"{learned_method}_over_learned_policy"]
    checks = {
        "paired_ci_lower_positive": comparison["ci_low"] > 0.0,
        "nonnegative_transfer_games": (
            nonnegative_games[learned_method]
            >= int(manifest["evaluation"]["global_nonnegative_games_minimum"])
        ),
        "completion_absolute_and_fraction": (
            completion[learned_method] >= required_completion
        ),
        "all_conditions_executed": False,
        "future_frames_scoring_only": True,
    }
    global_supported = all(
        value for key, value in checks.items() if key != "all_conditions_executed"
    )
    if global_supported:
        verdict = "GLOBAL_RERANKING_SUPPORTED_ACTIVE_PENDING"
    elif not semantic["behavior_prior_supported"]:
        verdict = "BEHAVIOR_PRIOR_BOTTLENECK"
    elif (
        metrics["oracle_milestone_temporal_ebm"]["mean_utility"]
        > metrics["policy_temporal_ebm"]["mean_utility"]
    ):
        verdict = "MILESTONE_PREDICTOR_BOTTLENECK"
    else:
        verdict = "SEMANTIC_RERANKING_BOTTLENECK"
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "semantic_result_checksum": semantic["semantic_result_checksum"],
        "verdict": verdict,
        "behavior_prior_supported": semantic["behavior_prior_supported"],
        "global_reranking_supported_offline": global_supported,
        "all_conditions_executed": False,
        "checks": checks,
        "metrics": metrics,
        "comparisons": comparisons,
        "nonnegative_games": nonnegative_games,
        "completion_capture": {
            "oracle": oracle_completion,
            "required": required_completion,
            "selected_by_method": completion,
        },
        "panels": len(panels),
        "arms": sum(len(panel.arms) for panel in panels),
        "topology": {
            "future_frames_used_by_policy": False,
            "v4_14_future_predictions_reused": True,
            "future_outcomes_used_for_scoring_only": True,
            "ebm_coefficient": float(manifest["evaluation"]["ebm_coefficient"]),
        },
        "active_validation": {
            "status": "PENDING_BOUNDED_RUN",
            "games": list(ACTIVE_VALIDATION_GAMES),
            "seeds": list(manifest["evaluation"]["active_seeds"]),
        },
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "transfer_decisions": _file_sha256(decisions_path),
            "transfer_predictions": _file_sha256(predictions_path),
            "demonstration_policy": _file_sha256(
                destination / "demonstration_policy.pt"
            ),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def _run_active_policy_controller(
    *,
    controller: str,
    game_id: str,
    seed: int,
    action_budget: int,
    maximum_resets: int,
    policy_model: Any,
    policy_parameters: Mapping[str, Any],
    temporal_model: Any,
    temporal_parameters: Mapping[str, Any],
    temporal_ebm: Any,
    sequence_table: Mapping[tuple[str, ...], float],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.m2.m3_execution_smoke import _reset_env
    from theory.non_ar25_active_micro_run import _env_dir
    from theory.real_env_option_adapter import snapshot_frame
    from theory.sage12.bound_mechanic_pilot import _legal_actions
    from theory.unified_cognition_ab_benchmark import (
        _is_terminal,
        _make_real_env,
    )

    if controller not in {
        "behavior_policy",
        "milestone_policy_temporal_ebm",
    }:
        raise ValueError(f"unknown V4.15 controller: {controller}")
    run_id = f"{game_id}:{seed}:{controller}"
    environment = _make_real_env(game_id, _env_dir())
    frame = _reset_env(environment)
    resets = 1
    actions_executed = 0
    episode_steps = 0
    levels = 0
    wins = 0
    game_overs = 0
    illegal_proposals = 0
    decision_latencies = []
    execution_latencies = []
    candidate_counts = []
    policy_belief = PolicyBelief()
    temporal_belief = TemporalBeliefState()
    traces = []
    stop_reason = "action_budget"
    while actions_executed < action_budget:
        before = snapshot_frame(frame)
        if _is_terminal(before.game_state):
            if resets >= maximum_resets:
                stop_reason = "maximum_resets"
                break
            frame = _reset_env(environment)
            resets += 1
            episode_steps = 0
            policy_belief = PolicyBelief()
            temporal_belief = TemporalBeliefState()
            continue
        legal = tuple(_legal_actions(environment))
        if not legal:
            stop_reason = "no_legal_actions"
            break
        candidate_counts.append(len(legal))
        decision_started = time.perf_counter()
        graphs = [
            _live_candidate_graph(
                game_id=game_id,
                policy_seed=seed,
                reset_index=resets - 1,
                step_index=actions_executed,
                frame=frame,
                legal=legal,
                action=action,
            )
            for action in legal
        ]
        policy = _score_candidate_graphs(
            policy_model,
            graphs,
            parameters=policy_parameters,
            device=device,
            belief=policy_belief,
        )
        selected_temporal_prediction = None
        energies = None
        if controller == "behavior_policy":
            scores = np.asarray(
                policy["behavior_scores"],
                dtype=np.float64,
            )
            action_plans = [
                (str(getattr(action, "name", "")).upper(),) * 3 for action in legal
            ]
        else:
            available = [str(getattr(action, "name", "")).upper() for action in legal]
            action_plans = [
                _candidate_action_plan(
                    str(getattr(action, "name", "")).upper(),
                    available,
                    sequence_table,
                )
                for action in legal
            ]
            graph_plans = [
                tuple(
                    graph if offset == 0 else _graph_for_action(graph, action_name)
                    for offset, action_name in enumerate(action_plan)
                )
                for graph, action_plan in zip(graphs, action_plans)
            ]
            temporal_predictions = _predict_candidate_rollouts(
                temporal_model,
                graph_plans,
                parameters=temporal_parameters,
                device=device,
                initial_belief=temporal_belief,
            )
            energies = np.asarray(
                [
                    temporal_ebm.energies((_prediction_features(prediction),))[0]
                    for prediction in temporal_predictions
                ],
                dtype=np.float64,
            )
            scores = _zscore(policy["learned_scores"]) - EBM_COEFFICIENT * _zscore(
                energies
            )
        maximum = float(np.max(scores))
        candidates = [
            index
            for index, value in enumerate(scores)
            if abs(float(value) - maximum) <= 1e-12
        ]
        selected_index = min(
            candidates,
            key=lambda index: hashlib.sha256(
                (
                    f"{run_id}:{actions_executed}:"
                    f"{_live_action_signature(legal[index])}"
                ).encode()
            ).hexdigest(),
        )
        if controller == "milestone_policy_temporal_ebm":
            selected_temporal_prediction = temporal_predictions[selected_index][0]
        decision_latencies.append(time.perf_counter() - decision_started)
        selected = legal[selected_index]
        execution_started = time.perf_counter()
        try:
            next_frame = _step_env_action(environment, selected)
        except Exception as exc:  # noqa: BLE001 - environment errors are external.
            illegal_proposals += 1
            traces.append(
                {
                    "format_version": FORMAT_VERSION,
                    "run_id": run_id,
                    "action_index": actions_executed,
                    "execution_error": f"{type(exc).__name__}:{exc}",
                }
            )
            stop_reason = "execution_error"
            break
        execution_latencies.append(time.perf_counter() - execution_started)
        after = snapshot_frame(next_frame)
        available_names = tuple(
            sorted({str(getattr(action, "name", "")).upper() for action in legal})
        )
        executed_trace = build_action_target_trace(
            game_id=game_id,
            source_split="source_validation",
            policy_seed=seed,
            reset_index=resets - 1,
            step_index=actions_executed,
            collection_phase="v4_15_active",
            available_action_names=available_names,
            selected_action_name=str(getattr(selected, "name", "")).upper(),
            selected_action_data=dict(getattr(selected, "action_args", {}) or {}),
            frame_before=before.grid,
            frame_after=after.grid,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
        )
        observed_effects, _applicable, _score, _evidence = compile_semantics(
            executed_trace
        )
        policy_belief = _advance_policy_belief(
            policy_model,
            graphs[selected_index],
            observed_effects,
            parameters=policy_parameters,
            device=device,
            belief=policy_belief,
        )
        if selected_temporal_prediction is not None:
            temporal_belief = selected_temporal_prediction.next_belief
        level_delta = max(
            0,
            int(after.levels_completed) - int(before.levels_completed),
        )
        is_win = str(after.game_state).upper() == "WIN"
        is_game_over = str(after.game_state).upper() == "GAME_OVER"
        levels += level_delta
        wins += int(is_win)
        game_overs += int(is_game_over)
        traces.append(
            {
                "format_version": FORMAT_VERSION,
                "run_id": run_id,
                "controller": controller,
                "game_id": game_id,
                "seed": seed,
                "reset_index": resets - 1,
                "action_index": actions_executed,
                "episode_step": episode_steps,
                "pre_state_sha256": grid_sha256(before.grid),
                "post_state_sha256": grid_sha256(after.grid),
                "candidate_count": len(legal),
                "selected_action": _live_action_signature(selected),
                "selected_plan": list(action_plans[selected_index]),
                "policy_score": float(
                    policy["behavior_scores"][selected_index]
                    if controller == "behavior_policy"
                    else policy["learned_scores"][selected_index]
                ),
                "temporal_energy": (
                    float(energies[selected_index]) if energies is not None else None
                ),
                "predicted_milestone": policy["predicted_milestone"],
                "levels_completed_before": before.levels_completed,
                "levels_completed_after": after.levels_completed,
                "game_state_after": after.game_state,
                "decision_seconds": decision_latencies[-1],
                "execution_seconds": execution_latencies[-1],
            }
        )
        actions_executed += 1
        episode_steps += 1
        frame = next_frame
        if _is_terminal(after.game_state):
            if resets >= maximum_resets:
                stop_reason = "maximum_resets"
                break
            frame = _reset_env(environment)
            resets += 1
            episode_steps = 0
            policy_belief = PolicyBelief()
            temporal_belief = TemporalBeliefState()
    return (
        {
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "controller": controller,
            "game_id": game_id,
            "seed": seed,
            "action_budget": action_budget,
            "maximum_resets": maximum_resets,
            "actions_executed": actions_executed,
            "resets": resets,
            "levels_completed": levels,
            "wins": wins,
            "game_overs": game_overs,
            "illegal_proposals": illegal_proposals,
            "stop_reason": stop_reason,
            "mean_candidates": float(np.mean(candidate_counts))
            if candidate_counts
            else 0.0,
            "decision_latency_seconds": {
                "mean": float(np.mean(decision_latencies))
                if decision_latencies
                else 0.0,
                "p95": float(np.quantile(decision_latencies, 0.95))
                if decision_latencies
                else 0.0,
            },
            "execution_latency_seconds": {
                "mean": float(np.mean(execution_latencies))
                if execution_latencies
                else 0.0,
            },
        },
        traces,
    )


def _active_metrics(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = {}
    for controller in sorted({str(row["controller"]) for row in runs}):
        selected = [row for row in runs if row["controller"] == controller]
        per_game = {}
        for game in ACTIVE_VALIDATION_GAMES:
            game_rows = [row for row in selected if row["game_id"] == game]
            per_game[game] = {
                "runs": len(game_rows),
                "actions": sum(int(row["actions_executed"]) for row in game_rows),
                "levels": sum(int(row["levels_completed"]) for row in game_rows),
                "wins": sum(int(row["wins"]) for row in game_rows),
                "game_overs": sum(int(row["game_overs"]) for row in game_rows),
            }
        output[controller] = {
            "runs": len(selected),
            "actions": sum(int(row["actions_executed"]) for row in selected),
            "levels": sum(int(row["levels_completed"]) for row in selected),
            "wins": sum(int(row["wins"]) for row in selected),
            "game_overs": sum(int(row["game_overs"]) for row in selected),
            "illegal_proposals": sum(int(row["illegal_proposals"]) for row in selected),
            "mean_decision_latency_seconds": float(
                np.mean(
                    [float(row["decision_latency_seconds"]["mean"]) for row in selected]
                )
            ),
            "per_game": per_game,
        }
    return output


def run_active_validation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    result_path = destination / "result.json"
    if not result_path.exists():
        raise FileNotFoundError("run V4.15 evaluate before active")
    result = _read_json(result_path)
    semantic = _read_json(destination / "semantic_result.json")
    device = (
        str(semantic["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    policy_model, policy_checkpoint = _load_policy_checkpoint(
        destination / "demonstration_policy.pt",
        device=device,
    )
    policy_parameters = policy_checkpoint["parameters"]
    v414_path = Path(v414_dir)
    temporal_model, temporal_checkpoint = _load_temporal_checkpoint(
        v414_path / "temporal_student.pt",
        device=device,
    )
    temporal_parameters = temporal_checkpoint["parameters"]
    temporal_ebm = _load_active_ebm(
        v414_path / "trajectory_ebm.pt",
        device=device,
    )
    temporal_records = load_temporal_records(v414_path)
    _action_table, sequence_table, _global_value = _action_sequence_tables(
        temporal_records
    )
    baseline_path = v414_path / "active_runs.jsonl"
    frozen_fingerprint = manifest["source_fingerprints"][baseline_path.as_posix()][
        "sha256"
    ]
    if _file_sha256(baseline_path) != frozen_fingerprint:
        raise ValueError("V4.14 active baseline drift")
    baseline_runs = [
        {
            **row,
            "format_version": FORMAT_VERSION,
            "source": "reused_v4_14_content_addressed",
        }
        for row in _read_jsonl(baseline_path)
        if row["controller"] == "action_sequence_only"
    ]
    if len(baseline_runs) != 9:
        raise ValueError("V4.15 expected nine V4.14 baseline runs")
    evaluation = manifest["evaluation"]
    runs = list(baseline_runs)
    traces = []
    started = time.perf_counter()
    for game_id in ACTIVE_VALIDATION_GAMES:
        for seed in evaluation["active_seeds"]:
            for controller in (
                "behavior_policy",
                "milestone_policy_temporal_ebm",
            ):
                run, run_traces = _run_active_policy_controller(
                    controller=controller,
                    game_id=game_id,
                    seed=int(seed),
                    action_budget=int(evaluation["active_action_budget"]),
                    maximum_resets=int(evaluation["active_maximum_resets"]),
                    policy_model=policy_model,
                    policy_parameters=policy_parameters,
                    temporal_model=temporal_model,
                    temporal_parameters=temporal_parameters,
                    temporal_ebm=temporal_ebm,
                    sequence_table=sequence_table,
                    device=device,
                )
                runs.append(run)
                traces.extend(run_traces)
    runs_path = destination / "active_runs.jsonl"
    traces_path = destination / "active_traces.jsonl"
    _write_jsonl(runs_path, runs)
    _write_jsonl(traces_path, traces)
    metrics = _active_metrics(runs)
    baseline_by_key = {
        (str(row["game_id"]), int(row["seed"])): row
        for row in runs
        if row["controller"] == "action_sequence_only"
    }
    paired = []
    for row in runs:
        if row["controller"] == "action_sequence_only":
            continue
        baseline = baseline_by_key[(str(row["game_id"]), int(row["seed"]))]
        paired.append(
            {
                "controller": row["controller"],
                "game_id": row["game_id"],
                "seed": row["seed"],
                "level_gain": int(row["levels_completed"])
                - int(baseline["levels_completed"]),
                "win_gain": int(row["wins"]) - int(baseline["wins"]),
                "game_over_delta": int(row["game_overs"]) - int(baseline["game_overs"]),
            }
        )
    active: dict[str, Any] = {
        "status": "COMPLETE",
        "games": list(ACTIVE_VALIDATION_GAMES),
        "seeds": list(evaluation["active_seeds"]),
        "controllers": [
            "action_sequence_only",
            "behavior_policy",
            "milestone_policy_temporal_ebm",
        ],
        "baseline_reused_from_v4_14": True,
        "baseline_sha256": frozen_fingerprint,
        "fresh_runs": 18,
        "total_runs": 27,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "paired": paired,
        "descriptive_only": True,
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "runs": _file_sha256(runs_path),
            "traces": _file_sha256(traces_path),
        },
    }
    active["active_checksum"] = _checksum(active)
    _write_json(destination / "active_validation.json", active)
    result["active_validation"] = active
    result["all_conditions_executed"] = True
    result["checks"]["all_conditions_executed"] = True
    result["global_reranking_supported"] = all(result["checks"].values())
    learned_levels = int(metrics["milestone_policy_temporal_ebm"]["levels"])
    behavior_levels = int(metrics["behavior_policy"]["levels"])
    if learned_levels > 0 and result["global_reranking_supported"]:
        result["verdict"] = "DEMONSTRATION_RERANKING_SUPPORTED"
    elif learned_levels > 0 or behavior_levels > 0:
        result["verdict"] = "LIVE_PROGRESS_WITH_OFFLINE_GATES_FAILED"
    elif not result["behavior_prior_supported"]:
        result["verdict"] = "BEHAVIOR_PRIOR_BOTTLENECK"
    else:
        result["verdict"] = "LIVE_POLICY_TRANSFER_BOTTLENECK"
    result["artifact_sha256"].update(
        {
            "active_validation": _file_sha256(destination / "active_validation.json"),
            "active_runs": _file_sha256(runs_path),
            "active_traces": _file_sha256(traces_path),
        }
    )
    result.pop("result_checksum", None)
    result["result_checksum"] = _checksum(result)
    _write_json(result_path, result)
    return active


def run_all(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    traces_dir: str | Path = DEFAULT_TRACES_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
    device: str = "auto",
) -> dict[str, Any]:
    manifest_path = Path(output_dir) / "frozen_manifest.json"
    if not manifest_path.exists():
        freeze_manifest(
            output_dir=output_dir,
            traces_dir=traces_dir,
            v414_dir=v414_dir,
        )
    teacher = compile_demonstrations(
        output_dir=output_dir,
        traces_dir=traces_dir,
    )
    semantic = train_demonstration_policy(
        output_dir=output_dir,
        requested_device=device,
    )
    integration = evaluate_transfer_reranking(
        output_dir=output_dir,
        v411_dir=v411_dir,
        v414_dir=v414_dir,
        requested_device=device,
    )
    active = run_active_validation(
        output_dir=output_dir,
        v414_dir=v414_dir,
        requested_device=device,
    )
    return {
        "manifest": load_manifest(output_dir),
        "teacher": teacher,
        "semantic": semantic,
        "integration": integration,
        "active": active,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "freeze",
        "compile",
        "train",
        "evaluate",
        "active",
        "run-all",
    ):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
        )
        child.add_argument(
            "--traces-dir",
            type=Path,
            default=DEFAULT_TRACES_DIR,
        )
        child.add_argument(
            "--v411-dir",
            type=Path,
            default=DEFAULT_V411_DIR,
        )
        child.add_argument(
            "--v414-dir",
            type=Path,
            default=DEFAULT_V414_DIR,
        )
        child.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
            v414_dir=args.v414_dir,
        )
    elif args.command == "compile":
        payload = compile_demonstrations(
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
        )
    elif args.command == "train":
        payload = train_demonstration_policy(
            output_dir=args.output_dir,
            requested_device=args.device,
        )
    elif args.command == "evaluate":
        payload = evaluate_transfer_reranking(
            output_dir=args.output_dir,
            v411_dir=args.v411_dir,
            v414_dir=args.v414_dir,
            requested_device=args.device,
        )
    elif args.command == "active":
        payload = run_active_validation(
            output_dir=args.output_dir,
            v414_dir=args.v414_dir,
            requested_device=args.device,
        )
    else:
        payload = run_all(
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
            v411_dir=args.v411_dir,
            v414_dir=args.v414_dir,
            device=args.device,
        )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MILESTONE_LABELS",
    "DemonstrationChoiceRecord",
    "PolicyBelief",
    "compile_demonstrations",
    "evaluate_transfer_reranking",
    "freeze_manifest",
    "load_demonstration_records",
    "load_manifest",
    "run_active_validation",
    "run_all",
    "tensorize_choices",
    "train_demonstration_policy",
]
