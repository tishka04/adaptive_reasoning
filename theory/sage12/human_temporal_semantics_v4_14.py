"""SAGE12 V4.14 human-trajectory temporal semantics.

This iteration turns the complete human JSONL trajectories into causal
object-relative student examples.  The post-action state and trajectory
suffix are available to the teacher only.  A compact DeepSets + GRU student
predicts immediate semantic effects, persistent role beliefs and multi-scale
progress, then supplies a deployable semantic rollout and pairwise EBM.

The command surface is intentionally artifact oriented::

    python -m theory.sage12.human_temporal_semantics_v4_14 freeze
    python -m theory.sage12.human_temporal_semantics_v4_14 compile
    python -m theory.sage12.human_temporal_semantics_v4_14 train
    python -m theory.sage12.human_temporal_semantics_v4_14 evaluate
    python -m theory.sage12.human_temporal_semantics_v4_14 active
    python -m theory.sage12.human_temporal_semantics_v4_14 run-all

No local semantic gate can skip the global panel evaluation.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import NEURO_HOLDOUT_V1, SOURCE_VALIDATION, short_game_id

from .action_target_data import (
    build_action_target_trace,
    grid_sha256,
)
from .compiler import SLOT_EFFECTS, SlotAnnotation
from .counterfactual_semantic_panels_v4_11 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V411_DIR,
    TeacherArm,
    load_raw_panels,
    load_teacher_panels,
)
from .energy import PairwiseTrajectoryEBM
from .integration_pilot_v4_7 import (
    DEFAULT_MODEL_PATH as DEFAULT_QWEN_MODEL_PATH,
    MAXIMUM_INPUT_TOKENS as QWEN_MAXIMUM_INPUT_TOKENS,
    QWEN_BATCH_SIZE,
    ConstrainedQwenBitDecoder,
)
from .object_relative_student_v4_9 import _token_id, _tokens
from .semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
    _checksum,
    _file_sha256,
    _read_json,
    _write_json,
    _write_jsonl,
    build_object_relative_graph,
    compile_semantics,
)


FORMAT_VERSION = "sage12-human-temporal-semantics-v4.14"
MANIFEST_VERSION = "sage12-human-temporal-semantics-manifest-v4.14"
RECORD_VERSION = "sage12-human-temporal-teacher-record-v4.14"
PREDICTION_VERSION = "sage12-human-temporal-prediction-v4.14"
RESULT_VERSION = "sage12-human-temporal-result-v4.14"
CHECKPOINT_VERSION = "sage12-human-temporal-checkpoint-v4.14"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "human_temporal_semantics_v4_14"
DEFAULT_TRACES_DIR = Path("human_traces")
DEFAULT_V413_DIR = Path("training") / "sage12" / "semantic_bottleneck_curve_v4_13"
SEED = 5_140

HUMAN_TRAIN_GAMES = ("ar25", "bp35", "cd82", "cn04", "dc22", "ft09")
TRANSFER_GAMES = (
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "sp80",
    "su15",
    "tr87",
    "tu93",
)
ACTIVE_VALIDATION_GAMES = tuple(SOURCE_VALIDATION)
FINAL_CONFIRMATION_GAMES = tuple(NEURO_HOLDOUT_V1)
ROLE_LABELS = (
    "controllable",
    "movable",
    "blocker",
    "consumable",
    "hazard",
    "goal_relevant",
)
HORIZONS = (4, 16, 64)
MAX_HISTORY = 32
MAX_DISTANCE = 128
MAXIMUM_NEIGHBORS = 16
ACTIVE_QWEN_REFRESH_STEPS = 128
ACTIVE_QWEN_BLEND_WEIGHT = 0.5
FORBIDDEN_STUDENT_FIELDS = (
    "game_id",
    "episode_id",
    "source_file",
    "frame_before",
    "frame_after",
    "frame_before_sha256",
    "frame_after_sha256",
    "row",
    "col",
    "object_id",
    "target_object_id",
    "hypothesis",
    "objective_guess",
    "game_type_guess",
    "color",
    "colour",
    "value",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _action_names(values: Sequence[Any]) -> tuple[str, ...]:
    rows = []
    for value in values:
        text = str(value).strip().upper()
        if text.startswith("ACTION"):
            rows.append(text)
        elif text.lstrip("-").isdigit():
            rows.append("RESET" if int(text) == 0 else f"ACTION{int(text)}")
        else:
            rows.append(text)
    return tuple(item for item in rows if item and item != "RESET")


def _file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _trace_paths(trace_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(trace_dir.glob("*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no human trace JSONL files under {trace_dir}")
    return paths


def _source_fingerprints(
    trace_dir: Path,
    v411_dir: Path,
    v413_dir: Path,
) -> dict[str, Any]:
    paths = list(_trace_paths(trace_dir))
    paths.extend(
        (
            v411_dir / "frozen_manifest.json",
            v411_dir / "teacher_panels.jsonl",
            v411_dir / "teacher_qa.json",
            v413_dir / "frozen_manifest.json",
            v413_dir / "result.json",
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
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v413_dir: str | Path = DEFAULT_V413_DIR,
) -> dict[str, Any]:
    """Freeze V4.14 before compiling any temporal result."""

    destination = Path(output_dir)
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "split": {
            "human_train": list(HUMAN_TRAIN_GAMES),
            "transfer_evaluation": list(TRANSFER_GAMES),
            "active_validation": list(ACTIVE_VALIDATION_GAMES),
            "final_confirmation_closed": list(FINAL_CONFIRMATION_GAMES),
            "legacy_sage11_registry_unchanged": True,
        },
        "source_fingerprints": _source_fingerprints(
            Path(traces_dir), Path(v411_dir), Path(v413_dir)
        ),
        "teacher": {
            "semantic_effects": list(SEMANTIC_EFFECTS),
            "roles": list(ROLE_LABELS),
            "horizons": list(HORIZONS),
            "maximum_distance": MAX_DISTANCE,
            "progress_gamma": 0.97,
            "danger_horizon": 8,
            "quit_is_automatic_failure": False,
            "human_text_is_teacher_only": True,
        },
        "student": {
            "model": "object_relative_deepsets_gru",
            "hash_buckets": 2048,
            "embedding_width": 32,
            "graph_hidden_width": 96,
            "temporal_hidden_width": 128,
            "maximum_neighbors": MAXIMUM_NEIGHBORS,
            "history_window": MAX_HISTORY,
            "epochs": 30,
            "learning_rate": 0.0015,
            "weight_decay": 0.0001,
            "positive_weight_cap": 20.0,
            "loss_weights": {
                "immediate_semantics": 1.0,
                "next_belief": 1.0,
                "multi_horizon": 0.5,
                "suffix_ranking": 0.5,
                "censored_distance": 0.25,
                "identity_confusion": 0.1,
            },
            "seed": SEED,
            "outer_diagnostic": "leave_one_human_game_out",
            "final_fit": "all_six_human_games",
        },
        "rollout": {
            "depth": 3,
            "future_descriptors_allowed": False,
            "feedback": "predicted_semantic_effects",
            "candidate_source": "legal_compiled_action_sequence",
        },
        "evaluation": {
            "all_conditions_run_unconditionally": True,
            "bootstrap_samples": 1_000,
            "bootstrap_seed": SEED + 1,
            "nonnegative_transfer_games_minimum": 5,
            "completion_capture_fraction_of_oracle_minimum": 0.5,
            "completion_capture_absolute_minimum": 1,
            "active_validation_seeds": [0, 1, 2],
            "active_validation_action_budget": 1_000,
            "active_validation_maximum_resets": 14,
        },
        "forbidden_student_fields": list(FORBIDDEN_STUDENT_FIELDS),
        "authority_promoted": False,
        "holdout_opened": False,
        "result_observed_at_freeze": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / "frozen_manifest.json")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.14 manifest")
    expected = str(manifest["manifest_checksum"])
    payload = dict(manifest)
    payload.pop("manifest_checksum")
    if _checksum(payload) != expected:
        raise ValueError("V4.14 manifest checksum mismatch")
    split = manifest["split"]
    if tuple(split["human_train"]) != HUMAN_TRAIN_GAMES:
        raise ValueError("V4.14 human-train split drift")
    if tuple(split["transfer_evaluation"]) != TRANSFER_GAMES:
        raise ValueError("V4.14 transfer split drift")
    return manifest


@dataclass(frozen=True)
class TemporalBeliefState:
    """Deployable recurrent semantic state; no game or object identity."""

    hidden: tuple[float, ...] = ()
    effect_probabilities: Mapping[str, float] = field(default_factory=dict)
    role_probabilities: Mapping[str, float] = field(default_factory=dict)
    progress_probabilities: Mapping[str, float] = field(default_factory=dict)
    risk_probability: float = 0.0
    step_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class TemporalSlotPrediction:
    """Prediction for one candidate action plus its successor belief."""

    effect_probabilities: Mapping[str, float]
    role_probabilities: Mapping[str, float]
    progress_probabilities: Mapping[str, float]
    danger_within_8: float
    normalized_distance: float
    next_belief: TemporalBeliefState

    def as_slot_annotation(
        self,
        slot_id: str,
        *,
        source: str = "human_temporal_v4_14",
    ) -> SlotAnnotation:
        return SlotAnnotation(
            slot_id=slot_id,
            effect_probabilities={
                effect: float(self.effect_probabilities[effect])
                for effect in SLOT_EFFECTS
            },
            source=source,
            support=0,
        )


@dataclass(frozen=True)
class TemporalTeacherRecord:
    example_id: str
    game_id: str
    episode_id: str
    step: int
    sequence_index: int
    source_file: str
    pre_state_sha256: str
    post_state_sha256: str
    graph: ObjectRelativeGraph
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    productive_score: float
    roles: Mapping[str, bool]
    horizon_progress: Mapping[str, bool]
    danger_within_8: bool
    steps_to_next_level: int
    distance_censored: bool
    discounted_progress: float
    human_teacher: Mapping[str, Any]
    format_version: str = RECORD_VERSION

    def __post_init__(self) -> None:
        if self.game_id not in HUMAN_TRAIN_GAMES:
            raise ValueError(
                f"V4.14 teacher record outside human train: {self.game_id}"
            )
        if set(self.labels) != set(SEMANTIC_EFFECTS):
            raise ValueError("V4.14 effect vocabulary drift")
        if set(self.applicable) != set(SEMANTIC_EFFECTS):
            raise ValueError("V4.14 applicability vocabulary drift")
        if set(self.roles) != set(ROLE_LABELS):
            raise ValueError("V4.14 role vocabulary drift")
        if set(self.horizon_progress) != {f"within_{value}" for value in HORIZONS}:
            raise ValueError("V4.14 horizon vocabulary drift")
        _assert_student_view_safe(self.graph.to_dict())

    @property
    def sequence_key(self) -> str:
        return f"{self.game_id}:{self.episode_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "example_id": self.example_id,
            "audit": {
                "game_id": self.game_id,
                "episode_id": self.episode_id,
                "step": self.step,
                "sequence_index": self.sequence_index,
                "source_file": self.source_file,
                "pre_state_sha256": self.pre_state_sha256,
                "post_state_sha256": self.post_state_sha256,
            },
            "student_view": {
                "model_graph": self.graph.to_dict(),
                "causal_history": {
                    "maximum_previous_transitions": MAX_HISTORY,
                    "ends_before_current_action": True,
                },
            },
            "teacher": {
                "labels": dict(self.labels),
                "applicable": dict(self.applicable),
                "productive_score": self.productive_score,
                "roles": dict(self.roles),
                "horizon_progress": dict(self.horizon_progress),
                "danger_within_8": self.danger_within_8,
                "steps_to_next_level": self.steps_to_next_level,
                "distance_censored": self.distance_censored,
                "discounted_progress": self.discounted_progress,
                "human_annotation": _json_safe(self.human_teacher),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalTeacherRecord":
        audit = payload["audit"]
        view = payload["student_view"]["model_graph"]
        teacher = payload["teacher"]
        return cls(
            example_id=str(payload["example_id"]),
            game_id=str(audit["game_id"]),
            episode_id=str(audit["episode_id"]),
            step=int(audit["step"]),
            sequence_index=int(audit["sequence_index"]),
            source_file=str(audit["source_file"]),
            pre_state_sha256=str(audit["pre_state_sha256"]),
            post_state_sha256=str(audit["post_state_sha256"]),
            graph=ObjectRelativeGraph(
                root=dict(view["root"]),
                neighbors=tuple(dict(row) for row in view["neighbors"]),
            ),
            labels={
                effect: bool(teacher["labels"][effect]) for effect in SEMANTIC_EFFECTS
            },
            applicable={
                effect: bool(teacher["applicable"][effect])
                for effect in SEMANTIC_EFFECTS
            },
            productive_score=float(teacher["productive_score"]),
            roles={role: bool(teacher["roles"][role]) for role in ROLE_LABELS},
            horizon_progress={
                f"within_{value}": bool(teacher["horizon_progress"][f"within_{value}"])
                for value in HORIZONS
            },
            danger_within_8=bool(teacher["danger_within_8"]),
            steps_to_next_level=int(teacher["steps_to_next_level"]),
            distance_censored=bool(teacher["distance_censored"]),
            discounted_progress=float(teacher["discounted_progress"]),
            human_teacher=dict(teacher.get("human_annotation", {})),
            format_version=str(payload["format_version"]),
        )


def _assert_student_view_safe(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).lower()
    for field_name in FORBIDDEN_STUDENT_FIELDS:
        if f'"{field_name.lower()}"' in encoded:
            raise ValueError(f"forbidden V4.14 student field: {field_name}")


def _episode_summaries(trace_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for path in sorted(trace_dir.glob("*.episodes.jsonl")):
        for row in _read_jsonl(path):
            key = (short_game_id(row["game_id"]), str(row["episode_id"]))
            output[key] = {**row, "_source_file": path.as_posix()}
    return output


def _step_rows(trace_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(trace_dir.glob("*.steps.jsonl")):
        for row in _read_jsonl(path):
            game = short_game_id(row["game_id"])
            if game not in HUMAN_TRAIN_GAMES:
                continue
            key = (game, str(row["episode_id"]))
            grouped[key].append({**row, "_source_file": path.as_posix()})
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("step", 0)))
    return grouped


def _future_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    index: int,
    level_before: Sequence[int],
) -> tuple[dict[str, bool], bool, int, bool, float]:
    level_events = []
    danger_events = []
    for offset in range(index, len(rows)):
        after = int(rows[offset].get("levels_completed_after", 0) or 0)
        before = int(level_before[offset])
        state = str(rows[offset].get("game_state_after", "")).upper()
        if after > before or state == "WIN":
            level_events.append(offset)
        if state == "GAME_OVER":
            danger_events.append(offset)
    next_index = level_events[0] if level_events else None
    raw_distance = (
        int(next_index - index + 1) if next_index is not None else MAX_DISTANCE
    )
    censored = next_index is None or raw_distance > MAX_DISTANCE
    distance = min(raw_distance, MAX_DISTANCE)
    progress = {
        f"within_{horizon}": bool(
            next_index is not None and next_index - index + 1 <= horizon
        )
        for horizon in HORIZONS
    }
    danger = bool(danger_events and danger_events[0] - index + 1 <= 8)
    discounted = 0.0 if censored else float(0.97**distance)
    return progress, danger, distance, censored, discounted


def _role_labels(
    labels: Mapping[str, bool],
    horizon: Mapping[str, bool],
    *,
    danger: bool,
) -> dict[str, bool]:
    return {
        "controllable": bool(labels["moved"]),
        "movable": bool(labels["target_moved"]),
        "blocker": bool(labels["path_opened"] or labels["path_closed"]),
        "consumable": bool(labels["target_removed"]),
        "hazard": bool(labels["risk"] or danger),
        "goal_relevant": bool(
            labels["productive"] or labels["level_complete"] or horizon["within_16"]
        ),
    }


def compile_human_teacher(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    traces_dir: str | Path = DEFAULT_TRACES_DIR,
) -> dict[str, Any]:
    """Compile every recorded human transition into a causal temporal row."""

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    trace_dir = Path(traces_dir)
    summaries = _episode_summaries(trace_dir)
    grouped = _step_rows(trace_dir)
    records: list[TemporalTeacherRecord] = []
    continuity_links = 0
    continuity_matches = 0
    orphan_sequences = 0
    for (game, episode_id), raw_rows in sorted(grouped.items()):
        summary = summaries.get((game, episode_id))
        if summary is None:
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
            level_before_rows = [int(row["_levels_before"]) for row in play_rows]
            for left, right in zip(play_rows, play_rows[1:]):
                continuity_links += 1
                continuity_matches += int(
                    left.get("frame_after") == right.get("frame_before")
                )
            for index, row in enumerate(play_rows):
                game_state_before = str(row["_game_state_before"])
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
                    collection_phase="human_trace_v4_14",
                    available_action_names=available,
                    selected_action_name=action,
                    selected_action_data=dict(row.get("action_args") or {}),
                    frame_before=row["frame_before"],
                    frame_after=row["frame_after"],
                    game_state_before=game_state_before,
                    game_state_after=str(row.get("game_state_after", "NOT_FINISHED")),
                    levels_completed_before=int(level_before_rows[index]),
                    levels_completed_after=int(
                        row.get(
                            "levels_completed_after",
                            level_before_rows[index],
                        )
                    ),
                )
                labels, applicable, score, _evidence = compile_semantics(trace)
                (
                    progress,
                    danger,
                    distance,
                    censored,
                    discounted,
                ) = _future_targets(
                    play_rows,
                    index=index,
                    level_before=level_before_rows,
                )
                roles = _role_labels(labels, progress, danger=danger)
                human = {
                    "intent": str(row.get("intent", "")),
                    "hypothesis": str(row.get("hypothesis", "")),
                    "cognitive_events": list(row.get("cognitive_events") or ()),
                    "game_type_guess": (
                        str(summary.get("game_type_guess", "")) if summary else ""
                    ),
                    "objective_guess": (
                        str(summary.get("objective_guess", "")) if summary else ""
                    ),
                    "discovered_mechanics": (
                        list(summary.get("discovered_mechanics") or ())
                        if summary
                        else []
                    ),
                    "discovered_mistakes": (
                        list(summary.get("discovered_mistakes") or ())
                        if summary
                        else []
                    ),
                }
                pre_hash = grid_sha256(row["frame_before"])
                post_hash = grid_sha256(row["frame_after"])
                example_id = (
                    "ht14_"
                    + _checksum(
                        {
                            "game": game,
                            "episode": sequence_id,
                            "step": int(row.get("step", index)),
                            "pre": pre_hash,
                            "action": action,
                            "args": row.get("action_args"),
                        }
                    )[:20]
                )
                records.append(
                    TemporalTeacherRecord(
                        example_id=example_id,
                        game_id=game,
                        episode_id=sequence_id,
                        step=int(row.get("step", index)),
                        sequence_index=index,
                        source_file=str(row["_source_file"]),
                        pre_state_sha256=pre_hash,
                        post_state_sha256=post_hash,
                        graph=build_object_relative_graph(trace),
                        labels=labels,
                        applicable=applicable,
                        productive_score=float(score),
                        roles=roles,
                        horizon_progress=progress,
                        danger_within_8=danger,
                        steps_to_next_level=distance,
                        distance_censored=censored,
                        discounted_progress=discounted,
                        human_teacher=human,
                    )
                )

    corpus_path = destination / "teacher_corpus.jsonl"
    _write_jsonl(corpus_path, (record.to_dict() for record in records))
    by_game = {}
    for game in HUMAN_TRAIN_GAMES:
        selected = [record for record in records if record.game_id == game]
        by_game[game] = {
            "records": len(selected),
            "episodes": len({record.episode_id for record in selected}),
            "immediate_positive": {
                effect: sum(record.labels[effect] for record in selected)
                for effect in SEMANTIC_EFFECTS
            },
            "role_positive": {
                role: sum(record.roles[role] for record in selected)
                for role in ROLE_LABELS
            },
            "horizon_positive": {
                key: sum(record.horizon_progress[key] for record in selected)
                for key in (f"within_{value}" for value in HORIZONS)
            },
            "danger_within_8": sum(record.danger_within_8 for record in selected),
            "uncensored_distance": sum(
                not record.distance_censored for record in selected
            ),
        }
    qa: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "records": len(records),
        "games": list(HUMAN_TRAIN_GAMES),
        "sequences": len({record.sequence_key for record in records}),
        "orphan_sequences": orphan_sequences,
        "continuity": {
            "links": continuity_links,
            "matches": continuity_matches,
            "fraction": (
                continuity_matches / continuity_links if continuity_links else 1.0
            ),
        },
        "student_view_forbidden_fields_absent": True,
        "by_game": by_game,
        "artifact_sha256": _file_sha256(corpus_path),
    }
    qa["teacher_ready"] = bool(
        records
        and set(by_game) == set(HUMAN_TRAIN_GAMES)
        and continuity_links == continuity_matches
    )
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def load_teacher_records(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[TemporalTeacherRecord, ...]:
    return tuple(
        TemporalTeacherRecord.from_dict(row)
        for row in _read_jsonl(Path(output_dir) / "teacher_corpus.jsonl")
    )


@dataclass(frozen=True)
class TensorizedTemporal:
    root_ids: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_mask: np.ndarray
    feedback: np.ndarray
    labels: np.ndarray
    applicable: np.ndarray
    roles: np.ndarray
    horizons: np.ndarray
    danger: np.ndarray
    distance: np.ndarray
    uncensored: np.ndarray
    next_belief: np.ndarray
    next_mask: np.ndarray
    sequences: tuple[tuple[int, ...], ...]


def _record_sequences(
    records: Sequence[TemporalTeacherRecord],
) -> tuple[tuple[int, ...], ...]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.sequence_key].append(index)
    return tuple(
        tuple(
            sorted(
                values,
                key=lambda index: (
                    records[index].sequence_index,
                    records[index].step,
                ),
            )
        )
        for _key, values in sorted(grouped.items())
    )


def _tensorize_graphs(
    records: Sequence[TemporalTeacherRecord],
    *,
    hash_buckets: int,
    maximum_neighbors: int,
    relation_shuffle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graphs = [
        record.graph.relation_shuffled() if relation_shuffle else record.graph
        for record in records
    ]
    root_rows = [_tokens("root", dict(graph.root)) for graph in graphs]
    neighbor_rows = [
        [
            _tokens("neighbor", dict(neighbor))
            for neighbor in graph.neighbors[:maximum_neighbors]
        ]
        for graph in graphs
    ]
    root_width = max(max((len(row) for row in root_rows), default=1), 1)
    node_width = max(
        max(
            (len(tokens) for neighbors in neighbor_rows for tokens in neighbors),
            default=1,
        ),
        1,
    )
    roots = np.zeros((len(records), root_width), dtype=np.int64)
    nodes = np.zeros(
        (len(records), maximum_neighbors, node_width),
        dtype=np.int64,
    )
    mask = np.zeros((len(records), maximum_neighbors), dtype=np.float32)
    for index, tokens in enumerate(root_rows):
        roots[index, : len(tokens)] = [
            _token_id(token, hash_buckets) for token in tokens
        ]
    for index, neighbors in enumerate(neighbor_rows):
        for neighbor_index, tokens in enumerate(neighbors):
            nodes[index, neighbor_index, : len(tokens)] = [
                _token_id(token, hash_buckets) for token in tokens
            ]
            mask[index, neighbor_index] = 1.0
    return roots, nodes, mask


def tensorize_temporal_records(
    records: Sequence[TemporalTeacherRecord],
    *,
    hash_buckets: int,
    maximum_neighbors: int,
    relation_shuffle: bool = False,
    history_shuffle: bool = False,
    shuffle_seed: int = SEED,
) -> TensorizedTemporal:
    roots, nodes, neighbor_mask = _tensorize_graphs(
        records,
        hash_buckets=hash_buckets,
        maximum_neighbors=maximum_neighbors,
        relation_shuffle=relation_shuffle,
    )
    labels = np.asarray(
        [
            [float(record.labels[effect]) for effect in SEMANTIC_EFFECTS]
            for record in records
        ],
        dtype=np.float32,
    )
    applicable = np.asarray(
        [
            [float(record.applicable[effect]) for effect in SEMANTIC_EFFECTS]
            for record in records
        ],
        dtype=np.float32,
    )
    roles = np.asarray(
        [[float(record.roles[role]) for role in ROLE_LABELS] for record in records],
        dtype=np.float32,
    )
    horizons = np.asarray(
        [
            [
                float(record.horizon_progress[f"within_{horizon}"])
                for horizon in HORIZONS
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    danger = np.asarray(
        [[float(record.danger_within_8)] for record in records],
        dtype=np.float32,
    )
    distance = np.asarray(
        [[float(record.steps_to_next_level) / MAX_DISTANCE] for record in records],
        dtype=np.float32,
    )
    uncensored = np.asarray(
        [[float(not record.distance_censored)] for record in records],
        dtype=np.float32,
    )
    sequences = _record_sequences(records)
    feedback = np.zeros_like(labels)
    next_belief = np.zeros(
        (len(records), len(SEMANTIC_EFFECTS) + len(ROLE_LABELS)),
        dtype=np.float32,
    )
    next_mask = np.zeros((len(records), 1), dtype=np.float32)
    for sequence in sequences:
        for offset, index in enumerate(sequence):
            if offset:
                feedback[index] = labels[sequence[offset - 1]]
            if offset + 1 < len(sequence):
                successor = sequence[offset + 1]
                next_belief[index] = np.concatenate(
                    (labels[successor], roles[successor])
                )
                next_mask[index] = 1.0
    if history_shuffle:
        rng = np.random.default_rng(shuffle_seed)
        for sequence in sequences:
            if len(sequence) <= 1:
                continue
            values = feedback[np.asarray(sequence, dtype=np.int64)].copy()
            rng.shuffle(values, axis=0)
            feedback[np.asarray(sequence, dtype=np.int64)] = values
    return TensorizedTemporal(
        root_ids=roots,
        neighbor_ids=nodes,
        neighbor_mask=neighbor_mask,
        feedback=feedback,
        labels=labels,
        applicable=applicable,
        roles=roles,
        horizons=horizons,
        danger=danger,
        distance=distance,
        uncensored=uncensored,
        next_belief=next_belief,
        next_mask=next_mask,
        sequences=sequences,
    )


def _torch_model(
    *,
    hash_buckets: int,
    embedding_width: int,
    graph_hidden_width: int,
    temporal_hidden_width: int,
    identity_classes: int,
) -> Any:
    import torch

    class HumanTemporalSemanticModel(torch.nn.Module):
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
                torch.nn.Linear(embedding_width * 3, graph_hidden_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(graph_hidden_width),
                torch.nn.Linear(graph_hidden_width, graph_hidden_width),
                torch.nn.GELU(),
            )
            self.temporal = torch.nn.GRU(
                graph_hidden_width + len(SEMANTIC_EFFECTS),
                temporal_hidden_width,
                batch_first=True,
            )
            self.effect_head = torch.nn.Linear(
                temporal_hidden_width,
                len(SEMANTIC_EFFECTS),
            )
            self.role_head = torch.nn.Linear(
                temporal_hidden_width,
                len(ROLE_LABELS),
            )
            self.horizon_head = torch.nn.Linear(
                temporal_hidden_width,
                len(HORIZONS) + 1,
            )
            self.distance_head = torch.nn.Sequential(
                torch.nn.Linear(temporal_hidden_width, 1),
                torch.nn.Sigmoid(),
            )
            self.next_belief_head = torch.nn.Linear(
                temporal_hidden_width,
                len(SEMANTIC_EFFECTS) + len(ROLE_LABELS),
            )
            self.identity_head = torch.nn.Sequential(
                torch.nn.Linear(temporal_hidden_width, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, max(1, identity_classes)),
            )

        @staticmethod
        def _mean_tokens(ids: Any, embeddings: Any) -> Any:
            token_mask = (ids != 0).to(embeddings.dtype)
            total = (embeddings * token_mask.unsqueeze(-1)).sum(dim=-2)
            denominator = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return total / denominator

        def encode_graph(
            self,
            root_ids: Any,
            neighbor_ids: Any,
            neighbor_mask: Any,
        ) -> Any:
            root = self._mean_tokens(root_ids, self.embedding(root_ids))
            nodes = self._mean_tokens(
                neighbor_ids,
                self.embedding(neighbor_ids),
            )
            nodes = self.node_encoder(nodes)
            mask = neighbor_mask.unsqueeze(-1)
            mean = (nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            masked = nodes.masked_fill(mask == 0, -1e4)
            maximum = masked.max(dim=1).values
            empty = neighbor_mask.sum(dim=1, keepdim=True) == 0
            maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
            return self.graph_trunk(torch.cat((root, mean, maximum), dim=-1))

        def forward(
            self,
            root_ids: Any,
            neighbor_ids: Any,
            neighbor_mask: Any,
            feedback: Any,
            hidden: Any = None,
        ) -> tuple[Any, ...]:
            graph = self.encode_graph(
                root_ids,
                neighbor_ids,
                neighbor_mask,
            )
            values = torch.cat((graph, feedback), dim=-1).unsqueeze(0)
            temporal, next_hidden = self.temporal(values, hidden)
            latent = temporal.squeeze(0)
            horizon = self.horizon_head(latent)
            return (
                self.effect_head(latent),
                self.role_head(latent),
                horizon[:, : len(HORIZONS)],
                horizon[:, len(HORIZONS) :],
                self.distance_head(latent),
                self.next_belief_head(latent),
                latent,
                next_hidden,
            )

    return HumanTemporalSemanticModel()


def _batch(
    tensors: TensorizedTemporal,
    indices: Sequence[int],
    *,
    device: str,
) -> tuple[Any, ...]:
    import torch

    selected = np.asarray(indices, dtype=np.int64)
    return (
        torch.as_tensor(
            tensors.root_ids[selected],
            dtype=torch.long,
            device=device,
        ),
        torch.as_tensor(
            tensors.neighbor_ids[selected],
            dtype=torch.long,
            device=device,
        ),
        torch.as_tensor(
            tensors.neighbor_mask[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.feedback[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.labels[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.applicable[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.roles[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.horizons[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.danger[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.distance[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.uncensored[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.next_belief[selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            tensors.next_mask[selected],
            dtype=torch.float32,
            device=device,
        ),
    )


def _masked_balanced_bce(
    logits: Any,
    targets: Any,
    mask: Any,
    positive_weight: Any,
) -> Any:
    import torch

    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def _positive_weights(
    values: np.ndarray,
    masks: np.ndarray,
    *,
    maximum: float,
) -> np.ndarray:
    positive = np.sum(values * masks, axis=0)
    negative = np.sum((1.0 - values) * masks, axis=0)
    weights = np.divide(
        negative,
        np.maximum(positive, 1.0),
        out=np.ones_like(positive, dtype=np.float64),
    )
    return np.clip(weights, 1.0, maximum).astype(np.float32)


def _pair_indices(
    records: Sequence[TemporalTeacherRecord],
    allowed: set[int],
) -> list[tuple[int, int]]:
    by_state: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in sorted(allowed):
        record = records[index]
        by_state[(record.game_id, record.pre_state_sha256)].append(index)
    pairs = []
    for values in by_state.values():
        for left_offset, left in enumerate(values):
            for right in values[left_offset + 1 :]:
                delta = (
                    records[left].discounted_progress
                    - float(records[left].danger_within_8)
                    - records[right].discounted_progress
                    + float(records[right].danger_within_8)
                )
                if abs(delta) <= 1e-9:
                    continue
                pairs.append((left, right) if delta > 0 else (right, left))
    return pairs


def _benchmark_device(parameters: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    timings = {}
    for device in devices:
        model = _torch_model(
            hash_buckets=int(parameters["hash_buckets"]),
            embedding_width=int(parameters["embedding_width"]),
            graph_hidden_width=int(parameters["graph_hidden_width"]),
            temporal_hidden_width=int(parameters["temporal_hidden_width"]),
            identity_classes=6,
        ).to(device)
        root = torch.randint(
            1,
            int(parameters["hash_buckets"]),
            (256, 12),
            device=device,
        )
        nodes = torch.randint(
            1,
            int(parameters["hash_buckets"]),
            (256, MAXIMUM_NEIGHBORS, 10),
            device=device,
        )
        mask = torch.ones(
            (256, MAXIMUM_NEIGHBORS),
            dtype=torch.float32,
            device=device,
        )
        feedback = torch.zeros(
            (256, len(SEMANTIC_EFFECTS)),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3):
            model(root, nodes, mask, feedback)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(20):
                model(root, nodes, mask, feedback)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings[device] = time.perf_counter() - started
    cpu = float(timings["cpu"])
    cuda = float(timings.get("cuda:0", math.inf))
    speedup = cpu / cuda if math.isfinite(cuda) and cuda > 0 else 0.0
    selected = "cuda:0" if speedup >= 1.2 else "cpu"
    return {
        "timings_seconds": timings,
        "cuda_speedup": speedup,
        "selected_device": selected,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _fit_model(
    records: Sequence[TemporalTeacherRecord],
    tensors: TensorizedTemporal,
    *,
    train_indices: np.ndarray,
    parameters: Mapping[str, Any],
    device: str,
    seed: int,
    epochs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_set = {int(value) for value in train_indices}
    train_games = sorted({records[index].game_id for index in train_set})
    game_index = {game: index for index, game in enumerate(train_games)}
    model = _torch_model(
        hash_buckets=int(parameters["hash_buckets"]),
        embedding_width=int(parameters["embedding_width"]),
        graph_hidden_width=int(parameters["graph_hidden_width"]),
        temporal_hidden_width=int(parameters["temporal_hidden_width"]),
        identity_classes=len(train_games),
    ).to(device)
    main_parameters = [
        value
        for name, value in model.named_parameters()
        if not name.startswith("identity_head.")
    ]
    optimizer = torch.optim.AdamW(
        main_parameters,
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    adversary = torch.optim.AdamW(
        model.identity_head.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    maximum_weight = float(parameters["positive_weight_cap"])
    index_array = np.asarray(sorted(train_set), dtype=np.int64)
    effect_weight = torch.as_tensor(
        _positive_weights(
            tensors.labels[index_array],
            tensors.applicable[index_array],
            maximum=maximum_weight,
        ),
        dtype=torch.float32,
        device=device,
    )
    role_weight = torch.as_tensor(
        _positive_weights(
            tensors.roles[index_array],
            np.ones_like(tensors.roles[index_array]),
            maximum=maximum_weight,
        ),
        dtype=torch.float32,
        device=device,
    )
    horizon_targets = np.concatenate(
        (tensors.horizons[index_array], tensors.danger[index_array]),
        axis=1,
    )
    horizon_weight = torch.as_tensor(
        _positive_weights(
            horizon_targets,
            np.ones_like(horizon_targets),
            maximum=maximum_weight,
        ),
        dtype=torch.float32,
        device=device,
    )
    belief_targets = tensors.next_belief[index_array]
    belief_masks = np.repeat(
        tensors.next_mask[index_array],
        belief_targets.shape[1],
        axis=1,
    )
    belief_weight = torch.as_tensor(
        _positive_weights(
            belief_targets,
            belief_masks,
            maximum=maximum_weight,
        ),
        dtype=torch.float32,
        device=device,
    )
    sequences = [
        tuple(index for index in sequence if index in train_set)
        for sequence in tensors.sequences
    ]
    sequences = [sequence for sequence in sequences if sequence]
    pairs = _pair_indices(records, train_set)
    losses: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    epoch_count = int(epochs if epochs is not None else parameters["epochs"])
    weights = parameters["loss_weights"]
    for epoch in range(epoch_count):
        order = list(sequences)
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for sequence in order:
            hidden = None
            for start in range(0, len(sequence), MAX_HISTORY):
                indices = sequence[start : start + MAX_HISTORY]
                batch = _batch(tensors, indices, device=device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(*batch[:4], hidden)
                (
                    effect_logits,
                    role_logits,
                    horizon_logits,
                    danger_logits,
                    distance,
                    belief_logits,
                    latent,
                    hidden,
                ) = outputs
                hidden = hidden.detach()
                semantic_loss = _masked_balanced_bce(
                    effect_logits,
                    batch[4],
                    batch[5],
                    effect_weight,
                )
                role_loss = _masked_balanced_bce(
                    role_logits,
                    batch[6],
                    torch.ones_like(batch[6]),
                    role_weight,
                )
                combined_horizon = torch.cat(
                    (horizon_logits, danger_logits),
                    dim=1,
                )
                combined_target = torch.cat((batch[7], batch[8]), dim=1)
                horizon_loss = _masked_balanced_bce(
                    combined_horizon,
                    combined_target,
                    torch.ones_like(combined_target),
                    horizon_weight,
                )
                belief_mask = batch[12].repeat(
                    1,
                    belief_logits.shape[1],
                )
                belief_loss = _masked_balanced_bce(
                    belief_logits,
                    batch[11],
                    belief_mask,
                    belief_weight,
                )
                distance_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        distance,
                        batch[9],
                        reduction="none",
                    )
                    * batch[10]
                ).sum() / batch[10].sum().clamp_min(1.0)
                identity_probabilities = torch.softmax(
                    model.identity_head(latent),
                    dim=-1,
                )
                uniform = torch.full_like(
                    identity_probabilities,
                    1.0 / identity_probabilities.shape[-1],
                )
                confusion = torch.nn.functional.kl_div(
                    torch.log(identity_probabilities.clamp_min(1e-8)),
                    uniform,
                    reduction="batchmean",
                )
                loss = (
                    float(weights["immediate_semantics"]) * (semantic_loss + role_loss)
                    + float(weights["next_belief"]) * belief_loss
                    + float(weights["multi_horizon"]) * horizon_loss
                    + float(weights["censored_distance"]) * distance_loss
                    + float(weights["identity_confusion"]) * confusion
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_parameters, 2.0)
                optimizer.step()

                adversary.zero_grad(set_to_none=True)
                identity_logits = model.identity_head(latent.detach())
                identity_targets = torch.as_tensor(
                    [game_index[records[index].game_id] for index in indices],
                    dtype=torch.long,
                    device=device,
                )
                identity_loss = torch.nn.functional.cross_entropy(
                    identity_logits,
                    identity_targets,
                )
                identity_loss.backward()
                adversary.step()
                losses["semantic"].append(float(semantic_loss.detach().cpu()))
                losses["roles"].append(float(role_loss.detach().cpu()))
                losses["horizon"].append(float(horizon_loss.detach().cpu()))
                losses["belief"].append(float(belief_loss.detach().cpu()))
                losses["distance"].append(float(distance_loss.detach().cpu()))

        if pairs:
            selected_pairs = list(pairs)
            random.Random(seed + 10_000 + epoch).shuffle(selected_pairs)
            selected_pairs = selected_pairs[: min(512, len(selected_pairs))]
            preferred = np.asarray(
                [left for left, _right in selected_pairs],
                dtype=np.int64,
            )
            rejected = np.asarray(
                [right for _left, right in selected_pairs],
                dtype=np.int64,
            )
            optimizer.zero_grad(set_to_none=True)
            left_batch = _batch(tensors, preferred, device=device)
            right_batch = _batch(tensors, rejected, device=device)
            left_outputs = model(*left_batch[:4])
            right_outputs = model(*right_batch[:4])
            left_score = (
                left_outputs[2][:, 1]
                + left_outputs[0][
                    :,
                    SEMANTIC_EFFECTS.index("productive"),
                ]
                - left_outputs[3][:, 0]
            )
            right_score = (
                right_outputs[2][:, 1]
                + right_outputs[0][
                    :,
                    SEMANTIC_EFFECTS.index("productive"),
                ]
                - right_outputs[3][:, 0]
            )
            ranking = torch.nn.functional.softplus(-(left_score - right_score)).mean()
            (float(weights["suffix_ranking"]) * ranking).backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, 2.0)
            optimizer.step()
            losses["ranking"].append(float(ranking.detach().cpu()))
    model.eval()
    return model, {
        "runtime_seconds": time.perf_counter() - started,
        "device": device,
        "epochs": epoch_count,
        "train_rows": len(train_set),
        "train_games": train_games,
        "ranking_pairs": len(pairs),
        "final_losses": {
            name: float(np.mean(values[-max(1, len(sequences)) :]))
            for name, values in losses.items()
            if values
        },
    }


def _predict_model(
    model: Any,
    tensors: TensorizedTemporal,
    *,
    indices: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    import torch

    allowed = {int(value) for value in indices}
    output = {
        "effects": np.zeros(
            (len(indices), len(SEMANTIC_EFFECTS)),
            dtype=np.float64,
        ),
        "roles": np.zeros(
            (len(indices), len(ROLE_LABELS)),
            dtype=np.float64,
        ),
        "horizons": np.zeros(
            (len(indices), len(HORIZONS)),
            dtype=np.float64,
        ),
        "danger": np.zeros((len(indices), 1), dtype=np.float64),
        "distance": np.zeros((len(indices), 1), dtype=np.float64),
        "latent": np.zeros(
            (len(indices), model.temporal.hidden_size),
            dtype=np.float64,
        ),
    }
    local = {int(index): offset for offset, index in enumerate(indices)}
    model.eval()
    with torch.inference_mode():
        for sequence in tensors.sequences:
            selected = tuple(index for index in sequence if index in allowed)
            if not selected:
                continue
            hidden = None
            for start in range(0, len(selected), MAX_HISTORY):
                chunk = selected[start : start + MAX_HISTORY]
                batch = _batch(tensors, chunk, device=device)
                values = model(*batch[:4], hidden)
                hidden = values[-1]
                rows = [local[index] for index in chunk]
                output["effects"][rows] = torch.sigmoid(values[0]).cpu().numpy()
                output["roles"][rows] = torch.sigmoid(values[1]).cpu().numpy()
                output["horizons"][rows] = torch.sigmoid(values[2]).cpu().numpy()
                output["danger"][rows] = torch.sigmoid(values[3]).cpu().numpy()
                output["distance"][rows] = values[4].cpu().numpy()
                output["latent"][rows] = values[6].cpu().numpy()
    return output


def _action_only_probabilities(
    records: Sequence[TemporalTeacherRecord],
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    output = np.zeros(
        (len(test_indices), len(SEMANTIC_EFFECTS)),
        dtype=np.float64,
    )
    for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
        global_rows = [
            records[index]
            for index in train_indices
            if records[index].applicable[effect]
        ]
        global_probability = (
            (sum(row.labels[effect] for row in global_rows) + 1.0)
            / (len(global_rows) + 2.0)
            if global_rows
            else 0.5
        )
        by_action: dict[str, list[TemporalTeacherRecord]] = defaultdict(list)
        for index in train_indices:
            record = records[int(index)]
            if record.applicable[effect]:
                by_action[str(record.graph.root["action_name"])].append(record)
        for local, index in enumerate(test_indices):
            action = str(records[int(index)].graph.root["action_name"])
            selected = by_action.get(action, [])
            output[local, effect_index] = (
                (sum(row.labels[effect] for row in selected) + 1.0)
                / (len(selected) + 2.0)
                if selected
                else global_probability
            )
    return output


def _binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    masks: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    per_label = {}
    briers = []
    balanced = []
    for index, name in enumerate(names):
        selected = masks[:, index] > 0.5
        truth = targets[selected, index]
        predicted = probabilities[selected, index]
        binary = predicted >= 0.5
        positives = truth == 1.0
        negatives = truth == 0.0
        positive_recall = float(np.mean(binary[positives])) if positives.any() else None
        negative_recall = (
            float(np.mean(~binary[negatives])) if negatives.any() else None
        )
        label_balanced = (
            0.5 * (positive_recall + negative_recall)
            if positive_recall is not None and negative_recall is not None
            else None
        )
        brier = float(np.mean((predicted - truth) ** 2)) if len(truth) else 0.0
        briers.append(brier)
        if label_balanced is not None:
            balanced.append(label_balanced)
        per_label[str(name)] = {
            "rows": int(len(truth)),
            "positives": int(np.sum(truth)),
            "prevalence": float(np.mean(truth)) if len(truth) else 0.0,
            "brier": brier,
            "positive_recall_at_0_5": positive_recall,
            "negative_recall_at_0_5": negative_recall,
            "balanced_accuracy_at_0_5": label_balanced,
        }
    return {
        "macro_brier": float(np.mean(briers)) if briers else 0.0,
        "macro_balanced_accuracy_at_0_5": (
            float(np.mean(balanced)) if balanced else 0.0
        ),
        "per_label": per_label,
    }


def _identity_probe(
    records: Sequence[TemporalTeacherRecord],
    latent: np.ndarray,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    labels = np.asarray([record.game_id for record in records])
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = LogisticRegression(
        max_iter=800,
        solver="lbfgs",
        random_state=SEED,
    )
    accuracy = float(np.mean(cross_val_score(model, latent, labels, cv=folds)))
    return {
        "accuracy": accuracy,
        "majority_accuracy": float(majority),
        "gain_over_majority": accuracy - majority,
    }


def _save_checkpoint(
    path: Path,
    model: Any,
    *,
    parameters: Mapping[str, Any],
    games: Sequence[str],
    manifest_checksum: str,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "parameters": dict(parameters),
            "games": list(games),
            "manifest_checksum": manifest_checksum,
            "state_dict": model.state_dict(),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    *,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported V4.14 checkpoint")
    parameters = payload["parameters"]
    model = _torch_model(
        hash_buckets=int(parameters["hash_buckets"]),
        embedding_width=int(parameters["embedding_width"]),
        graph_hidden_width=int(parameters["graph_hidden_width"]),
        temporal_hidden_width=int(parameters["temporal_hidden_width"]),
        identity_classes=len(payload["games"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def train_temporal_student(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Run outer human-game LOGO and fit the final six-game checkpoint."""

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa.get("teacher_ready"):
        raise RuntimeError("V4.14 temporal teacher is not ready")
    records = load_teacher_records(destination)
    parameters = dict(manifest["student"])
    tensors = tensorize_temporal_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=int(parameters["maximum_neighbors"]),
    )
    relation_tensors = tensorize_temporal_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=int(parameters["maximum_neighbors"]),
        relation_shuffle=True,
    )
    history_tensors = tensorize_temporal_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=int(parameters["maximum_neighbors"]),
        history_shuffle=True,
        shuffle_seed=SEED,
    )
    benchmark = _benchmark_device(parameters)
    if requested_device != "auto":
        selected_device = requested_device
    else:
        selected_device = str(benchmark["selected_device"])

    shape = (len(records), len(SEMANTIC_EFFECTS))
    predictions = np.zeros(shape, dtype=np.float64)
    relation_predictions = np.zeros(shape, dtype=np.float64)
    history_predictions = np.zeros(shape, dtype=np.float64)
    action_predictions = np.zeros(shape, dtype=np.float64)
    role_predictions = np.zeros(
        (len(records), len(ROLE_LABELS)),
        dtype=np.float64,
    )
    horizon_predictions = np.zeros(
        (len(records), len(HORIZONS)),
        dtype=np.float64,
    )
    danger_predictions = np.zeros((len(records), 1), dtype=np.float64)
    distance_predictions = np.zeros((len(records), 1), dtype=np.float64)
    latent = np.zeros(
        (len(records), int(parameters["temporal_hidden_width"])),
        dtype=np.float64,
    )
    folds = []
    for fold_index, held_out_game in enumerate(HUMAN_TRAIN_GAMES):
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
        model, summary = _fit_model(
            records,
            tensors,
            train_indices=train_indices,
            parameters=parameters,
            device=selected_device,
            seed=SEED + fold_index * 100,
        )
        full = _predict_model(
            model,
            tensors,
            indices=test_indices,
            device=selected_device,
        )
        relation = _predict_model(
            model,
            relation_tensors,
            indices=test_indices,
            device=selected_device,
        )
        history = _predict_model(
            model,
            history_tensors,
            indices=test_indices,
            device=selected_device,
        )
        predictions[test_indices] = full["effects"]
        relation_predictions[test_indices] = relation["effects"]
        history_predictions[test_indices] = history["effects"]
        role_predictions[test_indices] = full["roles"]
        horizon_predictions[test_indices] = full["horizons"]
        danger_predictions[test_indices] = full["danger"]
        distance_predictions[test_indices] = full["distance"]
        latent[test_indices] = full["latent"]
        action_predictions[test_indices] = _action_only_probabilities(
            records,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        folds.append(
            {
                "held_out_game": held_out_game,
                "training_rows": len(train_indices),
                "validation_rows": len(test_indices),
                **summary,
            }
        )
        del model
        try:
            import torch

            if selected_device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass

    prediction_rows = []
    for index, record in enumerate(records):
        prediction_rows.append(
            {
                "format_version": PREDICTION_VERSION,
                "example_id": record.example_id,
                "game_id": record.game_id,
                "episode_id": record.episode_id,
                "step": record.step,
                "probabilities": {
                    "temporal": {
                        effect: float(predictions[index, effect_index])
                        for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                    },
                    "relation_shuffle": {
                        effect: float(relation_predictions[index, effect_index])
                        for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                    },
                    "history_shuffle": {
                        effect: float(history_predictions[index, effect_index])
                        for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                    },
                    "action_only": {
                        effect: float(action_predictions[index, effect_index])
                        for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                    },
                    "roles": {
                        role: float(role_predictions[index, role_index])
                        for role_index, role in enumerate(ROLE_LABELS)
                    },
                    "progress": {
                        f"within_{horizon}": float(
                            horizon_predictions[index, horizon_index]
                        )
                        for horizon_index, horizon in enumerate(HORIZONS)
                    },
                    "danger_within_8": float(danger_predictions[index, 0]),
                    "normalized_distance": float(distance_predictions[index, 0]),
                },
            }
        )
    predictions_path = destination / "logo_predictions.jsonl"
    folds_path = destination / "folds.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(folds_path, folds)

    effect_targets = tensors.labels
    effect_masks = tensors.applicable
    role_masks = np.ones_like(tensors.roles)
    horizon_targets = np.concatenate((tensors.horizons, tensors.danger), axis=1)
    horizon_output = np.concatenate(
        (horizon_predictions, danger_predictions),
        axis=1,
    )
    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "records": len(records),
        "games": list(HUMAN_TRAIN_GAMES),
        "device_benchmark": benchmark,
        "selected_device": selected_device,
        "metrics": {
            "temporal_effects": _binary_metrics(
                effect_targets,
                predictions,
                effect_masks,
                SEMANTIC_EFFECTS,
            ),
            "relation_shuffle_effects": _binary_metrics(
                effect_targets,
                relation_predictions,
                effect_masks,
                SEMANTIC_EFFECTS,
            ),
            "history_shuffle_effects": _binary_metrics(
                effect_targets,
                history_predictions,
                effect_masks,
                SEMANTIC_EFFECTS,
            ),
            "action_only_effects": _binary_metrics(
                effect_targets,
                action_predictions,
                effect_masks,
                SEMANTIC_EFFECTS,
            ),
            "roles": _binary_metrics(
                tensors.roles,
                role_predictions,
                role_masks,
                ROLE_LABELS,
            ),
            "temporal_targets": _binary_metrics(
                horizon_targets,
                horizon_output,
                np.ones_like(horizon_targets),
                [
                    *(f"within_{value}" for value in HORIZONS),
                    "danger_within_8",
                ],
            ),
            "uncensored_distance_mae": float(
                np.sum(
                    np.abs(distance_predictions - tensors.distance) * tensors.uncensored
                )
                / np.maximum(np.sum(tensors.uncensored), 1.0)
            ),
        },
        "semantic_output_game_identity_probe": _identity_probe(
            records,
            latent,
        ),
        "folds": folds,
        "artifact_sha256": {
            "teacher_corpus": _file_sha256(destination / "teacher_corpus.jsonl"),
            "logo_predictions": _file_sha256(predictions_path),
            "folds": _file_sha256(folds_path),
        },
    }
    result["semantic_result_checksum"] = _checksum(result)
    _write_json(destination / "semantic_result.json", result)

    all_indices = np.arange(len(records), dtype=np.int64)
    final_model, final_summary = _fit_model(
        records,
        tensors,
        train_indices=all_indices,
        parameters=parameters,
        device=selected_device,
        seed=SEED + 900,
    )
    checkpoint_path = destination / "temporal_student.pt"
    _save_checkpoint(
        checkpoint_path,
        final_model,
        parameters=parameters,
        games=HUMAN_TRAIN_GAMES,
        manifest_checksum=manifest["manifest_checksum"],
    )
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(HUMAN_TRAIN_GAMES),
        "training": final_summary,
        "path": checkpoint_path.as_posix(),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _file_sha256(checkpoint_path),
        "selected_device": selected_device,
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


def _graph_for_action(
    graph: ObjectRelativeGraph,
    action_name: str,
) -> ObjectRelativeGraph:
    root = dict(graph.root)
    name = str(action_name).upper()
    directions = {
        "ACTION1": "up",
        "ACTION2": "down",
        "ACTION3": "left",
        "ACTION4": "right",
    }
    root["action_name"] = name
    root["action_family"] = (
        "move" if name in directions else "click" if name == "ACTION6" else "other"
    )
    root["requested_direction"] = directions.get(name, "none")
    return ObjectRelativeGraph(root=root, neighbors=graph.neighbors)


def _predict_graph_rollout(
    model: Any,
    graphs: Sequence[ObjectRelativeGraph],
    *,
    parameters: Mapping[str, Any],
    device: str,
    relation_shuffle: bool = False,
    initial_belief: TemporalBeliefState | None = None,
) -> list[TemporalSlotPrediction]:
    import torch

    selected_graphs = [
        graph.relation_shuffled() if relation_shuffle else graph for graph in graphs
    ]
    root_rows = [_tokens("root", dict(graph.root)) for graph in selected_graphs]
    neighbor_rows = [
        [
            _tokens("neighbor", dict(neighbor))
            for neighbor in graph.neighbors[: int(parameters["maximum_neighbors"])]
        ]
        for graph in selected_graphs
    ]
    root_width = max(max((len(row) for row in root_rows), default=1), 1)
    node_width = max(
        max(
            (len(tokens) for neighbors in neighbor_rows for tokens in neighbors),
            default=1,
        ),
        1,
    )
    roots = np.zeros((len(graphs), root_width), dtype=np.int64)
    nodes = np.zeros(
        (
            len(graphs),
            int(parameters["maximum_neighbors"]),
            node_width,
        ),
        dtype=np.int64,
    )
    masks = np.zeros(
        (len(graphs), int(parameters["maximum_neighbors"])),
        dtype=np.float32,
    )
    for index, tokens in enumerate(root_rows):
        roots[index, : len(tokens)] = [
            _token_id(token, int(parameters["hash_buckets"])) for token in tokens
        ]
    for index, neighbors in enumerate(neighbor_rows):
        for neighbor_index, tokens in enumerate(neighbors):
            nodes[index, neighbor_index, : len(tokens)] = [
                _token_id(token, int(parameters["hash_buckets"])) for token in tokens
            ]
            masks[index, neighbor_index] = 1.0
    if initial_belief is not None and initial_belief.hidden:
        hidden = torch.as_tensor(
            initial_belief.hidden,
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, -1)
    else:
        hidden = None
    feedback = torch.as_tensor(
        [
            float(initial_belief.effect_probabilities.get(effect, 0.0))
            if initial_belief is not None
            else 0.0
            for effect in SEMANTIC_EFFECTS
        ],
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    output = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(graphs)):
            values = model(
                torch.as_tensor(
                    roots[index : index + 1],
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    nodes[index : index + 1],
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    masks[index : index + 1],
                    dtype=torch.float32,
                    device=device,
                ),
                feedback,
                hidden,
            )
            effects = torch.sigmoid(values[0])[0].cpu().numpy()
            roles = torch.sigmoid(values[1])[0].cpu().numpy()
            progress = torch.sigmoid(values[2])[0].cpu().numpy()
            danger = float(torch.sigmoid(values[3])[0, 0].cpu())
            distance = float(values[4][0, 0].cpu())
            hidden = values[-1]
            feedback = torch.as_tensor(
                effects.reshape(1, -1),
                dtype=torch.float32,
                device=device,
            )
            belief = TemporalBeliefState(
                hidden=tuple(float(item) for item in hidden[:, -1, :].cpu().numpy()[0]),
                effect_probabilities={
                    effect: float(effects[effect_index])
                    for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                },
                role_probabilities={
                    role: float(roles[role_index])
                    for role_index, role in enumerate(ROLE_LABELS)
                },
                progress_probabilities={
                    f"within_{horizon}": float(progress[horizon_index])
                    for horizon_index, horizon in enumerate(HORIZONS)
                },
                risk_probability=danger,
                step_count=(
                    (initial_belief.step_count if initial_belief else 0) + index + 1
                ),
            )
            output.append(
                TemporalSlotPrediction(
                    effect_probabilities=belief.effect_probabilities,
                    role_probabilities=belief.role_probabilities,
                    progress_probabilities=belief.progress_probabilities,
                    danger_within_8=danger,
                    normalized_distance=distance,
                    next_belief=belief,
                )
            )
    return output


def _tensorize_graph_batch(
    graphs: Sequence[ObjectRelativeGraph],
    *,
    parameters: Mapping[str, Any],
    device: str,
) -> tuple[Any, Any, Any]:
    """Tensorize independent candidate graphs without joining their histories."""

    import torch

    root_rows = [_tokens("root", dict(graph.root)) for graph in graphs]
    neighbor_rows = [
        [
            _tokens("neighbor", dict(neighbor))
            for neighbor in graph.neighbors[: int(parameters["maximum_neighbors"])]
        ]
        for graph in graphs
    ]
    root_width = max(max((len(row) for row in root_rows), default=1), 1)
    node_width = max(
        max(
            (len(tokens) for neighbors in neighbor_rows for tokens in neighbors),
            default=1,
        ),
        1,
    )
    roots = np.zeros((len(graphs), root_width), dtype=np.int64)
    nodes = np.zeros(
        (
            len(graphs),
            int(parameters["maximum_neighbors"]),
            node_width,
        ),
        dtype=np.int64,
    )
    masks = np.zeros(
        (len(graphs), int(parameters["maximum_neighbors"])),
        dtype=np.float32,
    )
    for index, tokens in enumerate(root_rows):
        roots[index, : len(tokens)] = [
            _token_id(token, int(parameters["hash_buckets"])) for token in tokens
        ]
    for index, neighbors in enumerate(neighbor_rows):
        for neighbor_index, tokens in enumerate(neighbors):
            nodes[index, neighbor_index, : len(tokens)] = [
                _token_id(token, int(parameters["hash_buckets"])) for token in tokens
            ]
            masks[index, neighbor_index] = 1.0
    return (
        torch.as_tensor(roots, dtype=torch.long, device=device),
        torch.as_tensor(nodes, dtype=torch.long, device=device),
        torch.as_tensor(masks, dtype=torch.float32, device=device),
    )


def _blend_effect_probabilities(
    temporal: Mapping[str, float],
    qwen: Mapping[str, float] | None,
    *,
    weight: float = ACTIVE_QWEN_BLEND_WEIGHT,
) -> dict[str, float]:
    output = {effect: float(temporal.get(effect, 0.5)) for effect in SEMANTIC_EFFECTS}
    if qwen is None:
        return output
    alpha = min(max(float(weight), 0.0), 1.0)
    for effect in SLOT_EFFECTS:
        if effect in qwen:
            output[effect] = (1.0 - alpha) * output[effect] + alpha * float(
                qwen[effect]
            )
    return output


def _predict_candidate_rollouts(
    model: Any,
    plans: Sequence[Sequence[ObjectRelativeGraph]],
    *,
    parameters: Mapping[str, Any],
    device: str,
    initial_belief: TemporalBeliefState | None = None,
    qwen_priors: Sequence[Mapping[str, float] | None] | None = None,
) -> list[list[TemporalSlotPrediction]]:
    """Score independent depth-three candidates in one GPU batch per depth."""

    import torch

    if not plans:
        return []
    depths = {len(plan) for plan in plans}
    if len(depths) != 1 or not depths or next(iter(depths)) < 1:
        raise ValueError("candidate plans must share one positive depth")
    if qwen_priors is not None and len(qwen_priors) != len(plans):
        raise ValueError("Qwen priors must align with candidate plans")
    count = len(plans)
    if initial_belief is not None and initial_belief.hidden:
        hidden = (
            torch.as_tensor(
                initial_belief.hidden,
                dtype=torch.float32,
                device=device,
            )
            .reshape(1, 1, -1)
            .repeat(1, count, 1)
        )
    else:
        hidden = None
    feedback = torch.as_tensor(
        [
            [
                float(initial_belief.effect_probabilities.get(effect, 0.0))
                if initial_belief is not None
                else 0.0
                for effect in SEMANTIC_EFFECTS
            ]
            for _ in plans
        ],
        dtype=torch.float32,
        device=device,
    )
    output: list[list[TemporalSlotPrediction]] = [[] for _ in plans]
    model.eval()
    with torch.inference_mode():
        for depth in range(next(iter(depths))):
            graphs = [plan[depth] for plan in plans]
            roots, nodes, masks = _tensorize_graph_batch(
                graphs,
                parameters=parameters,
                device=device,
            )
            graph_latent = model.encode_graph(roots, nodes, masks)
            temporal_input = torch.cat(
                (graph_latent, feedback),
                dim=-1,
            ).unsqueeze(1)
            temporal, hidden = model.temporal(temporal_input, hidden)
            latent = temporal[:, 0, :]
            effect_values = torch.sigmoid(model.effect_head(latent))
            role_values = torch.sigmoid(model.role_head(latent))
            horizon_values = model.horizon_head(latent)
            progress_values = torch.sigmoid(horizon_values[:, : len(HORIZONS)])
            danger_values = torch.sigmoid(horizon_values[:, len(HORIZONS)])
            distance_values = model.distance_head(latent)[:, 0]
            blended_rows = []
            for index in range(count):
                temporal_effects = {
                    effect: float(effect_values[index, effect_index].cpu())
                    for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                }
                prior = (
                    qwen_priors[index]
                    if depth == 0 and qwen_priors is not None
                    else None
                )
                effects = _blend_effect_probabilities(
                    temporal_effects,
                    prior,
                )
                blended_rows.append([effects[effect] for effect in SEMANTIC_EFFECTS])
                roles = {
                    role: float(role_values[index, role_index].cpu())
                    for role_index, role in enumerate(ROLE_LABELS)
                }
                progress = {
                    f"within_{horizon}": float(
                        progress_values[index, horizon_index].cpu()
                    )
                    for horizon_index, horizon in enumerate(HORIZONS)
                }
                danger = float(danger_values[index].cpu())
                belief = TemporalBeliefState(
                    hidden=tuple(
                        float(item) for item in hidden[:, index, :].cpu().numpy()[0]
                    ),
                    effect_probabilities=effects,
                    role_probabilities=roles,
                    progress_probabilities=progress,
                    risk_probability=danger,
                    step_count=(
                        (initial_belief.step_count if initial_belief else 0) + depth + 1
                    ),
                )
                output[index].append(
                    TemporalSlotPrediction(
                        effect_probabilities=effects,
                        role_probabilities=roles,
                        progress_probabilities=progress,
                        danger_within_8=danger,
                        normalized_distance=float(distance_values[index].cpu()),
                        next_belief=belief,
                    )
                )
            feedback = torch.as_tensor(
                blended_rows,
                dtype=torch.float32,
                device=device,
            )
    return output


def _prediction_features(
    steps: Sequence[TemporalSlotPrediction],
) -> tuple[float, ...]:
    if not steps:
        return (0.0,) * 8
    returns = []
    successes = []
    failures = []
    productive = []
    entropies = []
    contradictions = []
    for offset, step in enumerate(steps):
        effects = step.effect_probabilities
        value = (
            4.0 * float(effects["level_complete"])
            + float(effects["target_created"])
            + float(effects["target_removed"])
            + float(effects["target_moved"])
            + float(effects["path_opened"])
            + 0.5 * float(effects["actor_approached_root"])
            + 0.25 * float(effects["moved"])
            - 4.0 * float(effects["game_over"])
            - 0.5 * float(effects["path_closed"])
        )
        returns.append((0.8**offset) * value)
        successes.append(float(effects["level_complete"]))
        failures.append(max(float(effects["game_over"]), step.danger_within_8))
        productive.append(float(effects["productive"]))
        probabilities = np.asarray(
            list(effects.values()),
            dtype=np.float64,
        )
        entropy = -(
            probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))
            + (1.0 - probabilities) * np.log(np.clip(1.0 - probabilities, 1e-8, 1.0))
        )
        entropies.append(float(np.mean(entropy)))
        contradictions.append(
            min(
                float(effects["path_opened"]),
                float(effects["path_closed"]),
            )
        )
    return (
        float(sum(returns)),
        float(max(successes)),
        float(max(failures)),
        float(sum(productive) / len(productive)),
        float(np.mean(entropies)),
        float(np.std(returns)),
        float(sum(contradictions)),
        float(len(steps) / 3.0),
    )


def _teacher_features(
    arm: TeacherArm,
) -> tuple[float, ...]:
    entropy = 0.0
    contradiction = float(arm.labels["path_opened"] and arm.labels["path_closed"])
    return (
        float(arm.horizon_return),
        float(arm.labels["level_complete"]),
        float(arm.labels["game_over"] or arm.labels["risk"]),
        float(arm.labels["productive"]),
        entropy,
        float(arm.horizon_uncertainty),
        contradiction,
        1.0,
    )


def _human_ebm_pairs(
    records: Sequence[TemporalTeacherRecord],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    by_example = {str(row["example_id"]): row for row in prediction_rows}
    features = {}
    for record in records:
        row = by_example[record.example_id]["probabilities"]
        step = TemporalSlotPrediction(
            effect_probabilities=dict(row["temporal"]),
            role_probabilities=dict(row["roles"]),
            progress_probabilities=dict(row["progress"]),
            danger_within_8=float(row["danger_within_8"]),
            normalized_distance=float(row["normalized_distance"]),
            next_belief=TemporalBeliefState(),
        )
        features[record.example_id] = _prediction_features((step,))
    by_state: dict[tuple[str, str], list[TemporalTeacherRecord]] = defaultdict(list)
    for record in records:
        by_state[(record.game_id, record.pre_state_sha256)].append(record)
    preferred = []
    rejected = []
    for values in by_state.values():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                left_value = (
                    left.discounted_progress
                    + 0.1 * left.productive_score
                    - float(left.danger_within_8)
                )
                right_value = (
                    right.discounted_progress
                    + 0.1 * right.productive_score
                    - float(right.danger_within_8)
                )
                if abs(left_value - right_value) <= 1e-9:
                    continue
                winner, loser = (
                    (left, right) if left_value > right_value else (right, left)
                )
                preferred.append(features[winner.example_id])
                rejected.append(features[loser.example_id])
    if len(preferred) > 4096:
        indices = np.random.default_rng(SEED).choice(
            len(preferred),
            size=4096,
            replace=False,
        )
        preferred = [preferred[int(index)] for index in indices]
        rejected = [rejected[int(index)] for index in indices]
    return preferred, rejected


def _action_sequence_tables(
    records: Sequence[TemporalTeacherRecord],
) -> tuple[dict[str, float], dict[tuple[str, ...], float], float]:
    sequences = _record_sequences(records)
    action_values: dict[str, list[float]] = defaultdict(list)
    sequence_values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    all_values = []
    for sequence in sequences:
        for offset, index in enumerate(sequence):
            record = records[index]
            action = str(record.graph.root["action_name"])
            value = (
                record.discounted_progress
                + 0.1 * record.productive_score
                - float(record.danger_within_8)
            )
            action_values[action].append(value)
            key = tuple(
                str(records[row].graph.root["action_name"])
                for row in sequence[offset : offset + 3]
            )
            sequence_values[key].append(value)
            all_values.append(value)
    return (
        {action: float(np.mean(values)) for action, values in action_values.items()},
        {key: float(np.mean(values)) for key, values in sequence_values.items()},
        float(np.mean(all_values)) if all_values else 0.0,
    )


def _paired_bootstrap_rows(
    learned: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    baseline_by_panel = {
        str(row["panel_id"]): float(row["utility"]) for row in baseline
    }
    differences = np.asarray(
        [
            float(row["utility"]) - baseline_by_panel[str(row["panel_id"])]
            for row in learned
        ],
        dtype=np.float64,
    )
    if not len(differences):
        return {"mean_gain": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            float(
                np.mean(
                    differences[rng.integers(0, len(differences), len(differences))]
                )
            )
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    return {
        "mean_gain": float(np.mean(differences)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _mean_features(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows:
        return (0.0,) * 8
    return tuple(
        float(value) for value in np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    )


def _template_score(arm: TeacherArm) -> tuple[float, str]:
    root = arm.graph.root
    score = (
        1.0 * float(root.get("root_occupied", 0))
        + 0.5 * float(root.get("path_status") == "open")
        + 0.25 * float(root.get("action_family") == "move")
        + 0.1 * float(root.get("action_family") == "click")
    )
    tie = json.dumps(
        {
            "action": arm.action_name,
            "data": dict(arm.action_data),
        },
        sort_keys=True,
    )
    return score, tie


def _v412_prediction_lookup(
    path: Path,
) -> dict[str, Mapping[str, float]]:
    output = {}
    if not path.exists():
        return output
    for row in _read_jsonl(path):
        output[str(row["trace_digest"])] = dict(
            row["probabilities"]["descriptive_distilled"]
        )
    return output


def _v412_features(probabilities: Mapping[str, float]) -> tuple[float, ...]:
    step = TemporalSlotPrediction(
        effect_probabilities={
            effect: float(probabilities.get(effect, 0.5)) for effect in SEMANTIC_EFFECTS
        },
        role_probabilities={role: 0.5 for role in ROLE_LABELS},
        progress_probabilities={f"within_{horizon}": 0.5 for horizon in HORIZONS},
        danger_within_8=float(probabilities.get("risk", 0.5)),
        normalized_distance=0.5,
        next_belief=TemporalBeliefState(),
    )
    return _prediction_features((step,))


def _summarize_decisions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_game = {}
    for game in sorted({str(row["game_id"]) for row in rows}):
        selected = [row for row in rows if row["game_id"] == game]
        per_game[game] = {
            "panels": len(selected),
            "mean_utility": float(np.mean([float(row["utility"]) for row in selected])),
            "mean_regret": float(np.mean([float(row["regret"]) for row in selected])),
            "oracle_action_accuracy": float(
                np.mean([bool(row["oracle_action"]) for row in selected])
            ),
        }
    return {
        "panels": len(rows),
        "mean_utility": float(np.mean([float(row["utility"]) for row in rows]))
        if rows
        else 0.0,
        "mean_regret": float(np.mean([float(row["regret"]) for row in rows]))
        if rows
        else 0.0,
        "oracle_action_accuracy": float(
            np.mean([bool(row["oracle_action"]) for row in rows])
        )
        if rows
        else 0.0,
        "per_game": per_game,
    }


def evaluate_transfer_and_global_chain(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v412_predictions: str | Path = (
        Path("training")
        / "sage12"
        / "descriptive_semantic_integration_v4_12"
        / "logo_predictions.jsonl"
    ),
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Evaluate semantics and the complete ranker on untraced games.

    V4.11 supplies executed outcomes for scoring only.  The V4.14 rollout
    consumes the root graph and proposed action names; it never reads the
    continuation frames or future graphs.
    """

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    semantic = _read_json(destination / "semantic_result.json")
    checkpoint_metadata = _read_json(destination / "checkpoint_metadata.json")
    device = (
        str(semantic["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    model, checkpoint = _load_checkpoint(
        destination / "temporal_student.pt",
        device=device,
    )
    if checkpoint["manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("V4.14 checkpoint/manifest mismatch")
    parameters = checkpoint["parameters"]
    human_records = load_teacher_records(destination)
    logo_rows = _read_jsonl(destination / "logo_predictions.jsonl")
    preferred, rejected = _human_ebm_pairs(human_records, logo_rows)
    if not preferred:
        raise RuntimeError("V4.14 has no human same-prestate EBM pairs")
    ebm = PairwiseTrajectoryEBM(
        input_width=8,
        hidden_width=32,
        seed=SEED,
    )
    ebm_losses = ebm.fit_pairs(
        preferred,
        rejected,
        epochs=150,
        learning_rate=0.003,
    )
    import torch

    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "input_width": 8,
            "hidden_width": 32,
            "trained_pairs": ebm.trained_pairs,
            "state_dict": ebm.model.state_dict(),
        },
        destination / "trajectory_ebm.pt",
    )

    teacher_panels = tuple(
        panel
        for panel in load_teacher_panels(v411_dir)
        if panel.game_id in TRANSFER_GAMES
    )
    raw_panels = {
        panel.panel_id: panel
        for panel in load_raw_panels(v411_dir)
        if panel.game_id in TRANSFER_GAMES
    }
    if {panel.game_id for panel in teacher_panels} != set(TRANSFER_GAMES):
        raise ValueError("V4.14 transfer panel coverage is incomplete")
    action_table, sequence_table, global_value = _action_sequence_tables(human_records)
    v412_lookup = _v412_prediction_lookup(Path(v412_predictions))

    transfer_truth = []
    transfer_temporal = []
    transfer_relation = []
    transfer_v412 = []
    transfer_action = []
    transfer_masks = []
    decisions = []
    panel_prediction_rows = []
    for panel in teacher_panels:
        raw_panel = raw_panels.get(panel.panel_id)
        if raw_panel is None:
            raise ValueError(f"missing raw V4.11 panel {panel.panel_id}")
        raw_by_arm = {arm.arm_index: arm for arm in raw_panel.arms}
        arm_rows = []
        for arm in panel.arms:
            raw_arm = raw_by_arm[arm.arm_index]
            rollout_features = []
            shuffled_features = []
            sequence_values = []
            rollout_predictions = []
            for continuation in raw_arm.continuations:
                action_names = [
                    arm.action_name,
                    *(trace.selected_action_name for trace in continuation),
                ]
                graphs = [
                    arm.graph,
                    *(
                        _graph_for_action(arm.graph, action_name)
                        for action_name in action_names[1:]
                    ),
                ]
                predicted = _predict_graph_rollout(
                    model,
                    graphs,
                    parameters=parameters,
                    device=device,
                )
                shuffled = _predict_graph_rollout(
                    model,
                    graphs,
                    parameters=parameters,
                    device=device,
                    relation_shuffle=True,
                )
                rollout_features.append(_prediction_features(predicted))
                shuffled_features.append(_prediction_features(shuffled))
                rollout_predictions.append(predicted)
                sequence_values.append(
                    sequence_table.get(
                        tuple(action_names),
                        action_table.get(arm.action_name, global_value),
                    )
                )
            temporal_features = _mean_features(rollout_features)
            relation_features = _mean_features(shuffled_features)
            temporal_energy = ebm.energies((temporal_features,))[0]
            relation_energy = ebm.energies((relation_features,))[0]
            teacher_features = _teacher_features(arm)
            true_world_energy = ebm.energies((teacher_features,))[0]
            v412_probabilities = v412_lookup.get(
                arm.trace_digest,
                {effect: 0.5 for effect in SEMANTIC_EFFECTS},
            )
            v412_features = _v412_features(v412_probabilities)
            v412_energy = ebm.energies((v412_features,))[0]
            first_prediction = rollout_predictions[0][0]
            transfer_truth.append(
                [float(arm.labels[effect]) for effect in SEMANTIC_EFFECTS]
            )
            transfer_masks.append(
                [float(arm.applicable[effect]) for effect in SEMANTIC_EFFECTS]
            )
            transfer_temporal.append(
                [
                    float(first_prediction.effect_probabilities[effect])
                    for effect in SEMANTIC_EFFECTS
                ]
            )
            relation_first = _predict_graph_rollout(
                model,
                (arm.graph,),
                parameters=parameters,
                device=device,
                relation_shuffle=True,
            )[0]
            transfer_relation.append(
                [
                    float(relation_first.effect_probabilities[effect])
                    for effect in SEMANTIC_EFFECTS
                ]
            )
            transfer_v412.append(
                [
                    float(v412_probabilities.get(effect, 0.5))
                    for effect in SEMANTIC_EFFECTS
                ]
            )
            transfer_action.append(
                [
                    float(
                        np.mean(
                            [
                                record.labels[effect]
                                for record in human_records
                                if record.graph.root["action_name"] == arm.action_name
                                and record.applicable[effect]
                            ]
                            or [0.5]
                        )
                    )
                    for effect in SEMANTIC_EFFECTS
                ]
            )
            arm_rows.append(
                {
                    "arm_index": arm.arm_index,
                    "action_name": arm.action_name,
                    "utility": float(arm.horizon_return),
                    "completion": bool(arm.labels["level_complete"]),
                    "temporal_features": temporal_features,
                    "relation_features": relation_features,
                    "teacher_features": teacher_features,
                    "temporal_energy": float(temporal_energy),
                    "relation_energy": float(relation_energy),
                    "true_world_energy": float(true_world_energy),
                    "v412_energy": float(v412_energy),
                    "action_only_score": float(
                        action_table.get(arm.action_name, global_value)
                    ),
                    "action_sequence_score": float(np.mean(sequence_values)),
                    "template_score": float(_template_score(arm)[0]),
                    "template_tie": _template_score(arm)[1],
                }
            )
        oracle = max(
            arm_rows,
            key=lambda row: (
                float(row["utility"]),
                -int(row["arm_index"]),
            ),
        )
        methods = {
            "deterministic_left": min(
                arm_rows,
                key=lambda row: int(row["arm_index"]),
            ),
            "action_only": max(
                arm_rows,
                key=lambda row: (
                    float(row["action_only_score"]),
                    -int(row["arm_index"]),
                ),
            ),
            "action_sequence_only": max(
                arm_rows,
                key=lambda row: (
                    float(row["action_sequence_score"]),
                    -int(row["arm_index"]),
                ),
            ),
            "deterministic_template": max(
                arm_rows,
                key=lambda row: (
                    float(row["template_score"]),
                    str(row["template_tie"]),
                ),
            ),
            "learned_v4_12_snapshot_ebm": min(
                arm_rows,
                key=lambda row: (
                    float(row["v412_energy"]),
                    int(row["arm_index"]),
                ),
            ),
            "temporal_rollout_ebm": min(
                arm_rows,
                key=lambda row: (
                    float(row["temporal_energy"]),
                    int(row["arm_index"]),
                ),
            ),
            "temporal_relation_shuffle_ebm": min(
                arm_rows,
                key=lambda row: (
                    float(row["relation_energy"]),
                    int(row["arm_index"]),
                ),
            ),
            "true_world_learned_ebm": min(
                arm_rows,
                key=lambda row: (
                    float(row["true_world_energy"]),
                    int(row["arm_index"]),
                ),
            ),
            "oracle_energy": oracle,
        }
        for method, selected in methods.items():
            decisions.append(
                {
                    "format_version": FORMAT_VERSION,
                    "panel_id": panel.panel_id,
                    "game_id": panel.game_id,
                    "method": method,
                    "selected_arm": int(selected["arm_index"]),
                    "oracle_arm": int(oracle["arm_index"]),
                    "utility": float(selected["utility"]),
                    "oracle_utility": float(oracle["utility"]),
                    "regret": float(oracle["utility"]) - float(selected["utility"]),
                    "oracle_action": (
                        int(selected["arm_index"]) == int(oracle["arm_index"])
                    ),
                    "completion_selected": bool(selected["completion"]),
                    "completion_available": any(
                        bool(row["completion"]) for row in arm_rows
                    ),
                }
            )
        panel_prediction_rows.append(
            {
                "format_version": PREDICTION_VERSION,
                "panel_id": panel.panel_id,
                "game_id": panel.game_id,
                "arms": arm_rows,
            }
        )

    decisions_path = destination / "transfer_decisions.jsonl"
    panel_path = destination / "transfer_predictions.jsonl"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(panel_path, panel_prediction_rows)
    methods = sorted({str(row["method"]) for row in decisions})
    by_method = {
        method: [row for row in decisions if row["method"] == method]
        for method in methods
    }
    metrics = {method: _summarize_decisions(rows) for method, rows in by_method.items()}
    primary = by_method["action_sequence_only"]
    comparisons = {
        f"{method}_over_action_sequence": _paired_bootstrap_rows(
            rows,
            primary,
            samples=int(manifest["evaluation"]["bootstrap_samples"]),
            seed=SEED + index,
        )
        for index, (method, rows) in enumerate(sorted(by_method.items()))
        if method != "action_sequence_only"
    }
    completion_opportunities = sum(
        bool(row["completion_available"]) for row in by_method["oracle_energy"]
    )
    completion_by_method = {
        method: sum(bool(row["completion_selected"]) for row in rows)
        for method, rows in by_method.items()
    }
    primary_per_game = metrics["action_sequence_only"]["per_game"]
    nonnegative_games = {
        method: sum(
            metrics[method]["per_game"][game]["mean_utility"]
            >= primary_per_game[game]["mean_utility"]
            for game in TRANSFER_GAMES
        )
        for method in methods
    }
    learned_method = "temporal_rollout_ebm"
    learned_comparison = comparisons[f"{learned_method}_over_action_sequence"]
    oracle_completion = completion_by_method["oracle_energy"]
    required_fraction = float(
        manifest["evaluation"]["completion_capture_fraction_of_oracle_minimum"]
    )
    required_absolute = int(
        manifest["evaluation"]["completion_capture_absolute_minimum"]
    )
    required_completion = max(
        required_absolute,
        math.ceil(required_fraction * oracle_completion),
    )
    checks = {
        "paired_bootstrap_ci_lower_positive": (learned_comparison["ci_low"] > 0.0),
        "nonnegative_transfer_games": (
            nonnegative_games[learned_method]
            >= int(manifest["evaluation"]["nonnegative_transfer_games_minimum"])
        ),
        "completion_absolute_and_fraction": (
            completion_by_method[learned_method] >= required_completion
        ),
        "all_conditions_executed": False,
        "future_descriptors_absent_from_rollout": True,
    }
    global_supported = all(checks.values())
    true_world_supported = (
        comparisons["true_world_learned_ebm_over_action_sequence"]["mean_gain"] > 0.0
    )
    semantic_targets = np.asarray(transfer_truth, dtype=np.float64)
    semantic_masks = np.asarray(transfer_masks, dtype=np.float64)
    semantic_metrics = {
        "temporal": _binary_metrics(
            semantic_targets,
            np.asarray(transfer_temporal, dtype=np.float64),
            semantic_masks,
            SEMANTIC_EFFECTS,
        ),
        "relation_shuffle": _binary_metrics(
            semantic_targets,
            np.asarray(transfer_relation, dtype=np.float64),
            semantic_masks,
            SEMANTIC_EFFECTS,
        ),
        "v4_12_snapshot": _binary_metrics(
            semantic_targets,
            np.asarray(transfer_v412, dtype=np.float64),
            semantic_masks,
            SEMANTIC_EFFECTS,
        ),
        "action_only": _binary_metrics(
            semantic_targets,
            np.asarray(transfer_action, dtype=np.float64),
            semantic_masks,
            SEMANTIC_EFFECTS,
        ),
    }
    if global_supported:
        verdict = "GLOBAL_CHAIN_SUPPORTED"
    elif not true_world_supported:
        verdict = "EBM_CONTROLLER_TRANSFER_BOTTLENECK"
    elif (
        semantic_metrics["temporal"]["macro_balanced_accuracy_at_0_5"]
        <= semantic_metrics["action_only"]["macro_balanced_accuracy_at_0_5"]
    ):
        verdict = "TEMPORAL_SEMANTIC_PREDICTOR_BOTTLENECK"
    else:
        verdict = "DEPLOYABLE_ROLLOUT_OR_ENERGY_BOTTLENECK"
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "semantic_result_checksum": semantic["semantic_result_checksum"],
        "verdict": verdict,
        "global_chain_supported": global_supported,
        "all_conditions_executed": False,
        "checks": checks,
        "transfer_games": list(TRANSFER_GAMES),
        "panels": len(teacher_panels),
        "arms": len(transfer_truth),
        "metrics": metrics,
        "comparisons": comparisons,
        "semantic_transfer": semantic_metrics,
        "completion_capture": {
            "opportunities": completion_opportunities,
            "oracle_selected": oracle_completion,
            "required_for_learned": required_completion,
            "selected_by_method": completion_by_method,
        },
        "nonnegative_games": nonnegative_games,
        "ebm": {
            "human_pairs": len(preferred),
            "epochs": 150,
            "initial_loss": float(ebm_losses[0]),
            "final_loss": float(ebm_losses[-1]),
            "checkpoint_sha256": _file_sha256(destination / "trajectory_ebm.pt"),
        },
        "qwen_contract": {
            "candidate_complete_compiler_preserved": True,
            "frozen_v4_7_decoding_unchanged": True,
            "new_panel_generations_run": False,
            "reason": (
                "V4.11 panels contain action sequences but not replay-prefix "
                "scene histories; the bounded active validation is the "
                "registered Qwen/controller execution surface."
            ),
        },
        "topology": {
            "true_future_descriptors_used_by_model": False,
            "continuation_action_names_used_as_proposed_plan": True,
            "continuation_frames_used_for_scoring_only": True,
            "candidate_sequence_source_is_recorded_panel": True,
            "live_win_rate_claimed": False,
        },
        "active_validation": {
            "status": "PENDING_BOUNDED_RUN",
            "games": list(ACTIVE_VALIDATION_GAMES),
            "seeds": list(manifest["evaluation"]["active_validation_seeds"]),
        },
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "transfer_decisions": _file_sha256(decisions_path),
            "transfer_predictions": _file_sha256(panel_path),
            "temporal_student": checkpoint_metadata["sha256"],
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def _live_action_signature(action: Any) -> str:
    return json.dumps(
        {
            "name": str(getattr(action, "name", "")).upper(),
            "action_args": dict(getattr(action, "action_args", {}) or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _live_candidate_graph(
    *,
    game_id: str,
    policy_seed: int,
    reset_index: int,
    step_index: int,
    frame: Any,
    legal: Sequence[Any],
    action: Any,
) -> ObjectRelativeGraph:
    from theory.real_env_option_adapter import snapshot_frame

    snapshot = snapshot_frame(frame)
    available = tuple(
        sorted(
            {
                str(getattr(candidate, "name", "")).upper()
                for candidate in legal
                if str(getattr(candidate, "name", "")).upper() != "RESET"
            }
        )
    )
    trace = build_action_target_trace(
        game_id=game_id,
        source_split="source_validation",
        policy_seed=policy_seed,
        reset_index=reset_index,
        step_index=step_index,
        collection_phase="v4_14_active_pre_action_only",
        available_action_names=available,
        selected_action_name=str(getattr(action, "name", "")).upper(),
        selected_action_data=dict(getattr(action, "action_args", {}) or {}),
        frame_before=snapshot.grid,
        frame_after=snapshot.grid,
        game_state_before=snapshot.game_state,
        game_state_after=snapshot.game_state,
        levels_completed_before=snapshot.levels_completed,
        levels_completed_after=snapshot.levels_completed,
    )
    return build_object_relative_graph(
        trace,
        maximum_neighbors=MAXIMUM_NEIGHBORS,
    )


def _compact_qwen_graph(graph: ObjectRelativeGraph) -> dict[str, Any]:
    root = graph.root
    return {
        "a": root.get("action_name", "unknown"),
        "f": root.get("action_family", "unknown"),
        "d": root.get("requested_direction", "none"),
        "k": root.get("root_kind", "unknown"),
        "o": root.get("root_occupied", 0),
        "af": root.get("root_affordance", "unknown"),
        "ar": root.get("actor_relation", "unknown"),
        "p": root.get("path_status", "unknown"),
        "b": root.get("boundary", "unknown"),
        "n": [
            [
                neighbor.get("direction", "unknown"),
                neighbor.get("proximity", "unknown"),
                neighbor.get("relative_size", "unknown"),
                neighbor.get("is_actor", 0),
                neighbor.get("aligned_row", 0),
                neighbor.get("aligned_col", 0),
            ]
            for neighbor in graph.neighbors[:4]
        ],
    }


def _active_qwen_prompt(
    left: ObjectRelativeGraph,
    right: ObjectRelativeGraph,
    *,
    recent_effects: Mapping[str, float],
) -> str:
    payload = {
        "task": "predict effects; output exactly 14 bits",
        "effect_order": list(SLOT_EFFECTS),
        "recent_effect_bits": {
            effect: int(float(recent_effects.get(effect, 0.0)) >= 0.5)
            for effect in SLOT_EFFECTS
        },
        "slots_0_then_1": [
            _compact_qwen_graph(left),
            _compact_qwen_graph(right),
        ],
        "answer": "",
    }
    prompt = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    lowered = prompt.lower()
    for token in FORBIDDEN_STUDENT_FIELDS:
        if f'"{token}"' in lowered:
            raise ValueError(f"forbidden V4.14 field in active Qwen prompt: {token}")
    return prompt


def _qwen_candidate_priors(
    decoder: ConstrainedQwenBitDecoder,
    graphs: Sequence[ObjectRelativeGraph],
    *,
    recent_effects: Mapping[str, float],
    run_id: str,
    step_index: int,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not graphs:
        return [], [], {"rows": 0, "inference_seconds": 0.0}
    pairs = [
        (index, min(index + 1, len(graphs) - 1)) for index in range(0, len(graphs), 2)
    ]
    prompts = [
        _active_qwen_prompt(
            graphs[left],
            graphs[right],
            recent_effects=recent_effects,
        )
        for left, right in pairs
    ]
    scores, runtime = decoder.score(
        prompts,
        bit_count=2 * len(SLOT_EFFECTS),
    )
    priors: list[dict[str, float] | None] = [None] * len(graphs)
    rows = []
    for pair_index, ((left, right), prompt, values) in enumerate(
        zip(pairs, prompts, scores)
    ):
        if len(values) != 2 * len(SLOT_EFFECTS):
            raise RuntimeError("active Qwen returned the wrong bit count")
        annotations = []
        for slot_index, candidate_index in enumerate((left, right)):
            start = slot_index * len(SLOT_EFFECTS)
            probabilities = {
                effect: float(values[start + effect_index])
                for effect_index, effect in enumerate(SLOT_EFFECTS)
            }
            if priors[candidate_index] is None:
                priors[candidate_index] = probabilities
            annotations.append(
                {
                    "candidate_index": candidate_index,
                    "effect_probabilities": probabilities,
                    "bits": "".join(
                        "1" if probabilities[effect] >= 0.5 else "0"
                        for effect in SLOT_EFFECTS
                    ),
                }
            )
        rows.append(
            {
                "format_version": PREDICTION_VERSION,
                "run_id": run_id,
                "step_index": step_index,
                "pair_index": pair_index,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "annotations": annotations,
                "strict_bitstream_valid": True,
                "compiler_slot_coverage": 1.0,
            }
        )
    if any(prior is None for prior in priors):
        raise RuntimeError("active Qwen did not cover every legal candidate")
    return (
        [dict(prior or {}) for prior in priors],
        rows,
        runtime,
    )


def _cache_qwen_by_action(
    legal: Sequence[Any],
    priors: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for action, prior in zip(legal, priors):
        grouped[str(getattr(action, "name", "")).upper()].append(prior)
    return {
        action_name: {
            effect: float(
                np.mean([float(probabilities[effect]) for probabilities in rows])
            )
            for effect in SLOT_EFFECTS
        }
        for action_name, rows in grouped.items()
    }


def _candidate_action_plan(
    action_name: str,
    available: Sequence[str],
    sequence_table: Mapping[tuple[str, ...], float],
    *,
    depth: int = 3,
) -> tuple[str, ...]:
    available_set = set(available)
    matches = [
        (float(value), sequence)
        for sequence, value in sequence_table.items()
        if sequence
        and sequence[0] == action_name
        and all(item in available_set for item in sequence)
    ]
    if matches:
        selected = max(matches, key=lambda row: (row[0], row[1]))[1]
    else:
        selected = (action_name,)
    return tuple(
        list(selected[:depth]) + [selected[-1]] * max(0, depth - len(selected))
    )


def _active_baseline_scores(
    legal: Sequence[Any],
    sequence_table: Mapping[tuple[str, ...], float],
    action_table: Mapping[str, float],
    global_value: float,
) -> list[float]:
    available = tuple(
        sorted({str(getattr(action, "name", "")).upper() for action in legal})
    )
    scores = []
    for action in legal:
        name = str(getattr(action, "name", "")).upper()
        candidates = [
            float(value)
            for sequence, value in sequence_table.items()
            if sequence
            and sequence[0] == name
            and all(item in set(available) for item in sequence)
        ]
        scores.append(
            max(
                candidates,
                default=float(action_table.get(name, global_value)),
            )
        )
    return scores


def _load_active_ebm(
    path: Path,
    *,
    device: str,
) -> PairwiseTrajectoryEBM:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    ebm = PairwiseTrajectoryEBM(
        input_width=int(payload["input_width"]),
        hidden_width=int(payload["hidden_width"]),
        seed=SEED,
    ).to(device)
    ebm.model.load_state_dict(payload["state_dict"])
    ebm.trained_pairs = int(payload["trained_pairs"])
    ebm.model.eval()
    return ebm


def _run_active_controller(
    *,
    controller: str,
    game_id: str,
    seed: int,
    action_budget: int,
    maximum_resets: int,
    qwen_refresh_steps: int,
    model: Any,
    parameters: Mapping[str, Any],
    ebm: PairwiseTrajectoryEBM,
    decoder: ConstrainedQwenBitDecoder,
    action_table: Mapping[str, float],
    sequence_table: Mapping[tuple[str, ...], float],
    global_value: float,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.m2.m3_execution_smoke import _reset_env
    from theory.non_ar25_active_micro_run import _env_dir
    from theory.real_env_option_adapter import snapshot_frame
    from theory.sage12.bound_mechanic_pilot import _legal_actions
    from theory.unified_cognition_ab_benchmark import (
        _is_terminal,
        _make_real_env,
    )

    if controller not in {"action_sequence_only", "qwen_temporal_ebm"}:
        raise ValueError(f"unknown V4.14 active controller: {controller}")
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
    terminal_events = 0
    candidate_counts = []
    decision_latencies = []
    execution_latencies = []
    belief = TemporalBeliefState()
    qwen_cache: dict[str, dict[str, float]] = {}
    qwen_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    qwen_requests = 0
    qwen_candidates = 0
    qwen_seconds = 0.0
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
            belief = TemporalBeliefState()
            qwen_cache = {}
            continue
        legal = tuple(_legal_actions(environment))
        if not legal:
            stop_reason = "no_legal_actions"
            break
        candidate_counts.append(len(legal))
        decision_started = time.perf_counter()
        qwen_refresh = False
        if controller == "action_sequence_only":
            scores = _active_baseline_scores(
                legal,
                sequence_table,
                action_table,
                global_value,
            )
            best_score = max(scores)
            candidates = [
                index
                for index, score in enumerate(scores)
                if abs(score - best_score) <= 1e-12
            ]
            selected_index = min(
                candidates,
                key=lambda index: hashlib.sha256(
                    (
                        f"{run_id}:{actions_executed}:"
                        f"{_live_action_signature(legal[index])}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            selected_energy = None
            selected_plan = _candidate_action_plan(
                str(getattr(legal[selected_index], "name", "")).upper(),
                [str(getattr(action, "name", "")).upper() for action in legal],
                sequence_table,
            )
            selected_prediction = None
        else:
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
            qwen_refresh = (
                not qwen_cache or episode_steps % max(1, qwen_refresh_steps) == 0
            )
            if qwen_refresh:
                current_priors, rows, runtime = _qwen_candidate_priors(
                    decoder,
                    graphs,
                    recent_effects=belief.effect_probabilities,
                    run_id=run_id,
                    step_index=actions_executed,
                )
                qwen_cache = _cache_qwen_by_action(legal, current_priors)
                qwen_rows.extend(rows)
                qwen_requests += int(runtime["rows"])
                qwen_candidates += len(legal)
                qwen_seconds += float(runtime["inference_seconds"])
            priors = [
                qwen_cache.get(str(getattr(action, "name", "")).upper())
                for action in legal
            ]
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
            predictions = _predict_candidate_rollouts(
                model,
                graph_plans,
                parameters=parameters,
                device=device,
                initial_belief=belief,
                qwen_priors=priors,
            )
            features = [_prediction_features(prediction) for prediction in predictions]
            energies = ebm.energies(features)
            selected_index = min(
                range(len(legal)),
                key=lambda index: (
                    float(energies[index]),
                    hashlib.sha256(
                        (
                            f"{run_id}:{actions_executed}:"
                            f"{_live_action_signature(legal[index])}"
                        ).encode("utf-8")
                    ).hexdigest(),
                ),
            )
            selected_energy = float(energies[selected_index])
            selected_plan = action_plans[selected_index]
            selected_prediction = predictions[selected_index][0]
        decision_latencies.append(time.perf_counter() - decision_started)
        selected = legal[selected_index]
        selected_signature = _live_action_signature(selected)
        before_digest = grid_sha256(before.grid)
        execution_started = time.perf_counter()
        try:
            next_frame = _step_env_action(environment, selected)
        except Exception as exc:
            illegal_proposals += 1
            traces.append(
                {
                    "format_version": FORMAT_VERSION,
                    "run_id": run_id,
                    "action_index": actions_executed,
                    "selected_action": selected_signature,
                    "execution_error": f"{type(exc).__name__}:{exc}",
                }
            )
            stop_reason = "execution_error"
            break
        execution_latencies.append(time.perf_counter() - execution_started)
        after = snapshot_frame(next_frame)
        level_delta = max(
            0,
            int(after.levels_completed) - int(before.levels_completed),
        )
        is_win = str(after.game_state).upper() == "WIN"
        is_game_over = str(after.game_state).upper() == "GAME_OVER"
        levels += level_delta
        wins += int(is_win)
        game_overs += int(is_game_over)
        terminal_events += int(_is_terminal(after.game_state))
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
                "pre_state_sha256": before_digest,
                "post_state_sha256": grid_sha256(after.grid),
                "candidate_count": len(legal),
                "selected_action": selected_signature,
                "selected_plan": list(selected_plan),
                "selected_energy": selected_energy,
                "qwen_refresh": qwen_refresh,
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
        if selected_prediction is not None:
            belief = selected_prediction.next_belief
        if _is_terminal(after.game_state):
            if resets >= maximum_resets:
                stop_reason = "maximum_resets"
                break
            frame = _reset_env(environment)
            resets += 1
            episode_steps = 0
            belief = TemporalBeliefState()
            qwen_cache = {}
    summary = {
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
        "terminal_events": terminal_events,
        "illegal_proposals": illegal_proposals,
        "stop_reason": stop_reason,
        "mean_candidates": float(np.mean(candidate_counts))
        if candidate_counts
        else 0.0,
        "decision_latency_seconds": {
            "mean": float(np.mean(decision_latencies)) if decision_latencies else 0.0,
            "p95": float(np.quantile(decision_latencies, 0.95))
            if decision_latencies
            else 0.0,
        },
        "execution_latency_seconds": {
            "mean": float(np.mean(execution_latencies)) if execution_latencies else 0.0,
        },
        "qwen": {
            "refresh_steps": qwen_refresh_steps,
            "requests": qwen_requests,
            "candidates_covered": qwen_candidates,
            "inference_seconds": qwen_seconds,
        },
    }
    return summary, traces, qwen_rows


def _active_summary(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
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
    requested_device: str = "auto",
    qwen_refresh_steps: int = ACTIVE_QWEN_REFRESH_STEPS,
) -> dict[str, Any]:
    """Run the frozen paired live panel without opening the final holdout."""

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    result_path = destination / "result.json"
    if not result_path.exists():
        raise FileNotFoundError("run V4.14 evaluate before active validation")
    result = _read_json(result_path)
    semantic = _read_json(destination / "semantic_result.json")
    device = (
        str(semantic["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    model, checkpoint = _load_checkpoint(
        destination / "temporal_student.pt",
        device=device,
    )
    parameters = checkpoint["parameters"]
    ebm = _load_active_ebm(
        destination / "trajectory_ebm.pt",
        device=device,
    )
    decoder = ConstrainedQwenBitDecoder(
        model_path=DEFAULT_QWEN_MODEL_PATH,
        device=device,
        batch_size=QWEN_BATCH_SIZE,
        maximum_input_tokens=QWEN_MAXIMUM_INPUT_TOKENS,
    )
    records = load_teacher_records(destination)
    action_table, sequence_table, global_value = _action_sequence_tables(records)
    evaluation = manifest["evaluation"]
    action_budget = int(evaluation["active_validation_action_budget"])
    maximum_resets = int(evaluation["active_validation_maximum_resets"])
    runs = []
    traces = []
    qwen_rows = []
    started = time.perf_counter()
    for game_id in ACTIVE_VALIDATION_GAMES:
        for seed in evaluation["active_validation_seeds"]:
            for controller in (
                "action_sequence_only",
                "qwen_temporal_ebm",
            ):
                run, run_traces, run_qwen = _run_active_controller(
                    controller=controller,
                    game_id=game_id,
                    seed=int(seed),
                    action_budget=action_budget,
                    maximum_resets=maximum_resets,
                    qwen_refresh_steps=qwen_refresh_steps,
                    model=model,
                    parameters=parameters,
                    ebm=ebm,
                    decoder=decoder,
                    action_table=action_table,
                    sequence_table=sequence_table,
                    global_value=global_value,
                    device=device,
                )
                runs.append(run)
                traces.extend(run_traces)
                qwen_rows.extend(run_qwen)
    runs_path = destination / "active_runs.jsonl"
    traces_path = destination / "active_traces.jsonl"
    qwen_path = destination / "active_qwen_outputs.jsonl"
    _write_jsonl(runs_path, runs)
    _write_jsonl(traces_path, traces)
    _write_jsonl(qwen_path, qwen_rows)
    metrics = _active_summary(runs)
    baseline_by_key = {
        (str(row["game_id"]), int(row["seed"])): row
        for row in runs
        if row["controller"] == "action_sequence_only"
    }
    paired = []
    for row in runs:
        if row["controller"] != "qwen_temporal_ebm":
            continue
        baseline = baseline_by_key[(str(row["game_id"]), int(row["seed"]))]
        paired.append(
            {
                "game_id": row["game_id"],
                "seed": row["seed"],
                "level_gain": int(row["levels_completed"])
                - int(baseline["levels_completed"]),
                "win_gain": int(row["wins"]) - int(baseline["wins"]),
                "game_over_delta": int(row["game_overs"]) - int(baseline["game_overs"]),
            }
        )
    qwen_runtime = {
        "refresh_steps": qwen_refresh_steps,
        "requests": sum(
            int(row["qwen"]["requests"])
            for row in runs
            if row["controller"] == "qwen_temporal_ebm"
        ),
        "candidates_covered": sum(
            int(row["qwen"]["candidates_covered"])
            for row in runs
            if row["controller"] == "qwen_temporal_ebm"
        ),
        "inference_seconds": sum(
            float(row["qwen"]["inference_seconds"])
            for row in runs
            if row["controller"] == "qwen_temporal_ebm"
        ),
        "strict_bitstream_validity": float(
            np.mean([bool(row["strict_bitstream_valid"]) for row in qwen_rows])
        )
        if qwen_rows
        else 0.0,
        "compiler_slot_coverage": float(
            np.mean([float(row["compiler_slot_coverage"]) for row in qwen_rows])
        )
        if qwen_rows
        else 0.0,
    }
    active = {
        "status": "COMPLETE",
        "games": list(ACTIVE_VALIDATION_GAMES),
        "seeds": list(evaluation["active_validation_seeds"]),
        "controllers": [
            "action_sequence_only",
            "qwen_temporal_ebm",
        ],
        "action_budget_per_run": action_budget,
        "maximum_resets_per_run": maximum_resets,
        "runs": len(runs),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "paired": paired,
        "mean_paired_level_gain": float(np.mean([row["level_gain"] for row in paired])),
        "mean_paired_win_gain": float(np.mean([row["win_gain"] for row in paired])),
        "qwen": qwen_runtime,
        "descriptive_only": True,
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "runs": _file_sha256(runs_path),
            "traces": _file_sha256(traces_path),
            "qwen_outputs": _file_sha256(qwen_path),
        },
    }
    active["active_checksum"] = _checksum(active)
    _write_json(destination / "active_validation.json", active)

    result["active_validation"] = active
    result["qwen_contract"] = {
        "candidate_complete_compiler_preserved": (
            qwen_runtime["compiler_slot_coverage"] == 1.0
        ),
        "frozen_v4_7_decoding_unchanged": True,
        "new_panel_generations_run": bool(qwen_rows),
        "decoder": "autoregressive token-logit mask to atomic 0/1",
        "sampling": False,
        "temperature": None,
        "refresh_steps": qwen_refresh_steps,
        "blend_weight": ACTIVE_QWEN_BLEND_WEIGHT,
        "runtime": qwen_runtime,
    }
    result["topology"]["live_validation_measured"] = True
    result["topology"]["live_win_rate_authority"] = False
    result["all_conditions_executed"] = True
    result["checks"]["all_conditions_executed"] = True
    result["global_chain_supported"] = all(result["checks"].values())
    result["artifact_sha256"].update(
        {
            "active_validation": _file_sha256(destination / "active_validation.json"),
            "active_runs": _file_sha256(runs_path),
            "active_traces": _file_sha256(traces_path),
            "active_qwen_outputs": _file_sha256(qwen_path),
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
    v413_dir: str | Path = DEFAULT_V413_DIR,
    device: str = "auto",
) -> dict[str, Any]:
    manifest_path = Path(output_dir) / "frozen_manifest.json"
    if not manifest_path.exists():
        freeze_manifest(
            output_dir=output_dir,
            traces_dir=traces_dir,
            v411_dir=v411_dir,
            v413_dir=v413_dir,
        )
    teacher = compile_human_teacher(
        output_dir=output_dir,
        traces_dir=traces_dir,
    )
    semantic = train_temporal_student(
        output_dir=output_dir,
        requested_device=device,
    )
    integration = evaluate_transfer_and_global_chain(
        output_dir=output_dir,
        v411_dir=v411_dir,
        requested_device=device,
    )
    active = run_active_validation(
        output_dir=output_dir,
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
            "--v413-dir",
            type=Path,
            default=DEFAULT_V413_DIR,
        )
        child.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
            v411_dir=args.v411_dir,
            v413_dir=args.v413_dir,
        )
    elif args.command == "compile":
        payload = compile_human_teacher(
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
        )
    elif args.command == "train":
        payload = train_temporal_student(
            output_dir=args.output_dir,
            requested_device=args.device,
        )
    elif args.command == "evaluate":
        payload = evaluate_transfer_and_global_chain(
            output_dir=args.output_dir,
            v411_dir=args.v411_dir,
            requested_device=args.device,
        )
    elif args.command == "active":
        payload = run_active_validation(
            output_dir=args.output_dir,
            requested_device=args.device,
        )
    else:
        payload = run_all(
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
            v411_dir=args.v411_dir,
            v413_dir=args.v413_dir,
            device=args.device,
        )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_VALIDATION_GAMES",
    "FINAL_CONFIRMATION_GAMES",
    "HUMAN_TRAIN_GAMES",
    "HORIZONS",
    "ROLE_LABELS",
    "TRANSFER_GAMES",
    "TemporalBeliefState",
    "TemporalSlotPrediction",
    "TemporalTeacherRecord",
    "compile_human_teacher",
    "evaluate_transfer_and_global_chain",
    "freeze_manifest",
    "load_manifest",
    "load_teacher_records",
    "run_active_validation",
    "run_all",
    "tensorize_temporal_records",
    "train_temporal_student",
]
