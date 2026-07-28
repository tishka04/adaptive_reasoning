"""SAGE12 V4.11 counterfactual semantic panels.

The post-transition teacher may inspect executed branches, while every student
input remains pre-action, action-aligned, identity-free, and source-only.
V4.11 learns causal preferences between actions replayed from the same state,
then distils relation residuals onto a root-only absolute-probability anchor.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN

from .action_aligned_semantics_v4_10 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V410_DIR,
    _AXIS_SHUFFLE,
    action_aligned_graph,
    load_pair_links as load_v410_pair_links,
    load_teacher_records as load_v410_records,
    validate_action_aligned_graph,
)
from .action_target_data import ActionTargetTrace, grid_sha256
from .compiler import SLOT_EFFECTS, SlotAnnotation
from .integration_pilot import load_complete_roots
from .integration_pilot_v4_7 import load_slot_examples
from .object_relative_student_v4_9 import (
    _action_only_probabilities,
    _batch_arrays,
    _brier_metrics,
    _completion_recall_at_8,
    _per_game_brier,
    _select_device,
    tensorize_records,
)
from .semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
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
)

FORMAT_VERSION = "sage12-counterfactual-panel-v4.11"
MANIFEST_VERSION = "sage12-counterfactual-semantics-manifest-v4.11"
TEACHER_VERSION = "sage12-counterfactual-teacher-panel-v4.11"
QA_VERSION = "sage12-counterfactual-teacher-qa-v4.11"
RESULT_VERSION = "sage12-counterfactual-semantics-result-v4.11"
PREDICTION_VERSION = "sage12-counterfactual-logo-prediction-v4.11"
SLOT_EXPORT_VERSION = "sage12-counterfactual-slot-annotations-v4.11"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "counterfactual_semantics_v4_11"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"

PROGRESS_HORIZON = 3
PROGRESS_GAMMA = 0.8
PROGRESS_DEADBAND = 0.25
CONTINUATION_ROLLOUTS = 2
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _action_key(name: str, data: Mapping[str, Any]) -> str:
    return _canonical({"name": str(name), "action_data": dict(data)})


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class PanelArm:
    """One immediate intervention plus two deterministic continuation rolls."""

    arm_index: int
    replay_pre_state_sha256: str
    immediate_trace: ActionTargetTrace
    continuations: tuple[tuple[ActionTargetTrace, ...], ...]

    def __post_init__(self) -> None:
        if self.arm_index < 0:
            raise ValueError("panel arm index must be non-negative")
        if len(self.continuations) != CONTINUATION_ROLLOUTS:
            raise ValueError("V4.11 requires exactly two continuation rolls")
        if any(len(row) > PROGRESS_HORIZON - 1 for row in self.continuations):
            raise ValueError("V4.11 continuation exceeds the frozen horizon")
        before = grid_sha256(self.immediate_trace.frame_before)
        if any(
            grid_sha256(row.frame_before) == ""
            for rollout in self.continuations
            for row in rollout
        ):
            raise ValueError("invalid continuation frame")
        if not before:
            raise ValueError("invalid immediate pre-state")

    @property
    def action_key(self) -> str:
        return _action_key(
            self.immediate_trace.selected_action_name,
            self.immediate_trace.selected_action_data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_index": self.arm_index,
            "replay_pre_state_sha256": self.replay_pre_state_sha256,
            "immediate_trace": self.immediate_trace.to_dict(),
            "continuations": [
                [trace.to_dict() for trace in rollout]
                for rollout in self.continuations
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PanelArm:
        return cls(
            arm_index=int(payload["arm_index"]),
            replay_pre_state_sha256=str(payload["replay_pre_state_sha256"]),
            immediate_trace=ActionTargetTrace.from_dict(payload["immediate_trace"]),
            continuations=tuple(
                tuple(ActionTargetTrace.from_dict(item) for item in rollout)
                for rollout in payload["continuations"]
            ),
        )


@dataclass(frozen=True)
class CounterfactualPanel:
    """Two to four actions executed from one replay-verified source state."""

    game_id: str
    policy_seed: int
    reset_index: int
    panel_index: int
    expected_pre_state_sha256: str
    pre_grid_sha256: str
    arms: tuple[PanelArm, ...]
    panel_id: str = ""
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported V4.11 panel format")
        if self.game_id not in SOURCE_TRAIN:
            raise ValueError("V4.11 panels are source-train only")
        if not 2 <= len(self.arms) <= 4:
            raise ValueError("V4.11 panels require two to four arms")
        if len({arm.action_key for arm in self.arms}) != len(self.arms):
            raise ValueError("V4.11 panel arms must be distinct")
        if any(
            arm.replay_pre_state_sha256 != self.expected_pre_state_sha256
            for arm in self.arms
        ):
            raise ValueError("V4.11 panel replay hashes differ")
        if any(
            grid_sha256(arm.immediate_trace.frame_before) != self.pre_grid_sha256
            for arm in self.arms
        ):
            raise ValueError("V4.11 panel arm grids differ")
        if any(
            arm.immediate_trace.game_id != self.game_id
            or arm.immediate_trace.source_split != "source_train"
            for arm in self.arms
        ):
            raise ValueError("V4.11 panel source firewall violation")
        if not self.panel_id:
            payload = {
                "game_id": self.game_id,
                "policy_seed": self.policy_seed,
                "reset_index": self.reset_index,
                "panel_index": self.panel_index,
                "pre_state": self.expected_pre_state_sha256,
                "actions": [arm.action_key for arm in self.arms],
            }
            object.__setattr__(self, "panel_id", _checksum(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "panel_id": self.panel_id,
            "game_id": self.game_id,
            "source_split": "source_train",
            "policy_seed": self.policy_seed,
            "reset_index": self.reset_index,
            "panel_index": self.panel_index,
            "expected_pre_state_sha256": self.expected_pre_state_sha256,
            "pre_grid_sha256": self.pre_grid_sha256,
            "arms": [arm.to_dict() for arm in self.arms],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CounterfactualPanel:
        if str(payload.get("source_split")) != "source_train":
            raise ValueError("V4.11 raw panel is not source-train")
        return cls(
            game_id=str(payload["game_id"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            panel_index=int(payload["panel_index"]),
            expected_pre_state_sha256=str(payload["expected_pre_state_sha256"]),
            pre_grid_sha256=str(payload["pre_grid_sha256"]),
            arms=tuple(PanelArm.from_dict(row) for row in payload["arms"]),
            panel_id=str(payload["panel_id"]),
            format_version=str(payload["format_version"]),
        )


@dataclass(frozen=True)
class TeacherArm:
    arm_index: int
    trace_digest: str
    exact_repeat_key: str
    action_name: str
    action_data: Mapping[str, Any]
    graph: ObjectRelativeGraph
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    immediate_score: float
    horizon_return: float
    horizon_uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_index": self.arm_index,
            "trace_digest": self.trace_digest,
            "exact_repeat_key": self.exact_repeat_key,
            "action_name": self.action_name,
            "action_data": _json_safe(self.action_data),
            "model_graph": self.graph.to_dict(),
            "teacher": {
                "labels": {effect: bool(self.labels[effect]) for effect in SEMANTIC_EFFECTS},
                "applicable": {
                    effect: bool(self.applicable[effect])
                    for effect in SEMANTIC_EFFECTS
                },
                "immediate_score": float(self.immediate_score),
                "horizon_return": float(self.horizon_return),
                "horizon_uncertainty": float(self.horizon_uncertainty),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TeacherArm:
        teacher = payload["teacher"]
        graph = payload["model_graph"]
        return cls(
            arm_index=int(payload["arm_index"]),
            trace_digest=str(payload["trace_digest"]),
            exact_repeat_key=str(payload["exact_repeat_key"]),
            action_name=str(payload["action_name"]),
            action_data=dict(payload["action_data"]),
            graph=ObjectRelativeGraph(
                root=dict(graph["root"]),
                neighbors=tuple(dict(row) for row in graph["neighbors"]),
            ),
            labels={
                effect: bool(teacher["labels"][effect]) for effect in SEMANTIC_EFFECTS
            },
            applicable={
                effect: bool(teacher["applicable"][effect])
                for effect in SEMANTIC_EFFECTS
            },
            immediate_score=float(teacher["immediate_score"]),
            horizon_return=float(teacher["horizon_return"]),
            horizon_uncertainty=float(teacher["horizon_uncertainty"]),
        )


@dataclass(frozen=True)
class TeacherPanel:
    panel_id: str
    game_id: str
    pre_state_sha256: str
    arms: tuple[TeacherArm, ...]
    format_version: str = TEACHER_VERSION

    def __post_init__(self) -> None:
        if self.format_version != TEACHER_VERSION:
            raise ValueError("unsupported V4.11 teacher-panel format")
        if self.game_id not in SOURCE_TRAIN:
            raise ValueError("V4.11 teacher panel outside source-train")
        if not 2 <= len(self.arms) <= 4:
            raise ValueError("invalid V4.11 teacher-panel arity")
        for arm in self.arms:
            validate_action_aligned_graph(arm.graph)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "panel_id": self.panel_id,
            "audit": {
                "game_id": self.game_id,
                "pre_state_sha256": self.pre_state_sha256,
            },
            "arms": [arm.to_dict() for arm in self.arms],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TeacherPanel:
        return cls(
            panel_id=str(payload["panel_id"]),
            game_id=str(payload["audit"]["game_id"]),
            pre_state_sha256=str(payload["audit"]["pre_state_sha256"]),
            arms=tuple(TeacherArm.from_dict(row) for row in payload["arms"]),
            format_version=str(payload["format_version"]),
        )


@dataclass(frozen=True)
class SemanticPanelPrediction:
    """Candidate-complete prediction; existing SlotAnnotation stays unchanged."""

    panel_id: str
    effect_probabilities: Mapping[str, Mapping[str, float]]
    progress_scores: Mapping[str, float]
    preference_probabilities: Mapping[str, Mapping[str, float]]

    def to_slot_annotations(self) -> tuple[SlotAnnotation, ...]:
        return tuple(
            SlotAnnotation(
                slot_id=slot_id,
                effect_probabilities={
                    effect: float(probabilities[effect]) for effect in SLOT_EFFECTS
                },
                source="counterfactual_panel_logo_v4_11",
                support=0,
            )
            for slot_id, probabilities in sorted(self.effect_probabilities.items())
        )


def _source_fingerprints(v410_dir: Path) -> dict[str, Any]:
    paths = (
        v410_dir / "frozen_manifest.json",
        v410_dir / "teacher_corpus.jsonl",
        v410_dir / "same_prestate_pairs.jsonl",
        v410_dir / "student_result.json",
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
    v410_dir: str | Path = DEFAULT_V410_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "source_games": list(SOURCE_TRAIN),
        "source_fingerprints": _source_fingerprints(Path(v410_dir)),
        "collection": {
            "source_only": True,
            "target_panels_per_game": 96,
            "minimum_panels_per_game": 80,
            "minimum_arms_per_panel": 2,
            "maximum_arms_per_panel": 4,
            "maximum_resets_per_game": 60,
            "action_budget_per_reset": 128,
            "policy_seeds": [6011, 6029, 6053, 6079, 6101],
            "progress_horizon": PROGRESS_HORIZON,
            "progress_gamma": PROGRESS_GAMMA,
            "continuation_rollouts": CONTINUATION_ROLLOUTS,
            "progress_deadband": PROGRESS_DEADBAND,
            "outcome_adaptive": False,
            "reject_v43_v410_exact_repeats": True,
        },
        "teacher_capacity": {
            "minimum_progress_discordant_panels_per_game": 20,
            "minimum_games_with_progress_capacity": 8,
            "effect_eligibility_minimum_discordant_comparisons": 100,
            "effect_eligibility_minimum_games": 4,
            "terminal_minimum_positive_arms": 20,
            "terminal_minimum_games": 4,
        },
        "training": {
            "seed": 5_110,
            "hash_buckets": 2048,
            "embedding_width": 32,
            "hidden_width": 96,
            "epochs": 30,
            "samples_per_game_per_epoch": 256,
            "maximum_pairs_per_epoch": 4096,
            "learning_rate": 0.0015,
            "weight_decay": 0.0001,
            "effect_pair_weight": 0.50,
            "progress_pair_weight": 0.50,
            "tie_consistency_weight": 0.05,
            "alpha_grid": list(ALPHA_GRID),
            "calibration": "inner_game_balanced_logit_shift",
        },
        "evaluation": {
            "outer_split": "leave_one_source_train_game_out",
            "primary_test_rows": "fresh_counterfactual_panels_only",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 51_100,
            "decision_thresholds": {
                "preference_gain_ci_lower_strictly_positive": True,
                "top1_regret_reduction_ci_lower_strictly_positive": True,
                "relation_shuffle_pair_degradation_ci_lower_strictly_positive": True,
                "absolute_brier_gain_ci_lower_strictly_positive": True,
                "absolute_ece_not_worse_than_root_only": True,
                "relation_shuffle_brier_degradation_strictly_positive": True,
                "nonnegative_games_minimum": 6,
                "identity_increment_ci_upper_maximum": 0.02,
                "neighbor_permutation_max_probability_delta": 1e-6,
                "arm_swap_max_complement_error": 1e-6,
                "terminal_recall_at_8_minimum_when_eligible": 0.20,
            },
            "confirmatory": False,
            "can_promote_live_authority": False,
            "can_fit_world_model_in_this_iteration": False,
            "can_fit_ebm_in_this_iteration": False,
        },
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / "frozen_manifest.json")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.11 manifest")
    expected = str(manifest["manifest_checksum"])
    check = dict(manifest)
    check.pop("manifest_checksum")
    if _checksum(check) != expected:
        raise ValueError("V4.11 manifest checksum mismatch")
    if tuple(manifest["source_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.11 source split drift")
    return manifest


def load_raw_panels(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[CounterfactualPanel, ...]:
    directory = Path(output_dir) / "source_train_shards"
    rows = []
    for game in SOURCE_TRAIN:
        path = directory / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(CounterfactualPanel.from_dict(row) for row in _read_jsonl(path))
    return tuple(rows)


def load_teacher_panels(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[TeacherPanel, ...]:
    return tuple(
        TeacherPanel.from_dict(row)
        for row in _read_jsonl(Path(output_dir) / "teacher_panels.jsonl")
    )


def _horizon_return(arm: PanelArm) -> tuple[float, float]:
    _labels, _applicable, immediate, _evidence = compile_semantics(
        arm.immediate_trace
    )
    returns = []
    for rollout in arm.continuations:
        value = float(immediate)
        for offset, trace in enumerate(rollout, start=1):
            _row_labels, _row_applicable, score, _row_evidence = compile_semantics(
                trace
            )
            value += (PROGRESS_GAMMA**offset) * float(score)
        returns.append(value)
    return float(np.mean(returns)), float(np.std(returns))


def compile_teacher_panels(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    collection = _read_json(destination / "collection_manifest.json")
    if collection.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ValueError("V4.11 collection/manifest mismatch")
    raw_panels = load_raw_panels(destination)
    teacher_panels = []
    for panel in raw_panels:
        arms = []
        for raw_arm in panel.arms:
            trace = raw_arm.immediate_trace
            labels, applicable, score, _evidence = compile_semantics(trace)
            horizon_return, uncertainty = _horizon_return(raw_arm)
            graph = action_aligned_graph(build_object_relative_graph(trace))
            arms.append(
                TeacherArm(
                    arm_index=raw_arm.arm_index,
                    trace_digest=trace.trace_digest,
                    exact_repeat_key=trace.exact_repeat_key(),
                    action_name=trace.selected_action_name,
                    action_data=dict(trace.selected_action_data),
                    graph=graph,
                    labels=labels,
                    applicable=applicable,
                    immediate_score=score,
                    horizon_return=horizon_return,
                    horizon_uncertainty=uncertainty,
                )
            )
        teacher_panels.append(
            TeacherPanel(
                panel_id=panel.panel_id,
                game_id=panel.game_id,
                pre_state_sha256=panel.expected_pre_state_sha256,
                arms=tuple(arms),
            )
        )
    teacher_panels.sort(key=lambda row: (row.game_id, row.panel_id))
    teacher_path = destination / "teacher_panels.jsonl"
    _write_jsonl(teacher_path, (row.to_dict() for row in teacher_panels))

    progress_discordant = Counter()
    effect_discordant = {effect: Counter() for effect in SEMANTIC_EFFECTS}
    effect_totals = Counter()
    terminal_positives = Counter()
    comparisons = 0
    for panel in teacher_panels:
        terminal_positives[panel.game_id] += sum(
            int(arm.labels["level_complete"]) for arm in panel.arms
        )
        panel_progress = False
        for left, right in itertools.combinations(panel.arms, 2):
            comparisons += 1
            if abs(left.horizon_return - right.horizon_return) >= PROGRESS_DEADBAND:
                panel_progress = True
            for effect in SEMANTIC_EFFECTS:
                if (
                    left.applicable[effect]
                    and right.applicable[effect]
                    and left.labels[effect] != right.labels[effect]
                ):
                    effect_discordant[effect][panel.game_id] += 1
                    effect_totals[effect] += 1
        progress_discordant[panel.game_id] += int(panel_progress)

    capacity = manifest["teacher_capacity"]
    eligible_effects = [
        effect
        for effect in SEMANTIC_EFFECTS
        if effect_totals[effect]
        >= int(capacity["effect_eligibility_minimum_discordant_comparisons"])
        and sum(value > 0 for value in effect_discordant[effect].values())
        >= int(capacity["effect_eligibility_minimum_games"])
    ]
    games_with_progress = sum(
        progress_discordant[game]
        >= int(capacity["minimum_progress_discordant_panels_per_game"])
        for game in SOURCE_TRAIN
    )
    terminal_positive_total = sum(terminal_positives.values())
    terminal_games = sum(value > 0 for value in terminal_positives.values())
    qa: dict[str, Any] = {
        "format_version": QA_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "collection_checksum": collection["collection_checksum"],
        "panels": len(teacher_panels),
        "arms": sum(len(panel.arms) for panel in teacher_panels),
        "comparisons": comparisons,
        "panels_per_game": {
            game: sum(panel.game_id == game for panel in teacher_panels)
            for game in SOURCE_TRAIN
        },
        "progress_discordant_panels_per_game": {
            game: int(progress_discordant[game]) for game in SOURCE_TRAIN
        },
        "games_with_progress_capacity": games_with_progress,
        "effect_discordant_comparisons": {
            effect: {
                "total": int(effect_totals[effect]),
                "per_game": {
                    game: int(effect_discordant[effect][game])
                    for game in SOURCE_TRAIN
                },
            }
            for effect in SEMANTIC_EFFECTS
        },
        "eligible_effects": eligible_effects,
        "terminal_capacity": {
            "positive_arms": int(terminal_positive_total),
            "games": int(terminal_games),
            "eligible": bool(
                terminal_positive_total
                >= int(capacity["terminal_minimum_positive_arms"])
                and terminal_games >= int(capacity["terminal_minimum_games"])
            ),
        },
        "all_graphs_action_aligned": all(
            _graph_is_valid(arm.graph)
            for panel in teacher_panels
            for arm in panel.arms
        ),
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {"teacher_panels": _file_sha256(teacher_path)},
    }
    qa["checks"] = {
        "collection_ready": bool(collection["collection_ready"]),
        "minimum_panels_each_game": all(
            qa["panels_per_game"][game]
            >= int(manifest["collection"]["minimum_panels_per_game"])
            for game in SOURCE_TRAIN
        ),
        "progress_capacity": (
            games_with_progress
            >= int(capacity["minimum_games_with_progress_capacity"])
        ),
        "action_aligned_firewall": bool(qa["all_graphs_action_aligned"]),
    }
    qa["teacher_ready"] = all(qa["checks"].values())
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def _graph_is_valid(graph: ObjectRelativeGraph) -> bool:
    try:
        validate_action_aligned_graph(graph)
    except ValueError:
        return False
    return True


def _teacher_arm_record(panel: TeacherPanel, arm: TeacherArm) -> SemanticTeacherRecord:
    return SemanticTeacherRecord(
        example_id=f"v411_{panel.panel_id}_{arm.arm_index}",
        game_id=panel.game_id,
        source_corpus="counterfactual_panel_v4_11",
        trace_digest=arm.trace_digest,
        exact_repeat_key=arm.exact_repeat_key,
        same_prestate_keys=(panel.panel_id,),
        graph=arm.graph,
        labels=arm.labels,
        applicable=arm.applicable,
        productive_score=arm.immediate_score,
        teacher_evidence={
            "panel_id": panel.panel_id,
            "horizon_return": arm.horizon_return,
            "horizon_uncertainty": arm.horizon_uncertainty,
        },
        format_version=TEACHER_VERSION,
    )


@dataclass(frozen=True)
class _Comparison:
    panel_id: str
    game_id: str
    left: int
    right: int
    horizon_delta: float
    fresh: bool


def _assemble_training_data(
    teacher_panels: Sequence[TeacherPanel],
) -> tuple[
    tuple[SemanticTeacherRecord, ...],
    tuple[_Comparison, ...],
    dict[str, tuple[int, ...]],
    set[int],
]:
    records = list(load_v410_records())
    by_digest = {record.trace_digest: index for index, record in enumerate(records)}
    comparisons = []
    groups: dict[str, tuple[int, ...]] = {}
    for link in load_v410_pair_links():
        left = by_digest.get(link.left_trace_digest)
        right = by_digest.get(link.right_trace_digest)
        if left is None or right is None:
            continue
        groups[f"legacy:{link.pair_id}"] = (left, right)
        comparisons.append(
            _Comparison(
                panel_id=f"legacy:{link.pair_id}",
                game_id=link.game_id,
                left=left,
                right=right,
                horizon_delta=(
                    records[left].productive_score - records[right].productive_score
                ),
                fresh=False,
            )
        )
    fresh_indices = set()
    for panel in teacher_panels:
        indices = []
        horizon = {}
        for arm in panel.arms:
            if arm.trace_digest in by_digest:
                raise ValueError("V4.11 fresh arm duplicates V4.10 teacher record")
            index = len(records)
            records.append(_teacher_arm_record(panel, arm))
            by_digest[arm.trace_digest] = index
            fresh_indices.add(index)
            indices.append(index)
            horizon[index] = arm.horizon_return
        groups[panel.panel_id] = tuple(indices)
        for left, right in itertools.combinations(indices, 2):
            comparisons.append(
                _Comparison(
                    panel_id=panel.panel_id,
                    game_id=panel.game_id,
                    left=left,
                    right=right,
                    horizon_delta=horizon[left] - horizon[right],
                    fresh=True,
                )
            )
    return tuple(records), tuple(comparisons), groups, fresh_indices


def _torch_model(
    *,
    hash_buckets: int,
    embedding_width: int,
    hidden_width: int,
) -> Any:
    import torch

    class CounterfactualPanelNet(torch.nn.Module):
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
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(embedding_width * 3, hidden_width),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
                torch.nn.LayerNorm(hidden_width),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.GELU(),
            )
            self.effect_head = torch.nn.Linear(hidden_width, len(SEMANTIC_EFFECTS))
            self.progress_head = torch.nn.Linear(hidden_width, 1)

        @staticmethod
        def _mean_tokens(ids: Any, embeddings: Any) -> Any:
            token_mask = (ids != 0).to(embeddings.dtype)
            total = (embeddings * token_mask.unsqueeze(-1)).sum(dim=-2)
            denominator = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return total / denominator

        def forward(
            self,
            root_ids: Any,
            neighbor_ids: Any,
            neighbor_mask: Any,
        ) -> tuple[Any, Any]:
            root = self._mean_tokens(root_ids, self.embedding(root_ids))
            nodes = self._mean_tokens(neighbor_ids, self.embedding(neighbor_ids))
            nodes = self.node_encoder(nodes)
            mask = neighbor_mask.unsqueeze(-1)
            mean = (nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            masked = nodes.masked_fill(mask == 0, -1e4)
            maximum = masked.max(dim=1).values
            empty = neighbor_mask.sum(dim=1, keepdim=True) == 0
            maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
            latent = self.trunk(torch.cat((root, mean, maximum), dim=-1))
            return self.effect_head(latent), self.progress_head(latent).squeeze(-1)

    return CounterfactualPanelNet()


def _balanced_indices(
    records: Sequence[SemanticTeacherRecord],
    train_indices: np.ndarray,
    *,
    samples_per_game: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for game in sorted({records[int(index)].game_id for index in train_indices}):
        eligible = np.asarray(
            [
                int(index)
                for index in train_indices
                if records[int(index)].game_id == game
            ],
            dtype=np.int64,
        )
        rows.extend(
            rng.choice(
                eligible,
                size=samples_per_game,
                replace=len(eligible) < samples_per_game,
            ).tolist()
        )
    result = np.asarray(rows, dtype=np.int64)
    rng.shuffle(result)
    return result


def _masked_effect_loss(logits: Any, labels: Any, masks: Any) -> Any:
    import torch

    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    per_effect = []
    for effect_index in range(raw.shape[1]):
        mask = masks[:, effect_index]
        per_effect.append(
            (raw[:, effect_index] * mask).sum() / mask.sum().clamp_min(1.0)
        )
    return torch.stack(per_effect).mean()


def _fit_model(
    records: Sequence[SemanticTeacherRecord],
    tensors: Any,
    *,
    train_indices: np.ndarray,
    comparisons: Sequence[_Comparison],
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
        hidden_width=int(parameters["hidden_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    train_set = {int(index) for index in train_indices}
    usable = [
        row
        for row in comparisons
        if row.left in train_set and row.right in train_set
    ]
    started = time.perf_counter()
    final_losses: dict[str, float] = {}
    for epoch in range(int(parameters["epochs"])):
        model.train()
        selected = _balanced_indices(
            records,
            train_indices,
            samples_per_game=int(parameters["samples_per_game_per_epoch"]),
            seed=seed + epoch,
        )
        root, nodes, mask, labels, applicable = _batch_arrays(tensors, selected)
        root = root.to(device)
        nodes = nodes.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        applicable = applicable.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _progress = model(root, nodes, mask)
        absolute = _masked_effect_loss(logits, labels, applicable)

        rng = random.Random(seed + 10_000 + epoch)
        chosen = list(usable)
        rng.shuffle(chosen)
        chosen = chosen[: int(parameters["maximum_pairs_per_epoch"])]
        left_indices = np.asarray([row.left for row in chosen], dtype=np.int64)
        right_indices = np.asarray([row.right for row in chosen], dtype=np.int64)
        left = _batch_arrays(tensors, left_indices)
        right = _batch_arrays(tensors, right_indices)
        left_logits, left_progress = model(
            left[0].to(device), left[1].to(device), left[2].to(device)
        )
        right_logits, right_progress = model(
            right[0].to(device), right[1].to(device), right[2].to(device)
        )
        left_labels = left[3].to(device)
        right_labels = right[3].to(device)
        pair_mask = (
            left[4].to(device)
            * right[4].to(device)
            * (left_labels != right_labels).to(torch.float32)
        )
        pair_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            left_logits - right_logits,
            left_labels,
            reduction="none",
        )
        per_effect_pair = []
        for effect_index in range(pair_raw.shape[1]):
            effect_mask = pair_mask[:, effect_index]
            per_effect_pair.append(
                (pair_raw[:, effect_index] * effect_mask).sum()
                / effect_mask.sum().clamp_min(1.0)
            )
        effect_pair = torch.stack(per_effect_pair).mean()

        progress_targets = torch.as_tensor(
            [row.horizon_delta > 0.0 for row in chosen],
            dtype=torch.float32,
            device=device,
        )
        progress_mask = torch.as_tensor(
            [
                row.fresh and abs(row.horizon_delta) >= PROGRESS_DEADBAND
                for row in chosen
            ],
            dtype=torch.float32,
            device=device,
        )
        progress_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            left_progress - right_progress,
            progress_targets,
            reduction="none",
        )
        progress_pair = (
            (progress_raw * progress_mask).sum()
            / progress_mask.sum().clamp_min(1.0)
        )
        tie_mask = torch.as_tensor(
            [
                row.fresh and abs(row.horizon_delta) < PROGRESS_DEADBAND
                for row in chosen
            ],
            dtype=torch.float32,
            device=device,
        )
        tie_consistency = (
            ((left_progress - right_progress) ** 2 * tie_mask).sum()
            / tie_mask.sum().clamp_min(1.0)
        )
        loss = (
            absolute
            + float(parameters["effect_pair_weight"]) * effect_pair
            + float(parameters["progress_pair_weight"]) * progress_pair
            + float(parameters["tie_consistency_weight"]) * tie_consistency
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        final_losses = {
            "absolute": float(absolute.detach().cpu()),
            "effect_pair": float(effect_pair.detach().cpu()),
            "progress_pair": float(progress_pair.detach().cpu()),
            "tie_consistency": float(tie_consistency.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
    return model, {
        "runtime_seconds": time.perf_counter() - started,
        "train_rows": len(train_indices),
        "train_comparisons": len(usable),
        "final_losses": final_losses,
    }


def _predict_model(
    model: Any,
    tensors: Any,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    progress = []
    model.eval()
    import torch

    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = _batch_arrays(tensors, selected)
            logits, scores = model(
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            rows.append(logits.cpu().numpy())
            progress.append(scores.cpu().numpy())
    return np.concatenate(rows), np.concatenate(progress)


def _centered_residual(
    full_logits: np.ndarray,
    root_logits: np.ndarray,
    indices: np.ndarray,
    groups: Mapping[str, tuple[int, ...]],
) -> np.ndarray:
    residual = full_logits - root_logits
    lookup = {int(global_index): local for local, global_index in enumerate(indices)}
    centered = np.zeros_like(residual)
    assigned = set()
    for group in groups.values():
        local = [lookup[index] for index in group if index in lookup]
        if len(local) < 2:
            continue
        values = residual[local]
        centered[local] = values - values.mean(axis=0, keepdims=True)
        assigned.update(local)
    for local in range(len(indices)):
        if local not in assigned:
            centered[local] = 0.0
    return centered


def _game_balanced_shifts(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    logits: np.ndarray,
    *,
    excluded_game: str | None = None,
) -> np.ndarray:
    shifts = np.zeros(len(SEMANTIC_EFFECTS), dtype=np.float64)
    games = sorted(
        {
            records[int(index)].game_id
            for index in indices
            if records[int(index)].game_id != excluded_game
        }
    )
    for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
        targets = []
        predictions = []
        for game in games:
            selected = [
                local
                for local, global_index in enumerate(indices)
                if records[int(global_index)].game_id == game
                and records[int(global_index)].applicable[effect]
            ]
            if not selected:
                continue
            targets.append(
                float(
                    np.mean(
                        [
                            records[int(indices[local])].labels[effect]
                            for local in selected
                        ]
                    )
                )
            )
            predictions.append(float(np.mean(_sigmoid(logits[selected, effect_index]))))
        target = float(np.mean(targets)) if targets else 0.5
        predicted = float(np.mean(predictions)) if predictions else 0.5
        shifts[effect_index] = _logit(target) - _logit(predicted)
    return shifts


def _select_alpha_and_shifts(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    root_logits: np.ndarray,
    residual: np.ndarray,
    *,
    alpha_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    games = sorted({records[int(index)].game_id for index in indices})
    alphas = np.zeros(len(SEMANTIC_EFFECTS), dtype=np.float64)
    scores: dict[str, Any] = {}
    for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
        candidates = []
        for alpha in alpha_grid:
            logits = root_logits + float(alpha) * residual
            held_scores = []
            for held in games:
                shift = _game_balanced_shifts(
                    records,
                    indices,
                    logits,
                    excluded_game=held,
                )[effect_index]
                selected = [
                    local
                    for local, global_index in enumerate(indices)
                    if records[int(global_index)].game_id == held
                    and records[int(global_index)].applicable[effect]
                ]
                if not selected:
                    continue
                targets = np.asarray(
                    [
                        records[int(indices[local])].labels[effect]
                        for local in selected
                    ],
                    dtype=np.float64,
                )
                probabilities = _sigmoid(
                    logits[selected, effect_index] + shift
                )
                held_scores.append(float(np.mean((probabilities - targets) ** 2)))
            candidates.append((float(np.mean(held_scores)), float(alpha)))
        candidates.sort(key=lambda row: (row[0], row[1]))
        alphas[effect_index] = candidates[0][1]
        scores[effect] = {
            "alpha": candidates[0][1],
            "inner_game_balanced_brier": candidates[0][0],
        }
    combined = root_logits + residual * alphas.reshape(1, -1)
    shifts = _game_balanced_shifts(records, indices, combined)
    return alphas, shifts, scores


def _calibrated(
    root_logits: np.ndarray,
    residual: np.ndarray,
    alphas: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    return _sigmoid(
        root_logits
        + residual * alphas.reshape(1, -1)
        + shifts.reshape(1, -1)
    )


def _ece(
    records: Sequence[SemanticTeacherRecord],
    indices: Sequence[int],
    probabilities: np.ndarray,
    effects: Sequence[str],
    *,
    bins: int = 10,
) -> float:
    rows = []
    lookup = {int(global_index): local for local, global_index in enumerate(indices)}
    for effect in effects:
        effect_index = SEMANTIC_EFFECTS.index(effect)
        selected = [
            int(index)
            for index in indices
            if records[int(index)].applicable[effect]
        ]
        if not selected:
            continue
        target = np.asarray(
            [records[index].labels[effect] for index in selected],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [probabilities[lookup[index], effect_index] for index in selected],
            dtype=np.float64,
        )
        value = 0.0
        for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
            upper = lower + 1.0 / bins
            mask = (predicted >= lower) & (
                predicted <= upper if upper >= 1.0 else predicted < upper
            )
            if mask.any():
                value += float(mask.mean()) * abs(
                    float(predicted[mask].mean()) - float(target[mask].mean())
                )
        rows.append(value)
    return float(np.mean(rows)) if rows else 0.0


def _fresh_pair_rows(
    comparisons: Sequence[_Comparison],
    *,
    game: str | None = None,
) -> list[_Comparison]:
    return [
        row
        for row in comparisons
        if row.fresh
        and abs(row.horizon_delta) >= PROGRESS_DEADBAND
        and (game is None or row.game_id == game)
    ]


def _preference_metrics(
    comparisons: Sequence[_Comparison],
    scores: np.ndarray,
) -> dict[str, Any]:
    rows = _fresh_pair_rows(comparisons)
    if not rows:
        return {"pairs": 0, "log_loss": 0.0, "accuracy": 0.0, "per_game": {}}
    probabilities = np.asarray(
        [
            float(_sigmoid(np.asarray([scores[row.left] - scores[row.right]]))[0])
            for row in rows
        ],
        dtype=np.float64,
    )
    targets = np.asarray([row.horizon_delta > 0.0 for row in rows], dtype=np.float64)
    losses = -(
        targets * np.log(np.clip(probabilities, 1e-8, 1.0))
        + (1.0 - targets) * np.log(np.clip(1.0 - probabilities, 1e-8, 1.0))
    )
    per_game = {}
    for game in SOURCE_TRAIN:
        selected = [index for index, row in enumerate(rows) if row.game_id == game]
        if not selected:
            continue
        per_game[game] = {
            "pairs": len(selected),
            "log_loss": float(np.mean(losses[selected])),
            "accuracy": float(
                np.mean((probabilities[selected] >= 0.5) == targets[selected])
            ),
        }
    return {
        "pairs": len(rows),
        "log_loss": float(np.mean(losses)),
        "accuracy": float(np.mean((probabilities >= 0.5) == targets)),
        "per_game": per_game,
    }


def _top1_regret(
    teacher_panels: Sequence[TeacherPanel],
    by_digest: Mapping[str, int],
    scores: np.ndarray,
) -> dict[str, Any]:
    regrets = {}
    for panel in teacher_panels:
        indices = [by_digest[arm.trace_digest] for arm in panel.arms]
        chosen = max(indices, key=lambda index: (scores[index], -index))
        returns = {by_digest[arm.trace_digest]: arm.horizon_return for arm in panel.arms}
        regrets[panel.panel_id] = {
            "game_id": panel.game_id,
            "regret": float(max(returns.values()) - returns[chosen]),
        }
    return {
        "panels": len(regrets),
        "mean_regret": float(np.mean([row["regret"] for row in regrets.values()])),
        "per_game": {
            game: float(
                np.mean(
                    [
                        row["regret"]
                        for row in regrets.values()
                        if row["game_id"] == game
                    ]
                )
            )
            for game in SOURCE_TRAIN
        },
        "rows": regrets,
    }


def _bootstrap_panel_difference(
    rows: Mapping[str, Mapping[str, float | str]],
    *,
    left_key: str,
    right_key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in rows.values():
        by_game[str(row["game_id"])].append(
            float(row[left_key]) - float(row[right_key])
        )
    rng = np.random.default_rng(seed)
    draws = []
    games = sorted(by_game)
    for _ in range(samples):
        values = []
        for game in games:
            game_rows = np.asarray(by_game[game], dtype=np.float64)
            selected = rng.choice(game_rows, size=len(game_rows), replace=True)
            values.append(float(np.mean(selected)))
        draws.append(float(np.mean(values)))
    return {
        "mean": float(
            np.mean([value for game in games for value in by_game[game]])
        ),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }


def _pair_loss_rows(
    comparisons: Sequence[_Comparison],
    variants: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | str]]:
    result = {}
    for row in _fresh_pair_rows(comparisons):
        target = float(row.horizon_delta > 0.0)
        payload: dict[str, float | str] = {"game_id": row.game_id}
        for name, scores in variants.items():
            probability = float(
                _sigmoid(np.asarray([scores[row.left] - scores[row.right]]))[0]
            )
            payload[name] = float(
                -target * math.log(max(probability, 1e-8))
                - (1.0 - target) * math.log(max(1.0 - probability, 1e-8))
            )
        result[f"{row.panel_id}:{row.left}:{row.right}"] = payload
    return result


def _panel_brier_rows(
    teacher_panels: Sequence[TeacherPanel],
    by_digest: Mapping[str, int],
    records: Sequence[SemanticTeacherRecord],
    variants: Mapping[str, np.ndarray],
    eligible_effects: Sequence[str],
) -> dict[str, dict[str, float | str]]:
    result = {}
    for panel in teacher_panels:
        payload: dict[str, float | str] = {"game_id": panel.game_id}
        indices = [by_digest[arm.trace_digest] for arm in panel.arms]
        for name, matrix in variants.items():
            values = []
            for index in indices:
                for effect in eligible_effects:
                    if not records[index].applicable[effect]:
                        continue
                    effect_index = SEMANTIC_EFFECTS.index(effect)
                    values.append(
                        (
                            float(matrix[index, effect_index])
                            - float(records[index].labels[effect])
                        )
                        ** 2
                    )
            payload[name] = float(np.mean(values)) if values else 0.0
        result[panel.panel_id] = payload
    return result


def _identity_predictions(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    features: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    labels = np.asarray([records[int(index)].game_id for index in indices])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = LogisticRegression(max_iter=600, solver="lbfgs", random_state=seed)
    predictions = cross_val_predict(model, features, labels, cv=folds, method="predict")
    return labels, predictions


def _identity_metric(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    features: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    labels, predictions = _identity_predictions(
        records,
        indices,
        features,
        seed=seed,
    )
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    correct = predictions == labels
    return (
        {
            "majority_accuracy": float(majority),
            "accuracy": float(np.mean(correct)),
            "gain_over_majority": float(np.mean(correct) - majority),
        },
        correct.astype(np.float64),
    )


def _bootstrap_identity_increment(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    full_correct: np.ndarray,
    root_correct: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rows = {
        str(local): {
            "game_id": records[int(index)].game_id,
            "full": float(full_correct[local]),
            "root": float(root_correct[local]),
        }
        for local, index in enumerate(indices)
    }
    return _bootstrap_panel_difference(
        rows,
        left_key="full",
        right_key="root",
        samples=samples,
        seed=seed,
    )


def _reverse_neighbor_tensors(tensors: Any) -> Any:
    return replace(
        tensors,
        neighbor_ids=tensors.neighbor_ids[:, ::-1].copy(),
        neighbor_mask=tensors.neighbor_mask[:, ::-1].copy(),
    )


def _shuffled_records(
    records: Sequence[SemanticTeacherRecord],
) -> tuple[SemanticTeacherRecord, ...]:
    return tuple(
        replace(
            record,
            graph=ObjectRelativeGraph(
                root=dict(record.graph.root),
                neighbors=tuple(
                    {
                        **dict(neighbor),
                        "axis_relation": _AXIS_SHUFFLE[
                            str(neighbor["axis_relation"])
                        ],
                    }
                    for neighbor in record.graph.neighbors
                ),
            ),
        )
        for record in records
    )


def _action_progress_scores(
    records: Sequence[SemanticTeacherRecord],
    teacher_panels: Sequence[TeacherPanel],
    train_games: set[str],
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for panel in teacher_panels:
        if panel.game_id not in train_games:
            continue
        for arm in panel.arms:
            values[arm.action_name].append(arm.horizon_return)
    fallback = float(np.mean([item for rows in values.values() for item in rows]))
    return {
        record.trace_digest: float(
            np.mean(values.get(str(record.graph.root.get("action_name")), [fallback]))
        )
        for record in records
    }


def evaluate_student(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa.get("teacher_ready"):
        result = {
            "format_version": RESULT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "teacher_qa_checksum": qa["qa_checksum"],
            "verdict": "COMPARATIVE_CAUSAL_TEACHER_CAPACITY_FAILED",
            "teacher_ready": False,
            "authority_promoted": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "historical_opened": False,
            "live_environment_opened": False,
        }
        result["result_checksum"] = _checksum(result)
        _write_json(destination / "student_result.json", result)
        return result

    teacher_panels = load_teacher_panels(destination)
    records, comparisons, groups, fresh_set = _assemble_training_data(teacher_panels)
    by_digest = {record.trace_digest: index for index, record in enumerate(records)}
    parameters = manifest["training"]
    selected_device = _select_device(device)
    maximum_neighbors = 16
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
    shuffled_tensors = tensorize_records(
        _shuffled_records(records),
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
    )
    reversed_tensors = _reverse_neighbor_tensors(tensors)

    count = len(records)
    effect_count = len(SEMANTIC_EFFECTS)
    full_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    root_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    action_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    shuffled_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    reversed_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    residual_features = np.zeros((count, effect_count), dtype=np.float64)
    full_progress = np.zeros(count, dtype=np.float64)
    root_progress = np.zeros(count, dtype=np.float64)
    shuffled_progress = np.zeros(count, dtype=np.float64)
    action_progress = np.zeros(count, dtype=np.float64)
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
        root_model, root_summary = _fit_model(
            records,
            root_tensors,
            train_indices=train_indices,
            comparisons=comparisons,
            parameters=parameters,
            device=selected_device,
            seed=int(parameters["seed"]) + fold_index * 100,
        )
        full_model, full_summary = _fit_model(
            records,
            tensors,
            train_indices=train_indices,
            comparisons=comparisons,
            parameters=parameters,
            device=selected_device,
            seed=int(parameters["seed"]) + fold_index * 100 + 1,
        )
        train_root, _train_root_progress = _predict_model(
            root_model,
            root_tensors,
            train_indices,
            device=selected_device,
        )
        train_full, _train_full_progress = _predict_model(
            full_model,
            tensors,
            train_indices,
            device=selected_device,
        )
        train_residual = _centered_residual(
            train_full,
            train_root,
            train_indices,
            groups,
        )
        alphas, shifts, alpha_summary = _select_alpha_and_shifts(
            records,
            train_indices,
            train_root,
            train_residual,
            alpha_grid=parameters["alpha_grid"],
        )
        root_shifts = _game_balanced_shifts(
            records,
            train_indices,
            train_root,
        )
        test_root, test_root_progress = _predict_model(
            root_model,
            root_tensors,
            test_indices,
            device=selected_device,
        )
        test_full, test_full_progress = _predict_model(
            full_model,
            tensors,
            test_indices,
            device=selected_device,
        )
        test_shuffle, test_shuffle_progress = _predict_model(
            full_model,
            shuffled_tensors,
            test_indices,
            device=selected_device,
        )
        test_reversed, _test_reversed_progress = _predict_model(
            full_model,
            reversed_tensors,
            test_indices,
            device=selected_device,
        )
        test_residual = _centered_residual(
            test_full,
            test_root,
            test_indices,
            groups,
        )
        shuffle_residual = _centered_residual(
            test_shuffle,
            test_root,
            test_indices,
            groups,
        )
        reversed_residual = _centered_residual(
            test_reversed,
            test_root,
            test_indices,
            groups,
        )
        full_probabilities[test_indices] = _calibrated(
            test_root,
            test_residual,
            alphas,
            shifts,
        )
        shuffled_probabilities[test_indices] = _calibrated(
            test_root,
            shuffle_residual,
            alphas,
            shifts,
        )
        reversed_probabilities[test_indices] = _calibrated(
            test_root,
            reversed_residual,
            alphas,
            shifts,
        )
        root_probabilities[test_indices] = _sigmoid(
            test_root + root_shifts.reshape(1, -1)
        )
        action_probabilities[test_indices] = _action_only_probabilities(
            records,
            train_indices,
            test_indices,
        )
        residual_features[test_indices] = test_residual
        full_progress[test_indices] = test_full_progress
        root_progress[test_indices] = test_root_progress
        shuffled_progress[test_indices] = test_shuffle_progress
        action_lookup = _action_progress_scores(
            records,
            teacher_panels,
            {game for game in SOURCE_TRAIN if game != held_out_game},
        )
        action_progress[test_indices] = [
            action_lookup[records[int(index)].trace_digest] for index in test_indices
        ]
        folds.append(
            {
                "held_out_game": held_out_game,
                "root_only": root_summary,
                "full": full_summary,
                "alpha_selection": alpha_summary,
            }
        )

    fresh_indices = np.asarray(sorted(fresh_set), dtype=np.int64)
    prediction_rows = []
    for index, record in enumerate(records):
        prediction_rows.append(
            {
                "format_version": PREDICTION_VERSION,
                "trace_digest": record.trace_digest,
                "example_id": record.example_id,
                "game_id": record.game_id,
                "fresh_panel_arm": index in fresh_set,
                "probabilities": {
                    variant: {
                        effect: float(matrix[index, effect_index])
                        for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                    }
                    for variant, matrix in (
                        ("comparative_distilled", full_probabilities),
                        ("root_only", root_probabilities),
                        ("action_only", action_probabilities),
                        ("relation_shuffle", shuffled_probabilities),
                    )
                },
                "progress_scores": {
                    "comparative": float(full_progress[index]),
                    "root_only": float(root_progress[index]),
                    "action_only": float(action_progress[index]),
                    "relation_shuffle": float(shuffled_progress[index]),
                },
            }
        )
    prediction_path = destination / "logo_predictions.jsonl"
    _write_jsonl(prediction_path, prediction_rows)

    preference_variants = {
        "comparative": full_progress,
        "root_only": root_progress,
        "action_only": action_progress,
        "relation_shuffle": shuffled_progress,
    }
    preference = {
        name: _preference_metrics(comparisons, scores)
        for name, scores in preference_variants.items()
    }
    pair_rows = _pair_loss_rows(comparisons, preference_variants)
    bootstrap_samples = int(manifest["evaluation"]["bootstrap_samples"])
    bootstrap_seed = int(manifest["evaluation"]["bootstrap_seed"])
    preference_gain_root = _bootstrap_panel_difference(
        pair_rows,
        left_key="root_only",
        right_key="comparative",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    preference_gain_action = _bootstrap_panel_difference(
        pair_rows,
        left_key="action_only",
        right_key="comparative",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    shuffle_pair_degradation = _bootstrap_panel_difference(
        pair_rows,
        left_key="relation_shuffle",
        right_key="comparative",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
    )
    top1 = {
        name: _top1_regret(teacher_panels, by_digest, scores)
        for name, scores in preference_variants.items()
    }
    regret_rows = {
        panel.panel_id: {
            "game_id": panel.game_id,
            "root": top1["root_only"]["rows"][panel.panel_id]["regret"],
            "full": top1["comparative"]["rows"][panel.panel_id]["regret"],
        }
        for panel in teacher_panels
    }
    regret_reduction = _bootstrap_panel_difference(
        regret_rows,
        left_key="root",
        right_key="full",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 3,
    )

    eligible_effects = tuple(str(row) for row in qa["eligible_effects"])
    fresh_records = tuple(records[int(index)] for index in fresh_indices)
    fresh_full = full_probabilities[fresh_indices]
    fresh_root = root_probabilities[fresh_indices]
    fresh_action = action_probabilities[fresh_indices]
    fresh_shuffle = shuffled_probabilities[fresh_indices]
    absolute_metrics = {
        "comparative_distilled": _brier_metrics(fresh_records, fresh_full),
        "root_only": _brier_metrics(fresh_records, fresh_root),
        "action_only": _brier_metrics(fresh_records, fresh_action),
        "relation_shuffle": _brier_metrics(fresh_records, fresh_shuffle),
    }
    brier_rows = _panel_brier_rows(
        teacher_panels,
        by_digest,
        records,
        {
            "full": full_probabilities,
            "root": root_probabilities,
            "shuffle": shuffled_probabilities,
        },
        eligible_effects,
    )
    brier_gain = _bootstrap_panel_difference(
        brier_rows,
        left_key="root",
        right_key="full",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 4,
    )
    shuffle_brier_degradation = _bootstrap_panel_difference(
        brier_rows,
        left_key="shuffle",
        right_key="full",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 5,
    )
    full_ece = _ece(
        records,
        fresh_indices,
        fresh_full,
        eligible_effects,
    )
    root_ece = _ece(
        records,
        fresh_indices,
        fresh_root,
        eligible_effects,
    )
    per_game_full = _per_game_brier(fresh_records, fresh_full)
    per_game_root = _per_game_brier(fresh_records, fresh_root)
    nonnegative_pair_games = sum(
        preference["comparative"]["per_game"].get(game, {}).get(
            "log_loss", float("inf")
        )
        <= preference["root_only"]["per_game"].get(game, {}).get(
            "log_loss", float("-inf")
        )
        for game in SOURCE_TRAIN
    )
    nonnegative_absolute_games = sum(
        per_game_full[game]["macro_brier"] <= per_game_root[game]["macro_brier"]
        for game in SOURCE_TRAIN
    )
    full_identity, full_correct = _identity_metric(
        records,
        fresh_indices,
        fresh_full,
        seed=5_111,
    )
    root_identity, root_correct = _identity_metric(
        records,
        fresh_indices,
        fresh_root,
        seed=5_111,
    )
    identity_increment = _bootstrap_identity_increment(
        records,
        fresh_indices,
        full_correct,
        root_correct,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 6,
    )
    permutation_delta = float(
        np.max(np.abs(full_probabilities - reversed_probabilities))
    )
    arm_swap_error = 0.0
    for row in _fresh_pair_rows(comparisons):
        forward = float(
            _sigmoid(np.asarray([full_progress[row.left] - full_progress[row.right]]))[0]
        )
        reverse = float(
            _sigmoid(np.asarray([full_progress[row.right] - full_progress[row.left]]))[0]
        )
        arm_swap_error = max(arm_swap_error, abs(forward + reverse - 1.0))

    terminal_capacity = bool(qa["terminal_capacity"]["eligible"])
    completion = _completion_recall_at_8(records, full_probabilities)
    thresholds = manifest["evaluation"]["decision_thresholds"]
    checks = {
        "teacher_ready": True,
        "preference_gain_over_root_ci": preference_gain_root["ci_lower"] > 0.0,
        "preference_gain_over_action_ci": preference_gain_action["ci_lower"] > 0.0,
        "top1_regret_reduction_ci": regret_reduction["ci_lower"] > 0.0,
        "relation_shuffle_pair_degradation_ci": (
            shuffle_pair_degradation["ci_lower"] > 0.0
        ),
        "pair_nonnegative_games": (
            nonnegative_pair_games >= int(thresholds["nonnegative_games_minimum"])
        ),
        "absolute_brier_gain_ci": brier_gain["ci_lower"] > 0.0,
        "absolute_ece_not_worse": full_ece <= root_ece,
        "relation_shuffle_brier_degradation": (
            shuffle_brier_degradation["mean"] > 0.0
        ),
        "absolute_nonnegative_games": (
            nonnegative_absolute_games
            >= int(thresholds["nonnegative_games_minimum"])
        ),
        "identity_increment": (
            identity_increment["ci_upper"]
            <= float(thresholds["identity_increment_ci_upper_maximum"])
        ),
        "neighbor_permutation_invariance": (
            permutation_delta
            <= float(thresholds["neighbor_permutation_max_probability_delta"])
        ),
        "arm_swap_antisymmetry": (
            arm_swap_error
            <= float(thresholds["arm_swap_max_complement_error"])
        ),
        "terminal_when_eligible": (
            not terminal_capacity
            or completion["recall_at_8"]
            >= float(thresholds["terminal_recall_at_8_minimum_when_eligible"])
        ),
    }
    comparative_keys = (
        "teacher_ready",
        "preference_gain_over_root_ci",
        "preference_gain_over_action_ci",
        "top1_regret_reduction_ci",
        "relation_shuffle_pair_degradation_ci",
        "pair_nonnegative_games",
        "arm_swap_antisymmetry",
    )
    absolute_keys = (
        "absolute_brier_gain_ci",
        "absolute_ece_not_worse",
        "relation_shuffle_brier_degradation",
        "absolute_nonnegative_games",
        "identity_increment",
        "neighbor_permutation_invariance",
        "terminal_when_eligible",
    )
    comparative_supported = all(checks[key] for key in comparative_keys)
    absolute_supported = all(checks[key] for key in absolute_keys)
    verdict = (
        "READY_FOR_SOURCE_WORLD_MODEL_PILOT"
        if comparative_supported and absolute_supported
        else "COMPARATIVE_SUPPORTED_ABSOLUTE_DISTILLATION_FAILED"
        if comparative_supported
        else "COMPARATIVE_CAUSAL_SEMANTICS_NOT_SUPPORTED"
    )
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "teacher_qa_checksum": qa["qa_checksum"],
        "verdict": verdict,
        "comparative_supported": comparative_supported,
        "absolute_distillation_supported": absolute_supported,
        "authority_promoted": False,
        "device": selected_device,
        "records": len(records),
        "fresh_panel_arms": len(fresh_indices),
        "eligible_effects": list(eligible_effects),
        "checks": checks,
        "preference": preference,
        "preference_gain_over_root": preference_gain_root,
        "preference_gain_over_action": preference_gain_action,
        "relation_shuffle_pair_degradation": shuffle_pair_degradation,
        "top1_regret": {
            name: {
                "panels": row["panels"],
                "mean_regret": row["mean_regret"],
                "per_game": row["per_game"],
            }
            for name, row in top1.items()
        },
        "top1_regret_reduction": regret_reduction,
        "absolute_metrics": absolute_metrics,
        "absolute_brier_gain_over_root": brier_gain,
        "relation_shuffle_brier_degradation": shuffle_brier_degradation,
        "absolute_ece": {
            "comparative_distilled": full_ece,
            "root_only": root_ece,
        },
        "nonnegative_games": {
            "preference": nonnegative_pair_games,
            "absolute": nonnegative_absolute_games,
        },
        "identity_probe": {
            "comparative_distilled": full_identity,
            "root_only": root_identity,
            "increment_bootstrap": identity_increment,
        },
        "neighbor_permutation_max_probability_delta": permutation_delta,
        "arm_swap_max_complement_error": arm_swap_error,
        "terminal_capacity_eligible": terminal_capacity,
        "completion_recall_at_8": completion,
        "folds": folds,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
        "world_model_fitted": False,
        "ebm_fitted": False,
        "artifact_sha256": {
            "teacher_panels": _file_sha256(destination / "teacher_panels.jsonl"),
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
                probabilities = lookup.get(arm.trace.trace_digest)
                if probabilities is None:
                    raise ValueError("V4.3 slot lacks V4.11 LOGO prediction")
                for variant, source in (
                    (
                        "comparative_distilled",
                        "counterfactual_panel_logo_v4_11",
                    ),
                    (
                        "relation_shuffle",
                        "counterfactual_panel_relation_shuffle_logo_v4_11",
                    ),
                ):
                    annotation = SlotAnnotation(
                        slot_id=item.slot.slot_id,
                        effect_probabilities={
                            effect: float(probabilities[variant][effect])
                            for effect in SLOT_EFFECTS
                        },
                        source=source,
                        support=0,
                    )
                    rows.append(
                        {
                            "format_version": SLOT_EXPORT_VERSION,
                            "slot_id": annotation.slot_id,
                            "example_id": item.example_id,
                            "game_id": item.game_id,
                            "variant": variant,
                            "effect_probabilities": dict(
                                annotation.effect_probabilities
                            ),
                            "source": annotation.source,
                            "support": annotation.support,
                        }
                    )
    path = destination / "v4_7_slot_annotations.jsonl"
    _write_jsonl(path, rows)
    summary: dict[str, Any] = {
        "format_version": SLOT_EXPORT_VERSION,
        "slots": len(examples),
        "rows": len(rows),
        "variants": ["comparative_distilled", "relation_shuffle"],
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
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "compile":
        payload = compile_teacher_panels(output_dir=args.output_dir)
    else:
        payload = evaluate_student(output_dir=args.output_dir, device=args.device)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CounterfactualPanel",
    "PanelArm",
    "SemanticPanelPrediction",
    "TeacherArm",
    "TeacherPanel",
    "compile_teacher_panels",
    "evaluate_student",
    "export_v47_annotations",
    "freeze_manifest",
    "load_manifest",
    "load_raw_panels",
    "load_teacher_panels",
]
