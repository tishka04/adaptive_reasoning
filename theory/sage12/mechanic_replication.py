"""Clean causal replication of the SAGE12 temporal mechanic pilot (V4.1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION

from . import mechanic_induction as v4
from .action_target_data import (
    EFFECT_LABELS,
    ActionTargetTrace,
    build_observation,
    conservative_match_objects,
    grid_sha256,
)
from .llm import TransformersJSONModel, TransformersModelConfig

FORMAT_VERSION = "sage12-mechanic-window-v4.1"
PREFLIGHT_FORMAT_VERSION = "sage12-mechanic-preflight-v4.1"
RESULT_FORMAT_VERSION = "sage12-mechanic-pilot-result-v4.1"
CALIBRATION_FORMAT_VERSION = "sage12-mechanic-calibration-v4.1"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "mechanic_induction_v4_1"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
V3_OUTPUT_DIR = Path("training") / "sage12" / "action_target_pilot_v3"
MODEL_MODES = (
    "structured",
    "context_ablation",
    "local_action",
    "global_action",
    "template",
)
BASELINE_MODES = ("local_action", "global_action", "template")
ROLE_STATES = ("translational", "non_translational", "ambiguous")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class ActorRoleState(str, Enum):
    """Causal reset-local status of the action-controlled role."""

    TRANSLATIONAL = "translational"
    NON_TRANSLATIONAL = "non_translational"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class SemanticTransitionEvent:
    """One outcome-bearing context event with audit-only role state."""

    action_name: str
    action_family: str
    anchor_condition: str
    effects: Mapping[str, bool]
    applicable: Mapping[str, bool]
    actor_role_known: bool
    actor_role_state: str

    def __post_init__(self) -> None:
        if self.anchor_condition not in v4.ANCHOR_CONDITIONS:
            raise ValueError("unsupported V4.1 anchor condition")
        if self.actor_role_state not in ROLE_STATES:
            raise ValueError("unsupported V4.1 actor role state")
        if set(self.effects) != set(EFFECT_LABELS):
            raise ValueError("V4.1 event requires complete effects")
        if set(self.applicable) != set(EFFECT_LABELS):
            raise ValueError("V4.1 event requires complete applicability masks")

    def model_view(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_family": self.action_family,
            "anchor_condition": self.anchor_condition,
            "effects": {
                label: bool(self.effects[label]) for label in EFFECT_LABELS
            },
            "applicable": {
                label: bool(self.applicable[label]) for label in EFFECT_LABELS
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.model_view(),
            "actor_role_known": bool(self.actor_role_known),
            "actor_role_state": self.actor_role_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SemanticTransitionEvent:
        effects = dict(payload.get("effects", {}))
        applicable = dict(payload.get("applicable", {}))
        state = str(payload.get("actor_role_state", ActorRoleState.AMBIGUOUS.value))
        return cls(
            action_name=str(payload["action_name"]),
            action_family=str(payload["action_family"]),
            anchor_condition=str(payload["anchor_condition"]),
            effects={label: bool(effects.get(label, False)) for label in EFFECT_LABELS},
            applicable={
                label: bool(applicable.get(label, False)) for label in EFFECT_LABELS
            },
            actor_role_known=bool(payload.get("actor_role_known", False)),
            actor_role_state=state,
        )


@dataclass(frozen=True)
class MechanicWindowRecord:
    """Eight causal observations followed by one outcome-blind query."""

    game_id: str
    source_split: str
    policy_seed: int
    reset_index: int
    query_step_index: int
    context: tuple[SemanticTransitionEvent, ...]
    query: v4.MechanicQuery
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    actor_role_known: bool
    actor_role_state: str
    window_digest: str = ""
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 V4.1 window format")
        if self.actor_role_state not in ROLE_STATES:
            raise ValueError("unsupported V4.1 query role state")
        if set(self.labels) != set(EFFECT_LABELS):
            raise ValueError("V4.1 window requires complete labels")
        if set(self.applicable) != set(EFFECT_LABELS):
            raise ValueError("V4.1 window requires complete masks")
        if not self.window_digest:
            object.__setattr__(
                self,
                "window_digest",
                _checksum(
                    {
                        "context": [item.model_view() for item in self.context],
                        "query": self.query.to_dict(),
                    }
                ),
            )

    @property
    def run_key(self) -> str:
        return f"{self.game_id}:{self.policy_seed}:{self.reset_index}"

    def model_view(self) -> dict[str, Any]:
        return {
            "context": [item.model_view() for item in self.context],
            "query": self.query.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "game_id": self.game_id,
            "source_split": self.source_split,
            "policy_seed": int(self.policy_seed),
            "reset_index": int(self.reset_index),
            "query_step_index": int(self.query_step_index),
            "context": [item.to_dict() for item in self.context],
            "query": self.query.to_dict(),
            "labels": {label: bool(self.labels[label]) for label in EFFECT_LABELS},
            "applicable": {
                label: bool(self.applicable[label]) for label in EFFECT_LABELS
            },
            "actor_role_known": bool(self.actor_role_known),
            "actor_role_state": self.actor_role_state,
            "window_digest": self.window_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MechanicWindowRecord:
        labels = dict(payload.get("labels", {}))
        applicable = dict(payload.get("applicable", {}))
        return cls(
            format_version=str(payload.get("format_version", FORMAT_VERSION)),
            game_id=str(payload["game_id"]),
            source_split=str(payload["source_split"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            query_step_index=int(payload["query_step_index"]),
            context=tuple(
                SemanticTransitionEvent.from_dict(item)
                for item in payload.get("context", ())
            ),
            query=v4.MechanicQuery.from_dict(dict(payload["query"])),
            labels={label: bool(labels.get(label, False)) for label in EFFECT_LABELS},
            applicable={
                label: bool(applicable.get(label, False)) for label in EFFECT_LABELS
            },
            actor_role_known=bool(payload.get("actor_role_known", False)),
            actor_role_state=str(
                payload.get("actor_role_state", ActorRoleState.AMBIGUOUS.value)
            ),
            window_digest=str(payload.get("window_digest", "")),
        )


def _shape_signature(item: Any) -> tuple[int, int, str]:
    """Stable internal-only component signature; never enters the model view."""
    r0, c0, r1, c1 = item.bbox
    height = max(1, int(r1) - int(r0) + 1)
    width = max(1, int(c1) - int(c0) + 1)
    aspect = "wide" if width >= 1.5 * height else "tall" if height >= 1.5 * width else "square"
    return int(item.value), int(item.area), aspect


class CausalRoleTracker:
    """Reset-local online role resolver with an explicit non-translation state."""

    def __init__(self) -> None:
        self._candidate_counts: Counter[tuple[int, int, str]] = Counter()
        self._actor_signature: tuple[int, int, str] | None = None
        self._previous_player_hypotheses: tuple[Any, ...] = ()
        self._steps = 0
        self._move_non_noops = 0
        self._missing_actor_steps = 0

    def reset(self) -> None:
        self.__init__()

    def observe(self, trace: ActionTargetTrace) -> SemanticTransitionEvent:
        before = build_observation(
            trace.frame_before,
            available_actions=trace.available_action_names,
            game_state=trace.game_state_before,
            levels_completed=trace.levels_completed_before,
            infer_players=True,
            prev_player_hypotheses=self._previous_player_hypotheses,
        )
        after = build_observation(
            trace.frame_after,
            available_actions=trace.available_action_names,
            game_state=trace.game_state_after,
            levels_completed=trace.levels_completed_after,
            infer_players=True,
            prev_player_hypotheses=before.player_candidates,
        )
        self._previous_player_hypotheses = tuple(after.player_candidates)
        matched = conservative_match_objects(before.objects, after.objects)
        before_by_id = {item.object_id: item for item in before.objects}
        after_by_id = {item.object_id: item for item in after.objects}
        maximum_area = max(1, int(np.asarray(trace.frame_before).size * 0.25))

        moving_signatures: list[tuple[int, int, str]] = []
        for left_id, right_id in matched.matched.items():
            left = before_by_id[left_id]
            right = after_by_id[right_id]
            if left.area > maximum_area or right.area > maximum_area:
                continue
            delta = (
                float(right.center[0]) - float(left.center[0]),
                float(right.center[1]) - float(left.center[1]),
            )
            if not np.allclose(delta, (0.0, 0.0)):
                moving_signatures.append(_shape_signature(left))

        unique_movers = [
            signature
            for signature, count in Counter(moving_signatures).items()
            if count == 1
        ]
        for signature in unique_movers:
            self._candidate_counts[signature] += 1

        if trace.anchor.target_object_id is not None:
            target = before_by_id.get(trace.anchor.target_object_id)
            if target is not None and target.area <= maximum_area:
                self._candidate_counts[_shape_signature(target)] += 2

        if before.best_player is not None and before.best_player.confidence >= 0.4:
            candidates = [
                item
                for item in before.objects
                if int(item.value) == int(before.best_player.value)
                and tuple(map(round, item.center))
                == tuple(map(round, before.best_player.position))
            ]
            if len(candidates) == 1 and candidates[0].area <= maximum_area:
                self._candidate_counts[_shape_signature(candidates[0])] += 2

        if self._actor_signature is None and self._candidate_counts:
            ranked = self._candidate_counts.most_common(2)
            lead = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)
            if ranked[0][1] >= 2 and lead >= 2:
                self._actor_signature = ranked[0][0]

        actor_applicable = False
        actor_displaced = False
        if self._actor_signature is not None:
            actor_pairs = [
                (before_by_id[left_id], after_by_id[right_id])
                for left_id, right_id in matched.matched.items()
                if _shape_signature(before_by_id[left_id]) == self._actor_signature
            ]
            if len(actor_pairs) == 1:
                actor_applicable = True
                left, right = actor_pairs[0]
                actor_displaced = not np.allclose(left.center, right.center)
                self._missing_actor_steps = 0
            else:
                self._missing_actor_steps += 1
                if self._missing_actor_steps >= 2:
                    self._actor_signature = None

        self._steps += 1
        if trace.anchor.action_family == "move" and not np.array_equal(
            np.asarray(trace.frame_before),
            np.asarray(trace.frame_after),
        ):
            self._move_non_noops += 1
        if self._actor_signature is not None:
            role_state = ActorRoleState.TRANSLATIONAL
        elif self._steps >= 8 and self._move_non_noops >= 3:
            role_state = ActorRoleState.NON_TRANSLATIONAL
        else:
            role_state = ActorRoleState.AMBIGUOUS

        effects = {
            label: bool(trace.effects.labels[label]) for label in EFFECT_LABELS
        }
        applicable = {
            label: bool(trace.effects.applicable[label]) for label in EFFECT_LABELS
        }
        effects["actor_displaced"] = bool(actor_displaced)
        applicable["actor_displaced"] = bool(actor_applicable)
        resolved = role_state is not ActorRoleState.AMBIGUOUS
        return SemanticTransitionEvent(
            action_name=trace.selected_action_name,
            action_family=trace.anchor.action_family,
            anchor_condition=self._anchor_condition(trace, before),
            effects=effects,
            applicable=applicable,
            actor_role_known=resolved,
            actor_role_state=role_state.value,
        )

    def _anchor_condition(self, trace: ActionTargetTrace, before: Any) -> str:
        anchor = trace.anchor
        if anchor.kind == "targetless":
            return "targetless"
        if anchor.kind == "clicked_empty":
            return "empty"
        if anchor.path_status == "open":
            return "open"
        if not anchor.occupied:
            return "empty"
        if anchor.target_object_id is None:
            return "unknown"
        target = next(
            (
                item
                for item in before.objects
                if item.object_id == anchor.target_object_id
            ),
            None,
        )
        if (
            target is not None
            and self._actor_signature is not None
            and _shape_signature(target) == self._actor_signature
        ):
            return "occupied_actor"
        return "occupied_object"


def build_mechanic_windows(
    traces: Sequence[ActionTargetTrace],
    *,
    context_length: int = 8,
) -> list[MechanicWindowRecord]:
    """Build causal windows without crossing a game, reset, or frame gap."""
    grouped: dict[tuple[str, int, int], list[ActionTargetTrace]] = defaultdict(list)
    for trace in traces:
        grouped[(trace.game_id, trace.policy_seed, trace.reset_index)].append(trace)
    windows: list[MechanicWindowRecord] = []
    seen: set[str] = set()
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: item.step_index)
        runs: list[list[ActionTargetTrace]] = []
        current: list[ActionTargetTrace] = []
        for row in rows:
            contiguous = bool(current) and (
                row.step_index == current[-1].step_index + 1
                and grid_sha256(current[-1].frame_after)
                == grid_sha256(row.frame_before)
            )
            if current and not contiguous:
                runs.append(current)
                current = []
            current.append(row)
        if current:
            runs.append(current)
        for run in runs:
            tracker = CausalRoleTracker()
            events = [tracker.observe(row) for row in run]
            for query_index in range(int(context_length), len(run)):
                row = run[query_index]
                event = events[query_index]
                digest = _checksum(
                    {
                        "trace_digests": [
                            item.trace_digest
                            for item in run[
                                query_index - context_length : query_index + 1
                            ]
                        ]
                    }
                )
                if digest in seen:
                    continue
                seen.add(digest)
                windows.append(
                    MechanicWindowRecord(
                        game_id=row.game_id,
                        source_split=row.source_split,
                        policy_seed=row.policy_seed,
                        reset_index=row.reset_index,
                        query_step_index=row.step_index,
                        context=tuple(
                            events[query_index - context_length : query_index]
                        ),
                        query=v4.MechanicQuery(
                            action_name=event.action_name,
                            action_family=event.action_family,
                            anchor_condition=event.anchor_condition,
                        ),
                        labels=event.effects,
                        applicable=event.applicable,
                        actor_role_known=event.actor_role_known,
                        actor_role_state=event.actor_role_state,
                        window_digest=digest,
                    )
                )
    return windows


def validate_model_view(window: MechanicWindowRecord) -> None:
    rendered = _canonical(window.model_view()).lower()
    forbidden = (
        window.game_id.lower(),
        "game_id",
        "policy_seed",
        "reset_index",
        "query_step_index",
        "frame_before",
        "frame_after",
        "grid",
        "trace_digest",
        "actor_role_state",
        "actor_role_known",
        '"x"',
        '"y"',
        '"row"',
        '"col"',
        '"value"',
        '"color"',
    )
    if any(token and token in rendered for token in forbidden):
        raise ValueError("V4.1 model view contains forbidden provenance")


@dataclass(frozen=True)
class CalibrationBundle:
    """Source-only calibration and decision thresholds for every model mode."""

    parameters: Mapping[str, Mapping[str, Mapping[str, float]]]
    thresholds: Mapping[str, Mapping[str, float]]
    source_oof_metrics: Mapping[str, Any]
    format_version: str = CALIBRATION_FORMAT_VERSION
    calibration_checksum: str = ""

    def __post_init__(self) -> None:
        if self.format_version != CALIBRATION_FORMAT_VERSION:
            raise ValueError("unsupported V4.1 calibration format")
        if not self.calibration_checksum:
            payload = self.to_dict(include_checksum=False)
            object.__setattr__(self, "calibration_checksum", _checksum(payload))

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        payload = {
            "format_version": self.format_version,
            "parameters": {
                mode: {
                    label: {
                        "slope": float(item["slope"]),
                        "intercept": float(item["intercept"]),
                    }
                    for label, item in labels.items()
                }
                for mode, labels in self.parameters.items()
            },
            "thresholds": {
                mode: {
                    label: float(value) for label, value in labels.items()
                }
                for mode, labels in self.thresholds.items()
            },
            "source_oof_metrics": self.source_oof_metrics,
        }
        if include_checksum:
            payload["calibration_checksum"] = self.calibration_checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationBundle:
        expected = str(payload.get("calibration_checksum", ""))
        bundle = cls(
            format_version=str(payload["format_version"]),
            parameters=dict(payload["parameters"]),
            thresholds=dict(payload["thresholds"]),
            source_oof_metrics=dict(payload["source_oof_metrics"]),
            calibration_checksum="",
        )
        if expected and bundle.calibration_checksum != expected:
            raise ValueError("V4.1 calibration checksum mismatch")
        return bundle


def _targets_masks(
    windows: Sequence[MechanicWindowRecord],
) -> tuple[np.ndarray, np.ndarray]:
    return v4._arrays(windows)


def _raw_matrices(
    windows: Sequence[MechanicWindowRecord],
    priors: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, np.ndarray], list[tuple[v4.MechanicEvidence, ...]]]:
    matrices: dict[str, np.ndarray] = {}
    evidence_rows: list[tuple[v4.MechanicEvidence, ...]] = []
    for mode in MODEL_MODES:
        matrix, evidence = v4._probability_matrix(windows, priors, mode=mode)
        matrices[mode] = matrix
        if mode == "structured":
            evidence_rows = list(evidence)
    return matrices, evidence_rows


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _fit_platt(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    if len(set(np.asarray(targets, dtype=int).tolist())) < 2:
        return {"slope": 1.0, "intercept": 0.0}
    model = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=307,
    )
    model.fit(_logits(probabilities).reshape(-1, 1), targets)
    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def _apply_parameter(
    probabilities: np.ndarray, parameter: Mapping[str, float]
) -> np.ndarray:
    values = (
        float(parameter["slope"]) * _logits(probabilities)
        + float(parameter["intercept"])
    )
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def apply_calibration(
    matrix: np.ndarray,
    bundle: CalibrationBundle,
    mode: str,
) -> np.ndarray:
    calibrated = np.asarray(matrix, dtype=np.float64).copy()
    for index, label in enumerate(EFFECT_LABELS):
        calibrated[:, index] = _apply_parameter(
            calibrated[:, index], bundle.parameters[mode][label]
        )
    return calibrated


def _select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    candidates = sorted(
        {0.5, *np.asarray(probabilities, dtype=float).tolist()}
    )
    scored = [
        (
            float(
                f1_score(
                    targets,
                    np.asarray(probabilities) >= threshold,
                    zero_division=0,
                )
            ),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored)[2]


def multilabel_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {label: 0.5 for label in EFFECT_LABELS}
    per_label: dict[str, Any] = {}
    for index, label in enumerate(EFFECT_LABELS):
        selected = masks[:, index].astype(bool)
        y_true = targets[selected, index]
        y_prob = probabilities[selected, index]
        y_pred = y_prob >= float(thresholds[label])
        per_label[label] = {
            "applicable": len(y_true),
            "positives": int(np.sum(y_true)),
            "negatives": int(len(y_true) - np.sum(y_true)),
            "threshold": float(thresholds[label]),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier": float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else 0.0,
            "ece": v4._ece(y_true, y_prob),
        }
    return {
        "macro_f1": float(np.mean([item["f1"] for item in per_label.values()])),
        "macro_brier": float(
            np.mean([item["brier"] for item in per_label.values()])
        ),
        "macro_ece": float(np.mean([item["ece"] for item in per_label.values()])),
        "per_label": per_label,
    }


def fit_source_calibration(
    windows: Sequence[MechanicWindowRecord],
) -> CalibrationBundle:
    """Fit calibration only from leave-one-source-game-out predictions."""
    targets, masks = _targets_masks(windows)
    matrices = {
        mode: np.zeros((len(windows), len(EFFECT_LABELS)), dtype=np.float64)
        for mode in MODEL_MODES
    }
    assigned = np.zeros(len(windows), dtype=bool)
    for game in SOURCE_TRAIN:
        train = [window for window in windows if window.game_id != game]
        held_indices = [
            index for index, window in enumerate(windows) if window.game_id == game
        ]
        if not held_indices:
            continue
        priors = v4.fit_source_priors(train)
        held = [windows[index] for index in held_indices]
        fold_matrices, _ = _raw_matrices(held, priors)
        for mode in MODEL_MODES:
            matrices[mode][held_indices] = fold_matrices[mode]
        assigned[held_indices] = True
    if not bool(np.all(assigned)):
        raise ValueError("V4.1 source OOF calibration left windows unassigned")

    parameters: dict[str, dict[str, dict[str, float]]] = {}
    calibrated: dict[str, np.ndarray] = {}
    thresholds: dict[str, dict[str, float]] = {}
    raw_metrics: dict[str, Any] = {}
    calibrated_metrics: dict[str, Any] = {}
    for mode in MODEL_MODES:
        parameters[mode] = {}
        thresholds[mode] = {}
        calibrated[mode] = matrices[mode].copy()
        for index, label in enumerate(EFFECT_LABELS):
            selected = masks[:, index].astype(bool)
            parameter = _fit_platt(
                matrices[mode][selected, index],
                targets[selected, index],
            )
            parameters[mode][label] = parameter
            calibrated[mode][:, index] = _apply_parameter(
                matrices[mode][:, index], parameter
            )
            thresholds[mode][label] = _select_threshold(
                calibrated[mode][selected, index],
                targets[selected, index],
            )
        raw_metrics[mode] = multilabel_metrics(
            targets, masks, matrices[mode]
        )
        calibrated_metrics[mode] = multilabel_metrics(
            targets,
            masks,
            calibrated[mode],
            thresholds=thresholds[mode],
        )
    return CalibrationBundle(
        parameters=parameters,
        thresholds=thresholds,
        source_oof_metrics={
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
    )


def compact_qwen_prompt(window: MechanicWindowRecord) -> str:
    """Short, outcome-blind mechanic history using a frozen codebook."""
    effect_codes = {
        "actor_displaced": "A",
        "target_created": "C",
        "target_removed": "R",
        "target_moved": "M",
    }
    rows = []
    for event in window.context:
        effects = "".join(
            effect_codes[label]
            for label in EFFECT_LABELS
            if event.applicable[label] and event.effects[label]
        ) or "-"
        applicable = "".join(
            effect_codes[label]
            for label in EFFECT_LABELS
            if event.applicable[label]
        ) or "-"
        rows.append(
            f"{event.action_name}/{event.action_family}/"
            f"{event.anchor_condition}/{effects}/{applicable}"
        )
    query = (
        f"{window.query.action_name}/{window.query.action_family}/"
        f"{window.query.anchor_condition}"
    )
    return (
        "Infer <=8 rules. Row=action/family/anchor/effects/applicable. "
        "Effects A=actor_displaced,C=target_created,R=target_removed,"
        "M=target_moved. s=e means exact, s=f family; z must be 0. JSON only.\n"
        f"H={';'.join(rows)}\nQ={query}"
    )


def compact_qwen_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["h"],
        "properties": {
            "h": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": ["s", "v", "a", "e", "z"],
                    "properties": {
                        "s": {"enum": ["e", "f"]},
                        "v": {"type": "string"},
                        "a": {"enum": list(v4.ANCHOR_CONDITIONS)},
                        "e": {"enum": list(EFFECT_LABELS)},
                        "z": {"const": 0},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }


def compile_compact_rule(
    payload: Mapping[str, Any],
    query: v4.MechanicQuery,
) -> v4.MechanicRule:
    if set(payload) != {"s", "v", "a", "e", "z"}:
        raise ValueError("compact rule has unexpected fields")
    kind = {"e": "exact", "f": "family"}.get(str(payload["s"]))
    if kind is None:
        raise ValueError("compact rule has invalid scope")
    value = str(payload["v"])
    allowed_value = query.action_name if kind == "exact" else query.action_family
    if value != allowed_value:
        raise ValueError("compact rule is not grounded to the query")
    if int(payload["z"]) != 0:
        raise ValueError("compact proposal support must be zero")
    effect = str(payload["e"])
    anchor = str(payload["a"])
    return v4.MechanicRule(
        rule_id=v4._rule_id(kind, value, anchor, effect),
        action_scope_kind=kind,
        action_scope_value=value,
        anchor_condition=anchor,
        effect=effect,
        support=0,
        source="local_llm",
    )


def _chat_token_count(tokenizer: Any, window: MechanicWindowRecord) -> int:
    messages = [
        {
            "role": "system",
            "content": (
                "Return one JSON value only. It must satisfy this schema: "
                + json.dumps(compact_qwen_schema(), sort_keys=True)
            ),
        },
        {"role": "user", "content": compact_qwen_prompt(window)},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = (
        encoded["input_ids"]
        if isinstance(encoded, Mapping) or hasattr(encoded, "keys")
        else encoded
    )
    return int(inputs.shape[-1])


def measure_qwen_token_budget(
    windows: Sequence[MechanicWindowRecord],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(frozen["qwen"]["path"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    counts = [_chat_token_count(tokenizer, window) for window in windows]
    return {
        "windows_checked": len(counts),
        "minimum_tokens": min(counts, default=0),
        "maximum_tokens": max(counts, default=0),
        "preflight_limit": int(frozen["qwen"]["preflight_maximum_input_tokens"]),
        "runtime_limit": int(frozen["qwen"]["maximum_input_tokens"]),
    }


def _window_quality(windows: Sequence[MechanicWindowRecord]) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    for label in EFFECT_LABELS:
        eligible = [window for window in windows if window.applicable[label]]
        positive = sum(int(window.labels[label]) for window in eligible)
        per_label[label] = {
            "applicable": len(eligible),
            "positives": positive,
            "negatives": len(eligible) - positive,
        }
    per_game: dict[str, Any] = {}
    for game in sorted({window.game_id for window in windows}):
        selected = [window for window in windows if window.game_id == game]
        states = Counter(window.actor_role_state for window in selected)
        per_game[game] = {
            "windows": len(selected),
            "actor_role_resolved_rate": sum(
                int(window.actor_role_state != ActorRoleState.AMBIGUOUS.value)
                for window in selected
            )
            / max(1, len(selected)),
            "actor_role_states": dict(sorted(states.items())),
        }
    states = Counter(window.actor_role_state for window in windows)
    return {
        "windows": len(windows),
        "unique_window_digests": len({window.window_digest for window in windows}),
        "actor_role_resolved_rate": sum(
            int(window.actor_role_state != ActorRoleState.AMBIGUOUS.value)
            for window in windows
        )
        / max(1, len(windows)),
        "actor_role_states": dict(sorted(states.items())),
        "per_label": per_label,
        "per_game": per_game,
    }


def _label_capacity(
    per_label: Mapping[str, Mapping[str, int]],
    minimum_positive: int,
    minimum_negative: int,
) -> bool:
    return all(
        int(item["positives"]) >= int(minimum_positive)
        and int(item["negatives"]) >= int(minimum_negative)
        for item in per_label.values()
    )


def _shuffle_context(
    windows: Sequence[MechanicWindowRecord],
    *,
    binding: bool,
) -> list[MechanicWindowRecord]:
    shuffled: list[MechanicWindowRecord] = []
    for window in windows:
        events = list(window.context)
        if len(events) > 1:
            offset = 1 + int(window.window_digest[:8], 16) % (len(events) - 1)
            rotated = events[offset:] + events[:offset]
            if binding:
                events = [
                    replace(event, anchor_condition=source.anchor_condition)
                    for event, source in zip(events, rotated)
                ]
            else:
                events = [
                    replace(
                        event,
                        effects=source.effects,
                        applicable=source.applicable,
                        actor_role_known=source.actor_role_known,
                        actor_role_state=source.actor_role_state,
                    )
                    for event, source in zip(events, rotated)
                ]
        shuffled.append(replace(window, context=tuple(events), window_digest=""))
    return shuffled


def _bootstrap_skill(
    windows: Sequence[MechanicWindowRecord],
    targets: np.ndarray,
    masks: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    model_thresholds: Mapping[str, float],
    baseline_thresholds: Mapping[str, float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    groups: dict[str, np.ndarray] = {}
    keys = np.asarray([window.run_key for window in windows])
    for key in sorted(set(keys)):
        groups[key] = np.flatnonzero(keys == key)
    by_game: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, indices in groups.items():
        by_game[key.split(":", 1)[0]].append(indices)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(samples)):
        sampled_parts = []
        for game in sorted(by_game):
            game_groups = by_game[game]
            choices = rng.integers(0, len(game_groups), size=len(game_groups))
            sampled_parts.extend(game_groups[index] for index in choices)
        sampled = np.concatenate(sampled_parts)
        left = multilabel_metrics(
            targets[sampled],
            masks[sampled],
            model[sampled],
            thresholds=model_thresholds,
        )
        right = multilabel_metrics(
            targets[sampled],
            masks[sampled],
            baseline[sampled],
            thresholds=baseline_thresholds,
        )
        values.append(v4._brier_skill(left, right))
    return {
        "samples": int(samples),
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = _read_json(Path(path))
    expected = str(payload.get("manifest_checksum", ""))
    check = dict(payload)
    check.pop("manifest_checksum", None)
    actual = _checksum(check)
    if expected != actual:
        raise ValueError(
            f"V4.1 frozen-manifest checksum mismatch: {actual} != {expected}"
        )
    if payload.get("format_version") != "sage12-mechanic-induction-v4.1":
        raise ValueError("unsupported SAGE12 V4.1 manifest")
    return payload


def run_source_train_preflight(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    traces = _load_traces(V3_OUTPUT_DIR / "shards", SOURCE_TRAIN)
    windows = build_mechanic_windows(
        traces, context_length=int(frozen["window"]["context_length"])
    )
    for window in windows:
        validate_model_view(window)
    priors = v4.fit_source_priors(windows)
    calibration = fit_source_calibration(windows)
    _write_jsonl_windows(destination / "source_train_windows.jsonl", windows)
    priors_payload: dict[str, Any] = {
        "format_version": "sage12-mechanic-priors-v4.1",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "prior_strength": float(frozen["model"]["prior_strength"]),
        "counts": priors,
    }
    priors_payload["priors_checksum"] = _checksum(priors_payload)
    _write_json(destination / "source_priors.json", priors_payload)
    _write_json(destination / "calibration.json", calibration.to_dict())

    quality = _window_quality(windows)
    static_rows = [
        {
            f"action:{window.query.action_name}": 1,
            f"family:{window.query.action_family}": 1,
            f"anchor:{window.query.anchor_condition}": 1,
        }
        for window in windows
    ]
    probe = v4._identity_probe(static_rows, [window.game_id for window in windows])
    action_probe = v4._identity_probe(
        [{f"action:{window.query.action_name}": 1} for window in windows],
        [window.game_id for window in windows],
    )
    identity_gain = probe["accuracy"] - action_probe["accuracy"]
    token_budget = measure_qwen_token_budget(windows, frozen)
    raw_oof = calibration.source_oof_metrics["raw"]["structured"]
    calibrated_oof = calibration.source_oof_metrics["calibrated"]["structured"]
    gates_cfg = frozen["gates"]
    gates = {
        "minimum_source_train_windows": len(windows)
        >= int(gates_cfg["minimum_source_train_windows"]),
        "source_train_label_capacity": _label_capacity(
            quality["per_label"],
            int(gates_cfg["minimum_source_train_positives_per_label"]),
            int(gates_cfg["minimum_source_train_negatives_per_label"]),
        ),
        "minimum_global_actor_role_resolution": quality[
            "actor_role_resolved_rate"
        ]
        >= float(gates_cfg["minimum_global_actor_role_resolution"]),
        "minimum_per_game_actor_role_resolution": all(
            item["actor_role_resolved_rate"]
            >= float(gates_cfg["minimum_per_game_actor_role_resolution"])
            for item in quality["per_game"].values()
        ),
        "static_identity_leakage": identity_gain
        <= float(gates_cfg["maximum_static_identity_gain_over_action"]),
        "source_oof_calibration": calibrated_oof["macro_ece"]
        <= float(gates_cfg["maximum_source_oof_macro_ece"]),
        "source_oof_brier_non_degradation": calibrated_oof["macro_brier"]
        - raw_oof["macro_brier"]
        <= float(gates_cfg["maximum_source_oof_brier_degradation"]),
        "qwen_prompt_budget": token_budget["maximum_tokens"]
        <= int(frozen["qwen"]["preflight_maximum_input_tokens"]),
        "model_view_firewall": True,
    }
    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": (
            "PASS_SOURCE_TRAIN_PREFLIGHT"
            if all(gates.values())
            else "FAIL_SOURCE_TRAIN_PREFLIGHT"
        ),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "rows": len(traces),
        "windows": len(windows),
        "quality": quality,
        "identity_probe": {
            "static": probe,
            "action_only": action_probe,
            "gain": identity_gain,
        },
        "source_oof": calibration.source_oof_metrics,
        "calibration_checksum": calibration.calibration_checksum,
        "priors_checksum": priors_payload["priors_checksum"],
        "qwen_token_budget": token_budget,
        "gates": gates,
        "source_validation_opened": False,
        "world_model_fit_authorized": False,
    }
    payload["preflight_checksum"] = _checksum(payload)
    _write_json(destination / "source_train_preflight.json", payload)
    return payload


def _output_contract(
    windows: Sequence[MechanicWindowRecord],
    evidence_rows: Sequence[Sequence[v4.MechanicEvidence]],
) -> dict[str, Any]:
    emitted = valid = support_zero = grounded = 0
    for window, row in zip(windows, evidence_rows):
        for evidence in row:
            emitted += 1
            raw = _canonical(evidence.rule.to_dict())
            rule = v4.MechanicRule.from_dict(json.loads(raw))
            valid += 1
            support_zero += int(rule.support == 0)
            hypothesis = v4.rule_to_semantic_hypothesis(
                rule,
                window.query,
                confidence=evidence.posterior_probability,
            )
            grounded += int(
                hypothesis.action_name == window.query.action_name
                and rule.matches_query(window.query)
            )
    denominator = max(1, emitted)
    return {
        "emitted_hypotheses": emitted,
        "strict_json_validity": valid / denominator,
        "support_zero_rate": support_zero / denominator,
        "grounded_hypothesis_rate": grounded / denominator,
    }


def _qwen_indices(windows: Sequence[MechanicWindowRecord], count: int) -> list[int]:
    selected: list[int] = []
    per_game = max(1, int(count) // len(SOURCE_VALIDATION))
    for game in SOURCE_VALIDATION:
        candidates = [
            (index, window)
            for index, window in enumerate(windows)
            if window.game_id == game
        ]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                (
                    item[1].query.action_name
                    + ":"
                    + item[1].query.anchor_condition
                    + ":"
                    + item[1].window_digest
                ).encode()
            ).hexdigest()
        )
        selected.extend(index for index, _ in candidates[:per_game])
    if len(selected) < count:
        chosen = set(selected)
        selected.extend(
            index
            for index in range(len(windows))
            if index not in chosen
        )
    return selected[:count]


def _qwen_generate(
    model: TransformersJSONModel,
    windows: Sequence[MechanicWindowRecord],
    indices: Sequence[int],
    *,
    maximum_tokens: int,
) -> tuple[list[tuple[v4.MechanicRule, ...]], list[dict[str, Any]]]:
    rows: list[tuple[v4.MechanicRule, ...]] = []
    outputs: list[dict[str, Any]] = []
    for index in indices:
        window = windows[index]
        raw = ""
        error = ""
        rules: tuple[v4.MechanicRule, ...] = ()
        try:
            raw = model.generate_json(
                prompt=compact_qwen_prompt(window),
                schema=compact_qwen_schema(),
                maximum_tokens=maximum_tokens,
            )
            payload = json.loads(raw)
            if set(payload) != {"h"} or not isinstance(payload["h"], list):
                raise ValueError("compact response must contain only h")
            rules = tuple(
                compile_compact_rule(item, window.query)
                for item in payload["h"][:8]
            )
        except Exception as exc:  # noqa: BLE001 - audited local-model failure
            error = f"{type(exc).__name__}: {exc}"
        rows.append(rules)
        outputs.append(
            {
                "window_digest": window.window_digest,
                "raw_response": raw[:4096],
                "parse_error": error,
                "hypotheses": [item.to_dict() for item in rules],
            }
        )
    return rows, outputs


def _score_qwen_rules(
    windows: Sequence[MechanicWindowRecord],
    indices: Sequence[int],
    rules: Sequence[Sequence[v4.MechanicRule]],
    priors: Mapping[str, Mapping[str, int]],
) -> np.ndarray:
    matrix = []
    for index, emitted in zip(indices, rules):
        window = windows[index]
        probabilities = {
            effect: v4._action_only_probability(
                window, effect, priors, use_context=False
            )
            for effect in EFFECT_LABELS
        }
        for rule in emitted:
            if rule.matches_query(window.query):
                evidence = v4.score_rule(rule, window.context, priors)
                probabilities[rule.effect] = evidence.posterior_probability
        matrix.append([probabilities[label] for label in EFFECT_LABELS])
    return np.asarray(matrix, dtype=np.float64)


def _evaluate_qwen(
    windows: Sequence[MechanicWindowRecord],
    priors: Mapping[str, Mapping[str, int]],
    bundle: CalibrationBundle,
    frozen: Mapping[str, Any],
    targets: np.ndarray,
    masks: np.ndarray,
    baseline: np.ndarray,
    baseline_name: str,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    count = min(int(frozen["qwen"]["contexts"]), len(windows))
    indices = _qwen_indices(windows, count)
    config = TransformersModelConfig(
        model_path=str(frozen["qwen"]["path"]),
        device=str(frozen["qwen"]["device"]),
        temperature=0.0,
        maximum_input_tokens=int(frozen["qwen"]["maximum_input_tokens"]),
    )
    model = TransformersJSONModel(config)
    started = time.perf_counter()
    rules, outputs = _qwen_generate(
        model,
        windows,
        indices,
        maximum_tokens=int(frozen["qwen"]["maximum_output_tokens"]),
    )
    raw = _score_qwen_rules(windows, indices, rules, priors)
    calibrated = apply_calibration(raw, bundle, "structured")
    subset = np.asarray(indices, dtype=int)
    metrics = multilabel_metrics(
        targets[subset],
        masks[subset],
        calibrated,
        thresholds=bundle.thresholds["structured"],
    )
    baseline_metrics = multilabel_metrics(
        targets[subset],
        masks[subset],
        baseline[subset],
        thresholds=bundle.thresholds[baseline_name],
    )
    skill = v4._brier_skill(metrics, baseline_metrics)

    shuffled_windows = _shuffle_context(windows, binding=False)
    shuffled_rules, shuffled_outputs = _qwen_generate(
        model,
        shuffled_windows,
        indices,
        maximum_tokens=int(frozen["qwen"]["maximum_output_tokens"]),
    )
    shuffled_raw = _score_qwen_rules(
        shuffled_windows, indices, shuffled_rules, priors
    )
    shuffled = apply_calibration(shuffled_raw, bundle, "structured")
    shuffled_metrics = multilabel_metrics(
        targets[subset],
        masks[subset],
        shuffled,
        thresholds=bundle.thresholds["structured"],
    )
    shuffle_drop = skill - v4._brier_skill(shuffled_metrics, baseline_metrics)

    valid = sum(not item["parse_error"] for item in outputs)
    emitted = sum(len(item) for item in rules)
    grounded = sum(
        int(rule.matches_query(windows[index].query))
        for index, row in zip(indices, rules)
        for rule in row
    )
    support_zero = sum(
        int(rule.support == 0) for row in rules for rule in row
    )
    productive = recalled = 0
    for index, row in zip(indices, rules):
        emitted_effects = {rule.effect for rule in row}
        for label in EFFECT_LABELS:
            if masks[index, EFFECT_LABELS.index(label)] and targets[
                index, EFFECT_LABELS.index(label)
            ]:
                productive += 1
                recalled += int(label in emitted_effects)
    per_game = {}
    for game in SOURCE_VALIDATION:
        selected_positions = [
            position
            for position, index in enumerate(indices)
            if windows[index].game_id == game
        ]
        if not selected_positions:
            continue
        positions = np.asarray(selected_positions, dtype=int)
        game_targets = targets[subset][positions]
        game_masks = masks[subset][positions]
        game_metric = multilabel_metrics(
            game_targets,
            game_masks,
            calibrated[positions],
            thresholds=bundle.thresholds["structured"],
        )
        game_baseline = multilabel_metrics(
            game_targets,
            game_masks,
            baseline[subset][positions],
            thresholds=bundle.thresholds[baseline_name],
        )
        per_game[game] = {
            "windows": len(positions),
            "brier_skill": v4._brier_skill(game_metric, game_baseline),
        }
    _write_jsonl_dicts(output_dir / "qwen_outputs.jsonl", outputs)
    _write_jsonl_dicts(
        output_dir / "qwen_outcome_shuffle_outputs.jsonl",
        shuffled_outputs,
    )
    rates = {
        "strict_json_validity": valid / max(1, len(indices)),
        "grounded_hypothesis_rate": grounded / max(1, emitted),
        "support_zero_rate": support_zero / max(1, emitted),
        "productive_effect_recall_at_8": recalled / max(1, productive),
    }
    qwen_gates = {
        "strict_json_validity": rates["strict_json_validity"]
        >= float(frozen["qwen_gates"]["minimum_strict_json_validity"]),
        "grounded_hypothesis_rate": rates["grounded_hypothesis_rate"]
        >= float(frozen["qwen_gates"]["minimum_grounded_hypothesis_rate"]),
        "support_zero_rate": rates["support_zero_rate"] == 1.0,
        "productive_effect_recall_at_8": rates[
            "productive_effect_recall_at_8"
        ]
        >= float(frozen["qwen_gates"]["minimum_productive_effect_recall_at_8"]),
        "outcome_shuffle_drop": shuffle_drop
        >= float(frozen["qwen_gates"]["minimum_outcome_shuffle_skill_drop"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
    }
    return {
        "status": "PASS" if all(qwen_gates.values()) else "FAIL_CLOSED",
        "authority_separate": True,
        "contexts": len(indices),
        **rates,
        "emitted_hypotheses": emitted,
        "metrics": metrics,
        "baseline": baseline_name,
        "brier_skill": skill,
        "outcome_shuffle": {
            "metrics": shuffled_metrics,
            "skill_drop": shuffle_drop,
        },
        "per_game": per_game,
        "gates": qwen_gates,
        "inference_seconds": time.perf_counter() - started,
        "device": str(frozen["qwen"]["device"]),
        "model": str(frozen["qwen"]["name"]),
    }


def _effect_authority(
    metrics: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    authority: dict[str, Any] = {}
    for label in EFFECT_LABELS:
        best = min(
            ("structured", "local_action", "global_action", "template"),
            key=lambda mode: metrics[mode]["per_label"][label]["brier"],
        )
        item = metrics[best]["per_label"][label]
        capacity = quality["per_label"][label]
        eligible = (
            float(item["brier"]) <= 0.10
            and float(item["f1"]) >= 0.20
            and int(capacity["positives"]) >= 30
            and int(capacity["negatives"]) >= 30
        )
        authority[label] = {
            "method": best,
            "brier": float(item["brier"]),
            "f1": float(item["f1"]),
            "eligible_for_v5": bool(eligible),
        }
    return authority


def run_evaluation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    run_qwen: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    preflight = _read_json(destination / "source_train_preflight.json")
    priors_payload = _read_json(destination / "source_priors.json")
    priors = dict(priors_payload["counts"])
    bundle = CalibrationBundle.from_dict(
        _read_json(destination / "calibration.json")
    )
    traces = _load_traces(destination / "shards", SOURCE_VALIDATION)
    windows = build_mechanic_windows(
        traces, context_length=int(frozen["window"]["context_length"])
    )
    for window in windows:
        validate_model_view(window)
    _write_jsonl_windows(destination / "validation_windows.jsonl", windows)
    targets, masks = _targets_masks(windows)
    raw_matrices, evidence_rows = _raw_matrices(windows, priors)
    calibrated_matrices = {
        mode: apply_calibration(matrix, bundle, mode)
        for mode, matrix in raw_matrices.items()
    }
    raw_metrics = {
        mode: multilabel_metrics(targets, masks, matrix)
        for mode, matrix in raw_matrices.items()
    }
    calibrated_metrics = {
        mode: multilabel_metrics(
            targets,
            masks,
            matrix,
            thresholds=bundle.thresholds[mode],
        )
        for mode, matrix in calibrated_matrices.items()
    }
    stronger_name = min(
        BASELINE_MODES,
        key=lambda name: calibrated_metrics[name]["macro_brier"],
    )
    raw_skill = v4._brier_skill(
        raw_metrics["structured"], raw_metrics[stronger_name]
    )
    calibrated_skill = v4._brier_skill(
        calibrated_metrics["structured"], calibrated_metrics[stronger_name]
    )
    f1_gain = (
        calibrated_metrics["structured"]["macro_f1"]
        - calibrated_metrics[stronger_name]["macro_f1"]
    )

    outcome_windows = _shuffle_context(windows, binding=False)
    outcome_raw, _ = v4._probability_matrix(
        outcome_windows, priors, mode="structured"
    )
    outcome_matrix = apply_calibration(outcome_raw, bundle, "structured")
    outcome_metrics = multilabel_metrics(
        targets,
        masks,
        outcome_matrix,
        thresholds=bundle.thresholds["structured"],
    )
    outcome_drop = calibrated_skill - v4._brier_skill(
        outcome_metrics, calibrated_metrics[stronger_name]
    )

    binding_windows = _shuffle_context(windows, binding=True)
    binding_raw, _ = v4._probability_matrix(
        binding_windows, priors, mode="structured"
    )
    binding_matrix = apply_calibration(binding_raw, bundle, "structured")
    binding_metrics = multilabel_metrics(
        targets,
        masks,
        binding_matrix,
        thresholds=bundle.thresholds["structured"],
    )
    context_gain = v4._brier_skill(
        calibrated_metrics["structured"],
        calibrated_metrics["context_ablation"],
    )
    bootstrap = _bootstrap_skill(
        windows,
        targets,
        masks,
        calibrated_matrices["structured"],
        calibrated_matrices[stronger_name],
        model_thresholds=bundle.thresholds["structured"],
        baseline_thresholds=bundle.thresholds[stronger_name],
        samples=int(frozen["evaluation"]["bootstrap_samples"]),
        seed=int(frozen["evaluation"]["random_seed"]),
    )
    per_game = {}
    for game in SOURCE_VALIDATION:
        selected = np.asarray([window.game_id == game for window in windows])
        model_metric = multilabel_metrics(
            targets[selected],
            masks[selected],
            calibrated_matrices["structured"][selected],
            thresholds=bundle.thresholds["structured"],
        )
        candidate_metrics = {
            mode: multilabel_metrics(
                targets[selected],
                masks[selected],
                calibrated_matrices[mode][selected],
                thresholds=bundle.thresholds[mode],
            )
            for mode in BASELINE_MODES
        }
        baseline_name = min(
            BASELINE_MODES,
            key=lambda mode: candidate_metrics[mode]["macro_brier"],
        )
        per_game[game] = {
            "windows": int(np.sum(selected)),
            "structured": model_metric,
            "stronger_baseline": baseline_name,
            "baseline": candidate_metrics[baseline_name],
            "brier_skill": v4._brier_skill(
                model_metric, candidate_metrics[baseline_name]
            ),
        }
    quality = _window_quality(windows)
    output_contract = _output_contract(windows, evidence_rows)
    qwen = (
        _evaluate_qwen(
            windows,
            priors,
            bundle,
            frozen,
            targets,
            masks,
            calibrated_matrices[stronger_name],
            stronger_name,
            output_dir=destination,
        )
        if run_qwen
        else {
            "status": "SKIPPED",
            "authority_separate": True,
        }
    )
    effect_authority = _effect_authority(calibrated_metrics, quality)
    gates_cfg = frozen["gates"]
    gates = {
        "minimum_prospective_windows": len(windows)
        >= int(gates_cfg["minimum_prospective_windows"]),
        "prospective_label_capacity": _label_capacity(
            quality["per_label"],
            int(gates_cfg["minimum_validation_positives_per_label"]),
            int(gates_cfg["minimum_validation_negatives_per_label"]),
        ),
        "minimum_global_actor_role_resolution": quality[
            "actor_role_resolved_rate"
        ]
        >= float(gates_cfg["minimum_global_actor_role_resolution"]),
        "minimum_per_game_actor_role_resolution": all(
            item["actor_role_resolved_rate"]
            >= float(gates_cfg["minimum_per_game_actor_role_resolution"])
            for item in quality["per_game"].values()
        ),
        "strict_json_validity": output_contract["strict_json_validity"] == 1.0,
        "support_zero_rate": output_contract["support_zero_rate"] == 1.0,
        "grounded_hypothesis_rate": output_contract[
            "grounded_hypothesis_rate"
        ]
        == 1.0,
        "minimum_raw_brier_skill": raw_skill
        >= float(gates_cfg["minimum_macro_brier_skill"]),
        "minimum_calibrated_brier_skill": calibrated_skill
        >= float(gates_cfg["minimum_macro_brier_skill"]),
        "bootstrap_lower_bound_positive": bootstrap["lower_95"] > 0.0,
        "minimum_macro_f1_gain": f1_gain
        >= float(gates_cfg["minimum_macro_f1_gain"]),
        "minimum_outcome_shuffle_drop": outcome_drop
        >= float(gates_cfg["minimum_outcome_shuffle_skill_drop"]),
        "minimum_context_gain": context_gain
        >= float(gates_cfg["minimum_context_brier_skill_gain"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
        "maximum_macro_ece": calibrated_metrics["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "source_preflight_passed": all(preflight["gates"].values()),
    }
    passed = all(gates.values())
    predictions = []
    for index, window in enumerate(windows):
        predictions.append(
            {
                "window_digest": window.window_digest,
                "game_id": window.game_id,
                "run_key": window.run_key,
                "query": window.query.to_dict(),
                "actor_role_state": window.actor_role_state,
                "labels": dict(window.labels),
                "applicable": dict(window.applicable),
                "raw_structured_probabilities": {
                    label: float(raw_matrices["structured"][index, label_index])
                    for label_index, label in enumerate(EFFECT_LABELS)
                },
                "calibrated_structured_probabilities": {
                    label: float(
                        calibrated_matrices["structured"][index, label_index]
                    )
                    for label_index, label in enumerate(EFFECT_LABELS)
                },
                "evidence": [item.to_dict() for item in evidence_rows[index]],
            }
        )
    _write_jsonl_dicts(destination / "predictions.jsonl", predictions)
    payload: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "all_structured_gates_passed": passed,
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": preflight["preflight_checksum"],
        "calibration_checksum": bundle.calibration_checksum,
        "rows": {
            "prospective_transitions": len(traces),
            "prospective_windows": len(windows),
        },
        "quality": quality,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "stronger_baseline": stronger_name,
        "raw_macro_brier_skill": raw_skill,
        "calibrated_macro_brier_skill": calibrated_skill,
        "calibrated_macro_f1_gain": f1_gain,
        "bootstrap_skill": bootstrap,
        "outcome_shuffle": {
            "metrics": outcome_metrics,
            "skill_drop": outcome_drop,
        },
        "binding_shuffle": {
            "metrics": binding_metrics,
            "skill": v4._brier_skill(
                binding_metrics, calibrated_metrics[stronger_name]
            ),
        },
        "context_brier_skill_gain": context_gain,
        "per_game": per_game,
        "effect_authority": effect_authority,
        "output_contract": output_contract,
        "qwen": qwen,
        "gates": gates,
        "firewall": {
            "source_only_calibration": True,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
            "controller_executed": False,
        },
        "world_model_fit_authorized": passed,
        "qwen_world_model_fit_authorized": bool(
            passed and qwen.get("status") == "PASS"
        ),
        "ebm_fit_authorized": False,
    }
    payload["result_checksum"] = _checksum(payload)
    _write_json(destination / "pilot_result.json", payload)
    return payload


def _load_traces(
    shard_dir: Path, games: Sequence[str]
) -> list[ActionTargetTrace]:
    rows: list[ActionTargetTrace] = []
    for game in games:
        path = shard_dir / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(ActionTargetTrace.from_dict(json.loads(line)))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_windows(
    path: Path, windows: Sequence[MechanicWindowRecord]
) -> None:
    _write_jsonl_dicts(path, [window.to_dict() for window in windows])


def _write_jsonl_dicts(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("preflight", "evaluate"),
        default="preflight",
    )
    parser.add_argument(
        "--frozen-manifest",
        default=str(DEFAULT_FROZEN_MANIFEST_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_source_train_preflight(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
        )
    else:
        result = run_evaluation(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
            run_qwen=not args.skip_qwen,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_FORMAT_VERSION",
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "ActorRoleState",
    "CalibrationBundle",
    "CausalRoleTracker",
    "MechanicWindowRecord",
    "SemanticTransitionEvent",
    "apply_calibration",
    "build_mechanic_windows",
    "compact_qwen_prompt",
    "compact_qwen_schema",
    "compile_compact_rule",
    "fit_source_calibration",
    "load_frozen_manifest",
    "measure_qwen_token_budget",
    "multilabel_metrics",
    "run_evaluation",
    "run_source_train_preflight",
    "validate_model_view",
]
