"""Invariant target-effect replication of SAGE12 temporal mechanics (V4.2).

V4.2 exposes only three target effects and a three-state causal anchor.  The
V4.1 implementation remains immutable and is used only through a padded
internal adapter whose actor effect is always inapplicable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION

from . import mechanic_induction as v4
from . import mechanic_replication as v41
from .action_target_data import EFFECT_LABELS, ActionTargetTrace
from .llm import TransformersJSONModel, TransformersModelConfig

FORMAT_VERSION = "sage12-target-mechanic-window-v4.2"
PREFLIGHT_FORMAT_VERSION = "sage12-target-mechanic-preflight-v4.2"
RESULT_FORMAT_VERSION = "sage12-target-mechanic-pilot-result-v4.2"
CALIBRATION_FORMAT_VERSION = "sage12-target-mechanic-calibration-v4.2"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "mechanic_induction_v4_2"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
V3_OUTPUT_DIR = Path("training") / "sage12" / "action_target_pilot_v3"

TARGET_EFFECT_LABELS = (
    "target_created",
    "target_removed",
    "target_moved",
)
COARSE_ANCHOR_CONDITIONS = ("occupied", "free", "none")
MODEL_MODES = v41.MODEL_MODES
BASELINE_MODES = v41.BASELINE_MODES
_TARGET_INDICES = tuple(EFFECT_LABELS.index(label) for label in TARGET_EFFECT_LABELS)
_INTERNAL_ANCHOR = {
    "occupied": "occupied_object",
    "free": "empty",
    "none": "targetless",
}
_EXTERNAL_ANCHOR = {value: key for key, value in _INTERNAL_ANCHOR.items()}


def coarsen_anchor(anchor: str) -> str:
    """Project V4.1 anchors onto the frozen invariant three-state vocabulary."""
    if anchor in {"occupied_actor", "occupied_object"}:
        return "occupied"
    if anchor in {"empty", "open"}:
        return "free"
    if anchor in {"targetless", "unknown"}:
        return "none"
    if anchor in COARSE_ANCHOR_CONDITIONS:
        return anchor
    raise ValueError(f"unsupported V4.2 anchor: {anchor}")


def _target_map(values: Mapping[str, Any]) -> dict[str, bool]:
    return {label: bool(values[label]) for label in TARGET_EFFECT_LABELS}


def _strict_target_map(values: Mapping[str, Any]) -> dict[str, bool]:
    if set(values) != set(TARGET_EFFECT_LABELS):
        raise ValueError("V4.2 public payload requires only target effects")
    return _target_map(values)


def _padded_map(values: Mapping[str, Any]) -> dict[str, bool]:
    return {
        label: bool(values[label]) if label in TARGET_EFFECT_LABELS else False
        for label in EFFECT_LABELS
    }


@dataclass(frozen=True)
class TargetMechanicQuery:
    action_name: str
    action_family: str
    anchor_condition: str

    def __post_init__(self) -> None:
        if self.anchor_condition not in COARSE_ANCHOR_CONDITIONS:
            raise ValueError("unsupported V4.2 query anchor")

    def to_dict(self) -> dict[str, str]:
        return {
            "action_name": self.action_name,
            "action_family": self.action_family,
            "anchor_condition": self.anchor_condition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetMechanicQuery:
        return cls(
            action_name=str(payload["action_name"]),
            action_family=str(payload["action_family"]),
            anchor_condition=str(payload["anchor_condition"]),
        )

    def as_internal(self) -> v4.MechanicQuery:
        return v4.MechanicQuery(
            self.action_name,
            self.action_family,
            _INTERNAL_ANCHOR[self.anchor_condition],
        )


@dataclass(frozen=True)
class TargetTransitionEvent:
    action_name: str
    action_family: str
    anchor_condition: str
    effects: Mapping[str, bool]
    applicable: Mapping[str, bool]
    actor_role_known: bool
    actor_role_state: str

    def __post_init__(self) -> None:
        if self.anchor_condition not in COARSE_ANCHOR_CONDITIONS:
            raise ValueError("unsupported V4.2 event anchor")
        if set(self.effects) != set(TARGET_EFFECT_LABELS):
            raise ValueError("V4.2 event requires only target effects")
        if set(self.applicable) != set(TARGET_EFFECT_LABELS):
            raise ValueError("V4.2 event requires target applicability")

    def model_view(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_family": self.action_family,
            "anchor_condition": self.anchor_condition,
            "effects": _target_map(self.effects),
            "applicable": _target_map(self.applicable),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.model_view(),
            "actor_role_known": bool(self.actor_role_known),
            "actor_role_state": self.actor_role_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetTransitionEvent:
        return cls(
            action_name=str(payload["action_name"]),
            action_family=str(payload["action_family"]),
            anchor_condition=str(payload["anchor_condition"]),
            effects=_strict_target_map(dict(payload["effects"])),
            applicable=_strict_target_map(dict(payload["applicable"])),
            actor_role_known=bool(payload.get("actor_role_known", False)),
            actor_role_state=str(payload.get("actor_role_state", "ambiguous")),
        )

    @classmethod
    def from_v41(
        cls, event: v41.SemanticTransitionEvent
    ) -> TargetTransitionEvent:
        return cls(
            action_name=event.action_name,
            action_family=event.action_family,
            anchor_condition=coarsen_anchor(event.anchor_condition),
            effects=_target_map(event.effects),
            applicable=_target_map(event.applicable),
            actor_role_known=event.actor_role_known,
            actor_role_state=event.actor_role_state,
        )

    def as_internal(self) -> v41.SemanticTransitionEvent:
        return v41.SemanticTransitionEvent(
            action_name=self.action_name,
            action_family=self.action_family,
            anchor_condition=_INTERNAL_ANCHOR[self.anchor_condition],
            effects=_padded_map(self.effects),
            applicable=_padded_map(self.applicable),
            actor_role_known=self.actor_role_known,
            actor_role_state=self.actor_role_state,
        )


@dataclass(frozen=True)
class TargetMechanicWindowRecord:
    game_id: str
    source_split: str
    policy_seed: int
    reset_index: int
    query_step_index: int
    context: tuple[TargetTransitionEvent, ...]
    query: TargetMechanicQuery
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    actor_role_known: bool
    actor_role_state: str
    excluded_actor_displaced: bool
    excluded_actor_applicable: bool
    window_digest: str = ""
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 V4.2 window format")
        if set(self.labels) != set(TARGET_EFFECT_LABELS):
            raise ValueError("V4.2 window requires only target labels")
        if set(self.applicable) != set(TARGET_EFFECT_LABELS):
            raise ValueError("V4.2 window requires target applicability")
        if len(self.context) != 8:
            raise ValueError("V4.2 window requires eight context transitions")
        if not self.window_digest:
            payload = self.to_dict(include_digest=False)
            object.__setattr__(
                self,
                "window_digest",
                v41._checksum(payload),
            )

    @property
    def run_key(self) -> str:
        return f"{self.game_id}:{self.policy_seed}:{self.reset_index}"

    def model_view(self) -> dict[str, Any]:
        return {
            "context": [event.model_view() for event in self.context],
            "query": self.query.to_dict(),
        }

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": self.format_version,
            "game_id": self.game_id,
            "source_split": self.source_split,
            "policy_seed": int(self.policy_seed),
            "reset_index": int(self.reset_index),
            "query_step_index": int(self.query_step_index),
            "context": [event.to_dict() for event in self.context],
            "query": self.query.to_dict(),
            "labels": _target_map(self.labels),
            "applicable": _target_map(self.applicable),
            "actor_role_known": bool(self.actor_role_known),
            "actor_role_state": self.actor_role_state,
            "excluded_effect_audit": {
                "actor_displaced": bool(self.excluded_actor_displaced),
                "applicable": bool(self.excluded_actor_applicable),
            },
        }
        if include_digest:
            payload["window_digest"] = self.window_digest
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> TargetMechanicWindowRecord:
        audit = dict(payload.get("excluded_effect_audit", {}))
        return cls(
            format_version=str(payload.get("format_version", FORMAT_VERSION)),
            game_id=str(payload["game_id"]),
            source_split=str(payload["source_split"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            query_step_index=int(payload["query_step_index"]),
            context=tuple(
                TargetTransitionEvent.from_dict(item)
                for item in payload.get("context", ())
            ),
            query=TargetMechanicQuery.from_dict(dict(payload["query"])),
            labels=_strict_target_map(dict(payload["labels"])),
            applicable=_strict_target_map(dict(payload["applicable"])),
            actor_role_known=bool(payload.get("actor_role_known", False)),
            actor_role_state=str(payload.get("actor_role_state", "ambiguous")),
            excluded_actor_displaced=bool(audit.get("actor_displaced", False)),
            excluded_actor_applicable=bool(audit.get("applicable", False)),
            window_digest=str(payload.get("window_digest", "")),
        )

    @classmethod
    def from_v41(
        cls, window: v41.MechanicWindowRecord
    ) -> TargetMechanicWindowRecord:
        return cls(
            game_id=window.game_id,
            source_split=window.source_split,
            policy_seed=window.policy_seed,
            reset_index=window.reset_index,
            query_step_index=window.query_step_index,
            context=tuple(
                TargetTransitionEvent.from_v41(event) for event in window.context
            ),
            query=TargetMechanicQuery(
                window.query.action_name,
                window.query.action_family,
                coarsen_anchor(window.query.anchor_condition),
            ),
            labels=_target_map(window.labels),
            applicable=_target_map(window.applicable),
            actor_role_known=window.actor_role_known,
            actor_role_state=window.actor_role_state,
            excluded_actor_displaced=bool(window.labels["actor_displaced"]),
            excluded_actor_applicable=bool(window.applicable["actor_displaced"]),
        )

    def as_internal(self) -> v41.MechanicWindowRecord:
        return v41.MechanicWindowRecord(
            game_id=self.game_id,
            source_split=self.source_split,
            policy_seed=self.policy_seed,
            reset_index=self.reset_index,
            query_step_index=self.query_step_index,
            context=tuple(event.as_internal() for event in self.context),
            query=self.query.as_internal(),
            labels=_padded_map(self.labels),
            applicable=_padded_map(self.applicable),
            actor_role_known=self.actor_role_known,
            actor_role_state=self.actor_role_state,
            window_digest=self.window_digest,
        )


def build_target_windows(
    traces: Sequence[ActionTargetTrace],
    *,
    context_length: int = 8,
) -> list[TargetMechanicWindowRecord]:
    if int(context_length) != 8:
        raise ValueError("V4.2 freezes context length at eight")
    return [
        TargetMechanicWindowRecord.from_v41(window)
        for window in v41.build_mechanic_windows(
            traces,
            context_length=context_length,
        )
    ]


def validate_model_view(window: TargetMechanicWindowRecord) -> None:
    rendered = v41._canonical(window.model_view()).lower()
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
        "actor_displaced",
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
        raise ValueError("V4.2 model view contains forbidden provenance")


@dataclass(frozen=True)
class TargetCalibrationBundle:
    parameters: Mapping[str, Mapping[str, Mapping[str, float]]]
    thresholds: Mapping[str, Mapping[str, float]]
    source_oof_metrics: Mapping[str, Any]
    format_version: str = CALIBRATION_FORMAT_VERSION
    calibration_checksum: str = ""

    def __post_init__(self) -> None:
        if self.format_version != CALIBRATION_FORMAT_VERSION:
            raise ValueError("unsupported V4.2 calibration format")
        for mode in MODEL_MODES:
            if set(self.parameters[mode]) != set(TARGET_EFFECT_LABELS):
                raise ValueError("V4.2 calibration contains non-target effects")
            if set(self.thresholds[mode]) != set(TARGET_EFFECT_LABELS):
                raise ValueError("V4.2 thresholds contain non-target effects")
        if not self.calibration_checksum:
            object.__setattr__(
                self,
                "calibration_checksum",
                v41._checksum(self.to_dict(include_checksum=False)),
            )

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
                mode: {label: float(value) for label, value in labels.items()}
                for mode, labels in self.thresholds.items()
            },
            "source_oof_metrics": self.source_oof_metrics,
        }
        if include_checksum:
            payload["calibration_checksum"] = self.calibration_checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetCalibrationBundle:
        expected = str(payload.get("calibration_checksum", ""))
        bundle = cls(
            format_version=str(payload["format_version"]),
            parameters=dict(payload["parameters"]),
            thresholds=dict(payload["thresholds"]),
            source_oof_metrics=dict(payload["source_oof_metrics"]),
        )
        if expected and expected != bundle.calibration_checksum:
            raise ValueError("V4.2 calibration checksum mismatch")
        return bundle


def _targets_masks(
    windows: Sequence[TargetMechanicWindowRecord],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [
            [int(window.labels[label]) for label in TARGET_EFFECT_LABELS]
            for window in windows
        ],
        dtype=np.int8,
    )
    masks = np.asarray(
        [
            [int(window.applicable[label]) for label in TARGET_EFFECT_LABELS]
            for window in windows
        ],
        dtype=np.int8,
    )
    return targets, masks


def _fit_priors(
    windows: Sequence[TargetMechanicWindowRecord],
) -> dict[str, dict[str, int]]:
    return v4.fit_source_priors([window.as_internal() for window in windows])


def _raw_matrices(
    windows: Sequence[TargetMechanicWindowRecord],
    priors: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, np.ndarray], list[tuple[v4.MechanicEvidence, ...]]]:
    matrices, evidence = v41._raw_matrices(
        [window.as_internal() for window in windows],
        priors,
    )
    projected = {
        mode: matrix[:, _TARGET_INDICES] for mode, matrix in matrices.items()
    }
    filtered = [
        tuple(
            item for item in row if item.rule.effect in TARGET_EFFECT_LABELS
        )
        for row in evidence
    ]
    return projected, filtered


def apply_calibration(
    matrix: np.ndarray,
    bundle: TargetCalibrationBundle,
    mode: str,
) -> np.ndarray:
    calibrated = np.asarray(matrix, dtype=np.float64).copy()
    for index, label in enumerate(TARGET_EFFECT_LABELS):
        calibrated[:, index] = v41._apply_parameter(
            calibrated[:, index],
            bundle.parameters[mode][label],
        )
    return calibrated


def multilabel_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {
        label: 0.5 for label in TARGET_EFFECT_LABELS
    }
    per_label: dict[str, Any] = {}
    for index, label in enumerate(TARGET_EFFECT_LABELS):
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
            "brier": (
                float(np.mean((y_prob - y_true) ** 2))
                if len(y_true)
                else 0.0
            ),
            "ece": v4._ece(y_true, y_prob),
        }
    return {
        "macro_f1": float(np.mean([item["f1"] for item in per_label.values()])),
        "macro_brier": float(
            np.mean([item["brier"] for item in per_label.values()])
        ),
        "macro_ece": float(
            np.mean([item["ece"] for item in per_label.values()])
        ),
        "per_label": per_label,
    }


def fit_source_calibration(
    windows: Sequence[TargetMechanicWindowRecord],
) -> TargetCalibrationBundle:
    targets, masks = _targets_masks(windows)
    matrices = {
        mode: np.zeros(
            (len(windows), len(TARGET_EFFECT_LABELS)),
            dtype=np.float64,
        )
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
        priors = _fit_priors(train)
        held = [windows[index] for index in held_indices]
        fold_matrices, _ = _raw_matrices(held, priors)
        for mode in MODEL_MODES:
            matrices[mode][held_indices] = fold_matrices[mode]
        assigned[held_indices] = True
    if not bool(np.all(assigned)):
        raise ValueError("V4.2 source OOF calibration left windows unassigned")

    parameters: dict[str, dict[str, dict[str, float]]] = {}
    thresholds: dict[str, dict[str, float]] = {}
    raw_metrics: dict[str, Any] = {}
    calibrated_metrics: dict[str, Any] = {}
    for mode in MODEL_MODES:
        parameters[mode] = {}
        thresholds[mode] = {}
        calibrated = matrices[mode].copy()
        for index, label in enumerate(TARGET_EFFECT_LABELS):
            selected = masks[:, index].astype(bool)
            parameter = v41._fit_platt(
                matrices[mode][selected, index],
                targets[selected, index],
            )
            parameters[mode][label] = parameter
            calibrated[:, index] = v41._apply_parameter(
                matrices[mode][:, index],
                parameter,
            )
            thresholds[mode][label] = v41._select_threshold(
                calibrated[selected, index],
                targets[selected, index],
            )
        raw_metrics[mode] = multilabel_metrics(
            targets,
            masks,
            matrices[mode],
        )
        calibrated_metrics[mode] = multilabel_metrics(
            targets,
            masks,
            calibrated,
            thresholds=thresholds[mode],
        )
    return TargetCalibrationBundle(
        parameters=parameters,
        thresholds=thresholds,
        source_oof_metrics={
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
    )


def compact_qwen_prompt(window: TargetMechanicWindowRecord) -> str:
    codes = {
        "target_created": "C",
        "target_removed": "R",
        "target_moved": "M",
    }
    rows = []
    for event in window.context:
        effects = "".join(
            codes[label]
            for label in TARGET_EFFECT_LABELS
            if event.applicable[label] and event.effects[label]
        ) or "-"
        applicable = "".join(
            codes[label]
            for label in TARGET_EFFECT_LABELS
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
        "Anchors occupied,free,none. Effects C=target_created,"
        "R=target_removed,M=target_moved. s=e exact,s=f family; "
        "z must be 0. JSON only.\n"
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
                        "a": {"enum": list(COARSE_ANCHOR_CONDITIONS)},
                        "e": {"enum": list(TARGET_EFFECT_LABELS)},
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
    query: TargetMechanicQuery,
) -> v4.MechanicRule:
    if set(payload) != {"s", "v", "a", "e", "z"}:
        raise ValueError("V4.2 compact rule has unexpected fields")
    kind = {"e": "exact", "f": "family"}.get(str(payload["s"]))
    if kind is None:
        raise ValueError("V4.2 compact rule has invalid scope")
    value = str(payload["v"])
    allowed = query.action_name if kind == "exact" else query.action_family
    if value != allowed:
        raise ValueError("V4.2 compact rule is not grounded")
    anchor = str(payload["a"])
    effect = str(payload["e"])
    if anchor not in COARSE_ANCHOR_CONDITIONS:
        raise ValueError("V4.2 compact rule has invalid anchor")
    if effect not in TARGET_EFFECT_LABELS:
        raise ValueError("V4.2 compact rule has invalid effect")
    if int(payload["z"]) != 0:
        raise ValueError("V4.2 proposal support must be zero")
    internal_anchor = _INTERNAL_ANCHOR[anchor]
    return v4.MechanicRule(
        rule_id=v4._rule_id(kind, value, internal_anchor, effect),
        action_scope_kind=kind,
        action_scope_value=value,
        anchor_condition=internal_anchor,
        effect=effect,
        support=0,
        source="local_llm",
    )


def _public_rule(rule: v4.MechanicRule) -> dict[str, Any]:
    payload = rule.to_dict()
    payload["anchor_condition"] = _EXTERNAL_ANCHOR[rule.anchor_condition]
    return payload


def _chat_token_count(
    tokenizer: Any,
    window: TargetMechanicWindowRecord,
) -> int:
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
    windows: Sequence[TargetMechanicWindowRecord],
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


def _window_quality(
    windows: Sequence[TargetMechanicWindowRecord],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    for label in TARGET_EFFECT_LABELS:
        eligible = [window for window in windows if window.applicable[label]]
        positives = sum(int(window.labels[label]) for window in eligible)
        per_label[label] = {
            "applicable": len(eligible),
            "positives": positives,
            "negatives": len(eligible) - positives,
        }
    per_game: dict[str, Any] = {}
    for game in sorted({window.game_id for window in windows}):
        selected = [window for window in windows if window.game_id == game]
        states = Counter(window.actor_role_state for window in selected)
        per_game[game] = {
            "windows": len(selected),
            "actor_role_resolved_rate": sum(
                int(window.actor_role_state != "ambiguous") for window in selected
            )
            / max(1, len(selected)),
            "actor_role_states": dict(sorted(states.items())),
        }
    actor_eligible = [
        window for window in windows if window.excluded_actor_applicable
    ]
    return {
        "windows": len(windows),
        "unique_window_digests": len({window.window_digest for window in windows}),
        "per_label": per_label,
        "per_game": per_game,
        "excluded_effect_audit": {
            "effect": "actor_displaced",
            "applicable": len(actor_eligible),
            "positives": sum(
                int(window.excluded_actor_displaced) for window in actor_eligible
            ),
            "authority": False,
        },
    }


def _label_capacity(
    per_label: Mapping[str, Mapping[str, int]],
    minimum_positive: int,
    minimum_negative: int,
) -> bool:
    return all(
        int(item["positives"]) >= minimum_positive
        and int(item["negatives"]) >= minimum_negative
        for item in per_label.values()
    )


def _identity_probe(
    windows: Sequence[TargetMechanicWindowRecord],
) -> dict[str, Any]:
    labels = [window.game_id for window in windows]
    action = v4._identity_probe(
        [{f"action:{window.query.action_name}": 1} for window in windows],
        labels,
    )
    static = v4._identity_probe(
        [
            {
                f"action:{window.query.action_name}": 1,
                f"family:{window.query.action_family}": 1,
                f"anchor:{window.query.anchor_condition}": 1,
            }
            for window in windows
        ],
        labels,
    )
    return {
        "action_only": action,
        "static": static,
        "gain": static["accuracy"] - action["accuracy"],
    }


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = v41._read_json(Path(path))
    expected = str(payload.get("manifest_checksum", ""))
    check = dict(payload)
    check.pop("manifest_checksum", None)
    actual = v41._checksum(check)
    if expected != actual:
        raise ValueError(
            f"V4.2 frozen-manifest checksum mismatch: {actual} != {expected}"
        )
    if payload.get("format_version") != "sage12-mechanic-induction-v4.2":
        raise ValueError("unsupported SAGE12 V4.2 manifest")
    if tuple(payload["effects"]["authoritative"]) != TARGET_EFFECT_LABELS:
        raise ValueError("V4.2 manifest changes the authoritative effects")
    return payload


def run_source_train_preflight(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    traces = v41._load_traces(V3_OUTPUT_DIR / "shards", SOURCE_TRAIN)
    windows = build_target_windows(
        traces,
        context_length=int(frozen["window"]["context_length"]),
    )
    for window in windows:
        validate_model_view(window)
    priors = _fit_priors(windows)
    calibration = fit_source_calibration(windows)
    v41._write_jsonl_dicts(
        destination / "source_train_windows.jsonl",
        [window.to_dict() for window in windows],
    )
    priors_payload: dict[str, Any] = {
        "format_version": "sage12-target-mechanic-priors-v4.2",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "prior_strength": float(frozen["model"]["prior_strength"]),
        "internal_anchor_adapter": dict(_INTERNAL_ANCHOR),
        "counts": priors,
    }
    priors_payload["priors_checksum"] = v41._checksum(priors_payload)
    v41._write_json(destination / "source_priors.json", priors_payload)
    v41._write_json(destination / "calibration.json", calibration.to_dict())

    quality = _window_quality(windows)
    identity = _identity_probe(windows)
    token_budget = measure_qwen_token_budget(windows, frozen)
    raw = calibration.source_oof_metrics["raw"]
    calibrated = calibration.source_oof_metrics["calibrated"]
    stronger = min(
        BASELINE_MODES,
        key=lambda mode: calibrated[mode]["macro_brier"],
    )
    source_skill = v4._brier_skill(
        calibrated["structured"],
        calibrated[stronger],
    )
    source_f1_gain = (
        calibrated["structured"]["macro_f1"]
        - calibrated[stronger]["macro_f1"]
    )
    source_context_gain = v4._brier_skill(
        calibrated["structured"],
        calibrated["context_ablation"],
    )
    gates_cfg = frozen["gates"]
    gates = {
        "minimum_source_train_windows": len(windows)
        >= int(gates_cfg["minimum_source_train_windows"]),
        "source_train_label_capacity": _label_capacity(
            quality["per_label"],
            int(gates_cfg["minimum_source_train_positives_per_label"]),
            int(gates_cfg["minimum_source_train_negatives_per_label"]),
        ),
        "static_identity_leakage": identity["gain"]
        <= float(gates_cfg["maximum_static_identity_gain_over_action"]),
        "source_oof_calibration": calibrated["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_source_oof_macro_ece"]),
        "source_oof_brier_non_degradation": (
            calibrated["structured"]["macro_brier"]
            - raw["structured"]["macro_brier"]
        )
        <= float(gates_cfg["maximum_source_oof_brier_degradation"]),
        "source_oof_brier_skill": source_skill
        >= float(gates_cfg["minimum_source_macro_brier_skill"]),
        "source_oof_macro_f1_gain": source_f1_gain
        >= float(gates_cfg["minimum_source_macro_f1_gain"]),
        "source_context_brier_skill_gain": source_context_gain
        >= float(gates_cfg["minimum_source_context_brier_skill_gain"]),
        "qwen_prompt_budget": token_budget["maximum_tokens"]
        <= int(frozen["qwen"]["preflight_maximum_input_tokens"]),
        "model_view_firewall": True,
        "actor_effect_excluded": True,
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
        "identity_probe": identity,
        "source_oof": calibration.source_oof_metrics,
        "source_stronger_baseline": stronger,
        "source_macro_brier_skill": source_skill,
        "source_macro_f1_gain": source_f1_gain,
        "source_context_brier_skill_gain": source_context_gain,
        "calibration_checksum": calibration.calibration_checksum,
        "priors_checksum": priors_payload["priors_checksum"],
        "qwen_token_budget": token_budget,
        "gates": gates,
        "source_validation_opened": False,
        "v5_protocol_authorized": False,
        "world_model_fit_authorized": False,
    }
    payload["preflight_checksum"] = v41._checksum(payload)
    v41._write_json(destination / "source_train_preflight.json", payload)
    return payload


def _shuffle_context(
    windows: Sequence[TargetMechanicWindowRecord],
    *,
    binding: bool,
) -> list[TargetMechanicWindowRecord]:
    shuffled = []
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
                    )
                    for event, source in zip(events, rotated)
                ]
        shuffled.append(
            replace(window, context=tuple(events), window_digest="")
        )
    return shuffled


def _bootstrap_skill(
    windows: Sequence[TargetMechanicWindowRecord],
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


def _output_contract(
    windows: Sequence[TargetMechanicWindowRecord],
    evidence_rows: Sequence[Sequence[v4.MechanicEvidence]],
) -> dict[str, Any]:
    emitted = valid = support_zero = grounded = 0
    for window, row in zip(windows, evidence_rows):
        internal_query = window.query.as_internal()
        for evidence in row:
            if evidence.rule.effect not in TARGET_EFFECT_LABELS:
                raise ValueError("V4.2 emitted a non-target effect")
            emitted += 1
            restored = v4.MechanicRule.from_dict(
                json.loads(v41._canonical(evidence.rule.to_dict()))
            )
            valid += 1
            support_zero += int(restored.support == 0)
            hypothesis = v4.rule_to_semantic_hypothesis(
                restored,
                internal_query,
                confidence=evidence.posterior_probability,
            )
            grounded += int(
                hypothesis.action_name == window.query.action_name
                and restored.matches_query(internal_query)
            )
    denominator = max(1, emitted)
    return {
        "emitted_hypotheses": emitted,
        "strict_json_validity": valid / denominator,
        "support_zero_rate": support_zero / denominator,
        "grounded_hypothesis_rate": grounded / denominator,
        "actor_effect_emitted": False,
    }


def _qwen_indices(
    windows: Sequence[TargetMechanicWindowRecord],
    count: int,
) -> list[int]:
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
            index for index in range(len(windows)) if index not in chosen
        )
    return selected[:count]


def _qwen_generate(
    model: TransformersJSONModel,
    windows: Sequence[TargetMechanicWindowRecord],
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
                raise ValueError("V4.2 compact response must contain only h")
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
                "hypotheses": [_public_rule(rule) for rule in rules],
            }
        )
    return rows, outputs


def _score_qwen_rules(
    windows: Sequence[TargetMechanicWindowRecord],
    indices: Sequence[int],
    rules: Sequence[Sequence[v4.MechanicRule]],
    priors: Mapping[str, Mapping[str, int]],
) -> np.ndarray:
    matrix = []
    for index, emitted in zip(indices, rules):
        window = windows[index]
        internal = window.as_internal()
        probabilities = {
            effect: v4._action_only_probability(
                internal,
                effect,
                priors,
                use_context=False,
            )
            for effect in TARGET_EFFECT_LABELS
        }
        for rule in emitted:
            if rule.effect not in TARGET_EFFECT_LABELS:
                raise ValueError("Qwen emitted an unauthorized effect")
            if rule.matches_query(internal.query):
                evidence = v4.score_rule(rule, internal.context, priors)
                probabilities[rule.effect] = evidence.posterior_probability
        matrix.append(
            [probabilities[label] for label in TARGET_EFFECT_LABELS]
        )
    return np.asarray(matrix, dtype=np.float64)


def _evaluate_qwen(
    windows: Sequence[TargetMechanicWindowRecord],
    priors: Mapping[str, Mapping[str, int]],
    bundle: TargetCalibrationBundle,
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
    model = TransformersJSONModel(
        TransformersModelConfig(
            model_path=str(frozen["qwen"]["path"]),
            device=str(frozen["qwen"]["device"]),
            temperature=0.0,
            maximum_input_tokens=int(frozen["qwen"]["maximum_input_tokens"]),
        )
    )
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
        shuffled_windows,
        indices,
        shuffled_rules,
        priors,
    )
    shuffled = apply_calibration(shuffled_raw, bundle, "structured")
    shuffled_metrics = multilabel_metrics(
        targets[subset],
        masks[subset],
        shuffled,
        thresholds=bundle.thresholds["structured"],
    )
    shuffle_drop = skill - v4._brier_skill(
        shuffled_metrics,
        baseline_metrics,
    )

    valid = sum(not item["parse_error"] for item in outputs)
    emitted = sum(len(row) for row in rules)
    grounded = sum(
        int(rule.matches_query(windows[index].query.as_internal()))
        for index, row in zip(indices, rules)
        for rule in row
    )
    support_zero = sum(
        int(rule.support == 0) for row in rules for rule in row
    )
    productive = recalled = 0
    for index, row in zip(indices, rules):
        emitted_effects = {rule.effect for rule in row}
        for label_index, label in enumerate(TARGET_EFFECT_LABELS):
            if masks[index, label_index] and targets[index, label_index]:
                productive += 1
                recalled += int(label in emitted_effects)
    per_game = {}
    for game in SOURCE_VALIDATION:
        positions = np.asarray(
            [
                position
                for position, index in enumerate(indices)
                if windows[index].game_id == game
            ],
            dtype=int,
        )
        if not len(positions):
            continue
        game_metrics = multilabel_metrics(
            targets[subset][positions],
            masks[subset][positions],
            calibrated[positions],
            thresholds=bundle.thresholds["structured"],
        )
        game_baseline = multilabel_metrics(
            targets[subset][positions],
            masks[subset][positions],
            baseline[subset][positions],
            thresholds=bundle.thresholds[baseline_name],
        )
        per_game[game] = {
            "windows": len(positions),
            "brier_skill": v4._brier_skill(game_metrics, game_baseline),
        }
    v41._write_jsonl_dicts(output_dir / "qwen_outputs.jsonl", outputs)
    v41._write_jsonl_dicts(
        output_dir / "qwen_outcome_shuffle_outputs.jsonl",
        shuffled_outputs,
    )
    rates = {
        "strict_json_validity": valid / max(1, len(indices)),
        "grounded_hypothesis_rate": grounded / max(1, emitted),
        "support_zero_rate": support_zero / max(1, emitted),
        "productive_effect_recall_at_8": recalled / max(1, productive),
    }
    gate_cfg = frozen["qwen_gates"]
    gates = {
        "strict_json_validity": rates["strict_json_validity"]
        >= float(gate_cfg["minimum_strict_json_validity"]),
        "grounded_hypothesis_rate": rates["grounded_hypothesis_rate"]
        >= float(gate_cfg["minimum_grounded_hypothesis_rate"]),
        "support_zero_rate": rates["support_zero_rate"] == 1.0,
        "productive_effect_recall_at_8": rates[
            "productive_effect_recall_at_8"
        ]
        >= float(gate_cfg["minimum_productive_effect_recall_at_8"]),
        "outcome_shuffle_drop": shuffle_drop
        >= float(gate_cfg["minimum_outcome_shuffle_skill_drop"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
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
        "gates": gates,
        "inference_seconds": time.perf_counter() - started,
        "device": str(frozen["qwen"]["device"]),
        "model": str(frozen["qwen"]["name"]),
    }


def _effect_authority(
    metrics: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    authority: dict[str, Any] = {}
    for label in TARGET_EFFECT_LABELS:
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
            "positives": int(capacity["positives"]),
            "negatives": int(capacity["negatives"]),
            "eligible_for_v5": bool(eligible),
        }
    authority["actor_displaced"] = {
        "method": "excluded",
        "eligible_for_v5": False,
        "reason": "outside V4.2 authoritative effect vocabulary",
    }
    return authority


def _validate_collection(
    destination: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    path = destination / "collection_manifest.json"
    payload = v41._read_json(path)
    if payload.get("format_version") != (
        "sage12-target-mechanic-collection-v4.2"
    ):
        raise ValueError("unsupported V4.2 collection manifest")
    if payload.get("status") != "COMPLETE":
        raise ValueError("V4.2 collection is incomplete")
    if payload.get("frozen_manifest_checksum") != frozen["manifest_checksum"]:
        raise ValueError("V4.2 collection/manifest checksum mismatch")
    if int(payload.get("rows", 0)) != int(
        frozen["collection"]["prospective_rows"]
    ):
        raise ValueError("V4.2 collection row count mismatch")
    expected = str(payload.get("report_checksum", ""))
    check = dict(payload)
    check.pop("report_checksum", None)
    if expected != v41._checksum(check):
        raise ValueError("V4.2 collection checksum mismatch")
    return payload


def run_evaluation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    run_qwen: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    preflight = v41._read_json(destination / "source_train_preflight.json")
    if (
        preflight.get("status") != "PASS_SOURCE_TRAIN_PREFLIGHT"
        or not all(dict(preflight.get("gates", {})).values())
    ):
        raise RuntimeError("V4.2 preflight did not authorize evaluation")
    if preflight.get("frozen_manifest_checksum") != frozen["manifest_checksum"]:
        raise RuntimeError("V4.2 preflight/manifest mismatch")
    _validate_collection(destination, frozen)
    priors_payload = v41._read_json(destination / "source_priors.json")
    priors = dict(priors_payload["counts"])
    bundle = TargetCalibrationBundle.from_dict(
        v41._read_json(destination / "calibration.json")
    )
    traces = v41._load_traces(destination / "shards", SOURCE_VALIDATION)
    windows = build_target_windows(
        traces,
        context_length=int(frozen["window"]["context_length"]),
    )
    for window in windows:
        validate_model_view(window)
    v41._write_jsonl_dicts(
        destination / "validation_windows.jsonl",
        [window.to_dict() for window in windows],
    )
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
    stronger = min(
        BASELINE_MODES,
        key=lambda mode: calibrated_metrics[mode]["macro_brier"],
    )
    raw_skill = v4._brier_skill(
        raw_metrics["structured"],
        raw_metrics[stronger],
    )
    calibrated_skill = v4._brier_skill(
        calibrated_metrics["structured"],
        calibrated_metrics[stronger],
    )
    f1_gain = (
        calibrated_metrics["structured"]["macro_f1"]
        - calibrated_metrics[stronger]["macro_f1"]
    )

    outcome_windows = _shuffle_context(windows, binding=False)
    outcome_raw, _ = _raw_matrices(outcome_windows, priors)
    outcome_matrix = apply_calibration(
        outcome_raw["structured"],
        bundle,
        "structured",
    )
    outcome_metrics = multilabel_metrics(
        targets,
        masks,
        outcome_matrix,
        thresholds=bundle.thresholds["structured"],
    )
    outcome_drop = calibrated_skill - v4._brier_skill(
        outcome_metrics,
        calibrated_metrics[stronger],
    )
    binding_windows = _shuffle_context(windows, binding=True)
    binding_raw, _ = _raw_matrices(binding_windows, priors)
    binding_matrix = apply_calibration(
        binding_raw["structured"],
        bundle,
        "structured",
    )
    binding_metrics = multilabel_metrics(
        targets,
        masks,
        binding_matrix,
        thresholds=bundle.thresholds["structured"],
    )
    binding_skill = v4._brier_skill(
        binding_metrics,
        calibrated_metrics[stronger],
    )
    binding_drop = calibrated_skill - binding_skill
    context_gain = v4._brier_skill(
        calibrated_metrics["structured"],
        calibrated_metrics["context_ablation"],
    )
    bootstrap = _bootstrap_skill(
        windows,
        targets,
        masks,
        calibrated_matrices["structured"],
        calibrated_matrices[stronger],
        model_thresholds=bundle.thresholds["structured"],
        baseline_thresholds=bundle.thresholds[stronger],
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
        candidates = {
            mode: multilabel_metrics(
                targets[selected],
                masks[selected],
                calibrated_matrices[mode][selected],
                thresholds=bundle.thresholds[mode],
            )
            for mode in BASELINE_MODES
        }
        game_baseline = min(
            BASELINE_MODES,
            key=lambda mode: candidates[mode]["macro_brier"],
        )
        per_game[game] = {
            "windows": int(np.sum(selected)),
            "structured": model_metric,
            "stronger_baseline": game_baseline,
            "baseline": candidates[game_baseline],
            "brier_skill": v4._brier_skill(
                model_metric,
                candidates[game_baseline],
            ),
        }
    quality = _window_quality(windows)
    identity = _identity_probe(windows)
    output_contract = _output_contract(windows, evidence_rows)
    if run_qwen:
        try:
            qwen = _evaluate_qwen(
                windows,
                priors,
                bundle,
                frozen,
                targets,
                masks,
                calibrated_matrices[stronger],
                stronger,
                output_dir=destination,
            )
        except Exception as exc:  # noqa: BLE001 - separate fail-closed branch
            qwen = {
                "status": "FAIL_RUNTIME",
                "authority_separate": True,
                "error": f"{type(exc).__name__}: {exc}",
                "device": str(frozen["qwen"]["device"]),
            }
    else:
        qwen = {"status": "SKIPPED", "authority_separate": True}
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
        "minimum_binding_shuffle_drop": binding_drop
        >= float(gates_cfg["minimum_binding_shuffle_skill_drop"]),
        "minimum_context_gain": context_gain
        >= float(gates_cfg["minimum_context_brier_skill_gain"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
        "maximum_macro_ece": calibrated_metrics["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "maximum_prospective_identity_gain": identity["gain"]
        <= float(gates_cfg["maximum_static_identity_gain_over_action"]),
        "effect_authority": all(
            effect_authority[label]["eligible_for_v5"]
            for label in TARGET_EFFECT_LABELS
        ),
        "source_preflight_passed": True,
        "actor_effect_excluded": True,
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
                "labels": dict(window.labels),
                "applicable": dict(window.applicable),
                "raw_structured_probabilities": {
                    label: float(raw_matrices["structured"][index, label_index])
                    for label_index, label in enumerate(TARGET_EFFECT_LABELS)
                },
                "calibrated_structured_probabilities": {
                    label: float(
                        calibrated_matrices["structured"][index, label_index]
                    )
                    for label_index, label in enumerate(TARGET_EFFECT_LABELS)
                },
                "evidence": [
                    {
                        **item.to_dict(),
                        "rule": _public_rule(item.rule),
                    }
                    for item in evidence_rows[index]
                ],
            }
        )
    v41._write_jsonl_dicts(destination / "predictions.jsonl", predictions)
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
        "identity_probe": identity,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "stronger_baseline": stronger,
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
            "skill": binding_skill,
            "skill_drop": binding_drop,
        },
        "context_brier_skill_gain": context_gain,
        "per_game": per_game,
        "effect_authority": effect_authority,
        "output_contract": output_contract,
        "qwen": qwen,
        "gates": gates,
        "firewall": {
            "source_only_calibration": True,
            "actor_effect_modelled": False,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
            "controller_executed": False,
        },
        "v5_protocol_authorized": passed,
        "qwen_v5_protocol_authorized": bool(
            passed and qwen.get("status") == "PASS"
        ),
        "world_model_fit_authorized": False,
        "ebm_fit_authorized": False,
    }
    payload["result_checksum"] = v41._checksum(payload)
    v41._write_json(destination / "pilot_result.json", payload)
    return payload


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
    "COARSE_ANCHOR_CONDITIONS",
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "TARGET_EFFECT_LABELS",
    "TargetCalibrationBundle",
    "TargetMechanicQuery",
    "TargetMechanicWindowRecord",
    "TargetTransitionEvent",
    "apply_calibration",
    "build_target_windows",
    "coarsen_anchor",
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
