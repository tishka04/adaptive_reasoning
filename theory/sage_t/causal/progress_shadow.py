"""Effect-grounded shadow ranking primitives for SAGE.T12.5b.

This module contains no environment access.  It learns an intervention-effect
table from one exact replay lineage, transports it to another lineage, and
scores every candidate through the frozen T12.5 progress posterior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .progress import (
    CausalProgressProgram,
    JointCausalProgressPosterior,
    JointProgressParticle,
    ProgressMilestone,
)

EFFECT_MODEL_FORMAT = "sage-t12.5b-empirical-action-effect-model-v1"


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _feature_parts(feature: str) -> tuple[str, str]:
    family, separator, key = str(feature).partition(".")
    if not separator or family not in {"role_counts", "predicate_counts"} or not key:
        raise ValueError(f"invalid progress-shadow feature: {feature}")
    return family, key


def projected_effect_step(
    step: Mapping[str, Any],
    *,
    features: Sequence[str],
) -> dict[str, Any]:
    """Keep only preregistered typed deltas; discard hashes and identities."""

    mechanism = dict(dict(step.get("delta", {})).get("mechanism", {}))
    projected: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for feature in features:
        family, key = _feature_parts(feature)
        raw = dict(dict(mechanism.get(family, {})).get(key, {}))
        before = int(raw.get("before", 0))
        after = int(raw.get("after", before))
        if before != after:
            projected[family][key] = {"after": after, "before": before}
    delta = {
        "exact_changed": bool(dict(step.get("delta", {})).get("exact_changed", False)),
        "mechanism": {family: dict(values) for family, values in projected.items()},
        "mechanism_empty": not any(projected.values()),
    }
    return {
        "action_name": str(step.get("action_name", "")).upper(),
        "available": bool(step.get("available", True)),
        "delta": delta,
        "position": int(step.get("position", 0)),
    }


def projection_vector(
    step: Mapping[str, Any], *, features: Sequence[str]
) -> tuple[int, ...]:
    mechanism = dict(dict(step.get("delta", {})).get("mechanism", {}))
    values = []
    for feature in features:
        family, key = _feature_parts(feature)
        raw = dict(dict(mechanism.get(family, {})).get(key, {}))
        values.append(int(raw.get("after", 0)) - int(raw.get("before", 0)))
    return tuple(values)


def step_from_projection(
    *,
    action_name: str,
    vector: Sequence[int],
    features: Sequence[str],
    position: int = 0,
) -> dict[str, Any]:
    if len(vector) != len(features):
        raise ValueError("effect projection has the wrong dimension")
    mechanism: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for feature, raw_delta in zip(features, vector):
        delta = int(raw_delta)
        if delta == 0:
            continue
        family, key = _feature_parts(feature)
        mechanism[family][key] = {"after": delta, "before": 0}
    return {
        "action_name": str(action_name).upper(),
        "available": True,
        "delta": {
            "exact_changed": bool(any(int(value) != 0 for value in vector)),
            "mechanism": {family: dict(values) for family, values in mechanism.items()},
            "mechanism_empty": not any(mechanism.values()),
        },
        "position": int(position),
    }


def progress_signature(
    step: Mapping[str, Any], milestones: Sequence[ProgressMilestone]
) -> tuple[bool, ...]:
    return tuple(milestone.matches(step) for milestone in milestones)


@dataclass(frozen=True)
class EffectPrediction:
    stage: int
    action_name: str
    projection: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    deterministic: bool

    @property
    def changed(self) -> bool:
        return any(value != 0 for value in self.projection)

    @property
    def magnitude(self) -> float:
        return float(sum(abs(value) for value in self.projection))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "changed": self.changed,
            "deterministic": self.deterministic,
            "evidence_ids": list(self.evidence_ids),
            "magnitude": self.magnitude,
            "projection": list(self.projection),
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EffectPrediction:
        return cls(
            stage=int(payload["stage"]),
            action_name=str(payload["action_name"]),
            projection=tuple(int(item) for item in payload.get("projection", ())),
            evidence_ids=tuple(map(str, payload.get("evidence_ids", ()))),
            deterministic=bool(payload["deterministic"]),
        )


@dataclass(frozen=True)
class EmpiricalActionEffectModel:
    features: tuple[str, ...]
    predictions: tuple[EffectPrediction, ...]
    induction_lineage_seed: int
    format_version: str = EFFECT_MODEL_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(map(str, self.features)))
        object.__setattr__(self, "predictions", tuple(self.predictions))
        if self.format_version != EFFECT_MODEL_FORMAT:
            raise ValueError("unsupported progress-shadow effect model")
        keys = {(item.stage, item.action_name) for item in self.predictions}
        if len(keys) != len(self.predictions):
            raise ValueError("duplicate progress-shadow effect prediction")

    @classmethod
    def fit(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        features: Sequence[str],
        induction_lineage_seed: int,
        expected_stages: Sequence[int],
        candidate_actions: Sequence[str],
    ) -> EmpiricalActionEffectModel:
        groups: dict[tuple[int, str], list[tuple[tuple[int, ...], str]]] = defaultdict(list)
        for record in records:
            if int(record.get("lineage_seed", -1)) != int(induction_lineage_seed):
                continue
            if not record.get("prefix_exact") or not record.get("branch_available"):
                raise ValueError("cannot fit effects from inexact or unavailable branches")
            key = (int(record["stage"]), str(record["action_name"]).upper())
            vector = projection_vector(
                dict(record["candidate_step"]), features=features
            )
            groups[key].append((vector, str(record["trial_id"])))
        expected = {
            (int(stage), str(action).upper())
            for stage in expected_stages
            for action in candidate_actions
        }
        if set(groups) != expected:
            raise ValueError("induction effect table is incomplete")
        predictions = []
        for key in sorted(groups):
            observations = groups[key]
            unique = {item[0] for item in observations}
            predictions.append(
                EffectPrediction(
                    stage=key[0],
                    action_name=key[1],
                    projection=observations[0][0],
                    evidence_ids=tuple(sorted(item[1] for item in observations)),
                    deterministic=len(unique) == 1,
                )
            )
        return cls(
            features=tuple(features),
            predictions=tuple(predictions),
            induction_lineage_seed=int(induction_lineage_seed),
        )

    @property
    def model_checksum(self) -> str:
        return _checksum(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "format_version": self.format_version,
            "induction_lineage_seed": self.induction_lineage_seed,
            "predictions": [item.to_dict() for item in self.predictions],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.safe_payload, "model_checksum": self.model_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EmpiricalActionEffectModel:
        model = cls(
            features=tuple(map(str, payload.get("features", ()))),
            predictions=tuple(
                EffectPrediction.from_dict(dict(item))
                for item in payload.get("predictions", ())
            ),
            induction_lineage_seed=int(payload["induction_lineage_seed"]),
            format_version=str(payload.get("format_version", "")),
        )
        if payload.get("model_checksum") not in {None, model.model_checksum}:
            raise ValueError("progress-shadow effect-model checksum mismatch")
        return model

    def prediction(self, stage: int, action_name: str) -> EffectPrediction:
        wanted = (int(stage), str(action_name).upper())
        for item in self.predictions:
            if (item.stage, item.action_name) == wanted:
                return item
        raise KeyError(wanted)

    def predicted_step(self, stage: int, action_name: str) -> dict[str, Any]:
        prediction = self.prediction(stage, action_name)
        return step_from_projection(
            action_name=prediction.action_name,
            vector=prediction.projection,
            features=self.features,
            position=stage,
        )


def posterior_from_snapshot(payload: Mapping[str, Any]) -> JointCausalProgressPosterior:
    snapshot = dict(payload.get("posterior", payload))
    particles = []
    for raw in snapshot.get("particles", ()):
        item = dict(raw)
        probability = float(item["probability"])
        if probability <= 0.0:
            raise ValueError("progress posterior particle has non-positive mass")
        program = CausalProgressProgram.from_dict(dict(item["progress_program"]))
        owner_hash = str(item["owner_program_hash"])
        if program.owner_program_hash != owner_hash:
            raise ValueError("progress posterior owner mismatch")
        particles.append(
            JointProgressParticle(
                owner_program_hash=owner_hash,
                progress_program=program,
                log_weight=math.log(probability),
                evidence_ids=tuple(map(str, item.get("evidence_ids", ()))),
                lineage=("t12_5b:sealed_parent",),
            )
        )
    posterior = JointCausalProgressPosterior(particles)
    if len(posterior.particles) != int(snapshot.get("joint_particle_count", -1)):
        raise ValueError("progress posterior particle count mismatch")
    return posterior


def rank_scores(scores: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(
        name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    )


def reciprocal_rank(ranking: Sequence[str], expected_action: str) -> float:
    try:
        return 1.0 / (tuple(ranking).index(str(expected_action)) + 1)
    except ValueError:
        return 0.0


def build_shadow_ranking(
    *,
    posterior: JointCausalProgressPosterior,
    model: EmpiricalActionEffectModel,
    stage: int,
    candidate_actions: Sequence[str],
    expected_actions: Sequence[str],
) -> dict[str, Any]:
    stage = int(stage)
    if not 0 <= stage < len(expected_actions):
        raise ValueError("progress-shadow stage is outside the option")
    actions = tuple(str(item).upper() for item in candidate_actions)
    expected = str(expected_actions[stage]).upper()
    prefix = tuple(
        model.predicted_step(index, expected_actions[index]) for index in range(stage)
    )
    causal_scores = {
        action: posterior.expected_potential(
            (*prefix, model.predicted_step(stage, action))
        )
        for action in actions
    }
    change_scores = {
        action: float(model.prediction(stage, action).changed) for action in actions
    }
    magnitude_scores = {
        action: model.prediction(stage, action).magnitude for action in actions
    }
    lexicographic_scores = {
        action: -float(index) for index, action in enumerate(sorted(actions))
    }
    action_only_scores = {action: float(action == expected) for action in actions}
    rankings = {
        "action_only": rank_scores(action_only_scores),
        "causal_progress": rank_scores(causal_scores),
        "change_only": rank_scores(change_scores),
        "magnitude_only": rank_scores(magnitude_scores),
        "lexicographic": rank_scores(lexicographic_scores),
    }
    alternatives = [value for name, value in causal_scores.items() if name != expected]
    return {
        "action_only_scores": action_only_scores,
        "candidate_actions": list(actions),
        "causal_margin": causal_scores[expected] - max(alternatives),
        "causal_scores": causal_scores,
        "change_scores": change_scores,
        "effect_model_checksum": model.model_checksum,
        "expected_action": expected,
        "lexicographic_scores": lexicographic_scores,
        "magnitude_scores": magnitude_scores,
        "rankings": {name: list(values) for name, values in rankings.items()},
        "stage": stage,
    }


__all__ = [
    "EffectPrediction",
    "EmpiricalActionEffectModel",
    "build_shadow_ranking",
    "posterior_from_snapshot",
    "progress_signature",
    "projected_effect_step",
    "projection_vector",
    "rank_scores",
    "reciprocal_rank",
    "step_from_projection",
]
