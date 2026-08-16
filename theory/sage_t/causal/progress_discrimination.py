"""Offline affordance and hard-contrast audit for SAGE.T12.5b.2."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .progress import JointCausalProgressPosterior, ProgressMilestone
from .progress_shadow import progress_signature, projection_vector

AFFORDANCE_REGISTRY_FORMAT = "sage-t12.5b.2-affordance-registry-v1"
CONTRAST_REGISTRY_FORMAT = "sage-t12.5b.2-contrast-registry-v1"


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


@dataclass(frozen=True)
class LocalAffordance:
    lineage_seed: int
    stage: int
    action_name: str
    repetition_count: int
    availability_count: int
    availability_deterministic: bool
    prefix_exact: bool
    prefix_checksum: str
    effect_deterministic: bool
    projection: tuple[int, ...] | None
    magnitude: float | None
    progress_gain: float | None
    milestone_signature: tuple[bool, ...] | None
    evidence_ids: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.availability_count == self.repetition_count

    @property
    def semantic_key(self) -> str | None:
        if not self.executable or self.milestone_signature is None:
            return None
        return _checksum(
            {
                "milestone_signature": list(self.milestone_signature),
                "stage": self.stage,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "availability_count": self.availability_count,
            "availability_deterministic": self.availability_deterministic,
            "effect_deterministic": self.effect_deterministic,
            "evidence_ids": list(self.evidence_ids),
            "executable": self.executable,
            "lineage_seed": self.lineage_seed,
            "magnitude": self.magnitude,
            "milestone_signature": (
                None
                if self.milestone_signature is None
                else list(self.milestone_signature)
            ),
            "prefix_checksum": self.prefix_checksum,
            "prefix_exact": self.prefix_exact,
            "progress_gain": self.progress_gain,
            "projection": None if self.projection is None else list(self.projection),
            "repetition_count": self.repetition_count,
            "semantic_key": self.semantic_key,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class AffordanceBinding:
    stage: int
    milestone_signature: tuple[bool, ...]
    induction_lineage_seed: int
    induction_action_name: str
    confirmation_lineage_seed: int
    confirmation_action_name: str
    semantic_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_names_equal": (
                self.induction_action_name == self.confirmation_action_name
            ),
            "confirmation_action_name": self.confirmation_action_name,
            "confirmation_lineage_seed": self.confirmation_lineage_seed,
            "induction_action_name": self.induction_action_name,
            "induction_lineage_seed": self.induction_lineage_seed,
            "matching_fields": ["stage", "milestone_signature"],
            "milestone_signature": list(self.milestone_signature),
            "semantic_key": self.semantic_key,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class HardContrast:
    lineage_seed: int
    stage: int
    progress_action: str
    distractor_action: str
    progress_gain: float
    distractor_progress_gain: float
    progress_magnitude: float
    distractor_magnitude: float

    @property
    def magnitude_gap(self) -> float:
        return self.distractor_magnitude - self.progress_magnitude

    def to_dict(self) -> dict[str, Any]:
        return {
            "distractor_action": self.distractor_action,
            "distractor_magnitude": self.distractor_magnitude,
            "distractor_progress_gain": self.distractor_progress_gain,
            "lineage_seed": self.lineage_seed,
            "magnitude_gap": self.magnitude_gap,
            "progress_action": self.progress_action,
            "progress_gain": self.progress_gain,
            "progress_magnitude": self.progress_magnitude,
            "stage": self.stage,
        }


def _local_affordances(
    *,
    trials: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    posterior: JointCausalProgressPosterior,
    milestones: Sequence[ProgressMilestone],
) -> tuple[LocalAffordance, ...]:
    groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for raw in trials:
        item = dict(raw)
        groups[
            (
                int(item["lineage_seed"]),
                int(item["stage"]),
                str(item["action_name"]).upper(),
            )
        ].append(item)

    output = []
    for (lineage_seed, stage, action_name), records in sorted(groups.items()):
        records = sorted(records, key=lambda item: int(item["repetition"]))
        available = tuple(bool(item["branch_available"]) for item in records)
        prefix_exact = all(bool(item["prefix_exact"]) for item in records)
        prefix_checksums = {
            _checksum([dict(step) for step in item.get("prefix_steps", ())])
            for item in records
        }
        if len(prefix_checksums) != 1:
            raise ValueError(
                "T12.5b.2 repetitions do not share one exact causal prefix"
            )
        executable = bool(available) and all(available)
        vectors = {
            projection_vector(
                dict(item["candidate_step"]), features=features
            )
            for item in records
            if item["branch_available"]
        }
        effect_deterministic = len(vectors) <= 1
        projection = next(iter(vectors)) if executable and vectors else None
        representative = dict(records[0])
        prefix = tuple(dict(step) for step in representative.get("prefix_steps", ()))
        step = dict(representative["candidate_step"])
        base = posterior.expected_potential(prefix)
        gain = posterior.expected_potential((*prefix, step)) - base if executable else None
        output.append(
            LocalAffordance(
                lineage_seed=lineage_seed,
                stage=stage,
                action_name=action_name,
                repetition_count=len(records),
                availability_count=sum(available),
                availability_deterministic=len(set(available)) == 1,
                prefix_exact=prefix_exact,
                prefix_checksum=next(iter(prefix_checksums)),
                effect_deterministic=effect_deterministic,
                projection=projection,
                magnitude=(
                    None
                    if projection is None
                    else float(sum(abs(value) for value in projection))
                ),
                progress_gain=None if gain is None else float(gain),
                milestone_signature=(
                    None if not executable else progress_signature(step, milestones)
                ),
                evidence_ids=tuple(str(item["trial_id"]) for item in records),
            )
        )
    return tuple(output)


def _progress_bindings(
    *,
    affordances: Sequence[LocalAffordance],
    stages: Sequence[int],
    induction_lineage_seed: int,
    confirmation_lineage_seed: int,
    milestone_count: int,
) -> tuple[AffordanceBinding, ...]:
    bindings = []
    for stage in stages:
        wanted = tuple(index == stage for index in range(milestone_count))
        matches = {}
        for seed in (induction_lineage_seed, confirmation_lineage_seed):
            candidates = [
                item
                for item in affordances
                if item.lineage_seed == seed
                and item.stage == stage
                and item.executable
                and item.milestone_signature == wanted
            ]
            if len(candidates) == 1:
                matches[seed] = candidates[0]
        if len(matches) != 2:
            continue
        source = matches[induction_lineage_seed]
        target = matches[confirmation_lineage_seed]
        if source.semantic_key != target.semantic_key or source.semantic_key is None:
            raise ValueError("T12.5b.2 semantic affordance key failed to transport")
        bindings.append(
            AffordanceBinding(
                stage=stage,
                milestone_signature=wanted,
                induction_lineage_seed=induction_lineage_seed,
                induction_action_name=source.action_name,
                confirmation_lineage_seed=confirmation_lineage_seed,
                confirmation_action_name=target.action_name,
                semantic_key=source.semantic_key,
            )
        )
    return tuple(bindings)


def audit_progress_discrimination(
    *,
    trials: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    posterior: JointCausalProgressPosterior,
    milestones: Sequence[ProgressMilestone],
    lineage_seeds: Sequence[int],
    stages: Sequence[int],
    expected_actions: Sequence[str],
    repetitions_per_branch: int,
    induction_lineage_seed: int,
    confirmation_lineage_seed: int,
    minimum_distractor_magnitude_gap: float,
) -> dict[str, Any]:
    """Audit locally executable affordances without creating new evidence."""

    affordances = _local_affordances(
        trials=trials,
        features=features,
        posterior=posterior,
        milestones=milestones,
    )
    contexts = {
        (int(seed), int(stage)) for seed in lineage_seeds for stage in stages
    }
    observed_contexts = {
        (item.lineage_seed, item.stage) for item in affordances
    }
    if observed_contexts != contexts:
        raise ValueError("T12.5b.2 local affordance context matrix is incomplete")

    bindings = _progress_bindings(
        affordances=affordances,
        stages=stages,
        induction_lineage_seed=induction_lineage_seed,
        confirmation_lineage_seed=confirmation_lineage_seed,
        milestone_count=len(milestones),
    )
    rankings = []
    hard_contrasts = []
    for seed, stage in sorted(contexts):
        local = [
            item
            for item in affordances
            if item.lineage_seed == seed and item.stage == stage and item.executable
        ]
        if not local:
            continue
        causal = sorted(
            local,
            key=lambda item: (
                -float(item.progress_gain or 0.0),
                item.action_name,
            ),
        )
        magnitude = sorted(
            local,
            key=lambda item: (-float(item.magnitude or 0.0), item.action_name),
        )
        expected = str(expected_actions[stage]).upper()
        rankings.append(
            {
                "causal_ranking": [item.action_name for item in causal],
                "causal_top1_correct": causal[0].action_name == expected,
                "expected_action": expected,
                "lineage_seed": seed,
                "magnitude_ranking": [item.action_name for item in magnitude],
                "magnitude_top1_correct": magnitude[0].action_name == expected,
                "rankings_disagree": causal[0].action_name != magnitude[0].action_name,
                "stage": stage,
            }
        )
        wanted_signature = tuple(
            index == stage for index in range(len(milestones))
        )
        progress_candidates = [
            item for item in local if item.milestone_signature == wanted_signature
        ]
        for progress in progress_candidates:
            for distractor in local:
                if distractor.action_name == progress.action_name:
                    continue
                if distractor.milestone_signature == wanted_signature:
                    continue
                if float(distractor.magnitude or 0.0) < float(
                    progress.magnitude or 0.0
                ) + float(minimum_distractor_magnitude_gap):
                    continue
                hard_contrasts.append(
                    HardContrast(
                        lineage_seed=seed,
                        stage=stage,
                        progress_action=progress.action_name,
                        distractor_action=distractor.action_name,
                        progress_gain=float(progress.progress_gain or 0.0),
                        distractor_progress_gain=float(
                            distractor.progress_gain or 0.0
                        ),
                        progress_magnitude=float(progress.magnitude or 0.0),
                        distractor_magnitude=float(distractor.magnitude or 0.0),
                    )
                )

    hard_lineages = sorted({item.lineage_seed for item in hard_contrasts})
    executable_counts = {
        f"{seed}:{stage}": sum(
            item.executable
            for item in affordances
            if item.lineage_seed == seed and item.stage == stage
        )
        for seed, stage in sorted(contexts)
    }
    unavailable = [item for item in affordances if not item.executable]
    causal_accuracy = sum(item["causal_top1_correct"] for item in rankings) / max(
        1, len(rankings)
    )
    magnitude_accuracy = sum(
        item["magnitude_top1_correct"] for item in rankings
    ) / max(1, len(rankings))
    return {
        "affordance_registry": {
            "affordances": [item.to_dict() for item in affordances],
            "bindings": [item.to_dict() for item in bindings],
            "format_version": AFFORDANCE_REGISTRY_FORMAT,
        },
        "contrast_registry": {
            "format_version": CONTRAST_REGISTRY_FORMAT,
            "hard_contrasts": [item.to_dict() for item in hard_contrasts],
            "rankings": rankings,
        },
        "metrics": {
            "affordance_binding_coverage": len(bindings) / max(1, len(stages)),
            "affordance_binding_count": len(bindings),
            "availability_is_deterministic": all(
                item.availability_deterministic for item in affordances
            ),
            "causal_top1_accuracy": causal_accuracy,
            "effect_is_deterministic_when_executable": all(
                item.effect_deterministic for item in affordances if item.executable
            ),
            "exact_prefix_rate": sum(item.prefix_exact for item in affordances)
            / max(1, len(affordances)),
            "executable_action_counts": executable_counts,
            "hard_contrast_count": len(hard_contrasts),
            "hard_contrast_lineage_count": len(hard_lineages),
            "hard_contrast_lineages": hard_lineages,
            "local_affordance_count": len(affordances),
            "magnitude_top1_accuracy": magnitude_accuracy,
            "minimum_executable_actions_per_context": min(executable_counts.values()),
            "progress_action_executable_in_every_context": all(
                any(
                    item.lineage_seed == seed
                    and item.stage == stage
                    and item.action_name == str(expected_actions[stage]).upper()
                    and item.executable
                    for item in affordances
                )
                for seed, stage in contexts
            ),
            "ranking_disagreement_count": sum(
                item["rankings_disagree"] for item in rankings
            ),
            "ranking_context_count": len(rankings),
            "repetition_count_is_exact": all(
                item.repetition_count == repetitions_per_branch
                for item in affordances
            ),
            "unavailable_affordance_count": len(unavailable),
            "unavailable_affordances": [
                {
                    "action_name": item.action_name,
                    "lineage_seed": item.lineage_seed,
                    "stage": item.stage,
                }
                for item in unavailable
            ],
        },
    }


__all__ = [
    "AFFORDANCE_REGISTRY_FORMAT",
    "CONTRAST_REGISTRY_FORMAT",
    "AffordanceBinding",
    "HardContrast",
    "LocalAffordance",
    "audit_progress_discrimination",
]
