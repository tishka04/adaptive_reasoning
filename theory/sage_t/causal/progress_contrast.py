"""Prospective affordance and hard-contrast analysis for SAGE.T12.5b.3."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .progress import JointCausalProgressPosterior, ProgressMilestone
from .progress_shadow import progress_signature, projection_vector

PROSPECTIVE_AFFORDANCE_FORMAT = "sage-t12.5b.3-affordance-registry-v1"
PROSPECTIVE_CONTRAST_FORMAT = "sage-t12.5b.3-hard-contrast-registry-v1"


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
class ProspectiveAffordance:
    lineage_seed: int
    stage: int
    context_id: str
    detour_depth: int
    action_name: str
    repetition_count: int
    availability_count: int
    availability_deterministic: bool
    context_valid: bool
    prefix_checksum: str
    effect_deterministic: bool
    projection: tuple[int, ...] | None
    magnitude: float | None
    progress_gain: float | None
    milestone_signature: tuple[bool, ...] | None
    evidence_ids: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return bool(
            self.context_valid
            and self.availability_count == self.repetition_count
        )

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
            "context_id": self.context_id,
            "context_valid": self.context_valid,
            "detour_depth": self.detour_depth,
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
            "progress_gain": self.progress_gain,
            "projection": None if self.projection is None else list(self.projection),
            "repetition_count": self.repetition_count,
            "semantic_key": self.semantic_key,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class ProspectiveHardContrast:
    lineage_seed: int
    stage: int
    context_id: str
    detour_depth: int
    progress_action: str
    distractor_action: str
    progress_gain: float
    distractor_progress_gain: float
    progress_magnitude: float
    distractor_magnitude: float

    @property
    def magnitude_gap(self) -> float:
        return self.distractor_magnitude - self.progress_magnitude

    @property
    def causal_correct(self) -> bool:
        return self.progress_gain > self.distractor_progress_gain

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal_correct": self.causal_correct,
            "context_id": self.context_id,
            "detour_depth": self.detour_depth,
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


def audit_prospective_progress_contrasts(
    *,
    trials: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    posterior: JointCausalProgressPosterior,
    milestones: Sequence[ProgressMilestone],
    lineage_seeds: Sequence[int],
    target_stage: int,
    context_ids: Sequence[str],
    candidate_actions: Sequence[str],
    repetitions_per_branch: int,
    minimum_distractor_magnitude_gap: float,
) -> dict[str, Any]:
    """Evaluate only observed deterministic branches from fixed detour contexts."""

    rows = tuple(dict(item) for item in trials)
    expected_contexts = {
        (int(seed), str(context_id))
        for seed in lineage_seeds
        for context_id in context_ids
    }
    observed_contexts = {
        (int(item["lineage_seed"]), str(item["context_id"])) for item in rows
    }
    if observed_contexts != expected_contexts:
        raise ValueError("T12.5b.3 prospective context matrix is incomplete")

    context_records: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in rows:
        if int(item["stage"]) != int(target_stage):
            raise ValueError("T12.5b.3 trial escaped the frozen target stage")
        context_records[(int(item["lineage_seed"]), str(item["context_id"]))].append(
            item
        )

    context_summaries: dict[tuple[int, str], dict[str, Any]] = {}
    expected_rows_per_context = len(candidate_actions) * int(repetitions_per_branch)
    for key, records in sorted(context_records.items()):
        context_hashes = {str(item["detour_context_hash"]) for item in records}
        prefix_checksums = {
            _checksum([dict(step) for step in item.get("prefix_steps", ())])
            for item in records
        }
        detour_depths = {int(item["detour_depth"]) for item in records}
        summary = {
            "candidate_schedule_complete": (
                len(records) == expected_rows_per_context
                and {
                    str(item["action_name"]).upper() for item in records
                }
                == {str(action).upper() for action in candidate_actions}
            ),
            "context_id": key[1],
            "detour_available": all(bool(item["detour_available"]) for item in records),
            "detour_context_deterministic": len(context_hashes) == 1,
            "detour_depth": next(iter(detour_depths)) if len(detour_depths) == 1 else -1,
            "detour_neutral": all(bool(item["detour_neutral"]) for item in records),
            "detour_terminal": any(bool(item["detour_terminal"]) for item in records),
            "lineage_seed": key[0],
            "original_prefix_exact": all(
                bool(item["original_prefix_exact"]) for item in records
            ),
            "prefix_exact": all(bool(item["prefix_exact"]) for item in records),
            "prefix_steps_deterministic": len(prefix_checksums) == 1,
            "trial_count": len(records),
        }
        summary["valid"] = all(
            (
                summary["candidate_schedule_complete"],
                summary["detour_available"],
                summary["detour_context_deterministic"],
                summary["detour_depth"] > 0,
                summary["detour_neutral"],
                not summary["detour_terminal"],
                summary["original_prefix_exact"],
                summary["prefix_exact"],
                summary["prefix_steps_deterministic"],
            )
        )
        context_summaries[key] = summary

    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in rows:
        groups[
            (
                int(item["lineage_seed"]),
                str(item["context_id"]),
                str(item["action_name"]).upper(),
            )
        ].append(item)

    affordances: list[ProspectiveAffordance] = []
    for (seed, context_id, action_name), records in sorted(groups.items()):
        records = sorted(records, key=lambda item: int(item["repetition"]))
        available = tuple(bool(item["branch_available"]) for item in records)
        vectors = {
            projection_vector(dict(item["candidate_step"]), features=features)
            for item in records
            if item["branch_available"]
        }
        effect_deterministic = len(vectors) <= 1
        context_valid = bool(context_summaries[(seed, context_id)]["valid"])
        executable = bool(context_valid and available and all(available))
        projection = next(iter(vectors)) if executable and vectors else None
        representative = records[0]
        prefix = tuple(dict(step) for step in representative.get("prefix_steps", ()))
        step = dict(representative["candidate_step"])
        base = posterior.expected_potential(prefix)
        gain = posterior.expected_potential((*prefix, step)) - base if executable else None
        affordances.append(
            ProspectiveAffordance(
                lineage_seed=seed,
                stage=int(target_stage),
                context_id=context_id,
                detour_depth=int(representative["detour_depth"]),
                action_name=action_name,
                repetition_count=len(records),
                availability_count=sum(available),
                availability_deterministic=len(set(available)) == 1,
                context_valid=context_valid,
                prefix_checksum=_checksum([dict(step) for step in prefix]),
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

    wanted = tuple(index == int(target_stage) for index in range(len(milestones)))
    bindings = []
    rankings = []
    hard_contrasts: list[ProspectiveHardContrast] = []
    for context_id in context_ids:
        progress_by_seed: dict[int, ProspectiveAffordance] = {}
        for seed in lineage_seeds:
            local = [
                item
                for item in affordances
                if item.lineage_seed == int(seed)
                and item.context_id == str(context_id)
                and item.executable
            ]
            if local:
                causal = sorted(
                    local,
                    key=lambda item: (-float(item.progress_gain or 0.0), item.action_name),
                )
                magnitude = sorted(
                    local,
                    key=lambda item: (-float(item.magnitude or 0.0), item.action_name),
                )
                rankings.append(
                    {
                        "causal_ranking": [item.action_name for item in causal],
                        "causal_top1_is_progress": (
                            causal[0].milestone_signature == wanted
                        ),
                        "context_id": str(context_id),
                        "lineage_seed": int(seed),
                        "magnitude_ranking": [item.action_name for item in magnitude],
                        "magnitude_top1_is_progress": (
                            magnitude[0].milestone_signature == wanted
                        ),
                        "rankings_disagree": (
                            causal[0].action_name != magnitude[0].action_name
                        ),
                        "stage": int(target_stage),
                    }
                )
            progress = [item for item in local if item.milestone_signature == wanted]
            if len(progress) == 1:
                progress_by_seed[int(seed)] = progress[0]
                for distractor in local:
                    if distractor.action_name == progress[0].action_name:
                        continue
                    if distractor.milestone_signature == wanted:
                        continue
                    if float(distractor.magnitude or 0.0) < float(
                        progress[0].magnitude or 0.0
                    ) + float(minimum_distractor_magnitude_gap):
                        continue
                    hard_contrasts.append(
                        ProspectiveHardContrast(
                            lineage_seed=int(seed),
                            stage=int(target_stage),
                            context_id=str(context_id),
                            detour_depth=progress[0].detour_depth,
                            progress_action=progress[0].action_name,
                            distractor_action=distractor.action_name,
                            progress_gain=float(progress[0].progress_gain or 0.0),
                            distractor_progress_gain=float(
                                distractor.progress_gain or 0.0
                            ),
                            progress_magnitude=float(progress[0].magnitude or 0.0),
                            distractor_magnitude=float(distractor.magnitude or 0.0),
                        )
                    )
        if len(progress_by_seed) == len(tuple(lineage_seeds)):
            semantic_keys = {item.semantic_key for item in progress_by_seed.values()}
            if len(semantic_keys) != 1 or None in semantic_keys:
                raise ValueError("T12.5b.3 semantic progress binding failed")
            bindings.append(
                {
                    "action_names": {
                        str(seed): progress_by_seed[int(seed)].action_name
                        for seed in lineage_seeds
                    },
                    "context_id": str(context_id),
                    "matching_fields": ["stage", "milestone_signature"],
                    "milestone_signature": list(wanted),
                    "semantic_key": next(iter(semantic_keys)),
                    "stage": int(target_stage),
                }
            )

    hard_by_lineage = {
        int(seed): sum(item.lineage_seed == int(seed) for item in hard_contrasts)
        for seed in lineage_seeds
    }
    hard_contexts_by_lineage = {
        int(seed): {
            item.context_id
            for item in hard_contrasts
            if item.lineage_seed == int(seed)
        }
        for seed in lineage_seeds
    }
    common_hard_contexts = sorted(
        set.intersection(
            *(set(values) for values in hard_contexts_by_lineage.values())
        )
        if hard_contexts_by_lineage
        else set()
    )
    executable_counts = {
        f"{seed}:{context_id}": sum(
            item.executable
            for item in affordances
            if item.lineage_seed == int(seed) and item.context_id == str(context_id)
        )
        for seed, context_id in sorted(expected_contexts)
    }
    unavailable = [item for item in affordances if not item.executable]
    valid_contexts_by_lineage = {
        int(seed): {
            context_id
            for (lineage_seed, context_id), summary in context_summaries.items()
            if lineage_seed == int(seed) and bool(summary["valid"])
        }
        for seed in lineage_seeds
    }
    common_valid_contexts = sorted(
        set.intersection(
            *(set(values) for values in valid_contexts_by_lineage.values())
        )
        if valid_contexts_by_lineage
        else set()
    )
    valid_executable_counts = [
        count
        for key, count in executable_counts.items()
        if bool(
            context_summaries[
                (int(key.split(":", 1)[0]), key.split(":", 1)[1])
            ]["valid"]
        )
    ]
    terminal_failures = sum(bool(item.get("terminal_failure")) for item in rows)
    causal_hard_accuracy = sum(item.causal_correct for item in hard_contrasts) / max(
        1, len(hard_contrasts)
    )
    magnitude_hard_accuracy = sum(
        item.progress_magnitude > item.distractor_magnitude
        for item in hard_contrasts
    ) / max(1, len(hard_contrasts))
    return {
        "affordance_registry": {
            "affordances": [item.to_dict() for item in affordances],
            "bindings": bindings,
            "contexts": [
                context_summaries[key] for key in sorted(context_summaries)
            ],
            "format_version": PROSPECTIVE_AFFORDANCE_FORMAT,
        },
        "contrast_registry": {
            "format_version": PROSPECTIVE_CONTRAST_FORMAT,
            "hard_contrasts": [item.to_dict() for item in hard_contrasts],
            "rankings": rankings,
        },
        "metrics": {
            "affordance_binding_count": len(bindings),
            "availability_is_deterministic": all(
                item.availability_deterministic for item in affordances
            ),
            "causal_hard_contrast_accuracy": causal_hard_accuracy,
            "causal_top1_accuracy": sum(
                item["causal_top1_is_progress"] for item in rankings
            )
            / max(1, len(rankings)),
            "common_hard_contrast_context_count": len(common_hard_contexts),
            "common_hard_contrast_contexts": common_hard_contexts,
            "common_valid_context_count": len(common_valid_contexts),
            "common_valid_contexts": common_valid_contexts,
            "context_count": len(context_summaries),
            "detour_availability_rate": sum(
                bool(item["detour_available"]) for item in context_summaries.values()
            )
            / max(1, len(context_summaries)),
            "detour_context_determinism_rate": sum(
                bool(item["detour_context_deterministic"])
                for item in context_summaries.values()
            )
            / max(1, len(context_summaries)),
            "detour_neutrality_rate": sum(
                bool(item["detour_neutral"]) for item in context_summaries.values()
            )
            / max(1, len(context_summaries)),
            "effect_is_deterministic_when_executable": all(
                item.effect_deterministic for item in affordances if item.executable
            ),
            "executable_action_counts": executable_counts,
            "hard_contrast_accuracy_gain": (
                causal_hard_accuracy - magnitude_hard_accuracy
            ),
            "hard_contrast_count": len(hard_contrasts),
            "hard_contrasts_per_lineage": hard_by_lineage,
            "magnitude_hard_contrast_accuracy": magnitude_hard_accuracy,
            "magnitude_top1_accuracy": sum(
                item["magnitude_top1_is_progress"] for item in rankings
            )
            / max(1, len(rankings)),
            "minimum_executable_actions_per_context": min(
                executable_counts.values(), default=0
            ),
            "minimum_executable_actions_per_valid_context": min(
                valid_executable_counts, default=0
            ),
            "original_prefix_exact_rate": sum(
                bool(item["original_prefix_exact"])
                for item in context_summaries.values()
            )
            / max(1, len(context_summaries)),
            "progress_affordance_lineage_count": len(
                {
                    int(seed)
                    for seed in lineage_seeds
                    if any(
                        item.lineage_seed == int(seed)
                        and item.milestone_signature == wanted
                        and item.executable
                        for item in affordances
                    )
                }
            ),
            "ranking_context_count": len(rankings),
            "ranking_disagreement_count": sum(
                item["rankings_disagree"] for item in rankings
            ),
            "repetition_count_is_exact": all(
                item.repetition_count == int(repetitions_per_branch)
                for item in affordances
            ),
            "terminal_failures": terminal_failures,
            "trial_count": len(rows),
            "unavailable_affordance_count": len(unavailable),
            "unavailable_affordances": [
                {
                    "action_name": item.action_name,
                    "context_id": item.context_id,
                    "lineage_seed": item.lineage_seed,
                }
                for item in unavailable
            ],
            "valid_context_count": sum(
                bool(item["valid"]) for item in context_summaries.values()
            ),
            "valid_contexts_per_lineage": {
                str(seed): len(values)
                for seed, values in valid_contexts_by_lineage.items()
            },
        },
    }


__all__ = [
    "PROSPECTIVE_AFFORDANCE_FORMAT",
    "PROSPECTIVE_CONTRAST_FORMAT",
    "ProspectiveAffordance",
    "ProspectiveHardContrast",
    "audit_prospective_progress_contrasts",
]
