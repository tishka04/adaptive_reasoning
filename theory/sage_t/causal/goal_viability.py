"""Goal-continuation viability audit for SAGE.T12.5b.5.

The experiment labels a first action by the observed outcome of a frozen,
re-grounded continuation toward a confirmed level-progress witness.  Labels
never depend on the causal score or on immediate effect magnitude.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .experiment import _signed


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
class ViabilityBranch:
    """One preregistered first-action intervention and goal continuation."""

    branch_id: str
    first_action: str
    program_actions: tuple[str, ...]
    goal_cursor_advance: bool
    transport_eligible: bool

    def __post_init__(self) -> None:
        first = str(self.first_action).upper()
        actions = tuple(str(item).upper() for item in self.program_actions)
        if not first.startswith("ACTION"):
            raise ValueError("viability branches require an ACTION first action")
        if not actions or actions[0] != first:
            raise ValueError("branch program must begin with its first action")
        object.__setattr__(self, "first_action", first)
        object.__setattr__(self, "program_actions", actions)
        expected_id = ">".join(actions)
        if str(self.branch_id) != expected_id:
            raise ValueError("branch id must be the ordered action program")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def viability_branch(
    first_action: str,
    *,
    goal_continuation: Sequence[str],
    transport_actions: Sequence[str],
) -> ViabilityBranch:
    """Build the frozen cursor-aware branch program.

    If the intervention matches the next witness step it consumes that step;
    otherwise the complete witness continuation remains to be re-grounded.
    """

    first = str(first_action).upper()
    continuation = tuple(str(item).upper() for item in goal_continuation)
    if not continuation:
        raise ValueError("goal continuation cannot be empty")
    advances = first == continuation[0]
    actions = continuation if advances else (first, *continuation)
    return ViabilityBranch(
        branch_id=">".join(actions),
        first_action=first,
        program_actions=actions,
        goal_cursor_advance=advances,
        transport_eligible=first in {str(item).upper() for item in transport_actions},
    )


def _branch_summaries(
    *,
    trials: Sequence[Mapping[str, Any]],
    expected_branches: Sequence[ViabilityBranch],
    repetitions_per_branch: int,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    expected = {branch.branch_id: branch for branch in expected_branches}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[str(row.get("program_id", ""))].append(row)

    expected_schedule = {
        (branch_id, repetition)
        for branch_id in expected
        for repetition in range(int(repetitions_per_branch))
    }
    observed_schedule = {
        (str(row.get("program_id", "")), int(row.get("repetition", -1)))
        for row in trials
    }
    prefix_payloads = {_canonical(row.get("prefix_steps", ())) for row in trials}
    context_hashes = {str(row.get("detour_context_hash", "")) for row in trials}
    context_exact = bool(
        trials
        and len(prefix_payloads) == 1
        and len(context_hashes) == 1
        and all(row.get("original_prefix_exact") is True for row in trials)
        and all(row.get("prefix_exact") is True for row in trials)
        and all(row.get("detour_available") is True for row in trials)
        and all(row.get("detour_neutral") is True for row in trials)
        and all(row.get("detour_terminal") is False for row in trials)
    )

    summaries: list[dict[str, Any]] = []
    repetitions_exact = True
    availability_deterministic = True
    effects_deterministic = True
    outcomes_deterministic = True
    transport_first_actions_available = True
    for branch_id, branch in sorted(expected.items()):
        records = sorted(
            grouped.get(branch_id, ()), key=lambda item: int(item.get("repetition", -1))
        )
        repetition_exact = bool(
            len(records) == int(repetitions_per_branch)
            and {int(item.get("repetition", -1)) for item in records}
            == set(range(int(repetitions_per_branch)))
        )
        action_exact = bool(
            records
            and all(
                tuple(str(value).upper() for value in item.get("program_actions", ()))
                == branch.program_actions
                for item in records
            )
        )
        availability_vectors = {
            tuple(bool(step.get("available")) for step in item.get("candidate_steps", ()))
            for item in records
        }
        effect_vectors = {
            _canonical(item.get("candidate_steps", ())) for item in records
        }
        outcome_vectors = {
            (
                int(item.get("executed_action_count", 0)),
                int(item.get("level_delta", 0)),
                bool(item.get("program_complete")),
                bool(item.get("terminal")),
                bool(item.get("terminal_failure")),
                str(item.get("terminal_state", "")),
            )
            for item in records
        }
        availability_is_deterministic = bool(records and len(availability_vectors) == 1)
        effect_is_deterministic = bool(records and len(effect_vectors) == 1)
        outcome_is_deterministic = bool(records and len(outcome_vectors) == 1)
        first_action_available = bool(
            records
            and all(
                item.get("candidate_steps")
                and item["candidate_steps"][0].get("available") is True
                for item in records
            )
        )
        progressed = bool(
            records
            and all(int(item.get("level_delta", 0)) > 0 for item in records)
        )
        terminal_failure = bool(
            records and all(bool(item.get("terminal_failure")) for item in records)
        )
        safe_progress = bool(
            repetition_exact
            and action_exact
            and availability_is_deterministic
            and effect_is_deterministic
            and outcome_is_deterministic
            and first_action_available
            and progressed
            and all(not bool(item.get("terminal_failure")) for item in records)
        )
        rejected = bool(
            repetition_exact
            and action_exact
            and availability_is_deterministic
            and outcome_is_deterministic
            and first_action_available
            and all(int(item.get("level_delta", 0)) <= 0 for item in records)
        )
        completed = bool(records and all(item.get("program_complete") for item in records))
        missing = bool(
            records
            and any(
                any(not bool(step.get("available")) for step in item.get("candidate_steps", ()))
                for item in records
            )
        )
        repetitions_exact = repetitions_exact and repetition_exact and action_exact
        availability_deterministic = (
            availability_deterministic and availability_is_deterministic
        )
        effects_deterministic = effects_deterministic and effect_is_deterministic
        outcomes_deterministic = outcomes_deterministic and outcome_is_deterministic
        if branch.transport_eligible:
            transport_first_actions_available = (
                transport_first_actions_available and first_action_available
            )
        representative = records[0] if records else {}
        summaries.append(
            {
                **branch.to_dict(),
                "availability_deterministic": availability_is_deterministic,
                "effect_deterministic": effect_is_deterministic,
                "evidence_ids": [str(item.get("trial_id", "")) for item in records],
                "first_action_available": first_action_available,
                "level_delta": int(representative.get("level_delta", 0)),
                "missing": missing,
                "outcome_deterministic": outcome_is_deterministic,
                "program_complete": completed,
                "rejected": rejected,
                "repetition_count": len(records),
                "repetitions_exact": repetition_exact,
                "safe_progress": safe_progress,
                "terminal_failure": terminal_failure,
            }
        )

    integrity = {
        "availability_is_deterministic": availability_deterministic,
        "context_replay_is_exact": context_exact,
        "effects_are_deterministic": effects_deterministic,
        "fixed_branch_schedule_completed": observed_schedule == expected_schedule,
        "outcomes_are_deterministic": outcomes_deterministic,
        "repetition_count_is_exact": repetitions_exact,
        "transport_first_actions_available": transport_first_actions_available,
    }
    return summaries, integrity


def audit_calibration_trials(
    *,
    trials: Sequence[Mapping[str, Any]],
    expected_branches: Sequence[ViabilityBranch],
    repetitions_per_branch: int,
) -> dict[str, Any]:
    summaries, integrity = _branch_summaries(
        trials=trials,
        expected_branches=expected_branches,
        repetitions_per_branch=repetitions_per_branch,
    )
    progress = sorted(
        (
            item
            for item in summaries
            if item["transport_eligible"]
            and item["goal_cursor_advance"]
            and item["safe_progress"]
        ),
        key=lambda item: (len(item["program_actions"]), tuple(item["program_actions"])),
    )
    controls = sorted(
        (
            item
            for item in summaries
            if item["transport_eligible"]
            and not item["goal_cursor_advance"]
            and item["rejected"]
        ),
        key=lambda item: (len(item["program_actions"]), tuple(item["program_actions"])),
    )
    selected_progress = progress[0] if progress else None
    selected_control = controls[0] if controls else None
    selection = {
        "control": selected_control,
        "control_rule": "shortest transportable cursor-mismatch branch with deterministic no progress",
        "progress": selected_progress,
        "progress_rule": "shortest transportable cursor-advance branch with deterministic safe progress",
        "score_used_for_branch_selection": False,
    }
    metrics = {
        **integrity,
        "branch_count": len(summaries),
        "cursor_advance_safe_progress_count": len(progress),
        "cursor_mismatch_rejected_count": len(controls),
        "missing_branch_count": sum(bool(item["missing"]) for item in summaries),
        "safe_progress_branch_count": sum(bool(item["safe_progress"]) for item in summaries),
        "terminal_failure_branch_count": sum(
            bool(item["terminal_failure"]) for item in summaries
        ),
        "viability_contrast_count": int(
            selected_progress is not None and selected_control is not None
        ),
    }
    return {
        "branch_registry": {
            "format_version": "sage-t12.5b.5-goal-viability-branch-registry-v1",
            "branches": summaries,
            "selection": selection,
        },
        "metrics": metrics,
        "selection": selection,
    }


def evaluation_registry_payload(
    *,
    manifest_checksum: str,
    protocol_checksum: str,
    calibration_evidence_checksum: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    progress = selection.get("progress")
    control = selection.get("control")
    if not isinstance(progress, Mapping) or not isinstance(control, Mapping):
        raise ValueError("a passed viability contrast is required for evaluation")
    return _signed(
        {
            "format_version": "sage-t12.5b.5-goal-viability-evaluation-registry-v1",
            "manifest_checksum": str(manifest_checksum),
            "protocol_checksum": str(protocol_checksum),
            "calibration_evidence_checksum": str(calibration_evidence_checksum),
            "binding": {
                "fields": ["target_stage", "goal_cursor_relation"],
                "uses_action_name_as_cross_game_semantics": False,
            },
            "branches": {
                "control": dict(control),
                "progress": dict(progress),
            },
        },
        "registry_checksum",
    )


def audit_evaluation_trials(
    *,
    trials: Sequence[Mapping[str, Any]],
    evaluation_registry: Mapping[str, Any],
    repetitions_per_branch: int,
) -> dict[str, Any]:
    registered = dict(evaluation_registry["branches"])
    branches = tuple(
        ViabilityBranch(
            branch_id=str(registered[name]["branch_id"]),
            first_action=str(registered[name]["first_action"]),
            program_actions=tuple(registered[name]["program_actions"]),
            goal_cursor_advance=bool(registered[name]["goal_cursor_advance"]),
            transport_eligible=bool(registered[name]["transport_eligible"]),
        )
        for name in ("progress", "control")
    )
    summaries, integrity = _branch_summaries(
        trials=trials,
        expected_branches=branches,
        repetitions_per_branch=repetitions_per_branch,
    )
    by_id = {item["branch_id"]: item for item in summaries}
    progress = by_id[str(registered["progress"]["branch_id"])]
    control = by_id[str(registered["control"]["branch_id"])]
    progress_transferred = bool(progress["safe_progress"])
    control_rejected = bool(control["rejected"] and not control["safe_progress"])
    metrics = {
        **integrity,
        "control_branch_rejected": control_rejected,
        "evaluation_branch_count": len(summaries),
        "goal_viability_contrast_transferred": progress_transferred and control_rejected,
        "progress_branch_transferred": progress_transferred,
    }
    return {
        "branch_registry": {
            "format_version": "sage-t12.5b.5-goal-viability-evaluation-branches-v1",
            "branches": summaries,
        },
        "metrics": metrics,
        "registered_control": control,
        "registered_progress": progress,
    }


__all__ = [
    "ViabilityBranch",
    "audit_calibration_trials",
    "audit_evaluation_trials",
    "evaluation_registry_payload",
    "viability_branch",
]
