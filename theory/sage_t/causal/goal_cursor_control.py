"""Paired equal-capacity goal-cursor control audit for SAGE.T12.5c."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


@dataclass(frozen=True)
class ControlArm:
    """One frozen two-slot intervention at the stage-3 goal cursor."""

    name: str
    program_actions: tuple[str, ...]
    goal_cursor_bound: bool

    def __post_init__(self) -> None:
        actions = tuple(str(item).upper() for item in self.program_actions)
        if str(self.name) not in {"goal_cursor", "binding_swap"}:
            raise ValueError("unknown T12.5c control arm")
        if len(actions) != 2 or any(not item.startswith("ACTION") for item in actions):
            raise ValueError("T12.5c arms require exactly two ACTION slots")
        object.__setattr__(self, "program_actions", actions)

    @property
    def program_id(self) -> str:
        return ">".join(self.program_actions)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "program_id": self.program_id}


@dataclass(frozen=True)
class ControlScheduleEntry:
    """One immutable position in the counterbalanced physical schedule."""

    order_index: int
    lineage_seed: int
    arm_name: str
    repetition: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_goal_cursor_control(
    *,
    trials: Sequence[Mapping[str, Any]],
    arms: Sequence[ControlArm],
    schedule: Sequence[ControlScheduleEntry],
) -> dict[str, Any]:
    """Audit the fixed paired schedule without score-derived labels."""

    program_to_name = {arm.program_id: arm.name for arm in arms}
    expected_order = tuple(
        (
            int(entry.lineage_seed),
            str(entry.arm_name),
            int(entry.repetition),
        )
        for entry in schedule
    )
    observed_order = tuple(
        (
            int(row.get("lineage_seed", -1)),
            program_to_name.get(str(row.get("program_id", "")), ""),
            int(row.get("repetition", -1)),
        )
        for row in trials
    )
    expected_lineages = sorted({int(entry.lineage_seed) for entry in schedule})
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        key = (
            int(row.get("lineage_seed", -1)),
            program_to_name.get(str(row.get("program_id", "")), ""),
        )
        grouped[key].append(row)

    context_exact_by_lineage: dict[str, bool] = {}
    for lineage in expected_lineages:
        records = [
            row for row in trials if int(row.get("lineage_seed", -1)) == lineage
        ]
        context_exact_by_lineage[str(lineage)] = bool(
            records
            and len({str(row.get("detour_context_hash", "")) for row in records}) == 1
            and len({_canonical(row.get("prefix_steps", ())) for row in records}) == 1
            and all(row.get("original_prefix_exact") is True for row in records)
            and all(row.get("prefix_exact") is True for row in records)
            and all(row.get("detour_available") is True for row in records)
            and all(row.get("detour_neutral") is True for row in records)
            and all(row.get("detour_terminal") is False for row in records)
        )

    cells: list[dict[str, Any]] = []
    availability_deterministic = True
    effects_deterministic = True
    outcomes_deterministic = True
    repetitions_exact = True
    actions_available_or_terminal = True
    for lineage in expected_lineages:
        for arm in arms:
            records = sorted(
                grouped.get((lineage, arm.name), ()),
                key=lambda item: int(item.get("repetition", -1)),
            )
            expected_repetitions = {
                int(item.repetition)
                for item in schedule
                if int(item.lineage_seed) == lineage and item.arm_name == arm.name
            }
            repetition_exact = bool(
                len(records) == len(expected_repetitions)
                and {int(row.get("repetition", -1)) for row in records}
                == expected_repetitions
            )
            program_exact = bool(
                records
                and all(
                    tuple(str(value).upper() for value in row.get("program_actions", ()))
                    == arm.program_actions
                    for row in records
                )
            )
            availability_vectors = {
                tuple(bool(step.get("available")) for step in row.get("candidate_steps", ()))
                for row in records
            }
            effect_vectors = {
                _canonical(row.get("candidate_steps", ())) for row in records
            }
            outcome_vectors = {
                (
                    int(row.get("executed_action_count", 0)),
                    int(row.get("level_delta", 0)),
                    bool(row.get("program_complete")),
                    bool(row.get("terminal")),
                    bool(row.get("terminal_failure")),
                    str(row.get("terminal_state", "")),
                )
                for row in records
            }
            availability_exact = bool(records and len(availability_vectors) == 1)
            effects_exact = bool(records and len(effect_vectors) == 1)
            outcomes_exact = bool(records and len(outcome_vectors) == 1)
            acquired_or_terminal = bool(
                records
                and all(
                    all(bool(step.get("available")) for step in row.get("candidate_steps", ()))
                    and (
                        int(row.get("executed_action_count", 0)) == len(arm.program_actions)
                        or bool(row.get("terminal"))
                        or int(row.get("level_delta", 0)) > 0
                    )
                    for row in records
                )
            )
            level_deltas = {int(row.get("level_delta", 0)) for row in records}
            safe_progress = bool(
                repetition_exact
                and program_exact
                and availability_exact
                and effects_exact
                and outcomes_exact
                and acquired_or_terminal
                and len(level_deltas) == 1
                and next(iter(level_deltas), 0) > 0
                and all(not bool(row.get("terminal_failure")) for row in records)
            )
            rejected = bool(
                repetition_exact
                and program_exact
                and availability_exact
                and outcomes_exact
                and acquired_or_terminal
                and records
                and all(int(row.get("level_delta", 0)) <= 0 for row in records)
            )
            availability_deterministic &= availability_exact
            effects_deterministic &= effects_exact
            outcomes_deterministic &= outcomes_exact
            repetitions_exact &= repetition_exact and program_exact
            actions_available_or_terminal &= acquired_or_terminal
            representative = records[0] if records else {}
            cells.append(
                {
                    **arm.to_dict(),
                    "actions_available_or_terminal": acquired_or_terminal,
                    "availability_deterministic": availability_exact,
                    "effect_deterministic": effects_exact,
                    "evidence_ids": [str(row.get("trial_id", "")) for row in records],
                    "level_delta": int(representative.get("level_delta", 0)),
                    "lineage_seed": lineage,
                    "outcome_deterministic": outcomes_exact,
                    "program_complete": bool(
                        records and all(row.get("program_complete") for row in records)
                    ),
                    "rejected": rejected,
                    "repetition_count": len(records),
                    "repetitions_exact": repetition_exact,
                    "safe_progress": safe_progress,
                    "terminal_failure": bool(
                        records and all(row.get("terminal_failure") for row in records)
                    ),
                }
            )

    by_cell = {
        (int(item["lineage_seed"]), str(item["name"])): item for item in cells
    }
    paired: list[dict[str, Any]] = []
    for lineage in expected_lineages:
        treatment = by_cell[(lineage, "goal_cursor")]
        control = by_cell[(lineage, "binding_swap")]
        level_delta_gain = int(treatment["level_delta"]) - int(control["level_delta"])
        paired.append(
            {
                "binding_swap_rejected": bool(control["rejected"]),
                "goal_cursor_safe_progress": bool(treatment["safe_progress"]),
                "level_delta_gain": level_delta_gain,
                "lineage_seed": lineage,
                "paired_advantage": bool(
                    treatment["safe_progress"]
                    and control["rejected"]
                    and level_delta_gain >= 1
                ),
            }
        )

    treatment_progress = bool(paired and all(item["goal_cursor_safe_progress"] for item in paired))
    control_rejected = bool(paired and all(item["binding_swap_rejected"] for item in paired))
    paired_advantage = bool(paired and all(item["paired_advantage"] for item in paired))
    metrics = {
        "actions_available_or_terminal": actions_available_or_terminal,
        "availability_is_deterministic": availability_deterministic,
        "binding_swap_control_rejected": control_rejected,
        "context_exact_by_lineage": context_exact_by_lineage,
        "context_replay_is_exact": bool(
            context_exact_by_lineage
            and all(context_exact_by_lineage.values())
        ),
        "effects_are_deterministic": effects_deterministic,
        "equal_capacity_horizon": bool(
            arms and len({len(arm.program_actions) for arm in arms}) == 1
        ),
        "expected_trial_count": len(schedule),
        "fixed_counterbalanced_schedule_completed": observed_order == expected_order,
        "goal_cursor_safe_progress": treatment_progress,
        "observed_trial_count": len(trials),
        "outcomes_are_deterministic": outcomes_deterministic,
        "paired_advantage_all_lineages": paired_advantage,
        "repetition_count_is_exact": repetitions_exact,
        "terminal_failure_heterogeneous_across_lineages": len(
            {
                bool(by_cell[(lineage, "binding_swap")]["terminal_failure"])
                for lineage in expected_lineages
            }
        )
        > 1,
    }
    return {
        "arm_registry": {
            "format_version": "sage-t12.5c-goal-cursor-control-arms-v1",
            "arms": cells,
            "paired_lineages": paired,
        },
        "metrics": metrics,
    }


__all__ = [
    "ControlArm",
    "ControlScheduleEntry",
    "audit_goal_cursor_control",
]
