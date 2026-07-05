"""M3 next controlled-experiment selector."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

from theory.m1.controlled_followup_experiment import ledger_spec_from_entry

from .scientific_planner_state import ScientificPlanningState


DEFAULT_PREFERRED_CONTROLS = ("ACTION3", "ACTION4", "ACTION1", "ACTION2")
CONTROL_REUSE_REASON = "no_unused_distinct_control_available"


@dataclass(frozen=True)
class PlannedControlledExperiment:
    """A planned target-vs-control experiment."""

    hypothesis_key: str
    game_id: str
    mechanic_family: str
    target_action: str
    control_action: str
    predicted_metric: str
    priority: float
    reason: str
    status: str = "PLANNED"
    control_reuse_reason: str = ""
    skipped_controls: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    open_questions: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "hypothesis_key": self.hypothesis_key,
            "game_id": self.game_id,
            "mechanic_family": self.mechanic_family,
            "target_action": self.target_action,
            "control_action": self.control_action,
            "predicted_metric": self.predicted_metric,
            "priority": round(float(self.priority), 4),
            "reason": self.reason,
            "status": self.status,
            "skipped_controls": [dict(item) for item in self.skipped_controls],
            "open_questions": list(self.open_questions),
        }
        if self.control_reuse_reason:
            payload["control_reuse_reason"] = self.control_reuse_reason
        return payload


def select_next_experiment(
    state: ScientificPlanningState,
    *,
    live_available_actions: Sequence[str],
    preferred_controls: Sequence[str] = DEFAULT_PREFERRED_CONTROLS,
) -> PlannedControlledExperiment:
    """Select the next controlled experiment from unresolved ledger entries."""
    candidates: list[tuple[float, str, Dict[str, Any], Dict[str, str], Tuple[str, ...]]] = []
    for entry in state.ledger_entries:
        if str(entry.get("status", "")).lower() != "unresolved":
            continue
        if not bool(entry.get("controlled_test_required", True)):
            continue
        spec = ledger_spec_from_entry(entry)
        controls = available_controls_for_target(
            live_available_actions,
            target_action=spec["target_action"],
            preferred_controls=preferred_controls,
        )
        if not controls:
            continue
        priority = score_entry_for_planning(state, entry, available_controls=controls)
        candidates.append((priority, spec["hypothesis_key"], dict(entry), spec, controls))

    if not candidates:
        raise ValueError("no unresolved hypothesis with an available control action")

    priority, _, entry, spec, controls = sorted(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )[0]
    control_action, reuse_reason = choose_control_action(
        state,
        hypothesis_key=spec["hypothesis_key"],
        available_controls=controls,
        preferred_controls=preferred_controls,
    )
    skipped = skipped_preferred_controls(
        live_available_actions,
        target_action=spec["target_action"],
        hypothesis_key=spec["hypothesis_key"],
        preferred_controls=preferred_controls,
    )
    questions = open_questions_for_selection(
        state,
        hypothesis_key=spec["hypothesis_key"],
        available_controls=controls,
    )
    return PlannedControlledExperiment(
        hypothesis_key=spec["hypothesis_key"],
        game_id=spec["game_id"],
        mechanic_family=spec["mechanic_family"],
        target_action=spec["target_action"],
        control_action=control_action,
        predicted_metric=spec["predicted_metric"],
        priority=priority,
        reason=planning_reason(
            state,
            hypothesis_key=spec["hypothesis_key"],
            available_controls=controls,
        ),
        control_reuse_reason=reuse_reason,
        skipped_controls=tuple(skipped),
        open_questions=tuple(questions),
    )


def score_entry_for_planning(
    state: ScientificPlanningState,
    entry: Dict[str, Any],
    *,
    available_controls: Sequence[str],
) -> float:
    spec = ledger_spec_from_entry(entry)
    key = spec["hypothesis_key"]
    already_tested_count = int(state.controlled_experiments_by_key.get(key, 0))
    is_untested = 1.0 if already_tested_count == 0 else 0.0
    has_support_but_unresolved = (
        1.0
        if state.support_events_by_key.get(key, 0) > 0
        and str(entry.get("status", "")).lower() == "unresolved"
        else 0.0
    )
    has_competing_controls = 1.0 if len(set(available_controls)) >= 2 else 0.0
    return (
        2.0 * is_untested
        + 1.0 * has_support_but_unresolved
        + 1.0 * has_competing_controls
        - 0.5 * float(already_tested_count)
    )


def available_controls_for_target(
    live_available_actions: Sequence[str],
    *,
    target_action: str,
    preferred_controls: Sequence[str] = DEFAULT_PREFERRED_CONTROLS,
) -> Tuple[str, ...]:
    live = {
        str(action)
        for action in live_available_actions
        if str(action) and str(action) not in {"RESET", str(target_action)}
    }
    preferred = [action for action in preferred_controls if action in live]
    others = sorted(action for action in live if action not in set(preferred_controls))
    return tuple(preferred + others)


def choose_control_action(
    state: ScientificPlanningState,
    *,
    hypothesis_key: str,
    available_controls: Sequence[str],
    preferred_controls: Sequence[str] = DEFAULT_PREFERRED_CONTROLS,
) -> tuple[str, str]:
    controls = tuple(available_controls)
    used = list(state.controls_used_by_key.get(hypothesis_key, ()))
    unused = [control for control in controls if control not in set(used)]
    if unused:
        return unused[0], ""

    counts = Counter(control for control in used if control in set(controls))
    order = {control: index for index, control in enumerate(controls)}
    preferred_order = {
        control: index for index, control in enumerate(preferred_controls)
    }
    repeated = sorted(
        controls,
        key=lambda control: (
            counts.get(control, 0),
            order.get(control, 999),
            preferred_order.get(control, 999),
            control,
        ),
    )[0]
    return repeated, CONTROL_REUSE_REASON


def skipped_preferred_controls(
    live_available_actions: Sequence[str],
    *,
    target_action: str,
    hypothesis_key: str,
    preferred_controls: Sequence[str] = DEFAULT_PREFERRED_CONTROLS,
) -> Tuple[Dict[str, Any], ...]:
    live = {str(action) for action in live_available_actions}
    skipped: list[Dict[str, Any]] = []
    for action in preferred_controls:
        if action == target_action:
            continue
        if action not in live:
            skipped.append(
                {
                    "hypothesis_key": hypothesis_key,
                    "action": action,
                    "reason": "action_not_available_at_reset",
                }
            )
    return tuple(skipped)


def open_questions_for_selection(
    state: ScientificPlanningState,
    *,
    hypothesis_key: str,
    available_controls: Sequence[str],
) -> Tuple[str, ...]:
    questions: list[str] = []
    controls = set(available_controls)
    used = set(state.controls_used_by_key.get(hypothesis_key, ()))
    if len(controls) < 2:
        questions.append("insufficient_distinct_controls")
    if state.contradiction_events_by_key.get(hypothesis_key, 0) > 0:
        questions.append("contradictory_control_observed")
    if (
        controls
        and controls.issubset(used)
        and state.support_events_by_key.get(hypothesis_key, 0) > 0
        and state.contradiction_events_by_key.get(hypothesis_key, 0) == 0
    ):
        questions.append("all_controls_support_hypothesis")
    return tuple(questions)


def planning_reason(
    state: ScientificPlanningState,
    *,
    hypothesis_key: str,
    available_controls: Sequence[str],
) -> str:
    return (
        "m3_priority_v1:"
        f"tested={state.controlled_experiments_by_key.get(hypothesis_key, 0)};"
        f"support_events={state.support_events_by_key.get(hypothesis_key, 0)};"
        f"available_controls={','.join(available_controls)}"
    )
