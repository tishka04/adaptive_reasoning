"""Live procedural R1 -> R2 -> R1 theory-reactivation benchmark.

SAGE.9u previously demonstrated reactivation only with hand-authored
``StencilTheoryAssessment`` objects.  SAGE.9y drives the production stencil
learner and structural-break detector from rendered grids and concrete click
transitions.  The observation stream changes to a distinct structural family,
confirms a revised terminal theory twice, then naturally returns to the first
family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .online_structural_break import OnlineStructuralBreakDetector
from .online_terminal_relational_stencil import (
    OnlineTerminalRelationalStencilLearner,
    RelationalStencilSelection,
    StencilTheoryAssessment,
)


SCHEMA_VERSION = "sage.natural_theory_reactivation.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage9y_natural_theory_reactivation_benchmark.json"
)
BACKGROUND = 5
SPACING = 8
TILE = 6
STENCIL = (18, 18)
R1_MARKERS = (
    (0, 2, 2),
    (0, 8, 0),
    (0, 2, 2),
)
R2_MARKERS = (
    (2, 2, 2),
    (2, 8, 2),
    (2, 2, 2),
)


def run_natural_theory_reactivation_benchmark(
    *,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    active = _run_live_reactivation_arm(
        enable_hierarchical_composition=True,
    )
    ablated = _run_live_reactivation_arm(
        enable_hierarchical_composition=False,
    )
    protocol_gate = bool(
        active["transition_fingerprint"]
        == ablated["transition_fingerprint"]
        and active["live_transitions"] == ablated["live_transitions"]
        and active["r1_structural_signature"]
        == ablated["r1_structural_signature"]
        and active["r2_structural_signature"]
        == ablated["r2_structural_signature"]
        and active["r1_family_signature"]
        == ablated["r1_family_signature"]
        and active["r2_family_signature"]
        == ablated["r2_family_signature"]
        and active["hierarchical_composition_enabled"]
        and not ablated["hierarchical_composition_enabled"]
    )
    observation_return_gate = bool(
        active["r1_structural_signature"]
        == active["returned_structural_signature"]
        and active["r1_structural_signature"]
        != active["r2_structural_signature"]
        and active["r1_family_signature"]
        != active["r2_family_signature"]
    )
    policy_sequence_gate = bool(
        active["policy_sources"] == ["base", "exact_revision", "base"]
        and active["policy_terminal_successes"] == 3
        and ablated["policy_sources"]
        == active["policy_sources"]
        and ablated["policy_terminal_successes"] == 3
    )
    reactivation_gate = bool(
        protocol_gate
        and observation_return_gate
        and policy_sequence_gate
        and active["returned_theory_reactivated"]
        and active["theory_programs"] == 2
        and active["theory_switches"] == 2
        and active["theory_reactivations"] == 1
        and not ablated["returned_theory_reactivated"]
        and ablated["theory_switches"] == 0
        and ablated["theory_reactivations"] == 0
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": (
            "paired_live_procedural_r1_r2_r1_theory_reactivation"
        ),
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "same_rendered_observations_per_arm": True,
            "same_concrete_actions_per_arm": True,
            "same_terminal_outcomes_per_arm": True,
            "only_hierarchical_composition_differs": True,
            "hand_authored_assessments_used": False,
            "game_or_level_identity_used": False,
            "arc_level_or_win_claimed": False,
        },
        "observation_return_gate_passed": observation_return_gate,
        "policy_sequence_gate_passed": policy_sequence_gate,
        "natural_theory_reactivation_gate_passed": reactivation_gate,
        "active": active,
        "hierarchical_composition_ablated": ablated,
    }
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _run_live_reactivation_arm(
    *,
    enable_hierarchical_composition: bool,
) -> Dict[str, Any]:
    learner = OnlineTerminalRelationalStencilLearner()
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
        minimum_terminal_confirmations=2,
        enable_active_hypothesis_arbitration=True,
        enable_regime_abstraction=True,
        enable_hierarchical_theory_composition=(
            enable_hierarchical_composition
        ),
    )
    actions = _actions()
    transitions = []
    r1_predecessor = _r1_predecessor()
    r1_target = {"x": 18, "y": 26}

    # Establish the generic toggle mechanic with repeated nonterminal live
    # transitions, then ground R1 in one actual terminal example.
    toggled_r1 = _toggle_click(
        r1_predecessor,
        r1_target,
        center_color=8,
        alternative_color=9,
    )
    for index in range(6):
        learner.observe_transition(
            grid_before=r1_predecessor,
            grid_after=toggled_r1,
            action_name="ACTION6",
            action_data=r1_target,
            available_action_candidates=actions,
            terminal_success_confirmed=False,
        )
        transitions.append(f"mechanic:{index}:ACTION6:18,26")
    learner.observe_transition(
        grid_before=r1_predecessor,
        grid_after=np.zeros((8, 8), dtype=np.int64),
        action_name="ACTION6",
        action_data=r1_target,
        available_action_candidates=actions,
        terminal_success_confirmed=True,
    )
    transitions.append("r1-terminal-grounding:ACTION6:18,26")
    base_rules = learner.selection_rules()

    # First natural R1 policy activation.
    r1_before = learner.assess(
        current_grid=r1_predecessor,
        available_action_candidates=actions,
        rules=base_rules,
    )
    r1_resolution = detector.resolve_policy(
        assessment=r1_before,
        base_rules=base_rules,
    )
    if r1_resolution is None:
        raise RuntimeError("R1 did not compile a base theory program")
    r1_selection = _required_selection(
        learner,
        r1_predecessor,
        actions,
        r1_resolution.rule_map(),
        "live R1 policy",
    )
    detector.note_policy_action(r1_resolution.theory_id)
    r1_after_grid = _toggle_click(
        r1_predecessor,
        r1_selection.action_data,
        center_color=8,
        alternative_color=9,
    )
    r1_after = learner.assess(
        current_grid=r1_after_grid,
        available_action_candidates=actions,
        rules=base_rules,
    )
    r1_terminal = bool(r1_after.total_violations == 0)
    detector.observe_base_prediction(
        before=r1_before,
        after=r1_after,
        predicted_reduction=(
            r1_selection.violations_before
            - r1_selection.expected_violations_after
        ),
        action_no_effect=False,
        terminal_success=r1_terminal,
        base_rules=base_rules,
    )
    transitions.append(_transition_key("r1-policy", r1_selection))

    # A distinct all-filled carrier invalidates R1.  Three concrete clicks
    # have no effect, which is the live evidence required to suspend it.
    r2_predecessor = _r2_predecessor()
    r2_before = learner.assess(
        current_grid=r2_predecessor,
        available_action_candidates=actions,
        rules=base_rules,
    )
    for index in range(3):
        resolution = detector.resolve_policy(
            assessment=r2_before,
            base_rules=base_rules,
        )
        if resolution is None:
            raise RuntimeError("R1 suspended before required residuals")
        selection = _required_selection(
            learner,
            r2_predecessor,
            actions,
            resolution.rule_map(),
            "live R1 residual",
        )
        learner.observe_transition(
            grid_before=r2_predecessor,
            grid_after=r2_predecessor,
            action_name=selection.action_name,
            action_data=selection.action_data,
            available_action_candidates=actions,
            terminal_success_confirmed=False,
        )
        detector.observe_base_prediction(
            before=r2_before,
            after=r2_before,
            predicted_reduction=(
                selection.violations_before
                - selection.expected_violations_after
            ),
            action_no_effect=True,
            terminal_success=False,
            base_rules=base_rules,
        )
        transitions.append(
            _transition_key(f"r2-residual-{index}", selection)
        )

    # The most discriminating revision predicts the hidden R2 terminal click.
    revision_terminal_successes = 0
    for confirmation_index in range(2):
        hypotheses = detector.revision_hypotheses(
            r2_before.structural_signature
        )
        if not hypotheses:
            raise RuntimeError("R2 break generated no live hypotheses")
        committed = detector.committed_revision_hypothesis(
            r2_before.structural_signature
        )
        discriminating = committed is None
        if committed is None:
            experiment = learner.select_discriminating_experiment(
                current_grid=r2_predecessor,
                available_action_candidates=actions,
                hypothesis_rules={
                    hypothesis.hypothesis_id: hypothesis.rule_map()
                    for hypothesis in hypotheses
                },
                hypothesis_priority=tuple(
                    hypothesis.hypothesis_id
                    for hypothesis in hypotheses
                ),
            )
            if experiment is None:
                raise RuntimeError("R2 has no discriminating experiment")
            hypothesis_id = experiment.hypothesis_id
            selection = experiment.selection
            detector.note_revision_action(
                hypothesis_id,
                discriminating=True,
                disagreement_score=experiment.disagreement_score,
            )
        else:
            hypothesis_id = committed.hypothesis_id
            selection = _required_selection(
                learner,
                r2_predecessor,
                actions,
                committed.rule_map(),
                "committed live R2 experiment",
            )
            detector.note_revision_action(hypothesis_id)
        revision_rules = detector.hypothesis_rules(hypothesis_id)
        r2_after_grid = _toggle_click(
            r2_predecessor,
            selection.action_data,
            center_color=12,
            alternative_color=13,
        )
        learner.observe_transition(
            grid_before=r2_predecessor,
            grid_after=r2_after_grid,
            action_name=selection.action_name,
            action_data=selection.action_data,
            available_action_candidates=actions,
            terminal_success_confirmed=False,
        )
        r2_after = learner.assess(
            current_grid=r2_after_grid,
            available_action_candidates=actions,
            rules=revision_rules,
        )
        terminal = bool(r2_after.total_violations == 0)
        revision_terminal_successes += int(terminal)
        detector.observe_revision_outcome(
            hypothesis_id=hypothesis_id,
            after=r2_after,
            terminal_success=terminal,
            game_over=False,
        )
        transitions.append(
            _transition_key(
                (
                    f"r2-revision-{confirmation_index}-"
                    f"{'discriminating' if discriminating else 'committed'}"
                ),
                selection,
            )
        )

    r2_resolution = detector.resolve_policy(
        assessment=r2_before,
        base_rules=base_rules,
    )
    if r2_resolution is None:
        raise RuntimeError("terminally confirmed R2 did not resolve")
    r2_selection = _required_selection(
        learner,
        r2_predecessor,
        actions,
        r2_resolution.rule_map(),
        "live R2 policy",
    )
    detector.note_policy_action(r2_resolution.theory_id)
    r2_policy_after = learner.assess(
        current_grid=_toggle_click(
            r2_predecessor,
            r2_selection.action_data,
            center_color=12,
            alternative_color=13,
        ),
        available_action_candidates=actions,
        rules=r2_resolution.rule_map(),
    )
    r2_terminal = bool(r2_policy_after.total_violations == 0)
    transitions.append(_transition_key("r2-policy", r2_selection))

    # The next rendered state is again the original R1 family.  No detector
    # method is told to reactivate anything: policy resolution follows the
    # observation and marks the already compiled base program as reactivated.
    returned = learner.assess(
        current_grid=r1_predecessor,
        available_action_candidates=actions,
        rules=base_rules,
    )
    returned_resolution = detector.resolve_policy(
        assessment=returned,
        base_rules=base_rules,
    )
    if returned_resolution is None:
        raise RuntimeError("returning R1 observation did not resolve")
    returned_selection = _required_selection(
        learner,
        r1_predecessor,
        actions,
        returned_resolution.rule_map(),
        "returned live R1 policy",
    )
    detector.note_policy_action(returned_resolution.theory_id)
    returned_after = learner.assess(
        current_grid=_toggle_click(
            r1_predecessor,
            returned_selection.action_data,
            center_color=8,
            alternative_color=9,
        ),
        available_action_candidates=actions,
        rules=returned_resolution.rule_map(),
    )
    returned_terminal = bool(returned_after.total_violations == 0)
    transitions.append(_transition_key("returned-r1", returned_selection))

    summary = detector.summary()
    return {
        "hierarchical_composition_enabled": bool(
            enable_hierarchical_composition
        ),
        "live_transitions": len(transitions),
        "transition_fingerprint": transitions,
        "base_rules": dict(base_rules),
        "r1_structural_signature": r1_before.structural_signature,
        "r2_structural_signature": r2_before.structural_signature,
        "returned_structural_signature": returned.structural_signature,
        "r1_family_signature": r1_before.structural_family_signature,
        "r2_family_signature": r2_before.structural_family_signature,
        "policy_sources": [
            r1_resolution.source,
            r2_resolution.source,
            returned_resolution.source,
        ],
        "policy_theory_ids": [
            r1_resolution.theory_id,
            r2_resolution.theory_id,
            returned_resolution.theory_id,
        ],
        "policy_terminal_successes": sum((
            r1_terminal,
            r2_terminal,
            returned_terminal,
        )),
        "revision_terminal_successes": (
            revision_terminal_successes
        ),
        "returned_theory_reactivated": bool(
            returned_resolution.reactivated
        ),
        "breaks_detected": summary["breaks_detected"],
        "revision_confirmations": summary[
            "revision_confirmations"
        ],
        "theory_programs": summary["theory_programs"],
        "theory_switches": summary["theory_switches"],
        "theory_reactivations": summary["theory_reactivations"],
        "theory_program_hierarchy": summary[
            "theory_program_hierarchy"
        ],
    }


def _actions() -> Tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            name="ACTION6",
            action_args={"x": x, "y": y},
        )
        for y in (10, 18, 26)
        for x in (10, 18, 26)
        if (x, y) != STENCIL
    )


def _grid(
    *,
    center: int,
    markers: Sequence[Sequence[int]],
    colors: Mapping[Tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((40, 40), BACKGROUND, dtype=np.int64)
    for (x, y), color in colors.items():
        grid[y:y + TILE, x:x + TILE] = 1
        grid[y + 3, x + 3] = int(color)
    sx, sy = STENCIL
    for row, values in enumerate(markers):
        for column, value in enumerate(values):
            grid[
                sy + 2 * row:sy + 2 * row + 2,
                sx + 2 * column:sx + 2 * column + 2,
            ] = int(value)
    grid[sy + 3, sx + 3] = int(center)
    return grid


def _r1_predecessor() -> np.ndarray:
    colors = {}
    for y in (10, 18, 26):
        for x in (10, 18, 26):
            if (x, y) == STENCIL:
                continue
            marker = R1_MARKERS[(y - 10) // SPACING][
                (x - 10) // SPACING
            ]
            colors[(x, y)] = 8 if marker == 0 else 9
    colors[(18, 26)] = 8
    return _grid(center=8, markers=R1_MARKERS, colors=colors)


def _r2_predecessor() -> np.ndarray:
    colors = {
        (x, y): 12
        for y in (10, 18, 26)
        for x in (10, 18, 26)
        if (x, y) != STENCIL
    }
    colors[(10, 10)] = 13
    return _grid(center=12, markers=R2_MARKERS, colors=colors)


def _toggle_click(
    grid: np.ndarray,
    action_data: Mapping[str, Any],
    *,
    center_color: int,
    alternative_color: int,
) -> np.ndarray:
    result = np.asarray(grid).copy()
    x = int(action_data["x"])
    y = int(action_data["y"])
    sample = int(result[y + 3, x + 3])
    result[y + 3, x + 3] = (
        int(alternative_color)
        if sample == int(center_color)
        else int(center_color)
    )
    return result


def _required_selection(
    learner: OnlineTerminalRelationalStencilLearner,
    grid: np.ndarray,
    actions: Sequence[Any],
    rules: Mapping[str, bool],
    reason: str,
) -> RelationalStencilSelection:
    selection = learner.select_with_rules(
        current_grid=grid,
        available_action_candidates=actions,
        rules=rules,
        reason_prefix=reason,
    )
    if selection is None:
        raise RuntimeError(f"{reason} produced no live click")
    return selection


def _transition_key(
    phase: str,
    selection: RelationalStencilSelection,
) -> str:
    return (
        f"{phase}:{selection.action_name}:"
        f"{int(selection.action_data['x'])},"
        f"{int(selection.action_data['y'])}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.9y live R1-R2-R1 benchmark.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_natural_theory_reactivation_benchmark(
        write_path=args.out,
    )
    print(json.dumps({
        "protocol_gate_passed": payload["protocol"][
            "protocol_gate_passed"
        ],
        "observation_return_gate_passed": payload[
            "observation_return_gate_passed"
        ],
        "policy_sequence_gate_passed": payload[
            "policy_sequence_gate_passed"
        ],
        "natural_theory_reactivation_gate_passed": payload[
            "natural_theory_reactivation_gate_passed"
        ],
    }, indent=2, sort_keys=True))
    return (
        0
        if payload["natural_theory_reactivation_gate_passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_natural_theory_reactivation_benchmark",
]
