"""Compact paired procedural proof for SAGE.10b through SAGE.10e."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .online_frontier_exploration import OnlineFrontierExplorer
from .online_level_route_memory import OnlineLevelRouteMemory


SCHEMA_VERSION = "sage.sage10b_plus_procedural.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage10b_plus_procedural_benchmark.json"
)


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: dict = field(default_factory=dict)


def run_sage10b_plus_benchmark(
    *,
    write_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> Dict[str, Any]:
    relay_active = _relay_case(enabled=True)
    relay_ablated = _relay_case(enabled=False)
    stall_active = _stall_case(enabled=True)
    stall_ablated = _stall_case(enabled=False)
    rearm_active = _rearm_case(enabled=True)
    rearm_ablated = _rearm_case(enabled=False)
    route_active = _route_case(enable_shortening=True)
    route_ablated = _route_case(enable_shortening=False)
    gates = {
        "sage10b_relay_necessary_for_credit": bool(
            relay_active["terminal_credits"] == 1
            and relay_active["relays_created"] == 1
            and relay_ablated["terminal_credits"] == 0
            and relay_ablated["relays_created"] == 0
        ),
        "sage10c_generalized_stall_necessary_for_intervention": bool(
            stall_active["interventions"] == 1
            and stall_active["new_states"] == 1
            and stall_ablated["interventions"] == 0
        ),
        "sage10d_level_rearm_necessary_after_terminal_retreat": bool(
            rearm_active["rearmed_interventions"] == 1
            and rearm_ablated["rearmed_interventions"] == 0
        ),
        "sage10e_shortening_candidate_requires_terminal_verification": bool(
            route_active["shortening_candidates"] > 0
            and route_active["shortening_confirmations"] == 1
            and route_active["actions_saved"] > 0
            and route_ablated["shortening_candidates"] == 0
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "paired_sage10b_through_sage10e_procedural_proof",
        "protocol": {
            "paired_active_and_isolated_ablation": True,
            "candidate_only_until_terminal_confirmation": True,
            "reset_and_danger_destroy_eligibility": True,
            "game_specific_rules_used": False,
            "coordinates_or_palette_semantics_used": False,
            "a32_or_a33_written": False,
        },
        "relay": {
            "active": relay_active,
            "ablated": relay_ablated,
        },
        "generalized_stall": {
            "active": stall_active,
            "ablated": stall_ablated,
        },
        "per_level_rearm": {
            "active": rearm_active,
            "ablated": rearm_ablated,
        },
        "route_shortening": {
            "active": route_active,
            "ablated": route_ablated,
        },
        "gates": gates,
        "all_procedural_gates_passed": all(gates.values()),
    }
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _grid() -> np.ndarray:
    grid = np.zeros((7, 9), dtype=np.int32)
    grid[2:4, 2:4] = 2
    return grid


def _diagnostics() -> Dict[str, int]:
    return {
        "branch_actions": 8,
        "actions_since_terminal_improvement": 8,
        "window_actions": 8,
        "max_hash_repeat": 1,
        "unique_states_in_window": 8,
    }


def _relay_case(*, enabled: bool) -> Dict[str, int]:
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        delayed_terminal_credit_window=1,
        enable_subeffect_eligibility_relay=enabled,
    )
    before = _grid()
    action = _Action("ACTION6", {"x": 2, "y": 2})
    selected = explorer.select(
        current_grid=before,
        available_actions=("ACTION6",),
        available_action_candidates=(action,),
        branch_diagnostics=_diagnostics(),
    )
    if selected is None:
        return {"relays_created": 0, "terminal_credits": 0}
    after = before.copy()
    after[2, 2] = 8
    explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
        causal_effect_signature="effect-a",
    )
    explorer.note_transition(
        terminal_success=False,
        grid_before=before,
        grid_after=after,
        action_data=selected.action_data,
        causal_effect_signature="effect-a",
    )
    middle = after.copy()
    middle[3, 2] = 8
    explorer.note_transition(
        terminal_success=False,
        grid_before=after,
        grid_after=middle,
        action_data={"x": 2, "y": 2},
        causal_effect_signature="effect-b",
        causally_linked_effect_signatures=("effect-a",),
    )
    explorer.note_transition(terminal_success=True)
    summary = explorer.summary()
    return {
        "relays_created": int(summary["subeffect_relays_created"]),
        "terminal_credits": int(summary["delayed_terminal_credits"]),
    }


def _stall_case(*, enabled: bool) -> Dict[str, int]:
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=3,
        minimum_failed_branches=0,
        enable_generalized_stall_detection=enabled,
    )
    before = _grid()
    selected = explorer.select(
        current_grid=before,
        available_actions=("ACTION1",),
        available_action_candidates=(_Action("ACTION1"),),
        branch_diagnostics=_diagnostics(),
    )
    if selected is None:
        return {"interventions": 0, "new_states": 0}
    after = before.copy()
    after[0, 0] = 7
    outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    return {
        "interventions": 1,
        "new_states": int(bool(outcome["novel_state"])),
    }


def _rearm_case(*, enabled: bool) -> Dict[str, int]:
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        minimum_failed_branches=0,
        enable_per_level_rearming=enabled,
    )
    grid = _grid()
    first = explorer.select(
        current_grid=grid,
        available_actions=("ACTION1",),
        available_action_candidates=(_Action("ACTION1"),),
        branch_diagnostics=_diagnostics(),
    )
    if first is None:
        return {"rearmed_interventions": 0}
    explorer.observe_transition(
        grid_before=grid,
        grid_after=grid,
        action_name=first.action_name,
        action_data=first.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=True,
    )
    explorer.note_transition(terminal_success=True)
    explorer.start_branch()
    explorer.note_level_change()
    rearmed = explorer.select(
        current_grid=grid,
        available_actions=("ACTION2",),
        available_action_candidates=(_Action("ACTION2"),),
        branch_diagnostics=_diagnostics(),
    )
    return {"rearmed_interventions": int(rearmed is not None)}


def _route_case(*, enable_shortening: bool) -> Dict[str, int]:
    memory = OnlineLevelRouteMemory(
        enable_shortening=enable_shortening,
    )
    for index, action in enumerate(("ACTION1", "ACTION2", "ACTION3")):
        memory.observe_transition(
            state_signature_before=f"s{index}",
            state_signature_after=(
                "level-1" if index == 2 else f"s{index + 1}"
            ),
            action_name=action,
            action_data=None,
            level_progressed=index == 2,
            won=False,
            game_over=False,
        )
    if enable_shortening:
        memory.start_branch()
        selection = memory.select(
            state_signature="s0",
            available_actions=("ACTION1", "ACTION2", "ACTION3"),
        )
        if selection is not None:
            memory.observe_transition(
                state_signature_before="s0",
                state_signature_after="level-1",
                action_name=selection.action.action_name,
                action_data=selection.action.data,
                level_progressed=True,
                won=False,
                game_over=False,
            )
    summary = memory.summary()
    return {
        "shortening_candidates": int(
            summary["shortening_candidates"]
        ),
        "shortening_confirmations": int(
            summary["shortening_confirmations"]
        ),
        "actions_saved": int(summary["shortening_actions_saved"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.10b+ paired procedural proof.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_sage10b_plus_benchmark(write_path=args.out)
    print(json.dumps(payload["gates"], indent=2, sort_keys=True))
    return 0 if payload["all_procedural_gates_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "SCHEMA_VERSION",
    "run_sage10b_plus_benchmark",
]
