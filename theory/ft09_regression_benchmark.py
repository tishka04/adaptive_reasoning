"""SAGE.10e authority-ordering regression gates on the historical ft09 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from .live_transition_loop import build_observation
from .online_multiform_relational_learner import (
    OnlineMultiformRelationalLearner,
)
from .sage10b_plus_benchmark import run_sage10b_plus_benchmark
from .unified_cognition_ab_benchmark import _env_dir, _run_arm
from .unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


SCHEMA_VERSION = "sage.sage10e_authority_repair_ft09.v1"
DEFAULT_GAME_ID = "ft09-0d8bbf25"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage10e_authority_repair_ft09_regression.json"
)

ArmRunner = Callable[
    [str, Callable[[str], UnifiedCognitiveController]],
    Mapping[str, Any],
]

_LAYER_FLAGS: tuple[tuple[str, Dict[str, bool]], ...] = (
    (
        "post_9u_baseline",
        {
            "enable_frontier_oriented_exploration": False,
            "enable_delayed_frontier_terminal_credit": False,
            "enable_subeffect_eligibility_relay": False,
            "enable_generalized_frontier_stall_detection": False,
            "enable_per_level_frontier_rearming": False,
            "enable_terminal_multiform_relational_induction": False,
            "enable_level_route_memory": False,
            "enable_level_route_shortening": False,
        },
    ),
    (
        "plus_9v",
        {
            "enable_frontier_oriented_exploration": True,
            "enable_delayed_frontier_terminal_credit": False,
            "enable_subeffect_eligibility_relay": False,
            "enable_generalized_frontier_stall_detection": False,
            "enable_per_level_frontier_rearming": False,
            "enable_terminal_multiform_relational_induction": False,
            "enable_level_route_memory": False,
            "enable_level_route_shortening": False,
        },
    ),
    (
        "plus_9w",
        {
            "enable_frontier_oriented_exploration": True,
            "enable_delayed_frontier_terminal_credit": False,
            "enable_subeffect_eligibility_relay": False,
            "enable_generalized_frontier_stall_detection": False,
            "enable_per_level_frontier_rearming": False,
            "enable_terminal_multiform_relational_induction": True,
            "enable_level_route_memory": False,
            "enable_level_route_shortening": False,
        },
    ),
    (
        "plus_10a",
        {
            "enable_frontier_oriented_exploration": True,
            "enable_delayed_frontier_terminal_credit": True,
            "enable_subeffect_eligibility_relay": False,
            "enable_generalized_frontier_stall_detection": False,
            "enable_per_level_frontier_rearming": False,
            "enable_terminal_multiform_relational_induction": True,
            "enable_level_route_memory": False,
            "enable_level_route_shortening": False,
        },
    ),
    ("full", {}),
)


def run_ft09_authority_repair_benchmark(
    *,
    game_id: str = DEFAULT_GAME_ID,
    seed: int = 0,
    action_budget_per_reset: int = 160,
    resets: int = 14,
    arm_runner: ArmRunner | None = None,
    write_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Run the competence, monotonicity, liveness, and pre-emption gates."""
    if arm_runner is None:
        def arm_runner(
            layer_name: str,
            controller_factory: Callable[
                [str],
                UnifiedCognitiveController,
            ],
        ) -> Mapping[str, Any]:
            del layer_name
            return _run_arm(
                arm="unified",
                game_id=str(game_id),
                seed=int(seed),
                action_budget_per_reset=int(action_budget_per_reset),
                resets=int(resets),
                env_dir=_env_dir(),
                env_factory=None,
                controller_factory=controller_factory,
            )

    layers: Dict[str, Dict[str, Any]] = {}
    for layer_name, flags in _LAYER_FLAGS:
        def controller_factory(
            controller_game_id: str,
            *,
            _flags: Mapping[str, bool] = flags,
        ) -> UnifiedCognitiveController:
            return UnifiedCognitiveController(
                controller_game_id,
                config=UnifiedCognitiveConfig(**dict(_flags)),
            )

        layers[layer_name] = _compact_arm(
            arm_runner(layer_name, controller_factory)
        )

    baseline = layers["post_9u_baseline"]
    full = layers["full"]
    layer_monotonicity = {
        layer_name: bool(
            row["max_level_reached"]
            >= baseline["max_level_reached"]
            and row["wins"] >= baseline["wins"]
        )
        for layer_name, row in layers.items()
    }
    procedural = run_sage10b_plus_benchmark(write_path=None)
    multiform_liveness = _multiform_liveness_probe()
    mechanisms_live = bool(
        procedural["all_procedural_gates_passed"]
        and procedural["relay"]["active"]["relays_created"] > 0
        and procedural["generalized_stall"]["active"][
            "interventions"
        ] > 0
        and procedural["per_level_rearm"]["active"][
            "rearmed_interventions"
        ] > 0
        and procedural["route_shortening"]["active"][
            "shortening_candidates"
        ] > 0
        and multiform_liveness["selections"] > 0
        and multiform_liveness["demotions"] > 0
        and multiform_liveness["reactivations"] > 0
    )
    gates = {
        "g1_ft09_level6_and_win": bool(
            full["max_level_reached"] >= 6 and full["wins"] >= 1
        ),
        "g2_layer_monotonicity": all(layer_monotonicity.values()),
        "g3_intended_mechanisms_remain_live": mechanisms_live,
        "g4_zero_protected_route_preemptions": all(
            row["protected_route_preemptions"] == 0
            for row in layers.values()
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "sage10e_strict_authority_ordering_repair",
        "protocol": {
            "game_id": str(game_id),
            "seed": int(seed),
            "action_budget_per_reset": int(action_budget_per_reset),
            "resets": int(resets),
            "protected_tier": [
                "progressive_terminal_route",
                "frontier_reacquisition",
                "terminal_relational_stencil",
                "exact_terminal_replay",
                "terminal_observed_level_route",
            ],
            "adaptive_arbitration": "deferred_to_sage11_shadow_mode",
        },
        "layers": layers,
        "layer_monotonicity": layer_monotonicity,
        "procedural_liveness": {
            "sage10b_plus": procedural,
            "multiform": multiform_liveness,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    if write_path is not None:
        _write_json(payload, write_path)
    return payload


def _compact_arm(arm: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "max_level_reached": int(
            arm.get("max_level_reached", 0) or 0
        ),
        "levels_completed": int(
            arm.get("levels_completed_delta", 0) or 0
        ),
        "wins": int(arm.get("wins", 0) or 0),
        "actions_executed": int(
            arm.get("actions_executed", 0) or 0
        ),
        "protected_route_preemptions": int(
            arm.get("protected_route_preemptions", 0) or 0
        ),
        "progressive_route_actions": int(
            arm.get("progressive_terminal_route_actions", 0) or 0
        ),
        "level_route_replay_actions": int(
            arm.get("level_route_replay_actions", 0) or 0
        ),
        "frontier_experiments": int(
            arm.get("frontier_experiments", 0) or 0
        ),
        "frontier_per_level_rearms": int(
            arm.get("frontier_per_level_rearms", 0) or 0
        ),
        "multiform_selections": int(
            arm.get("terminal_multiform_selections", 0) or 0
        ),
        "multiform_demotions": int(
            arm.get("terminal_multiform_demotions", 0) or 0
        ),
        "controller_errors": list(
            arm.get("controller_errors", ()) or ()
        ),
    }


def _multiform_liveness_probe() -> Dict[str, int]:
    learner = OnlineMultiformRelationalLearner(
        minimum_terminal_support=2,
    )
    empty = np.zeros((7, 9), dtype=np.int32)
    for color in (2, 7):
        before = empty.copy()
        before[2:4, 2:4] = color
        learner.start_branch()
        learner.observe_transition(
            observation_before=_observation(before),
            observation_after=_observation(empty),
            action_name="ACTION6",
            action_data={"x": 2, "y": 2},
            terminal_success=True,
            game_over=False,
        )
    target = empty.copy()
    target[2:4, 2:4] = 9
    selection = learner.select(
        observation=_observation(target),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(
                name="ACTION6",
                action_args={"x": 2, "y": 2},
            ),
        ),
    )
    initial_selections = int(selection is not None)
    for _ in range(2):
        if selection is None:
            break
        learner.observe_transition(
            observation_before=_observation(target),
            observation_after=_observation(target),
            action_name=selection.action_name,
            action_data=selection.action_data,
            terminal_success=False,
            game_over=False,
        )
        selection = learner.select(
            observation=_observation(target),
            available_actions=("ACTION6",),
            available_action_candidates=(
                SimpleNamespace(
                    name="ACTION6",
                    action_args={"x": 2, "y": 2},
                ),
            ),
        )
        initial_selections += int(selection is not None)
    learner.start_branch()
    support = empty.copy()
    support[2:4, 2:4] = 5
    learner.observe_transition(
        observation_before=_observation(support),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data={"x": 2, "y": 2},
        terminal_success=True,
        game_over=False,
    )
    reactivated = learner.select(
        observation=_observation(target),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(
                name="ACTION6",
                action_args={"x": 2, "y": 2},
            ),
        ),
    )
    summary = learner.summary()
    return {
        "terminal_examples": int(summary["terminal_examples"]),
        "confirmed_patterns": int(summary["confirmed_patterns"]),
        "selections": initial_selections + int(reactivated is not None),
        "demotions": int(summary["demotions"]),
        "reactivations": int(summary["reactivations"]),
    }


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=("ACTION6",),
        game_state="NOT_FINISHED",
        levels_completed=0,
        infer_players=False,
    )


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.10e ft09 authority regression gates.",
    )
    parser.add_argument("--game", default=DEFAULT_GAME_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--resets", type=int, default=14)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    payload = run_ft09_authority_repair_benchmark(
        game_id=args.game,
        seed=args.seed,
        action_budget_per_reset=args.budget,
        resets=args.resets,
        write_path=args.out,
    )
    return 0 if payload["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
