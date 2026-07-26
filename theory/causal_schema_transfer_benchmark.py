"""Frozen source-to-target causal-schema transfer benchmark.

The protocol deliberately separates source learning from target evaluation:

1. run one source controller online;
2. freeze its terminal-grounded causal-schema library;
3. construct fresh target controllers from that immutable snapshot;
4. compare transfer-active and transfer-ablated target arms on identical
   seeds, reset states, and budgets.

The source controller is never reused on a target, so no mutable route,
palette, coordinate, or exact-state memory can cross the boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import game_splits

from theory.non_ar25_active_micro_run import _env_dir

from .online_transferable_causal_schema import FrozenCausalSchemaLibrary
from .unified_cognition_ab_benchmark import EnvFactory, _run_arm
from .unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


SCHEMA_VERSION = "sage.causal_schema_transfer.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage10f_causal_schema_transfer_benchmark.json"
)


def run_causal_schema_transfer_benchmark(
    *,
    source_game_id: str,
    target_game_ids: Sequence[str],
    seed: int = 0,
    source_action_budget_per_reset: int = 160,
    source_resets: int = 14,
    target_action_budget_per_reset: int = 160,
    target_resets: int = 4,
    minimum_source_terminal_support: int = 1,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Run a frozen source -> fresh target A/B protocol."""
    env_directory = (
        Path(environments_dir)
        if environments_dir is not None
        else _env_dir()
    )
    source_holder: Dict[str, UnifiedCognitiveController] = {}

    def source_factory(game_id: str) -> UnifiedCognitiveController:
        controller = UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(
                enable_transferable_causal_schema_priors=False,
            ),
        )
        source_holder["controller"] = controller
        return controller

    source = _run_arm(
        arm="unified",
        game_id=str(source_game_id),
        seed=int(seed),
        action_budget_per_reset=max(
            0,
            int(source_action_budget_per_reset),
        ),
        resets=max(1, int(source_resets)),
        env_dir=env_directory,
        env_factory=env_factory,
        controller_factory=source_factory,
    )
    source_controller = source_holder["controller"]
    library = source_controller.freeze_transferable_causal_schemas(
        minimum_terminal_support=minimum_source_terminal_support,
    )

    targets = _evaluate_targets(
        library=library,
        target_game_ids=target_game_ids,
        seed=seed,
        action_budget_per_reset=target_action_budget_per_reset,
        resets=target_resets,
        env_directory=env_directory,
        env_factory=env_factory,
    )

    source_export = dict(
        (
            source.get("controller_summary", {})
            or {}
        ).get("transferable_causal_schema_export", {})
        or {}
    )
    target_probe_count = sum(
        int(
            target["transfer_mechanisms"].get(
                "candidate_probe_selections",
                0,
            )
            or 0
        )
        for target in targets
    )
    target_confirmation_count = sum(
        int(
            target["transfer_mechanisms"].get(
                "effect_confirmations",
                0,
            )
            or 0
        )
        for target in targets
    )
    target_promotion_count = sum(
        int(
            target["transfer_mechanisms"].get("promotions", 0)
            or 0
        )
        for target in targets
    )
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "source_game_id": str(source_game_id),
            "target_game_ids": [
                str(item) for item in target_game_ids
            ],
            "seed": int(seed),
            "source_action_budget_per_reset": int(
                source_action_budget_per_reset
            ),
            "source_resets": int(source_resets),
            "target_action_budget_per_reset": int(
                target_action_budget_per_reset
            ),
            "target_resets": int(target_resets),
            "source_controller_reused_on_targets": False,
            "library_frozen_before_target_evaluation": True,
        },
        "source": _compact_arm(source),
        "source_export": source_export,
        "frozen_library": library.to_dict(),
        "targets": targets,
        "aggregate": {
            "source_schemas": len(library.schemas),
            "target_probe_selections": target_probe_count,
            "target_effect_confirmations": target_confirmation_count,
            "target_promotions": target_promotion_count,
            "target_level_delta": sum(
                int(target["delta"]["levels_completed"])
                for target in targets
            ),
            "target_win_delta": sum(
                int(target["delta"]["wins"])
                for target in targets
            ),
        },
    }
    payload["gates"] = {
        "G1_frozen_abstract_library": bool(
            library.schemas
            and not source_export.get("contains_game_identity", True)
            and not source_export.get("contains_palette_values", True)
            and not source_export.get("contains_coordinates", True)
            and not source_export.get(
                "contains_grid_or_state_hashes",
                True,
            )
        ),
        "G2_source_evidence_is_probe_only": all(
            not bool(
                target["transfer_mechanisms"].get(
                    "source_evidence_grants_policy_authority",
                    True,
                )
            )
            and bool(
                target["transfer_mechanisms"].get(
                    "promotion_requires_target_terminal",
                    False,
                )
            )
            for target in targets
        ),
        "G3_target_mechanism_activated": bool(target_probe_count),
        "G4_zero_protected_preemptions": all(
            int(
                target["transfer_active"].get(
                    "protected_route_preemptions",
                    0,
                )
                or 0
            )
            == 0
            for target in targets
        ),
        "G5_target_terminal_promotion_observed": bool(
            target_promotion_count
        ),
    }
    if write_path is not None:
        write_causal_schema_transfer_benchmark(payload, write_path)
    return payload


def run_frozen_causal_schema_target_benchmark(
    *,
    library: FrozenCausalSchemaLibrary,
    target_game_ids: Sequence[str],
    seed: int = 0,
    action_budget_per_reset: int = 160,
    resets: int = 4,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate an already frozen library without rerunning its source."""
    env_directory = (
        Path(environments_dir)
        if environments_dir is not None
        else _env_dir()
    )
    targets = _evaluate_targets(
        library=library,
        target_game_ids=target_game_ids,
        seed=seed,
        action_budget_per_reset=action_budget_per_reset,
        resets=resets,
        env_directory=env_directory,
        env_factory=env_factory,
    )
    probe_count = sum(
        int(
            target["transfer_mechanisms"].get(
                "candidate_probe_selections",
                0,
            )
            or 0
        )
        for target in targets
    )
    confirmation_count = sum(
        int(
            target["transfer_mechanisms"].get(
                "effect_confirmations",
                0,
            )
            or 0
        )
        for target in targets
    )
    promotion_count = sum(
        int(target["transfer_mechanisms"].get("promotions", 0) or 0)
        for target in targets
    )
    payload: Dict[str, Any] = {
        "schema_version": "sage.causal_schema_target_only.v1",
        "protocol": {
            "target_game_ids": [
                str(item) for item in target_game_ids
            ],
            "seed": int(seed),
            "action_budget_per_reset": int(action_budget_per_reset),
            "resets": int(resets),
            "source_controller_available": False,
            "library_imported_frozen": True,
        },
        "frozen_library": library.to_dict(),
        "targets": targets,
        "aggregate": {
            "source_schemas": len(library.schemas),
            "target_probe_selections": probe_count,
            "target_effect_confirmations": confirmation_count,
            "target_promotions": promotion_count,
            "target_level_delta": sum(
                int(target["delta"]["levels_completed"])
                for target in targets
            ),
            "target_win_delta": sum(
                int(target["delta"]["wins"])
                for target in targets
            ),
        },
        "gates": {
            "G1_frozen_library_imported": bool(library.schemas),
            "G2_source_evidence_is_probe_only": all(
                not bool(
                    target["transfer_mechanisms"].get(
                        "source_evidence_grants_policy_authority",
                        True,
                    )
                )
                and bool(
                    target["transfer_mechanisms"].get(
                        "promotion_requires_target_terminal",
                        False,
                    )
                )
                for target in targets
            ),
            "G3_target_mechanism_activated": bool(probe_count),
            "G4_zero_protected_preemptions": all(
                int(
                    target["transfer_active"].get(
                        "protected_route_preemptions",
                        0,
                    )
                    or 0
                )
                == 0
                for target in targets
            ),
            "G5_target_terminal_promotion_observed": bool(
                promotion_count
            ),
        },
    }
    if write_path is not None:
        write_causal_schema_transfer_benchmark(payload, write_path)
    return payload


def write_causal_schema_transfer_benchmark(
    payload: Mapping[str, Any],
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _target_factory(
    *,
    library: FrozenCausalSchemaLibrary,
    transfer_enabled: bool,
):
    def factory(game_id: str) -> UnifiedCognitiveController:
        return UnifiedCognitiveController(
            game_id,
            frozen_causal_schema_library=library,
            config=UnifiedCognitiveConfig(
                enable_transferable_causal_schema_priors=transfer_enabled,
            ),
        )

    return factory


def _evaluate_targets(
    *,
    library: FrozenCausalSchemaLibrary,
    target_game_ids: Sequence[str],
    seed: int,
    action_budget_per_reset: int,
    resets: int,
    env_directory: Path,
    env_factory: EnvFactory | None,
) -> list[Dict[str, Any]]:
    targets = []
    for target_game_id in tuple(str(item) for item in target_game_ids):
        baseline = _run_arm(
            arm="unified",
            game_id=target_game_id,
            seed=int(seed),
            action_budget_per_reset=max(
                0,
                int(action_budget_per_reset),
            ),
            resets=max(1, int(resets)),
            env_dir=env_directory,
            env_factory=env_factory,
            controller_factory=_target_factory(
                library=FrozenCausalSchemaLibrary(),
                transfer_enabled=False,
            ),
        )
        active = _run_arm(
            arm="unified",
            game_id=target_game_id,
            seed=int(seed),
            action_budget_per_reset=max(
                0,
                int(action_budget_per_reset),
            ),
            resets=max(1, int(resets)),
            env_dir=env_directory,
            env_factory=env_factory,
            controller_factory=_target_factory(
                library=library,
                transfer_enabled=True,
            ),
        )
        active_transfer = dict(
            (
                active.get("controller_summary", {})
                or {}
            ).get("transferable_causal_schema_transfer", {})
            or {}
        )
        targets.append({
            "game_id": target_game_id,
            "same_fresh_reset_states": (
                baseline.get("reset_visual_digests")
                == active.get("reset_visual_digests")
            ),
            "same_action_budget": (
                baseline.get("configured_action_budget")
                == active.get("configured_action_budget")
            ),
            "transfer_ablated": _compact_arm(baseline),
            "transfer_active": _compact_arm(active),
            "delta": {
                "levels_completed": (
                    int(active.get("levels_completed_delta", 0) or 0)
                    - int(baseline.get("levels_completed_delta", 0) or 0)
                ),
                "wins": (
                    int(active.get("wins", 0) or 0)
                    - int(baseline.get("wins", 0) or 0)
                ),
                "max_level_reached": (
                    int(active.get("max_level_reached", 0) or 0)
                    - int(baseline.get("max_level_reached", 0) or 0)
                ),
            },
            "transfer_mechanisms": active_transfer,
        })
    return targets


def _compact_arm(arm: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "game_id": str(arm.get("game_id", "")),
        "actions_executed": int(arm.get("actions_executed", 0) or 0),
        "max_level_reached": int(
            arm.get("max_level_reached", 0) or 0
        ),
        "levels_completed": int(
            arm.get("levels_completed_delta", 0) or 0
        ),
        "wins": int(arm.get("wins", 0) or 0),
        "controller_errors": list(arm.get("controller_errors", []) or []),
        "protected_route_preemptions": int(
            arm.get("protected_route_preemptions", 0) or 0
        ),
        "frontier_eligibility_assessments": int(
            arm.get("frontier_eligibility_assessments", 0) or 0
        ),
        "frontier_context_actuator_demotions": int(
            arm.get("frontier_context_actuator_demotions", 0) or 0
        ),
        "causal_schema_probe_selections": int(
            arm.get("causal_schema_probe_selections", 0) or 0
        ),
        "causal_schema_effect_confirmations": int(
            arm.get("causal_schema_effect_confirmations", 0) or 0
        ),
        "causal_schema_cross_family_adapter_probes": int(
            arm.get(
                "causal_schema_cross_family_adapter_probes",
                0,
            )
            or 0
        ),
        "causal_schema_cross_family_adapter_confirmations": int(
            arm.get(
                "causal_schema_cross_family_adapter_confirmations",
                0,
            )
            or 0
        ),
        "causal_schema_terminal_backcredits": int(
            arm.get("causal_schema_terminal_backcredits", 0) or 0
        ),
        "causal_schema_promotions": int(
            arm.get("causal_schema_promotions", 0) or 0
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen source-to-target causal-schema transfer.",
    )
    parser.add_argument("--source", default="")
    parser.add_argument(
        "--library-in",
        default="",
        help=(
            "Reuse a frozen library JSON (or a prior benchmark containing "
            "frozen_library) and skip source training."
        ),
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-budget", type=int, default=160)
    parser.add_argument("--source-resets", type=int, default=14)
    parser.add_argument("--target-budget", type=int, default=160)
    parser.add_argument("--target-resets", type=int, default=4)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(list(argv) if argv is not None else None)
    targets = [
        game_splits.resolve_full_game_id(item.strip())
        for item in str(args.targets).split(",")
        if item.strip()
    ]
    if args.library_in:
        imported_payload = json.loads(
            Path(args.library_in).read_text(encoding="utf-8")
        )
        library_payload = imported_payload.get(
            "frozen_library",
            imported_payload,
        )
        library = FrozenCausalSchemaLibrary.from_dict(library_payload)
        payload = run_frozen_causal_schema_target_benchmark(
            library=library,
            target_game_ids=targets,
            seed=args.seed,
            action_budget_per_reset=args.target_budget,
            resets=args.target_resets,
            environments_dir=args.environments_dir,
            write_path=args.out,
        )
    else:
        if not args.source:
            parser.error("--source is required unless --library-in is used")
        source = game_splits.resolve_full_game_id(str(args.source))
        payload = run_causal_schema_transfer_benchmark(
            source_game_id=source,
            target_game_ids=targets,
            seed=args.seed,
            source_action_budget_per_reset=args.source_budget,
            source_resets=args.source_resets,
            target_action_budget_per_reset=args.target_budget,
            target_resets=args.target_resets,
            environments_dir=args.environments_dir,
            write_path=args.out,
        )
    print(json.dumps({
        "output": str(args.out),
        "aggregate": payload["aggregate"],
        "gates": payload["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "run_causal_schema_transfer_benchmark",
    "run_frozen_causal_schema_target_benchmark",
    "write_causal_schema_transfer_benchmark",
]
