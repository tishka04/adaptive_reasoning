"""Paired held-out A/B benchmark for the unified cognitive controller.

Each arm receives a fresh offline environment, the same game/seed/reset/action
budget, and the same deterministic legacy proposal policy.  The legacy-only
arm executes that proposal directly; the unified arm may override it and learns
only from transitions observed during that arm.  Evaluation outcomes are never
fed into another game or seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

import game_splits

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import (
    _configure_offline_env,
    _env_dir,
    _valid_actions,
)
from theory.real_env_option_adapter import snapshot_frame

from .unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


DEFAULT_OUTPUT_PATH = (
    Path("diagnostics") / "sage" / "unified_cognition_ab_held_out.json"
)
DEFAULT_HELD_OUT_GAMES = tuple(
    game_splits.resolve("public_unseen_split", full_ids=True)
)
SCHEMA_VERSION = "sage.unified_cognition_ab_held_out.v22"
WIN_STATES = {"WIN", "WON", "VICTORY"}
TERMINAL_STATES = WIN_STATES | {"GAME_OVER", "TERMINATED", "FINISHED"}
EXPERIMENT_SOURCES = {
    "discriminating_experiment",
    "relational_experiment",
    "terminal_objective_probe",
    "terminal_objective_discriminator",
    "terminal_objective_ablation",
    "temporal_subgoal_probe",
    "causal_option_downstream_probe",
    "causal_option_effect_subgoal_probe",
    "causal_option_mediated_discrimination",
    "causal_option_mode_restoration",
    "causal_option_mediated_exploitation",
    "causal_option_mediated_exploitation_restoration",
    "causal_option_mediated_replication",
}

EnvFactory = Callable[[str], Any]
ControllerFactory = Callable[[str], UnifiedCognitiveController]


def _causal_disabled_controller(game_id: str) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(enable_causal_subgoal_induction=False),
    )


def _causal_effect_credit_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(enable_causal_effect_credit=False),
    )


def _causal_hierarchical_options_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_causal_hierarchical_options=False
        ),
    )


def _effect_conditioned_downstream_subgoals_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_effect_conditioned_downstream_subgoals=False
        ),
    )


def _state_conditioned_directional_control_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_state_conditioned_directional_control=False
        ),
    )


def _persistent_directional_pursuit_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_persistent_directional_pursuit=False
        ),
    )


def _entity_anchored_interventions_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_entity_anchored_interventions=False
        ),
    )


def _active_entity_causal_binding_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_active_entity_causal_binding=False
        ),
    )


def _mediated_entity_effect_induction_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_mediated_entity_effect_induction=False
        ),
    )


def _active_mediated_replication_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_active_mediated_replication=False
        ),
    )


def _active_mediated_discrimination_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_active_mediated_discrimination=False
        ),
    )


def _active_mode_restoration_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_active_mode_restoration=False
        ),
    )


def _terminal_mediated_exploitation_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_terminal_mediated_exploitation=False
        ),
    )


def _successor_policy_chaining_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_successor_policy_chaining=False,
            enable_active_successor_exploration=False,
        ),
    )


def _successor_structural_transfer_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_successor_structural_transfer=False,
        ),
    )


def _horizon_stable_learning_epochs_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_horizon_stable_learning_epochs=False,
        ),
    )


def _online_horizon_learning_arbiter_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_online_horizon_learning_arbiter=False,
        ),
    )


def _online_mediated_anti_unification_disabled_controller(
    game_id: str,
) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            enable_online_mediated_anti_unification=False
        ),
    )


@dataclass(frozen=True)
class _ExecutableAction:
    name: str
    raw_action: Any
    action_args: Dict[str, Any] = field(default_factory=dict)


class SharedLegacyProposalPolicy:
    """Deterministic balanced legal policy shared exactly by both A/B arms."""

    def __init__(self, *, game_id: str, seed: int, reset_index: int) -> None:
        self.rng = random.Random(
            f"unified-ab:{game_id}:{int(seed)}:{int(reset_index)}"
        )
        self.family_counts: Counter[str] = Counter()
        self.concrete_counts: Counter[str] = Counter()

    def select(self, actions: Sequence[Any]) -> Any | None:
        legal = [
            action
            for action in actions
            if str(getattr(action, "name", "")) not in {"", "RESET"}
        ]
        if not legal:
            return None
        minimum_family = min(
            self.family_counts[str(getattr(action, "name", ""))]
            for action in legal
        )
        family_candidates = [
            action
            for action in legal
            if self.family_counts[str(getattr(action, "name", ""))]
            == minimum_family
        ]
        minimum_concrete = min(
            self.concrete_counts[_action_identity(action)]
            for action in family_candidates
        )
        concrete_candidates = [
            action
            for action in family_candidates
            if self.concrete_counts[_action_identity(action)] == minimum_concrete
        ]
        selected = concrete_candidates[
            self.rng.randrange(len(concrete_candidates))
        ]
        name = str(getattr(selected, "name", ""))
        self.family_counts[name] += 1
        self.concrete_counts[_action_identity(selected)] += 1
        return selected


def run_unified_cognition_ab_benchmark(
    *,
    game_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (0,),
    action_budget_per_reset: int = 40,
    resets: int = 1,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
    controller_factory: ControllerFactory | None = None,
    enable_causal_subgoal_induction: bool = True,
    enable_causal_effect_credit: bool = True,
    enable_causal_hierarchical_options: bool = True,
    enable_effect_conditioned_downstream_subgoals: bool = True,
    enable_state_conditioned_directional_control: bool = True,
    enable_persistent_directional_pursuit: bool = True,
    enable_entity_anchored_interventions: bool = True,
    enable_active_entity_causal_binding: bool = True,
    enable_mediated_entity_effect_induction: bool = True,
    enable_online_mediated_anti_unification: bool = True,
    enable_active_mediated_discrimination: bool = True,
    enable_active_mode_restoration: bool = True,
    enable_terminal_mediated_exploitation: bool = True,
    enable_successor_policy_chaining: bool = True,
    enable_successor_structural_transfer: bool = True,
    enable_active_mediated_replication: bool = True,
    enable_horizon_stable_learning_epochs: bool = True,
    enable_online_horizon_learning_arbiter: bool = True,
    write_path: str | Path | None = None,
    include_traces: bool = False,
) -> Dict[str, Any]:
    """Run paired legacy-only/unified online-learning episodes."""
    games = tuple(str(game) for game in (game_ids or DEFAULT_HELD_OUT_GAMES))
    normalized_seeds = tuple(int(seed) for seed in seeds)
    budget = max(0, int(action_budget_per_reset))
    reset_count = max(1, int(resets))
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    effective_controller_factory = controller_factory
    if effective_controller_factory is None and not enable_causal_subgoal_induction:
        effective_controller_factory = _causal_disabled_controller
    elif effective_controller_factory is None and not enable_causal_effect_credit:
        effective_controller_factory = _causal_effect_credit_disabled_controller
    elif (
        effective_controller_factory is None
        and not enable_causal_hierarchical_options
    ):
        effective_controller_factory = (
            _causal_hierarchical_options_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_effect_conditioned_downstream_subgoals
    ):
        effective_controller_factory = (
            _effect_conditioned_downstream_subgoals_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_state_conditioned_directional_control
    ):
        effective_controller_factory = (
            _state_conditioned_directional_control_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_persistent_directional_pursuit
    ):
        effective_controller_factory = (
            _persistent_directional_pursuit_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_entity_anchored_interventions
    ):
        effective_controller_factory = (
            _entity_anchored_interventions_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_active_entity_causal_binding
    ):
        effective_controller_factory = (
            _active_entity_causal_binding_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_mediated_entity_effect_induction
    ):
        effective_controller_factory = (
            _mediated_entity_effect_induction_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_online_mediated_anti_unification
    ):
        effective_controller_factory = (
            _online_mediated_anti_unification_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_active_mediated_discrimination
    ):
        effective_controller_factory = (
            _active_mediated_discrimination_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_active_mode_restoration
    ):
        effective_controller_factory = (
            _active_mode_restoration_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_terminal_mediated_exploitation
    ):
        effective_controller_factory = (
            _terminal_mediated_exploitation_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_successor_policy_chaining
    ):
        effective_controller_factory = (
            _successor_policy_chaining_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_successor_structural_transfer
    ):
        effective_controller_factory = (
            _successor_structural_transfer_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_active_mediated_replication
    ):
        effective_controller_factory = (
            _active_mediated_replication_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_horizon_stable_learning_epochs
    ):
        effective_controller_factory = (
            _horizon_stable_learning_epochs_disabled_controller
        )
    elif (
        effective_controller_factory is None
        and not enable_online_horizon_learning_arbiter
    ):
        effective_controller_factory = (
            _online_horizon_learning_arbiter_disabled_controller
        )

    pairs: List[Dict[str, Any]] = []
    for game_id in games:
        for seed in normalized_seeds:
            legacy = _run_arm(
                arm="legacy_only",
                game_id=game_id,
                seed=seed,
                action_budget_per_reset=budget,
                resets=reset_count,
                env_dir=env_dir,
                env_factory=env_factory,
                controller_factory=effective_controller_factory,
            )
            unified = _run_arm(
                arm="unified",
                game_id=game_id,
                seed=seed,
                action_budget_per_reset=budget,
                resets=reset_count,
                env_dir=env_dir,
                env_factory=env_factory,
                controller_factory=effective_controller_factory,
            )
            reset_match = (
                legacy["reset_visual_digests"]
                == unified["reset_visual_digests"]
                and len(legacy["reset_visual_digests"]) == reset_count
            )
            pairs.append({
                "game_id": game_id,
                "seed": seed,
                "same_fresh_reset_states": reset_match,
                "same_action_budget": (
                    legacy["configured_action_budget"]
                    == unified["configured_action_budget"]
                    == budget * reset_count
                ),
                "same_reset_count": (
                    legacy["resets_executed"]
                    == unified["resets_executed"]
                    == reset_count
                ),
                "legacy_only": legacy,
                "unified": unified,
                "delta": _pair_delta(legacy, unified),
            })

    payload = _summarize_benchmark(
        pairs,
        games=games,
        seeds=normalized_seeds,
        action_budget_per_reset=budget,
        resets=reset_count,
        causal_subgoal_induction_enabled=enable_causal_subgoal_induction,
        causal_effect_credit_enabled=enable_causal_effect_credit,
        causal_hierarchical_options_enabled=(
            enable_causal_hierarchical_options
        ),
        effect_conditioned_downstream_subgoals_enabled=(
            enable_effect_conditioned_downstream_subgoals
        ),
        state_conditioned_directional_control_enabled=(
            enable_state_conditioned_directional_control
        ),
        persistent_directional_pursuit_enabled=(
            enable_persistent_directional_pursuit
        ),
        entity_anchored_interventions_enabled=(
            enable_entity_anchored_interventions
        ),
        active_entity_causal_binding_enabled=(
            enable_active_entity_causal_binding
        ),
        mediated_entity_effect_induction_enabled=(
            enable_mediated_entity_effect_induction
        ),
        online_mediated_anti_unification_enabled=(
            enable_online_mediated_anti_unification
        ),
        active_mediated_discrimination_enabled=(
            enable_active_mediated_discrimination
        ),
        active_mode_restoration_enabled=enable_active_mode_restoration,
        terminal_mediated_exploitation_enabled=(
            enable_terminal_mediated_exploitation
        ),
        successor_policy_chaining_enabled=(
            enable_successor_policy_chaining
        ),
        successor_structural_transfer_enabled=(
            enable_successor_structural_transfer
        ),
        active_mediated_replication_enabled=(
            enable_active_mediated_replication
        ),
        horizon_stable_learning_epochs_enabled=(
            enable_horizon_stable_learning_epochs
        ),
        online_horizon_learning_arbiter_enabled=(
            enable_online_horizon_learning_arbiter
        ),
    )
    if not include_traces:
        _omit_step_traces(payload)
    if write_path is not None:
        write_unified_cognition_ab_benchmark(payload, write_path)
    return payload


def write_unified_cognition_ab_benchmark(
    payload: Mapping[str, Any],
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_arm(
    *,
    arm: str,
    game_id: str,
    seed: int,
    action_budget_per_reset: int,
    resets: int,
    env_dir: Path,
    env_factory: EnvFactory | None,
    controller_factory: ControllerFactory | None,
) -> Dict[str, Any]:
    controller = (
        (
            controller_factory(game_id)
            if controller_factory is not None
            else UnifiedCognitiveController(game_id)
        )
        if arm == "unified"
        else None
    )
    attempts: List[Dict[str, Any]] = []
    decision_sources: Counter[str] = Counter()
    controller_errors: List[str] = []
    for reset_index in range(resets):
        if controller is not None:
            controller.on_reset()
        policy = SharedLegacyProposalPolicy(
            game_id=game_id,
            seed=seed,
            reset_index=reset_index,
        )
        attempt = _run_attempt(
            arm=arm,
            game_id=game_id,
            reset_index=reset_index,
            action_budget=action_budget_per_reset,
            env_dir=env_dir,
            env_factory=env_factory,
            policy=policy,
            controller=controller,
            decision_sources=decision_sources,
            controller_errors=controller_errors,
        )
        attempts.append(attempt)

    failure_causes = Counter(
        str(attempt.get("failure_cause", ""))
        for attempt in attempts
        if str(attempt.get("failure_cause", ""))
    )
    total_actions = sum(int(attempt["actions_executed"]) for attempt in attempts)
    experiments = sum(
        count
        for source, count in decision_sources.items()
        if source in EXPERIMENT_SOURCES
    )
    controller_summary = controller.summary() if controller is not None else {}
    terminal_summary = dict(
        controller_summary.get("terminal_objectives", {}) or {}
    )
    temporal_summary = dict(
        controller_summary.get("temporal_goal_composition", {}) or {}
    )
    causal_summary = dict(
        controller_summary.get("causal_subgoal_graph", {}) or {}
    )
    causal_option_summary = dict(
        controller_summary.get("causal_hierarchical_options", {}) or {}
    )
    effect_subgoal_summary = dict(
        causal_option_summary.get("effect_conditioned_subgoals", {}) or {}
    )
    directional_summary = dict(
        effect_subgoal_summary.get(
            "state_conditioned_directional_model",
            {},
        ) or {}
    )
    persistent_summary = dict(
        causal_option_summary.get(
            "persistent_directional_pursuit",
            {},
        ) or {}
    )
    entity_binding_summary = dict(
        causal_option_summary.get(
            "active_entity_causal_binding",
            {},
        ) or {}
    )
    mediated_effect_summary = dict(
        causal_option_summary.get(
            "mediated_entity_effect_induction",
            {},
        ) or {}
    )
    mediated_replication_summary = dict(
        causal_option_summary.get(
            "active_mediated_replication",
            {},
        ) or {}
    )
    mediated_discrimination_summary = dict(
        causal_option_summary.get(
            "active_mediated_discrimination",
            {},
        ) or {}
    )
    mediated_exploitation_summary = dict(
        causal_option_summary.get(
            "terminal_mediated_exploitation",
            {},
        ) or {}
    )
    horizon_arbiter_summary = dict(
        controller_summary.get(
            "online_horizon_learning_arbiter",
            {},
        ) or {}
    )
    return {
        "arm": arm,
        "game_id": game_id,
        "seed": seed,
        "configured_action_budget": action_budget_per_reset * resets,
        "action_budget_per_reset": action_budget_per_reset,
        "resets_configured": resets,
        "resets_executed": len(attempts),
        "fresh_environment_instances": len(attempts),
        "reset_visual_digests": [
            str(attempt.get("reset_visual_digest", "")) for attempt in attempts
        ],
        "actions_executed": total_actions,
        "operator_plan_actions": decision_sources["operator_plan"],
        "operator_plan_actions_since_objective_progress": int(
            controller_summary.get(
                "operator_plan_actions_since_objective_progress",
                0,
            )
            or 0
        ),
        "operator_plan_streak_peak": int(
            controller_summary.get("operator_plan_streak_peak", 0) or 0
        ),
        "operator_plan_budget_blocks": int(
            controller_summary.get("operator_plan_budget_blocks", 0) or 0
        ),
        "operator_plan_progress_resets": int(
            controller_summary.get("operator_plan_progress_resets", 0) or 0
        ),
        "horizon_arbiter_evaluations": int(
            horizon_arbiter_summary.get("evaluations", 0) or 0
        ),
        "horizon_arbiter_reservations": int(
            horizon_arbiter_summary.get("reservations", 0) or 0
        ),
        "horizon_arbiter_releases": int(
            horizon_arbiter_summary.get("releases", 0) or 0
        ),
        "horizon_arbiter_causal_uncertainty_reservations": int(
            horizon_arbiter_summary.get(
                "causal_uncertainty_reservations",
                0,
            )
            or 0
        ),
        "horizon_arbiter_terminal_test_reservations": int(
            horizon_arbiter_summary.get(
                "terminal_test_reservations",
                0,
            )
            or 0
        ),
        "horizon_arbiter_priority_peak": int(
            horizon_arbiter_summary.get("priority_peak", 0) or 0
        ),
        "levels_completed_delta": sum(
            int(attempt["levels_completed_delta"]) for attempt in attempts
        ),
        "max_level_reached": max(
            (int(attempt["max_level_reached"]) for attempt in attempts),
            default=0,
        ),
        "wins": sum(int(bool(attempt["win"])) for attempt in attempts),
        "experiment_actions": experiments,
        "experiment_cost_rate": round(experiments / total_actions, 6)
        if total_actions
        else 0.0,
        "promoted_option_actions": (
            decision_sources["terminal_objective_option"]
            + decision_sources["terminal_objective_preparation"]
        ),
        "option_preparation_actions": (
            decision_sources["terminal_objective_preparation"]
        ),
        "terminal_objective_probe_actions": decision_sources[
            "terminal_objective_probe"
        ],
        "terminal_objective_discriminator_actions": decision_sources[
            "terminal_objective_discriminator"
        ],
        "terminal_objective_ablation_actions": decision_sources[
            "terminal_objective_ablation"
        ],
        "terminal_objective_grounded_actions": decision_sources[
            "terminal_objective_option"
        ],
        "temporal_subgoal_probe_actions": decision_sources[
            "temporal_subgoal_probe"
        ],
        "temporal_subgoal_option_actions": decision_sources[
            "temporal_subgoal_option"
        ],
        "generated_goal_hypotheses": int(
            terminal_summary.get("objectives", 0) or 0
        ),
        "objective_distance_reductions": int(
            terminal_summary.get("distance_reductions", 0) or 0
        ),
        "objective_nonterminal_completions": int(
            terminal_summary.get("nonterminal_completions", 0) or 0
        ),
        "objective_ambiguous_terminal_events": int(
            terminal_summary.get("ambiguous_terminal_events", 0) or 0
        ),
        "terminal_supported_objectives": int(
            terminal_summary.get("terminal_supported_objectives", 0) or 0
        ),
        "refuted_objectives": int(
            terminal_summary.get("refuted_objectives", 0) or 0
        ),
        "unsafe_goal_plan_failures": int(
            terminal_summary.get("unsafe_plan_failures", 0) or 0
        ),
        "temporal_plans_generated": int(
            temporal_summary.get("plans_generated_total", 0) or 0
        ),
        "temporal_plan_starts": int(
            temporal_summary.get("plan_starts", 0) or 0
        ),
        "temporal_plan_actions": int(
            temporal_summary.get("actions", 0) or 0
        ),
        "temporal_progress_events": int(
            temporal_summary.get("progress_events", 0) or 0
        ),
        "temporal_step_completions": int(
            temporal_summary.get("step_completions", 0) or 0
        ),
        "temporal_local_completions": int(
            temporal_summary.get("local_completions", 0) or 0
        ),
        "temporal_nonterminal_completions": int(
            temporal_summary.get("nonterminal_completions", 0) or 0
        ),
        "temporal_plan_stalls": int(
            temporal_summary.get("stalls", 0) or 0
        ),
        "temporal_plan_abandonments": int(
            temporal_summary.get("abandonments", 0) or 0
        ),
        "temporal_unsafe_failures": int(
            temporal_summary.get("unsafe_failures", 0) or 0
        ),
        "temporal_terminal_bypasses": int(
            temporal_summary.get("terminal_bypasses", 0) or 0
        ),
        "terminal_supported_temporal_plans": int(
            temporal_summary.get("terminal_supported_plans", 0) or 0
        ),
        "refuted_temporal_plans": int(
            temporal_summary.get("refuted_plans", 0) or 0
        ),
        "causal_dependency_plans": int(
            temporal_summary.get("causal_dependency_plans", 0) or 0
        ),
        "causal_dependency_plan_starts": int(
            temporal_summary.get("causal_dependency_plan_starts", 0) or 0
        ),
        "causal_dependency_plan_actions": int(
            temporal_summary.get("causal_dependency_plan_actions", 0) or 0
        ),
        "causal_dependency_progress_events": int(
            temporal_summary.get("causal_dependency_progress_events", 0) or 0
        ),
        "causal_dependency_step_completions": int(
            temporal_summary.get("causal_dependency_step_completions", 0) or 0
        ),
        "causal_edges_generated": int(
            causal_summary.get("edges_generated_total", 0) or 0
        ),
        "causal_blocked_target_events": int(
            causal_summary.get("blocked_target_events", 0) or 0
        ),
        "causal_edge_trials": int(causal_summary.get("trials", 0) or 0),
        "causal_edge_actions": int(causal_summary.get("actions", 0) or 0),
        "causal_edge_source_progress_events": int(
            causal_summary.get("source_progress_events", 0) or 0
        ),
        "causal_edge_support_events": int(
            causal_summary.get("support_events", 0) or 0
        ),
        "causal_edge_contradictions": int(
            causal_summary.get("contradictions", 0) or 0
        ),
        "causal_availability_successes": int(
            causal_summary.get("availability_successes", 0) or 0
        ),
        "causal_availability_failures": int(
            causal_summary.get("availability_failures", 0) or 0
        ),
        "causal_cochange_supports": int(
            causal_summary.get("cochange_supports", 0) or 0
        ),
        "confirmed_causal_edges": int(
            causal_summary.get("confirmed_edges", 0) or 0
        ),
        "refuted_causal_edges": int(
            causal_summary.get("refuted_edges", 0) or 0
        ),
        "causal_edge_plan_failures": int(
            causal_summary.get("plan_failures", 0) or 0
        ),
        "causal_edge_unsafe_failures": int(
            causal_summary.get("unsafe_failures", 0) or 0
        ),
        "causal_effect_observations": int(
            causal_summary.get("effect_observations", 0) or 0
        ),
        "causal_effect_guided_actions": int(
            causal_summary.get("effect_guided_actions", 0) or 0
        ),
        "causal_productive_effect_signatures": int(
            causal_summary.get("productive_effect_signatures", 0) or 0
        ),
        "causal_productive_intervention_signatures": int(
            causal_summary.get("productive_intervention_signatures", 0) or 0
        ),
        "causal_delayed_credit_events": int(
            causal_summary.get("delayed_credit_events", 0) or 0
        ),
        "causal_expired_credit_windows": int(
            causal_summary.get("expired_credit_windows", 0) or 0
        ),
        "causal_cross_branch_confirmations": int(
            causal_summary.get("cross_branch_confirmations", 0) or 0
        ),
        "causal_reserved_confirmation_starts": int(
            temporal_summary.get("reserved_confirmation_starts", 0) or 0
        ),
        "causal_options_compiled": int(
            causal_option_summary.get("compiled_total", 0) or 0
        ),
        "causal_option_opening_events": int(
            causal_option_summary.get("opening_events", 0) or 0
        ),
        "causal_option_rollouts": int(
            causal_option_summary.get("rollouts", 0) or 0
        ),
        "causal_option_downstream_actions": int(
            causal_option_summary.get("downstream_actions", 0) or 0
        ),
        "causal_option_downstream_effects": int(
            causal_option_summary.get("downstream_effects", 0) or 0
        ),
        "causal_option_downstream_progress_events": int(
            causal_option_summary.get("downstream_progress_events", 0) or 0
        ),
        "causal_option_target_completions": int(
            causal_option_summary.get("target_completions", 0) or 0
        ),
        "causal_option_nonterminal_rollouts": int(
            causal_option_summary.get("nonterminal_rollouts", 0) or 0
        ),
        "causal_option_unsafe_rollouts": int(
            causal_option_summary.get("unsafe_rollouts", 0) or 0
        ),
        "causal_option_terminal_credited_events": int(
            causal_option_summary.get("credited_terminal_events", 0) or 0
        ),
        "terminal_supported_causal_options": int(
            causal_option_summary.get("terminal_supported_options", 0) or 0
        ),
        "terminal_refuted_causal_options": int(
            causal_option_summary.get("terminal_refuted_options", 0) or 0
        ),
        "causal_option_censored_openings": int(
            causal_option_summary.get("censored_openings", 0) or 0
        ),
        "entity_anchored_candidate_signatures": int(
            causal_option_summary.get(
                "entity_anchored_candidate_signatures",
                0,
            )
            or 0
        ),
        "entity_anchored_transfer_signatures": int(
            causal_option_summary.get(
                "entity_anchored_transfer_signatures",
                0,
            )
            or 0
        ),
        "entity_anchored_selections": int(
            causal_option_summary.get("entity_anchored_selections", 0) or 0
        ),
        "entity_binding_observations": int(
            entity_binding_summary.get("observations", 0) or 0
        ),
        "entity_binding_matched_entities": int(
            entity_binding_summary.get("matched_entities", 0) or 0
        ),
        "entity_binding_transformed_entities": int(
            entity_binding_summary.get("transformed_entities", 0) or 0
        ),
        "entity_binding_moved_entities": int(
            entity_binding_summary.get("moved_entities", 0) or 0
        ),
        "entity_binding_removed_entities": int(
            entity_binding_summary.get("removed_entities", 0) or 0
        ),
        "entity_binding_ambiguous_entities": int(
            entity_binding_summary.get("ambiguous_entities", 0) or 0
        ),
        "entity_binding_tracks_created": int(
            entity_binding_summary.get("tracks_created", 0) or 0
        ),
        "entity_binding_tracks_reused": int(
            entity_binding_summary.get("tracks_reused", 0) or 0
        ),
        "entity_binding_models": int(
            entity_binding_summary.get("binding_models", 0) or 0
        ),
        "entity_binding_carrier_progress_events": int(
            entity_binding_summary.get("carrier_progress_events", 0) or 0
        ),
        "entity_binding_carrier_regression_events": int(
            entity_binding_summary.get("carrier_regression_events", 0) or 0
        ),
        "entity_binding_noncarrier_progress_events": int(
            entity_binding_summary.get("noncarrier_progress_events", 0) or 0
        ),
        "entity_binding_conflicts": int(
            entity_binding_summary.get("binding_conflicts", 0) or 0
        ),
        "entity_binding_predictions": int(
            entity_binding_summary.get("predictions", 0) or 0
        ),
        "entity_binding_controlled_contrast_predictions": int(
            entity_binding_summary.get(
                "controlled_contrast_predictions",
                0,
            ) or 0
        ),
        "entity_binding_controlled_contrast_selections": int(
            entity_binding_summary.get(
                "controlled_contrast_selections",
                0,
            ) or 0
        ),
        "entity_binding_progressive_carrier_selections": int(
            entity_binding_summary.get(
                "progressive_carrier_selections",
                0,
            ) or 0
        ),
        "entity_binding_blocked_misbound_actions": int(
            entity_binding_summary.get("blocked_misbound_actions", 0) or 0
        ),
        "mediated_effect_observations": int(
            mediated_effect_summary.get("observations", 0) or 0
        ),
        "mediated_effect_scene_correspondences": int(
            mediated_effect_summary.get("scene_correspondences", 0) or 0
        ),
        "mediated_effect_changed_entities": int(
            mediated_effect_summary.get("changed_entities", 0) or 0
        ),
        "mediated_effect_moved_entities": int(
            mediated_effect_summary.get("moved_entities", 0) or 0
        ),
        "mediated_effect_transformed_entities": int(
            mediated_effect_summary.get("transformed_entities", 0) or 0
        ),
        "mediated_effect_removed_entities": int(
            mediated_effect_summary.get("removed_entities", 0) or 0
        ),
        "mediated_effect_appeared_entities": int(
            mediated_effect_summary.get("appeared_entities", 0) or 0
        ),
        "mediated_effect_ambiguous_entities": int(
            mediated_effect_summary.get("ambiguous_entities", 0) or 0
        ),
        "mediated_effect_tracks_created": int(
            mediated_effect_summary.get("tracks_created", 0) or 0
        ),
        "mediated_effect_tracks_reused": int(
            mediated_effect_summary.get("tracks_reused", 0) or 0
        ),
        "mediated_effect_progress_with_indirect_candidates": int(
            mediated_effect_summary.get(
                "progress_with_indirect_candidates",
                0,
            ) or 0
        ),
        "mediated_effect_ambiguous_progress_candidate_sets": int(
            mediated_effect_summary.get(
                "ambiguous_progress_candidate_sets",
                0,
            ) or 0
        ),
        "mediated_effect_no_candidate_progress_events": int(
            mediated_effect_summary.get(
                "no_candidate_progress_events",
                0,
            ) or 0
        ),
        "mediated_effect_direct_target_progress_events": int(
            mediated_effect_summary.get(
                "direct_target_progress_events",
                0,
            ) or 0
        ),
        "mediated_effect_models": int(
            mediated_effect_summary.get("mediated_effect_models", 0) or 0
        ),
        "mediated_effect_supported_hyperedges": int(
            mediated_effect_summary.get("supported_hyperedges", 0) or 0
        ),
        "mediated_abstraction_hypotheses": int(
            mediated_effect_summary.get(
                "abstract_hyperedge_hypotheses", 0
            ) or 0
        ),
        "mediated_abstraction_supported_hyperedges": int(
            mediated_effect_summary.get(
                "supported_abstract_hyperedges", 0
            ) or 0
        ),
        "mediated_abstraction_control_contexts": int(
            mediated_effect_summary.get("abstract_control_contexts", 0) or 0
        ),
        "mediated_abstraction_regression_contexts": int(
            mediated_effect_summary.get(
                "abstract_regression_contexts", 0
            ) or 0
        ),
        "mediated_effect_predictions": int(
            mediated_effect_summary.get("predictions", 0) or 0
        ),
        "mediated_effect_supported_predictions": int(
            mediated_effect_summary.get("supported_predictions", 0) or 0
        ),
        "mediated_effect_controlled_contrast_predictions": int(
            mediated_effect_summary.get(
                "controlled_contrast_predictions",
                0,
            ) or 0
        ),
        "mediated_effect_supported_selections": int(
            mediated_effect_summary.get("supported_selections", 0) or 0
        ),
        "mediated_effect_controlled_contrast_selections": int(
            mediated_effect_summary.get(
                "controlled_contrast_selections",
                0,
            ) or 0
        ),
        "mediated_effect_blocked_contradicted_actions": int(
            mediated_effect_summary.get(
                "blocked_contradicted_actions",
                0,
            ) or 0
        ),
        "mediated_discrimination_requests_created": int(
            mediated_discrimination_summary.get("requests_created", 0) or 0
        ),
        "mediated_discrimination_pending_requests": int(
            mediated_discrimination_summary.get("pending_requests", 0) or 0
        ),
        "mediated_discrimination_active_requests": int(
            mediated_discrimination_summary.get("active_requests", 0) or 0
        ),
        "mediated_discrimination_cross_branch_activations": int(
            mediated_discrimination_summary.get(
                "cross_branch_activations", 0
            ) or 0
        ),
        "mediated_discrimination_predictions": int(
            mediated_discrimination_summary.get("predictions", 0) or 0
        ),
        "mediated_discrimination_mode_mismatch_blocks": int(
            mediated_discrimination_summary.get(
                "mode_mismatch_blocks", 0
            ) or 0
        ),
        "mediated_discrimination_no_single_feature_blocks": int(
            mediated_discrimination_summary.get(
                "no_single_feature_blocks", 0
            ) or 0
        ),
        "mediated_discrimination_selections": int(
            mediated_discrimination_summary.get("selections", 0) or 0
        ),
        "mediated_discrimination_preparation_actions": int(
            mediated_discrimination_summary.get(
                "preparation_actions", 0
            ) or 0
        ),
        "mediated_discrimination_preparation_starts": int(
            temporal_summary.get(
                "mediated_discrimination_preparation_starts", 0
            ) or 0
        ),
        "mediated_discrimination_feature_requirements": int(
            mediated_discrimination_summary.get(
                "feature_requirements", 0
            ) or 0
        ),
        "mediated_discrimination_feature_eliminations": int(
            mediated_discrimination_summary.get(
                "feature_eliminations", 0
            ) or 0
        ),
        "mediated_discrimination_inconclusive_attempts": int(
            mediated_discrimination_summary.get(
                "inconclusive_attempts", 0
            ) or 0
        ),
        "mediated_discrimination_expirations": int(
            mediated_discrimination_summary.get("expirations", 0) or 0
        ),
        "mediated_restoration_actions": decision_sources[
            "causal_option_mode_restoration"
        ],
        "mediated_restoration_predictions": int(
            mediated_discrimination_summary.get(
                "restoration_predictions", 0
            ) or 0
        ),
        "mediated_restoration_selections": int(
            mediated_discrimination_summary.get(
                "restoration_selections", 0
            ) or 0
        ),
        "mediated_restoration_steps_confirmed": int(
            mediated_discrimination_summary.get(
                "restoration_steps_confirmed", 0
            ) or 0
        ),
        "mediated_restoration_targets_reached": int(
            mediated_discrimination_summary.get(
                "restoration_targets_reached", 0
            ) or 0
        ),
        "mediated_restoration_failures": int(
            mediated_discrimination_summary.get(
                "restoration_failures", 0
            ) or 0
        ),
        "mediated_restoration_unavailable_contexts": int(
            mediated_discrimination_summary.get(
                "restoration_unavailable_contexts", 0
            ) or 0
        ),
        "mediated_exploitation_actions": decision_sources[
            "causal_option_mediated_exploitation"
        ],
        "mediated_exploitation_restoration_actions": decision_sources[
            "causal_option_mediated_exploitation_restoration"
        ],
        "mediated_exploitation_policies_compiled": int(
            mediated_exploitation_summary.get("compiled", 0) or 0
        ),
        "mediated_exploitation_policy_revisions": int(
            mediated_exploitation_summary.get("revisions", 0) or 0
        ),
        "mediated_exploitation_activations": int(
            mediated_exploitation_summary.get("activations", 0) or 0
        ),
        "mediated_exploitation_predictions": int(
            mediated_exploitation_summary.get("predictions", 0) or 0
        ),
        "mediated_exploitation_mode_mismatch_blocks": int(
            mediated_exploitation_summary.get(
                "mode_mismatch_blocks", 0
            ) or 0
        ),
        "mediated_exploitation_constraint_mismatch_blocks": int(
            mediated_exploitation_summary.get(
                "constraint_mismatch_blocks", 0
            ) or 0
        ),
        "mediated_exploitation_duplicate_action_blocks": int(
            mediated_exploitation_summary.get(
                "duplicate_action_blocks", 0
            ) or 0
        ),
        "mediated_exploitation_selections": int(
            mediated_exploitation_summary.get("selections", 0) or 0
        ),
        "mediated_exploitation_preparation_actions": int(
            mediated_exploitation_summary.get(
                "preparation_actions", 0
            ) or 0
        ),
        "mediated_exploitation_preparation_starts": int(
            temporal_summary.get(
                "mediated_exploitation_preparation_starts", 0
            ) or 0
        ),
        "mediated_exploitation_progress_events": int(
            mediated_exploitation_summary.get("progress_events", 0) or 0
        ),
        "mediated_exploitation_nonprogress_events": int(
            mediated_exploitation_summary.get("nonprogress_events", 0) or 0
        ),
        "mediated_exploitation_terminal_events": int(
            mediated_exploitation_summary.get("terminal_events", 0) or 0
        ),
        "mediated_exploitation_refutations": int(
            mediated_exploitation_summary.get("refutations", 0) or 0
        ),
        "mediated_exploitation_unsafe_failures": int(
            mediated_exploitation_summary.get("unsafe_failures", 0) or 0
        ),
        "mediated_exploitation_restoration_predictions": int(
            mediated_exploitation_summary.get(
                "restoration_predictions", 0
            ) or 0
        ),
        "mediated_exploitation_restoration_selections": int(
            mediated_exploitation_summary.get(
                "restoration_selections", 0
            ) or 0
        ),
        "mediated_exploitation_restoration_steps_confirmed": int(
            mediated_exploitation_summary.get(
                "restoration_steps_confirmed", 0
            ) or 0
        ),
        "mediated_exploitation_restoration_targets_reached": int(
            mediated_exploitation_summary.get(
                "restoration_targets_reached", 0
            ) or 0
        ),
        "mediated_exploitation_restoration_failures": int(
            mediated_exploitation_summary.get(
                "restoration_failures", 0
            ) or 0
        ),
        "mediated_successor_states_captured": int(
            mediated_exploitation_summary.get(
                "successor_states_captured", 0
            ) or 0
        ),
        "mediated_successor_nonprogress_states_captured": int(
            mediated_exploitation_summary.get(
                "nonprogress_states_captured", 0
            ) or 0
        ),
        "mediated_successor_known_actions_compiled": int(
            mediated_exploitation_summary.get(
                "successor_known_actions_compiled", 0
            ) or 0
        ),
        "mediated_successor_exploration_actions": int(
            mediated_exploitation_summary.get(
                "successor_exploration_actions", 0
            ) or 0
        ),
        "mediated_successor_action_selections": int(
            mediated_exploitation_summary.get(
                "successor_action_selections", 0
            ) or 0
        ),
        "mediated_successor_progress_events": int(
            mediated_exploitation_summary.get(
                "successor_progress_events", 0
            ) or 0
        ),
        "mediated_successor_nonprogress_events": int(
            mediated_exploitation_summary.get(
                "successor_nonprogress_events", 0
            ) or 0
        ),
        "mediated_successor_terminal_events": int(
            mediated_exploitation_summary.get(
                "successor_terminal_events", 0
            ) or 0
        ),
        "mediated_successor_dead_ends": int(
            mediated_exploitation_summary.get("successor_dead_ends", 0) or 0
        ),
        "mediated_successor_cycle_blocks": int(
            mediated_exploitation_summary.get("cycle_blocks", 0) or 0
        ),
        "mediated_successor_obsolete_restoration_blocks": int(
            mediated_exploitation_summary.get(
                "obsolete_restoration_blocks", 0
            ) or 0
        ),
        "mediated_successor_maximum_chain_depth": int(
            mediated_exploitation_summary.get("maximum_chain_depth", 0) or 0
        ),
        "mediated_successor_structural_policy_classes": int(
            mediated_exploitation_summary.get(
                "structural_policy_classes", 0
            ) or 0
        ),
        "mediated_successor_structural_transfer_predictions": int(
            mediated_exploitation_summary.get(
                "structural_transfer_predictions", 0
            ) or 0
        ),
        "mediated_successor_structural_transfer_selections": int(
            mediated_exploitation_summary.get(
                "structural_transfer_selections", 0
            ) or 0
        ),
        "mediated_successor_structural_transfer_progress_events": int(
            mediated_exploitation_summary.get(
                "structural_transfer_progress_events", 0
            ) or 0
        ),
        "mediated_successor_structural_transfer_nonprogress_events": int(
            mediated_exploitation_summary.get(
                "structural_transfer_nonprogress_events", 0
            ) or 0
        ),
        "mediated_successor_structural_transfer_blocks": int(
            mediated_exploitation_summary.get(
                "structural_transfer_blocks", 0
            ) or 0
        ),
        "mediated_replication_requests_created": int(
            mediated_replication_summary.get("requests_created", 0) or 0
        ),
        "mediated_replication_pending_requests": int(
            mediated_replication_summary.get("pending_requests", 0) or 0
        ),
        "mediated_replication_cross_branch_activations": int(
            mediated_replication_summary.get(
                "cross_branch_activations", 0
            ) or 0
        ),
        "mediated_replication_same_branch_blocks": int(
            mediated_replication_summary.get("same_branch_blocks", 0) or 0
        ),
        "mediated_replication_exact_predictions": int(
            mediated_replication_summary.get(
                "exact_replication_predictions", 0
            ) or 0
        ),
        "mediated_replication_selections": int(
            mediated_replication_summary.get("selections", 0) or 0
        ),
        "mediated_replication_preparation_actions": int(
            mediated_replication_summary.get("preparation_actions", 0) or 0
        ),
        "mediated_replication_preparation_starts": int(
            temporal_summary.get(
                "mediated_replication_preparation_starts", 0
            ) or 0
        ),
        "mediated_replication_confirmations": int(
            mediated_replication_summary.get("confirmations", 0) or 0
        ),
        "mediated_replication_refutations": int(
            mediated_replication_summary.get("refutations", 0) or 0
        ),
        "mediated_replication_expirations": int(
            mediated_replication_summary.get("expirations", 0) or 0
        ),
        "effect_conditioned_goal_candidates_generated": int(
            controller_summary.get(
                "effect_conditioned_goal_candidates_generated",
                0,
            )
            or 0
        ),
        "effect_conditioned_subgoals_generated": int(
            effect_subgoal_summary.get("generated_total", 0) or 0
        ),
        "effect_conditioned_subgoal_links": int(
            effect_subgoal_summary.get("effect_links", 0) or 0
        ),
        "productive_effect_subgoal_links": int(
            effect_subgoal_summary.get("productive_effect_links", 0) or 0
        ),
        "effect_conditioned_subgoal_selections": int(
            effect_subgoal_summary.get("selections", 0) or 0
        ),
        "effect_conditioned_subgoal_guided_actions": int(
            effect_subgoal_summary.get("guided_actions", 0) or 0
        ),
        "effect_conditioned_subgoal_progress_events": int(
            effect_subgoal_summary.get("progress_events", 0) or 0
        ),
        "effect_conditioned_trigger_progress_events": int(
            effect_subgoal_summary.get("trigger_progress_events", 0) or 0
        ),
        "effect_conditioned_pursuit_progress_events": int(
            effect_subgoal_summary.get("pursuit_progress_events", 0) or 0
        ),
        "effect_conditioned_subgoal_completions": int(
            effect_subgoal_summary.get("completion_events", 0) or 0
        ),
        "effect_conditioned_subgoal_replayed_actions": int(
            effect_subgoal_summary.get("replayed_actions", 0) or 0
        ),
        "progress_supported_effect_conditioned_subgoals": int(
            dict(effect_subgoal_summary.get("statuses", {}) or {}).get(
                "progress_supported",
                0,
            )
            or 0
        ),
        "refuted_effect_conditioned_subgoals": int(
            dict(effect_subgoal_summary.get("statuses", {}) or {}).get(
                "refuted",
                0,
            )
            or 0
        ),
        "failed_effect_conditioned_subgoal_pursuits": int(
            effect_subgoal_summary.get("failed_pursuits", 0) or 0
        ),
        "censored_effect_conditioned_subgoal_pursuits": int(
            effect_subgoal_summary.get("censored_pursuits", 0) or 0
        ),
        "directional_effect_observations": int(
            directional_summary.get("observations", 0) or 0
        ),
        "directional_trigger_observations": int(
            directional_summary.get("trigger_observations", 0) or 0
        ),
        "directional_pursuit_observations": int(
            directional_summary.get("pursuit_observations", 0) or 0
        ),
        "directional_progress_events": int(
            directional_summary.get("progress_events", 0) or 0
        ),
        "directional_regression_events": int(
            directional_summary.get("regression_events", 0) or 0
        ),
        "directional_stall_events": int(
            directional_summary.get("stall_events", 0) or 0
        ),
        "directional_latent_modes": int(
            directional_summary.get("latent_modes", 0) or 0
        ),
        "directional_mode_action_models": int(
            directional_summary.get("mode_action_models", 0) or 0
        ),
        "directional_entity_anchored_action_models": int(
            directional_summary.get("entity_anchored_action_models", 0) or 0
        ),
        "directional_structural_transfer_classes": int(
            directional_summary.get("structural_transfer_classes", 0) or 0
        ),
        "directional_entity_alias_conflicts": int(
            directional_summary.get("entity_alias_conflicts", 0) or 0
        ),
        "directional_reversible_action_objectives": int(
            directional_summary.get("reversible_action_objectives", 0) or 0
        ),
        "directional_predictions": int(
            directional_summary.get("predictions", 0) or 0
        ),
        "directional_exact_mode_predictions": int(
            directional_summary.get("exact_mode_predictions", 0) or 0
        ),
        "directional_mode_contrast_predictions": int(
            directional_summary.get("mode_contrast_predictions", 0) or 0
        ),
        "directional_bridge_predictions": int(
            directional_summary.get("bridge_predictions", 0) or 0
        ),
        "directional_structural_transfer_predictions": int(
            directional_summary.get("structural_transfer_predictions", 0) or 0
        ),
        "directional_entity_contrast_predictions": int(
            directional_summary.get("entity_contrast_predictions", 0) or 0
        ),
        "directional_progressive_selections": int(
            directional_summary.get("progressive_selections", 0) or 0
        ),
        "directional_bridge_selections": int(
            directional_summary.get("bridge_selections", 0) or 0
        ),
        "directional_structural_transfer_selections": int(
            directional_summary.get("structural_transfer_selections", 0) or 0
        ),
        "directional_entity_contrast_selections": int(
            directional_summary.get("entity_contrast_selections", 0) or 0
        ),
        "directional_mode_contrast_selections": int(
            directional_summary.get("mode_contrast_selections", 0) or 0
        ),
        "directional_blocked_regressive_actions": int(
            directional_summary.get("blocked_regressive_actions", 0) or 0
        ),
        "directional_blocked_structural_regressions": int(
            directional_summary.get("blocked_structural_regressions", 0) or 0
        ),
        "persistent_pursuit_commitment_selections": int(
            persistent_summary.get("commitment_selections", 0) or 0
        ),
        "persistent_pursuit_resumed_commitments": int(
            persistent_summary.get("resumed_commitments", 0) or 0
        ),
        "persistent_pursuit_continuation_actions": int(
            persistent_summary.get("continuation_actions", 0) or 0
        ),
        "persistent_pursuit_directional_policy_actions": int(
            persistent_summary.get("directional_policy_actions", 0) or 0
        ),
        "persistent_pursuit_bridge_actions": int(
            persistent_summary.get("bridge_actions", 0) or 0
        ),
        "persistent_pursuit_entity_contrast_actions": int(
            persistent_summary.get("entity_contrast_actions", 0) or 0
        ),
        "persistent_pursuit_entity_binding_contrast_actions": int(
            persistent_summary.get(
                "entity_binding_contrast_actions",
                0,
            ) or 0
        ),
        "persistent_pursuit_mediated_effect_policy_actions": int(
            persistent_summary.get("mediated_effect_policy_actions", 0) or 0
        ),
        "persistent_pursuit_mediated_effect_contrast_actions": int(
            persistent_summary.get(
                "mediated_effect_contrast_actions",
                0,
            ) or 0
        ),
        "persistent_pursuit_mode_contrast_actions": int(
            persistent_summary.get("mode_contrast_actions", 0) or 0
        ),
        "persistent_pursuit_progress_events": int(
            persistent_summary.get("continuation_progress_events", 0) or 0
        ),
        "persistent_pursuit_repeated_progress_events": int(
            persistent_summary.get("repeated_progress_events", 0) or 0
        ),
        "persistent_pursuit_completed_objectives": int(
            persistent_summary.get("completed_objectives", 0) or 0
        ),
        "persistent_pursuit_attempt_budget_extensions": int(
            persistent_summary.get("attempt_budget_extensions", 0) or 0
        ),
        "persistent_pursuit_rollout_budget_extensions": int(
            persistent_summary.get("rollout_budget_extensions", 0) or 0
        ),
        "persistent_pursuit_credit_window_extensions": int(
            persistent_summary.get("credit_window_extensions", 0) or 0
        ),
        "persistent_pursuit_longest_continuation": int(
            persistent_summary.get("longest_continuation", 0) or 0
        ),
        "causal_option_dynamic_budget_extensions": int(
            causal_option_summary.get("dynamic_budget_extensions", 0) or 0
        ),
        "causal_option_budget_pruned_rollouts": int(
            causal_option_summary.get("budget_pruned_rollouts", 0) or 0
        ),
        "decision_sources": dict(decision_sources),
        "failure_causes": dict(failure_causes),
        "controller_errors": controller_errors,
        "attempts": attempts,
        "controller_summary": controller_summary if controller is not None else None,
    }


def _run_attempt(
    *,
    arm: str,
    game_id: str,
    reset_index: int,
    action_budget: int,
    env_dir: Path,
    env_factory: EnvFactory | None,
    policy: SharedLegacyProposalPolicy,
    controller: UnifiedCognitiveController | None,
    decision_sources: Counter[str],
    controller_errors: List[str],
) -> Dict[str, Any]:
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, env_dir)
        )
        frame = _reset_env(env)
        initial = snapshot_frame(frame)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _blocked_attempt(reset_index, f"environment_setup_error:{exc}")

    initial_level = int(initial.levels_completed)
    max_level = initial_level
    positive_level_delta = 0
    trace: List[Dict[str, Any]] = []
    failure = ""
    for step in range(action_budget):
        before = snapshot_frame(frame)
        if _is_terminal(before.game_state):
            break
        legal_actions = tuple(_valid_actions(env))
        proposal = policy.select(legal_actions)
        if proposal is None:
            failure = "no_legal_action"
            break
        selected = proposal
        source = "legacy_only"
        decision_payload: Dict[str, Any] = {}
        if controller is not None:
            try:
                decision = controller.select_action(
                    current_grid=before.grid,
                    available_actions=_available_action_names(legal_actions),
                    legacy_action=str(getattr(proposal, "name", "")),
                    legacy_action_data=dict(
                        getattr(proposal, "action_args", {}) or {}
                    ),
                    game_state=before.game_state,
                    levels_completed=before.levels_completed,
                )
                materialized = _materialize_decision(legal_actions, decision)
                if materialized is None:
                    raise ValueError(
                        f"unavailable_decision:{decision.action_name}:"
                        f"{decision.action_data}"
                    )
                selected = materialized
                source = decision.source
                decision_payload = decision.to_dict()
            except Exception as exc:
                controller_errors.append(f"select:{game_id}:{reset_index}:{step}:{exc}")
                selected = proposal
                source = "controller_error_legacy_fallback"
        decision_sources[source] += 1
        try:
            after_frame = _step_env_action(env, selected)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
        except Exception as exc:  # pragma: no cover - integration failure path
            failure = f"environment_step_error:{type(exc).__name__}"
            break
        if controller is not None:
            try:
                controller.observe_transition(
                    action=str(getattr(selected, "name", "")),
                    action_data=dict(getattr(selected, "action_args", {}) or {}),
                    grid_before=before.grid,
                    grid_after=after.grid,
                    available_actions=_available_action_names(legal_actions),
                    game_state_before=before.game_state,
                    game_state_after=after.game_state,
                    levels_completed_before=before.levels_completed,
                    levels_completed_after=after.levels_completed,
                )
            except Exception as exc:
                controller_errors.append(f"observe:{game_id}:{reset_index}:{step}:{exc}")
        step_delta = max(0, int(after.levels_completed) - int(before.levels_completed))
        positive_level_delta += step_delta
        max_level = max(max_level, int(after.levels_completed))
        trace.append({
            "step": step,
            "action": str(getattr(selected, "name", "")),
            "action_args": dict(getattr(selected, "action_args", {}) or {}),
            "decision_source": source,
            "decision": decision_payload,
            "levels_before": int(before.levels_completed),
            "levels_after": int(after.levels_completed),
            "game_state_after": str(after.game_state),
        })
        frame = after_frame
        if _is_terminal(after.game_state):
            break

    final = snapshot_frame(frame)
    win = str(final.game_state).upper() in WIN_STATES
    if not failure:
        failure = _classify_attempt_outcome(
            game_state=final.game_state,
            win=win,
            level_delta=positive_level_delta,
            actions_executed=len(trace),
            action_budget=action_budget,
        )
    return {
        "reset_index": reset_index,
        "status": "EXECUTED",
        "reset_visual_digest": _visual_digest(initial.grid),
        "actions_executed": len(trace),
        "levels_completed_delta": positive_level_delta,
        "max_level_reached": max_level,
        "final_game_state": str(final.game_state),
        "win": win,
        "failure_cause": failure,
        "trace": trace,
    }


def _materialize_decision(
    legal_actions: Sequence[Any],
    decision: Any,
) -> _ExecutableAction | None:
    name = str(decision.action_name)
    args = dict(decision.action_data or {})
    matches = [
        action
        for action in legal_actions
        if str(getattr(action, "name", "")) == name
    ]
    if not matches:
        return None
    for action in matches:
        if dict(getattr(action, "action_args", {}) or {}) == args:
            return _ExecutableAction(
                name=name,
                raw_action=getattr(action, "raw_action", action),
                action_args=args,
            )
    if args:
        base = matches[0]
        return _ExecutableAction(
            name=name,
            raw_action=getattr(base, "raw_action", base),
            action_args=args,
        )
    base = next(
        (
            action
            for action in matches
            if not dict(getattr(action, "action_args", {}) or {})
        ),
        matches[0],
    )
    return _ExecutableAction(
        name=name,
        raw_action=getattr(base, "raw_action", base),
        action_args=dict(getattr(base, "action_args", {}) or {}),
    )


def _summarize_benchmark(
    pairs: Sequence[Mapping[str, Any]],
    *,
    games: Sequence[str],
    seeds: Sequence[int],
    action_budget_per_reset: int,
    resets: int,
    causal_subgoal_induction_enabled: bool,
    causal_effect_credit_enabled: bool,
    causal_hierarchical_options_enabled: bool,
    effect_conditioned_downstream_subgoals_enabled: bool,
    state_conditioned_directional_control_enabled: bool,
    persistent_directional_pursuit_enabled: bool,
    entity_anchored_interventions_enabled: bool,
    active_entity_causal_binding_enabled: bool,
    mediated_entity_effect_induction_enabled: bool,
    online_mediated_anti_unification_enabled: bool,
    active_mediated_discrimination_enabled: bool,
    active_mode_restoration_enabled: bool,
    terminal_mediated_exploitation_enabled: bool,
    successor_policy_chaining_enabled: bool,
    successor_structural_transfer_enabled: bool,
    active_mediated_replication_enabled: bool,
    horizon_stable_learning_epochs_enabled: bool,
    online_horizon_learning_arbiter_enabled: bool,
) -> Dict[str, Any]:
    legacy = _aggregate_arm(pairs, "legacy_only")
    unified = _aggregate_arm(pairs, "unified")
    reset_gate = all(bool(pair["same_fresh_reset_states"]) for pair in pairs)
    protocol_gate = bool(
        pairs
        and reset_gate
        and all(bool(pair["same_action_budget"]) for pair in pairs)
        and all(bool(pair["same_reset_count"]) for pair in pairs)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "paired_held_out_online_learning_ab",
        "held_out_games": list(games),
        "seeds": list(seeds),
        "action_budget_per_reset": action_budget_per_reset,
        "resets_per_game_seed_arm": resets,
        "paired_protocol": {
            "fresh_environment_per_reset": True,
            "same_games": True,
            "same_seeds": True,
            "same_reset_count": all(
                bool(pair["same_reset_count"]) for pair in pairs
            ),
            "same_action_budget": all(
                bool(pair["same_action_budget"]) for pair in pairs
            ),
            "same_reset_visual_states": reset_gate,
            "shared_legacy_proposal_policy": True,
            "online_learning_within_arm_only": True,
            "evaluation_outcomes_used_for_training_or_tuning": False,
            "causal_subgoal_induction_enabled_in_unified": bool(
                causal_subgoal_induction_enabled
            ),
            "causal_effect_credit_enabled_in_unified": bool(
                causal_effect_credit_enabled
            ),
            "causal_hierarchical_options_enabled_in_unified": bool(
                causal_hierarchical_options_enabled
            ),
            "effect_conditioned_downstream_subgoals_enabled_in_unified": bool(
                effect_conditioned_downstream_subgoals_enabled
            ),
            "state_conditioned_directional_control_enabled_in_unified": bool(
                state_conditioned_directional_control_enabled
            ),
            "persistent_directional_pursuit_enabled_in_unified": bool(
                persistent_directional_pursuit_enabled
            ),
            "entity_anchored_interventions_enabled_in_unified": bool(
                entity_anchored_interventions_enabled
            ),
            "active_entity_causal_binding_enabled_in_unified": bool(
                active_entity_causal_binding_enabled
            ),
            "mediated_entity_effect_induction_enabled_in_unified": bool(
                mediated_entity_effect_induction_enabled
            ),
            "online_mediated_anti_unification_enabled_in_unified": bool(
                online_mediated_anti_unification_enabled
            ),
            "active_mediated_discrimination_enabled_in_unified": bool(
                active_mediated_discrimination_enabled
            ),
            "active_mode_restoration_enabled_in_unified": bool(
                active_mode_restoration_enabled
            ),
            "terminal_mediated_exploitation_enabled_in_unified": bool(
                terminal_mediated_exploitation_enabled
            ),
            "successor_policy_chaining_enabled_in_unified": bool(
                successor_policy_chaining_enabled
            ),
            "successor_structural_transfer_enabled_in_unified": bool(
                successor_structural_transfer_enabled
            ),
            "active_mediated_replication_enabled_in_unified": bool(
                active_mediated_replication_enabled
            ),
            "horizon_stable_learning_epochs_enabled_in_unified": bool(
                horizon_stable_learning_epochs_enabled
            ),
            "online_horizon_learning_arbiter_enabled_in_unified": bool(
                online_horizon_learning_arbiter_enabled
            ),
            "protocol_gate_passed": protocol_gate,
        },
        "baseline_definition": (
            "both arms use the same deterministic balanced legal proposal "
            "algorithm and seed; legacy_only executes its proposal directly "
            "and unified receives its arm's proposal through legacy_action"
        ),
        "metrics": {
            "legacy_only": legacy,
            "unified": unified,
            "delta_unified_minus_legacy": {
                "levels_completed": (
                    unified["levels_completed"] - legacy["levels_completed"]
                ),
                "wins": unified["wins"] - legacy["wins"],
                "experiment_actions": (
                    unified["experiment_actions"] - legacy["experiment_actions"]
                ),
                "experiment_cost_rate": round(
                    unified["experiment_cost_rate"]
                    - legacy["experiment_cost_rate"],
                    6,
                ),
            },
        },
        "failure_causes": {
            "legacy_only": legacy["failure_causes"],
            "unified": unified["failure_causes"],
        },
        "arc_progress_observed": bool(
            unified["levels_completed"] > legacy["levels_completed"]
            or unified["wins"] > legacy["wins"]
        ),
        "pairs": list(pairs),
    }


def _aggregate_arm(
    pairs: Sequence[Mapping[str, Any]],
    arm: str,
) -> Dict[str, Any]:
    rows = [dict(pair[arm]) for pair in pairs]
    actions = sum(int(row["actions_executed"]) for row in rows)
    experiments = sum(int(row["experiment_actions"]) for row in rows)
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(dict(row.get("failure_causes", {}) or {}))
    reset_attempts = sum(int(row["resets_executed"]) for row in rows)
    return {
        "game_seed_runs": len(rows),
        "episodes": reset_attempts,
        "reset_attempts": reset_attempts,
        "actions_executed": actions,
        "operator_plan_actions": sum(
            int(row["operator_plan_actions"]) for row in rows
        ),
        "operator_plan_actions_since_objective_progress": sum(
            int(row["operator_plan_actions_since_objective_progress"])
            for row in rows
        ),
        "operator_plan_streak_peak": max(
            (int(row["operator_plan_streak_peak"]) for row in rows),
            default=0,
        ),
        "operator_plan_budget_blocks": sum(
            int(row["operator_plan_budget_blocks"]) for row in rows
        ),
        "operator_plan_progress_resets": sum(
            int(row["operator_plan_progress_resets"]) for row in rows
        ),
        "horizon_arbiter_evaluations": sum(
            int(row["horizon_arbiter_evaluations"]) for row in rows
        ),
        "horizon_arbiter_reservations": sum(
            int(row["horizon_arbiter_reservations"]) for row in rows
        ),
        "horizon_arbiter_releases": sum(
            int(row["horizon_arbiter_releases"]) for row in rows
        ),
        "horizon_arbiter_causal_uncertainty_reservations": sum(
            int(row["horizon_arbiter_causal_uncertainty_reservations"])
            for row in rows
        ),
        "horizon_arbiter_terminal_test_reservations": sum(
            int(row["horizon_arbiter_terminal_test_reservations"])
            for row in rows
        ),
        "horizon_arbiter_priority_peak": max(
            (int(row["horizon_arbiter_priority_peak"]) for row in rows),
            default=0,
        ),
        "levels_completed": sum(
            int(row["levels_completed_delta"]) for row in rows
        ),
        "wins": sum(int(row["wins"]) for row in rows),
        "experiment_actions": experiments,
        "experiment_cost_rate": round(experiments / actions, 6)
        if actions
        else 0.0,
        "promoted_option_actions": sum(
            int(row["promoted_option_actions"]) for row in rows
        ),
        "option_preparation_actions": sum(
            int(row["option_preparation_actions"]) for row in rows
        ),
        "terminal_objective_probe_actions": sum(
            int(row["terminal_objective_probe_actions"]) for row in rows
        ),
        "terminal_objective_grounded_actions": sum(
            int(row["terminal_objective_grounded_actions"]) for row in rows
        ),
        "terminal_objective_discriminator_actions": sum(
            int(row["terminal_objective_discriminator_actions"]) for row in rows
        ),
        "terminal_objective_ablation_actions": sum(
            int(row["terminal_objective_ablation_actions"]) for row in rows
        ),
        "temporal_subgoal_probe_actions": sum(
            int(row["temporal_subgoal_probe_actions"]) for row in rows
        ),
        "temporal_subgoal_option_actions": sum(
            int(row["temporal_subgoal_option_actions"]) for row in rows
        ),
        "generated_goal_hypotheses": sum(
            int(row["generated_goal_hypotheses"]) for row in rows
        ),
        "objective_distance_reductions": sum(
            int(row["objective_distance_reductions"]) for row in rows
        ),
        "objective_nonterminal_completions": sum(
            int(row["objective_nonterminal_completions"]) for row in rows
        ),
        "objective_ambiguous_terminal_events": sum(
            int(row["objective_ambiguous_terminal_events"]) for row in rows
        ),
        "terminal_supported_objectives": sum(
            int(row["terminal_supported_objectives"]) for row in rows
        ),
        "refuted_objectives": sum(
            int(row["refuted_objectives"]) for row in rows
        ),
        "unsafe_goal_plan_failures": sum(
            int(row["unsafe_goal_plan_failures"]) for row in rows
        ),
        "temporal_plans_generated": sum(
            int(row["temporal_plans_generated"]) for row in rows
        ),
        "temporal_plan_starts": sum(
            int(row["temporal_plan_starts"]) for row in rows
        ),
        "temporal_plan_actions": sum(
            int(row["temporal_plan_actions"]) for row in rows
        ),
        "temporal_progress_events": sum(
            int(row["temporal_progress_events"]) for row in rows
        ),
        "temporal_step_completions": sum(
            int(row["temporal_step_completions"]) for row in rows
        ),
        "temporal_local_completions": sum(
            int(row["temporal_local_completions"]) for row in rows
        ),
        "temporal_nonterminal_completions": sum(
            int(row["temporal_nonterminal_completions"]) for row in rows
        ),
        "temporal_plan_stalls": sum(
            int(row["temporal_plan_stalls"]) for row in rows
        ),
        "temporal_plan_abandonments": sum(
            int(row["temporal_plan_abandonments"]) for row in rows
        ),
        "temporal_unsafe_failures": sum(
            int(row["temporal_unsafe_failures"]) for row in rows
        ),
        "temporal_terminal_bypasses": sum(
            int(row["temporal_terminal_bypasses"]) for row in rows
        ),
        "terminal_supported_temporal_plans": sum(
            int(row["terminal_supported_temporal_plans"]) for row in rows
        ),
        "refuted_temporal_plans": sum(
            int(row["refuted_temporal_plans"]) for row in rows
        ),
        "causal_dependency_plans": sum(
            int(row["causal_dependency_plans"]) for row in rows
        ),
        "causal_dependency_plan_starts": sum(
            int(row["causal_dependency_plan_starts"]) for row in rows
        ),
        "causal_dependency_plan_actions": sum(
            int(row["causal_dependency_plan_actions"]) for row in rows
        ),
        "causal_dependency_progress_events": sum(
            int(row["causal_dependency_progress_events"]) for row in rows
        ),
        "causal_dependency_step_completions": sum(
            int(row["causal_dependency_step_completions"]) for row in rows
        ),
        "causal_edges_generated": sum(
            int(row["causal_edges_generated"]) for row in rows
        ),
        "causal_blocked_target_events": sum(
            int(row["causal_blocked_target_events"]) for row in rows
        ),
        "causal_edge_trials": sum(
            int(row["causal_edge_trials"]) for row in rows
        ),
        "causal_edge_actions": sum(
            int(row["causal_edge_actions"]) for row in rows
        ),
        "causal_edge_source_progress_events": sum(
            int(row["causal_edge_source_progress_events"]) for row in rows
        ),
        "causal_edge_support_events": sum(
            int(row["causal_edge_support_events"]) for row in rows
        ),
        "causal_edge_contradictions": sum(
            int(row["causal_edge_contradictions"]) for row in rows
        ),
        "causal_availability_successes": sum(
            int(row["causal_availability_successes"]) for row in rows
        ),
        "causal_availability_failures": sum(
            int(row["causal_availability_failures"]) for row in rows
        ),
        "causal_cochange_supports": sum(
            int(row["causal_cochange_supports"]) for row in rows
        ),
        "confirmed_causal_edges": sum(
            int(row["confirmed_causal_edges"]) for row in rows
        ),
        "refuted_causal_edges": sum(
            int(row["refuted_causal_edges"]) for row in rows
        ),
        "causal_edge_plan_failures": sum(
            int(row["causal_edge_plan_failures"]) for row in rows
        ),
        "causal_edge_unsafe_failures": sum(
            int(row["causal_edge_unsafe_failures"]) for row in rows
        ),
        "causal_effect_observations": sum(
            int(row["causal_effect_observations"]) for row in rows
        ),
        "causal_effect_guided_actions": sum(
            int(row["causal_effect_guided_actions"]) for row in rows
        ),
        "causal_productive_effect_signatures": sum(
            int(row["causal_productive_effect_signatures"]) for row in rows
        ),
        "causal_productive_intervention_signatures": sum(
            int(row["causal_productive_intervention_signatures"])
            for row in rows
        ),
        "causal_delayed_credit_events": sum(
            int(row["causal_delayed_credit_events"]) for row in rows
        ),
        "causal_expired_credit_windows": sum(
            int(row["causal_expired_credit_windows"]) for row in rows
        ),
        "causal_cross_branch_confirmations": sum(
            int(row["causal_cross_branch_confirmations"]) for row in rows
        ),
        "causal_reserved_confirmation_starts": sum(
            int(row["causal_reserved_confirmation_starts"])
            for row in rows
        ),
        "causal_options_compiled": sum(
            int(row["causal_options_compiled"]) for row in rows
        ),
        "causal_option_opening_events": sum(
            int(row["causal_option_opening_events"]) for row in rows
        ),
        "causal_option_rollouts": sum(
            int(row["causal_option_rollouts"]) for row in rows
        ),
        "causal_option_downstream_actions": sum(
            int(row["causal_option_downstream_actions"]) for row in rows
        ),
        "causal_option_downstream_effects": sum(
            int(row["causal_option_downstream_effects"]) for row in rows
        ),
        "causal_option_downstream_progress_events": sum(
            int(row["causal_option_downstream_progress_events"])
            for row in rows
        ),
        "causal_option_target_completions": sum(
            int(row["causal_option_target_completions"]) for row in rows
        ),
        "causal_option_nonterminal_rollouts": sum(
            int(row["causal_option_nonterminal_rollouts"]) for row in rows
        ),
        "causal_option_unsafe_rollouts": sum(
            int(row["causal_option_unsafe_rollouts"]) for row in rows
        ),
        "causal_option_terminal_credited_events": sum(
            int(row["causal_option_terminal_credited_events"])
            for row in rows
        ),
        "terminal_supported_causal_options": sum(
            int(row["terminal_supported_causal_options"]) for row in rows
        ),
        "terminal_refuted_causal_options": sum(
            int(row["terminal_refuted_causal_options"]) for row in rows
        ),
        "causal_option_censored_openings": sum(
            int(row["causal_option_censored_openings"]) for row in rows
        ),
        "entity_anchored_candidate_signatures": sum(
            int(row["entity_anchored_candidate_signatures"]) for row in rows
        ),
        "entity_anchored_transfer_signatures": sum(
            int(row["entity_anchored_transfer_signatures"]) for row in rows
        ),
        "entity_anchored_selections": sum(
            int(row["entity_anchored_selections"]) for row in rows
        ),
        "entity_binding_observations": sum(
            int(row["entity_binding_observations"]) for row in rows
        ),
        "entity_binding_matched_entities": sum(
            int(row["entity_binding_matched_entities"]) for row in rows
        ),
        "entity_binding_transformed_entities": sum(
            int(row["entity_binding_transformed_entities"]) for row in rows
        ),
        "entity_binding_moved_entities": sum(
            int(row["entity_binding_moved_entities"]) for row in rows
        ),
        "entity_binding_removed_entities": sum(
            int(row["entity_binding_removed_entities"]) for row in rows
        ),
        "entity_binding_ambiguous_entities": sum(
            int(row["entity_binding_ambiguous_entities"]) for row in rows
        ),
        "entity_binding_tracks_created": sum(
            int(row["entity_binding_tracks_created"]) for row in rows
        ),
        "entity_binding_tracks_reused": sum(
            int(row["entity_binding_tracks_reused"]) for row in rows
        ),
        "entity_binding_models": sum(
            int(row["entity_binding_models"]) for row in rows
        ),
        "entity_binding_carrier_progress_events": sum(
            int(row["entity_binding_carrier_progress_events"])
            for row in rows
        ),
        "entity_binding_carrier_regression_events": sum(
            int(row["entity_binding_carrier_regression_events"])
            for row in rows
        ),
        "entity_binding_noncarrier_progress_events": sum(
            int(row["entity_binding_noncarrier_progress_events"])
            for row in rows
        ),
        "entity_binding_conflicts": sum(
            int(row["entity_binding_conflicts"]) for row in rows
        ),
        "entity_binding_predictions": sum(
            int(row["entity_binding_predictions"]) for row in rows
        ),
        "entity_binding_controlled_contrast_predictions": sum(
            int(row["entity_binding_controlled_contrast_predictions"])
            for row in rows
        ),
        "entity_binding_controlled_contrast_selections": sum(
            int(row["entity_binding_controlled_contrast_selections"])
            for row in rows
        ),
        "entity_binding_progressive_carrier_selections": sum(
            int(row["entity_binding_progressive_carrier_selections"])
            for row in rows
        ),
        "entity_binding_blocked_misbound_actions": sum(
            int(row["entity_binding_blocked_misbound_actions"])
            for row in rows
        ),
        "mediated_effect_observations": sum(
            int(row["mediated_effect_observations"]) for row in rows
        ),
        "mediated_effect_scene_correspondences": sum(
            int(row["mediated_effect_scene_correspondences"])
            for row in rows
        ),
        "mediated_effect_changed_entities": sum(
            int(row["mediated_effect_changed_entities"]) for row in rows
        ),
        "mediated_effect_moved_entities": sum(
            int(row["mediated_effect_moved_entities"]) for row in rows
        ),
        "mediated_effect_transformed_entities": sum(
            int(row["mediated_effect_transformed_entities"]) for row in rows
        ),
        "mediated_effect_removed_entities": sum(
            int(row["mediated_effect_removed_entities"]) for row in rows
        ),
        "mediated_effect_appeared_entities": sum(
            int(row["mediated_effect_appeared_entities"]) for row in rows
        ),
        "mediated_effect_ambiguous_entities": sum(
            int(row["mediated_effect_ambiguous_entities"]) for row in rows
        ),
        "mediated_effect_tracks_created": sum(
            int(row["mediated_effect_tracks_created"]) for row in rows
        ),
        "mediated_effect_tracks_reused": sum(
            int(row["mediated_effect_tracks_reused"]) for row in rows
        ),
        "mediated_effect_progress_with_indirect_candidates": sum(
            int(row["mediated_effect_progress_with_indirect_candidates"])
            for row in rows
        ),
        "mediated_effect_ambiguous_progress_candidate_sets": sum(
            int(row["mediated_effect_ambiguous_progress_candidate_sets"])
            for row in rows
        ),
        "mediated_effect_no_candidate_progress_events": sum(
            int(row["mediated_effect_no_candidate_progress_events"])
            for row in rows
        ),
        "mediated_effect_direct_target_progress_events": sum(
            int(row["mediated_effect_direct_target_progress_events"])
            for row in rows
        ),
        "mediated_effect_models": sum(
            int(row["mediated_effect_models"]) for row in rows
        ),
        "mediated_effect_supported_hyperedges": sum(
            int(row["mediated_effect_supported_hyperedges"]) for row in rows
        ),
        "mediated_abstraction_hypotheses": sum(
            int(row["mediated_abstraction_hypotheses"]) for row in rows
        ),
        "mediated_abstraction_supported_hyperedges": sum(
            int(row["mediated_abstraction_supported_hyperedges"])
            for row in rows
        ),
        "mediated_abstraction_control_contexts": sum(
            int(row["mediated_abstraction_control_contexts"])
            for row in rows
        ),
        "mediated_abstraction_regression_contexts": sum(
            int(row["mediated_abstraction_regression_contexts"])
            for row in rows
        ),
        "mediated_effect_predictions": sum(
            int(row["mediated_effect_predictions"]) for row in rows
        ),
        "mediated_effect_supported_predictions": sum(
            int(row["mediated_effect_supported_predictions"])
            for row in rows
        ),
        "mediated_effect_controlled_contrast_predictions": sum(
            int(row["mediated_effect_controlled_contrast_predictions"])
            for row in rows
        ),
        "mediated_effect_supported_selections": sum(
            int(row["mediated_effect_supported_selections"])
            for row in rows
        ),
        "mediated_effect_controlled_contrast_selections": sum(
            int(row["mediated_effect_controlled_contrast_selections"])
            for row in rows
        ),
        "mediated_effect_blocked_contradicted_actions": sum(
            int(row["mediated_effect_blocked_contradicted_actions"])
            for row in rows
        ),
        "mediated_discrimination_requests_created": sum(
            int(row["mediated_discrimination_requests_created"])
            for row in rows
        ),
        "mediated_discrimination_pending_requests": sum(
            int(row["mediated_discrimination_pending_requests"])
            for row in rows
        ),
        "mediated_discrimination_active_requests": sum(
            int(row["mediated_discrimination_active_requests"])
            for row in rows
        ),
        "mediated_discrimination_cross_branch_activations": sum(
            int(row["mediated_discrimination_cross_branch_activations"])
            for row in rows
        ),
        "mediated_discrimination_predictions": sum(
            int(row["mediated_discrimination_predictions"])
            for row in rows
        ),
        "mediated_discrimination_mode_mismatch_blocks": sum(
            int(row["mediated_discrimination_mode_mismatch_blocks"])
            for row in rows
        ),
        "mediated_discrimination_no_single_feature_blocks": sum(
            int(row["mediated_discrimination_no_single_feature_blocks"])
            for row in rows
        ),
        "mediated_discrimination_selections": sum(
            int(row["mediated_discrimination_selections"])
            for row in rows
        ),
        "mediated_discrimination_preparation_actions": sum(
            int(row["mediated_discrimination_preparation_actions"])
            for row in rows
        ),
        "mediated_discrimination_preparation_starts": sum(
            int(row["mediated_discrimination_preparation_starts"])
            for row in rows
        ),
        "mediated_discrimination_feature_requirements": sum(
            int(row["mediated_discrimination_feature_requirements"])
            for row in rows
        ),
        "mediated_discrimination_feature_eliminations": sum(
            int(row["mediated_discrimination_feature_eliminations"])
            for row in rows
        ),
        "mediated_discrimination_inconclusive_attempts": sum(
            int(row["mediated_discrimination_inconclusive_attempts"])
            for row in rows
        ),
        "mediated_discrimination_expirations": sum(
            int(row["mediated_discrimination_expirations"])
            for row in rows
        ),
        "mediated_restoration_actions": sum(
            int(row["mediated_restoration_actions"]) for row in rows
        ),
        "mediated_restoration_predictions": sum(
            int(row["mediated_restoration_predictions"]) for row in rows
        ),
        "mediated_restoration_selections": sum(
            int(row["mediated_restoration_selections"]) for row in rows
        ),
        "mediated_restoration_steps_confirmed": sum(
            int(row["mediated_restoration_steps_confirmed"])
            for row in rows
        ),
        "mediated_restoration_targets_reached": sum(
            int(row["mediated_restoration_targets_reached"])
            for row in rows
        ),
        "mediated_restoration_failures": sum(
            int(row["mediated_restoration_failures"]) for row in rows
        ),
        "mediated_restoration_unavailable_contexts": sum(
            int(row["mediated_restoration_unavailable_contexts"])
            for row in rows
        ),
        "mediated_exploitation_actions": sum(
            int(row["mediated_exploitation_actions"]) for row in rows
        ),
        "mediated_exploitation_restoration_actions": sum(
            int(row["mediated_exploitation_restoration_actions"])
            for row in rows
        ),
        "mediated_exploitation_policies_compiled": sum(
            int(row["mediated_exploitation_policies_compiled"])
            for row in rows
        ),
        "mediated_exploitation_policy_revisions": sum(
            int(row["mediated_exploitation_policy_revisions"])
            for row in rows
        ),
        "mediated_exploitation_activations": sum(
            int(row["mediated_exploitation_activations"])
            for row in rows
        ),
        "mediated_exploitation_predictions": sum(
            int(row["mediated_exploitation_predictions"])
            for row in rows
        ),
        "mediated_exploitation_mode_mismatch_blocks": sum(
            int(row["mediated_exploitation_mode_mismatch_blocks"])
            for row in rows
        ),
        "mediated_exploitation_constraint_mismatch_blocks": sum(
            int(row["mediated_exploitation_constraint_mismatch_blocks"])
            for row in rows
        ),
        "mediated_exploitation_duplicate_action_blocks": sum(
            int(row["mediated_exploitation_duplicate_action_blocks"])
            for row in rows
        ),
        "mediated_exploitation_selections": sum(
            int(row["mediated_exploitation_selections"])
            for row in rows
        ),
        "mediated_exploitation_preparation_actions": sum(
            int(row["mediated_exploitation_preparation_actions"])
            for row in rows
        ),
        "mediated_exploitation_preparation_starts": sum(
            int(row["mediated_exploitation_preparation_starts"])
            for row in rows
        ),
        "mediated_exploitation_progress_events": sum(
            int(row["mediated_exploitation_progress_events"])
            for row in rows
        ),
        "mediated_exploitation_nonprogress_events": sum(
            int(row["mediated_exploitation_nonprogress_events"])
            for row in rows
        ),
        "mediated_exploitation_terminal_events": sum(
            int(row["mediated_exploitation_terminal_events"])
            for row in rows
        ),
        "mediated_exploitation_refutations": sum(
            int(row["mediated_exploitation_refutations"])
            for row in rows
        ),
        "mediated_exploitation_unsafe_failures": sum(
            int(row["mediated_exploitation_unsafe_failures"])
            for row in rows
        ),
        "mediated_exploitation_restoration_predictions": sum(
            int(row["mediated_exploitation_restoration_predictions"])
            for row in rows
        ),
        "mediated_exploitation_restoration_selections": sum(
            int(row["mediated_exploitation_restoration_selections"])
            for row in rows
        ),
        "mediated_exploitation_restoration_steps_confirmed": sum(
            int(row[
                "mediated_exploitation_restoration_steps_confirmed"
            ])
            for row in rows
        ),
        "mediated_exploitation_restoration_targets_reached": sum(
            int(row[
                "mediated_exploitation_restoration_targets_reached"
            ])
            for row in rows
        ),
        "mediated_exploitation_restoration_failures": sum(
            int(row["mediated_exploitation_restoration_failures"])
            for row in rows
        ),
        "mediated_successor_states_captured": sum(
            int(row["mediated_successor_states_captured"])
            for row in rows
        ),
        "mediated_successor_nonprogress_states_captured": sum(
            int(row["mediated_successor_nonprogress_states_captured"])
            for row in rows
        ),
        "mediated_successor_known_actions_compiled": sum(
            int(row["mediated_successor_known_actions_compiled"])
            for row in rows
        ),
        "mediated_successor_exploration_actions": sum(
            int(row["mediated_successor_exploration_actions"])
            for row in rows
        ),
        "mediated_successor_action_selections": sum(
            int(row["mediated_successor_action_selections"])
            for row in rows
        ),
        "mediated_successor_progress_events": sum(
            int(row["mediated_successor_progress_events"])
            for row in rows
        ),
        "mediated_successor_nonprogress_events": sum(
            int(row["mediated_successor_nonprogress_events"])
            for row in rows
        ),
        "mediated_successor_terminal_events": sum(
            int(row["mediated_successor_terminal_events"])
            for row in rows
        ),
        "mediated_successor_dead_ends": sum(
            int(row["mediated_successor_dead_ends"])
            for row in rows
        ),
        "mediated_successor_cycle_blocks": sum(
            int(row["mediated_successor_cycle_blocks"])
            for row in rows
        ),
        "mediated_successor_obsolete_restoration_blocks": sum(
            int(row["mediated_successor_obsolete_restoration_blocks"])
            for row in rows
        ),
        "mediated_successor_maximum_chain_depth": max(
            (
                int(row["mediated_successor_maximum_chain_depth"])
                for row in rows
            ),
            default=0,
        ),
        "mediated_successor_structural_policy_classes": sum(
            int(row["mediated_successor_structural_policy_classes"])
            for row in rows
        ),
        "mediated_successor_structural_transfer_predictions": sum(
            int(row["mediated_successor_structural_transfer_predictions"])
            for row in rows
        ),
        "mediated_successor_structural_transfer_selections": sum(
            int(row["mediated_successor_structural_transfer_selections"])
            for row in rows
        ),
        "mediated_successor_structural_transfer_progress_events": sum(
            int(
                row[
                    "mediated_successor_structural_transfer_progress_events"
                ]
            )
            for row in rows
        ),
        "mediated_successor_structural_transfer_nonprogress_events": sum(
            int(
                row[
                    "mediated_successor_structural_transfer_nonprogress_events"
                ]
            )
            for row in rows
        ),
        "mediated_successor_structural_transfer_blocks": sum(
            int(row["mediated_successor_structural_transfer_blocks"])
            for row in rows
        ),
        "mediated_replication_requests_created": sum(
            int(row["mediated_replication_requests_created"])
            for row in rows
        ),
        "mediated_replication_pending_requests": sum(
            int(row["mediated_replication_pending_requests"])
            for row in rows
        ),
        "mediated_replication_cross_branch_activations": sum(
            int(row["mediated_replication_cross_branch_activations"])
            for row in rows
        ),
        "mediated_replication_same_branch_blocks": sum(
            int(row["mediated_replication_same_branch_blocks"])
            for row in rows
        ),
        "mediated_replication_exact_predictions": sum(
            int(row["mediated_replication_exact_predictions"])
            for row in rows
        ),
        "mediated_replication_selections": sum(
            int(row["mediated_replication_selections"])
            for row in rows
        ),
        "mediated_replication_preparation_actions": sum(
            int(row["mediated_replication_preparation_actions"])
            for row in rows
        ),
        "mediated_replication_preparation_starts": sum(
            int(row["mediated_replication_preparation_starts"])
            for row in rows
        ),
        "mediated_replication_confirmations": sum(
            int(row["mediated_replication_confirmations"])
            for row in rows
        ),
        "mediated_replication_refutations": sum(
            int(row["mediated_replication_refutations"])
            for row in rows
        ),
        "mediated_replication_expirations": sum(
            int(row["mediated_replication_expirations"])
            for row in rows
        ),
        "effect_conditioned_goal_candidates_generated": sum(
            int(row["effect_conditioned_goal_candidates_generated"])
            for row in rows
        ),
        "effect_conditioned_subgoals_generated": sum(
            int(row["effect_conditioned_subgoals_generated"]) for row in rows
        ),
        "effect_conditioned_subgoal_links": sum(
            int(row["effect_conditioned_subgoal_links"]) for row in rows
        ),
        "productive_effect_subgoal_links": sum(
            int(row["productive_effect_subgoal_links"]) for row in rows
        ),
        "effect_conditioned_subgoal_selections": sum(
            int(row["effect_conditioned_subgoal_selections"]) for row in rows
        ),
        "effect_conditioned_subgoal_guided_actions": sum(
            int(row["effect_conditioned_subgoal_guided_actions"])
            for row in rows
        ),
        "effect_conditioned_subgoal_progress_events": sum(
            int(row["effect_conditioned_subgoal_progress_events"])
            for row in rows
        ),
        "effect_conditioned_trigger_progress_events": sum(
            int(row["effect_conditioned_trigger_progress_events"])
            for row in rows
        ),
        "effect_conditioned_pursuit_progress_events": sum(
            int(row["effect_conditioned_pursuit_progress_events"])
            for row in rows
        ),
        "effect_conditioned_subgoal_completions": sum(
            int(row["effect_conditioned_subgoal_completions"])
            for row in rows
        ),
        "effect_conditioned_subgoal_replayed_actions": sum(
            int(row["effect_conditioned_subgoal_replayed_actions"])
            for row in rows
        ),
        "progress_supported_effect_conditioned_subgoals": sum(
            int(row["progress_supported_effect_conditioned_subgoals"])
            for row in rows
        ),
        "refuted_effect_conditioned_subgoals": sum(
            int(row["refuted_effect_conditioned_subgoals"])
            for row in rows
        ),
        "failed_effect_conditioned_subgoal_pursuits": sum(
            int(row["failed_effect_conditioned_subgoal_pursuits"])
            for row in rows
        ),
        "censored_effect_conditioned_subgoal_pursuits": sum(
            int(row["censored_effect_conditioned_subgoal_pursuits"])
            for row in rows
        ),
        "directional_effect_observations": sum(
            int(row["directional_effect_observations"]) for row in rows
        ),
        "directional_trigger_observations": sum(
            int(row["directional_trigger_observations"]) for row in rows
        ),
        "directional_pursuit_observations": sum(
            int(row["directional_pursuit_observations"]) for row in rows
        ),
        "directional_progress_events": sum(
            int(row["directional_progress_events"]) for row in rows
        ),
        "directional_regression_events": sum(
            int(row["directional_regression_events"]) for row in rows
        ),
        "directional_stall_events": sum(
            int(row["directional_stall_events"]) for row in rows
        ),
        "directional_latent_modes": sum(
            int(row["directional_latent_modes"]) for row in rows
        ),
        "directional_mode_action_models": sum(
            int(row["directional_mode_action_models"]) for row in rows
        ),
        "directional_entity_anchored_action_models": sum(
            int(row["directional_entity_anchored_action_models"])
            for row in rows
        ),
        "directional_structural_transfer_classes": sum(
            int(row["directional_structural_transfer_classes"])
            for row in rows
        ),
        "directional_entity_alias_conflicts": sum(
            int(row["directional_entity_alias_conflicts"]) for row in rows
        ),
        "directional_reversible_action_objectives": sum(
            int(row["directional_reversible_action_objectives"])
            for row in rows
        ),
        "directional_predictions": sum(
            int(row["directional_predictions"]) for row in rows
        ),
        "directional_exact_mode_predictions": sum(
            int(row["directional_exact_mode_predictions"]) for row in rows
        ),
        "directional_mode_contrast_predictions": sum(
            int(row["directional_mode_contrast_predictions"]) for row in rows
        ),
        "directional_bridge_predictions": sum(
            int(row["directional_bridge_predictions"]) for row in rows
        ),
        "directional_structural_transfer_predictions": sum(
            int(row["directional_structural_transfer_predictions"])
            for row in rows
        ),
        "directional_entity_contrast_predictions": sum(
            int(row["directional_entity_contrast_predictions"])
            for row in rows
        ),
        "directional_progressive_selections": sum(
            int(row["directional_progressive_selections"]) for row in rows
        ),
        "directional_bridge_selections": sum(
            int(row["directional_bridge_selections"]) for row in rows
        ),
        "directional_structural_transfer_selections": sum(
            int(row["directional_structural_transfer_selections"])
            for row in rows
        ),
        "directional_entity_contrast_selections": sum(
            int(row["directional_entity_contrast_selections"])
            for row in rows
        ),
        "directional_mode_contrast_selections": sum(
            int(row["directional_mode_contrast_selections"]) for row in rows
        ),
        "directional_blocked_regressive_actions": sum(
            int(row["directional_blocked_regressive_actions"]) for row in rows
        ),
        "directional_blocked_structural_regressions": sum(
            int(row["directional_blocked_structural_regressions"])
            for row in rows
        ),
        "persistent_pursuit_commitment_selections": sum(
            int(row["persistent_pursuit_commitment_selections"])
            for row in rows
        ),
        "persistent_pursuit_resumed_commitments": sum(
            int(row["persistent_pursuit_resumed_commitments"])
            for row in rows
        ),
        "persistent_pursuit_continuation_actions": sum(
            int(row["persistent_pursuit_continuation_actions"])
            for row in rows
        ),
        "persistent_pursuit_directional_policy_actions": sum(
            int(row["persistent_pursuit_directional_policy_actions"])
            for row in rows
        ),
        "persistent_pursuit_bridge_actions": sum(
            int(row["persistent_pursuit_bridge_actions"])
            for row in rows
        ),
        "persistent_pursuit_entity_contrast_actions": sum(
            int(row["persistent_pursuit_entity_contrast_actions"])
            for row in rows
        ),
        "persistent_pursuit_entity_binding_contrast_actions": sum(
            int(row["persistent_pursuit_entity_binding_contrast_actions"])
            for row in rows
        ),
        "persistent_pursuit_mediated_effect_policy_actions": sum(
            int(row["persistent_pursuit_mediated_effect_policy_actions"])
            for row in rows
        ),
        "persistent_pursuit_mediated_effect_contrast_actions": sum(
            int(row["persistent_pursuit_mediated_effect_contrast_actions"])
            for row in rows
        ),
        "persistent_pursuit_mode_contrast_actions": sum(
            int(row["persistent_pursuit_mode_contrast_actions"])
            for row in rows
        ),
        "persistent_pursuit_progress_events": sum(
            int(row["persistent_pursuit_progress_events"])
            for row in rows
        ),
        "persistent_pursuit_repeated_progress_events": sum(
            int(row["persistent_pursuit_repeated_progress_events"])
            for row in rows
        ),
        "persistent_pursuit_completed_objectives": sum(
            int(row["persistent_pursuit_completed_objectives"])
            for row in rows
        ),
        "persistent_pursuit_attempt_budget_extensions": sum(
            int(row["persistent_pursuit_attempt_budget_extensions"])
            for row in rows
        ),
        "persistent_pursuit_rollout_budget_extensions": sum(
            int(row["persistent_pursuit_rollout_budget_extensions"])
            for row in rows
        ),
        "persistent_pursuit_credit_window_extensions": sum(
            int(row["persistent_pursuit_credit_window_extensions"])
            for row in rows
        ),
        "persistent_pursuit_longest_continuation": max(
            (
                int(row["persistent_pursuit_longest_continuation"])
                for row in rows
            ),
            default=0,
        ),
        "causal_option_dynamic_budget_extensions": sum(
            int(row["causal_option_dynamic_budget_extensions"])
            for row in rows
        ),
        "causal_option_budget_pruned_rollouts": sum(
            int(row["causal_option_budget_pruned_rollouts"])
            for row in rows
        ),
        "controller_errors": sum(
            len(row.get("controller_errors", []) or []) for row in rows
        ),
        "failure_causes": dict(failures),
    }


def _pair_delta(
    legacy: Mapping[str, Any],
    unified: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "levels_completed": (
            int(unified["levels_completed_delta"])
            - int(legacy["levels_completed_delta"])
        ),
        "wins": int(unified["wins"]) - int(legacy["wins"]),
        "experiment_actions": (
            int(unified["experiment_actions"])
            - int(legacy["experiment_actions"])
        ),
        "actions_executed": (
            int(unified["actions_executed"])
            - int(legacy["actions_executed"])
        ),
    }


def _omit_step_traces(payload: Dict[str, Any]) -> None:
    """Keep committed artifacts compact while retaining exact accounting."""
    for pair in payload.get("pairs", []) or []:
        for arm in ("legacy_only", "unified"):
            for attempt in pair.get(arm, {}).get("attempts", []) or []:
                trace = list(attempt.pop("trace", []) or [])
                attempt["step_trace_omitted"] = True
                attempt["step_trace_length"] = len(trace)
        summary = pair.get("unified", {}).get("controller_summary") or {}
        terminal = summary.get("terminal_objectives") or {}
        hypotheses = list(terminal.pop("hypotheses", []) or [])
        if hypotheses:
            terminal["hypothesis_details_omitted"] = True
            terminal["hypothesis_detail_count"] = len(hypotheses)
            terminal["objective_family_counts"] = dict(Counter(
                str(item.get("family", "unknown")) for item in hypotheses
            ))
        temporal = summary.get("temporal_goal_composition") or {}
        temporal_hypotheses = list(temporal.pop("hypotheses", []) or [])
        if temporal_hypotheses:
            temporal["hypothesis_details_omitted"] = True
            temporal["hypothesis_detail_count"] = len(temporal_hypotheses)
        causal = summary.get("causal_subgoal_graph") or {}
        causal_hypotheses = list(causal.pop("hypotheses", []) or [])
        if causal_hypotheses:
            causal["hypothesis_details_omitted"] = True
            causal["hypothesis_detail_count"] = len(causal_hypotheses)
        causal_options = summary.get("causal_hierarchical_options") or {}
        option_hypotheses = list(
            causal_options.pop("hypotheses", []) or []
        )
        if option_hypotheses:
            causal_options["hypothesis_details_omitted"] = True
            causal_options["hypothesis_detail_count"] = len(
                option_hypotheses
            )
        effect_subgoals = (
            causal_options.get("effect_conditioned_subgoals") or {}
        )
        effect_hypotheses = list(
            effect_subgoals.pop("hypotheses", []) or []
        )
        if effect_hypotheses:
            effect_subgoals["hypothesis_details_omitted"] = True
            effect_subgoals["hypothesis_detail_count"] = len(
                effect_hypotheses
            )
        directional = (
            effect_subgoals.get("state_conditioned_directional_model") or {}
        )
        directional_hypotheses = list(
            directional.pop("hypotheses", []) or []
        )
        if directional_hypotheses:
            directional["hypothesis_details_omitted"] = True
            directional["hypothesis_detail_count"] = len(
                directional_hypotheses
            )


def _blocked_attempt(reset_index: int, reason: str) -> Dict[str, Any]:
    return {
        "reset_index": reset_index,
        "status": "BLOCKED",
        "reset_visual_digest": "",
        "actions_executed": 0,
        "levels_completed_delta": 0,
        "max_level_reached": 0,
        "final_game_state": "",
        "win": False,
        "failure_cause": reason,
        "trace": [],
    }


def _classify_attempt_outcome(
    *,
    game_state: str,
    win: bool,
    level_delta: int,
    actions_executed: int,
    action_budget: int,
) -> str:
    state = str(game_state).upper()
    if win:
        return "win"
    if state in TERMINAL_STATES:
        return (
            "terminal_after_level_progress"
            if level_delta > 0
            else "terminal_before_level_progress"
        )
    if actions_executed >= action_budget:
        return (
            "budget_exhausted_after_level_progress"
            if level_delta > 0
            else "budget_exhausted_no_level_progress"
        )
    return "stopped_without_terminal"


def _available_action_names(actions: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for action in actions:
        name = str(getattr(action, "name", ""))
        if name and name != "RESET" and name not in result:
            result.append(name)
    return result


def _action_identity(action: Any) -> str:
    return json.dumps(
        {
            "name": str(getattr(action, "name", "")),
            "action_args": dict(getattr(action, "action_args", {}) or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _visual_digest(grid: Any) -> str:
    array = np.asarray(grid, dtype=np.int32)
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def _is_terminal(state: Any) -> bool:
    return str(state).upper() in TERMINAL_STATES


def _make_real_env(game_id: str, environments_dir: Path) -> Any:
    _configure_offline_env(environments_dir)
    return _make_env(game_id, environments_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired unified-cognition held-out A/B benchmark.",
    )
    parser.add_argument(
        "--games",
        default=",".join(DEFAULT_HELD_OUT_GAMES),
        help="Comma-separated full or short game ids.",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--resets", type=int, default=1)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument(
        "--disable-causal-subgoals",
        action="store_true",
        help="Run an auditable unified-controller ablation without SAGE.8n.",
    )
    parser.add_argument(
        "--disable-causal-effect-credit",
        action="store_true",
        help="Ablate SAGE.8o effect memory and reserved confirmations only.",
    )
    parser.add_argument(
        "--disable-causal-hierarchical-options",
        action="store_true",
        help="Ablate SAGE.8p downstream causal-option exploitation only.",
    )
    parser.add_argument(
        "--disable-effect-conditioned-downstream-subgoals",
        action="store_true",
        help="Ablate SAGE.8q effect-to-subgoal learning and adaptive budget only.",
    )
    parser.add_argument(
        "--disable-state-conditioned-directional-control",
        action="store_true",
        help="Ablate SAGE.8r latent-mode directional action control only.",
    )
    parser.add_argument(
        "--disable-persistent-directional-pursuit",
        action="store_true",
        help="Ablate SAGE.8s progress-gated persistent pursuit only.",
    )
    parser.add_argument(
        "--disable-entity-anchored-interventions",
        action="store_true",
        help="Ablate SAGE.8t entity/structural-role action identities only.",
    )
    parser.add_argument(
        "--disable-active-entity-causal-binding",
        action="store_true",
        help="Ablate SAGE.8u online target/effect causal bindings only.",
    )
    parser.add_argument(
        "--disable-mediated-entity-effect-induction",
        action="store_true",
        help="Ablate SAGE.8v indirect affected-entity induction only.",
    )
    parser.add_argument(
        "--disable-active-mediated-replication",
        action="store_true",
        help="Ablate SAGE.8w cross-branch mediated replication only.",
    )
    parser.add_argument(
        "--disable-online-mediated-anti-unification",
        action="store_true",
        help="Ablate SAGE.8x online structural carrier abstraction only.",
    )
    parser.add_argument(
        "--disable-active-mediated-discrimination",
        action="store_true",
        help="Ablate SAGE.8y one-feature abstraction controls only.",
    )
    parser.add_argument(
        "--disable-active-mode-restoration",
        action="store_true",
        help="Ablate SAGE.8z learned latent-mode restoration only.",
    )
    parser.add_argument(
        "--disable-terminal-mediated-exploitation",
        action="store_true",
        help="Ablate SAGE.9 revised-lattice terminal exploitation only.",
    )
    parser.add_argument(
        "--disable-successor-policy-chaining",
        action="store_true",
        help="Ablate SAGE.9a/9b successor chaining and active exploration.",
    )
    parser.add_argument(
        "--disable-successor-structural-transfer",
        action="store_true",
        help="Ablate SAGE.9c online successor analogy transfer only.",
    )
    parser.add_argument(
        "--disable-horizon-stable-learning-epochs",
        action="store_true",
        help="Ablate SAGE.9d horizon-stable learning epochs only.",
    )
    parser.add_argument(
        "--disable-online-horizon-learning-arbiter",
        action="store_true",
        help="Ablate SAGE.9e online epistemic action allocation only.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    games = [
        game_splits.resolve_full_game_id(item.strip())
        for item in str(args.games).split(",")
        if item.strip()
    ]
    seeds = [
        int(item.strip())
        for item in str(args.seeds).split(",")
        if item.strip()
    ]
    payload = run_unified_cognition_ab_benchmark(
        game_ids=games,
        seeds=seeds,
        action_budget_per_reset=args.budget,
        resets=args.resets,
        environments_dir=args.environments_dir,
        write_path=args.out,
        include_traces=args.include_traces,
        enable_causal_subgoal_induction=(
            not args.disable_causal_subgoals
        ),
        enable_causal_effect_credit=(
            not args.disable_causal_effect_credit
        ),
        enable_causal_hierarchical_options=(
            not args.disable_causal_hierarchical_options
        ),
        enable_effect_conditioned_downstream_subgoals=(
            not args.disable_effect_conditioned_downstream_subgoals
        ),
        enable_state_conditioned_directional_control=(
            not args.disable_state_conditioned_directional_control
        ),
        enable_persistent_directional_pursuit=(
            not args.disable_persistent_directional_pursuit
        ),
        enable_entity_anchored_interventions=(
            not args.disable_entity_anchored_interventions
        ),
        enable_active_entity_causal_binding=(
            not args.disable_active_entity_causal_binding
        ),
        enable_mediated_entity_effect_induction=(
            not args.disable_mediated_entity_effect_induction
        ),
        enable_online_mediated_anti_unification=(
            not args.disable_online_mediated_anti_unification
        ),
        enable_active_mediated_discrimination=(
            not args.disable_active_mediated_discrimination
        ),
        enable_active_mode_restoration=(
            not args.disable_active_mode_restoration
        ),
        enable_terminal_mediated_exploitation=(
            not args.disable_terminal_mediated_exploitation
        ),
        enable_successor_policy_chaining=(
            not args.disable_successor_policy_chaining
        ),
        enable_successor_structural_transfer=(
            not args.disable_successor_structural_transfer
        ),
        enable_active_mediated_replication=(
            not args.disable_active_mediated_replication
        ),
        enable_horizon_stable_learning_epochs=(
            not args.disable_horizon_stable_learning_epochs
        ),
        enable_online_horizon_learning_arbiter=(
            not args.disable_online_horizon_learning_arbiter
        ),
    )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0 if payload["paired_protocol"]["protocol_gate_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HELD_OUT_GAMES",
    "DEFAULT_OUTPUT_PATH",
    "SharedLegacyProposalPolicy",
    "run_unified_cognition_ab_benchmark",
    "write_unified_cognition_ab_benchmark",
]
