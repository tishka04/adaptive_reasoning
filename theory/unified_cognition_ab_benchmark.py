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

from .unified_cognitive_controller import UnifiedCognitiveController


DEFAULT_OUTPUT_PATH = (
    Path("diagnostics") / "sage" / "unified_cognition_ab_held_out.json"
)
DEFAULT_HELD_OUT_GAMES = tuple(
    game_splits.resolve("public_unseen_split", full_ids=True)
)
SCHEMA_VERSION = "sage.unified_cognition_ab_held_out.v4"
WIN_STATES = {"WIN", "WON", "VICTORY"}
TERMINAL_STATES = WIN_STATES | {"GAME_OVER", "TERMINATED", "FINISHED"}
EXPERIMENT_SOURCES = {
    "discriminating_experiment",
    "relational_experiment",
    "terminal_objective_probe",
    "terminal_objective_discriminator",
    "terminal_objective_ablation",
    "temporal_subgoal_probe",
}

EnvFactory = Callable[[str], Any]


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
    write_path: str | Path | None = None,
    include_traces: bool = False,
) -> Dict[str, Any]:
    """Run paired legacy-only/unified online-learning episodes."""
    games = tuple(str(game) for game in (game_ids or DEFAULT_HELD_OUT_GAMES))
    normalized_seeds = tuple(int(seed) for seed in seeds)
    budget = max(0, int(action_budget_per_reset))
    reset_count = max(1, int(resets))
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

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
            )
            unified = _run_arm(
                arm="unified",
                game_id=game_id,
                seed=seed,
                action_budget_per_reset=budget,
                resets=reset_count,
                env_dir=env_dir,
                env_factory=env_factory,
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
) -> Dict[str, Any]:
    controller = (
        UnifiedCognitiveController(game_id)
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
