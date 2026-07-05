"""A37 closed-loop rollout for scope-conditioned action policy.

A37 keeps one environment alive across several steps and uses only A33 + A35
to decide whether a confirmed mechanic should drive the next action. It does
not re-evaluate truth and it does not write scientific verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.a33.confirmed_mechanics_registry import (
    DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
)
from theory.a35.confirmed_mechanic_scope_map import (
    DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    context_id,
)
from theory.a36.scope_conditioned_policy_probe import (
    DEFAULT_BASELINE_ORDER,
    ELIGIBLE_SCOPE,
    context_matches_scope,
    metric_signal,
)
from theory.m1.controlled_followup_experiment import (
    _select_concrete_action,
    _step_env_action,
    measure_required_observation,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH = (
    Path("diagnostics") / "a37" / "scope_conditioned_policy_rollout.json"
)
DEFAULT_BUDGET = 4
TRUTH_STATUS = "NOT_REEVALUATED_BY_A37"


@dataclass(frozen=True)
class ScopeConditionedRolloutDecision:
    """One policy decision before execution."""

    key: str
    action: str
    fallback_action: str
    predicted_metric: str
    selected_from_confirmed_mechanic: bool
    scope_used: str
    context_signature: Tuple[str, ...]
    context_match: bool
    context_match_reason: str
    decision_reason: str


@dataclass(frozen=True)
class PolicyRolloutStep:
    """One executed step of the closed-loop rollout."""

    step: int
    key: str
    game_id: str
    context_signature: Tuple[str, ...]
    context_id: str
    policy_selected_action: str
    fallback_action: str
    predicted_metric: str
    selected_from_confirmed_mechanic: bool
    scope_used: str
    context_match: bool
    context_match_reason: str
    decision_reason: str
    action_args: Dict[str, Any] = field(default_factory=dict)
    measurement: Dict[str, Any] = field(default_factory=dict)
    selected_signal: float = 0.0
    functional_progress: bool = False
    useful_new_state: bool = False
    usage_contradiction: bool = False
    repeated_usefulness: bool = False
    dead_end_or_cycle: bool = False
    state_signature_before: str = ""
    state_signature_after: str = ""
    levels_before: int = 0
    levels_after: int = 0
    game_state_before: str = ""
    game_state_after: str = ""
    env_actions: int = 0
    error: str = ""
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "key": self.key,
            "game_id": self.game_id,
            "context_signature": list(self.context_signature),
            "context_id": self.context_id,
            "policy_selected_action": self.policy_selected_action,
            "fallback_action": self.fallback_action,
            "predicted_metric": self.predicted_metric,
            "selected_from_confirmed_mechanic": self.selected_from_confirmed_mechanic,
            "scope_used": self.scope_used,
            "context_match": self.context_match,
            "context_match_reason": self.context_match_reason,
            "decision_reason": self.decision_reason,
            "action_args": dict(self.action_args),
            "measurement": dict(self.measurement),
            "selected_signal": self.selected_signal,
            "functional_progress": self.functional_progress,
            "useful_new_state": self.useful_new_state,
            "usage_contradiction": self.usage_contradiction,
            "repeated_usefulness": self.repeated_usefulness,
            "dead_end_or_cycle": self.dead_end_or_cycle,
            "state_signature_before": self.state_signature_before,
            "state_signature_after": self.state_signature_after,
            "levels_before": int(self.levels_before),
            "levels_after": int(self.levels_after),
            "game_state_before": self.game_state_before,
            "game_state_after": self.game_state_after,
            "env_actions": int(self.env_actions),
            "error": self.error,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_scope_conditioned_policy_rollout(
    *,
    registry_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    scope_map_path: str | Path = DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    budget: int = DEFAULT_BUDGET,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Dict[str, Any]:
    """Run a short closed-loop rollout using only A33 + A35 as policy memory."""
    registry_payload = _load_json(registry_path)
    scope_payload = _load_json(scope_map_path)
    registry_entries = [
        dict(entry)
        for entry in registry_payload.get("confirmed_mechanics", []) or []
        if isinstance(entry, Mapping)
    ]
    scopes_by_key = _scope_maps_by_key(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    steps = execute_scope_conditioned_rollout(
        registry_entries,
        scopes_by_key=scopes_by_key,
        environments_dir=env_dir,
        budget=budget,
        baseline_order=baseline_order,
    )
    return {
        "config": {
            "registry_path": str(registry_path),
            "scope_map_path": str(scope_map_path),
            "environments_dir": str(env_dir),
            "budget": int(budget),
            "baseline_order": list(baseline_order),
            "inputs_read": ["A33", "A35"],
            "artifacts_not_read": ["M3", "A32", "A34", "A36"],
        },
        "summary": summarize_rollout_steps(steps),
        "rollout_steps": [step.to_dict() for step in steps],
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def execute_scope_conditioned_rollout(
    registry_entries: Sequence[Mapping[str, Any]],
    *,
    scopes_by_key: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
    budget: int = DEFAULT_BUDGET,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Tuple[PolicyRolloutStep, ...]:
    """Execute a short policy rollout in one live environment."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    if not registry_entries:
        return tuple()
    first_entry = dict(registry_entries[0])
    game_id = str(first_entry.get("game_id", ""))
    env_dir = Path(environments_dir)
    _configure_offline_env(env_dir)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(env_dir),
    )
    env = arc.make(game_id)
    current_frame = env.step(GameAction.RESET)

    action_history: list[str] = []
    fallback_counts: Counter[str] = Counter()
    useful_keys: set[str] = set()
    seen_states: set[str] = set()
    steps: list[PolicyRolloutStep] = []
    initial = snapshot_frame(current_frame)
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(current_frame)
        before_signature = state_signature(
            before.grid,
            before.levels_completed,
            before.game_state,
        )
        decision = select_rollout_decision(
            registry_entries,
            scopes_by_key=scopes_by_key,
            action_history=tuple(action_history),
            fallback_counts=fallback_counts,
            baseline_order=baseline_order,
        )
        selected_action = select_available_action(
            env,
            desired_action=decision.action,
            predicted_metric=decision.predicted_metric,
            require_metric_args=decision.selected_from_confirmed_mechanic,
        )
        selected_from_mechanic = decision.selected_from_confirmed_mechanic
        decision_reason = decision.decision_reason
        if selected_action is None:
            fallback = select_available_fallback(
                env,
                treatment_action=decision.action,
                fallback_counts=fallback_counts,
                baseline_order=baseline_order,
            )
            if fallback is None:
                steps.append(
                    error_step(
                        step_index,
                        first_entry,
                        decision,
                        game_id=game_id,
                        error="no_available_policy_or_fallback_action",
                    )
                )
                break
            selected_action = fallback
            selected_from_mechanic = False
            decision_reason = "confirmed_action_unavailable_fallback_executed"

        after_frame = _step_env_action(env, selected_action)
        if after_frame is None:
            steps.append(
                error_step(
                    step_index,
                    first_entry,
                    decision,
                    game_id=game_id,
                    error="env_step_returned_no_frame",
                )
            )
            break
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        action_name = str(getattr(selected_action, "name", decision.action))
        action_args = dict(getattr(selected_action, "action_args", {}) or {})
        measurement = measure_required_observation(
            before.grid,
            after.grid,
            required_observation=decision.predicted_metric,
            action_args=action_args,
        )
        selected_signal = metric_signal(measurement, decision.predicted_metric)
        no_regression = score_or_level_unchanged_or_improved(
            before_levels=before.levels_completed,
            after_levels=after.levels_completed,
            game_state_after=after.game_state,
        )
        after_signature = state_signature(
            after.grid,
            after.levels_completed,
            after.game_state,
        )
        cycle = after_signature in seen_states
        functional = bool(
            measurement.get("changed", False)
            and selected_signal > 0
            and no_regression
        )
        useful_new_state = bool(functional and not cycle)
        usage_contradiction = bool(selected_from_mechanic and selected_signal <= 0)
        repeated = bool(
            selected_from_mechanic
            and useful_new_state
            and decision.key in useful_keys
        )
        step = PolicyRolloutStep(
            step=step_index,
            key=decision.key,
            game_id=game_id,
            context_signature=decision.context_signature,
            context_id=context_id(decision.context_signature),
            policy_selected_action=action_name,
            fallback_action=decision.fallback_action,
            predicted_metric=decision.predicted_metric,
            selected_from_confirmed_mechanic=selected_from_mechanic,
            scope_used=decision.scope_used,
            context_match=decision.context_match,
            context_match_reason=decision.context_match_reason,
            decision_reason=decision_reason,
            action_args=action_args,
            measurement=measurement,
            selected_signal=selected_signal,
            functional_progress=functional,
            useful_new_state=useful_new_state,
            usage_contradiction=usage_contradiction,
            repeated_usefulness=repeated,
            dead_end_or_cycle=cycle,
            state_signature_before=before_signature,
            state_signature_after=after_signature,
            levels_before=before.levels_completed,
            levels_after=after.levels_completed,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            env_actions=1,
        )
        steps.append(step)
        seen_states.add(after_signature)
        if selected_from_mechanic and useful_new_state:
            useful_keys.add(decision.key)
        if not selected_from_mechanic:
            fallback_counts[action_name] += 1
        action_history.append(action_name)
        current_frame = after_frame
    return tuple(steps)


def select_rollout_decision(
    registry_entries: Sequence[Mapping[str, Any]],
    *,
    scopes_by_key: Mapping[str, Mapping[str, Any]],
    action_history: Sequence[str],
    fallback_counts: Mapping[str, int],
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> ScopeConditionedRolloutDecision:
    """Select a confirmed mechanic action or a neutral fallback."""
    for entry in registry_entries:
        key = str(entry.get("key", ""))
        scope_map = scopes_by_key.get(key, {})
        if str(scope_map.get("scope_assessment", "")) != ELIGIBLE_SCOPE:
            continue
        signature, matched, reason = context_signature_for_scope(
            action_history,
            scope_map,
        )
        if matched:
            return ScopeConditionedRolloutDecision(
                key=key,
                action=str(entry.get("action", "")),
                fallback_action=fallback_action_for_rollout(
                    treatment_action=str(entry.get("action", "")),
                    fallback_counts=fallback_counts,
                    baseline_order=baseline_order,
                ),
                predicted_metric=str(entry.get("predicted_metric", "")),
                selected_from_confirmed_mechanic=True,
                scope_used=str(scope_map.get("scope_assessment", "")),
                context_signature=signature,
                context_match=True,
                context_match_reason=reason,
                decision_reason="covered_scope_prioritize_confirmed_mechanic",
            )

    entry = dict(registry_entries[0]) if registry_entries else {}
    action = fallback_action_for_rollout(
        treatment_action=str(entry.get("action", "")),
        fallback_counts=fallback_counts,
        baseline_order=baseline_order,
    )
    signature = fallback_context_signature(action_history, scopes_by_key.values())
    return ScopeConditionedRolloutDecision(
        key=str(entry.get("key", "")),
        action=action,
        fallback_action=action,
        predicted_metric=str(entry.get("predicted_metric", "")),
        selected_from_confirmed_mechanic=False,
        scope_used=str(scopes_by_key.get(str(entry.get("key", "")), {}).get("scope_assessment", "")),
        context_signature=signature,
        context_match=False,
        context_match_reason="no_covered_rollout_context",
        decision_reason="fallback_neutral_exploration",
    )


def context_signature_for_scope(
    action_history: Sequence[str],
    scope_map: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], bool, str]:
    """Find the longest covered suffix of the rollout action history."""
    history = tuple(str(action) for action in action_history)
    if not history:
        matched, reason = context_matches_scope(scope_map, live_context_sequence=())
        return (), matched, reason
    max_len = max_context_length(scope_map)
    for length in range(min(max_len + 1, len(history)), 0, -1):
        suffix = history[-length:]
        matched, reason = context_matches_scope(
            scope_map,
            live_context_sequence=suffix,
        )
        if matched:
            return suffix, True, reason
    return fallback_context_signature(history, (scope_map,)), False, "no_covered_rollout_suffix"


def fallback_context_signature(
    action_history: Sequence[str],
    scope_maps: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    if not action_history:
        return ()
    max_len = max([max_context_length(scope_map) for scope_map in scope_maps] or [1])
    return tuple(str(action) for action in action_history[-max(1, max_len) :])


def max_context_length(scope_map: Mapping[str, Any]) -> int:
    lengths = [
        len(probe.get("context_sequence", []) or [])
        for probe in scope_map.get("context_probes", []) or []
        if isinstance(probe, Mapping)
    ]
    return max(lengths or [0])


def fallback_action_for_rollout(
    *,
    treatment_action: str,
    fallback_counts: Mapping[str, int],
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> str:
    candidates = [
        str(action)
        for action in baseline_order
        if str(action) and str(action) not in {"RESET", str(treatment_action)}
    ]
    if not candidates:
        return ""
    return min(
        candidates,
        key=lambda action: (int(fallback_counts.get(action, 0) or 0), candidates.index(action)),
    )


def select_available_action(
    env: Any,
    *,
    desired_action: str,
    predicted_metric: str,
    require_metric_args: bool,
) -> Any | None:
    return _select_concrete_action(
        _valid_actions(env),
        action_name=desired_action,
        required_observation=predicted_metric if require_metric_args else "",
    )


def select_available_fallback(
    env: Any,
    *,
    treatment_action: str,
    fallback_counts: Mapping[str, int],
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Any | None:
    ordered = sorted(
        [
            str(action)
            for action in baseline_order
            if str(action) and str(action) not in {"RESET", str(treatment_action)}
        ],
        key=lambda action: (int(fallback_counts.get(action, 0) or 0), baseline_order.index(action)),
    )
    for action_name in ordered:
        selected = select_available_action(
            env,
            desired_action=action_name,
            predicted_metric="",
            require_metric_args=False,
        )
        if selected is not None:
            return selected
    for action in _valid_actions(env):
        action_name = str(getattr(action, "name", ""))
        if action_name and action_name not in {"RESET", str(treatment_action)}:
            return action
    return None


def summarize_rollout_steps(
    steps: Sequence[PolicyRolloutStep],
) -> Dict[str, Any]:
    return {
        "policy_steps": len(steps),
        "policy_steps_from_confirmed_mechanic": len(
            [step for step in steps if step.selected_from_confirmed_mechanic]
        ),
        "functional_progress_steps": len(
            [step for step in steps if step.functional_progress]
        ),
        "useful_new_states": len([step for step in steps if step.useful_new_state]),
        "usage_contradictions": len(
            [step for step in steps if step.usage_contradiction]
        ),
        "fallback_steps": len(
            [step for step in steps if not step.selected_from_confirmed_mechanic]
        ),
        "repeated_usefulness": len(
            [step for step in steps if step.repeated_usefulness]
        ),
        "cycle_or_dead_end_detected": any(step.dead_end_or_cycle for step in steps),
        "dead_end_or_cycle_steps": len(
            [step for step in steps if step.dead_end_or_cycle]
        ),
        "errors": len([step for step in steps if step.error]),
        "live_env_actions": sum(step.env_actions for step in steps),
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def score_or_level_unchanged_or_improved(
    *,
    before_levels: int,
    after_levels: int,
    game_state_after: str,
) -> bool:
    state = str(game_state_after).lower()
    regressed_state = "lost" in state or "game_over" in state or "failed" in state
    return int(after_levels) >= int(before_levels) and not regressed_state


def state_signature(grid: Any, levels_completed: int, game_state: str) -> str:
    array = np.asarray(grid, dtype=np.int32)
    digest = hashlib.sha1(array.tobytes()).hexdigest()[:16]
    return f"{tuple(array.shape)}:{digest}:{int(levels_completed)}:{game_state}"


def error_step(
    step_index: int,
    entry: Mapping[str, Any],
    decision: ScopeConditionedRolloutDecision,
    *,
    game_id: str,
    error: str,
) -> PolicyRolloutStep:
    return PolicyRolloutStep(
        step=step_index,
        key=decision.key or str(entry.get("key", "")),
        game_id=game_id,
        context_signature=decision.context_signature,
        context_id=context_id(decision.context_signature),
        policy_selected_action=decision.action,
        fallback_action=decision.fallback_action,
        predicted_metric=decision.predicted_metric,
        selected_from_confirmed_mechanic=decision.selected_from_confirmed_mechanic,
        scope_used=decision.scope_used,
        context_match=decision.context_match,
        context_match_reason=decision.context_match_reason,
        decision_reason=decision.decision_reason,
        error=error,
    )


def write_scope_conditioned_policy_rollout(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scope_maps_by_key(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("key", "")): dict(row)
        for row in payload.get("scope_maps", []) or []
        if isinstance(row, Mapping) and str(row.get("key", ""))
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A37 scope-conditioned closed-loop policy rollout.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-map",
        type=Path,
        default=DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--out", type=Path, default=DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_scope_conditioned_policy_rollout(
        registry_path=args.registry,
        scope_map_path=args.scope_map,
        environments_dir=args.environments_dir,
        budget=args.budget,
    )
    write_scope_conditioned_policy_rollout(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "truth_status": TRUTH_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
