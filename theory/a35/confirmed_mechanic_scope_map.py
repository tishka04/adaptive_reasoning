"""A35 confirmed mechanic contextual scope mapper.

A35 tests where an A33-confirmed mechanic is useful. It does not re-evaluate
truth and it does not write scientific verdicts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.a33.confirmed_mechanics_registry import (
    DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
)
from theory.a34.confirmed_mechanic_usage_probe import (
    DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH,
    DEFAULT_BASELINE_ORDER,
    choose_baseline_action,
    metric_signal,
    utility_assessment,
)
from theory.m1.controlled_followup_experiment import (
    _select_concrete_action,
    _step_env_action,
    measure_required_observation,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH = (
    Path("diagnostics") / "a35" / "confirmed_mechanic_scope_map.json"
)
DEFAULT_CONTEXT_SEQUENCES: Tuple[Tuple[str, ...], ...] = (
    (),
    ("ACTION3",),
    ("ACTION4",),
    ("ACTION3", "ACTION4"),
)
TRUTH_STATUS = "NOT_REEVALUATED_BY_A35"


@dataclass(frozen=True)
class ContextualMechanicScopeProbe:
    """One baseline-vs-treatment probe in a short context."""

    key: str
    game_id: str
    context_id: str
    context_sequence: Tuple[str, ...]
    predicted_metric: str
    baseline_action: str
    treatment_action: str
    baseline_measurement: Dict[str, Any] = field(default_factory=dict)
    treatment_measurement: Dict[str, Any] = field(default_factory=dict)
    baseline_signal: float = 0.0
    treatment_signal: float = 0.0
    utility_assessment: str = "NOT_USEFUL"
    local_patch_before_after_observed: bool = False
    useful_new_state: bool = False
    functional_progress: bool = False
    usage_contradiction: bool = False
    score_or_level_unchanged_or_improved: bool = True
    env_actions: int = 0
    error: str = ""
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "context_id": self.context_id,
            "context_sequence": list(self.context_sequence),
            "predicted_metric": self.predicted_metric,
            "baseline_action": self.baseline_action,
            "treatment_action": self.treatment_action,
            "baseline_measurement": dict(self.baseline_measurement),
            "treatment_measurement": dict(self.treatment_measurement),
            "baseline_signal": self.baseline_signal,
            "treatment_signal": self.treatment_signal,
            "utility_assessment": self.utility_assessment,
            "local_patch_before_after_observed": self.local_patch_before_after_observed,
            "useful_new_state": self.useful_new_state,
            "functional_progress": self.functional_progress,
            "usage_contradiction": self.usage_contradiction,
            "score_or_level_unchanged_or_improved": (
                self.score_or_level_unchanged_or_improved
            ),
            "env_actions": int(self.env_actions),
            "error": self.error,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


@dataclass(frozen=True)
class ConfirmedMechanicScopeMap:
    """Scope assessment for one A33-confirmed mechanic."""

    key: str
    game_id: str
    action: str
    mechanic_family: str
    predicted_metric: str
    known_scope_from_a33: str
    a34_usage_reference_loaded: bool
    a34_utility_assessment: str = ""
    a34_functional_progress: bool = False
    context_probes: Tuple[ContextualMechanicScopeProbe, ...] = field(
        default_factory=tuple
    )
    scope_assessment: str = "UNSTABLE_OR_NOT_USEFUL"
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "mechanic": {
                "action": self.action,
                "mechanic_family": self.mechanic_family,
                "predicted_metric": self.predicted_metric,
                "known_scope_from_a33": self.known_scope_from_a33,
            },
            "a34_usage_reference_loaded": self.a34_usage_reference_loaded,
            "a34_utility_assessment": self.a34_utility_assessment,
            "a34_functional_progress": self.a34_functional_progress,
            "context_probes": [probe.to_dict() for probe in self.context_probes],
            "scope_assessment": self.scope_assessment,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_confirmed_mechanic_scope_map(
    *,
    registry_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    usage_probe_path: str | Path = DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    context_sequences: Sequence[Sequence[str]] = DEFAULT_CONTEXT_SEQUENCES,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Dict[str, Any]:
    """Map the short-context utility scope of A33-confirmed mechanics."""
    registry_payload = _load_json(registry_path)
    usage_payload = _load_json(usage_probe_path)
    usage_by_key = _usage_probe_by_key(usage_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    maps = [
        build_scope_map_for_mechanic(
            entry,
            usage_reference=usage_by_key.get(str(entry.get("key", "")), {}),
            environments_dir=env_dir,
            context_sequences=context_sequences,
            baseline_order=baseline_order,
        )
        for entry in registry_payload.get("confirmed_mechanics", []) or []
        if isinstance(entry, Mapping)
    ]
    return {
        "config": {
            "registry_path": str(registry_path),
            "usage_probe_path": str(usage_probe_path),
            "environments_dir": str(env_dir),
            "context_sequences": [list(sequence) for sequence in context_sequences],
            "baseline_order": list(baseline_order),
        },
        "summary": summarize_scope_maps(maps),
        "scope_maps": [scope_map.to_dict() for scope_map in maps],
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_scope_map_for_mechanic(
    entry: Mapping[str, Any],
    *,
    usage_reference: Mapping[str, Any],
    environments_dir: str | Path,
    context_sequences: Sequence[Sequence[str]] = DEFAULT_CONTEXT_SEQUENCES,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> ConfirmedMechanicScopeMap:
    """Probe one mechanic across the configured short contexts."""
    probes = tuple(
        probe_context_for_mechanic(
            entry,
            tuple(str(action) for action in context_sequence),
            environments_dir=environments_dir,
            baseline_order=baseline_order,
        )
        for context_sequence in context_sequences
    )
    return ConfirmedMechanicScopeMap(
        key=str(entry.get("key", "")),
        game_id=str(entry.get("game_id", "")),
        action=str(entry.get("action", "")),
        mechanic_family=str(entry.get("mechanic_family", "")),
        predicted_metric=str(entry.get("predicted_metric", "")),
        known_scope_from_a33=str(entry.get("known_scope", "")),
        a34_usage_reference_loaded=bool(usage_reference),
        a34_utility_assessment=str(usage_reference.get("utility_assessment", "")),
        a34_functional_progress=bool(usage_reference.get("functional_progress", False)),
        context_probes=probes,
        scope_assessment=scope_assessment_from_contexts(probes),
    )


def probe_context_for_mechanic(
    entry: Mapping[str, Any],
    context_sequence: Sequence[str],
    *,
    environments_dir: str | Path,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> ContextualMechanicScopeProbe:
    """Run treatment and neutral baseline after the same context prefix."""
    key = str(entry.get("key", ""))
    game_id = str(entry.get("game_id", ""))
    treatment_action = str(entry.get("action", ""))
    predicted_metric = str(entry.get("predicted_metric", ""))
    context = tuple(str(action) for action in context_sequence)
    context_name = context_id(context)
    try:
        treatment = execute_contextual_action_measurement(
            game_id,
            context,
            treatment_action,
            predicted_metric,
            environments_dir=environments_dir,
        )
        if treatment.get("error"):
            return _error_probe(
                key,
                game_id,
                context_name,
                context,
                predicted_metric,
                treatment_action,
                error=str(treatment.get("error", "")),
            )
        baseline_action = choose_baseline_action(
            tuple(treatment.get("available_actions_before", []) or ()),
            treatment_action=treatment_action,
            baseline_order=baseline_order,
        )
        baseline = execute_contextual_action_measurement(
            game_id,
            context,
            baseline_action,
            predicted_metric,
            environments_dir=environments_dir,
            metric_action_args=dict(treatment.get("metric_action_args", {}) or {}),
        )
        if baseline.get("error"):
            return _error_probe(
                key,
                game_id,
                context_name,
                context,
                predicted_metric,
                treatment_action,
                baseline_action=baseline_action,
                error=str(baseline.get("error", "")),
            )
    except Exception as exc:  # pragma: no cover - integration failure path
        return _error_probe(
            key,
            game_id,
            context_name,
            context,
            predicted_metric,
            treatment_action,
            error=f"context_probe_failed:{exc}",
        )

    baseline_measurement = dict(baseline.get("measurement", {}) or {})
    treatment_measurement = dict(treatment.get("measurement", {}) or {})
    baseline_signal = metric_signal(baseline_measurement, predicted_metric)
    treatment_signal = metric_signal(treatment_measurement, predicted_metric)
    contradiction = treatment_signal < baseline_signal
    no_regression = score_or_level_unchanged_or_improved(treatment)
    useful = bool(
        treatment_measurement.get("changed", False)
        and treatment_signal > baseline_signal
        and no_regression
    )
    return ContextualMechanicScopeProbe(
        key=key,
        game_id=game_id,
        context_id=context_name,
        context_sequence=context,
        predicted_metric=predicted_metric,
        baseline_action=str(baseline.get("action", "")),
        treatment_action=str(treatment.get("action", "")),
        baseline_measurement=baseline_measurement,
        treatment_measurement=treatment_measurement,
        baseline_signal=baseline_signal,
        treatment_signal=treatment_signal,
        utility_assessment=utility_assessment(
            baseline_signal=baseline_signal,
            treatment_signal=treatment_signal,
            contradiction=contradiction,
        ),
        local_patch_before_after_observed=bool(
            predicted_metric == "local_patch_before_after"
            and treatment_measurement.get("local_changed_pixels", 0)
        ),
        useful_new_state=useful,
        functional_progress=useful,
        usage_contradiction=contradiction,
        score_or_level_unchanged_or_improved=no_regression,
        env_actions=int(treatment.get("env_actions", 0) or 0)
        + int(baseline.get("env_actions", 0) or 0),
    )


def execute_contextual_action_measurement(
    game_id: str,
    context_sequence: Sequence[str],
    action_name: str,
    predicted_metric: str,
    *,
    environments_dir: str | Path,
    metric_action_args: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Replay a short context from reset, then measure one action."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    env_dir = Path(environments_dir)
    _configure_offline_env(env_dir)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(env_dir),
    )
    env = arc.make(game_id)
    current_frame = env.step(GameAction.RESET)
    env_actions = 0
    for prefix_action_name in context_sequence:
        selected_prefix = _select_concrete_action(
            _valid_actions(env),
            action_name=str(prefix_action_name),
            required_observation="",
        )
        if selected_prefix is None:
            return {
                "action": action_name,
                "context_sequence": list(context_sequence),
                "env_actions": env_actions,
                "error": f"context_action_not_available:{prefix_action_name}",
            }
        current_frame = _step_env_action(env, selected_prefix)
        env_actions += 1
        if current_frame is None:
            return {
                "action": action_name,
                "context_sequence": list(context_sequence),
                "env_actions": env_actions,
                "error": f"context_action_returned_no_frame:{prefix_action_name}",
            }

    before_actions = _valid_actions(env)
    before_action_names = tuple(
        sorted(
            {
                str(getattr(action, "name", ""))
                for action in before_actions
                if str(getattr(action, "name", ""))
            }
        )
    )
    before = snapshot_frame(current_frame)
    selection_metric = "" if metric_action_args is not None else predicted_metric
    selected_action = _select_concrete_action(
        before_actions,
        action_name=action_name,
        required_observation=selection_metric,
    )
    if selected_action is None:
        return {
            "action": action_name,
            "context_sequence": list(context_sequence),
            "available_actions_before": list(before_action_names),
            "env_actions": env_actions,
            "error": f"action_not_available:{action_name}",
        }
    after_frame = _step_env_action(env, selected_action)
    env_actions += 1
    if after_frame is None:
        return {
            "action": action_name,
            "context_sequence": list(context_sequence),
            "available_actions_before": list(before_action_names),
            "env_actions": env_actions,
            "error": "action_returned_no_frame",
        }
    after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
    action_args = dict(getattr(selected_action, "action_args", {}) or {})
    measurement_args = dict(metric_action_args or action_args)
    measurement = measure_required_observation(
        before.grid,
        after.grid,
        required_observation=predicted_metric,
        action_args=measurement_args,
    )
    return {
        "action": action_name,
        "context_sequence": list(context_sequence),
        "available_actions_before": list(before_action_names),
        "action_args": action_args,
        "metric_action_args": measurement_args,
        "measurement": measurement,
        "levels_before": int(before.levels_completed),
        "levels_after": int(after.levels_completed),
        "game_state_before": before.game_state,
        "game_state_after": after.game_state,
        "env_actions": env_actions,
        "error": "",
    }


def scope_assessment_from_contexts(
    probes: Sequence[ContextualMechanicScopeProbe],
) -> str:
    valid = [probe for probe in probes if not probe.error]
    if not valid:
        return "UNSTABLE_OR_NOT_USEFUL"
    useful = [
        probe
        for probe in valid
        if probe.functional_progress and not probe.usage_contradiction
    ]
    if not useful:
        return "UNSTABLE_OR_NOT_USEFUL"
    reset_useful = any(probe.context_id == "reset_exact" for probe in useful)
    if len(useful) == len(valid):
        return "CONTEXTUALLY_STABLE"
    if reset_useful and len(useful) == 1:
        return "LOCAL_ONLY"
    return "PRECONDITION_DEPENDENT"


def summarize_scope_maps(
    scope_maps: Sequence[ConfirmedMechanicScopeMap],
) -> Dict[str, Any]:
    contexts = [
        probe for scope_map in scope_maps for probe in scope_map.context_probes
    ]
    by_scope = Counter(scope_map.scope_assessment for scope_map in scope_maps)
    return {
        "mechanics_mapped": len(scope_maps),
        "contexts_tested": len(contexts),
        "functional_progress_contexts": len(
            [probe for probe in contexts if probe.functional_progress]
        ),
        "useful_new_state_contexts": len(
            [probe for probe in contexts if probe.useful_new_state]
        ),
        "usage_contradictions": len(
            [probe for probe in contexts if probe.usage_contradiction]
        ),
        "errors": len([probe for probe in contexts if probe.error]),
        "scope_assessments": dict(sorted(by_scope.items())),
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def score_or_level_unchanged_or_improved(measurement_run: Mapping[str, Any]) -> bool:
    before = int(measurement_run.get("levels_before", 0) or 0)
    after = int(measurement_run.get("levels_after", 0) or 0)
    state = str(measurement_run.get("game_state_after", "")).lower()
    regressed_state = "lost" in state or "game_over" in state or "failed" in state
    return after >= before and not regressed_state


def context_id(context_sequence: Sequence[str]) -> str:
    if not context_sequence:
        return "reset_exact"
    return "after_" + "_then_".join(str(action) for action in context_sequence)


def write_confirmed_mechanic_scope_map(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _usage_probe_by_key(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("key", "")): dict(row)
        for row in payload.get("usage_probes", []) or []
        if isinstance(row, Mapping) and str(row.get("key", ""))
    }


def _error_probe(
    key: str,
    game_id: str,
    context_name: str,
    context: Tuple[str, ...],
    predicted_metric: str,
    treatment_action: str,
    *,
    baseline_action: str = "",
    error: str,
) -> ContextualMechanicScopeProbe:
    return ContextualMechanicScopeProbe(
        key=key,
        game_id=game_id,
        context_id=context_name,
        context_sequence=context,
        predicted_metric=predicted_metric,
        baseline_action=baseline_action,
        treatment_action=treatment_action,
        error=error,
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A35 confirmed mechanic contextual scope mapper.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--usage-probe",
        type=Path,
        default=DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_confirmed_mechanic_scope_map(
        registry_path=args.registry,
        usage_probe_path=args.usage_probe,
        environments_dir=args.environments_dir,
    )
    write_confirmed_mechanic_scope_map(payload, args.out)
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
