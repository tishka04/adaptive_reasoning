"""M1.3l controlled follow-up experiment.

M1.3l runs one controlled follow-up for an unresolved M1 mechanic hypothesis
that has already entered the scientific ledger. It can add support or
contradiction events, but it never confirms/refutes the hypothesis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .polymorphic_a25_adapter import (
    _select_concrete_action,
    _step_env_action,
    measure_required_observation,
)
from .scientific_integration_pretest import (
    DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
)

DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "controlled_experiment_results.json"
)
DEFAULT_CONTROL_ACTIONS = ("ACTION3", "ACTION4", "ACTION1", "ACTION2")


@dataclass(frozen=True)
class ControlledExperiment:
    """One controlled follow-up for an unresolved mechanic hypothesis."""

    hypothesis_key: str
    game_id: str
    mechanic_family: str
    target_action: str
    control_action: str
    baseline_sequence: Tuple[str, ...]
    perturbation_sequence: Tuple[str, ...]
    predicted_metric: str
    observed_baseline: Dict[str, Any] = field(default_factory=dict)
    observed_perturbation: Dict[str, Any] = field(default_factory=dict)
    delta: Dict[str, Any] = field(default_factory=dict)
    support_events: int = 0
    contradiction_events: int = 0
    env_actions: int = 0
    controlled_experiments_run: int = 0
    status: str = "UNRESOLVED"
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_key": self.hypothesis_key,
            "game_id": self.game_id,
            "mechanic_family": self.mechanic_family,
            "target_action": self.target_action,
            "control_action": self.control_action,
            "baseline_sequence": list(self.baseline_sequence),
            "perturbation_sequence": list(self.perturbation_sequence),
            "predicted_metric": self.predicted_metric,
            "observed_baseline": dict(self.observed_baseline),
            "observed_perturbation": dict(self.observed_perturbation),
            "delta": dict(self.delta),
            "support_events": int(self.support_events),
            "contradiction_events": int(self.contradiction_events),
            "env_actions": int(self.env_actions),
            "controlled_experiments_run": int(self.controlled_experiments_run),
            "status": self.status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
        }


def run_controlled_followup_experiment(
    *,
    scientific_integration_path: str | Path = DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    control_actions: Sequence[str] = DEFAULT_CONTROL_ACTIONS,
) -> Dict[str, Any]:
    """Run a first controlled follow-up for the first unresolved ledger entry."""
    payload = json.loads(Path(scientific_integration_path).read_text(encoding="utf-8"))
    entries = [
        dict(entry)
        for entry in payload.get("ledger_entries", []) or []
        if str(entry.get("status", "")).lower() == "unresolved"
        and bool(entry.get("controlled_test_required"))
    ]
    if not entries:
        raise ValueError("no unresolved controlled-test ledger entry found")
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    experiment = execute_controlled_followup(
        entries[0],
        environments_dir=env_dir,
        control_actions=control_actions,
    )
    return {
        "config": {
            "scientific_integration_path": str(scientific_integration_path),
            "environments_dir": str(env_dir),
            "control_actions": list(control_actions),
        },
        "summary": summarize_controlled_experiments([experiment]),
        "controlled_experiments": [experiment.to_dict()],
        "updated_ledger_entries": [updated_ledger_entry_from_experiment(experiment)],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def execute_controlled_followup(
    ledger_entry: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    control_actions: Sequence[str] = DEFAULT_CONTROL_ACTIONS,
) -> ControlledExperiment:
    """Execute target and control actions from fresh resets."""
    spec = ledger_spec_from_entry(ledger_entry)
    baseline_action = select_control_action(
        spec["target_action"],
        control_actions=control_actions,
    )
    try:
        target = execute_single_action_measurement(
            spec["game_id"],
            spec["target_action"],
            spec["predicted_metric"],
            environments_dir=environments_dir,
        )
        baseline = execute_single_action_measurement(
            spec["game_id"],
            baseline_action,
            spec["predicted_metric"],
            environments_dir=environments_dir,
            metric_action_args=target["metric_action_args"],
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return ControlledExperiment(
            hypothesis_key=spec["hypothesis_key"],
            game_id=spec["game_id"],
            mechanic_family=spec["mechanic_family"],
            target_action=spec["target_action"],
            control_action=baseline_action,
            baseline_sequence=("RESET", baseline_action),
            perturbation_sequence=("RESET", spec["target_action"]),
            predicted_metric=spec["predicted_metric"],
            delta={"error": f"controlled_followup_failed:{exc}"},
        )
    delta = controlled_delta(
        baseline["measurement"],
        target["measurement"],
        predicted_metric=spec["predicted_metric"],
    )
    support, contradiction = support_contradiction_from_delta(delta)
    return ControlledExperiment(
        hypothesis_key=spec["hypothesis_key"],
        game_id=spec["game_id"],
        mechanic_family=spec["mechanic_family"],
        target_action=spec["target_action"],
        control_action=baseline_action,
        baseline_sequence=("RESET", baseline_action),
        perturbation_sequence=("RESET", spec["target_action"]),
        predicted_metric=spec["predicted_metric"],
        observed_baseline=baseline["measurement"],
        observed_perturbation=target["measurement"],
        delta=delta,
        support_events=support,
        contradiction_events=contradiction,
        env_actions=int(baseline["env_actions"]) + int(target["env_actions"]),
        controlled_experiments_run=1,
    )


def execute_single_action_measurement(
    game_id: str,
    action_name: str,
    predicted_metric: str,
    *,
    environments_dir: str | Path,
    metric_action_args: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    env = arc.make(game_id)
    before_frame = env.step(GameAction.RESET)
    before = snapshot_frame(before_frame)
    selection_metric = "" if metric_action_args is not None else predicted_metric
    selected_action = _select_concrete_action(
        _valid_actions(env),
        action_name=action_name,
        required_observation=selection_metric,
    )
    if selected_action is None:
        raise ValueError(f"no concrete action available for {action_name}")
    after_frame = _step_env_action(env, selected_action)
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
        "action_args": action_args,
        "metric_action_args": measurement_args,
        "measurement": measurement,
        "env_actions": 1,
    }


def controlled_delta(
    baseline: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    *,
    predicted_metric: str,
) -> Dict[str, Any]:
    baseline_signal = metric_signal(baseline, predicted_metric)
    perturbation_signal = metric_signal(perturbation, predicted_metric)
    effect_size = perturbation_signal - baseline_signal
    return {
        "predicted_metric": predicted_metric,
        "baseline_signal": baseline_signal,
        "perturbation_signal": perturbation_signal,
        "effect_size": effect_size,
        "absolute_effect_size": abs(effect_size),
        "direction": "support" if effect_size > 0 else "contradiction" if effect_size < 0 else "neutral",
    }


def metric_signal(
    measurement: Mapping[str, Any],
    predicted_metric: str,
) -> float:
    metric = str(predicted_metric)
    if metric == "local_patch_before_after":
        return float(measurement.get("local_changed_pixels", 0) or 0)
    if metric == "object_counts_before_after":
        return abs(float(measurement.get("object_count_delta", 0) or 0))
    if metric == "contact_graph_before_after":
        return float(len(measurement.get("contact_pairs_added", []) or [])) + float(
            len(measurement.get("contact_pairs_removed", []) or [])
        )
    if metric == "object_positions_before_after":
        return float(measurement.get("moved_component_count", 0) or 0)
    if metric == "object_shape_zone_before_after":
        return float(len(measurement.get("zone_delta", {}) or {}))
    if metric == "terminal_state_after_rollout":
        if "terminal_rate" in measurement:
            return float(measurement.get("terminal_rate", 0.0) or 0.0)
        return 1.0 if bool(measurement.get("terminal_state_after_rollout")) else 0.0
    if metric == "levels_completed_after_rollout":
        return float(measurement.get("levels_completed_after_rollout", 0) or 0)
    return float(measurement.get("changed_pixels", 0) or 0)


def support_contradiction_from_delta(delta: Mapping[str, Any]) -> Tuple[int, int]:
    effect = float(delta.get("effect_size", 0.0) or 0.0)
    if effect > 0:
        return 1, 0
    if effect < 0:
        return 0, 1
    return 0, 0


def ledger_spec_from_entry(entry: Mapping[str, Any]) -> Dict[str, str]:
    key = str(entry.get("key", ""))
    parts = key.split("::")
    if len(parts) >= 5 and parts[0] == "mechanic_prediction":
        return {
            "hypothesis_key": key,
            "game_id": parts[1],
            "target_action": parts[2],
            "mechanic_family": parts[3],
            "predicted_metric": parts[4],
        }
    description = str(entry.get("description", ""))
    tokens = description.split()
    action = tokens[0] if tokens else ""
    family = tokens[1] if len(tokens) > 1 else ""
    metric = tokens[-1] if tokens else ""
    return {
        "hypothesis_key": key,
        "game_id": str(entry.get("game_id", "")),
        "target_action": action,
        "mechanic_family": family,
        "predicted_metric": metric,
    }


def select_control_action(
    target_action: str,
    *,
    control_actions: Sequence[str] = DEFAULT_CONTROL_ACTIONS,
) -> str:
    target = str(target_action)
    for action in control_actions:
        if str(action) != target:
            return str(action)
    raise ValueError("no control action available")


def updated_ledger_entry_from_experiment(
    experiment: ControlledExperiment,
) -> Dict[str, Any]:
    return {
        "key": experiment.hypothesis_key,
        "game_id": experiment.game_id,
        "status": experiment.status,
        "support": 0,
        "contradictions": 0,
        "support_events": int(experiment.support_events),
        "contradiction_events": int(experiment.contradiction_events),
        "controlled_experiments_run": int(experiment.controlled_experiments_run),
        "controlled_test_required": True,
        "revision_performed": False,
        "observation_counted_as_confirmation": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def summarize_controlled_experiments(
    experiments: Sequence[ControlledExperiment],
) -> Dict[str, Any]:
    return {
        "hypotheses_tested": len(experiments),
        "controlled_experiments_run": sum(
            experiment.controlled_experiments_run for experiment in experiments
        ),
        "env_actions": sum(experiment.env_actions for experiment in experiments),
        "support_events": sum(experiment.support_events for experiment in experiments),
        "contradiction_events": sum(
            experiment.contradiction_events for experiment in experiments
        ),
        "unresolved_hypotheses": len(
            [experiment for experiment in experiments if experiment.status == "UNRESOLVED"]
        ),
        "revision_performed": any(
            experiment.revision_performed for experiment in experiments
        ),
        "observation_counted_as_confirmation": any(
            experiment.observation_counted_as_confirmation
            for experiment in experiments
        ),
        "wrong_confirmations": sum(
            experiment.wrong_confirmations for experiment in experiments
        ),
    }


def write_controlled_experiment_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3l controlled follow-up experiment.",
    )
    parser.add_argument(
        "--scientific-integration",
        type=Path,
        default=DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--control-action",
        action="append",
        default=[],
        help="Control action priority. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_controlled_followup_experiment(
        scientific_integration_path=args.scientific_integration,
        environments_dir=args.environments_dir,
        control_actions=tuple(args.control_action) or DEFAULT_CONTROL_ACTIONS,
    )
    write_controlled_experiment_results(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
                "observation_counted_as_confirmation": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
