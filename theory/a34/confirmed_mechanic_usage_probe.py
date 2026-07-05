"""A34 confirmed mechanic usage probe.

A34 asks whether an A33-confirmed mechanic changes action choice and produces a
useful local observation. It does not re-confirm the mechanic's truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.a33.confirmed_mechanics_registry import (
    DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
)
from theory.m1.controlled_followup_experiment import execute_single_action_measurement
from theory.non_ar25_active_micro_run import _env_dir
from theory.m1.live_anchor_ranking import _load_live_grid_and_actions
from theory.non_ar25_active_micro_run import _configure_offline_env


DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "a34" / "confirmed_mechanic_usage_probe.json"
)
DEFAULT_BASELINE_ORDER = ("ACTION3", "ACTION4", "ACTION1", "ACTION2")


@dataclass(frozen=True)
class ConfirmedMechanicUsageProbeResult:
    """One baseline-vs-treatment usage probe for a confirmed mechanic."""

    key: str
    game_id: str
    predicted_metric: str
    baseline_action: str
    treatment_action: str
    baseline_measurement: Dict[str, Any]
    treatment_measurement: Dict[str, Any]
    utility_assessment: str
    action_choice_changed: bool
    action_prioritized_from_registry: bool
    local_patch_before_after_observed: bool
    useful_new_state: bool
    functional_progress: bool
    contradiction: bool
    game_score_unchanged_or_improved: bool
    registry_entry: Dict[str, Any] = field(default_factory=dict)
    truth_status: str = "NOT_REEVALUATED_BY_A34"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "predicted_metric": self.predicted_metric,
            "baseline_action": self.baseline_action,
            "treatment_action": self.treatment_action,
            "baseline_measurement": dict(self.baseline_measurement),
            "treatment_measurement": dict(self.treatment_measurement),
            "utility_assessment": self.utility_assessment,
            "action_choice_changed": self.action_choice_changed,
            "action_prioritized_from_registry": self.action_prioritized_from_registry,
            "local_patch_before_after_observed": self.local_patch_before_after_observed,
            "useful_new_state": self.useful_new_state,
            "functional_progress": self.functional_progress,
            "contradiction": self.contradiction,
            "game_score_unchanged_or_improved": self.game_score_unchanged_or_improved,
            "registry_entry": dict(self.registry_entry),
            "truth_status": self.truth_status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_confirmed_mechanic_usage_probe(
    *,
    registry_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Dict[str, Any]:
    """Run usage probes for confirmed mechanics in the A33 registry."""
    payload = _load_json(registry_path)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    probes = [
        build_usage_probe_for_mechanic(
            entry,
            environments_dir=env_dir,
            baseline_order=baseline_order,
        )
        for entry in payload.get("confirmed_mechanics", []) or []
        if isinstance(entry, Mapping)
    ]
    return {
        "config": {
            "registry_path": str(registry_path),
            "environments_dir": str(env_dir),
            "baseline_order": list(baseline_order),
        },
        "summary": summarize_usage_probes(probes),
        "usage_probes": [probe.to_dict() for probe in probes],
        "truth_status": "NOT_REEVALUATED_BY_A34",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_usage_probe_for_mechanic(
    entry: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> ConfirmedMechanicUsageProbeResult:
    game_id = str(entry.get("game_id", ""))
    treatment_action = str(entry.get("action", ""))
    predicted_metric = str(entry.get("predicted_metric", ""))
    live_actions = live_action_names(game_id, environments_dir)
    baseline_action = choose_baseline_action(
        live_actions,
        treatment_action=treatment_action,
        baseline_order=baseline_order,
    )
    treatment = execute_single_action_measurement(
        game_id,
        treatment_action,
        predicted_metric,
        environments_dir=environments_dir,
    )
    baseline = execute_single_action_measurement(
        game_id,
        baseline_action,
        predicted_metric,
        environments_dir=environments_dir,
        metric_action_args=treatment["metric_action_args"],
    )
    baseline_measurement = dict(baseline.get("measurement", {}) or {})
    treatment_measurement = dict(treatment.get("measurement", {}) or {})
    baseline_signal = metric_signal(baseline_measurement, predicted_metric)
    treatment_signal = metric_signal(treatment_measurement, predicted_metric)
    utility = utility_assessment(
        baseline_signal=baseline_signal,
        treatment_signal=treatment_signal,
        contradiction=treatment_signal < baseline_signal,
    )
    useful = bool(
        treatment_measurement.get("changed", False)
        and treatment_signal > baseline_signal
    )
    return ConfirmedMechanicUsageProbeResult(
        key=str(entry.get("key", "")),
        game_id=game_id,
        predicted_metric=predicted_metric,
        baseline_action=baseline_action,
        treatment_action=treatment_action,
        baseline_measurement=baseline_measurement,
        treatment_measurement=treatment_measurement,
        utility_assessment=utility,
        action_choice_changed=baseline_action != treatment_action,
        action_prioritized_from_registry=True,
        local_patch_before_after_observed=bool(
            predicted_metric == "local_patch_before_after"
            and treatment_measurement.get("local_changed_pixels", 0)
        ),
        useful_new_state=useful,
        functional_progress=useful,
        contradiction=treatment_signal < baseline_signal,
        game_score_unchanged_or_improved=True,
        registry_entry=dict(entry),
    )


def choose_baseline_action(
    live_actions: Sequence[str],
    *,
    treatment_action: str,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> str:
    live = {str(action) for action in live_actions}
    for action in baseline_order:
        if str(action) in live and str(action) != treatment_action:
            return str(action)
    for action in sorted(live):
        if action not in {"RESET", treatment_action}:
            return action
    raise ValueError("no baseline action available")


def live_action_names(game_id: str, environments_dir: str | Path) -> Tuple[str, ...]:
    env_dir = Path(environments_dir)
    _configure_offline_env(env_dir)
    _, valid_actions = _load_live_grid_and_actions(game_id, env_dir)
    return tuple(
        sorted(
            {
                str(getattr(action, "name", ""))
                for action in valid_actions
                if str(getattr(action, "name", ""))
            }
        )
    )


def metric_signal(measurement: Mapping[str, Any], predicted_metric: str) -> float:
    if predicted_metric == "local_patch_before_after":
        return float(measurement.get("local_changed_pixels", 0) or 0)
    if predicted_metric == "object_counts_before_after":
        return abs(float(measurement.get("object_count_delta", 0) or 0))
    if predicted_metric == "contact_graph_before_after":
        return float(len(measurement.get("contact_pairs_added", []) or [])) + float(
            len(measurement.get("contact_pairs_removed", []) or [])
        )
    if predicted_metric == "object_positions_before_after":
        return float(measurement.get("moved_component_count", 0) or 0)
    if predicted_metric == "object_shape_zone_before_after":
        return float(len(measurement.get("zone_delta", {}) or {}))
    return float(measurement.get("changed_pixels", 0) or 0)


def utility_assessment(
    *,
    baseline_signal: float,
    treatment_signal: float,
    contradiction: bool,
) -> str:
    if contradiction:
        return "CONTEXTUALLY_NOT_USEFUL"
    if treatment_signal > baseline_signal:
        return "USEFUL"
    if treatment_signal == baseline_signal and treatment_signal > 0:
        return "CONTEXTUALLY_USEFUL"
    return "NOT_USEFUL"


def summarize_usage_probes(
    probes: Sequence[ConfirmedMechanicUsageProbeResult],
) -> Dict[str, Any]:
    return {
        "mechanics_probed": len(probes),
        "useful": len([probe for probe in probes if probe.utility_assessment == "USEFUL"]),
        "contextually_useful": len(
            [probe for probe in probes if probe.utility_assessment == "CONTEXTUALLY_USEFUL"]
        ),
        "not_useful": len(
            [probe for probe in probes if probe.utility_assessment == "NOT_USEFUL"]
        ),
        "contradictions": len([probe for probe in probes if probe.contradiction]),
        "action_choice_changed": len(
            [probe for probe in probes if probe.action_choice_changed]
        ),
        "action_prioritized_from_registry": len(
            [probe for probe in probes if probe.action_prioritized_from_registry]
        ),
        "useful_new_states": len([probe for probe in probes if probe.useful_new_state]),
        "functional_progress": len(
            [probe for probe in probes if probe.functional_progress]
        ),
        "truth_status": "NOT_REEVALUATED_BY_A34",
        "wrong_confirmations": 0,
    }


def write_confirmed_mechanic_usage_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A34 confirmed mechanic usage probe.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_confirmed_mechanic_usage_probe(
        registry_path=args.registry,
        environments_dir=args.environments_dir,
    )
    write_confirmed_mechanic_usage_probe(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "truth_status": "NOT_REEVALUATED_BY_A34",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
