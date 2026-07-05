"""M3.7 batch runner for M2 candidate experiment requests."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m1.controlled_followup_experiment import (
    controlled_delta,
    support_contradiction_from_delta,
)
from theory.m1.polymorphic_a25_adapter import measure_required_observation
from theory.m2.m3_execution_smoke import (
    execute_contextual_m2_request,
    source_hypothesis_live_state_signature,
)
from theory.m2.schema import M2_TRUTH_STATUS, M3CandidateExperimentRequest
from theory.m2.testability_compiler import (
    DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    load_m3_requests_payload,
    requests_from_payload,
)
from theory.non_ar25_active_micro_run import _env_dir


DEFAULT_M3_M2_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "m2_candidate_experiment_results.json"
)
DEFAULT_OFFLINE_TRACE_DATASET_PATH = (
    Path("training") / "m2_arc_lewm_transitions.jsonl"
)
DEFAULT_SECONDARY_METRICS = (
    "changed_pixels",
    "object_positions_before_after",
    "contact_graph_before_after",
    "topology_before_after",
)
RAW_CHANGED_PIXELS_METRIC = "changed_pixels"


def run_m2_candidate_experiment_queue(
    *,
    m2_requests_path: str | Path = DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    offline_trace_dataset_path: str | Path = DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    secondary_metrics: Sequence[str] = DEFAULT_SECONDARY_METRICS,
    max_requests: int | None = None,
) -> Dict[str, Any]:
    """Execute READY_FOR_M3 M2 requests as M3 candidate-only experiments."""
    payload = load_m3_requests_payload(m2_requests_path)
    requests = [
        request
        for request in requests_from_payload(payload)
        if request.status == "READY_FOR_M3"
    ]
    if max_requests is not None:
        requests = requests[: max(0, int(max_requests))]

    target_context_signatures = {
        request.source_hypothesis_id: source_hypothesis_live_state_signature(
            payload,
            request.source_hypothesis_id,
        )
        for request in requests
    }

    experiments: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    for request in requests:
        try:
            primary = execute_metric_experiment(
                request,
                environments_dir=environments_dir,
                offline_trace_dataset_path=offline_trace_dataset_path,
                metric_role="primary",
                target_context_signature=target_context_signatures.get(
                    request.source_hypothesis_id,
                    "",
                ),
            )
        except Exception as exc:  # pragma: no cover - integration failure path
            blocked.append(
                {
                    "request_id": request.request_id,
                    "source_hypothesis_id": request.source_hypothesis_id,
                    "reason": f"primary_execution_failed:{exc}",
                }
            )
            continue
        experiments.append(primary)
        if not should_escalate_metrics(primary):
            continue
        for metric in secondary_metrics:
            if metric == request.metric:
                continue
            try:
                secondary = execute_metric_experiment(
                    replace(request, metric=str(metric)),
                    environments_dir=environments_dir,
                    offline_trace_dataset_path=offline_trace_dataset_path,
                    metric_role="secondary",
                    target_context_signature=target_context_signatures.get(
                        request.source_hypothesis_id,
                        "",
                    ),
                    escalation_reason=(
                        "primary_metric_neutral_but_changed_pixels_effect_observed"
                    ),
                )
            except Exception as exc:  # pragma: no cover - integration failure path
                blocked.append(
                    {
                        "request_id": request.request_id,
                        "source_hypothesis_id": request.source_hypothesis_id,
                        "metric": str(metric),
                        "reason": f"secondary_execution_failed:{exc}",
                    }
                )
                continue
            experiments.append(secondary)

    return build_results_payload(
        m2_requests_path=m2_requests_path,
        environments_dir=environments_dir,
        offline_trace_dataset_path=offline_trace_dataset_path,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
        secondary_metrics=secondary_metrics,
    )


def execute_metric_experiment(
    request: M3CandidateExperimentRequest,
    *,
    environments_dir: str | Path | None,
    offline_trace_dataset_path: str | Path = DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    metric_role: str,
    target_context_signature: str = "",
    escalation_reason: str = "",
) -> Dict[str, Any]:
    if is_offline_trace_context_request(request):
        row = execute_offline_trace_context_request(
            request,
            offline_trace_dataset_path=offline_trace_dataset_path,
        )
    else:
        row = execute_contextual_m2_request(
            request,
            environments_dir=environments_dir,
            target_context_signature=target_context_signature,
        )
    row = apply_metric_grounding_guard(
        row,
        metric=request.metric,
    )
    neutral = neutral_events_for_experiment(row)
    row.update(
        {
            "request_id": request.request_id,
            "source_hypothesis_id": request.source_hypothesis_id,
            "metric": request.metric,
            "metric_role": metric_role,
            "neutral_events": neutral,
            "revision_status": "CANDIDATE_ONLY",
            "status": "UNRESOLVED",
            "support": 0,
            "contradictions": 0,
            "controlled_test_required": True,
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "trace_support_counted_as_proof": False,
            "prior_counted_as_proof": False,
            "observation_counted_as_confirmation": False,
        }
    )
    if escalation_reason:
        row["metric_escalation_reason"] = escalation_reason
    return row


def is_offline_trace_context_request(request: M3CandidateExperimentRequest) -> bool:
    return (
        str(request.replayability or "") == "OFFLINE_TRACE_CONTEXT_ONLY"
        and str(request.context_state_origin or "") == "human_trace_frame_before"
    )


def execute_offline_trace_context_request(
    request: M3CandidateExperimentRequest,
    *,
    offline_trace_dataset_path: str | Path = DEFAULT_OFFLINE_TRACE_DATASET_PATH,
) -> Dict[str, Any]:
    """Execute an ARC-LeWM request from a trace frame-before context.

    This path is intentionally trace-grounded: the target observation is the
    exact source transition and the baseline is estimated from matched human
    trace transitions where the dynamic control action was taken.
    """
    rows = load_offline_trace_rows(offline_trace_dataset_path)
    source = find_offline_source_transition(request, rows)
    _validate_offline_source_matches_request(request, source)
    control_action = choose_offline_trace_control_action(request, source)
    control_rows, control_policy = matched_offline_control_rows(
        rows,
        source=source,
        control_action=control_action,
    )
    if not control_rows:
        raise ValueError(f"no_offline_trace_controls_available:{control_action}")

    target = offline_trace_measurement(
        [source],
        metric=request.metric,
        action_name=request.target_action,
        action_args=request.target_action_args,
        signal_source="offline_trace_source_transition",
    )
    control = offline_trace_measurement(
        control_rows,
        metric=request.metric,
        action_name=control_action,
        action_args=None,
        signal_source="offline_trace_matched_dynamic_controls",
    )
    delta = controlled_delta(
        control["measurement"],
        target["measurement"],
        predicted_metric=request.metric,
    )
    support, contradiction = support_contradiction_from_delta(delta)
    source_transition_id = source_transition_id_for_row(source)
    return {
        "hypothesis_key": request.source_hypothesis_id,
        "game_id": request.game_id,
        "mechanic_family": "frontier_conditioned_candidate",
        "target_action": request.target_action,
        "target_action_args": (
            dict(request.target_action_args)
            if request.target_action_args is not None
            else None
        ),
        "control_action": control_action,
        "context_replay": list(request.context_replay),
        "context_replay_args": (
            [dict(item) for item in request.context_replay_args]
            if request.context_replay_args is not None
            else None
        ),
        "target_context_signature": source_transition_id,
        "baseline_sequence": [
            f"OFFLINE_TRACE_MATCHED::{control_action}",
        ],
        "perturbation_sequence": [
            source_transition_id,
            request.target_action,
        ],
        "predicted_metric": request.metric,
        "observed_baseline": control["measurement"],
        "observed_perturbation": target["measurement"],
        "delta": delta,
        "baseline_signal": control["signal"],
        "perturbation_signal": target["signal"],
        "support_events": support,
        "contradiction_events": contradiction,
        "controlled_experiments_run": 1,
        "env_actions": 0,
        "execution_mode": "offline_trace_context",
        "replayability": request.replayability,
        "context_state_origin": request.context_state_origin,
        "source_transition_id": source_transition_id,
        "source_episode_id": request.source_episode_id or str(source.get("episode_id", "")),
        "source_step": (
            request.source_step
            if request.source_step is not None
            else int(source.get("step", 0) or 0)
        ),
        "offline_trace_dataset_path": str(offline_trace_dataset_path),
        "matched_control_policy": control_policy,
        "matched_control_samples": len(control_rows),
        "target_trace_samples": 1,
        "dynamic_available_actions": list(source.get("available_actions_t", []) or []),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def load_offline_trace_rows(path: str | Path) -> tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return tuple(rows)


def find_offline_source_transition(
    request: M3CandidateExperimentRequest,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_transition_id = str(request.source_transition_id or "")
    for row in rows:
        if source_transition_id and source_transition_id_for_row(row) == source_transition_id:
            return dict(row)
    for row in rows:
        if (
            str(row.get("game_id", "")) == request.game_id
            and str(row.get("episode_id", "")) == str(request.source_episode_id or "")
            and int(row.get("step", 0) or 0) == int(request.source_step or -1)
        ):
            return dict(row)
    raise ValueError(f"offline_source_transition_not_found:{source_transition_id}")


def source_transition_id_for_row(row: Mapping[str, Any]) -> str:
    return (
        f"m2_14d::{row.get('game_id', 'unknown')}::"
        f"{row.get('episode_id', 'episode')}::{int(row.get('step', 0) or 0):04d}"
    )


def choose_offline_trace_control_action(
    request: M3CandidateExperimentRequest,
    source: Mapping[str, Any],
) -> str:
    live = {
        str(action)
        for action in source.get("available_actions_t", []) or []
        if str(action)
    }
    for action in request.suggested_control_actions:
        action_name = str(action)
        if action_name in live and action_name != request.target_action:
            return action_name
    for action_name in sorted(live):
        if action_name not in {"RESET", request.target_action}:
            return action_name
    raise ValueError("no_offline_trace_control_available")


def matched_offline_control_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    control_action: str,
) -> tuple[tuple[Dict[str, Any], ...], str]:
    game_id = str(source.get("game_id", ""))
    source_actions = {
        str(action)
        for action in source.get("available_actions_t", []) or []
        if str(action)
    }
    same_action = [
        dict(row)
        for row in rows
        if str(row.get("game_id", "")) == game_id
        and str(row.get("action", "")) == control_action
    ]
    strict = [
        row
        for row in same_action
        if {
            str(action)
            for action in row.get("available_actions_t", []) or []
            if str(action)
        }
        == source_actions
    ]
    if strict:
        return tuple(strict), "same_game_same_available_actions"
    available = [
        row
        for row in same_action
        if control_action in {str(action) for action in row.get("available_actions_t", []) or []}
    ]
    if available:
        return tuple(available), "same_game_control_available"
    return tuple(same_action), "same_game_same_control_action"


def _validate_offline_source_matches_request(
    request: M3CandidateExperimentRequest,
    source: Mapping[str, Any],
) -> None:
    if str(source.get("action", "")) != request.target_action:
        raise ValueError(
            "offline_source_action_mismatch:"
            f"{source.get('action', '')}!={request.target_action}"
        )
    expected_args = (
        {str(key): str(value) for key, value in dict(request.target_action_args).items()}
        if request.target_action_args is not None
        else None
    )
    if expected_args is None:
        return
    actual_args = {
        str(key): str(value)
        for key, value in dict(source.get("action_args", {}) or {}).items()
    }
    if actual_args != expected_args:
        raise ValueError(
            "offline_source_action_args_mismatch:"
            f"{actual_args}!={expected_args}"
        )


def offline_trace_measurement(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    action_name: str,
    action_args: Mapping[str, Any] | None,
    signal_source: str,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError(f"no_offline_trace_rows_for_measurement:{action_name}")
    if metric == "terminal_state_after_rollout":
        terminal_count = sum(1 for row in rows if bool(row.get("terminal_t1", False)))
        samples = len(rows)
        measurement = {
            "metric": metric,
            "measurable": True,
            "signal_source": signal_source,
            "terminal_state_after_rollout": terminal_count > 0 if samples == 1 else None,
            "terminal_count": terminal_count,
            "terminal_rate": terminal_count / max(samples, 1),
            "samples": samples,
            "changed_pixels": _mean_changed_pixels(rows),
        }
        return {
            "action": action_name,
            "action_args": dict(action_args or {}),
            "metric_action_args": dict(action_args or {}),
            "measurement": measurement,
            "signal": float(measurement["terminal_rate"]),
        }
    if metric == "levels_completed_after_rollout":
        samples = len(rows)
        completed = [
            float(row.get("level_delta", 0) or 0)
            for row in rows
        ]
        mean_completed = sum(completed) / max(samples, 1)
        measurement = {
            "metric": metric,
            "measurable": True,
            "signal_source": signal_source,
            "levels_completed_after_rollout": mean_completed,
            "samples": samples,
            "changed_pixels": _mean_changed_pixels(rows),
        }
        return {
            "action": action_name,
            "action_args": dict(action_args or {}),
            "metric_action_args": dict(action_args or {}),
            "measurement": measurement,
            "signal": mean_completed,
        }
    measurement = _mean_grid_measurement(
        rows,
        metric=metric,
        action_args=dict(action_args or {}),
    )
    measurement["signal_source"] = signal_source
    return {
        "action": action_name,
        "action_args": dict(action_args or {}),
        "metric_action_args": dict(action_args or {}),
        "measurement": measurement,
        "signal": float(measurement.get("changed_pixels", 0) or 0),
    }


def _mean_grid_measurement(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    action_args: Mapping[str, Any],
) -> Dict[str, Any]:
    measured = [
        measure_required_observation(
            row.get("grid_t", []),
            row.get("grid_t1", []),
            required_observation=metric,
            action_args=action_args,
        )
        for row in rows
    ]
    samples = len(measured)
    if samples == 1:
        single = dict(measured[0])
        single["samples"] = 1
        return single
    numeric_keys = {
        key
        for item in measured
        for key, value in item.items()
        if isinstance(value, (int, float, bool))
    }
    averaged: Dict[str, Any] = {
        "metric": metric,
        "measurable": all(bool(item.get("measurable")) for item in measured),
        "samples": samples,
    }
    for key in sorted(numeric_keys):
        averaged[key] = sum(float(item.get(key, 0) or 0) for item in measured) / max(samples, 1)
    return averaged


def _mean_changed_pixels(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(_changed_pixels(row) for row in rows) / len(rows)


def _changed_pixels(row: Mapping[str, Any]) -> int:
    before = row.get("grid_t", []) or []
    after = row.get("grid_t1", []) or []
    changed = 0
    for before_row, after_row in zip(before, after):
        for before_value, after_value in zip(before_row, after_row):
            if before_value != after_value:
                changed += 1
    return changed


def should_escalate_metrics(experiment: Mapping[str, Any]) -> bool:
    """Escalate when primary metric is neutral but global pixels changed."""
    if str(experiment.get("metric_role", "primary")) != "primary":
        return False
    if int(experiment.get("support_events", 0) or 0) != 0:
        return False
    if int(experiment.get("contradiction_events", 0) or 0) != 0:
        return False
    target = experiment.get("observed_perturbation", {}) or {}
    baseline = experiment.get("observed_baseline", {}) or {}
    target_pixels = int(target.get("changed_pixels", 0) or 0)
    baseline_pixels = int(baseline.get("changed_pixels", 0) or 0)
    return target_pixels > baseline_pixels


def neutral_events_for_experiment(experiment: Mapping[str, Any]) -> int:
    support = int(experiment.get("support_events", 0) or 0)
    contradiction = int(experiment.get("contradiction_events", 0) or 0)
    run_count = int(experiment.get("controlled_experiments_run", 0) or 0)
    return 1 if run_count > 0 and support == 0 and contradiction == 0 else 0


def apply_metric_grounding_guard(
    experiment: Mapping[str, Any],
    *,
    metric: str,
) -> Dict[str, Any]:
    """Ensure only grounded metric measurements can create events."""
    row = dict(experiment)
    baseline = dict(row.get("observed_baseline", {}) or {})
    perturbation = dict(row.get("observed_perturbation", {}) or {})
    row["observed_baseline"] = baseline
    row["observed_perturbation"] = perturbation

    if metric == RAW_CHANGED_PIXELS_METRIC:
        if "changed_pixels" in baseline and "changed_pixels" in perturbation:
            for observation in (baseline, perturbation):
                observation["metric"] = RAW_CHANGED_PIXELS_METRIC
                observation["measurable"] = True
                observation["signal_source"] = "raw_changed_pixels"
                observation["changed"] = int(observation.get("changed_pixels", 0) or 0) > 0
            row.update(
                {
                    "metric_grounding_status": "grounded_raw_signal",
                    "signal_source": "raw_changed_pixels",
                    "diagnostic_only": False,
                    "event_counting_policy": "raw_changed_pixels_can_count_events",
                }
            )
            row["delta"] = _delta_with_counting_flags(
                row.get("delta", {}),
                signal_source="raw_changed_pixels",
                support_events=int(row.get("support_events", 0) or 0),
                contradiction_events=int(row.get("contradiction_events", 0) or 0),
            )
            return row
        return _mark_diagnostic_only(
            row,
            reason="raw_changed_pixels_missing",
        )

    if _observations_match_metric(baseline, perturbation, metric):
        row.update(
            {
                "metric_grounding_status": "grounded_metric_extractor",
                "signal_source": "metric_extractor",
                "diagnostic_only": False,
                "event_counting_policy": "grounded_metric_can_count_events",
            }
        )
        row["delta"] = _delta_with_counting_flags(
            row.get("delta", {}),
            signal_source="metric_extractor",
            support_events=int(row.get("support_events", 0) or 0),
            contradiction_events=int(row.get("contradiction_events", 0) or 0),
        )
        return row

    return _mark_diagnostic_only(
        row,
        reason="metric_measurement_not_grounded",
    )


def _observations_match_metric(
    baseline: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    metric: str,
) -> bool:
    return (
        bool(baseline.get("measurable"))
        and bool(perturbation.get("measurable"))
        and str(baseline.get("metric", "")) == metric
        and str(perturbation.get("metric", "")) == metric
    )


def _mark_diagnostic_only(
    row: Mapping[str, Any],
    *,
    reason: str,
) -> Dict[str, Any]:
    updated = dict(row)
    raw_support = int(updated.get("support_events", 0) or 0)
    raw_contradiction = int(updated.get("contradiction_events", 0) or 0)
    updated.update(
        {
            "metric_grounding_status": "diagnostic_only",
            "metric_grounding_reason": reason,
            "signal_source": "ungrounded_metric",
            "diagnostic_only": True,
            "event_counting_policy": "ungrounded_metric_cannot_count_events",
            "raw_support_events": raw_support,
            "raw_contradiction_events": raw_contradiction,
            "support_events": 0,
            "contradiction_events": 0,
        }
    )
    delta = dict(updated.get("delta", {}) or {})
    delta["raw_direction"] = str(delta.get("direction", ""))
    delta["direction"] = "diagnostic_only"
    delta["counted_as_support"] = False
    delta["counted_as_contradiction"] = False
    delta["signal_source"] = "ungrounded_metric"
    updated["delta"] = delta
    return updated


def _delta_with_counting_flags(
    delta: Mapping[str, Any],
    *,
    signal_source: str,
    support_events: int,
    contradiction_events: int,
) -> Dict[str, Any]:
    updated = dict(delta or {})
    updated["signal_source"] = signal_source
    updated["counted_as_support"] = support_events > 0
    updated["counted_as_contradiction"] = contradiction_events > 0
    return updated


def build_results_payload(
    *,
    m2_requests_path: str | Path,
    environments_dir: str | Path | None,
    offline_trace_dataset_path: str | Path,
    requests: Sequence[M3CandidateExperimentRequest],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    secondary_metrics: Sequence[str],
) -> Dict[str, Any]:
    support_events = sum(int(row.get("support_events", 0) or 0) for row in experiments)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in experiments
    )
    neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in experiments)
    diagnostic_only_experiments = [
        row for row in experiments if bool(row.get("diagnostic_only"))
    ]
    return {
        "config": {
            "m2_requests_path": str(m2_requests_path),
            "environments_dir": str(environments_dir or _env_dir()),
            "offline_trace_dataset_path": str(offline_trace_dataset_path),
            "secondary_metrics": list(secondary_metrics),
            "schema_version": "m3.m2_candidate_experiment_results.v1",
            "inputs_read": ["M2"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
        },
        "summary": {
            "m2_requests_ready_for_m3": len(requests),
            "m2_requests_executed": len(
                {str(row.get("request_id", "")) for row in experiments}
            ),
            "controlled_experiments_run": sum(
                int(row.get("controlled_experiments_run", 0) or 0)
                for row in experiments
            ),
            "primary_metric_experiments": len(
                [row for row in experiments if row.get("metric_role") == "primary"]
            ),
            "secondary_metric_experiments": len(
                [row for row in experiments if row.get("metric_role") == "secondary"]
            ),
            "metric_escalations": len(
                {
                    str(row.get("request_id", ""))
                    for row in experiments
                    if str(row.get("metric_role", "")) == "secondary"
                }
            ),
            "support_events": support_events,
            "contradiction_events": contradiction_events,
            "neutral_events": neutral_events,
            "diagnostic_only_experiments": len(diagnostic_only_experiments),
            "grounded_metric_experiments": len(
                [
                    row
                    for row in experiments
                    if str(row.get("metric_grounding_status", "")).startswith(
                        "grounded"
                    )
                ]
            ),
            "raw_changed_pixels_experiments": len(
                [
                    row
                    for row in experiments
                    if row.get("signal_source") == "raw_changed_pixels"
                ]
            ),
            "grounding_suppressed_support_events": sum(
                int(row.get("raw_support_events", 0) or 0)
                for row in diagnostic_only_experiments
            ),
            "grounding_suppressed_contradiction_events": sum(
                int(row.get("raw_contradiction_events", 0) or 0)
                for row in diagnostic_only_experiments
            ),
            "blocked_experiments": len(blocked),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M2_TRUTH_STATUS,
            "m2_artifacts_mutated_by_m3": False,
            "a32_remains_only_verdict_location": True,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "controlled_experiments": [dict(row) for row in experiments],
        "blocked_experiments": [dict(row) for row in blocked],
        "updated_candidate_records": updated_candidate_records(experiments),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def updated_candidate_records(
    experiments: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    by_key: dict[str, Dict[str, Any]] = {}
    for row in experiments:
        key = str(row.get("source_hypothesis_id") or row.get("hypothesis_key", ""))
        if not key:
            continue
        record = by_key.setdefault(
            key,
            {
                "key": key,
                "game_id": str(row.get("game_id", "")),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "contradictions": 0,
                "support_events": 0,
                "contradiction_events": 0,
                "neutral_events": 0,
                "controlled_experiments_run": 0,
                "controlled_test_required": True,
                "truth_status": M2_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
            },
        )
        record["support_events"] += int(row.get("support_events", 0) or 0)
        record["contradiction_events"] += int(row.get("contradiction_events", 0) or 0)
        record["neutral_events"] += int(row.get("neutral_events", 0) or 0)
        record["controlled_experiments_run"] += int(
            row.get("controlled_experiments_run", 0) or 0
        )
    return list(by_key.values())


def write_m2_candidate_experiment_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M3_M2_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M3.7 M2 candidate queue.")
    parser.add_argument(
        "--m2-requests",
        type=Path,
        default=DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--offline-trace-dataset",
        type=Path,
        default=DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_M3_M2_RESULTS_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_m2_candidate_experiment_queue(
        m2_requests_path=args.m2_requests,
        environments_dir=args.environments_dir,
        offline_trace_dataset_path=args.offline_trace_dataset,
        max_requests=args.max_requests,
    )
    write_m2_candidate_experiment_results(payload, args.out)
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
