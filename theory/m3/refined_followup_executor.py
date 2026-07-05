"""M3.10 executor for refined follow-up experiment requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.m1.controlled_followup_experiment import (
    controlled_delta,
    metric_signal,
    support_contradiction_from_delta,
)
from theory.m1.polymorphic_a25_adapter import (
    _select_concrete_action,
    _step_env_action,
    measure_required_observation,
)
from theory.m2.m3_execution_smoke import (
    _execute_named_action,
    _make_env,
    _reset_env,
)
from theory.non_ar25_active_micro_run import (
    _configure_offline_env,
    _env_dir,
    _valid_actions,
)
from theory.real_env_option_adapter import snapshot_frame

from .m2_candidate_experiment_runner import (
    apply_metric_grounding_guard,
    neutral_events_for_experiment,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_planner import (
    DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    READY_FOR_M3_FOLLOWUP,
)


DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "refined_followup_experiment_results.json"
)
SAME_ARGS_AS_PREVIOUS_SKILL_OCCURRENCE = "same_args_as_previous_skill_occurrence"
TARGET_ARGS_EXPLICIT = "explicit_target_action_args"
TARGET_ARGS_DYNAMIC = "dynamic_target_action_selection"


def run_refined_followup_execution(
    *,
    followup_requests_path: str | Path = DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    target_action_arg_policy: str = SAME_ARGS_AS_PREVIOUS_SKILL_OCCURRENCE,
) -> Dict[str, Any]:
    payload = _load_json(followup_requests_path)
    requests = [
        dict(request)
        for request in payload.get("followup_experiment_requests", []) or []
        if str(request.get("status", "")) == READY_FOR_M3_FOLLOWUP
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    experiments: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    for request in requests:
        resolved_args, resolved_policy = resolve_target_action_args(
            request,
            policy=target_action_arg_policy,
        )
        controls = available_followup_controls(
            request,
            environments_dir=env_dir,
        )
        if not controls:
            blocked.append(
                blocked_followup_row(
                    request,
                    reason="no_dynamic_control_available",
                    target_action_args=resolved_args,
                    target_action_arg_policy=resolved_policy,
                )
            )
            continue
        for metric in request.get("metrics", []) or []:
            for control_action in controls:
                try:
                    experiment = execute_metric_followup_experiment(
                        request,
                        metric=str(metric),
                        control_action=str(control_action),
                        target_action_args=resolved_args,
                        target_action_arg_policy=resolved_policy,
                        environments_dir=env_dir,
                    )
                except Exception as exc:  # pragma: no cover - integration failure path
                    blocked.append(
                        blocked_followup_row(
                            request,
                            reason=f"execution_failed:{exc}",
                            metric=str(metric),
                            control_action=str(control_action),
                            target_action_args=resolved_args,
                            target_action_arg_policy=resolved_policy,
                        )
                    )
                    continue
                experiments.append(experiment)

    return build_followup_results_payload(
        followup_requests_path=followup_requests_path,
        environments_dir=env_dir,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
        target_action_arg_policy=target_action_arg_policy,
    )


def resolve_target_action_args(
    request: Mapping[str, Any],
    *,
    policy: str = SAME_ARGS_AS_PREVIOUS_SKILL_OCCURRENCE,
) -> Tuple[Dict[str, Any] | None, str]:
    explicit = request.get("target_action_args")
    if isinstance(explicit, Mapping):
        return dict(explicit), TARGET_ARGS_EXPLICIT
    if policy != SAME_ARGS_AS_PREVIOUS_SKILL_OCCURRENCE:
        return None, TARGET_ARGS_DYNAMIC
    target = str(request.get("target_action", ""))
    replay = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index in range(len(replay) - 1, -1, -1):
        if str(replay[index]) != target:
            continue
        if index < len(replay_args) and isinstance(replay_args[index], Mapping):
            return dict(replay_args[index]), SAME_ARGS_AS_PREVIOUS_SKILL_OCCURRENCE
    return None, TARGET_ARGS_DYNAMIC


def available_followup_controls(
    request: Mapping[str, Any],
    *,
    environments_dir: str | Path,
) -> Tuple[str, ...]:
    env = _make_env(str(request.get("game_id", "")), environments_dir)
    _reset_env(env)
    replay = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index, action_name in enumerate(replay):
        action_args = replay_args[index] if index < len(replay_args) else None
        _execute_named_action(
            env,
            str(action_name),
            required_observation="",
            action_args=action_args if isinstance(action_args, Mapping) else None,
        )
    live = {
        str(getattr(action, "name", ""))
        for action in _valid_actions(env)
        if str(getattr(action, "name", ""))
    }
    target = str(request.get("target_action", ""))
    controls: list[str] = []
    for action in request.get("suggested_control_actions", []) or []:
        text = str(action)
        if text and text in live and text != target and text not in controls:
            controls.append(text)
    return tuple(controls)


def execute_metric_followup_experiment(
    request: Mapping[str, Any],
    *,
    metric: str,
    control_action: str,
    target_action_args: Mapping[str, Any] | None,
    target_action_arg_policy: str,
    environments_dir: str | Path,
) -> Dict[str, Any]:
    control_action_args = previous_action_args(
        request,
        action_name=control_action,
    )
    target = execute_exact_action_measurement(
        request,
        action_name=str(request.get("target_action", "")),
        action_args=target_action_args,
        metric=metric,
        environments_dir=environments_dir,
        measurement_action_args=target_action_args,
    )
    control = execute_exact_action_measurement(
        request,
        action_name=control_action,
        action_args=control_action_args,
        metric=metric,
        environments_dir=environments_dir,
        measurement_action_args=target_action_args,
    )
    delta = controlled_delta(
        control["measurement"],
        target["measurement"],
        predicted_metric=metric,
    )
    support, contradiction = support_contradiction_from_delta(delta)
    row: Dict[str, Any] = {
        "request_id": str(request.get("request_id", "")),
        "source_refined_hypothesis_id": str(
            request.get("source_refined_hypothesis_id", "")
        ),
        "source_hypothesis_ids": list(request.get("source_hypothesis_ids", []) or []),
        "game_id": str(request.get("game_id", "")),
        "hypothesis_tested": str(request.get("hypothesis_tested", "")),
        "context_replay": list(request.get("context_replay", []) or []),
        "context_replay_args": _context_args_list(
            request.get("context_replay_args")
        ),
        "target_action": str(request.get("target_action", "")),
        "target_action_args": (
            dict(target_action_args) if target_action_args is not None else None
        ),
        "target_action_arg_policy": target_action_arg_policy,
        "target_action_args_resolved": target_action_args is not None,
        "control_action": control_action,
        "control_action_args": (
            dict(control_action_args) if control_action_args is not None else None
        ),
        "metric": metric,
        "predicted_metric": metric,
        "observed_baseline": control["measurement"],
        "observed_perturbation": target["measurement"],
        "delta": delta,
        "baseline_signal": metric_signal(control["measurement"], metric),
        "perturbation_signal": metric_signal(target["measurement"], metric),
        "support_events": support,
        "contradiction_events": contradiction,
        "controlled_experiments_run": 1,
        "env_actions": int(target["env_actions"]) + int(control["env_actions"]),
        "exact_replay": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }
    row = apply_metric_grounding_guard(row, metric=metric)
    row["neutral_events"] = neutral_events_for_experiment(row)
    return row


def execute_exact_action_measurement(
    request: Mapping[str, Any],
    *,
    action_name: str,
    action_args: Mapping[str, Any] | None,
    metric: str,
    environments_dir: str | Path,
    measurement_action_args: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    env = _make_env(str(request.get("game_id", "")), environments_dir)
    frame = _reset_env(env)
    env_actions = 0
    replay = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index, replay_action in enumerate(replay):
        replay_action_args = replay_args[index] if index < len(replay_args) else None
        frame = _execute_named_action(
            env,
            str(replay_action),
            required_observation="",
            action_args=(
                replay_action_args if isinstance(replay_action_args, Mapping) else None
            ),
        )
        env_actions += 1
    before = snapshot_frame(frame)
    if action_args is not None:
        after_frame = _execute_named_action(
            env,
            action_name,
            required_observation="",
            action_args=action_args,
        )
        selected_action_args = dict(action_args)
    else:
        selected = _select_concrete_action(
            _valid_actions(env),
            action_name=action_name,
            required_observation=metric,
        )
        if selected is None:
            selected = _select_concrete_action(
                _valid_actions(env),
                action_name=action_name,
                required_observation="",
            )
        if selected is None:
            raise ValueError(f"no_concrete_action_available:{action_name}")
        selected_action_args = dict(getattr(selected, "action_args", {}) or {})
        after_frame = _step_env_action(env, selected)
    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    measurement_args = dict(measurement_action_args or selected_action_args)
    measurement = measure_required_observation(
        before.grid,
        after.grid,
        required_observation=metric,
        action_args=measurement_args,
    )
    return {
        "action": action_name,
        "action_args": selected_action_args,
        "measurement_action_args": measurement_args,
        "measurement": measurement,
        "env_actions": env_actions + 1,
    }


def previous_action_args(
    request: Mapping[str, Any],
    *,
    action_name: str,
) -> Dict[str, Any] | None:
    replay = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index in range(len(replay) - 1, -1, -1):
        if str(replay[index]) != action_name:
            continue
        if index < len(replay_args) and isinstance(replay_args[index], Mapping):
            return dict(replay_args[index])
    return None


def blocked_followup_row(
    request: Mapping[str, Any],
    *,
    reason: str,
    target_action_args: Mapping[str, Any] | None,
    target_action_arg_policy: str,
    metric: str = "",
    control_action: str = "",
) -> Dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "source_refined_hypothesis_id": str(
            request.get("source_refined_hypothesis_id", "")
        ),
        "game_id": str(request.get("game_id", "")),
        "metric": metric,
        "control_action": control_action,
        "target_action": str(request.get("target_action", "")),
        "target_action_args": (
            dict(target_action_args) if target_action_args is not None else None
        ),
        "target_action_arg_policy": target_action_arg_policy,
        "blocked_reason": reason,
        "status": "BLOCKED_NOT_EXECUTED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0,
        "controlled_experiments_run": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_followup_results_payload(
    *,
    followup_requests_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    target_action_arg_policy: str,
) -> Dict[str, Any]:
    support_events = sum(int(row.get("support_events", 0) or 0) for row in experiments)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in experiments
    )
    neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in experiments)
    return {
        "config": {
            "followup_requests_path": str(followup_requests_path),
            "environments_dir": str(environments_dir),
            "schema_version": "m3.refined_followup_results.v1",
            "target_action_arg_policy": target_action_arg_policy,
            "inputs_read": ["M3.9"],
            "artifacts_not_modified": ["M2", "M3.8", "M3.9", "A32", "A33"],
        },
        "summary": {
            "followup_requests_consumed": len(requests),
            "followup_requests_executed": len(
                {str(row.get("request_id", "")) for row in experiments}
            ),
            "controlled_experiments_run": sum(
                int(row.get("controlled_experiments_run", 0) or 0)
                for row in experiments
            ),
            "metrics_executed": len(
                {str(row.get("metric", "")) for row in experiments}
            ),
            "controls_executed": len(
                {str(row.get("control_action", "")) for row in experiments}
            ),
            "target_action_args_resolved": all(
                bool(row.get("target_action_args_resolved")) for row in experiments
            )
            if experiments
            else False,
            "target_action_arg_policy": target_action_arg_policy,
            "support_events": support_events,
            "contradiction_events": contradiction_events,
            "neutral_events": neutral_events,
            "diagnostic_only_experiments": len(
                [row for row in experiments if bool(row.get("diagnostic_only"))]
            ),
            "blocked_experiments": len(blocked),
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "controlled_test_required": True,
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a32_remains_only_verdict_location": True,
        },
        "controlled_experiments": [dict(row) for row in experiments],
        "blocked_experiments": [dict(row) for row in blocked],
        "updated_candidate_records": updated_candidate_records(experiments),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
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
        key = str(row.get("source_refined_hypothesis_id", ""))
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
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
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


def write_refined_followup_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_args_list(raw: Any) -> list[Dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute M3.10 refined follow-up requests.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--target-action-arg-policy",
        default=SAME_ARGS_AS_PREVIOUS_SKILL_OCCURRENCE,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_refined_followup_execution(
        followup_requests_path=args.requests,
        environments_dir=args.environments_dir,
        target_action_arg_policy=args.target_action_arg_policy,
    )
    write_refined_followup_results(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
