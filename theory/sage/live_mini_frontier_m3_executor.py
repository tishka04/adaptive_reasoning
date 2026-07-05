"""SAGE.5d M3 execution for live mini-frontier requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from theory.m1.controlled_followup_experiment import controlled_delta
from theory.m1.polymorphic_a25_adapter import (
    _step_env_action,
    measure_required_observation,
)
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m2.schema import M2_TRUTH_STATUS
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .live_mini_frontier_generation import DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH
from .live_prefix_counterfactual_collector import (
    LivePrefixAction,
    select_live_action,
    state_signature_from_frame,
)


DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage5d_live_mini_frontier_m3_results.json"
)
SAGE5D_SCHEMA_VERSION = "sage.live_mini_frontier_m3_execution.v1"
SAGE5D_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5D"
SAGE5D_EXECUTED = "SAGE_LIVE_MINI_FRONTIER_M3_EXECUTED_CANDIDATE_ONLY"
SAGE5D_NO_EXACT_REPLAY = "SAGE_LIVE_MINI_FRONTIER_M3_NO_EXACT_REPLAY_CANDIDATE_ONLY"
DEFAULT_MIN_REQUESTS = 4
DEFAULT_MAX_REQUESTS = 8

EnvFactory = Callable[[str], Any]


def run_sage5d_live_mini_frontier_m3_execution(
    *,
    source_sage5c_path: str | Path = DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    min_requests: int = DEFAULT_MIN_REQUESTS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute a stratified subset of SAGE.5c M3 requests in live-prefix mode."""
    source = _load_json(source_sage5c_path)
    requests = select_sage5c_mini_frontier_requests(
        source,
        min_requests=min_requests,
        max_requests=max_requests,
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    experiments: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for request in requests:
        result = execute_live_prefix_mini_frontier_request(
            request,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        if result["execution_status"] == "EXECUTED":
            experiments.append(result)
        else:
            blocked.append(result)

    summary = summarize_sage5d_execution(
        selected_requests=requests,
        experiments=experiments,
        blocked=blocked,
        source=source,
        min_requests=min_requests,
        max_requests=max_requests,
    )
    payload = {
        "config": {
            "schema_version": SAGE5D_SCHEMA_VERSION,
            "source_sage5c_path": str(source_sage5c_path),
            "environments_dir": str(env_dir),
            "min_requests": int(min_requests),
            "max_requests": int(max_requests),
            "execution_mode": "live_prefix_replay_context",
            "benchmark_run": False,
            "inputs_read": ["SAGE.5c"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5c_summary": dict(source.get("summary", {}) or {}),
        "selected_requests": [dict(row) for row in requests],
        "controlled_experiments": experiments,
        "blocked_replay_events": blocked,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "support_events_counted_as_support": False,
        "mini_frontier_execution_counted_as_evidence": False,
        "generated_requests_counted_as_support": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5d_live_mini_frontier_m3_results(payload, output_path)
    return payload


def select_sage5c_mini_frontier_requests(
    source: Mapping[str, Any],
    *,
    min_requests: int = DEFAULT_MIN_REQUESTS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> tuple[Dict[str, Any], ...]:
    """Select a small stratified READY_FOR_M3 subset from SAGE.5c."""
    ready = [
        dict(row)
        for row in source.get("mini_frontier_m3_requests", []) or []
        if str(row.get("status", "")) == "READY_FOR_M3"
        and str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
        and str(row.get("context_state_origin", "")) == "sage5_live_prefix_frame_before"
    ]
    ready.sort(key=_request_sort_key)
    if not ready:
        return tuple()

    selected: List[Dict[str, Any]] = []

    def add_first(predicate: Callable[[Mapping[str, Any]], bool]) -> None:
        match = next((row for row in ready if predicate(row) and row not in selected), None)
        if match is not None and len(selected) < max_requests:
            selected.append(match)

    add_first(lambda row: str(row.get("hypothesis_family", "")) == "object_delta_candidate")
    add_first(
        lambda row: str(row.get("hypothesis_family", ""))
        == "local_patch_change_candidate"
    )
    add_first(lambda row: str(row.get("target_action", "")) == "ACTION5")
    add_first(lambda row: str(row.get("target_action", "")) == "ACTION6")

    for row in ready:
        if len(selected) >= max_requests:
            break
        if row not in selected:
            selected.append(row)

    target = min(max_requests, max(min_requests, len(selected)))
    for row in ready:
        if len(selected) >= target:
            break
        if row not in selected:
            selected.append(row)
    return tuple(selected[: max(0, int(max_requests))])


def execute_live_prefix_mini_frontier_request(
    request: Mapping[str, Any],
    *,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Replay a SAGE.5c prefix and compare target against a dynamic control."""
    control_action = _choose_control_action(request)
    if not control_action:
        return _blocked_result(request, "no_dynamic_control_available")

    target = _execute_request_arm(
        request,
        action_name=str(request.get("target_action", "")),
        action_args=dict(request.get("target_action_args", {}) or {}),
        arm="target",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if target["status"] != "EXECUTED":
        return _blocked_result(
            request,
            str(target.get("reason", "target_execution_blocked")),
            target_arm=target,
        )

    control = _execute_request_arm(
        request,
        action_name=control_action,
        action_args={},
        arm="dynamic_control",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if control["status"] != "EXECUTED":
        return _blocked_result(
            request,
            str(control.get("reason", "control_execution_blocked")),
            target_arm=target,
            control_arm=control,
        )

    delta = controlled_delta(
        control["measurement_for_delta"],
        target["measurement_for_delta"],
        predicted_metric=str(request.get("metric", "")),
    )
    support, contradiction = _support_contradiction_from_live_delta(delta)
    return {
        "execution_status": "EXECUTED",
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": str(request.get("game_id", "")),
        "execution_mode": "live_prefix_replay_context",
        "metric": str(request.get("metric", "")),
        "hypothesis_family": str(request.get("hypothesis_family", "")),
        "target_action": str(request.get("target_action", "")),
        "target_action_args": (
            dict(request.get("target_action_args", {}) or {})
            if request.get("target_action_args") is not None
            else None
        ),
        "control_action": control_action,
        "context_replay": list(request.get("context_replay", []) or []),
        "context_replay_args": [
            dict(item) for item in request.get("context_replay_args", []) or []
        ],
        "context_snapshot_hash": str(request.get("context_snapshot_hash", "")),
        "target_context_signature_verified": bool(
            target.get("context_snapshot_hash_verified", False)
        ),
        "control_context_signature_verified": bool(
            control.get("context_snapshot_hash_verified", False)
        ),
        "live_prefix_replay_exact": bool(
            target.get("context_snapshot_hash_verified", False)
            and control.get("context_snapshot_hash_verified", False)
        ),
        "target_measurement": dict(target["measurement"]),
        "control_measurement": dict(control["measurement"]),
        "target_signal": float(target["measurement_for_delta"]["local_changed_pixels"]),
        "control_signal": float(control["measurement_for_delta"]["local_changed_pixels"]),
        "controlled_delta": delta,
        "support_events": support,
        "contradiction_events": contradiction,
        "neutral_events": 1 if support == 0 and contradiction == 0 else 0,
        "blocked_replay_events": 0,
        "execution_failures": 0,
        "support": 0,
        "truth_status": SAGE5D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_support": False,
        "observation_counted_as_confirmation": False,
    }


def _execute_request_arm(
    request: Mapping[str, Any],
    *,
    action_name: str,
    action_args: Mapping[str, Any],
    arm: str,
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    game_id = str(request.get("game_id", ""))
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, environments_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {"status": "BLOCKED", "reason": f"env_setup_failed:{exc}", "arm": arm}

    for prefix_action in _prefix_actions_from_request(request):
        selected = select_live_action(
            env,
            prefix_action.name,
            action_args=prefix_action.action_args,
        )
        if selected is None:
            return {
                "status": "BLOCKED",
                "reason": f"prefix_action_unavailable:{prefix_action.name}",
                "arm": arm,
                "replay_state_signature": state_signature_from_frame(frame),
            }
        frame = _step_env_action(env, selected)

    replay_signature = state_signature_from_frame(frame)
    expected_signature = str(request.get("context_snapshot_hash", ""))
    if replay_signature != expected_signature:
        return {
            "status": "BLOCKED",
            "reason": "context_snapshot_hash_mismatch",
            "arm": arm,
            "replay_state_signature": replay_signature,
            "target_state_signature": expected_signature,
        }

    selected_action = select_live_action(env, action_name, action_args=action_args)
    if selected_action is None:
        return {
            "status": "BLOCKED",
            "reason": f"{arm}_action_unavailable:{action_name}",
            "arm": arm,
            "replay_state_signature": replay_signature,
        }

    before_frame = frame
    before = snapshot_frame(before_frame)
    after_frame = _step_env_action(env, selected_action)
    after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
    selected_args = dict(getattr(selected_action, "action_args", {}) or {})
    measurement_args = dict(action_args or selected_args)
    measurement = measure_required_observation(
        before.grid,
        after.grid,
        required_observation=str(request.get("metric", "")),
        action_args=measurement_args,
    )
    measurement_for_delta = _measurement_for_delta(
        measurement,
        metric=str(request.get("metric", "")),
    )
    return {
        "status": "EXECUTED",
        "arm": arm,
        "context_snapshot_hash_verified": True,
        "replay_state_signature": replay_signature,
        "before_signature": state_signature_from_frame(before_frame),
        "after_signature": state_signature_from_frame(after_frame),
        "action": action_name,
        "action_args": measurement_args,
        "measurement": {
            **measurement,
            "action": action_name,
            "action_args": measurement_args,
            "observed_signal": measurement_for_delta["local_changed_pixels"],
            "observed_signal_source": measurement_for_delta["observed_signal_source"],
        },
        "measurement_for_delta": measurement_for_delta,
    }


def summarize_sage5d_execution(
    *,
    selected_requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    min_requests: int,
    max_requests: int,
) -> Dict[str, Any]:
    executed = len(experiments)
    blocked_count = len(blocked)
    support_events = sum(int(row.get("support_events", 0) or 0) for row in experiments)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in experiments
    )
    replay_exact = sum(
        1 for row in experiments if bool(row.get("live_prefix_replay_exact", False))
    )
    families = _counts(row.get("hypothesis_family", "") for row in experiments)
    actions = _counts(row.get("target_action", "") for row in experiments)
    gate_passed = executed >= 1 and replay_exact >= 1
    return {
        "source_sage5c_outcome_status": str(source.get("outcome_status", "")),
        "source_sage5c_effective_requests_generated": int(
            (source.get("summary", {}) or {}).get("effective_requests_generated", 0) or 0
        ),
        "selected_requests": len(selected_requests),
        "min_requests": int(min_requests),
        "max_requests": int(max_requests),
        "requests_executed": executed,
        "requests_blocked": blocked_count,
        "live_prefix_replay_exact_events": replay_exact,
        "context_snapshot_hash_verified_events": replay_exact,
        "support_events": support_events,
        "contradiction_events": contradiction_events,
        "blocked_replay_events": blocked_count,
        "execution_failures": sum(
            1
            for row in blocked
            if "failed" in str(row.get("blocked_reason", "")).lower()
        ),
        "neutral_events": sum(int(row.get("neutral_events", 0) or 0) for row in experiments),
        "families_executed": families,
        "target_actions_executed": actions,
        "object_delta_candidate_executed": int(families.get("object_delta_candidate", 0))
        > 0,
        "local_patch_change_candidate_executed": int(
            families.get("local_patch_change_candidate", 0)
        )
        > 0,
        "action5_executed": int(actions.get("ACTION5", 0)) > 0,
        "action6_executed": int(actions.get("ACTION6", 0)) > 0,
        "gate_passed": gate_passed,
        "outcome_status": SAGE5D_EXECUTED if gate_passed else SAGE5D_NO_EXACT_REPLAY,
        "support": 0,
        "truth_status": SAGE5D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "policy_result_counted_as_confirmation": False,
        "support_events_counted_as_support": False,
        "mini_frontier_execution_counted_as_evidence": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def write_sage5d_live_mini_frontier_m3_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _choose_control_action(request: Mapping[str, Any]) -> str:
    controls = [str(item) for item in request.get("suggested_control_actions", []) or []]
    target = str(request.get("target_action", ""))
    return next((item for item in controls if item and item != target), "")


def _prefix_actions_from_request(request: Mapping[str, Any]) -> tuple[LivePrefixAction, ...]:
    actions = list(request.get("context_replay", []) or [])
    args = list(request.get("context_replay_args", []) or [])
    normalized: List[LivePrefixAction] = []
    for index, action in enumerate(actions):
        action_args = args[index] if index < len(args) and isinstance(args[index], Mapping) else {}
        normalized.append(LivePrefixAction(name=str(action), action_args=dict(action_args)))
    return tuple(normalized)


def _measurement_for_delta(
    measurement: Mapping[str, Any],
    *,
    metric: str,
) -> Dict[str, Any]:
    if metric == "local_patch_before_after":
        if bool(measurement.get("local_patch_available", False)):
            signal = int(measurement.get("local_changed_pixels", 0) or 0)
            source = "local_changed_pixels"
        else:
            signal = int(measurement.get("changed_pixels", 0) or 0)
            source = "changed_pixels_fallback_for_unparameterized_local_patch"
        return {
            **dict(measurement),
            "local_changed_pixels": signal,
            "observed_signal_source": source,
        }
    return {
        **dict(measurement),
        "local_changed_pixels": int(measurement.get("changed_pixels", 0) or 0),
        "observed_signal_source": "changed_pixels_default",
    }


def _support_contradiction_from_live_delta(delta: Mapping[str, Any]) -> tuple[int, int]:
    effect = float(delta.get("effect_size", 0.0) or 0.0)
    if effect > 0:
        return 1, 0
    if effect < 0:
        return 0, 1
    return 0, 0


def _blocked_result(
    request: Mapping[str, Any],
    reason: str,
    *,
    target_arm: Mapping[str, Any] | None = None,
    control_arm: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "execution_status": "BLOCKED",
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": str(request.get("game_id", "")),
        "execution_mode": "live_prefix_replay_context",
        "blocked_reason": reason,
        "target_arm": dict(target_arm or {}),
        "control_arm": dict(control_arm or {}),
        "support_events": 0,
        "contradiction_events": 0,
        "blocked_replay_events": 1,
        "execution_failures": 1 if "failed" in reason.lower() else 0,
        "support": 0,
        "truth_status": SAGE5D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_support": False,
    }


def _make_real_env(game_id: str, environments_dir: str | Path | None) -> Any:
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    return _make_env(game_id, env_dir)


def _request_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (
        int(row.get("source_step", 0) or 0),
        str(row.get("request_id", "")),
    )


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.5d live-prefix M3 execution for SAGE.5c requests.",
    )
    parser.add_argument("--source-sage5c", default=str(DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH))
    parser.add_argument("--min-requests", type=int, default=DEFAULT_MIN_REQUESTS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    args = parser.parse_args(argv)
    run_sage5d_live_mini_frontier_m3_execution(
        source_sage5c_path=args.source_sage5c,
        environments_dir=args.environments_dir,
        output_path=args.out,
        min_requests=args.min_requests,
        max_requests=args.max_requests,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
