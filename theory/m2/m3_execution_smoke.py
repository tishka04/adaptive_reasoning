"""M3-style contextual execution smoke for one M2 request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.a37.scope_conditioned_policy_rollout import state_signature

from .schema import M2_TRUTH_STATUS, M3CandidateExperimentRequest
from .testability_compiler import (
    DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    load_m3_requests_payload,
    requests_from_payload,
)


DEFAULT_M2_M3_EXECUTION_SMOKE_PATH = (
    Path("diagnostics") / "m2" / "m3_execution_smoke.json"
)


def select_execution_smoke_request(
    requests: Sequence[M3CandidateExperimentRequest],
    *,
    target_action: str = "ACTION4",
    context_replay: Sequence[str] = ("ACTION6", "ACTION3"),
    metric: str = "local_patch_before_after",
) -> M3CandidateExperimentRequest | None:
    desired_replay = tuple(context_replay)
    for request in requests:
        if (
            request.status == "READY_FOR_M3"
            and request.target_action == target_action
            and tuple(request.context_replay) == desired_replay
            and request.metric == metric
        ):
            return request
    return None


def run_m3_execution_smoke(
    *,
    m3_requests_path: str | Path = DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
) -> Dict[str, Any]:
    payload = load_m3_requests_payload(m3_requests_path)
    requests = requests_from_payload(payload)
    request = select_execution_smoke_request(requests)
    if request is None:
        return _blocked_payload(
            m3_requests_path=m3_requests_path,
            reason="no_matching_ready_m2_request",
        )
    target_context_signature = source_hypothesis_live_state_signature(
        payload,
        request.source_hypothesis_id,
    )
    try:
        experiment = execute_contextual_m2_request(
            request,
            environments_dir=environments_dir,
            target_context_signature=target_context_signature,
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return _blocked_payload(
            m3_requests_path=m3_requests_path,
            reason=f"contextual_execution_failed:{exc}",
            selected_request=request.to_dict(),
        )
    summary = {
        "m3_context_replay_exact": tuple(request.context_replay)
        == ("ACTION6", "ACTION3"),
        "m3_executed_target_vs_control": (
            int(experiment.get("controlled_experiments_run", 0) or 0) == 1
        ),
        "support_events": int(experiment.get("support_events", 0) or 0),
        "contradiction_events": int(experiment.get("contradiction_events", 0) or 0),
        "m2_truth_status_unchanged": M2_TRUTH_STATUS,
        "m2_artifacts_mutated_by_m3": False,
        "a32_remains_only_verdict_location": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    return {
        "config": {
            "m3_requests_path": str(m3_requests_path),
            "environments_dir": str(environments_dir or _env_dir()),
            "schema_version": "m2.m3_execution_smoke.v1",
        },
        "selected_request": request.to_dict(),
        "controlled_experiment": experiment,
        "summary": summary,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def execute_contextual_m2_request(
    request: M3CandidateExperimentRequest,
    *,
    environments_dir: str | Path | None = None,
    target_context_signature: str = "",
) -> Dict[str, Any]:
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    replay_action_args = find_replay_action_args_for_signature(
        request.game_id,
        context_replay=request.context_replay,
        target_signature=target_context_signature,
        environments_dir=env_dir,
    )
    control_action = choose_dynamic_control_action(
        request,
        env_dir,
        context_replay_args=replay_action_args,
    )
    target = execute_sequence_measurement(
        request.game_id,
        context_replay=request.context_replay,
        context_replay_args=replay_action_args,
        action_name=request.target_action,
        metric=request.metric,
        environments_dir=env_dir,
    )
    control = execute_sequence_measurement(
        request.game_id,
        context_replay=request.context_replay,
        context_replay_args=replay_action_args,
        action_name=control_action,
        metric=request.metric,
        environments_dir=env_dir,
        metric_action_args=target["metric_action_args"],
    )
    delta = controlled_delta(
        control["measurement"],
        target["measurement"],
        predicted_metric=request.metric,
    )
    support, contradiction = support_contradiction_from_delta(delta)
    return {
        "hypothesis_key": request.source_hypothesis_id,
        "game_id": request.game_id,
        "mechanic_family": "frontier_conditioned_candidate",
        "target_action": request.target_action,
        "control_action": control_action,
        "context_replay": list(request.context_replay),
        "context_replay_args": (
            [dict(item) for item in replay_action_args]
            if replay_action_args is not None
            else None
        ),
        "target_context_signature": target_context_signature,
        "baseline_sequence": ["RESET", *request.context_replay, control_action],
        "perturbation_sequence": ["RESET", *request.context_replay, request.target_action],
        "predicted_metric": request.metric,
        "observed_baseline": control["measurement"],
        "observed_perturbation": target["measurement"],
        "delta": delta,
        "baseline_signal": metric_signal(control["measurement"], request.metric),
        "perturbation_signal": metric_signal(target["measurement"], request.metric),
        "support_events": support,
        "contradiction_events": contradiction,
        "controlled_experiments_run": 1,
        "env_actions": int(control["env_actions"]) + int(target["env_actions"]),
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


def choose_dynamic_control_action(
    request: M3CandidateExperimentRequest,
    environments_dir: str | Path,
    *,
    context_replay_args: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    live = set(
        load_available_action_names_after_replay(
            request.game_id,
            context_replay=request.context_replay,
            context_replay_args=context_replay_args,
            environments_dir=environments_dir,
        )
    )
    for action in request.suggested_control_actions:
        if action in live and action != request.target_action:
            return action
    for action in sorted(live):
        if action not in {"RESET", request.target_action}:
            return action
    raise ValueError("no_dynamic_control_available")


def load_available_action_names_after_replay(
    game_id: str,
    *,
    context_replay: Sequence[str],
    context_replay_args: Sequence[Mapping[str, Any]] | None = None,
    environments_dir: str | Path,
) -> tuple[str, ...]:
    env = _make_env(game_id, environments_dir)
    _reset_env(env)
    replay_args = list(context_replay_args or [])
    for index, action_name in enumerate(context_replay):
        exact_args = replay_args[index] if index < len(replay_args) else None
        _execute_named_action(
            env,
            str(action_name),
            required_observation="",
            action_args=exact_args,
        )
    return tuple(
        sorted(
            {
                str(getattr(action, "name", ""))
                for action in _valid_actions(env)
                if str(getattr(action, "name", ""))
            }
        )
    )


def execute_sequence_measurement(
    game_id: str,
    *,
    context_replay: Sequence[str],
    context_replay_args: Sequence[Mapping[str, Any]] | None = None,
    action_name: str,
    metric: str,
    environments_dir: str | Path,
    metric_action_args: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    env = _make_env(game_id, environments_dir)
    frame = _reset_env(env)
    env_actions = 0
    replay_args = list(context_replay_args or [])
    for index, replay_action in enumerate(context_replay):
        exact_args = replay_args[index] if index < len(replay_args) else None
        frame = _execute_named_action(
            env,
            str(replay_action),
            required_observation="",
            action_args=exact_args,
        )
        env_actions += 1
    before = snapshot_frame(frame)
    selected = _select_concrete_action(
        _valid_actions(env),
        action_name=action_name,
        required_observation=metric if metric_action_args is None else "",
    )
    if selected is None and metric_action_args is None:
        selected = _select_concrete_action(
            _valid_actions(env),
            action_name=action_name,
            required_observation="",
        )
    if selected is None:
        raise ValueError(f"no_concrete_action_available:{action_name}")
    after_frame = _step_env_action(env, selected)
    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    action_args = dict(getattr(selected, "action_args", {}) or {})
    measurement_args = dict(metric_action_args or action_args)
    measurement = measure_required_observation(
        before.grid,
        after.grid,
        required_observation=metric,
        action_args=measurement_args,
    )
    return {
        "action": action_name,
        "context_replay": list(context_replay),
        "action_args": action_args,
        "metric_action_args": measurement_args,
        "measurement": measurement,
        "env_actions": env_actions + 1,
    }


def source_hypothesis_live_state_signature(
    m3_payload: Mapping[str, Any],
    source_hypothesis_id: str,
) -> str:
    hypothesis_path = str(
        (m3_payload.get("config") or {}).get("source_hypothesis_path", "")
    )
    if not hypothesis_path or not Path(hypothesis_path).exists():
        return ""
    payload = json.loads(Path(hypothesis_path).read_text(encoding="utf-8"))
    for batch in payload.get("hypothesis_batches", []) or []:
        for hypothesis in batch.get("candidate_hypotheses", []) or []:
            if str(hypothesis.get("hypothesis_id", "")) != str(source_hypothesis_id):
                continue
            snapshot = hypothesis.get("context_snapshot", {}) or {}
            return str(snapshot.get("live_state_signature", ""))
    return ""


def find_replay_action_args_for_signature(
    game_id: str,
    *,
    context_replay: Sequence[str],
    target_signature: str,
    environments_dir: str | Path,
    max_candidates_per_action: int = 30,
) -> tuple[Dict[str, Any], ...] | None:
    if not target_signature or not context_replay:
        return None

    def search(prefix_args: tuple[Dict[str, Any], ...]) -> tuple[Dict[str, Any], ...] | None:
        env = _make_env(game_id, environments_dir)
        frame = _reset_env(env)
        for index, action_args in enumerate(prefix_args):
            frame = _execute_named_action(
                env,
                str(context_replay[index]),
                required_observation="",
                action_args=action_args,
            )
        if len(prefix_args) == len(context_replay):
            snapshot = snapshot_frame(frame)
            signature = state_signature(
                snapshot.grid,
                snapshot.levels_completed,
                snapshot.game_state,
            )
            return prefix_args if signature == target_signature else None
        next_action = str(context_replay[len(prefix_args)])
        for candidate in _matching_actions(env, next_action)[:max_candidates_per_action]:
            action_args = dict(getattr(candidate, "action_args", {}) or {})
            result = search((*prefix_args, action_args))
            if result is not None:
                return result
        return None

    return search(tuple())


def write_m3_execution_smoke(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_M3_EXECUTION_SMOKE_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_env(game_id: str, environments_dir: str | Path) -> Any:
    from arc_agi import Arcade, OperationMode

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    return arc.make(game_id)


def _reset_env(env: Any) -> Any:
    from arcengine import GameAction

    return env.step(GameAction.RESET)


def _execute_named_action(
    env: Any,
    action_name: str,
    *,
    required_observation: str,
    action_args: Mapping[str, Any] | None = None,
) -> Any:
    if action_args is not None:
        selected = _select_action_with_args(env, action_name, action_args)
    else:
        selected = _select_concrete_action(
            _valid_actions(env),
            action_name=action_name,
            required_observation=required_observation,
        )
    if selected is None:
        raise ValueError(f"no_concrete_action_available:{action_name}")
    return _step_env_action(env, selected)


def _matching_actions(env: Any, action_name: str) -> list[Any]:
    return [
        action
        for action in _valid_actions(env)
        if str(getattr(action, "name", "")) == str(action_name)
    ]


def _select_action_with_args(
    env: Any,
    action_name: str,
    action_args: Mapping[str, Any],
) -> Any | None:
    expected = {str(key): str(value) for key, value in dict(action_args).items()}
    for action in _matching_actions(env, action_name):
        actual = {
            str(key): str(value)
            for key, value in dict(getattr(action, "action_args", {}) or {}).items()
        }
        if actual == expected:
            return action
    return None


def _blocked_payload(
    *,
    m3_requests_path: str | Path,
    reason: str,
    selected_request: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "config": {
            "m3_requests_path": str(m3_requests_path),
            "schema_version": "m2.m3_execution_smoke.v1",
        },
        "selected_request": dict(selected_request or {}),
        "controlled_experiment": None,
        "summary": {
            "m3_context_replay_exact": False,
            "m3_executed_target_vs_control": False,
            "blocked_reason": reason,
            "support_events": 0,
            "contradiction_events": 0,
            "m2_truth_status_unchanged": M2_TRUTH_STATUS,
            "m2_artifacts_mutated_by_m3": False,
            "a32_remains_only_verdict_location": True,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one contextual M2 request smoke.")
    parser.add_argument(
        "--m3-requests",
        type=Path,
        default=DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_M2_M3_EXECUTION_SMOKE_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_m3_execution_smoke(
        m3_requests_path=args.m3_requests,
        environments_dir=args.environments_dir,
    )
    write_m3_execution_smoke(payload, args.out)
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
