"""M3.7d replication for ARC-LeWM terminal-risk candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m2.arc_lewm_signal_extractor import DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH
from theory.m2.normalizer import default_controls_for_action
from theory.m2.schema import (
    FalsificationCriterion,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_READY_FOR_M3_STATUS,
    M2_TO_M3_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    M3CandidateExperimentRequest,
)
from theory.m2.testability_compiler import write_m3_requests_payload

from .m2_candidate_experiment_runner import (
    DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    run_m2_candidate_experiment_queue,
    source_transition_id_for_row,
)


DEFAULT_ARC_LEWM_TERMINAL_RISK_REPLICATION_REQUESTS_PATH = (
    Path("diagnostics") / "m3" / "arc_lewm_terminal_risk_replication_requests.json"
)
DEFAULT_ARC_LEWM_TERMINAL_RISK_REPLICATION_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "arc_lewm_terminal_risk_replication_results.json"
)
REPLICATION_SCHEMA_VERSION = "m3.arc_lewm_terminal_risk_replication.v1"


def run_arc_lewm_terminal_risk_replication(
    *,
    signal_report_path: str | Path = DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH,
    offline_trace_dataset_path: str | Path = DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    requests_output_path: str | Path = DEFAULT_ARC_LEWM_TERMINAL_RISK_REPLICATION_REQUESTS_PATH,
    output_path: str | Path | None = None,
    max_requests: int = 12,
) -> Dict[str, Any]:
    signal_report = _load_json(signal_report_path)
    requests_payload = build_terminal_risk_replication_requests_payload(
        signal_report,
        max_requests=max_requests,
        signal_report_path=str(signal_report_path),
    )
    write_m3_requests_payload(requests_payload, requests_output_path)
    runner_payload = run_m2_candidate_experiment_queue(
        m2_requests_path=requests_output_path,
        offline_trace_dataset_path=offline_trace_dataset_path,
        secondary_metrics=(),
    )
    payload = build_replication_result_payload(
        signal_report_path=str(signal_report_path),
        requests_path=str(requests_output_path),
        offline_trace_dataset_path=str(offline_trace_dataset_path),
        requests_payload=requests_payload,
        runner_payload=runner_payload,
    )
    if output_path is not None:
        write_arc_lewm_terminal_risk_replication_results(payload, output_path)
    return payload


def build_terminal_risk_replication_requests_payload(
    signal_report: Mapping[str, Any],
    *,
    max_requests: int = 12,
    signal_report_path: str = "",
) -> Dict[str, Any]:
    rows = select_terminal_risk_signal_rows(signal_report, max_requests=max_requests)
    requests = [
        terminal_signal_row_to_request(row, index=index)
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "config": {
            "schema_version": M2_TO_M3_SCHEMA_VERSION,
            "replication_schema_version": REPLICATION_SCHEMA_VERSION,
            "source_signal_report_path": signal_report_path,
            "source_signal_family": "terminal_like_latent_neighborhoods",
            "planner": "arc_lewm_terminal_risk_replication_planner",
        },
        "experiment_requests": [request.to_dict() for request in requests],
        "summary": {
            "source_terminal_rows_selected": len(rows),
            "experiment_requests": len(requests),
            "ready_for_m3": len(requests),
            "blocked_not_testable": 0,
            "distinct_source_transition_ids": len(
                {request.source_transition_id for request in requests}
            ),
            "distinct_source_episodes": len(
                {request.source_episode_id for request in requests}
            ),
            "distinct_games": len({request.game_id for request in requests}),
            "distinct_target_actions": len({request.target_action for request in requests}),
            "truth_status": M2_TRUTH_STATUS,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "world_model_score_counted_as_support": False,
            "world_model_counted_as_evidence": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
    }


def select_terminal_risk_signal_rows(
    signal_report: Mapping[str, Any],
    *,
    max_requests: int,
) -> tuple[Dict[str, Any], ...]:
    terminal = (
        signal_report.get("signals", {})
        or {}
    ).get("terminal_like_latent_neighborhoods", {}) or {}
    rows = terminal.get("highest_surprise_terminal_transitions", []) or []
    selected: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        action = str(row.get("action", ""))
        if not action or action == "RESET":
            continue
        game_id = str(row.get("game_id", ""))
        episode_id = str(row.get("episode_id", ""))
        if not game_id or game_id == "unknown_game" or not episode_id:
            continue
        available = {str(item) for item in row.get("available_actions_t", []) or []}
        if available and action not in available:
            continue
        if action == "ACTION6" and not dict(row.get("action_args", {}) or {}):
            continue
        transition_id = source_transition_id_for_row(row)
        if transition_id in seen:
            continue
        seen.add(transition_id)
        row["source_transition_id"] = transition_id
        selected.append(row)
        if len(selected) >= max(0, int(max_requests)):
            break
    return tuple(selected)


def terminal_signal_row_to_request(
    row: Mapping[str, Any],
    *,
    index: int,
) -> M3CandidateExperimentRequest:
    action = str(row.get("action", ""))
    available_actions = tuple(str(item) for item in row.get("available_actions_t", []) or [])
    source_transition_id = str(
        row.get("source_transition_id") or source_transition_id_for_row(row)
    )
    controls = default_controls_for_action(action, available_actions)
    return M3CandidateExperimentRequest(
        request_id=f"m2m3::m2_14_lewm::terminal_risk_replication::{index:03d}",
        source_hypothesis_id=f"m2_14_lewm::terminal_risk_replication::{index:03d}",
        game_id=str(row.get("game_id", "")),
        context_replay=(),
        context_replay_args=None,
        context_snapshot_hash=source_transition_id,
        target_action=action,
        target_action_args=(
            dict(row.get("action_args", {}) or {})
            if dict(row.get("action_args", {}) or {})
            else None
        ),
        suggested_control_actions=tuple(
            control for control in controls if control != action
        ),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric="terminal_state_after_rollout",
        expected_signal="target_action_terminal_rate_exceeds_matched_dynamic_controls",
        falsification_criterion=FalsificationCriterion(
            metric="terminal_state_after_rollout",
            support_condition="target_action_terminal_rate > best_control_terminal_rate",
            failure_condition="target_action_terminal_rate <= best_control_terminal_rate",
            minimum_effect_size=1,
        ),
        status=M2_READY_FOR_M3_STATUS,
        source_episode_id=str(row.get("episode_id", "")),
        source_step=int(row.get("step", 0) or 0),
        source_transition_id=source_transition_id,
        context_state_origin="human_trace_frame_before",
        replayability="OFFLINE_TRACE_CONTEXT_ONLY",
        blocking_reason=None,
        truth_status=M2_TRUTH_STATUS,
        support=0,
        controlled_test_required=True,
        revision_performed=False,
        wrong_confirmations=0,
    )


def build_replication_result_payload(
    *,
    signal_report_path: str,
    requests_path: str,
    offline_trace_dataset_path: str,
    requests_payload: Mapping[str, Any],
    runner_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    experiments = [
        dict(row) for row in runner_payload.get("controlled_experiments", []) or []
    ]
    blocked = [dict(row) for row in runner_payload.get("blocked_experiments", []) or []]
    summary = summarize_replication_experiments(experiments, blocked)
    return {
        "config": {
            "schema_version": REPLICATION_SCHEMA_VERSION,
            "source_signal_report_path": signal_report_path,
            "replication_requests_path": requests_path,
            "offline_trace_dataset_path": offline_trace_dataset_path,
            "inputs_read": ["M2", "M3.7c"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
        },
        "replication_requests": list(requests_payload.get("experiment_requests", []) or []),
        "controlled_experiments": experiments,
        "blocked_experiments": blocked,
        "runner_summary": dict(runner_payload.get("summary", {}) or {}),
        "summary": {
            **summary,
            "m2_requests_ready_for_m3": int(
                (runner_payload.get("summary", {}) or {}).get(
                    "m2_requests_ready_for_m3", 0
                )
                or 0
            ),
            "m2_requests_executed": int(
                (runner_payload.get("summary", {}) or {}).get(
                    "m2_requests_executed", 0
                )
                or 0
            ),
            "support": 0,
            "truth_status": M2_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "a32_remains_only_verdict_location": True,
        },
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def summarize_replication_experiments(
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    support_rows = [
        row for row in experiments if int(row.get("support_events", 0) or 0) > 0
    ]
    contradiction_rows = [
        row for row in experiments if int(row.get("contradiction_events", 0) or 0) > 0
    ]
    neutral_rows = [
        row
        for row in experiments
        if int(row.get("support_events", 0) or 0) == 0
        and int(row.get("contradiction_events", 0) or 0) == 0
    ]
    support_transition_ids = [
        str(row.get("source_transition_id", "")) for row in support_rows
    ]
    support_episode_ids = [
        str(row.get("source_episode_id", "")) for row in support_rows
    ]
    context_keys = [
        _context_key(row)
        for row in support_rows
    ]
    duplicate_source_support_events = len(support_transition_ids) - len(
        set(support_transition_ids)
    )
    context_reused_support_events = len(context_keys) - len(set(context_keys))
    control_policies = Counter(
        str(row.get("matched_control_policy", "")) for row in experiments
    )
    return {
        "controlled_experiments_run": sum(
            int(row.get("controlled_experiments_run", 0) or 0)
            for row in experiments
        ),
        "replication_experiments": len(experiments),
        "blocked_experiments": len(blocked),
        "support_events": sum(
            int(row.get("support_events", 0) or 0) for row in experiments
        ),
        "contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in experiments
        ),
        "neutral_events": len(neutral_rows),
        "supporting_source_transition_ids": len(set(support_transition_ids)),
        "supporting_source_episodes": len(set(support_episode_ids)),
        "supporting_games": len({str(row.get("game_id", "")) for row in support_rows}),
        "supporting_target_actions": len(
            {str(row.get("target_action", "")) for row in support_rows}
        ),
        "independent_source_support_events": len(set(support_transition_ids)),
        "duplicate_source_support_events": duplicate_source_support_events,
        "unique_context_support_events": len(set(context_keys)),
        "context_reused_support_events": context_reused_support_events,
        "contradicting_source_transition_ids": len(
            {str(row.get("source_transition_id", "")) for row in contradiction_rows}
        ),
        "target_trace_samples_total": sum(
            int(row.get("target_trace_samples", 0) or 0) for row in experiments
        ),
        "matched_control_samples_total": sum(
            int(row.get("matched_control_samples", 0) or 0) for row in experiments
        ),
        "same_game_same_available_actions_controls": int(
            control_policies.get("same_game_same_available_actions", 0)
        ),
        "all_controls_same_game_same_available_actions": all(
            str(row.get("matched_control_policy", ""))
            == "same_game_same_available_actions"
            for row in experiments
        )
        if experiments
        else False,
        "replication_breadth_low": len(set(support_transition_ids)) < 2
        or len(set(support_episode_ids)) < 2,
        "grounded_metric_experiments": len(
            [
                row
                for row in experiments
                if str(row.get("metric_grounding_status", "")).startswith("grounded")
            ]
        ),
    }


def write_arc_lewm_terminal_risk_replication_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_ARC_LEWM_TERMINAL_RISK_REPLICATION_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_key(row: Mapping[str, Any]) -> str:
    actions = ",".join(str(item) for item in row.get("dynamic_available_actions", []) or [])
    return (
        f"{row.get('game_id', '')}|{row.get('target_action', '')}|"
        f"{row.get('control_action', '')}|{actions}"
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate ARC-LeWM terminal-risk signals through M3 offline traces.",
    )
    parser.add_argument(
        "--signal-report",
        type=Path,
        default=DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--offline-trace-dataset",
        type=Path,
        default=DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    )
    parser.add_argument(
        "--requests-out",
        type=Path,
        default=DEFAULT_ARC_LEWM_TERMINAL_RISK_REPLICATION_REQUESTS_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ARC_LEWM_TERMINAL_RISK_REPLICATION_RESULTS_PATH,
    )
    parser.add_argument("--max-requests", type=int, default=12)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_arc_lewm_terminal_risk_replication(
        signal_report_path=args.signal_report,
        offline_trace_dataset_path=args.offline_trace_dataset,
        requests_output_path=args.requests_out,
        output_path=args.out,
        max_requests=args.max_requests,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "requests_path": str(args.requests_out),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
