"""M3 runner for M2.15 fused LLM + ARC-LeWM candidate requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m2.schema import M2_TRUTH_STATUS
from theory.m2.testability_compiler import load_m3_requests_payload, requests_from_payload

from .m2_candidate_experiment_runner import (
    DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    run_m2_candidate_experiment_queue,
)


DEFAULT_FUSED_M2_REQUESTS_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_m3_candidate_requests.json"
)
DEFAULT_M3_7D_REPLICATION_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "arc_lewm_terminal_risk_replication_results.json"
)
DEFAULT_FUSED_M3_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "fused_llm_wm_experiment_results.json"
)
FUSED_M3_RESULTS_SCHEMA_VERSION = "m3.fused_llm_wm_experiment_results.v1"


def run_fused_llm_wm_experiment_runner(
    *,
    m2_requests_path: str | Path = DEFAULT_FUSED_M2_REQUESTS_PATH,
    m3_7d_results_path: str | Path = DEFAULT_M3_7D_REPLICATION_RESULTS_PATH,
    offline_trace_dataset_path: str | Path = DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    request_payload = load_m3_requests_payload(m2_requests_path)
    requests = requests_from_payload(request_payload)
    skipped_blocked = [
        request.to_dict()
        for request in requests
        if request.status == "BLOCKED_NOT_TESTABLE"
    ]
    runner_payload = run_m2_candidate_experiment_queue(
        m2_requests_path=m2_requests_path,
        offline_trace_dataset_path=offline_trace_dataset_path,
        secondary_metrics=(),
    )
    m3_7d_payload = _load_json(m3_7d_results_path)
    reuse_index = m3_7d_source_index(m3_7d_payload)
    experiments = annotate_fused_experiments(
        runner_payload.get("controlled_experiments", []) or [],
        reuse_index=reuse_index,
    )
    payload = build_fused_m3_results_payload(
        m2_requests_path=str(m2_requests_path),
        m3_7d_results_path=str(m3_7d_results_path),
        offline_trace_dataset_path=str(offline_trace_dataset_path),
        request_payload=request_payload,
        runner_payload=runner_payload,
        experiments=experiments,
        skipped_blocked_requests=skipped_blocked,
    )
    if output_path is not None:
        write_fused_llm_wm_experiment_results(payload, output_path)
    return payload


def m3_7d_source_index(
    payload: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("controlled_experiments", []) or []:
        if not isinstance(row, Mapping):
            continue
        source_transition_id = str(row.get("source_transition_id", "") or "")
        if not source_transition_id:
            continue
        index[source_transition_id] = {
            "m3_7d_request_id": str(row.get("request_id", "")),
            "m3_7d_source_hypothesis_id": str(row.get("source_hypothesis_id", "")),
            "m3_7d_support_events": int(row.get("support_events", 0) or 0),
            "m3_7d_contradiction_events": int(
                row.get("contradiction_events", 0) or 0
            ),
            "m3_7d_matched_control_samples": int(
                row.get("matched_control_samples", 0) or 0
            ),
        }
    return index


def annotate_fused_experiments(
    experiments: Sequence[Mapping[str, Any]],
    *,
    reuse_index: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    annotated: list[Dict[str, Any]] = []
    for row in experiments:
        experiment = dict(row)
        source_transition_id = str(experiment.get("source_transition_id", "") or "")
        reuse = dict(reuse_index.get(source_transition_id, {}) or {})
        support_events = int(experiment.get("support_events", 0) or 0)
        reused = bool(reuse)
        experiment.update(
            {
                "source_transition_reused_from_m3_7d": reused,
                "new_independent_terminal_risk_evidence": (
                    support_events > 0 and not reused
                ),
                "fusion_hypothesis_routing_validated": bool(
                    source_transition_id
                    and str(experiment.get("execution_mode", ""))
                    == "offline_trace_context"
                    and str(experiment.get("metric", ""))
                    == "terminal_state_after_rollout"
                ),
                "support_counting_policy": (
                    "source_reuse_counts_as_routing_validation_not_new_independent_evidence"
                    if reused
                    else "new_source_can_count_as_independent_event_only_candidate"
                ),
                "support": 0,
                "truth_status": M2_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "wrong_confirmations": 0,
            }
        )
        if reuse:
            experiment["m3_7d_reuse_reference"] = reuse
        annotated.append(experiment)
    return annotated


def build_fused_m3_results_payload(
    *,
    m2_requests_path: str,
    m3_7d_results_path: str,
    offline_trace_dataset_path: str,
    request_payload: Mapping[str, Any],
    runner_payload: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    skipped_blocked_requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = summarize_fused_experiments(experiments, skipped_blocked_requests)
    return {
        "config": {
            "schema_version": FUSED_M3_RESULTS_SCHEMA_VERSION,
            "m2_requests_path": m2_requests_path,
            "m3_7d_results_path": m3_7d_results_path,
            "offline_trace_dataset_path": offline_trace_dataset_path,
            "inputs_read": ["M2.15", "M3.7d"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
        },
        "source_m2_summary": dict(request_payload.get("summary", {}) or {}),
        "runner_summary": dict(runner_payload.get("summary", {}) or {}),
        "controlled_experiments": [dict(row) for row in experiments],
        "skipped_m2_blocked_requests": [dict(row) for row in skipped_blocked_requests],
        "runner_blocked_experiments": [
            dict(row) for row in runner_payload.get("blocked_experiments", []) or []
        ],
        "summary": {
            **summary,
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
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def summarize_fused_experiments(
    experiments: Sequence[Mapping[str, Any]],
    skipped_blocked_requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    support_rows = [
        row for row in experiments if int(row.get("support_events", 0) or 0) > 0
    ]
    contradiction_rows = [
        row for row in experiments if int(row.get("contradiction_events", 0) or 0) > 0
    ]
    reused_support_rows = [
        row
        for row in support_rows
        if bool(row.get("source_transition_reused_from_m3_7d"))
    ]
    new_support_rows = [
        row
        for row in support_rows
        if bool(row.get("new_independent_terminal_risk_evidence"))
    ]
    source_ids = [str(row.get("source_transition_id", "")) for row in support_rows]
    reused_source_ids = [
        str(row.get("source_transition_id", "")) for row in reused_support_rows
    ]
    new_source_ids = [
        str(row.get("source_transition_id", "")) for row in new_support_rows
    ]
    return {
        "fused_requests_executed": len(experiments),
        "fused_requests_skipped_blocked": len(skipped_blocked_requests),
        "controlled_experiments_run": sum(
            int(row.get("controlled_experiments_run", 0) or 0)
            for row in experiments
        ),
        "support_events": sum(
            int(row.get("support_events", 0) or 0) for row in experiments
        ),
        "contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in experiments
        ),
        "neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in experiments
        ),
        "source_transition_reused_from_m3_7d": bool(experiments)
        and all(bool(row.get("source_transition_reused_from_m3_7d")) for row in experiments),
        "source_transition_reused_from_m3_7d_count": len(
            {
                str(row.get("source_transition_id", ""))
                for row in experiments
                if bool(row.get("source_transition_reused_from_m3_7d"))
            }
        ),
        "fusion_hypothesis_routing_validated": bool(experiments)
        and all(bool(row.get("fusion_hypothesis_routing_validated")) for row in experiments),
        "new_independent_terminal_risk_evidence": bool(new_support_rows),
        "independent_source_support_events": len(set(new_source_ids)),
        "reused_source_support_events": len(reused_support_rows),
        "supporting_source_transition_ids": len(set(source_ids)),
        "reused_supporting_source_transition_ids": len(set(reused_source_ids)),
        "new_supporting_source_transition_ids": len(set(new_source_ids)),
        "contradicting_source_transition_ids": len(
            {str(row.get("source_transition_id", "")) for row in contradiction_rows}
        ),
        "matched_control_samples_total": sum(
            int(row.get("matched_control_samples", 0) or 0) for row in experiments
        ),
        "target_trace_samples_total": sum(
            int(row.get("target_trace_samples", 0) or 0) for row in experiments
        ),
        "all_controls_same_game_same_available_actions": bool(experiments)
        and all(
            str(row.get("matched_control_policy", ""))
            == "same_game_same_available_actions"
            for row in experiments
        ),
        "blocked_request_ids": [
            str(row.get("request_id", "")) for row in skipped_blocked_requests
        ],
    }


def write_fused_llm_wm_experiment_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_FUSED_M3_RESULTS_PATH,
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
        description="Run M3 experiments for M2.15 fused LLM + WM requests.",
    )
    parser.add_argument("--m2-requests", type=Path, default=DEFAULT_FUSED_M2_REQUESTS_PATH)
    parser.add_argument("--m3-7d-results", type=Path, default=DEFAULT_M3_7D_REPLICATION_RESULTS_PATH)
    parser.add_argument(
        "--offline-trace-dataset",
        type=Path,
        default=DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_FUSED_M3_RESULTS_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_fused_llm_wm_experiment_runner(
        m2_requests_path=args.m2_requests,
        m3_7d_results_path=args.m3_7d_results,
        offline_trace_dataset_path=args.offline_trace_dataset,
        output_path=args.out,
    )
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
