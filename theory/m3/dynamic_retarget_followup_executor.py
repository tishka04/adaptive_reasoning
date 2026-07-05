"""M3.12 executor for dynamic retarget follow-up requests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .dynamic_retarget_followup_planner import (
    DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH,
    DYNAMIC_RETARGET_POLICY,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_executor import (
    available_followup_controls,
    blocked_followup_row,
    execute_metric_followup_experiment,
)
from .refined_followup_planner import READY_FOR_M3_FOLLOWUP


DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "dynamic_retarget_followup_results.json"
)


def run_dynamic_retarget_followup_execution(
    *,
    retarget_requests_path: str | Path = DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
) -> Dict[str, Any]:
    payload = _load_json(retarget_requests_path)
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
        target_args = dict(request.get("target_action_args", {}) or {})
        controls = available_followup_controls(
            request,
            environments_dir=env_dir,
        )
        if not controls:
            blocked.append(
                blocked_followup_row(
                    request,
                    reason="no_dynamic_control_available",
                    target_action_args=target_args,
                    target_action_arg_policy=DYNAMIC_RETARGET_POLICY,
                )
            )
            continue
        for metric in request.get("metrics", []) or []:
            for control_action in controls:
                try:
                    row = execute_metric_followup_experiment(
                        request,
                        metric=str(metric),
                        control_action=str(control_action),
                        target_action_args=target_args,
                        target_action_arg_policy=DYNAMIC_RETARGET_POLICY,
                        environments_dir=env_dir,
                    )
                except Exception as exc:  # pragma: no cover - integration failure path
                    blocked.append(
                        blocked_followup_row(
                            request,
                            reason=f"execution_failed:{exc}",
                            metric=str(metric),
                            control_action=str(control_action),
                            target_action_args=target_args,
                            target_action_arg_policy=DYNAMIC_RETARGET_POLICY,
                        )
                    )
                    continue
                row.update(
                    {
                        "candidate_arg_rank": int(
                            request.get("candidate_arg_rank", 0) or 0
                        ),
                        "candidate_arg_score": float(
                            request.get("candidate_arg_score", 0.0) or 0.0
                        ),
                        "candidate_arg_generation_sources": list(
                            request.get("candidate_arg_generation_sources", []) or []
                        ),
                        "source_followup_request_id": str(
                            request.get("source_followup_request_id", "")
                        ),
                        "target_action_arg_policy": DYNAMIC_RETARGET_POLICY,
                    }
                )
                experiments.append(row)

    per_arg = per_arg_results(experiments)
    mechanism = mechanism_summary(per_arg)
    return build_dynamic_retarget_results_payload(
        retarget_requests_path=retarget_requests_path,
        environments_dir=env_dir,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
        per_arg=per_arg,
        mechanism=mechanism,
    )


def per_arg_results(
    experiments: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_key: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in experiments:
        by_key[_args_key(row.get("target_action_args", {}) or {})].append(dict(row))

    results: list[Dict[str, Any]] = []
    for rows in by_key.values():
        first = rows[0]
        support_events = sum(int(row.get("support_events", 0) or 0) for row in rows)
        contradiction_events = sum(
            int(row.get("contradiction_events", 0) or 0) for row in rows
        )
        neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in rows)
        grounded_support_metrics = sorted(
            {
                str(row.get("metric", ""))
                for row in rows
                if int(row.get("support_events", 0) or 0) > 0
                and not bool(row.get("diagnostic_only"))
            }
        )
        best_support_effect = max(
            [
                float((row.get("delta", {}) or {}).get("effect_size", 0.0) or 0.0)
                for row in rows
                if int(row.get("support_events", 0) or 0) > 0
            ]
            or [0.0]
        )
        results.append(
            {
                "source_refined_hypothesis_id": str(
                    first.get("source_refined_hypothesis_id", "")
                ),
                "target_action": str(first.get("target_action", "")),
                "target_action_args": dict(first.get("target_action_args", {}) or {}),
                "target_action_arg_policy": DYNAMIC_RETARGET_POLICY,
                "candidate_arg_rank": int(first.get("candidate_arg_rank", 0) or 0),
                "candidate_arg_score": float(
                    first.get("candidate_arg_score", 0.0) or 0.0
                ),
                "candidate_arg_generation_sources": list(
                    first.get("candidate_arg_generation_sources", []) or []
                ),
                "metrics_tested": sorted(
                    {str(row.get("metric", "")) for row in rows if row.get("metric")}
                ),
                "controls_tested": sorted(
                    {
                        str(row.get("control_action", ""))
                        for row in rows
                        if row.get("control_action")
                    }
                ),
                "controlled_experiments_run": sum(
                    int(row.get("controlled_experiments_run", 0) or 0)
                    for row in rows
                ),
                "support_events": support_events,
                "contradiction_events": contradiction_events,
                "neutral_events": neutral_events,
                "grounded_support_metrics": grounded_support_metrics,
                "arg_has_grounded_support": bool(grounded_support_metrics),
                "best_support_effect_size": best_support_effect,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
            }
        )
    return tuple(
        sorted(
            results,
            key=lambda row: (
                not bool(row["arg_has_grounded_support"]),
                int(row.get("candidate_arg_rank", 0) or 0),
                _args_key(row.get("target_action_args", {}) or {}),
            ),
        )
    )


def mechanism_summary(per_arg: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    supported = [row for row in per_arg if bool(row.get("arg_has_grounded_support"))]
    best = best_arg_result(supported)
    return {
        "tested_candidate_args": len(per_arg),
        "args_with_grounded_support": len(supported),
        "args_with_contradictions": len(
            [row for row in per_arg if int(row.get("contradiction_events", 0) or 0) > 0]
        ),
        "args_neutral_only": len(
            [
                row
                for row in per_arg
                if int(row.get("support_events", 0) or 0) == 0
                and int(row.get("contradiction_events", 0) or 0) == 0
            ]
        ),
        "best_arg": dict(best.get("target_action_args", {})) if best else None,
        "best_arg_support_events": int(best.get("support_events", 0) or 0)
        if best
        else 0,
        "best_arg_grounded_support_metrics": list(
            best.get("grounded_support_metrics", []) if best else []
        ),
        "mechanism_support_events": 1 if supported else 0,
        "arg_level_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in per_arg
        ),
        "arg_level_support_events_counted_as_mechanism_support": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_remains_only_verdict_location": True,
    }


def best_arg_result(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("support_events", 0) or 0),
            -float(row.get("best_support_effect_size", 0.0) or 0.0),
            int(row.get("candidate_arg_rank", 9999) or 9999),
        ),
    )[0]


def build_dynamic_retarget_results_payload(
    *,
    retarget_requests_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    per_arg: Sequence[Mapping[str, Any]],
    mechanism: Mapping[str, Any],
) -> Dict[str, Any]:
    support_events = sum(int(row.get("support_events", 0) or 0) for row in experiments)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in experiments
    )
    neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in experiments)
    return {
        "config": {
            "retarget_requests_path": str(retarget_requests_path),
            "environments_dir": str(environments_dir),
            "schema_version": "m3.dynamic_retarget_followup_results.v1",
            "inputs_read": ["M3.11"],
            "artifacts_not_modified": ["M2", "M3.8", "M3.9", "M3.10", "M3.11", "A32", "A33"],
            "target_action_arg_policy": DYNAMIC_RETARGET_POLICY,
        },
        "summary": {
            "retarget_requests_consumed": len(requests),
            "retarget_requests_executed": len(
                {str(row.get("request_id", "")) for row in experiments}
            ),
            "controlled_experiments_run": sum(
                int(row.get("controlled_experiments_run", 0) or 0)
                for row in experiments
            ),
            "tested_candidate_args": len(per_arg),
            "args_with_grounded_support": int(
                mechanism.get("args_with_grounded_support", 0) or 0
            ),
            "mechanism_support_events": int(
                mechanism.get("mechanism_support_events", 0) or 0
            ),
            "arg_level_support_events": support_events,
            "arg_level_support_events_counted_as_mechanism_support": False,
            "contradiction_events": contradiction_events,
            "neutral_events": neutral_events,
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
        "per_arg_results": [dict(row) for row in per_arg],
        "mechanism_summary": dict(mechanism),
        "blocked_experiments": [dict(row) for row in blocked],
        "updated_candidate_records": updated_candidate_records(per_arg, mechanism),
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
    per_arg: Sequence[Mapping[str, Any]],
    mechanism: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    if not per_arg:
        return []
    key = str(per_arg[0].get("source_refined_hypothesis_id", ""))
    if not key:
        return []
    return [
        {
            "key": key,
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "contradictions": 0,
            "tested_candidate_args": len(per_arg),
            "args_with_grounded_support": int(
                mechanism.get("args_with_grounded_support", 0) or 0
            ),
            "mechanism_support_events": int(
                mechanism.get("mechanism_support_events", 0) or 0
            ),
            "arg_level_support_events": int(
                mechanism.get("arg_level_support_events", 0) or 0
            ),
            "arg_level_support_events_counted_as_mechanism_support": False,
            "controlled_experiments_run": sum(
                int(row.get("controlled_experiments_run", 0) or 0)
                for row in per_arg
            ),
            "controlled_test_required": True,
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        }
    ]


def write_dynamic_retarget_followup_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _args_key(args: Mapping[str, Any]) -> str:
    return json.dumps({str(key): args[key] for key in sorted(args)}, sort_keys=True)


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute M3.12 dynamic retarget follow-up requests.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_followup_execution(
        retarget_requests_path=args.requests,
        environments_dir=args.environments_dir,
    )
    write_dynamic_retarget_followup_results(payload, args.out)
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
