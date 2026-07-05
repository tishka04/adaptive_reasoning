"""Controlled M2 -> M3 handoff validation helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m3.next_experiment_selector import select_next_experiment
from theory.m3.scientific_planner_state import (
    build_scientific_planning_state_from_payloads,
)

from .schema import M2_TRUTH_STATUS, M3CandidateExperimentRequest
from .testability_compiler import (
    DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    load_m3_requests_payload,
    requests_from_payload,
)
from .validators import validate_m3_request


DEFAULT_M2_M3_HANDOFF_VALIDATION_PATH = (
    Path("diagnostics") / "m2" / "m3_handoff_validation.json"
)


def request_to_ledger_entry(
    request: M3CandidateExperimentRequest,
) -> Dict[str, Any]:
    return {
        "key": request.source_hypothesis_id,
        "game_id": request.game_id,
        "description": (
            f"{request.target_action} frontier_conditioned_candidate via "
            f"{request.metric}"
        ),
        "status": "UNRESOLVED",
        "support": 0,
        "contradictions": 0,
        "controlled_test_required": True,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def ledger_payload_from_requests(
    requests: Sequence[M3CandidateExperimentRequest],
) -> Dict[str, Any]:
    return {
        "ledger_entries": [
            request_to_ledger_entry(request)
            for request in requests
            if request.status == "READY_FOR_M3"
        ]
    }


def run_m3_handoff_validation(
    *,
    m3_requests_path: str | Path = DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    live_available_actions: Sequence[str] = ("ACTION3", "ACTION4", "ACTION6"),
) -> Dict[str, Any]:
    payload = load_m3_requests_payload(m3_requests_path)
    requests = requests_from_payload(payload)
    invalid = [
        request.request_id
        for request in requests
        if not validate_m3_request(request).valid
    ]
    ready = [request for request in requests if request.status == "READY_FOR_M3"]
    plan_payload: Dict[str, Any] | None = None
    can_rank = False
    can_execute_at_least_one = False
    if ready and not invalid:
        state = build_scientific_planning_state_from_payloads(
            ledger_payload=ledger_payload_from_requests(ready),
            budget=1,
        )
        plan = select_next_experiment(
            state,
            live_available_actions=live_available_actions,
        )
        plan_payload = plan.to_dict()
        can_rank = True
        can_execute_at_least_one = bool(plan.target_action and plan.control_action)
    summary = {
        "m3_requests_loadable": bool(requests),
        "m3_selector_can_rank_m2_requests": can_rank,
        "m3_can_execute_at_least_one_request": can_execute_at_least_one,
        "invalid_m3_requests": len(invalid),
        "m2_truth_status_unchanged": M2_TRUTH_STATUS,
        "m2_artifacts_mutated_by_m3": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    return {
        "config": {
            "m3_requests_path": str(m3_requests_path),
            "live_available_actions": list(live_available_actions),
            "schema_version": "m2.m3_handoff.v1",
        },
        "summary": summary,
        "selected_plan": plan_payload,
        "invalid_request_ids": invalid,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_m3_handoff_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_M3_HANDOFF_VALIDATION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate M2 requests for M3.")
    parser.add_argument(
        "--m3-requests",
        type=Path,
        default=DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--live-action",
        action="append",
        default=[],
        help="Live action name available to M3. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_M2_M3_HANDOFF_VALIDATION_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_m3_handoff_validation(
        m3_requests_path=args.m3_requests,
        live_available_actions=tuple(args.live_action) or ("ACTION3", "ACTION4", "ACTION6"),
    )
    write_m3_handoff_validation(payload, args.out)
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
