"""P1 handoff of candidate-policy utility to later scientific review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from theory.p1.bp35_sage_candidate_policy_probe import (
    DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MATRIX_OUTPUT_PATH,
    DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH,
    TRUTH_STATUS,
)


DEFAULT_P1_BP35_CANDIDATE_POLICY_UTILITY_HANDOFF_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_candidate_policy_utility_handoff.json"
)
DEFAULT_SCOPE_CONSOLIDATION_PATH = (
    Path("diagnostics") / "m3" / "a32_requested_patch_similarity_scope_consolidation.json"
)
HANDOFF_TYPE = "AGENTIC_UTILITY_CANDIDATE_ONLY"
SUPPORTED_CANDIDATE_ONLY = "SUPPORTED_CANDIDATE_ONLY"
BLOCKED_INVALID_INPUT = "BLOCKED_INVALID_INPUT"


def build_candidate_policy_utility_handoff(
    *,
    probe_payload: Mapping[str, Any],
    matrix_payload: Mapping[str, Any],
    scope_payload: Mapping[str, Any],
    probe_path: str | Path = DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH,
    matrix_path: str | Path = DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MATRIX_OUTPUT_PATH,
    scope_path: str | Path = DEFAULT_SCOPE_CONSOLIDATION_PATH,
) -> dict[str, Any]:
    scope_row = _first_scope_consolidation(scope_payload)
    aggregate = dict(matrix_payload.get("aggregate", {}) or {})
    probe_summary = dict(probe_payload.get("summary", {}) or {})
    probe_comparison = dict(probe_payload.get("comparison", {}) or {})
    blocking_reasons = validate_utility_handoff_inputs(
        scope_row=scope_row,
        aggregate=aggregate,
        probe_summary=probe_summary,
        probe_comparison=probe_comparison,
    )
    accepted = not blocking_reasons
    handoff_item = {
        "handoff_id": "p1_4::bp35::ACTION4_ACTION6::candidate_policy_utility",
        "handoff_type": HANDOFF_TYPE,
        "agentic_utility_status": (
            SUPPORTED_CANDIDATE_ONLY if accepted else BLOCKED_INVALID_INPUT
        ),
        "source_scope_consolidation_id": str(
            scope_row.get("scope_consolidation_id", "")
        ),
        "source_policy_probe_path": str(probe_path),
        "source_policy_matrix_path": str(matrix_path),
        "source_scope_consolidation_path": str(scope_path),
        "candidate_rule_family": str(scope_row.get("candidate_rule_family", "")),
        "candidate_mechanic": str(scope_row.get("candidate_mechanic", "")),
        "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "candidate_beats_baseline_runs": int(
            aggregate.get("candidate_beats_baseline_runs", 0) or 0
        ),
        "candidate_beats_action4_only_runs": int(
            aggregate.get("candidate_beats_action4_only_runs", 0) or 0
        ),
        "candidate_beats_baseline_ratio": float(
            aggregate.get("candidate_beats_baseline_ratio", 0.0) or 0.0
        ),
        "candidate_beats_action4_only_ratio": float(
            aggregate.get("candidate_beats_action4_only_ratio", 0.0) or 0.0
        ),
        "candidate_mean_progress_delta_vs_baseline": float(
            aggregate.get("candidate_mean_progress_delta_vs_baseline", 0.0) or 0.0
        ),
        "candidate_mean_failure_like_selection_delta_vs_baseline": float(
            aggregate.get(
                "candidate_mean_failure_like_selection_delta_vs_baseline",
                0.0,
            )
            or 0.0
        ),
        "single_probe_candidate_better_than_baseline": bool(
            probe_comparison.get("candidate_policy_better_than_baseline_on_any_axis")
        ),
        "single_probe_candidate_progress_proxy_delta": float(
            probe_comparison.get("candidate_progress_proxy_delta", 0.0) or 0.0
        ),
        "patch_similarity_attribution_signal_candidate_only": bool(
            aggregate.get("patch_similarity_attribution_signal_candidate_only")
        ),
        "policy_result_counted_as_mechanistic_confirmation": False,
        "policy_result_counted_as_a33_readiness": False,
        "ready_for_a32_utility_review": accepted,
        "ready_for_a32_utility_review_is_not_verdict": True,
        "a33_ready": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "blocking_reasons": blocking_reasons,
    }
    return {
        "config": {
            "policy_probe_path": str(probe_path),
            "policy_matrix_path": str(matrix_path),
            "scope_consolidation_path": str(scope_path),
            "schema_version": "p1.candidate_policy_utility_handoff.v1",
            "inputs_read": ["P1.1", "P1.2/P1.3", "M3.24"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A32", "A33", "M3"],
        },
        "handoff_type": HANDOFF_TYPE,
        "agentic_utility_status": (
            SUPPORTED_CANDIDATE_ONLY if accepted else BLOCKED_INVALID_INPUT
        ),
        "candidate_beats_baseline_runs": handoff_item[
            "candidate_beats_baseline_runs"
        ],
        "candidate_beats_action4_only_runs": handoff_item[
            "candidate_beats_action4_only_runs"
        ],
        "patch_similarity_attribution_signal_candidate_only": handoff_item[
            "patch_similarity_attribution_signal_candidate_only"
        ],
        "policy_result_counted_as_mechanistic_confirmation": False,
        "a33_ready": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "candidate_policy_utility_handoffs": [handoff_item] if accepted else [],
        "rejected_handoff_candidates": [] if accepted else [handoff_item],
        "summary": {
            "handoffs_produced": 1 if accepted else 0,
            "handoffs_rejected": 0 if accepted else 1,
            "agentic_utility_status": (
                SUPPORTED_CANDIDATE_ONLY if accepted else BLOCKED_INVALID_INPUT
            ),
            "policy_result_counted_as_mechanistic_confirmation": False,
            "a33_ready": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def validate_utility_handoff_inputs(
    *,
    scope_row: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    probe_summary: Mapping[str, Any],
    probe_comparison: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not scope_row:
        reasons.append("missing_scope_consolidation")
    if not bool(scope_row.get("ready_for_agent_policy_probe")):
        reasons.append("scope_not_ready_for_agent_policy_probe")
    if bool(scope_row.get("a33_ready")):
        reasons.append("scope_marked_a33_ready")
    if int(scope_row.get("support", 0) or 0) != 0:
        reasons.append("scope_support_nonzero")
    if str(scope_row.get("revision_status", "")) != "CANDIDATE_ONLY":
        reasons.append("scope_revision_status_not_candidate_only")
    if not bool(probe_summary.get("candidate_policy_probe_ready")):
        reasons.append("single_probe_not_ready")
    if int(probe_summary.get("support", 0) or 0) != 0:
        reasons.append("single_probe_support_nonzero")
    if bool(probe_summary.get("candidate_policy_counted_as_confirmation")):
        reasons.append("single_probe_counted_as_confirmation")
    if not bool(
        probe_comparison.get("candidate_policy_better_than_baseline_on_any_axis")
    ):
        reasons.append("single_probe_no_candidate_utility_signal")
    if not bool(aggregate.get("robust_candidate_policy_utility_signal")):
        reasons.append("matrix_no_robust_utility_signal")
    if int(aggregate.get("candidate_beats_baseline_runs", 0) or 0) <= 0:
        reasons.append("matrix_candidate_never_beats_baseline")
    if int(aggregate.get("candidate_beats_action4_only_runs", 0) or 0) <= 0:
        reasons.append("matrix_candidate_never_beats_action4_only")
    if not bool(aggregate.get("patch_similarity_attribution_signal_candidate_only")):
        reasons.append("matrix_no_patch_similarity_attribution_signal")
    if int(aggregate.get("support", 0) or 0) != 0:
        reasons.append("matrix_support_nonzero")
    if bool(aggregate.get("candidate_policy_counted_as_confirmation")):
        reasons.append("matrix_counted_as_confirmation")
    return reasons


def build_candidate_policy_utility_handoff_from_paths(
    *,
    probe_path: str | Path = DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH,
    matrix_path: str | Path = DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MATRIX_OUTPUT_PATH,
    scope_path: str | Path = DEFAULT_SCOPE_CONSOLIDATION_PATH,
) -> dict[str, Any]:
    return build_candidate_policy_utility_handoff(
        probe_payload=_load_json(probe_path),
        matrix_payload=_load_json(matrix_path),
        scope_payload=_load_json(scope_path),
        probe_path=probe_path,
        matrix_path=matrix_path,
        scope_path=scope_path,
    )


def write_candidate_policy_utility_handoff(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P1_BP35_CANDIDATE_POLICY_UTILITY_HANDOFF_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _first_scope_consolidation(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("scope_consolidations", []) or []
    for row in rows:
        if isinstance(row, Mapping):
            return dict(row)
    return {}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build P1 candidate-policy utility handoff.",
    )
    parser.add_argument("--probe", type=Path, default=DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MATRIX_OUTPUT_PATH)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_CONSOLIDATION_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P1_BP35_CANDIDATE_POLICY_UTILITY_HANDOFF_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_candidate_policy_utility_handoff_from_paths(
        probe_path=args.probe,
        matrix_path=args.matrix,
        scope_path=args.scope,
    )
    write_candidate_policy_utility_handoff(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "handoff_type": payload["handoff_type"],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
