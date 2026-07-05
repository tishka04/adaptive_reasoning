"""P2.G4 risk-aware post-stop frontier records.

P3.G4 establishes candidate-only risk-aware policy utility after safe stop, but
still observes no objective completion. P2.G4 records that as a new frontier
without writing M2/M3/A32/A33 or treating policy utility as a scientific verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.policy_frontier_records import TRUTH_STATUS


DEFAULT_P3_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_PATH = (
    Path("diagnostics")
    / "p3"
    / "risk_targeted_contextual_post_stop_policy_validation.json"
)
DEFAULT_P2_RISK_AWARE_POST_STOP_FRONTIER_RECORDS_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "risk_aware_post_stop_frontier_records.json"
)

P3G4_RISK_AWARE_STATUS = "RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY"
RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER = (
    "RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER"
)
RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION = (
    "RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION"
)
OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION = (
    "objective_completion_after_risk_aware_safe_conversion"
)
RISK_AWARE_FRONTIER_RECORDED_CANDIDATE_ONLY = (
    "RISK_AWARE_POST_STOP_FRONTIER_RECORDED_CANDIDATE_ONLY"
)


def run_risk_aware_post_stop_frontier_records(
    *,
    p3_risk_targeted_validation_path: str
    | Path = DEFAULT_P3_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_PATH,
) -> Dict[str, Any]:
    payload = _load_json(p3_risk_targeted_validation_path)
    _validate_p3g4_source(payload)
    summary = dict(payload.get("summary", {}) or {})
    records = []
    if risk_aware_post_stop_frontier_is_triggered(summary):
        record = build_risk_aware_post_stop_frontier_record(
            summary,
            source_path=p3_risk_targeted_validation_path,
        )
        validate_risk_aware_post_stop_frontier_record(record)
        records.append(record)

    return {
        "config": {
            "schema_version": "p2.risk_aware_post_stop_frontier_records.v1",
            "p3_risk_targeted_validation_path": str(p3_risk_targeted_validation_path),
            "inputs_read": ["P3.G4"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "direct_downstream_write_performed": False,
        },
        "risk_aware_post_stop_frontier_records": records,
        "summary": {
            "source_policy_utility_status": str(
                summary.get("policy_utility_status", "")
            ),
            "frontier_records": len(records),
            "frontier_record_status": (
                RISK_AWARE_FRONTIER_RECORDED_CANDIDATE_ONLY if records else ""
            ),
            "frontier_type": (
                RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
                if records
                else ""
            ),
            "frontier_reason": (
                RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION if records else ""
            ),
            "blocked_capability": (
                OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
                if records
                else ""
            ),
            "terminal_safety_observed": bool(
                float(summary.get("terminal_rate", 1.0) or 0.0) == 0.0
            ),
            "oos_utility_observed": bool(
                float(summary.get("mean_delta_vs_hold", 0.0) or 0.0) > 0.0
                and bool(summary.get("improvement_over_action6_only", False))
            ),
            "risk_aware_selection_observed": bool(
                int(summary.get("static_extension_terminal_options", 0) or 0) > 0
                and int(summary.get("unsafe_extension_options_avoided", 0) or 0) > 0
            ),
            "objective_completion_signal": bool(
                summary.get("objective_completion_signal", False)
            ),
            "ready_for_risk_aware_objective_frontier_review": bool(records),
            "ready_for_m2_or_m3": False,
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "risk_aware_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def risk_aware_post_stop_frontier_is_triggered(summary: Mapping[str, Any]) -> bool:
    return bool(
        str(summary.get("policy_utility_status", "")) == P3G4_RISK_AWARE_STATUS
        and bool(summary.get("execution_performed", False))
        and bool(summary.get("source_cells_rerun", False))
        and not bool(summary.get("adapter_relearned", True))
        and not bool(summary.get("selection_uses_risk_targeted_candidate_outcomes", True))
        and float(summary.get("terminal_rate", 1.0) or 0.0) == 0.0
        and float(summary.get("mean_delta_vs_hold", 0.0) or 0.0) > 0.0
        and bool(summary.get("improvement_over_action6_only", False))
        and int(summary.get("static_extension_terminal_options", 0) or 0) > 0
        and int(summary.get("unsafe_extension_options_avoided", 0) or 0) > 0
        and not bool(summary.get("objective_completion_signal", False))
        and int(summary.get("objective_completion_runs", 0) or 0) == 0
    )


def build_risk_aware_post_stop_frontier_record(
    summary: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> Dict[str, Any]:
    baseline_aggregates = dict(summary.get("baseline_aggregates", {}) or {})
    contextual_policy = dict(summary.get("contextual_policy_aggregate", {}) or {})
    risk_stats = dict(summary.get("risk_targeted_extension_risk_stats", {}) or {})
    frontier_id = "p2g4::bp35::risk_aware_post_stop_no_objective_completion"
    return {
        "frontier_id": frontier_id,
        "source": "P3.G4",
        "source_p3_risk_targeted_validation_path": str(source_path),
        "frontier_type": RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER,
        "frontier_reason": RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
        "frontier_record_status": RISK_AWARE_FRONTIER_RECORDED_CANDIDATE_ONLY,
        "game_id": "bp35-0a0ad940",
        "blocked_capability": (
            OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
        ),
        "candidate_policy_state": (
            "risk_aware_terminal_safe_post_stop_selector_improves_proxy_progress_"
            "without_objective_completion"
        ),
        "source_policy_utility_status": str(summary.get("policy_utility_status", "")),
        "evidence": {
            "accepted_risk_targeted_safe_stops": int(
                summary.get("accepted_risk_targeted_safe_stops", 0) or 0
            ),
            "terminal_rate": float(summary.get("terminal_rate", 0.0) or 0.0),
            "mean_terminal_adjusted_progress": float(
                summary.get("mean_terminal_adjusted_progress", 0.0) or 0.0
            ),
            "mean_delta_vs_hold": float(
                summary.get("mean_delta_vs_hold", 0.0) or 0.0
            ),
            "mean_delta_vs_action6_only": float(
                summary.get("mean_delta_vs_action6_only", 0.0) or 0.0
            ),
            "improvement_over_action6_only": bool(
                summary.get("improvement_over_action6_only", False)
            ),
            "selected_extension_count": int(
                summary.get("selected_extension_count", 0) or 0
            ),
            "selected_action6_only_count": int(
                summary.get("selected_action6_only_count", 0) or 0
            ),
            "static_extension_terminal_options": int(
                summary.get("static_extension_terminal_options", 0) or 0
            ),
            "static_extension_terminal_safe_stops": int(
                summary.get("static_extension_terminal_safe_stops", 0) or 0
            ),
            "unsafe_extension_options_avoided": int(
                summary.get("unsafe_extension_options_avoided", 0) or 0
            ),
            "objective_completion_signal": bool(
                summary.get("objective_completion_signal", False)
            ),
            "objective_completion_runs": int(
                summary.get("objective_completion_runs", 0) or 0
            ),
            "adapter_relearned": bool(summary.get("adapter_relearned", True)),
            "source_cells_rerun": bool(summary.get("source_cells_rerun", False)),
            "selection_uses_risk_targeted_candidate_outcomes": bool(
                summary.get("selection_uses_risk_targeted_candidate_outcomes", True)
            ),
        },
        "policy_aggregate_snapshot": {
            "contextual_policy": _aggregate_snapshot(contextual_policy),
            "hold_or_stop_state": _aggregate_snapshot(
                dict(baseline_aggregates.get("hold_or_stop_state", {}) or {})
            ),
            "ACTION6": _aggregate_snapshot(
                dict(baseline_aggregates.get("ACTION6", {}) or {})
            ),
            "ACTION6,ACTION3": _aggregate_snapshot(
                dict(baseline_aggregates.get("ACTION6,ACTION3", {}) or {})
            ),
            "ACTION6,ACTION4": _aggregate_snapshot(
                dict(baseline_aggregates.get("ACTION6,ACTION4", {}) or {})
            ),
        },
        "risk_region_snapshot": {
            "action6_action3_terminal_rate": float(
                risk_stats.get("action6_action3_terminal_rate", 0.0) or 0.0
            ),
            "action6_action4_terminal_rate": float(
                risk_stats.get("action6_action4_terminal_rate", 0.0) or 0.0
            ),
            "static_extension_terminal_options": int(
                risk_stats.get("static_extension_terminal_options", 0) or 0
            ),
            "static_extension_terminal_safe_stops": int(
                risk_stats.get("static_extension_terminal_safe_stops", 0) or 0
            ),
            "unsafe_extension_options_avoided_by_selector": int(
                risk_stats.get("unsafe_extension_options_avoided_by_selector", 0) or 0
            ),
            "terminal_extension_records": [
                _terminal_extension_snapshot(record)
                for record in risk_stats.get("terminal_extension_records", []) or []
                if isinstance(record, Mapping)
            ],
        },
        "blocked_capability_hypotheses": [
            "proxy_progress_not_completion_condition",
            "objective_readiness_detector_missing",
            "terminal_commit_or_submit_action_missing",
            "goal_representation_missing_beyond_safe_progress",
            "conversion_state_useful_but_not_completion_trigger",
        ],
        "scientific_questions": [
            "Which signal marks a post-stop state as ready for objective completion rather than more relation progress?",
            "Does a terminal-safe conversion require a distinct commit/submit action after ACTION6-led movement?",
            "Which objective feature should replace terminal-adjusted relation progress as the selector target?",
            "Do ACTION6-led extensions move into a useful pre-completion zone without triggering level change?",
            "Can a completion discriminator separate safe progress from objective readiness before terminal risk rises?",
        ],
        "desired_hypothesis_families": [
            "objective_readiness_detection",
            "post_conversion_commit_action_search",
            "goal_state_representation_beyond_safe_progress",
            "proxy_progress_vs_completion_discriminator",
            "risk_aware_selector_completion_gap",
        ],
        "requested_experiment_styles": [
            "post_selector_objective_readiness_probe",
            "post_conversion_commit_action_matrix",
            "terminal_safe_progress_vs_completion_discriminator",
            "horizon_conditioned_completion_trigger_search",
            "risk_aware_policy_ablation_with_completion_metrics",
        ],
        "ready_for_risk_aware_objective_frontier_review": True,
        "ready_for_m2_or_m3": False,
        "ready_for_direct_downstream_write": False,
        "ready_for_saturation_handoff": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "risk_aware_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "risk_aware_policy_counted_as_objective_solution": False,
    }


def validate_risk_aware_post_stop_frontier_record(record: Mapping[str, Any]) -> None:
    if not record.get("frontier_id"):
        raise ValueError("frontier_id is required")
    if str(record.get("frontier_type", "")) != (
        RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
    ):
        raise ValueError("frontier_type must be risk-aware post-stop frontier")
    if str(record.get("frontier_reason", "")) != (
        RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION
    ):
        raise ValueError("frontier_reason must describe utility without completion")
    if str(record.get("blocked_capability", "")) != (
        OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
    ):
        raise ValueError("blocked_capability must be risk-aware objective completion")
    evidence = dict(record.get("evidence", {}) or {})
    if float(evidence.get("terminal_rate", 1.0) or 0.0) != 0.0:
        raise ValueError("risk-aware frontier requires terminal-safe selector")
    if float(evidence.get("mean_delta_vs_hold", 0.0) or 0.0) <= 0.0:
        raise ValueError("risk-aware frontier requires utility over hold")
    if not bool(evidence.get("improvement_over_action6_only", False)):
        raise ValueError("risk-aware frontier requires improvement over ACTION6")
    if int(evidence.get("static_extension_terminal_options", 0) or 0) <= 0:
        raise ValueError("risk-aware frontier requires reproduced extension risk")
    if int(evidence.get("unsafe_extension_options_avoided", 0) or 0) <= 0:
        raise ValueError("risk-aware frontier requires avoided unsafe extensions")
    if bool(evidence.get("objective_completion_signal", False)):
        raise ValueError("risk-aware frontier requires no objective completion")
    if int(evidence.get("objective_completion_runs", 0) or 0) != 0:
        raise ValueError("risk-aware frontier requires zero completion runs")
    if bool(evidence.get("adapter_relearned", True)):
        raise ValueError("risk-aware frontier requires frozen adapter")
    if not bool(evidence.get("source_cells_rerun", False)):
        raise ValueError("risk-aware frontier requires rerun execution cells")
    if bool(evidence.get("selection_uses_risk_targeted_candidate_outcomes", True)):
        raise ValueError("risk-aware selector cannot use risk-targeted outcomes")
    if not list(record.get("blocked_capability_hypotheses", []) or []):
        raise ValueError("risk-aware frontier needs blocked capability hypotheses")
    if not list(record.get("desired_hypothesis_families", []) or []):
        raise ValueError("risk-aware frontier needs hypothesis families")
    if not list(record.get("requested_experiment_styles", []) or []):
        raise ValueError("risk-aware frontier needs experiment styles")
    if not list(record.get("scientific_questions", []) or []):
        raise ValueError("risk-aware frontier needs scientific questions")
    if not bool(record.get("ready_for_risk_aware_objective_frontier_review", False)):
        raise ValueError("record must be ready for risk-aware objective review")
    if bool(record.get("ready_for_m2_or_m3", False)):
        raise ValueError("P2.G4 records are review-ready, not direct M2/M3 handoffs")
    if bool(record.get("ready_for_direct_downstream_write", False)):
        raise ValueError("P2.G4 cannot be ready for direct downstream write")
    if bool(record.get("ready_for_saturation_handoff", False)):
        raise ValueError("risk-aware frontier cannot be saturation handoff")
    if bool(record.get("a33_ready", False)):
        raise ValueError("risk-aware frontier cannot be A33-ready")
    if str(record.get("status", "")) != "UNRESOLVED":
        raise ValueError("risk-aware frontier must remain unresolved")
    if int(record.get("support", 0) or 0) != 0:
        raise ValueError("risk-aware frontier support must remain 0")
    if str(record.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("risk-aware frontier must remain candidate-only")
    if str(record.get("truth_status", "")) != TRUTH_STATUS:
        raise ValueError("risk-aware frontier truth_status must remain P2-local")
    if bool(record.get("revision_performed", False)):
        raise ValueError("risk-aware frontier revision_performed must be false")
    if int(record.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("risk-aware frontier wrong_confirmations must remain 0")
    if bool(record.get("risk_aware_frontier_counted_as_confirmation", False)):
        raise ValueError("risk-aware frontier cannot count as confirmation")
    if bool(record.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot count as scientific verdict")
    if bool(record.get("risk_aware_policy_counted_as_objective_solution", False)):
        raise ValueError("risk-aware policy cannot count as objective solution")


def _validate_p3g4_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G4 source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("P3.G4 source summary support must remain 0")
    if str(payload.get("revision_status", summary.get("revision_status", ""))) != (
        "CANDIDATE_ONLY"
    ):
        raise ValueError("P3.G4 source must remain candidate-only")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)) or bool(
        summary.get("policy_result_counted_as_scientific_verdict", False)
    ):
        raise ValueError("P3.G4 policy result cannot be scientific verdict")
    if bool(payload.get("adapter_counted_as_mechanic_confirmation", False)) or bool(
        summary.get("adapter_counted_as_mechanic_confirmation", False)
    ):
        raise ValueError("P3.G4 adapter cannot be mechanic confirmation")
    if bool(payload.get("a32_write_performed", False)) or bool(
        summary.get("a32_write_performed", False)
    ):
        raise ValueError("P3.G4 source must not write A32")
    if bool(payload.get("a33_write_performed", False)) or bool(
        summary.get("a33_write_performed", False)
    ):
        raise ValueError("P3.G4 source must not write A33")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("P3.G4 source revision_performed must be false")
    if int(payload.get("wrong_confirmations", summary.get("wrong_confirmations", 0)) or 0) != 0:
        raise ValueError("P3.G4 source wrong_confirmations must remain 0")


def _aggregate_snapshot(aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "condition": str(aggregate.get("condition", "")),
        "runs": int(aggregate.get("runs", 0) or 0),
        "mean_terminal_adjusted_progress": float(
            aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "mean_delta_vs_hold": float(aggregate.get("mean_delta_vs_hold", 0.0) or 0.0),
        "terminal_rate": float(aggregate.get("terminal_rate", 0.0) or 0.0),
        "objective_completion_runs": int(
            aggregate.get("objective_completion_runs", 0) or 0
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def _terminal_extension_snapshot(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "safe_stop_id": str(record.get("safe_stop_id", "")),
        "sampling_family": str(record.get("sampling_family", "")),
        "terminal_horizon_band": str(record.get("terminal_horizon_band", "")),
        "hold_baseline_band": str(record.get("hold_baseline_band", "")),
        "action_or_sequence": [
            str(action) for action in record.get("action_or_sequence", []) or []
        ],
        "candidate_terminal_adjusted_progress_after_stop": float(
            record.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def write_risk_aware_post_stop_frontier_records(
    payload: Mapping[str, Any],
    output_path: str
    | Path = DEFAULT_P2_RISK_AWARE_POST_STOP_FRONTIER_RECORDS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P2.G4 risk-aware post-stop frontier extraction.",
    )
    parser.add_argument(
        "--p3-risk-targeted-validation",
        type=Path,
        default=DEFAULT_P3_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_RISK_AWARE_POST_STOP_FRONTIER_RECORDS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_risk_aware_post_stop_frontier_records(
        p3_risk_targeted_validation_path=args.p3_risk_targeted_validation,
    )
    write_risk_aware_post_stop_frontier_records(payload, args.out)
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
