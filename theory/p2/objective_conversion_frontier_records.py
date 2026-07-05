"""P2.G1 objective-conversion frontier records from terminal-safe policy probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.policy_frontier_records import TRUTH_STATUS


DEFAULT_P3_OBJECTIVE_AWARE_CONSOLIDATION_PATH = (
    Path("diagnostics")
    / "p3"
    / "objective_aware_abstract_policy_utility_consolidation.json"
)
DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_RECORDS_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "objective_conversion_frontier_records.json"
)

OBJECTIVE_CONVERSION_FRONTIER = "OBJECTIVE_CONVERSION_FRONTIER"
TERMINAL_SAFE_BUT_PASSIVE = "TERMINAL_SAFE_BUT_PASSIVE"
OBJECTIVE_CONVERSION_AFTER_SAFE_STOP = "objective_conversion_after_safe_stop"
P3G1_SAFE_BUT_PASSIVE_STATUS = "POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY"


def run_objective_conversion_frontier_records(
    *,
    p3_objective_aware_consolidation_path: str | Path = (
        DEFAULT_P3_OBJECTIVE_AWARE_CONSOLIDATION_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(p3_objective_aware_consolidation_path)
    _validate_p3g1_source(payload)
    summary = dict(payload.get("summary", {}) or {})
    records = []
    if objective_conversion_frontier_is_triggered(summary):
        record = build_objective_conversion_frontier_record(
            summary,
            source_path=p3_objective_aware_consolidation_path,
        )
        validate_objective_conversion_frontier_record(record)
        records.append(record)
    return {
        "config": {
            "schema_version": "p2.objective_conversion_frontier_records.v1",
            "p3_objective_aware_consolidation_path": str(
                p3_objective_aware_consolidation_path
            ),
            "inputs_read": ["P3.G1"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "direct_downstream_write_performed": False,
        },
        "objective_conversion_frontier_records": records,
        "summary": {
            "source_policy_utility_status": str(
                summary.get("policy_utility_status", "")
            ),
            "frontier_records": len(records),
            "frontier_type": OBJECTIVE_CONVERSION_FRONTIER if records else "",
            "frontier_reason": TERMINAL_SAFE_BUT_PASSIVE if records else "",
            "terminal_safe_but_passive_frontiers": len(records),
            "ready_for_objective_conversion_review": bool(records),
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
        "objective_conversion_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def objective_conversion_frontier_is_triggered(summary: Mapping[str, Any]) -> bool:
    return bool(
        str(summary.get("policy_utility_status", "")) == P3G1_SAFE_BUT_PASSIVE_STATUS
        and bool(summary.get("terminal_rate_reduced_vs_p3g0", False))
        and not bool(summary.get("objective_completion_signal", False))
        and float(summary.get("best_objective_terminal_rate", 1.0)) == 0.0
    )


def build_objective_conversion_frontier_record(
    summary: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> Dict[str, Any]:
    frontier_id = "p2g1::bp35::terminal_safe_but_passive::objective_conversion"
    return {
        "frontier_id": frontier_id,
        "source": "P3.G1",
        "source_p3_objective_aware_consolidation_path": str(source_path),
        "frontier_type": OBJECTIVE_CONVERSION_FRONTIER,
        "frontier_reason": TERMINAL_SAFE_BUT_PASSIVE,
        "game_id": "bp35-0a0ad940",
        "blocked_capability": OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
        "candidate_policy_state": "terminal_safe_relation_policy_stops_or_avoids_terminal_but_does_not_complete_objective",
        "source_policy_utility_status": str(summary.get("policy_utility_status", "")),
        "evidence": {
            "p3g0_mean_progress_proxy": float(
                summary.get("p3g0_mean_progress_proxy", 0.0) or 0.0
            ),
            "best_objective_mean_progress_proxy": float(
                summary.get("best_objective_mean_progress_proxy", 0.0) or 0.0
            ),
            "p3g0_terminal_rate": float(summary.get("p3g0_terminal_rate", 0.0) or 0.0),
            "best_objective_terminal_rate": float(
                summary.get("best_objective_terminal_rate", 0.0) or 0.0
            ),
            "p3g0_terminal_adjusted_progress": float(
                summary.get("p3g0_terminal_adjusted_progress", 0.0) or 0.0
            ),
            "best_objective_terminal_adjusted_progress": float(
                summary.get("best_objective_terminal_adjusted_progress", 0.0)
                or 0.0
            ),
            "progress_proxy_preserved_vs_p3g0": bool(
                summary.get("progress_proxy_preserved_vs_p3g0", False)
            ),
            "terminal_rate_reduced_vs_p3g0": bool(
                summary.get("terminal_rate_reduced_vs_p3g0", False)
            ),
            "objective_completion_signal": bool(
                summary.get("objective_completion_signal", False)
            ),
            "best_objective_aware_condition": str(
                summary.get("best_objective_aware_condition", "")
            ),
            "best_objective_aware_lambda_terminal_risk": summary.get(
                "best_objective_aware_lambda_terminal_risk"
            ),
        },
        "scientific_questions": [
            "After terminal-safe stop/avoidance, which action converts relation progress into objective completion?",
            "Which target relation should replace generic distance_decreases when objective completion remains false?",
            "Does a specific post-safe-stop ACTION3, ACTION4, or ACTION6 sequence create objective progress without re-entering terminal risk?",
            "Which objective-readiness signal is missing from the abstract symbolic model?",
        ],
        "desired_hypothesis_families": [
            "post_safe_stop_objective_conversion",
            "subgoal_target_reselection",
            "objective_readiness_condition",
            "terminal_safe_sequence_search",
        ],
        "requested_experiment_styles": [
            "stop_state_action_matrix",
            "post_safe_stop_short_sequence_probe",
            "relation_target_ablation_after_safe_stop",
            "objective_completion_vs_relation_progress_discriminator",
        ],
        "ready_for_objective_conversion_review": True,
        "ready_for_m2_or_m3": False,
        "ready_for_saturation_handoff": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "objective_conversion_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }


def validate_objective_conversion_frontier_record(record: Mapping[str, Any]) -> None:
    if not record.get("frontier_id"):
        raise ValueError("frontier_id is required")
    if str(record.get("frontier_type", "")) != OBJECTIVE_CONVERSION_FRONTIER:
        raise ValueError("frontier_type must be OBJECTIVE_CONVERSION_FRONTIER")
    if str(record.get("frontier_reason", "")) != TERMINAL_SAFE_BUT_PASSIVE:
        raise ValueError("frontier_reason must be TERMINAL_SAFE_BUT_PASSIVE")
    if str(record.get("blocked_capability", "")) != OBJECTIVE_CONVERSION_AFTER_SAFE_STOP:
        raise ValueError("blocked_capability must be objective conversion after safe stop")
    evidence = dict(record.get("evidence", {}) or {})
    if not bool(evidence.get("terminal_rate_reduced_vs_p3g0", False)):
        raise ValueError("objective conversion frontier requires terminal-rate reduction")
    if bool(evidence.get("objective_completion_signal", False)):
        raise ValueError("objective conversion frontier requires no objective completion")
    if not bool(record.get("ready_for_objective_conversion_review", False)):
        raise ValueError("record must be ready for objective conversion review")
    if bool(record.get("ready_for_m2_or_m3", False)):
        raise ValueError("P2.G1 records are review-ready, not direct M2/M3 handoffs")
    if bool(record.get("ready_for_saturation_handoff", False)):
        raise ValueError("objective conversion frontier cannot be saturation handoff")
    if bool(record.get("a33_ready", False)):
        raise ValueError("objective conversion frontier cannot be A33-ready")
    if str(record.get("status", "")) != "UNRESOLVED":
        raise ValueError("objective conversion frontier must remain unresolved")
    if int(record.get("support", 0) or 0) != 0:
        raise ValueError("objective conversion frontier support must remain 0")
    if str(record.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("objective conversion frontier must remain candidate-only")
    if str(record.get("truth_status", "")) != TRUTH_STATUS:
        raise ValueError("objective conversion frontier truth_status must remain P2-local")
    if bool(record.get("revision_performed", False)):
        raise ValueError("objective conversion frontier revision_performed must be false")
    if int(record.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("objective conversion frontier wrong_confirmations must remain 0")
    if bool(record.get("objective_conversion_frontier_counted_as_confirmation", False)):
        raise ValueError("objective conversion frontier cannot count as confirmation")
    if bool(record.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot count as scientific verdict")


def _validate_p3g1_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G1 source support must remain 0")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)) or bool(
        summary.get("policy_result_counted_as_scientific_verdict", False)
    ):
        raise ValueError("P3.G1 policy result cannot be scientific verdict")
    if bool(payload.get("candidate_model_counted_as_confirmed_mechanic", False)) or bool(
        summary.get("candidate_model_counted_as_confirmed_mechanic", False)
    ):
        raise ValueError("P3.G1 model cannot be confirmed mechanic")
    if bool(payload.get("a32_write_performed", False)) or bool(
        summary.get("a32_write_performed", False)
    ):
        raise ValueError("P3.G1 source must not write A32")
    if bool(payload.get("a33_write_performed", False)) or bool(
        summary.get("a33_write_performed", False)
    ):
        raise ValueError("P3.G1 source must not write A33")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("P3.G1 source revision_performed must be false")
    if int(payload.get("wrong_confirmations", summary.get("wrong_confirmations", 0)) or 0) != 0:
        raise ValueError("P3.G1 source wrong_confirmations must remain 0")


def write_objective_conversion_frontier_records(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_RECORDS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P2.G1 objective-conversion frontier extraction.",
    )
    parser.add_argument(
        "--p3-objective-aware-consolidation",
        type=Path,
        default=DEFAULT_P3_OBJECTIVE_AWARE_CONSOLIDATION_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_RECORDS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_frontier_records(
        p3_objective_aware_consolidation_path=args.p3_objective_aware_consolidation,
    )
    write_objective_conversion_frontier_records(payload, args.out)
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
