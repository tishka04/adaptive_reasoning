"""M3.G0.3 candidate-only consolidation of generic mechanic observations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .generic_mechanic_experiment_executor import (
    DEFAULT_GENERIC_MECHANIC_EXPERIMENT_RESULTS_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_MECHANIC_EVIDENCE_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "generic_mechanic_evidence_consolidation.json"
)

SUPPORTED_CANDIDATE_ONLY = "SUPPORTED_CANDIDATE_ONLY"
NEUTRAL_CANDIDATE_ONLY = "NEUTRAL_CANDIDATE_ONLY"
MIXED_CANDIDATE_ONLY = "MIXED_CANDIDATE_ONLY"
CONTRADICTED_CANDIDATE_ONLY = "CONTRADICTED_CANDIDATE_ONLY"
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
DUPLICATE_EVIDENCE_ONLY = "DUPLICATE_EVIDENCE_ONLY"
LOW_RESET_ONLY = "LOW_RESET_ONLY"


def run_generic_mechanic_evidence_consolidation(
    *,
    generic_results_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_EXPERIMENT_RESULTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(generic_results_path)
    validate_generic_results_source(payload)
    hypothesis_records = consolidate_hypothesis_records(payload)
    family_records = consolidate_family_records(hypothesis_records)
    context_diversity = assess_context_diversity(payload)
    return {
        "config": {
            "generic_results_path": str(generic_results_path),
            "schema_version": "m3.generic_mechanic_evidence_consolidation.v1",
            "inputs_read": ["M3.G0.2"],
            "artifacts_not_modified": ["M1", "M2", "A32", "A33"],
            "execution_performed": False,
            "consolidation_policy": {
                "support_events_are_not_scientific_support": True,
                "duplicate_links_are_not_independent": True,
                "reset_only_context_is_insufficient_for_model_induction": True,
                "semantic_interpretation_for_dynamic_invariants_remains_unknown": True,
            },
        },
        "summary": summarize_generic_consolidation(
            source_payload=payload,
            hypothesis_records=hypothesis_records,
            family_records=family_records,
            context_diversity=context_diversity,
        ),
        "context_diversity": context_diversity,
        "hypothesis_consolidations": hypothesis_records,
        "family_consolidations": family_records,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "support_events_counted_as_scientific_support": False,
        "duplicate_links_counted_as_independent": False,
        "semantic_interpretation_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def consolidate_hypothesis_records(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    summary = dict(payload.get("summary", {}) or {})
    independent_contexts = independent_context_count(payload)
    context_diversity_label = str(assess_context_diversity(payload)["assessment"])
    records: list[Dict[str, Any]] = []
    for row in payload.get("request_results", []) or []:
        request = dict(row)
        support_events = int(request.get("support_events", 0) or 0)
        contradiction_events = int(request.get("contradiction_events", 0) or 0)
        neutral_events = int(request.get("neutral_events", 0) or 0)
        cells_linked = int(request.get("cells_linked", 0) or 0)
        status = candidate_status(
            support_events=support_events,
            contradiction_events=contradiction_events,
            neutral_events=neutral_events,
            independent_contexts=independent_contexts,
            context_diversity_label=context_diversity_label,
        )
        if status == SUPPORTED_CANDIDATE_ONLY and int(summary.get("unique_execution_cells", 0) or 0) <= 3:
            status = DUPLICATE_EVIDENCE_ONLY
        records.append(
            {
                "consolidation_id": hypothesis_consolidation_id(request),
                "request_id": str(request.get("request_id", "")),
                "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
                "source_mechanic_family": str(request.get("source_mechanic_family", "")),
                "test_type": str(request.get("test_type", "")),
                "candidate_status": status,
                "observation_interpretation": str(
                    request.get("observation_interpretation", "")
                ),
                "raw_support_events": support_events,
                "raw_contradiction_events": contradiction_events,
                "raw_neutral_events": neutral_events,
                "cells_linked": cells_linked,
                "unique_execution_cells": int(summary.get("unique_execution_cells", 0) or 0),
                "independent_contexts": independent_contexts,
                "context_diversity_assessment": context_diversity_label,
                "support_events_counted_as_scientific_support": False,
                "duplicate_links_counted_as_independent": False,
                "requires_contextual_followup": context_diversity_label == LOW_RESET_ONLY,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                "observation_counted_as_confirmation": False,
                "source_confidence_counted_as_support": False,
            }
        )
    return tuple(records)


def consolidate_family_records(
    hypothesis_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in hypothesis_records:
        by_family[str(row.get("source_mechanic_family", ""))].append(row)
    records: list[Dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        statuses = Counter(str(row.get("candidate_status", "")) for row in rows)
        raw_support = sum(int(row.get("raw_support_events", 0) or 0) for row in rows)
        raw_contradiction = sum(
            int(row.get("raw_contradiction_events", 0) or 0) for row in rows
        )
        raw_neutral = sum(int(row.get("raw_neutral_events", 0) or 0) for row in rows)
        family_status = family_candidate_status(statuses)
        records.append(
            {
                "family": family,
                "candidate_status": family_status,
                "hypotheses": len(rows),
                "status_counts": dict(sorted(statuses.items())),
                "raw_support_events": int(raw_support),
                "raw_contradiction_events": int(raw_contradiction),
                "raw_neutral_events": int(raw_neutral),
                "support_events_counted_as_scientific_support": False,
                "duplicate_links_counted_as_independent": False,
                "requires_contextual_followup": any(
                    bool(row.get("requires_contextual_followup")) for row in rows
                ),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "wrong_confirmations": 0,
                "family_consolidation_counted_as_confirmation": False,
            }
        )
    return tuple(records)


def candidate_status(
    *,
    support_events: int,
    contradiction_events: int,
    neutral_events: int,
    independent_contexts: int,
    context_diversity_label: str,
) -> str:
    if independent_contexts <= 0:
        return INSUFFICIENT_CONTEXT
    if support_events > 0 and contradiction_events > 0:
        return MIXED_CANDIDATE_ONLY
    if contradiction_events > 0 and support_events == 0:
        return CONTRADICTED_CANDIDATE_ONLY
    if support_events > 0:
        return (
            DUPLICATE_EVIDENCE_ONLY
            if context_diversity_label == LOW_RESET_ONLY
            else SUPPORTED_CANDIDATE_ONLY
        )
    if neutral_events > 0:
        return NEUTRAL_CANDIDATE_ONLY
    return INSUFFICIENT_CONTEXT


def family_candidate_status(statuses: Mapping[str, int]) -> str:
    if int(statuses.get(MIXED_CANDIDATE_ONLY, 0) or 0) > 0:
        return MIXED_CANDIDATE_ONLY
    if int(statuses.get(CONTRADICTED_CANDIDATE_ONLY, 0) or 0) > 0 and int(
        statuses.get(SUPPORTED_CANDIDATE_ONLY, 0) or 0
    ) == 0:
        return CONTRADICTED_CANDIDATE_ONLY
    if int(statuses.get(SUPPORTED_CANDIDATE_ONLY, 0) or 0) > 0:
        return SUPPORTED_CANDIDATE_ONLY
    if int(statuses.get(DUPLICATE_EVIDENCE_ONLY, 0) or 0) > 0:
        return DUPLICATE_EVIDENCE_ONLY
    if int(statuses.get(NEUTRAL_CANDIDATE_ONLY, 0) or 0) > 0:
        return NEUTRAL_CANDIDATE_ONLY
    return INSUFFICIENT_CONTEXT


def assess_context_diversity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    independent_contexts = independent_context_count(payload)
    unique_cells = int(summary.get("unique_execution_cells", 0) or 0)
    planned_cells = int(summary.get("planned_execution_cells", 0) or 0)
    duplicate_cells = int(summary.get("duplicate_execution_cells", 0) or 0)
    replay_policies = {
        str(row.get("replay_policy", ""))
        for row in payload.get("execution_cells", []) or []
    }
    assessment = (
        LOW_RESET_ONLY
        if independent_contexts <= 1 and replay_policies <= {"initial_reset_same_state"}
        else "CONTEXT_DIVERSITY_PRESENT"
    )
    return {
        "assessment": assessment,
        "unique_execution_cells": unique_cells,
        "planned_execution_cells": planned_cells,
        "duplicate_execution_cells": duplicate_cells,
        "duplicate_execution_cells_counted_as_independent": False,
        "independent_contexts": independent_contexts,
        "replay_policies": sorted(replay_policies),
        "requires_contextual_followup_before_symbolic_model": assessment == LOW_RESET_ONLY,
        "support_events_counted_as_scientific_support": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def summarize_generic_consolidation(
    *,
    source_payload: Mapping[str, Any],
    hypothesis_records: Sequence[Mapping[str, Any]],
    family_records: Sequence[Mapping[str, Any]],
    context_diversity: Mapping[str, Any],
) -> Dict[str, Any]:
    source_summary = dict(source_payload.get("summary", {}) or {})
    status_counts = Counter(str(row.get("candidate_status", "")) for row in hypothesis_records)
    return {
        "generic_requests_consumed": int(
            source_summary.get("generic_requests_consumed", len(hypothesis_records)) or 0
        ),
        "hypothesis_consolidations": len(hypothesis_records),
        "family_consolidations": len(family_records),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "source_support_events": int(source_summary.get("support_events", 0) or 0),
        "source_contradiction_events": int(
            source_summary.get("contradiction_events", 0) or 0
        ),
        "source_neutral_events": int(source_summary.get("neutral_events", 0) or 0),
        "unique_execution_cells": int(source_summary.get("unique_execution_cells", 0) or 0),
        "planned_execution_cells": int(source_summary.get("planned_execution_cells", 0) or 0),
        "duplicate_execution_cells": int(source_summary.get("duplicate_execution_cells", 0) or 0),
        "independent_contexts": int(context_diversity.get("independent_contexts", 0) or 0),
        "context_diversity_assessment": str(context_diversity.get("assessment", "")),
        "support_events_counted_as_scientific_support": False,
        "duplicate_links_counted_as_independent": False,
        "contextual_followup_required_before_symbolic_model": bool(
            context_diversity.get("requires_contextual_followup_before_symbolic_model")
        ),
        "execution_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def independent_context_count(payload: Mapping[str, Any]) -> int:
    signatures = {
        str(row.get("replay_policy", "")) or "unknown"
        for row in payload.get("execution_cells", []) or []
    }
    return len(signatures)


def validate_generic_results_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.2 support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.2 summary support must remain 0")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M3.G0.2 must not perform revision")
    if int(payload.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("M3.G0.2 contains wrong confirmations")
    if bool(payload.get("semantic_interpretation_counted_as_confirmation", False)):
        raise ValueError("semantic interpretation cannot be confirmation")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.2 cannot write A32/A33")


def hypothesis_consolidation_id(request: Mapping[str, Any]) -> str:
    request_id = str(request.get("request_id", "unknown")).replace("::", "_")
    return f"m3g0_3::{request_id}"


def write_generic_mechanic_evidence_consolidation(
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_GENERIC_MECHANIC_EVIDENCE_CONSOLIDATION_OUTPUT_PATH,
) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate generic M3.G0 mechanic evidence candidate-only."
    )
    parser.add_argument(
        "--generic-results",
        default=str(DEFAULT_GENERIC_MECHANIC_EXPERIMENT_RESULTS_OUTPUT_PATH),
        help="Path to diagnostics/m3/generic_mechanic_experiment_results.json.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_GENERIC_MECHANIC_EVIDENCE_CONSOLIDATION_OUTPUT_PATH),
        help="Output path for M3.G0.3 evidence consolidation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_mechanic_evidence_consolidation(
        generic_results_path=args.generic_results,
    )
    write_generic_mechanic_evidence_consolidation(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "hypothesis_consolidations": payload["summary"][
                    "hypothesis_consolidations"
                ],
                "context_diversity_assessment": payload["summary"][
                    "context_diversity_assessment"
                ],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
