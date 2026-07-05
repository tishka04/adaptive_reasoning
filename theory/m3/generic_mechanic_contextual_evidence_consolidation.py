"""M3.G0.6 candidate-only consolidation of contextual generic evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .generic_mechanic_contextual_followup_executor import (
    DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_RESULTS_OUTPUT_PATH,
    CONTEXT_REPRODUCED_CANDIDATE_ONLY,
    CONTEXT_FAILED_CANDIDATE_ONLY,
    MIXED_CONTEXT_CANDIDATE_ONLY,
    TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY,
    MULTI_PREFIX_CONTEXTS,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_EVIDENCE_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "generic_mechanic_contextual_evidence_consolidation.json"
)

CONTEXT_STABLE_CANDIDATE_ONLY = "CONTEXT_STABLE_CANDIDATE_ONLY"
CONTEXT_MIXED_CANDIDATE_ONLY = "CONTEXT_MIXED_CANDIDATE_ONLY"
TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY = (
    "TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY"
)
CONTRADICTION_REQUIRES_FOLLOWUP = "CONTRADICTION_REQUIRES_FOLLOWUP"
READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY = (
    "READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY"
)


def run_generic_mechanic_contextual_evidence_consolidation(
    *,
    contextual_results_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_RESULTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(contextual_results_path)
    validate_contextual_results_source(payload)
    hypothesis_records = consolidate_contextual_hypothesis_records(payload)
    contradiction_records = contradiction_followup_records(hypothesis_records)
    family_records = consolidate_contextual_family_records(hypothesis_records)
    symbolic_readiness = symbolic_model_readiness(family_records, contradiction_records)
    return {
        "config": {
            "contextual_results_path": str(contextual_results_path),
            "schema_version": "m3.generic_mechanic_contextual_evidence_consolidation.v1",
            "inputs_read": ["M3.G0.5"],
            "artifacts_not_modified": ["M1", "M2", "A32", "A33"],
            "execution_performed": False,
            "consolidation_policy": {
                "contextual_events_are_not_scientific_support": True,
                "semantic_interpretation_for_dynamic_invariants_remains_unknown": True,
                "contradictions_are_localized_not_global_refutations": True,
                "symbolic_model_readiness_is_candidate_only": True,
            },
        },
        "summary": summarize_contextual_consolidation(
            source_payload=payload,
            hypothesis_records=hypothesis_records,
            family_records=family_records,
            contradiction_records=contradiction_records,
            symbolic_readiness=symbolic_readiness,
        ),
        "hypothesis_consolidations": hypothesis_records,
        "family_consolidations": family_records,
        "contradiction_followups": contradiction_records,
        "symbolic_model_readiness": symbolic_readiness,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "contextual_events_counted_as_scientific_support": False,
        "semantic_interpretation_counted_as_confirmation": False,
        "contradiction_counted_as_refutation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def consolidate_contextual_hypothesis_records(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    summary = dict(payload.get("summary", {}) or {})
    independent_contexts = int(summary.get("independent_contexts", 0) or 0)
    diversity = str(summary.get("context_diversity_assessment", ""))
    records: list[Dict[str, Any]] = []
    for row in payload.get("request_results", []) or []:
        request = dict(row)
        raw_status = str(request.get("candidate_status", ""))
        support_events = int(request.get("support_events", 0) or 0)
        contradiction_events = int(request.get("contradiction_events", 0) or 0)
        neutral_events = int(request.get("neutral_events", 0) or 0)
        status = contextual_candidate_status(
            raw_status=raw_status,
            support_events=support_events,
            contradiction_events=contradiction_events,
            neutral_events=neutral_events,
        )
        records.append(
            {
                "consolidation_id": contextual_consolidation_id(request),
                "request_id": str(request.get("request_id", "")),
                "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
                "source_mechanic_family": str(request.get("source_mechanic_family", "")),
                "followup_family": str(request.get("followup_family", "")),
                "candidate_status": status,
                "source_candidate_status": raw_status,
                "observation_interpretation": str(
                    request.get("observation_interpretation", "")
                ),
                "raw_support_events": support_events,
                "raw_contradiction_events": contradiction_events,
                "raw_neutral_events": neutral_events,
                "cells_linked": int(request.get("cells_linked", 0) or 0),
                "executed_cells": int(request.get("executed_cells", 0) or 0),
                "blocked_cells": int(request.get("blocked_cells", 0) or 0),
                "request_independent_contexts": int(
                    request.get("independent_contexts", 0) or 0
                ),
                "source_independent_contexts": independent_contexts,
                "context_diversity_assessment": diversity,
                "remaining_semantics_unknown": request.get(
                    "remaining_semantics_unknown"
                ),
                "semantic_interpretation": "unknown",
                "contextual_events_counted_as_scientific_support": False,
                "contradiction_counted_as_refutation": False,
                "followup_required_for_contradiction": contradiction_events > 0,
                "ready_for_symbolic_model_candidate_only": (
                    status
                    in {
                        CONTEXT_STABLE_CANDIDATE_ONLY,
                        TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY,
                    }
                    and diversity == MULTI_PREFIX_CONTEXTS
                ),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                "observation_counted_as_confirmation": False,
            }
        )
    return tuple(records)


def contextual_candidate_status(
    *,
    raw_status: str,
    support_events: int,
    contradiction_events: int,
    neutral_events: int,
) -> str:
    if contradiction_events > 0 and support_events > 0:
        return CONTEXT_MIXED_CANDIDATE_ONLY
    if contradiction_events > 0:
        return (
            TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY
            if raw_status == CONTEXT_FAILED_CANDIDATE_ONLY
            else CONTRADICTION_REQUIRES_FOLLOWUP
        )
    if raw_status == TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY:
        return TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY
    if raw_status == CONTEXT_REPRODUCED_CANDIDATE_ONLY and support_events > 0:
        return CONTEXT_STABLE_CANDIDATE_ONLY
    if raw_status == MIXED_CONTEXT_CANDIDATE_ONLY or neutral_events > 0:
        return CONTEXT_MIXED_CANDIDATE_ONLY
    return CONTRADICTION_REQUIRES_FOLLOWUP


def contradiction_followup_records(
    hypothesis_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    records: list[Dict[str, Any]] = []
    for row in hypothesis_records:
        if int(row.get("raw_contradiction_events", 0) or 0) <= 0:
            continue
        records.append(
            {
                "contradiction_id": f"m3g0_6::contradiction::{len(records) + 1:03d}",
                "source_consolidation_id": str(row.get("consolidation_id", "")),
                "source_request_id": str(row.get("request_id", "")),
                "source_hypothesis_id": str(row.get("source_hypothesis_id", "")),
                "source_mechanic_family": str(row.get("source_mechanic_family", "")),
                "candidate_status": str(row.get("candidate_status", "")),
                "raw_contradiction_events": int(
                    row.get("raw_contradiction_events", 0) or 0
                ),
                "observation_interpretation": str(
                    row.get("observation_interpretation", "")
                ),
                "contradiction_counted_as_refutation": False,
                "followup_required_for_contradiction": True,
                "recommended_followup_family": "contextual_contradiction_disambiguation",
                "semantic_interpretation": "unknown",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "wrong_confirmations": 0,
            }
        )
    return tuple(records)


def consolidate_contextual_family_records(
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
        family_status = contextual_family_status(statuses)
        records.append(
            {
                "family": family,
                "candidate_status": family_status,
                "hypotheses": len(rows),
                "status_counts": dict(sorted(statuses.items())),
                "raw_support_events": int(raw_support),
                "raw_contradiction_events": int(raw_contradiction),
                "contextual_events_counted_as_scientific_support": False,
                "contradiction_counted_as_refutation": False,
                "ready_for_symbolic_model_candidate_only": family_status
                == READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY,
                "semantic_interpretation": "unknown",
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "wrong_confirmations": 0,
                "family_consolidation_counted_as_confirmation": False,
            }
        )
    return tuple(records)


def contextual_family_status(statuses: Mapping[str, int]) -> str:
    if int(statuses.get(CONTEXT_MIXED_CANDIDATE_ONLY, 0) or 0) > 0:
        return CONTEXT_MIXED_CANDIDATE_ONLY
    if int(statuses.get(TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY, 0) or 0) > 0 and int(
        statuses.get(CONTEXT_STABLE_CANDIDATE_ONLY, 0) or 0
    ) == 0 and int(statuses.get(TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY, 0) or 0) == 0:
        return TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY
    if int(statuses.get(CONTEXT_STABLE_CANDIDATE_ONLY, 0) or 0) > 0 or int(
        statuses.get(TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY, 0) or 0
    ) > 0:
        return READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY
    if int(statuses.get(CONTRADICTION_REQUIRES_FOLLOWUP, 0) or 0) > 0:
        return CONTRADICTION_REQUIRES_FOLLOWUP
    return CONTEXT_MIXED_CANDIDATE_ONLY


def symbolic_model_readiness(
    family_records: Sequence[Mapping[str, Any]],
    contradiction_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ready_families = [
        str(row.get("family", ""))
        for row in family_records
        if bool(row.get("ready_for_symbolic_model_candidate_only"))
    ]
    blocked_families = [
        str(row.get("family", ""))
        for row in family_records
        if int(row.get("raw_contradiction_events", 0) or 0) > 0
    ]
    return {
        "ready_for_symbolic_model_candidate_only": bool(ready_families),
        "ready_family_count": len(ready_families),
        "ready_families": ready_families,
        "blocked_or_followup_families": blocked_families,
        "contradictions_require_followup": len(contradiction_records) > 0,
        "symbolic_model_induction_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def summarize_contextual_consolidation(
    *,
    source_payload: Mapping[str, Any],
    hypothesis_records: Sequence[Mapping[str, Any]],
    family_records: Sequence[Mapping[str, Any]],
    contradiction_records: Sequence[Mapping[str, Any]],
    symbolic_readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    source_summary = dict(source_payload.get("summary", {}) or {})
    status_counts = Counter(str(row.get("candidate_status", "")) for row in hypothesis_records)
    return {
        "contextual_followup_requests_consumed": int(
            source_summary.get(
                "contextual_followup_requests_consumed",
                len(hypothesis_records),
            )
            or 0
        ),
        "hypothesis_consolidations": len(hypothesis_records),
        "family_consolidations": len(family_records),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "source_support_events": int(source_summary.get("support_events", 0) or 0),
        "source_contradiction_events": int(
            source_summary.get("contradiction_events", 0) or 0
        ),
        "source_neutral_events": int(source_summary.get("neutral_events", 0) or 0),
        "unique_contextual_execution_cells": int(
            source_summary.get("unique_contextual_execution_cells", 0) or 0
        ),
        "planned_contextual_cells": int(
            source_summary.get("planned_contextual_cells", 0) or 0
        ),
        "duplicate_contextual_cells": int(
            source_summary.get("duplicate_contextual_cells", 0) or 0
        ),
        "duplicate_contextual_cells_counted_as_independent": False,
        "independent_contexts": int(source_summary.get("independent_contexts", 0) or 0),
        "context_diversity_assessment": str(
            source_summary.get("context_diversity_assessment", "")
        ),
        "contradiction_followups": len(contradiction_records),
        "contradiction_counted_as_refutation": False,
        "contextual_events_counted_as_scientific_support": False,
        "ready_for_symbolic_model_candidate_only": bool(
            symbolic_readiness.get("ready_for_symbolic_model_candidate_only")
        ),
        "symbolic_model_induction_performed": False,
        "execution_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_contextual_results_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.5 support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.5 summary support must remain 0")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M3.G0.5 must not perform revision")
    if int(payload.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("M3.G0.5 contains wrong confirmations")
    if bool(payload.get("contextual_signal_counted_as_scientific_support", False)):
        raise ValueError("contextual signals cannot be scientific support")
    if bool(payload.get("semantic_interpretation_counted_as_confirmation", False)):
        raise ValueError("semantic interpretation cannot be confirmation")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.5 cannot write A32/A33")


def contextual_consolidation_id(request: Mapping[str, Any]) -> str:
    request_id = str(request.get("request_id", "unknown")).replace("::", "_")
    return f"m3g0_6::{request_id}"


def write_generic_mechanic_contextual_evidence_consolidation(
    payload: Mapping[str, Any],
    out_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_EVIDENCE_CONSOLIDATION_OUTPUT_PATH
    ),
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
        description="Consolidate contextual M3.G0 mechanic evidence candidate-only."
    )
    parser.add_argument(
        "--contextual-results",
        default=str(DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_RESULTS_OUTPUT_PATH),
        help="Path to diagnostics/m3/generic_mechanic_contextual_followup_results.json.",
    )
    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_EVIDENCE_CONSOLIDATION_OUTPUT_PATH
        ),
        help="Output path for M3.G0.6 contextual evidence consolidation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_mechanic_contextual_evidence_consolidation(
        contextual_results_path=args.contextual_results,
    )
    write_generic_mechanic_contextual_evidence_consolidation(payload, args.out)
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
                "ready_for_symbolic_model_candidate_only": payload["summary"][
                    "ready_for_symbolic_model_candidate_only"
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
