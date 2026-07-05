"""M3.24 light consolidation for A32 requested patch-similarity scope tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .a32_requested_patch_similarity_followup_executor import (
    ALTERNATE_REPOSITIONING_CONTEXT_PROBE,
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH,
    OUTSIDE_KNOWN_Y12_REGION_PROBE,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "a32_requested_patch_similarity_scope_consolidation.json"
)

SCOPE_EXPANDED_CANDIDATE_ONLY = "SCOPE_EXPANDED_CANDIDATE_ONLY"
SCOPE_LIMITED_REINFORCED_CANDIDATE_ONLY = "SCOPE_LIMITED_REINFORCED_CANDIDATE_ONLY"
SCOPE_CONTRADICTED_CANDIDATE_ONLY = "SCOPE_CONTRADICTED_CANDIDATE_ONLY"
SCOPE_UNRESOLVED_CANDIDATE_ONLY = "SCOPE_UNRESOLVED_CANDIDATE_ONLY"


def run_a32_requested_patch_similarity_scope_consolidation(
    *,
    a32_followup_results_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(a32_followup_results_path)
    consolidation = consolidate_patch_similarity_scope(payload)
    consolidations = [consolidation] if consolidation else []
    return {
        "config": {
            "a32_followup_results_path": str(a32_followup_results_path),
            "schema_version": "m3.a32_requested_patch_similarity_scope_consolidation.v1",
            "inputs_read": ["M3.23"],
            "artifacts_not_modified": ["A32", "A33"],
            "execution_performed": False,
            "consolidation_policy": {
                "scope_expansion_is_candidate_only": True,
                "outside_boundary_failures_are_not_global_refutation": True,
                "diagnostic_contradictions_do_not_refute_success": True,
                "a33_ready_forced_false": True,
            },
        },
        "summary": summarize_scope_consolidations(consolidations),
        "scope_consolidations": consolidations,
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
        "rule_counted_as_confirmation": False,
        "scope_expansion_counted_as_confirmation": False,
    }


def consolidate_patch_similarity_scope(
    payload: Mapping[str, Any],
) -> Dict[str, Any] | None:
    family_summary = {
        str(row.get("followup_family", "")): dict(row)
        for row in payload.get("family_summary", []) or []
        if isinstance(row, Mapping)
    }
    per_signature = [
        dict(row)
        for row in payload.get("per_signature_execution", []) or []
        if isinstance(row, Mapping)
    ]
    if not family_summary and not per_signature:
        return None

    outside = family_summary.get(OUTSIDE_KNOWN_Y12_REGION_PROBE, {})
    alternate = family_summary.get(ALTERNATE_REPOSITIONING_CONTEXT_PROBE, {})
    carried_success_args = seed_args_from_experiments(
        payload.get("controlled_experiments", []) or [],
        "seed_successful_args",
    )
    carried_failed_args = seed_args_from_experiments(
        payload.get("controlled_experiments", []) or [],
        "seed_failed_args",
    )
    alternate_success_args = target_args_for_family(
        per_signature,
        ALTERNATE_REPOSITIONING_CONTEXT_PROBE,
        require_success=True,
    )
    outside_boundary_args = target_args_for_family(
        per_signature,
        OUTSIDE_KNOWN_Y12_REGION_PROBE,
        require_success=False,
    )
    initial_context_supported = len(carried_success_args) >= 2
    alternate_context_supported = (
        int(alternate.get("args_with_success_metric_support", 0) or 0) > 0
        and int(alternate.get("success_metric_contradiction_events", 0) or 0) == 0
    )
    outside_region_boundary_reinforced = (
        int(outside.get("args_with_success_metric_contradiction", 0) or 0) > 0
        and int(outside.get("args_with_success_metric_support", 0) or 0) == 0
        and not bool(outside.get("outside_boundary_failures_counted_as_rule_refutation"))
    )
    scope_assessment = assess_scope(
        initial_context_supported=initial_context_supported,
        alternate_context_supported=alternate_context_supported,
        outside_region_boundary_reinforced=outside_region_boundary_reinforced,
        alternate_success_metric_contradictions=int(
            alternate.get("success_metric_contradiction_events", 0) or 0
        ),
    )
    first_experiment = first_mapping(payload.get("controlled_experiments", []) or [])
    first_signature = first_mapping(per_signature)
    success_metrics = metrics_from_payload(payload, role="success_metrics")
    diagnostic_metrics = metrics_from_payload(payload, role="diagnostic_metrics")
    return {
        "scope_consolidation_id": scope_consolidation_id(first_signature),
        "source_a32_queue_item_id": str(
            first_signature.get("source_a32_queue_item_id", "")
        ),
        "source_a32_decision": str(first_signature.get("source_a32_decision", "")),
        "source_a32_recommended_next_step": str(
            first_signature.get("source_a32_recommended_next_step", "")
        ),
        "a32_decision_counted_as_confirmation": False,
        "game_id": str(first_signature.get("game_id", first_experiment.get("game_id", ""))),
        "candidate_rule_family": str(
            first_signature.get(
                "candidate_rule_family",
                first_experiment.get("candidate_rule_family", ""),
            )
        ),
        "target_action": str(
            first_signature.get("target_action", first_experiment.get("target_action", ""))
        ),
        "scope_assessment": scope_assessment,
        "initial_context_supported": initial_context_supported,
        "initial_context_support_source": "carried_from_m3_20_seed_successful_args",
        "alternate_context_supported": alternate_context_supported,
        "outside_region_boundary_reinforced": outside_region_boundary_reinforced,
        "known_success_args": [dict(row) for row in carried_success_args],
        "known_failed_args": [dict(row) for row in carried_failed_args],
        "alternate_context_success_args": [dict(row) for row in alternate_success_args],
        "outside_boundary_args": [dict(row) for row in outside_boundary_args],
        "supported_contexts": supported_contexts(per_signature),
        "boundary_contexts": boundary_contexts(per_signature),
        "success_metrics": success_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "changed_pixels_role": "effect_radar_not_success_metric",
        "source_success_metric_support_events": int(
            payload.get("summary", {}).get("success_metric_support_events", 0) or 0
        ),
        "source_success_metric_contradiction_events": int(
            payload.get("summary", {}).get("success_metric_contradiction_events", 0)
            or 0
        ),
        "source_diagnostic_contradiction_events": int(
            payload.get("summary", {}).get("diagnostic_contradiction_events", 0) or 0
        ),
        "outside_boundary_failures_counted_as_rule_refutation": False,
        "diagnostic_contradictions_counted_as_refutation": False,
        "a33_ready": False,
        "ready_for_agent_policy_probe": (
            scope_assessment == SCOPE_EXPANDED_CANDIDATE_ONLY
        ),
        "agent_policy_probe_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "recommended_agent_policy_probe": {
            "use_repositioning_action": "ACTION4",
            "select_target_action": "ACTION6",
            "prefer_patch_similar_success_like_args": True,
            "avoid_known_failure_like_args": [dict(row) for row in carried_failed_args],
            "do_not_treat_candidate_as_confirmed_skill": True,
        },
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
        "rule_counted_as_confirmation": False,
        "scope_expansion_counted_as_confirmation": False,
    }


def assess_scope(
    *,
    initial_context_supported: bool,
    alternate_context_supported: bool,
    outside_region_boundary_reinforced: bool,
    alternate_success_metric_contradictions: int,
) -> str:
    if (
        initial_context_supported
        and alternate_context_supported
        and outside_region_boundary_reinforced
    ):
        return SCOPE_EXPANDED_CANDIDATE_ONLY
    if alternate_success_metric_contradictions > 0 and not alternate_context_supported:
        return SCOPE_CONTRADICTED_CANDIDATE_ONLY
    if initial_context_supported and outside_region_boundary_reinforced:
        return SCOPE_LIMITED_REINFORCED_CANDIDATE_ONLY
    return SCOPE_UNRESOLVED_CANDIDATE_ONLY


def summarize_scope_consolidations(
    consolidations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "scope_consolidations": len(consolidations),
        "scope_assessments": sorted(
            {
                str(row.get("scope_assessment", ""))
                for row in consolidations
                if row.get("scope_assessment")
            }
        ),
        "scope_expanded_candidate_only": len(
            [
                row
                for row in consolidations
                if row.get("scope_assessment") == SCOPE_EXPANDED_CANDIDATE_ONLY
            ]
        ),
        "ready_for_agent_policy_probe": any(
            bool(row.get("ready_for_agent_policy_probe")) for row in consolidations
        ),
        "a33_ready": any(bool(row.get("a33_ready")) for row in consolidations),
        "a33_write_performed": False,
        "a32_write_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "scope_expansion_counted_as_confirmation": False,
    }


def target_args_for_family(
    per_signature: Sequence[Mapping[str, Any]],
    family: str,
    *,
    require_success: bool,
) -> Tuple[Dict[str, Any], ...]:
    values = []
    for row in per_signature:
        if family not in (row.get("followup_families", []) or []):
            continue
        has_success = int(row.get("success_metric_support_events", 0) or 0) > 0
        if require_success and not has_success:
            continue
        if row.get("target_action_args"):
            values.append(dict(row.get("target_action_args", {}) or {}))
    return dedupe_args(values)


def supported_contexts(per_signature: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    contexts = []
    for row in per_signature:
        if int(row.get("success_metric_support_events", 0) or 0) <= 0:
            continue
        contexts.append(
            {
                "context_replay": list(row.get("context_replay", []) or []),
                "context_replay_args": _context_args_list(
                    row.get("context_replay_args")
                ),
                "target_action_args": dict(row.get("target_action_args", {}) or {}),
                "scope_interpretation": str(row.get("scope_interpretation", "")),
            }
        )
    return dedupe_context_rows(contexts)


def boundary_contexts(per_signature: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    contexts = []
    for row in per_signature:
        if OUTSIDE_KNOWN_Y12_REGION_PROBE not in (
            row.get("followup_families", []) or []
        ):
            continue
        contexts.append(
            {
                "context_replay": list(row.get("context_replay", []) or []),
                "context_replay_args": _context_args_list(
                    row.get("context_replay_args")
                ),
                "target_action_args": dict(row.get("target_action_args", {}) or {}),
                "scope_interpretation": str(row.get("scope_interpretation", "")),
            }
        )
    return dedupe_context_rows(contexts)


def metrics_from_payload(payload: Mapping[str, Any], *, role: str) -> list[str]:
    values: list[str] = []
    for row in payload.get("controlled_experiments", []) or []:
        if not isinstance(row, Mapping):
            continue
        for metric in row.get(role, []) or []:
            text = str(metric)
            if text and text not in values:
                values.append(text)
    if values:
        return values
    for row in payload.get("family_summary", []) or []:
        if not isinstance(row, Mapping):
            continue
        for metric in row.get(role, []) or []:
            text = str(metric)
            if text and text not in values:
                values.append(text)
    return values


def seed_args_from_experiments(
    experiments: Sequence[Any],
    field_name: str,
) -> Tuple[Dict[str, Any], ...]:
    values = []
    for row in experiments:
        if not isinstance(row, Mapping):
            continue
        for item in row.get(field_name, []) or []:
            if isinstance(item, Mapping):
                values.append(dict(item))
    return dedupe_args(values)


def first_mapping(values: Sequence[Any]) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def scope_consolidation_id(first_signature: Mapping[str, Any]) -> str:
    game = str(first_signature.get("game_id", "unknown-game"))
    target = str(first_signature.get("target_action", "unknown-target"))
    return f"m3_24::{game}::patch_similarity_scope::{target}"


def dedupe_args(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_stable_json(dict(value)): dict(value) for value in values if value}
    return tuple(by_key[key] for key in sorted(by_key))


def dedupe_context_rows(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_stable_json(dict(value)): dict(value) for value in values}
    return tuple(by_key[key] for key in sorted(by_key))


def _context_args_list(raw: Any) -> list[Dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_a32_requested_patch_similarity_scope_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
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
        description="Consolidate M3.24 patch-similarity scope follow-up results.",
    )
    parser.add_argument(
        "--followup-results",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_requested_patch_similarity_scope_consolidation(
        a32_followup_results_path=args.followup_results,
    )
    write_a32_requested_patch_similarity_scope_consolidation(payload, args.out)
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
