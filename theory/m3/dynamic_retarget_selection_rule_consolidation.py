"""M3.17 candidate-only consolidation of retarget selection-rule follow-ups."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .dynamic_retarget_selection_followup_executor import (
    DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS, METRIC_ORDER


DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "dynamic_retarget_selection_rule_consolidation.json"
)


@dataclass(frozen=True)
class RetargetSelectionRuleConsolidation:
    """Candidate-only summary of which selection rule currently explains best."""

    selection_rule_consolidation_id: str
    source_selection_rule_candidate_id: str
    source_mechanism_candidate_id: str
    game_id: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    best_current_rule_family: str
    confidence_basis: str
    important_caveat: str
    unique_new_successful_args: Tuple[Dict[str, Any], ...]
    known_successful_retargets: Tuple[Dict[str, Any], ...]
    known_failed_retargets: Tuple[Dict[str, Any], ...]
    excluded_args_for_next_expansion: Tuple[Dict[str, Any], ...]
    duplicate_resolution_counted_as_independent: bool
    success_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    diagnostic_metric_interpretation: Dict[str, Any]
    rule_family_assessments: Tuple[Dict[str, Any], ...]
    row_or_band_directly_tested: bool
    row_or_band_not_directly_contradicted: bool
    status: str = "UNRESOLVED"
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False
    rule_counted_as_confirmation: bool = False
    blocked_probe_counted_as_contradiction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_rule_consolidation_id": self.selection_rule_consolidation_id,
            "source_selection_rule_candidate_id": self.source_selection_rule_candidate_id,
            "source_mechanism_candidate_id": self.source_mechanism_candidate_id,
            "game_id": self.game_id,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "best_current_rule_family": self.best_current_rule_family,
            "confidence_basis": self.confidence_basis,
            "important_caveat": self.important_caveat,
            "unique_new_successful_args": [
                dict(item) for item in self.unique_new_successful_args
            ],
            "known_successful_retargets": [
                dict(item) for item in self.known_successful_retargets
            ],
            "known_failed_retargets": [
                dict(item) for item in self.known_failed_retargets
            ],
            "excluded_args_for_next_expansion": [
                dict(item) for item in self.excluded_args_for_next_expansion
            ],
            "duplicate_resolution_counted_as_independent": (
                self.duplicate_resolution_counted_as_independent
            ),
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "diagnostic_metric_interpretation": dict(
                self.diagnostic_metric_interpretation
            ),
            "rule_family_assessments": [
                dict(item) for item in self.rule_family_assessments
            ],
            "row_or_band_directly_tested": self.row_or_band_directly_tested,
            "row_or_band_not_directly_contradicted": (
                self.row_or_band_not_directly_contradicted
            ),
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
            "rule_counted_as_confirmation": self.rule_counted_as_confirmation,
            "blocked_probe_counted_as_contradiction": (
                self.blocked_probe_counted_as_contradiction
            ),
        }


def run_dynamic_retarget_selection_rule_consolidation(
    *,
    selection_followup_results_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(selection_followup_results_path)
    consolidations = consolidate_selection_followup_results(payload)
    return {
        "config": {
            "selection_followup_results_path": str(selection_followup_results_path),
            "schema_version": "m3.dynamic_retarget_selection_rule_consolidation.v1",
            "inputs_read": ["M3.16"],
            "artifacts_not_modified": [
                "M2",
                "M3.8",
                "M3.9",
                "M3.10",
                "M3.11",
                "M3.12",
                "M3.13",
                "M3.14",
                "M3.15",
                "M3.16",
                "A32",
                "A33",
            ],
            "consolidation_policy": {
                "blocked_probes_are_not_contradictions": True,
                "duplicate_resolved_args_are_not_independent": True,
                "diagnostic_metrics_do_not_decide_retarget_success": True,
                "support_forced_to_zero": True,
                "execution_performed": False,
            },
        },
        "summary": summarize_consolidations(
            source_payload=payload,
            consolidations=consolidations,
        ),
        "selection_rule_consolidations": [
            consolidation.to_dict() for consolidation in consolidations
        ],
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
        "blocked_probe_counted_as_contradiction": False,
    }


def consolidate_selection_followup_results(
    payload: Mapping[str, Any],
) -> Tuple[RetargetSelectionRuleConsolidation, ...]:
    per_request = [
        dict(row) for row in payload.get("per_request_results", []) or []
    ]
    if not per_request:
        return ()
    experiments = [
        dict(row) for row in payload.get("controlled_experiments", []) or []
    ]
    rule_summary = [
        dict(row) for row in payload.get("rule_family_summary", []) or []
    ]
    resolutions = [
        dict(row) for row in payload.get("target_arg_resolutions", []) or []
    ]
    summary = dict(payload.get("summary", {}) or {})

    first_observation = first_executed_row(experiments) or first_request_row(per_request)
    source_selection_rule_id = str(
        first_observation.get("source_selection_rule_candidate_id", "")
    )
    source_mechanism_id = str(
        first_observation.get("source_mechanism_candidate_id", "")
    )
    game_id = str(first_observation.get("game_id", ""))
    target_action = str(first_observation.get("target_action", ""))
    context_replay = tuple(
        str(item) for item in first_observation.get("context_replay", []) or []
    )
    context_replay_args = _context_args_tuple(
        first_observation.get("context_replay_args")
    )

    success_args = unique_successful_args(per_request)
    known_successes = known_retargets_from_experiments(
        experiments,
        field_name="known_successful_retargets",
    )
    known_failures = known_retargets_from_experiments(
        experiments,
        field_name="known_failed_retargets",
    )
    replay_target_args = replay_args_for_target(
        context_replay=context_replay,
        context_replay_args=context_replay_args,
        target_action=target_action,
    )
    excluded_for_next = dedupe_args(
        (
            *known_successes,
            *known_failures,
            *success_args,
            *replay_target_args,
        )
    )
    success_metrics = tuple(
        _metric_sort(
            {
                str(metric)
                for row in per_request
                for metric in row.get("grounded_success_metrics", []) or []
            }
        )
    )
    diagnostic_metrics = tuple(
        _metric_sort(
            {
                str(metric)
                for row in experiments
                for metric in row.get("diagnostic_metrics", []) or []
            }
        )
    )
    best_family = best_current_rule_family(rule_summary)
    row_or_band_tested = rule_family_controlled_experiments(
        rule_summary,
        "row_or_band_dependent_retarget",
    ) > 0
    consolidation = RetargetSelectionRuleConsolidation(
        selection_rule_consolidation_id=selection_rule_consolidation_id(
            source_selection_rule_id=source_selection_rule_id,
            game_id=game_id,
            target_action=target_action,
        ),
        source_selection_rule_candidate_id=source_selection_rule_id,
        source_mechanism_candidate_id=source_mechanism_id,
        game_id=game_id,
        context_replay=context_replay,
        context_replay_args=context_replay_args,
        target_action=target_action,
        best_current_rule_family=best_family,
        confidence_basis=confidence_basis_for(best_family, rule_summary, summary),
        important_caveat=important_caveat_for(rule_summary),
        unique_new_successful_args=tuple(success_args),
        known_successful_retargets=tuple(known_successes),
        known_failed_retargets=tuple(known_failures),
        excluded_args_for_next_expansion=tuple(excluded_for_next),
        duplicate_resolution_counted_as_independent=bool(
            summary.get("duplicate_resolved_target_arg_sets_counted_as_independent")
        ),
        success_metrics=success_metrics,
        diagnostic_metrics=diagnostic_metrics,
        diagnostic_metric_interpretation=diagnostic_metric_interpretation(
            experiments=experiments,
            success_metrics=success_metrics,
            diagnostic_metrics=diagnostic_metrics,
        ),
        rule_family_assessments=tuple(
            build_rule_family_assessments(
                rule_summary=rule_summary,
                resolutions=resolutions,
                summary=summary,
            )
        ),
        row_or_band_directly_tested=row_or_band_tested,
        row_or_band_not_directly_contradicted=not row_or_band_tested,
    )
    return (consolidation,)


def build_rule_family_assessments(
    *,
    rule_summary: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    assessments: list[Dict[str, Any]] = []
    for row in sorted(rule_summary, key=lambda item: str(item.get("rule_family", ""))):
        family = str(row.get("rule_family", ""))
        controlled = int(row.get("controlled_experiments_run", 0) or 0)
        blocked = int(row.get("blocked_requests", 0) or 0)
        support_events = int(row.get("success_metric_support_events", 0) or 0)
        if family == "row_or_band_dependent_retarget" and controlled == 0:
            assessment = "not_directly_tested_blocked_or_excluded"
            direct_contradiction = False
        elif support_events > 0:
            assessment = "supported_candidate_only_by_executed_followups"
            direct_contradiction = False
        else:
            assessment = "unresolved_no_grounded_support"
            direct_contradiction = False
        assessments.append(
            {
                "rule_family": family,
                "assessment": assessment,
                "controlled_experiments_run": controlled,
                "blocked_requests": blocked,
                "requests_with_success_metric_support": int(
                    row.get("requests_with_success_metric_support", 0) or 0
                ),
                "success_metric_support_events": support_events,
                "success_metric_contradiction_events": int(
                    row.get("success_metric_contradiction_events", 0) or 0
                ),
                "diagnostic_support_events": int(
                    row.get("diagnostic_support_events", 0) or 0
                ),
                "diagnostic_contradiction_events": int(
                    row.get("diagnostic_contradiction_events", 0) or 0
                ),
                "blocked_reason_counts": blocked_reason_counts(
                    resolutions,
                    rule_family=family,
                ),
                "directly_tested": controlled > 0,
                "directly_contradicted": direct_contradiction,
                "blocked_probe_counted_as_contradiction": False,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "wrong_confirmations": 0,
            }
        )
    if int(summary.get("diagnostic_contradiction_events", 0) or 0) > 0:
        assessments.append(
            {
                "rule_family": "specific_effect_over_global_pixels",
                "assessment": "methodological_support_candidate_only",
                "basis": (
                    "Diagnostic metrics produced contradictions while grounded "
                    "success metrics supported the retarget."
                ),
                "directly_tested": False,
                "directly_contradicted": False,
                "changed_pixels_role": "effect_radar_not_retarget_success_metric",
                "diagnostic_contradiction_events": int(
                    summary.get("diagnostic_contradiction_events", 0) or 0
                ),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "wrong_confirmations": 0,
            }
        )
    return tuple(assessments)


def summarize_consolidations(
    *,
    source_payload: Mapping[str, Any],
    consolidations: Sequence[RetargetSelectionRuleConsolidation],
) -> Dict[str, Any]:
    summary = dict(source_payload.get("summary", {}) or {})
    best_families = {
        item.best_current_rule_family for item in consolidations if item.best_current_rule_family
    }
    return {
        "selection_followup_results_consumed": 1 if source_payload else 0,
        "selection_rule_consolidations": len(consolidations),
        "best_current_rule_families": sorted(best_families),
        "unique_new_successful_args": sum(
            len(item.unique_new_successful_args) for item in consolidations
        ),
        "row_or_band_directly_tested": any(
            item.row_or_band_directly_tested for item in consolidations
        ),
        "row_or_band_not_directly_contradicted": all(
            item.row_or_band_not_directly_contradicted for item in consolidations
        )
        if consolidations
        else True,
        "blocked_probe_counted_as_contradiction": False,
        "duplicate_resolution_counted_as_independent": any(
            item.duplicate_resolution_counted_as_independent
            for item in consolidations
        ),
        "source_success_metric_support_events": int(
            summary.get("success_metric_support_events", 0) or 0
        ),
        "source_diagnostic_contradiction_events": int(
            summary.get("diagnostic_contradiction_events", 0) or 0
        ),
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_remains_only_verdict_location": True,
    }


def unique_successful_args(
    per_request: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    values: dict[str, Dict[str, Any]] = {}
    for row in per_request:
        if not bool(row.get("request_has_success_metric_support")):
            continue
        for args in row.get("resolved_target_action_args", []) or []:
            if isinstance(args, Mapping):
                values[_args_key(args)] = dict(args)
    return tuple(values[key] for key in sorted(values))


def known_retargets_from_experiments(
    experiments: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> Tuple[Dict[str, Any], ...]:
    values: list[Dict[str, Any]] = []
    for row in experiments:
        for args in row.get(field_name, []) or []:
            if isinstance(args, Mapping):
                values.append(dict(args))
    return tuple(dedupe_args(values))


def replay_args_for_target(
    *,
    context_replay: Sequence[str],
    context_replay_args: Sequence[Mapping[str, Any]] | None,
    target_action: str,
) -> Tuple[Dict[str, Any], ...]:
    if context_replay_args is None:
        return ()
    values: list[Dict[str, Any]] = []
    for action, args in zip(context_replay, context_replay_args):
        if str(action) == str(target_action) and args:
            values.append(dict(args))
    return tuple(dedupe_args(values))


def dedupe_args(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_args_key(dict(value)): dict(value) for value in values if value}
    return tuple(by_key[key] for key in sorted(by_key))


def best_current_rule_family(rule_summary: Sequence[Mapping[str, Any]]) -> str:
    scored = sorted(
        rule_summary,
        key=lambda row: (
            int(row.get("requests_with_success_metric_support", 0) or 0),
            int(row.get("success_metric_support_events", 0) or 0),
            int(row.get("controlled_experiments_run", 0) or 0),
            -int(row.get("blocked_requests", 0) or 0),
            str(row.get("rule_family", "")),
        ),
        reverse=True,
    )
    return str(scored[0].get("rule_family", "")) if scored else ""


def confidence_basis_for(
    best_family: str,
    rule_summary: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    if (
        best_family == "local_patch_transformability"
        and int(summary.get("success_metric_support_events", 0) or 0) > 0
    ):
        return "only_executed_followups_support_patch_similarity"
    if not rule_summary:
        return "no_followups_available"
    return "candidate_only_followup_balance"


def important_caveat_for(rule_summary: Sequence[Mapping[str, Any]]) -> str:
    row_controlled = rule_family_controlled_experiments(
        rule_summary,
        "row_or_band_dependent_retarget",
    )
    if row_controlled == 0:
        return (
            "row_or_band probes were mostly not executable, so they are not "
            "direct contradictions"
        )
    return "row_or_band probes require more discriminating follow-ups"


def diagnostic_metric_interpretation(
    *,
    experiments: Sequence[Mapping[str, Any]],
    success_metrics: Sequence[str],
    diagnostic_metrics: Sequence[str],
) -> Dict[str, Any]:
    diagnostic_contradictions = {
        str(row.get("metric", ""))
        for row in experiments
        if str(row.get("metric_role", "")) == "diagnostic_metric"
        and int(row.get("contradiction_events", 0) or 0) > 0
    }
    return {
        "success_metrics": list(success_metrics),
        "diagnostic_metrics": list(diagnostic_metrics),
        "diagnostic_metrics_with_contradictions": _metric_sort(
            diagnostic_contradictions
        ),
        "changed_pixels_role": (
            "effect_radar_not_retarget_success_metric"
            if "changed_pixels" in diagnostic_metrics
            else "not_tested"
        ),
        "contact_graph_role": (
            "diagnostic_not_decisive_for_current_retarget"
            if "contact_graph_before_after" in diagnostic_metrics
            else "not_tested"
        ),
        "diagnostic_metrics_decide_retarget_success": False,
    }


def blocked_reason_counts(
    resolutions: Sequence[Mapping[str, Any]],
    *,
    rule_family: str,
) -> Dict[str, int]:
    counts: dict[str, int] = {}
    for row in resolutions:
        if str(row.get("rule_family", "")) != rule_family:
            continue
        reason = str(row.get("blocked_reason", "") or "")
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def rule_family_controlled_experiments(
    rule_summary: Sequence[Mapping[str, Any]],
    rule_family: str,
) -> int:
    return sum(
        int(row.get("controlled_experiments_run", 0) or 0)
        for row in rule_summary
        if str(row.get("rule_family", "")) == rule_family
    )


def first_executed_row(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    for row in rows:
        if int(row.get("controlled_experiments_run", 0) or 0) > 0:
            return dict(row)
    return None


def first_request_row(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(rows[0]) if rows else {}


def selection_rule_consolidation_id(
    *,
    source_selection_rule_id: str,
    game_id: str,
    target_action: str,
) -> str:
    parts = [part for part in str(source_selection_rule_id).split("::") if part]
    if len(parts) >= 3:
        return "::".join(["m3_17", parts[1], parts[2], "selection_rule_consolidation"])
    game_token = str(game_id).split("-", 1)[0] or "unknown_game"
    return "::".join(["m3_17", game_token, str(target_action), "selection_rule_consolidation"])


def write_dynamic_retarget_selection_rule_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_sort(metrics: set[str] | Sequence[str]) -> list[str]:
    def key(metric: str) -> Tuple[int, str]:
        try:
            return (METRIC_ORDER.index(metric), metric)
        except ValueError:
            return (len(METRIC_ORDER), metric)

    return sorted({str(metric) for metric in metrics if str(metric)}, key=key)


def _context_args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _args_key(args: Mapping[str, Any]) -> str:
    return json.dumps({str(key): args[key] for key in sorted(args)}, sort_keys=True)


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate M3.16 selection-rule follow-up evidence.",
    )
    parser.add_argument(
        "--selection-followup-results",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_selection_rule_consolidation(
        selection_followup_results_path=args.selection_followup_results,
    )
    write_dynamic_retarget_selection_rule_consolidation(payload, args.out)
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
