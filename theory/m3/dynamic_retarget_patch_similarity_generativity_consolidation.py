"""M3.20 candidate-only consolidation of patch-similarity generativity."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .dynamic_retarget_patch_similarity_expansion_executor import (
    DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS, METRIC_ORDER


DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "dynamic_retarget_patch_similarity_generativity_consolidation.json"
)

PATCH_GENERATIVITY_STATUS = (
    "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
)


@dataclass(frozen=True)
class PatchSimilarityGenerativityConsolidation:
    """Candidate-only generativity summary for patch-similarity retargeting."""

    generativity_consolidation_id: str
    source_selection_rule_consolidation_id: str
    source_selection_rule_candidate_id: str
    source_mechanism_candidate_id: str
    game_id: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    candidate_rule_family: str
    candidate_generativity: str
    initial_success_args: Tuple[Dict[str, Any], ...]
    prior_expansion_successes: Tuple[Dict[str, Any], ...]
    new_executed_expansion_successes: Tuple[Dict[str, Any], ...]
    successful_args_total: Tuple[Dict[str, Any], ...]
    failed_args: Tuple[Dict[str, Any], ...]
    new_expansion_successes: Tuple[Dict[str, Any], ...]
    pattern_hypothesis: str
    success_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    changed_pixels_role: str
    source_success_metric_support_events: int
    source_success_metric_contradiction_events: int
    source_diagnostic_contradiction_events: int
    source_neutral_events: int
    ready_for_a32_revision_queue: bool
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
    generative_sequence_counted_as_confirmation: bool = False
    a32_queue_ready_is_not_verdict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generativity_consolidation_id": self.generativity_consolidation_id,
            "source_selection_rule_consolidation_id": (
                self.source_selection_rule_consolidation_id
            ),
            "source_selection_rule_candidate_id": (
                self.source_selection_rule_candidate_id
            ),
            "source_mechanism_candidate_id": self.source_mechanism_candidate_id,
            "game_id": self.game_id,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "candidate_rule_family": self.candidate_rule_family,
            "candidate_generativity": self.candidate_generativity,
            "initial_success_args": [
                dict(item) for item in self.initial_success_args
            ],
            "prior_expansion_successes": [
                dict(item) for item in self.prior_expansion_successes
            ],
            "new_executed_expansion_successes": [
                dict(item) for item in self.new_executed_expansion_successes
            ],
            "successful_args_total": [
                dict(item) for item in self.successful_args_total
            ],
            "failed_args": [dict(item) for item in self.failed_args],
            "new_expansion_successes": [
                dict(item) for item in self.new_expansion_successes
            ],
            "pattern_hypothesis": self.pattern_hypothesis,
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "changed_pixels_role": self.changed_pixels_role,
            "source_success_metric_support_events": int(
                self.source_success_metric_support_events
            ),
            "source_success_metric_contradiction_events": int(
                self.source_success_metric_contradiction_events
            ),
            "source_diagnostic_contradiction_events": int(
                self.source_diagnostic_contradiction_events
            ),
            "source_neutral_events": int(self.source_neutral_events),
            "ready_for_a32_revision_queue": self.ready_for_a32_revision_queue,
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
            "generative_sequence_counted_as_confirmation": (
                self.generative_sequence_counted_as_confirmation
            ),
            "a32_queue_ready_is_not_verdict": self.a32_queue_ready_is_not_verdict,
        }


def run_dynamic_retarget_patch_similarity_generativity_consolidation(
    *,
    expansion_results_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(expansion_results_path)
    consolidations = consolidate_patch_similarity_generativity(payload)
    return {
        "config": {
            "expansion_results_path": str(expansion_results_path),
            "schema_version": (
                "m3.dynamic_retarget_patch_similarity_generativity_consolidation.v1"
            ),
            "inputs_read": ["M3.19"],
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
                "M3.17",
                "M3.18",
                "M3.19",
                "A32",
                "A33",
            ],
            "consolidation_policy": {
                "execution_performed": False,
                "support_forced_to_zero": True,
                "success_metric_contradictions_block_a32_readiness": True,
                "diagnostic_metric_contradictions_do_not_refute_success": True,
                "generative_sequence_is_not_confirmation": True,
            },
        },
        "summary": summarize_generativity_consolidations(
            source_payload=payload,
            consolidations=consolidations,
        ),
        "generativity_consolidations": [
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
        "generative_sequence_counted_as_confirmation": False,
    }


def consolidate_patch_similarity_generativity(
    payload: Mapping[str, Any],
) -> Tuple[PatchSimilarityGenerativityConsolidation, ...]:
    per_signature = [
        dict(row) for row in payload.get("per_signature_execution", []) or []
    ]
    supported_signatures = [
        row
        for row in per_signature
        if bool(row.get("signature_has_success_metric_support"))
        and str(row.get("status", "")) != "BLOCKED_NOT_EXECUTED"
    ]
    if not supported_signatures:
        return ()

    experiments = [
        dict(row) for row in payload.get("controlled_experiments", []) or []
    ]
    summary = dict(payload.get("summary", {}) or {})
    first = supported_signatures[0]

    seed_successes = seed_args_from_experiments(
        experiments,
        field_name="seed_successful_args",
    )
    seed_failures = seed_args_from_experiments(
        experiments,
        field_name="seed_failed_args",
    )
    initial_successes, prior_expansion_successes = split_initial_and_prior_expansion(
        seed_successes
    )
    new_executed_successes = dedupe_args(
        [
            dict(row.get("target_action_args", {}) or {})
            for row in supported_signatures
            if row.get("target_action_args")
        ]
    )
    new_expansion_successes = dedupe_args(
        [*prior_expansion_successes, *new_executed_successes]
    )
    successful_total = dedupe_args([*seed_successes, *new_executed_successes])
    success_metrics = tuple(success_metrics_from_payload(payload))
    diagnostic_metrics = tuple(diagnostic_metrics_from_payload(payload))
    ready_for_a32 = (
        len(new_expansion_successes) >= 2
        and bool(new_executed_successes)
        and int(summary.get("success_metric_contradiction_events", 0) or 0) == 0
    )

    consolidation = PatchSimilarityGenerativityConsolidation(
        generativity_consolidation_id=generativity_consolidation_id(first),
        source_selection_rule_consolidation_id=str(
            first.get("source_selection_rule_consolidation_id", "")
        ),
        source_selection_rule_candidate_id=str(
            first.get("source_selection_rule_candidate_id", "")
        ),
        source_mechanism_candidate_id=str(
            first.get("source_mechanism_candidate_id", "")
        ),
        game_id=str(first.get("game_id", "")),
        context_replay=tuple(str(item) for item in first.get("context_replay", []) or []),
        context_replay_args=_context_args_tuple(first.get("context_replay_args")),
        target_action=str(first.get("target_action", "")),
        candidate_rule_family=str(first.get("rule_family", "")),
        candidate_generativity=PATCH_GENERATIVITY_STATUS,
        initial_success_args=tuple(initial_successes),
        prior_expansion_successes=tuple(prior_expansion_successes),
        new_executed_expansion_successes=tuple(new_executed_successes),
        successful_args_total=tuple(successful_total),
        failed_args=tuple(seed_failures),
        new_expansion_successes=tuple(new_expansion_successes),
        pattern_hypothesis=pattern_hypothesis_for(successful_total),
        success_metrics=success_metrics,
        diagnostic_metrics=diagnostic_metrics,
        changed_pixels_role=(
            "effect_radar_not_success_metric"
            if "changed_pixels" in diagnostic_metrics
            else "not_tested"
        ),
        source_success_metric_support_events=int(
            summary.get("success_metric_support_events", 0) or 0
        ),
        source_success_metric_contradiction_events=int(
            summary.get("success_metric_contradiction_events", 0) or 0
        ),
        source_diagnostic_contradiction_events=int(
            summary.get("diagnostic_contradiction_events", 0) or 0
        ),
        source_neutral_events=int(summary.get("neutral_events", 0) or 0),
        ready_for_a32_revision_queue=ready_for_a32,
    )
    return (consolidation,)


def summarize_generativity_consolidations(
    *,
    source_payload: Mapping[str, Any],
    consolidations: Sequence[PatchSimilarityGenerativityConsolidation],
) -> Dict[str, Any]:
    source_summary = dict(source_payload.get("summary", {}) or {})
    return {
        "expansion_results_consumed": 1 if source_payload else 0,
        "patch_similarity_generativity_consolidations": len(consolidations),
        "candidate_rule_families": sorted(
            {
                item.candidate_rule_family
                for item in consolidations
                if item.candidate_rule_family
            }
        ),
        "candidate_generativity": sorted(
            {
                item.candidate_generativity
                for item in consolidations
                if item.candidate_generativity
            }
        ),
        "successful_args_total_count": sum(
            len(item.successful_args_total) for item in consolidations
        ),
        "failed_args_count": sum(len(item.failed_args) for item in consolidations),
        "new_expansion_successes_count": sum(
            len(item.new_expansion_successes) for item in consolidations
        ),
        "ready_for_a32_revision_queue": any(
            item.ready_for_a32_revision_queue for item in consolidations
        ),
        "ready_for_a32_revision_queue_is_not_verdict": True,
        "source_success_metric_support_events": int(
            source_summary.get("success_metric_support_events", 0) or 0
        ),
        "source_success_metric_contradiction_events": int(
            source_summary.get("success_metric_contradiction_events", 0) or 0
        ),
        "source_diagnostic_contradiction_events": int(
            source_summary.get("diagnostic_contradiction_events", 0) or 0
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
        "generative_sequence_counted_as_confirmation": False,
    }


def seed_args_from_experiments(
    experiments: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> Tuple[Dict[str, Any], ...]:
    values: list[Dict[str, Any]] = []
    for row in experiments:
        raw = row.get(field_name)
        if isinstance(raw, Mapping):
            values.append(dict(raw))
            continue
        for args in raw or []:
            if isinstance(args, Mapping):
                values.append(dict(args))
    return dedupe_args(values)


def split_initial_and_prior_expansion(
    seed_successes: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    deduped = tuple(dict(item) for item in seed_successes)
    if len(deduped) >= 5:
        return deduped[:-1], deduped[-1:]
    return deduped, ()


def success_metrics_from_payload(payload: Mapping[str, Any]) -> list[str]:
    per_signature = payload.get("per_signature_execution", []) or []
    metrics = {
        str(metric)
        for row in per_signature
        for metric in row.get("grounded_success_metrics", []) or []
    }
    if metrics:
        return _metric_sort(metrics)
    experiments = payload.get("controlled_experiments", []) or []
    return _metric_sort(
        {
            str(row.get("metric", ""))
            for row in experiments
            if str(row.get("metric_role", "")) == "success_metric"
            and int(row.get("support_events", 0) or 0) > 0
        }
    )


def diagnostic_metrics_from_payload(payload: Mapping[str, Any]) -> list[str]:
    experiments = payload.get("controlled_experiments", []) or []
    metrics = {
        str(metric)
        for row in experiments
        for metric in row.get("diagnostic_metrics", []) or []
    }
    if metrics:
        return _metric_sort(metrics)
    return _metric_sort(
        {
            str(row.get("metric", ""))
            for row in experiments
            if str(row.get("metric_role", "")) == "diagnostic_metric"
        }
    )


def pattern_hypothesis_for(successful_args: Sequence[Mapping[str, Any]]) -> str:
    y_counts: dict[Any, int] = {}
    for args in successful_args:
        y = args.get("y")
        y_counts[y] = y_counts.get(y, 0) + 1
    if any(count >= 3 for count in y_counts.values()):
        return "success_like_patch_line_or_region_after_repositioning"
    return "success_like_patch_region_after_repositioning"


def generativity_consolidation_id(row: Mapping[str, Any]) -> str:
    game_token = str(row.get("game_id", "")).split("-", 1)[0] or "unknown_game"
    return "::".join(
        [
            "m3_20",
            game_token,
            "ACTION4_ACTION6",
            "patch_similarity_generativity",
        ]
    )


def dedupe_args(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_args_key(dict(value)): dict(value) for value in values if value}
    return tuple(by_key[key] for key in sorted(by_key))


def write_dynamic_retarget_patch_similarity_generativity_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH
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
        description="Consolidate M3.20 patch-similarity generativity.",
    )
    parser.add_argument(
        "--expansion-results",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_patch_similarity_generativity_consolidation(
        expansion_results_path=args.expansion_results,
    )
    write_dynamic_retarget_patch_similarity_generativity_consolidation(
        payload,
        args.out,
    )
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
