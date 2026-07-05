"""M3.13 consolidation of dynamic retarget observations into candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .dynamic_retarget_followup_executor import (
    DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH,
)
from .m2_observation_refinement import (
    M3_REFINEMENT_TRUTH_STATUS,
    METRIC_ORDER,
    context_token,
)


DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "dynamic_retarget_mechanism_candidates.json"
)


@dataclass(frozen=True)
class DynamicRetargetMechanismCandidate:
    """Candidate-only mechanism compressed from M3.12 retarget tests."""

    mechanism_candidate_id: str
    source_refined_hypothesis_id: str
    source_hypothesis_ids: Tuple[str, ...]
    game_id: str
    candidate_mechanic: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    initial_consumed_args: Dict[str, Any] | None
    repositioning_action: str | None
    target_action: str
    target_action_arg_policy: str
    successful_retargets: Tuple[Dict[str, Any], ...]
    failed_retargets: Tuple[Dict[str, Any], ...]
    best_arg: Dict[str, Any] | None
    positive_metrics: Tuple[str, ...]
    non_decisive_or_negative_metrics: Tuple[str, ...]
    controls_tested: Tuple[str, ...]
    tested_candidate_args: int
    args_with_grounded_support: int
    mechanism_support_events: int
    arg_level_support_events: int
    contradiction_events: int
    neutral_events: int
    derived_from_arg_results: Tuple[Dict[str, Any], ...]
    derived_from_mechanism_summary: Dict[str, Any]
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
    mechanism_support_events_counted_as_support: bool = False
    arg_level_support_events_counted_as_mechanism_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_candidate_id": self.mechanism_candidate_id,
            "source_refined_hypothesis_id": self.source_refined_hypothesis_id,
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "game_id": self.game_id,
            "candidate_mechanic": self.candidate_mechanic,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "initial_consumed_args": (
                dict(self.initial_consumed_args)
                if self.initial_consumed_args is not None
                else None
            ),
            "repositioning_action": self.repositioning_action,
            "target_action": self.target_action,
            "target_action_arg_policy": self.target_action_arg_policy,
            "successful_retargets": [
                dict(item) for item in self.successful_retargets
            ],
            "failed_retargets": [dict(item) for item in self.failed_retargets],
            "best_arg": dict(self.best_arg) if self.best_arg is not None else None,
            "positive_metrics": list(self.positive_metrics),
            "non_decisive_or_negative_metrics": list(
                self.non_decisive_or_negative_metrics
            ),
            "controls_tested": list(self.controls_tested),
            "tested_candidate_args": int(self.tested_candidate_args),
            "args_with_grounded_support": int(self.args_with_grounded_support),
            "mechanism_support_events": int(self.mechanism_support_events),
            "arg_level_support_events": int(self.arg_level_support_events),
            "arg_level_support_events_counted_as_mechanism_support": (
                self.arg_level_support_events_counted_as_mechanism_support
            ),
            "mechanism_support_events_counted_as_support": (
                self.mechanism_support_events_counted_as_support
            ),
            "contradiction_events": int(self.contradiction_events),
            "neutral_events": int(self.neutral_events),
            "selection_problem_open": bool(
                self.successful_retargets and self.failed_retargets
            ),
            "selection_problem": (
                "distinguish_successful_from_failed_retarget_args"
                if self.successful_retargets and self.failed_retargets
                else None
            ),
            "metric_interpretation": {
                "success_metrics": list(self.positive_metrics),
                "non_decisive_or_negative_metrics": list(
                    self.non_decisive_or_negative_metrics
                ),
                "changed_pixels_role": changed_pixels_role(
                    self.positive_metrics,
                    self.non_decisive_or_negative_metrics,
                ),
            },
            "derived_from_arg_results": [
                dict(item) for item in self.derived_from_arg_results
            ],
            "derived_from_mechanism_summary": dict(
                self.derived_from_mechanism_summary
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
        }


def run_dynamic_retarget_mechanism_consolidation(
    *,
    retarget_results_path: str | Path = DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(retarget_results_path)
    per_arg = [dict(row) for row in payload.get("per_arg_results", []) or []]
    experiments = [
        dict(row) for row in payload.get("controlled_experiments", []) or []
    ]
    candidates = build_dynamic_retarget_mechanism_candidates(
        per_arg=per_arg,
        experiments=experiments,
        mechanism_summary=dict(payload.get("mechanism_summary", {}) or {}),
    )
    return {
        "config": {
            "retarget_results_path": str(retarget_results_path),
            "schema_version": "m3.dynamic_retarget_mechanism_candidates.v1",
            "inputs_read": ["M3.12"],
            "artifacts_not_modified": [
                "M2",
                "M3.8",
                "M3.9",
                "M3.10",
                "M3.11",
                "M3.12",
                "A32",
                "A33",
            ],
            "consolidation_policy": {
                "arg_level_events_are_observations": True,
                "arg_level_support_events_counted_as_mechanism_support": False,
                "mechanism_support_events_counted_as_support": False,
                "changed_pixels_can_be_non_decisive_for_retarget_success": True,
            },
        },
        "summary": summarize_mechanism_candidates(
            per_arg=per_arg,
            experiments=experiments,
            candidates=candidates,
        ),
        "mechanism_candidates": [candidate.to_dict() for candidate in candidates],
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
        "mechanism_support_events_counted_as_support": False,
    }


def build_dynamic_retarget_mechanism_candidates(
    *,
    per_arg: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    mechanism_summary: Mapping[str, Any],
) -> Tuple[DynamicRetargetMechanismCandidate, ...]:
    by_source: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in per_arg:
        source_id = str(row.get("source_refined_hypothesis_id", ""))
        if source_id:
            by_source[source_id].append(dict(row))

    candidates: list[DynamicRetargetMechanismCandidate] = []
    for source_id, rows in sorted(by_source.items()):
        successful = [
            dict(row.get("target_action_args", {}) or {})
            for row in rows
            if bool(row.get("arg_has_grounded_support"))
        ]
        if not successful:
            continue
        failed = [
            dict(row.get("target_action_args", {}) or {})
            for row in rows
            if not bool(row.get("arg_has_grounded_support"))
        ]
        source_experiments = [
            dict(row)
            for row in experiments
            if str(row.get("source_refined_hypothesis_id", "")) == source_id
        ]
        first = source_experiments[0] if source_experiments else rows[0]
        context_replay = tuple(
            str(action) for action in first.get("context_replay", []) or []
        )
        context_replay_args = _context_args_tuple(first.get("context_replay_args"))
        target_action = str(first.get("target_action", ""))
        game_id = str(first.get("game_id", ""))
        positive_metrics = tuple(
            _metric_sort(
                {
                    str(metric)
                    for row in rows
                    for metric in row.get("grounded_support_metrics", []) or []
                    if str(metric)
                }
            )
        )
        metrics_tested = {
            str(metric)
            for row in rows
            for metric in row.get("metrics_tested", []) or []
            if str(metric)
        }
        non_decisive = tuple(_metric_sort(metrics_tested - set(positive_metrics)))
        source_hypothesis_ids = tuple(
            sorted(
                {
                    str(source)
                    for row in source_experiments
                    for source in row.get("source_hypothesis_ids", []) or []
                    if str(source)
                }
            )
        )
        controls_tested = tuple(
            sorted(
                {
                    str(control)
                    for row in rows
                    for control in row.get("controls_tested", []) or []
                    if str(control)
                }
            )
        )
        source_mechanism = mechanism_summary_for_source(
            source_id=source_id,
            rows=rows,
            mechanism_summary=mechanism_summary,
        )
        candidates.append(
            DynamicRetargetMechanismCandidate(
                mechanism_candidate_id=mechanism_candidate_id(
                    game_id=game_id,
                    context_replay=context_replay,
                    target_action=target_action,
                ),
                source_refined_hypothesis_id=source_id,
                source_hypothesis_ids=source_hypothesis_ids,
                game_id=game_id,
                candidate_mechanic=candidate_mechanic_name(target_action),
                context_replay=context_replay,
                context_replay_args=context_replay_args,
                initial_consumed_args=initial_consumed_args(
                    context_replay=context_replay,
                    context_replay_args=context_replay_args,
                    target_action=target_action,
                ),
                repositioning_action=repositioning_action(context_replay),
                target_action=target_action,
                target_action_arg_policy=str(
                    first.get("target_action_arg_policy", "")
                ),
                successful_retargets=tuple(successful),
                failed_retargets=tuple(failed),
                best_arg=(
                    dict(source_mechanism.get("best_arg", {}) or {})
                    if source_mechanism.get("best_arg") is not None
                    else None
                ),
                positive_metrics=positive_metrics,
                non_decisive_or_negative_metrics=non_decisive,
                controls_tested=controls_tested,
                tested_candidate_args=len(rows),
                args_with_grounded_support=len(successful),
                mechanism_support_events=(
                    1 if successful else 0
                ),
                arg_level_support_events=sum(
                    int(row.get("support_events", 0) or 0) for row in rows
                ),
                contradiction_events=sum(
                    int(row.get("contradiction_events", 0) or 0) for row in rows
                ),
                neutral_events=sum(
                    int(row.get("neutral_events", 0) or 0) for row in rows
                ),
                derived_from_arg_results=tuple(
                    compact_arg_result(row) for row in rows
                ),
                derived_from_mechanism_summary=source_mechanism,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.mechanism_candidate_id))


def mechanism_summary_for_source(
    *,
    source_id: str,
    rows: Sequence[Mapping[str, Any]],
    mechanism_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    if len({str(row.get("source_refined_hypothesis_id", "")) for row in rows}) == 1:
        return {
            "source_refined_hypothesis_id": source_id,
            "tested_candidate_args": len(rows),
            "args_with_grounded_support": len(
                [row for row in rows if bool(row.get("arg_has_grounded_support"))]
            ),
            "best_arg": dict(mechanism_summary.get("best_arg", {}) or {}),
            "mechanism_support_events": (
                1
                if any(bool(row.get("arg_has_grounded_support")) for row in rows)
                else 0
            ),
            "arg_level_support_events": sum(
                int(row.get("support_events", 0) or 0) for row in rows
            ),
            "arg_level_support_events_counted_as_mechanism_support": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            "wrong_confirmations": 0,
        }
    return dict(mechanism_summary)


def compact_arg_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "target_action_args": dict(row.get("target_action_args", {}) or {}),
        "arg_has_grounded_support": bool(row.get("arg_has_grounded_support")),
        "support_events": int(row.get("support_events", 0) or 0),
        "contradiction_events": int(row.get("contradiction_events", 0) or 0),
        "neutral_events": int(row.get("neutral_events", 0) or 0),
        "grounded_support_metrics": [
            str(metric) for metric in row.get("grounded_support_metrics", []) or []
        ],
        "metrics_tested": [str(metric) for metric in row.get("metrics_tested", []) or []],
        "candidate_arg_rank": int(row.get("candidate_arg_rank", 0) or 0),
        "candidate_arg_score": float(row.get("candidate_arg_score", 0.0) or 0.0),
    }


def summarize_mechanism_candidates(
    *,
    per_arg: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    candidates: Sequence[DynamicRetargetMechanismCandidate],
) -> Dict[str, Any]:
    positive_metrics = {
        metric for candidate in candidates for metric in candidate.positive_metrics
    }
    non_decisive = {
        metric
        for candidate in candidates
        for metric in candidate.non_decisive_or_negative_metrics
    }
    return {
        "source_retarget_args": len(per_arg),
        "source_controlled_experiments": len(experiments),
        "candidate_mechanisms": len(candidates),
        "successful_retargets": sum(
            len(candidate.successful_retargets) for candidate in candidates
        ),
        "failed_retargets": sum(
            len(candidate.failed_retargets) for candidate in candidates
        ),
        "positive_metrics": _metric_sort(positive_metrics),
        "non_decisive_or_negative_metrics": _metric_sort(non_decisive),
        "mechanism_support_events": sum(
            int(candidate.mechanism_support_events) for candidate in candidates
        ),
        "mechanism_support_events_counted_as_support": False,
        "arg_level_support_events": sum(
            int(candidate.arg_level_support_events) for candidate in candidates
        ),
        "arg_level_support_events_counted_as_mechanism_support": False,
        "contradiction_events": sum(
            int(candidate.contradiction_events) for candidate in candidates
        ),
        "neutral_events": sum(int(candidate.neutral_events) for candidate in candidates),
        "selection_problem_candidates": len(
            [
                candidate
                for candidate in candidates
                if candidate.successful_retargets and candidate.failed_retargets
            ]
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


def candidate_mechanic_name(target_action: str) -> str:
    token = str(target_action).lower() or "target_action"
    return f"repositioning_opens_new_{token}_target_region"


def mechanism_candidate_id(
    *,
    game_id: str,
    context_replay: Sequence[str],
    target_action: str,
) -> str:
    game_token = str(game_id).split("-", 1)[0] or "unknown_game"
    return "::".join(
        [
            "m3_13",
            game_token,
            context_token(context_replay),
            str(target_action),
            "retarget_region",
        ]
    )


def initial_consumed_args(
    *,
    context_replay: Sequence[str],
    context_replay_args: Sequence[Mapping[str, Any]] | None,
    target_action: str,
) -> Dict[str, Any] | None:
    if context_replay_args is None:
        return None
    for action, args in zip(context_replay, context_replay_args):
        if str(action) == str(target_action) and args:
            return dict(args)
    return None


def repositioning_action(context_replay: Sequence[str]) -> str | None:
    if not context_replay:
        return None
    return str(context_replay[-1])


def changed_pixels_role(
    positive_metrics: Sequence[str],
    non_decisive_or_negative_metrics: Sequence[str],
) -> str:
    if "changed_pixels" in positive_metrics:
        return "success_metric"
    if "changed_pixels" in non_decisive_or_negative_metrics:
        return "effect_radar_not_retarget_success_metric"
    return "not_tested"


def write_dynamic_retarget_mechanism_candidates(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH,
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


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate M3.12 retarget results into mechanism candidates.",
    )
    parser.add_argument(
        "--retarget-results",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_mechanism_consolidation(
        retarget_results_path=args.retarget_results,
    )
    write_dynamic_retarget_mechanism_candidates(payload, args.out)
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
