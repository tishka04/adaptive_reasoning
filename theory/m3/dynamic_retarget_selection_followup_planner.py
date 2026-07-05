"""M3.15 planner for discriminating dynamic retarget selection rules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .dynamic_retarget_selection_rule_induction import (
    DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_planner import DEFAULT_REACTIVATION_METRICS


DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "dynamic_retarget_selection_followup_requests.json"
)
READY_FOR_M3_SELECTION_FOLLOWUP = "READY_FOR_M3_SELECTION_FOLLOWUP"
DEFAULT_MAX_SELECTION_FOLLOWUP_REQUESTS = 8
EXPLICIT_RETARGET_ARG_POLICY = "explicit_selection_rule_probe_args"
LOCAL_PATCH_SIMILARITY_POLICY = "local_patch_similarity_after_repositioning"


@dataclass(frozen=True)
class SelectionRuleFollowupRequest:
    """A candidate-only request for discriminating retarget selection rules."""

    request_id: str
    source_selection_rule_candidate_id: str
    source_mechanism_candidate_id: str
    source_refined_hypothesis_id: str
    source_rule_id: str | None
    game_id: str
    rule_family: str
    probe_family: str
    hypothesis_tested: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    target_action_args: Dict[str, Any] | None
    target_action_arg_policy: str
    known_successful_retargets: Tuple[Dict[str, Any], ...]
    known_failed_retargets: Tuple[Dict[str, Any], ...]
    excluded_args: Tuple[Dict[str, Any], ...]
    seed_successful_args: Tuple[Dict[str, Any], ...]
    seed_failed_args: Tuple[Dict[str, Any], ...]
    suggested_control_actions: Tuple[str, ...]
    control_policy: str
    metrics: Tuple[str, ...]
    success_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    expected_signal: str
    falsification_criterion: str
    discriminates_rule_families: Tuple[str, ...]
    planning_rationale: str
    status: str = READY_FOR_M3_SELECTION_FOLLOWUP
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
    followup_request_counted_as_support: bool = False
    execution_performed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_selection_rule_candidate_id": (
                self.source_selection_rule_candidate_id
            ),
            "source_mechanism_candidate_id": self.source_mechanism_candidate_id,
            "source_refined_hypothesis_id": self.source_refined_hypothesis_id,
            "source_rule_id": self.source_rule_id,
            "game_id": self.game_id,
            "rule_family": self.rule_family,
            "probe_family": self.probe_family,
            "hypothesis_tested": self.hypothesis_tested,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "target_action_args": (
                dict(self.target_action_args)
                if self.target_action_args is not None
                else None
            ),
            "target_action_arg_policy": self.target_action_arg_policy,
            "known_successful_retargets": [
                dict(item) for item in self.known_successful_retargets
            ],
            "known_failed_retargets": [
                dict(item) for item in self.known_failed_retargets
            ],
            "excluded_args": [dict(item) for item in self.excluded_args],
            "seed_successful_args": [
                dict(item) for item in self.seed_successful_args
            ],
            "seed_failed_args": [dict(item) for item in self.seed_failed_args],
            "suggested_control_actions": list(self.suggested_control_actions),
            "control_policy": self.control_policy,
            "metrics": list(self.metrics),
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "expected_signal": self.expected_signal,
            "falsification_criterion": self.falsification_criterion,
            "discriminates_rule_families": list(self.discriminates_rule_families),
            "planning_rationale": self.planning_rationale,
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
            "followup_request_counted_as_support": (
                self.followup_request_counted_as_support
            ),
            "execution_performed": self.execution_performed,
        }


def run_dynamic_retarget_selection_followup_planning(
    *,
    selection_rules_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH
    ),
    max_followup_requests: int = DEFAULT_MAX_SELECTION_FOLLOWUP_REQUESTS,
) -> Dict[str, Any]:
    payload = _load_json(selection_rules_path)
    rule_sets = [
        dict(row) for row in payload.get("selection_rule_sets", []) or []
    ]
    all_requests = build_selection_followup_requests(rule_sets)
    limited_requests = tuple(all_requests[: max(0, int(max_followup_requests))])
    return {
        "config": {
            "selection_rules_path": str(selection_rules_path),
            "schema_version": "m3.dynamic_retarget_selection_followups.v1",
            "inputs_read": ["M3.14"],
            "artifacts_not_modified": [
                "M2",
                "M3.8",
                "M3.9",
                "M3.10",
                "M3.11",
                "M3.12",
                "M3.13",
                "M3.14",
                "A32",
                "A33",
            ],
            "execution_performed": False,
            "max_followup_requests": int(max_followup_requests),
            "request_truncation_policy": "deterministic_priority_order",
        },
        "summary": summarize_selection_followup_requests(
            rule_sets=rule_sets,
            all_requests=all_requests,
            emitted_requests=limited_requests,
            max_followup_requests=max_followup_requests,
        ),
        "followup_experiment_requests": [
            request.to_dict() for request in limited_requests
        ],
        "truncated_followup_experiment_requests": [
            request.to_dict()
            for request in all_requests[len(limited_requests) :]
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
        "followup_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def build_selection_followup_requests(
    rule_sets: Sequence[Mapping[str, Any]],
) -> Tuple[SelectionRuleFollowupRequest, ...]:
    requests: list[SelectionRuleFollowupRequest] = []
    for rule_set in rule_sets:
        requests.extend(same_x_different_band_requests(rule_set))
        requests.extend(same_y_neighbor_x_requests(rule_set))
        requests.extend(local_patch_similarity_requests(rule_set))
    return tuple(requests)


def same_x_different_band_requests(
    rule_set: Mapping[str, Any],
) -> Tuple[SelectionRuleFollowupRequest, ...]:
    requests: list[SelectionRuleFollowupRequest] = []
    contrasts = dict(rule_set.get("observed_contrasts", {}) or {})
    for contrast in contrasts.get("same_x_mixed_outcomes", []) or []:
        x_value = _safe_int(dict(contrast).get("x"))
        if x_value is None:
            continue
        known_y = {
            int(y)
            for y in dict(contrast).get("successful_y_values", []) or []
            if _safe_int(y) is not None
        }
        known_y.update(
            int(y)
            for y in dict(contrast).get("failed_y_values", []) or []
            if _safe_int(y) is not None
        )
        for y_value in candidate_band_values(known_y):
            args = {"x": x_value, "y": y_value}
            if args_key(args) in tested_arg_keys(rule_set):
                continue
            requests.append(
                build_selection_followup_request(
                    rule_set=rule_set,
                    rule_family="row_or_band_dependent_retarget",
                    probe_family="same_x_different_band_probe",
                    target_action_args=args,
                    target_action_arg_policy=EXPLICIT_RETARGET_ARG_POLICY,
                    source_rule_id=rule_id_for_family(
                        rule_set, "row_or_band_dependent_retarget"
                    ),
                    expected_signal=(
                        "same_x_new_band_probe_separates_band_rule_from_patch_rule"
                    ),
                    falsification_criterion=(
                        "New y-band probes at the mixed x do not follow any "
                        "stable band relation on grounded success metrics."
                    ),
                    discriminates_rule_families=(
                        "row_or_band_dependent_retarget",
                        "local_patch_transformability",
                    ),
                    planning_rationale=(
                        "M3.14 observed the same x with both success and failure; "
                        "new y bands test whether the band relation explains the "
                        "contrast."
                    ),
                )
            )
    return tuple(requests)


def same_y_neighbor_x_requests(
    rule_set: Mapping[str, Any],
) -> Tuple[SelectionRuleFollowupRequest, ...]:
    requests: list[SelectionRuleFollowupRequest] = []
    contrasts = dict(rule_set.get("observed_contrasts", {}) or {})
    for contrast in contrasts.get("same_y_mixed_outcomes", []) or []:
        y_value = _safe_int(dict(contrast).get("y"))
        if y_value is None:
            continue
        known_x = {
            int(x)
            for x in dict(contrast).get("successful_x_values", []) or []
            if _safe_int(x) is not None
        }
        known_x.update(
            int(x)
            for x in dict(contrast).get("failed_x_values", []) or []
            if _safe_int(x) is not None
        )
        for x_value in candidate_neighbor_x_values(known_x):
            args = {"x": x_value, "y": y_value}
            if args_key(args) in tested_arg_keys(rule_set):
                continue
            requests.append(
                build_selection_followup_request(
                    rule_set=rule_set,
                    rule_family="row_or_band_dependent_retarget",
                    probe_family="same_y_neighbor_x_probe",
                    target_action_args=args,
                    target_action_arg_policy=EXPLICIT_RETARGET_ARG_POLICY,
                    source_rule_id=rule_id_for_family(
                        rule_set, "row_or_band_dependent_retarget"
                    ),
                    expected_signal=(
                        "same_y_neighbor_probe_separates_spatial_boundary_from_patch_rule"
                    ),
                    falsification_criterion=(
                        "Neighbor x probes on the mixed y band do not reveal a "
                        "stable boundary or local-patch contrast."
                    ),
                    discriminates_rule_families=(
                        "row_or_band_dependent_retarget",
                        "local_patch_transformability",
                    ),
                    planning_rationale=(
                        "M3.14 observed y=0 with both success and failure; "
                        "neighbor x probes test whether x=30 is a spatial "
                        "boundary or a local-patch outlier."
                    ),
                )
            )
    return tuple(requests)


def local_patch_similarity_requests(
    rule_set: Mapping[str, Any],
) -> Tuple[SelectionRuleFollowupRequest, ...]:
    if not rule_id_for_family(rule_set, "local_patch_transformability"):
        return ()
    return (
        build_selection_followup_request(
            rule_set=rule_set,
            rule_family="local_patch_transformability",
            probe_family="local_patch_success_similarity_probe",
            target_action_args=None,
            target_action_arg_policy=LOCAL_PATCH_SIMILARITY_POLICY,
            source_rule_id=rule_id_for_family(
                rule_set, "local_patch_transformability"
            ),
            expected_signal=(
                "success_patch_similar_targets_match_grounded_success_metrics"
            ),
            falsification_criterion=(
                "Targets selected as similar to successful local patches fail "
                "on local_patch/object_positions metrics."
            ),
            discriminates_rule_families=(
                "local_patch_transformability",
                "row_or_band_dependent_retarget",
            ),
            planning_rationale=(
                "M3.15 should ask M3.16 to resolve targets with local patches "
                "similar to successful retargets under the same causal replay."
            ),
            seed_successful_args=tuple(
                dict(item) for item in rule_set.get("successful_retargets", []) or []
            ),
            seed_failed_args=(),
        ),
        build_selection_followup_request(
            rule_set=rule_set,
            rule_family="local_patch_transformability",
            probe_family="local_patch_failure_similarity_probe",
            target_action_args=None,
            target_action_arg_policy=LOCAL_PATCH_SIMILARITY_POLICY,
            source_rule_id=rule_id_for_family(
                rule_set, "local_patch_transformability"
            ),
            expected_signal=(
                "failure_patch_similar_targets_fail_grounded_success_metrics"
            ),
            falsification_criterion=(
                "Targets selected as similar to the failed local patch succeed "
                "on local_patch/object_positions metrics."
            ),
            discriminates_rule_families=(
                "local_patch_transformability",
                "row_or_band_dependent_retarget",
            ),
            planning_rationale=(
                "M3.15 should ask M3.16 to resolve targets with local patches "
                "similar to the failed retarget under the same causal replay."
            ),
            seed_successful_args=(),
            seed_failed_args=tuple(
                dict(item) for item in rule_set.get("failed_retargets", []) or []
            ),
        ),
    )


def build_selection_followup_request(
    *,
    rule_set: Mapping[str, Any],
    rule_family: str,
    probe_family: str,
    target_action_args: Mapping[str, Any] | None,
    target_action_arg_policy: str,
    source_rule_id: str | None,
    expected_signal: str,
    falsification_criterion: str,
    discriminates_rule_families: Sequence[str],
    planning_rationale: str,
    seed_successful_args: Sequence[Mapping[str, Any]] = (),
    seed_failed_args: Sequence[Mapping[str, Any]] = (),
) -> SelectionRuleFollowupRequest:
    request_id = selection_followup_request_id(
        selection_rule_candidate_id=str(
            rule_set.get("selection_rule_candidate_id", "")
        ),
        probe_family=probe_family,
        target_action_args=target_action_args,
        index=next_probe_index(rule_set, probe_family, target_action_args),
    )
    success_metrics = tuple(
        str(metric) for metric in rule_set.get("positive_metrics", []) or []
    )
    diagnostic_metrics = tuple(
        str(metric)
        for metric in rule_set.get("non_decisive_or_negative_metrics", []) or []
    )
    metrics = ordered_unique((*success_metrics, *diagnostic_metrics)) or tuple(
        DEFAULT_REACTIVATION_METRICS
    )
    return SelectionRuleFollowupRequest(
        request_id=request_id,
        source_selection_rule_candidate_id=str(
            rule_set.get("selection_rule_candidate_id", "")
        ),
        source_mechanism_candidate_id=str(
            rule_set.get("source_mechanism_candidate_id", "")
        ),
        source_refined_hypothesis_id=str(
            rule_set.get("source_refined_hypothesis_id", "")
        ),
        source_rule_id=source_rule_id,
        game_id=str(rule_set.get("game_id", "")),
        rule_family=rule_family,
        probe_family=probe_family,
        hypothesis_tested=(
            f"{rule_family} explains which {rule_set.get('target_action', '')} "
            "retargets are valid after repositioning"
        ),
        context_replay=tuple(
            str(action) for action in rule_set.get("context_replay", []) or []
        ),
        context_replay_args=_context_args_tuple(rule_set.get("context_replay_args")),
        target_action=str(rule_set.get("target_action", "")),
        target_action_args=(
            dict(target_action_args) if target_action_args is not None else None
        ),
        target_action_arg_policy=target_action_arg_policy,
        known_successful_retargets=tuple(
            dict(item) for item in rule_set.get("successful_retargets", []) or []
        ),
        known_failed_retargets=tuple(
            dict(item) for item in rule_set.get("failed_retargets", []) or []
        ),
        excluded_args=tuple(
            dict(item)
            for item in (
                list(rule_set.get("successful_retargets", []) or [])
                + list(rule_set.get("failed_retargets", []) or [])
            )
        ),
        seed_successful_args=tuple(dict(item) for item in seed_successful_args),
        seed_failed_args=tuple(dict(item) for item in seed_failed_args),
        suggested_control_actions=("ACTION3", "ACTION4"),
        control_policy="m3_dynamic_available_controls",
        metrics=metrics,
        success_metrics=success_metrics,
        diagnostic_metrics=diagnostic_metrics,
        expected_signal=expected_signal,
        falsification_criterion=falsification_criterion,
        discriminates_rule_families=tuple(
            str(item) for item in discriminates_rule_families
        ),
        planning_rationale=planning_rationale,
    )


def summarize_selection_followup_requests(
    *,
    rule_sets: Sequence[Mapping[str, Any]],
    all_requests: Sequence[SelectionRuleFollowupRequest],
    emitted_requests: Sequence[SelectionRuleFollowupRequest],
    max_followup_requests: int,
) -> Dict[str, Any]:
    by_probe = {
        family: len([request for request in emitted_requests if request.probe_family == family])
        for family in sorted({request.probe_family for request in emitted_requests})
    }
    return {
        "selection_rule_sets_consumed": len(rule_sets),
        "candidate_followup_requests": len(all_requests),
        "followup_requests_generated": len(emitted_requests),
        "truncated_followup_requests": max(0, len(all_requests) - len(emitted_requests)),
        "max_followup_requests": int(max_followup_requests),
        "request_budget_respected": len(emitted_requests) <= int(max_followup_requests),
        "probe_families": by_probe,
        "explicit_arg_requests": len(
            [
                request
                for request in emitted_requests
                if request.target_action_args is not None
            ]
        ),
        "dynamic_arg_resolution_requests": len(
            [
                request
                for request in emitted_requests
                if request.target_action_args is None
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
        "followup_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def candidate_band_values(known_y_values: set[int]) -> Tuple[int, ...]:
    candidates = (6, 12, 18, 24)
    return tuple(value for value in candidates if value not in known_y_values)


def candidate_neighbor_x_values(known_x_values: set[int]) -> Tuple[int, ...]:
    candidates = (6, 12, 18, 24, 30, 36)
    return tuple(value for value in candidates if value not in known_x_values)


def tested_arg_keys(rule_set: Mapping[str, Any]) -> set[str]:
    return {
        args_key(item)
        for item in (
            list(rule_set.get("successful_retargets", []) or [])
            + list(rule_set.get("failed_retargets", []) or [])
        )
    }


def rule_id_for_family(rule_set: Mapping[str, Any], rule_family: str) -> str | None:
    for rule in rule_set.get("candidate_rules", []) or []:
        if str(dict(rule).get("rule_family", "")) == rule_family:
            return str(dict(rule).get("rule_id", ""))
    return None


def selection_followup_request_id(
    *,
    selection_rule_candidate_id: str,
    probe_family: str,
    target_action_args: Mapping[str, Any] | None,
    index: int,
) -> str:
    source = selection_rule_candidate_id.replace("::", "_") or "unknown_selection_rule"
    if target_action_args is None:
        args_token = f"dynamic_{index:02d}"
    else:
        args_token = "_".join(
            f"{key}{target_action_args[key]}" for key in sorted(target_action_args)
        )
    return f"m3_15::{source}::{probe_family}::{args_token}"


def next_probe_index(
    rule_set: Mapping[str, Any],
    probe_family: str,
    target_action_args: Mapping[str, Any] | None,
) -> int:
    if target_action_args is not None:
        return 0
    seeds = (
        len(rule_set.get("successful_retargets", []) or [])
        + len(rule_set.get("failed_retargets", []) or [])
    )
    family_offset = 1 if "success" in probe_family else 2
    return seeds + family_offset


def ordered_unique(values: Sequence[str]) -> Tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def args_key(args: Mapping[str, Any]) -> str:
    return json.dumps({str(key): args[key] for key in sorted(args)}, sort_keys=True)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _context_args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def write_dynamic_retarget_selection_followup_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH
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
        description="Build M3.15 selection-rule follow-up requests.",
    )
    parser.add_argument(
        "--selection-rules",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--max-followup-requests",
        type=int,
        default=DEFAULT_MAX_SELECTION_FOLLOWUP_REQUESTS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_selection_followup_planning(
        selection_rules_path=args.selection_rules,
        max_followup_requests=args.max_followup_requests,
    )
    write_dynamic_retarget_selection_followup_requests(payload, args.out)
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
