"""M3.14 candidate-only induction of dynamic retarget selection rules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .dynamic_retarget_mechanism_consolidation import (
    DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "dynamic_retarget_selection_rules.json"
)


@dataclass(frozen=True)
class RetargetSelectionRuleSet:
    """Candidate-only rule set for distinguishing retarget affordances."""

    selection_rule_candidate_id: str
    source_mechanism_candidate_id: str
    source_refined_hypothesis_id: str
    game_id: str
    candidate_mechanic: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    repositioning_action: str
    target_action: str
    initial_consumed_args: Dict[str, Any] | None
    successful_retargets: Tuple[Dict[str, Any], ...]
    failed_retargets: Tuple[Dict[str, Any], ...]
    positive_metrics: Tuple[str, ...]
    non_decisive_or_negative_metrics: Tuple[str, ...]
    candidate_rules: Tuple[Dict[str, Any], ...]
    observed_contrasts: Dict[str, Any]
    followup_test_hints: Tuple[Dict[str, Any], ...]
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_rule_candidate_id": self.selection_rule_candidate_id,
            "source_mechanism_candidate_id": self.source_mechanism_candidate_id,
            "source_refined_hypothesis_id": self.source_refined_hypothesis_id,
            "game_id": self.game_id,
            "candidate_mechanic": self.candidate_mechanic,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "repositioning_action": self.repositioning_action,
            "target_action": self.target_action,
            "initial_consumed_args": (
                dict(self.initial_consumed_args)
                if self.initial_consumed_args is not None
                else None
            ),
            "successful_retargets": [
                dict(item) for item in self.successful_retargets
            ],
            "failed_retargets": [dict(item) for item in self.failed_retargets],
            "positive_metrics": list(self.positive_metrics),
            "non_decisive_or_negative_metrics": list(
                self.non_decisive_or_negative_metrics
            ),
            "candidate_rules": [dict(rule) for rule in self.candidate_rules],
            "observed_contrasts": dict(self.observed_contrasts),
            "followup_test_hints": [dict(item) for item in self.followup_test_hints],
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
        }


def run_dynamic_retarget_selection_rule_induction(
    *,
    mechanism_candidates_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(mechanism_candidates_path)
    mechanism_candidates = [
        dict(row) for row in payload.get("mechanism_candidates", []) or []
    ]
    rule_sets = induce_retarget_selection_rule_sets(mechanism_candidates)
    return {
        "config": {
            "mechanism_candidates_path": str(mechanism_candidates_path),
            "schema_version": "m3.dynamic_retarget_selection_rules.v1",
            "inputs_read": ["M3.13"],
            "artifacts_not_modified": [
                "M2",
                "M3.8",
                "M3.9",
                "M3.10",
                "M3.11",
                "M3.12",
                "M3.13",
                "A32",
                "A33",
            ],
            "induction_policy": {
                "rules_are_candidate_only": True,
                "requires_success_and_failure_contrast": True,
                "changed_pixels_not_used_as_success_metric": True,
                "execution_performed": False,
                "support_forced_to_zero": True,
            },
        },
        "summary": summarize_rule_sets(
            mechanism_candidates=mechanism_candidates,
            rule_sets=rule_sets,
        ),
        "selection_rule_sets": [rule_set.to_dict() for rule_set in rule_sets],
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
    }


def induce_retarget_selection_rule_sets(
    mechanism_candidates: Sequence[Mapping[str, Any]],
) -> Tuple[RetargetSelectionRuleSet, ...]:
    rule_sets: list[RetargetSelectionRuleSet] = []
    for candidate in mechanism_candidates:
        successes = tuple(
            dict(item) for item in candidate.get("successful_retargets", []) or []
        )
        failures = tuple(
            dict(item) for item in candidate.get("failed_retargets", []) or []
        )
        if not successes or not failures:
            continue
        positive_metrics = tuple(
            str(item) for item in candidate.get("positive_metrics", []) or []
        )
        non_decisive = tuple(
            str(item)
            for item in candidate.get("non_decisive_or_negative_metrics", []) or []
        )
        observed_contrasts = retarget_contrasts(successes, failures)
        rules = candidate_rules_for_contrasts(
            candidate=candidate,
            observed_contrasts=observed_contrasts,
            positive_metrics=positive_metrics,
            non_decisive_or_negative_metrics=non_decisive,
        )
        if not rules:
            continue
        context_replay = tuple(
            str(item) for item in candidate.get("context_replay", []) or []
        )
        rule_sets.append(
            RetargetSelectionRuleSet(
                selection_rule_candidate_id=selection_rule_candidate_id(
                    game_id=str(candidate.get("game_id", "")),
                    repositioning_action=str(
                        candidate.get("repositioning_action", "")
                    ),
                    target_action=str(candidate.get("target_action", "")),
                ),
                source_mechanism_candidate_id=str(
                    candidate.get("mechanism_candidate_id", "")
                ),
                source_refined_hypothesis_id=str(
                    candidate.get("source_refined_hypothesis_id", "")
                ),
                game_id=str(candidate.get("game_id", "")),
                candidate_mechanic=str(candidate.get("candidate_mechanic", "")),
                context_replay=context_replay,
                context_replay_args=_context_args_tuple(
                    candidate.get("context_replay_args")
                ),
                repositioning_action=str(candidate.get("repositioning_action", "")),
                target_action=str(candidate.get("target_action", "")),
                initial_consumed_args=(
                    dict(candidate.get("initial_consumed_args", {}) or {})
                    if candidate.get("initial_consumed_args") is not None
                    else None
                ),
                successful_retargets=successes,
                failed_retargets=failures,
                positive_metrics=positive_metrics,
                non_decisive_or_negative_metrics=non_decisive,
                candidate_rules=tuple(rules),
                observed_contrasts=observed_contrasts,
                followup_test_hints=tuple(
                    followup_hints_for_contrasts(
                        target_action=str(candidate.get("target_action", "")),
                        observed_contrasts=observed_contrasts,
                        positive_metrics=positive_metrics,
                    )
                ),
            )
        )
    return tuple(
        sorted(rule_sets, key=lambda item: item.selection_rule_candidate_id)
    )


def candidate_rules_for_contrasts(
    *,
    candidate: Mapping[str, Any],
    observed_contrasts: Mapping[str, Any],
    positive_metrics: Sequence[str],
    non_decisive_or_negative_metrics: Sequence[str],
) -> list[Dict[str, Any]]:
    rules: list[Dict[str, Any]] = []
    base_id = selection_rule_candidate_id(
        game_id=str(candidate.get("game_id", "")),
        repositioning_action=str(candidate.get("repositioning_action", "")),
        target_action=str(candidate.get("target_action", "")),
    )
    if observed_contrasts.get("same_x_mixed_outcomes") or observed_contrasts.get(
        "same_y_mixed_outcomes"
    ):
        rules.append(
            candidate_rule(
                rule_id=f"{base_id}::row_or_band_dependent_retarget",
                rule_family="row_or_band_dependent_retarget",
                description=(
                    "Retarget validity depends on a coordinate-and-band "
                    "interaction, not on x or y alone. The current contrast "
                    "keeps both the successful and failed examples open as "
                    "candidate-only evidence."
                ),
                predicate_features=[
                    "target_arg_x",
                    "target_arg_y",
                    "spatial_band_after_repositioning",
                    "same_x_mixed_outcome",
                    "same_y_mixed_outcome",
                ],
                positive_examples=list(candidate.get("successful_retargets", []) or []),
                counterexamples=list(candidate.get("failed_retargets", []) or []),
                observed_contrasts=dict(observed_contrasts),
                falsification_criterion=(
                    "Additional retargets sharing the proposed band relation fail, "
                    "or retargets matching the failed band relation succeed under "
                    "the same replay and grounded metrics."
                ),
            )
        )
    if {
        "local_patch_before_after",
        "object_positions_before_after",
    }.intersection(set(positive_metrics)):
        rules.append(
            candidate_rule(
                rule_id=f"{base_id}::local_patch_transformability",
                rule_family="local_patch_transformability",
                description=(
                    "A retarget is valid when the local patch around the target "
                    "is transformable by the target action and the object-position "
                    "effect is specific to the retargeted action."
                ),
                predicate_features=[
                    "local_patch_before_after",
                    "object_positions_before_after",
                    "target_local_patch_signature",
                    "post_repositioning_object_layout",
                ],
                positive_examples=list(candidate.get("successful_retargets", []) or []),
                counterexamples=list(candidate.get("failed_retargets", []) or []),
                observed_contrasts=dict(observed_contrasts),
                falsification_criterion=(
                    "A retarget with a similar local patch fails, or a retarget "
                    "without the transformable local patch succeeds on grounded "
                    "local_patch/object_positions metrics."
                ),
            )
        )
    if "changed_pixels" in non_decisive_or_negative_metrics:
        rules.append(
            candidate_rule(
                rule_id=f"{base_id}::specific_effect_over_global_pixels",
                rule_family="specific_effect_over_global_pixels",
                description=(
                    "Retarget success should be selected by grounded local/object "
                    "effects rather than by maximizing global changed_pixels, "
                    "which can be larger for controls."
                ),
                predicate_features=[
                    "local_patch_before_after",
                    "object_positions_before_after",
                    "changed_pixels_as_effect_radar",
                ],
                positive_examples=list(candidate.get("successful_retargets", []) or []),
                counterexamples=list(candidate.get("failed_retargets", []) or []),
                observed_contrasts=dict(observed_contrasts),
                falsification_criterion=(
                    "Global changed_pixels alone predicts successful retargets "
                    "better than the grounded local/object metrics on follow-up "
                    "retargets from the same causal replay."
                ),
            )
        )
    return rules


def candidate_rule(
    *,
    rule_id: str,
    rule_family: str,
    description: str,
    predicate_features: Sequence[str],
    positive_examples: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    observed_contrasts: Mapping[str, Any],
    falsification_criterion: str,
) -> Dict[str, Any]:
    return {
        "rule_id": rule_id,
        "rule_family": rule_family,
        "description": description,
        "predicate_features": [str(item) for item in predicate_features],
        "positive_examples": [dict(item) for item in positive_examples],
        "counterexamples": [dict(item) for item in counterexamples],
        "observed_contrasts": dict(observed_contrasts),
        "falsification_criterion": falsification_criterion,
        "testable": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "rule_counted_as_confirmation": False,
    }


def retarget_contrasts(
    successful_retargets: Sequence[Mapping[str, Any]],
    failed_retargets: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    success_points = [xy_tuple(item) for item in successful_retargets]
    failed_points = [xy_tuple(item) for item in failed_retargets]
    success_points = [item for item in success_points if item is not None]
    failed_points = [item for item in failed_points if item is not None]
    success_x = sorted({x for x, _ in success_points})
    failed_x = sorted({x for x, _ in failed_points})
    success_y = sorted({y for _, y in success_points})
    failed_y = sorted({y for _, y in failed_points})
    same_x = []
    for x in sorted(set(success_x).intersection(failed_x)):
        same_x.append(
            {
                "x": x,
                "successful_y_values": sorted(
                    {y for point_x, y in success_points if point_x == x}
                ),
                "failed_y_values": sorted(
                    {y for point_x, y in failed_points if point_x == x}
                ),
            }
        )
    same_y = []
    for y in sorted(set(success_y).intersection(failed_y)):
        same_y.append(
            {
                "y": y,
                "successful_x_values": sorted(
                    {x for x, point_y in success_points if point_y == y}
                ),
                "failed_x_values": sorted(
                    {x for x, point_y in failed_points if point_y == y}
                ),
            }
        )
    return {
        "successful_points": [
            {"x": int(x), "y": int(y)} for x, y in sorted(success_points)
        ],
        "failed_points": [
            {"x": int(x), "y": int(y)} for x, y in sorted(failed_points)
        ],
        "successful_x_values": [int(item) for item in success_x],
        "failed_x_values": [int(item) for item in failed_x],
        "successful_y_values": [int(item) for item in success_y],
        "failed_y_values": [int(item) for item in failed_y],
        "same_x_mixed_outcomes": same_x,
        "same_y_mixed_outcomes": same_y,
        "pure_x_rule_blocked": bool(same_x),
        "pure_y_rule_blocked": bool(same_y),
    }


def followup_hints_for_contrasts(
    *,
    target_action: str,
    observed_contrasts: Mapping[str, Any],
    positive_metrics: Sequence[str],
) -> list[Dict[str, Any]]:
    hints: list[Dict[str, Any]] = []
    for contrast in observed_contrasts.get("same_x_mixed_outcomes", []) or []:
        x = contrast.get("x")
        hints.append(
            {
                "hint_family": "same_x_different_band_probe",
                "target_action": target_action,
                "description": (
                    "Probe additional y bands at the same x to test whether "
                    "the band relation explains success versus failure."
                ),
                "fixed_x": x,
                "metrics": [str(metric) for metric in positive_metrics],
                "status": "UNRESOLVED",
                "support": 0,
            }
        )
    for contrast in observed_contrasts.get("same_y_mixed_outcomes", []) or []:
        y = contrast.get("y")
        hints.append(
            {
                "hint_family": "same_y_neighbor_x_probe",
                "target_action": target_action,
                "description": (
                    "Probe neighboring x values on the mixed-outcome band to "
                    "separate coordinate effects from local-patch effects."
                ),
                "fixed_y": y,
                "metrics": [str(metric) for metric in positive_metrics],
                "status": "UNRESOLVED",
                "support": 0,
            }
        )
    hints.append(
        {
            "hint_family": "local_patch_similarity_probe",
            "target_action": target_action,
            "description": (
                "Compare targets with local patches similar to successful and "
                "failed retargets under the same replay."
            ),
            "metrics": [str(metric) for metric in positive_metrics],
            "status": "UNRESOLVED",
            "support": 0,
        }
    )
    return hints


def summarize_rule_sets(
    *,
    mechanism_candidates: Sequence[Mapping[str, Any]],
    rule_sets: Sequence[RetargetSelectionRuleSet],
) -> Dict[str, Any]:
    candidate_rules = [
        rule for rule_set in rule_sets for rule in rule_set.candidate_rules
    ]
    return {
        "mechanism_candidates_consumed": len(mechanism_candidates),
        "selection_rule_sets": len(rule_sets),
        "candidate_rules": len(candidate_rules),
        "rules_with_falsification": len(
            [rule for rule in candidate_rules if rule.get("falsification_criterion")]
        ),
        "rule_families": sorted(
            {str(rule.get("rule_family", "")) for rule in candidate_rules}
        ),
        "selection_problem_sets": len(
            [
                rule_set
                for rule_set in rule_sets
                if rule_set.successful_retargets and rule_set.failed_retargets
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


def selection_rule_candidate_id(
    *,
    game_id: str,
    repositioning_action: str,
    target_action: str,
) -> str:
    game_token = str(game_id).split("-", 1)[0] or "unknown_game"
    return "::".join(
        [
            "m3_14",
            game_token,
            f"{repositioning_action}_{target_action}",
            "retarget_selection_rule",
        ]
    )


def xy_tuple(args: Mapping[str, Any]) -> Tuple[int, int] | None:
    try:
        return (int(args["x"]), int(args["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def write_dynamic_retarget_selection_rules(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        description="Induce M3.14 dynamic retarget selection rules.",
    )
    parser.add_argument(
        "--mechanism-candidates",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_selection_rule_induction(
        mechanism_candidates_path=args.mechanism_candidates,
    )
    write_dynamic_retarget_selection_rules(payload, args.out)
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
