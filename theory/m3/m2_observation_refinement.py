"""M3.8 refinement from M2 experiment observations to candidate hypotheses."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .m2_candidate_experiment_runner import DEFAULT_M3_M2_RESULTS_PATH


DEFAULT_REFINED_M2_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "refined_candidate_hypotheses_from_m2.json"
)
M3_REFINEMENT_TRUTH_STATUS = "NOT_EVALUATED_BY_M3"
METRIC_ORDER = (
    "local_patch_before_after",
    "changed_pixels",
    "object_positions_before_after",
    "object_counts_before_after",
    "contact_graph_before_after",
    "object_shape_zone_before_after",
    "topology_before_after",
)


@dataclass(frozen=True)
class ExperimentalObservationGroup:
    """One deduplicated experimental signature from M3.7b."""

    experimental_signature: Dict[str, Any]
    source_hypothesis_ids: Tuple[str, ...]
    request_ids: Tuple[str, ...]
    role: str
    metric: str
    signal_source: str
    support_events: int
    contradiction_events: int
    neutral_events: int
    raw_support_events: int
    diagnostic_only: bool
    baseline_signal: float
    perturbation_signal: float
    effect_size: float
    metric_grounding_status: str

    def label(self) -> str:
        if self.role == "positive":
            return f"{self.metric}_support"
        if self.role == "contradiction":
            return f"{self.metric}_contradiction"
        if self.role == "neutral":
            return f"{self.metric}_neutral"
        return self.metric

    def base_key(self) -> Tuple[str, str, str]:
        return (
            str(self.experimental_signature.get("game_id", "")),
            _stable_json(self.experimental_signature.get("context_replay", [])),
            str(self.experimental_signature.get("target_action", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experimental_signature": dict(self.experimental_signature),
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "request_ids": list(self.request_ids),
            "role": self.role,
            "metric": self.metric,
            "signal_source": self.signal_source,
            "support_events": int(self.support_events),
            "contradiction_events": int(self.contradiction_events),
            "neutral_events": int(self.neutral_events),
            "raw_support_events": int(self.raw_support_events),
            "diagnostic_only": self.diagnostic_only,
            "baseline_signal": float(self.baseline_signal),
            "perturbation_signal": float(self.perturbation_signal),
            "effect_size": float(self.effect_size),
            "metric_grounding_status": self.metric_grounding_status,
        }


@dataclass(frozen=True)
class RefinedCandidateHypothesis:
    """Candidate-only refined hypothesis derived from M3 observations."""

    refined_hypothesis_id: str
    source_hypothesis_ids: Tuple[str, ...]
    game_id: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    control_actions: Tuple[str, ...]
    observed_effect_family: str
    candidate_mechanic: str
    mechanistic_signature: Dict[str, Any]
    positive_observations: Tuple[str, ...] = ()
    neutral_observations: Tuple[str, ...] = ()
    diagnostic_only_observations: Tuple[str, ...] = ()
    derived_from_observations: Tuple[Dict[str, Any], ...] = ()
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
    input_support_events_counted_as_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refined_hypothesis_id": self.refined_hypothesis_id,
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "game_id": self.game_id,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "control_actions": list(self.control_actions),
            "observed_effect_family": self.observed_effect_family,
            "candidate_mechanic": self.candidate_mechanic,
            "mechanistic_signature": dict(self.mechanistic_signature),
            "positive_observations": list(self.positive_observations),
            "neutral_observations": list(self.neutral_observations),
            "diagnostic_only_observations": list(
                self.diagnostic_only_observations
            ),
            "derived_from_observations": [
                dict(row) for row in self.derived_from_observations
            ],
            "evidence_summary": {
                "grounded_positive_observations": len(self.positive_observations),
                "neutral_observations": len(self.neutral_observations),
                "diagnostic_only_observations": len(
                    self.diagnostic_only_observations
                ),
                "input_support_events": sum(
                    int(row.get("support_events", 0) or 0)
                    for row in self.derived_from_observations
                ),
                "unique_grounded_positive_signatures": len(
                    [
                        row
                        for row in self.derived_from_observations
                        if row.get("role") == "positive"
                    ]
                ),
                "input_support_events_counted_as_support": False,
            },
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
            "input_support_events_counted_as_support": (
                self.input_support_events_counted_as_support
            ),
        }


def run_m2_observation_refinement(
    *,
    experiment_results_path: str | Path = DEFAULT_M3_M2_RESULTS_PATH,
) -> Dict[str, Any]:
    payload = _load_json(experiment_results_path)
    experiments = [
        dict(row) for row in payload.get("controlled_experiments", []) or []
    ]
    experimental_groups = deduplicate_experimental_observations(experiments)
    refined = build_refined_candidate_hypotheses(experimental_groups)
    return {
        "config": {
            "experiment_results_path": str(experiment_results_path),
            "schema_version": "m3.m2_observation_refinement.v1",
            "inputs_read": ["M3.7b"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "grouping_policy": {
                "experimental_signature": [
                    "game_id",
                    "context_replay",
                    "context_replay_args",
                    "target_action",
                    "control_action",
                    "metric",
                    "signal_source",
                ],
                "mechanistic_signature": [
                    "game_id",
                    "context_replay",
                    "target_action",
                    "observed_effect_family",
                ],
            },
        },
        "summary": summarize_refinement(
            experiments=experiments,
            experimental_groups=experimental_groups,
            refined=refined,
        ),
        "experimental_observation_groups": [
            group.to_dict() for group in experimental_groups
        ],
        "refined_candidate_hypotheses": [
            hypothesis.to_dict() for hypothesis in refined
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
        "input_support_events_counted_as_support": False,
    }


def deduplicate_experimental_observations(
    experiments: Sequence[Mapping[str, Any]],
) -> Tuple[ExperimentalObservationGroup, ...]:
    by_key: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in experiments:
        signature = experimental_signature(row)
        by_key[_stable_json(signature)].append(dict(row))

    groups = [
        build_experimental_observation_group(rows)
        for _, rows in sorted(by_key.items(), key=lambda item: item[0])
    ]
    return tuple(sorted(groups, key=_observation_sort_key))


def experimental_signature(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "game_id": str(row.get("game_id", "")),
        "context_replay": [str(item) for item in row.get("context_replay", []) or []],
        "context_replay_args": _context_replay_args(row),
        "target_action": str(row.get("target_action", "")),
        "control_action": str(row.get("control_action", "")),
        "metric": str(row.get("metric", row.get("predicted_metric", ""))),
        "signal_source": str(row.get("signal_source", "")),
    }


def build_experimental_observation_group(
    rows: Sequence[Mapping[str, Any]],
) -> ExperimentalObservationGroup:
    first = dict(rows[0])
    signature = experimental_signature(first)
    metric = str(signature.get("metric", ""))
    support_events = 1 if any(int(row.get("support_events", 0) or 0) > 0 for row in rows) else 0
    contradiction_events = (
        1 if any(int(row.get("contradiction_events", 0) or 0) > 0 for row in rows) else 0
    )
    diagnostic_only = any(bool(row.get("diagnostic_only")) for row in rows)
    neutral_events = (
        1
        if not support_events
        and not contradiction_events
        and any(int(row.get("neutral_events", 0) or 0) > 0 for row in rows)
        else 0
    )
    raw_support_events = (
        1 if any(int(row.get("raw_support_events", 0) or 0) > 0 for row in rows) else 0
    )
    role = observation_role(
        support_events=support_events,
        contradiction_events=contradiction_events,
        neutral_events=neutral_events,
        diagnostic_only=diagnostic_only,
    )
    delta = dict(first.get("delta", {}) or {})
    return ExperimentalObservationGroup(
        experimental_signature=signature,
        source_hypothesis_ids=tuple(
            sorted(
                {
                    str(row.get("source_hypothesis_id") or row.get("hypothesis_key", ""))
                    for row in rows
                    if str(row.get("source_hypothesis_id") or row.get("hypothesis_key", ""))
                }
            )
        ),
        request_ids=tuple(
            sorted({str(row.get("request_id", "")) for row in rows if row.get("request_id")})
        ),
        role=role,
        metric=metric,
        signal_source=str(signature.get("signal_source", "")),
        support_events=support_events,
        contradiction_events=contradiction_events,
        neutral_events=neutral_events,
        raw_support_events=raw_support_events,
        diagnostic_only=diagnostic_only,
        baseline_signal=float(first.get("baseline_signal", delta.get("baseline_signal", 0.0)) or 0.0),
        perturbation_signal=float(
            first.get("perturbation_signal", delta.get("perturbation_signal", 0.0))
            or 0.0
        ),
        effect_size=float(delta.get("effect_size", 0.0) or 0.0),
        metric_grounding_status=str(first.get("metric_grounding_status", "")),
    )


def observation_role(
    *,
    support_events: int,
    contradiction_events: int,
    neutral_events: int,
    diagnostic_only: bool,
) -> str:
    if diagnostic_only:
        return "diagnostic_only"
    if support_events > 0:
        return "positive"
    if contradiction_events > 0:
        return "contradiction"
    if neutral_events > 0:
        return "neutral"
    return "unclassified"


def build_refined_candidate_hypotheses(
    experimental_groups: Sequence[ExperimentalObservationGroup],
) -> Tuple[RefinedCandidateHypothesis, ...]:
    by_base: dict[Tuple[str, str, str], list[ExperimentalObservationGroup]] = defaultdict(list)
    for group in experimental_groups:
        by_base[group.base_key()].append(group)

    hypotheses: list[RefinedCandidateHypothesis] = []
    for groups in by_base.values():
        positives = [group for group in groups if group.role == "positive"]
        if not positives:
            continue
        family = observed_effect_family(positives)
        candidate_mechanic = candidate_mechanic_for_family(family)
        first = groups[0].experimental_signature
        game_id = str(first.get("game_id", ""))
        context_replay = tuple(str(item) for item in first.get("context_replay", []) or [])
        target_action = str(first.get("target_action", ""))
        source_hypothesis_ids = tuple(
            sorted(
                {
                    source
                    for group in groups
                    for source in group.source_hypothesis_ids
                    if source
                }
            )
        )
        mechanistic_signature = {
            "game_id": game_id,
            "context_replay": list(context_replay),
            "target_action": target_action,
            "observed_effect_family": family,
        }
        hypotheses.append(
            RefinedCandidateHypothesis(
                refined_hypothesis_id=refined_hypothesis_id(
                    game_id=game_id,
                    context_replay=context_replay,
                    target_action=target_action,
                    effect_family=family,
                ),
                source_hypothesis_ids=source_hypothesis_ids,
                game_id=game_id,
                context_replay=context_replay,
                context_replay_args=_context_args_tuple(first.get("context_replay_args")),
                target_action=target_action,
                control_actions=tuple(
                    sorted(
                        {
                            str(group.experimental_signature.get("control_action", ""))
                            for group in groups
                            if group.experimental_signature.get("control_action")
                        }
                    )
                ),
                observed_effect_family=family,
                candidate_mechanic=candidate_mechanic,
                mechanistic_signature=mechanistic_signature,
                positive_observations=tuple(
                    group.label() for group in sorted(positives, key=_observation_sort_key)
                ),
                neutral_observations=tuple(
                    group.label()
                    for group in sorted(
                        [group for group in groups if group.role == "neutral"],
                        key=_observation_sort_key,
                    )
                ),
                diagnostic_only_observations=tuple(
                    group.label()
                    for group in sorted(
                        [group for group in groups if group.role == "diagnostic_only"],
                        key=_observation_sort_key,
                    )
                ),
                derived_from_observations=tuple(
                    group.to_dict()
                    for group in sorted(groups, key=_observation_sort_key)
                ),
            )
        )

    return tuple(sorted(hypotheses, key=lambda item: item.refined_hypothesis_id))


def observed_effect_family(
    positive_groups: Sequence[ExperimentalObservationGroup],
) -> str:
    metrics = {group.metric for group in positive_groups}
    if "object_positions_before_after" in metrics and "changed_pixels" in metrics:
        return "global_motion"
    if "object_positions_before_after" in metrics:
        return "object_repositioning"
    if "contact_graph_before_after" in metrics:
        return "contact_graph_reconfiguration"
    if "local_patch_before_after" in metrics:
        return "local_patch_effect"
    if "changed_pixels" in metrics:
        return "global_grid_change"
    return "grounded_observable_change"


def candidate_mechanic_for_family(effect_family: str) -> str:
    if effect_family == "global_motion":
        return "global_object_repositioning_after_consumption"
    if effect_family == "object_repositioning":
        return "object_repositioning_after_consumption"
    if effect_family == "contact_graph_reconfiguration":
        return "contact_graph_reconfiguration_after_consumption"
    if effect_family == "local_patch_effect":
        return "local_patch_effect_after_consumption"
    if effect_family == "global_grid_change":
        return "global_grid_change_after_consumption"
    return "grounded_observable_change_after_consumption"


def refined_hypothesis_id(
    *,
    game_id: str,
    context_replay: Sequence[str],
    target_action: str,
    effect_family: str,
) -> str:
    game_token = str(game_id).split("-", 1)[0] or "unknown_game"
    return "::".join(
        [
            "m3_8",
            game_token,
            context_token(context_replay),
            str(target_action),
            str(effect_family),
        ]
    )


def context_token(context_replay: Sequence[str]) -> str:
    tokens = []
    for action in context_replay:
        text = str(action)
        if text.startswith("ACTION"):
            suffix = text.removeprefix("ACTION")
            tokens.append(f"A{suffix}")
        else:
            tokens.append(text)
    return "_".join(tokens) or "reset"


def summarize_refinement(
    *,
    experiments: Sequence[Mapping[str, Any]],
    experimental_groups: Sequence[ExperimentalObservationGroup],
    refined: Sequence[RefinedCandidateHypothesis],
) -> Dict[str, Any]:
    source_ids = {
        str(row.get("source_hypothesis_id") or row.get("hypothesis_key", ""))
        for row in experiments
        if str(row.get("source_hypothesis_id") or row.get("hypothesis_key", ""))
    }
    refined_source_ids = {
        source
        for hypothesis in refined
        for source in hypothesis.source_hypothesis_ids
    }
    return {
        "source_experiments": len(experiments),
        "source_hypotheses_consumed": len(source_ids),
        "experimental_signature_groups": len(experimental_groups),
        "unique_grounded_positive_signatures": len(
            [group for group in experimental_groups if group.role == "positive"]
        ),
        "unique_neutral_signatures": len(
            [group for group in experimental_groups if group.role == "neutral"]
        ),
        "unique_diagnostic_only_signatures": len(
            [group for group in experimental_groups if group.role == "diagnostic_only"]
        ),
        "input_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in experiments
        ),
        "input_support_events_after_signature_dedup": sum(
            int(group.support_events) for group in experimental_groups
        ),
        "input_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in experiments
        ),
        "refined_candidate_hypotheses": len(refined),
        "mechanistic_signature_groups": len(refined),
        "merged_source_hypotheses": len(refined_source_ids),
        "duplicate_source_hypotheses_merged": any(
            len(item.source_hypothesis_ids) > 1 for item in refined
        ),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "input_support_events_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def write_refined_candidate_hypotheses(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_REFINED_M2_HYPOTHESES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _observation_sort_key(group: ExperimentalObservationGroup) -> Tuple[int, str, str]:
    metric = group.metric
    try:
        metric_index = METRIC_ORDER.index(metric)
    except ValueError:
        metric_index = len(METRIC_ORDER)
    return (metric_index, metric, group.signal_source)


def _context_replay_args(row: Mapping[str, Any]) -> list[Dict[str, Any]] | None:
    raw = row.get("context_replay_args")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _context_args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build M3.8 refined candidate hypotheses from M2 observations.",
    )
    parser.add_argument(
        "--m3-results",
        type=Path,
        default=DEFAULT_M3_M2_RESULTS_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REFINED_M2_HYPOTHESES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_m2_observation_refinement(
        experiment_results_path=args.m3_results,
    )
    write_refined_candidate_hypotheses(payload, args.out)
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
