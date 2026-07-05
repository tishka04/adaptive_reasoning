"""M1.3h experiment value estimates.

M1.3h ranks unresolved mechanic experiments by expected usefulness. It does not
execute actions, does not revise hypotheses, and never confirms anything.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .mechanic_grounded_candidates import (
    DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
)
from .polymorphic_a25_adapter import DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH
from .polymorphic_a25_pretest import (
    DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    TESTABLE,
)

DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "experiment_value_estimates.json"
)

DEFAULT_SCORE_WEIGHTS = {
    "delta_score": 0.4,
    "novelty_score": 0.3,
    "diversity_score": 0.3,
}


@dataclass(frozen=True)
class ExperimentalValueEstimate:
    """Value estimate for one unresolved mechanic experiment candidate."""

    candidate_id: str
    game_id: str
    candidate_type: str
    action: str
    required_observation: str
    score: float
    expected_information_gain: float
    delta_score: float
    novelty_score: float
    diversity_score: float
    expected_delta_magnitude: float
    expected_state_change: str
    expected_novelty: str
    expected_disambiguation_power: float
    score_basis: str
    recommended: bool = False
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def with_recommendation(self, recommended: bool) -> "ExperimentalValueEstimate":
        return ExperimentalValueEstimate(
            candidate_id=self.candidate_id,
            game_id=self.game_id,
            candidate_type=self.candidate_type,
            action=self.action,
            required_observation=self.required_observation,
            score=self.score,
            expected_information_gain=self.expected_information_gain,
            delta_score=self.delta_score,
            novelty_score=self.novelty_score,
            diversity_score=self.diversity_score,
            expected_delta_magnitude=self.expected_delta_magnitude,
            expected_state_change=self.expected_state_change,
            expected_novelty=self.expected_novelty,
            expected_disambiguation_power=self.expected_disambiguation_power,
            score_basis=self.score_basis,
            recommended=bool(recommended),
            status=self.status,
            trace_support_counted_as_proof=self.trace_support_counted_as_proof,
            prior_counted_as_proof=self.prior_counted_as_proof,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "candidate_type": self.candidate_type,
            "action": self.action,
            "required_observation": self.required_observation,
            "score": round(float(self.score), 4),
            "expected_information_gain": round(
                float(self.expected_information_gain),
                4,
            ),
            "delta_score": round(float(self.delta_score), 4),
            "novelty_score": round(float(self.novelty_score), 4),
            "diversity_score": round(float(self.diversity_score), 4),
            "expected_delta_magnitude": round(
                float(self.expected_delta_magnitude),
                6,
            ),
            "expected_state_change": self.expected_state_change,
            "expected_novelty": self.expected_novelty,
            "expected_disambiguation_power": round(
                float(self.expected_disambiguation_power),
                4,
            ),
            "score_basis": self.score_basis,
            "recommended": self.recommended,
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def run_experiment_value_estimation(
    *,
    pretest_path: str | Path = DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    candidates_path: str | Path = DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
    observed_experiments_path: str | Path | None = DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH,
    game_id: str | None = None,
    max_recommended: int = 5,
    score_weights: Mapping[str, float] = DEFAULT_SCORE_WEIGHTS,
) -> Dict[str, Any]:
    """Rank testable mechanic candidates by expected experiment value."""
    rows = load_testable_pretest_rows(pretest_path, game_id=game_id)
    support_by_id = load_candidate_support_by_id(candidates_path)
    observed_by_id = load_observed_experiments_by_id(observed_experiments_path)
    estimates = estimate_experiment_values(
        rows,
        support_by_id=support_by_id,
        observed_by_id=observed_by_id,
        score_weights=score_weights,
    )
    estimates = mark_recommended_estimates(
        estimates,
        max_recommended=max_recommended,
    )
    return {
        "config": {
            "pretest_path": str(pretest_path),
            "candidates_path": str(candidates_path),
            "observed_experiments_path": (
                str(observed_experiments_path)
                if observed_experiments_path is not None
                else None
            ),
            "game_id": game_id,
            "max_recommended": int(max_recommended),
            "score_weights": dict(score_weights),
        },
        "summary": summarize_experiment_value_estimates(estimates),
        "estimates": [estimate.to_dict() for estimate in estimates],
        "recommended": [
            estimate.to_dict() for estimate in estimates if estimate.recommended
        ],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def load_testable_pretest_rows(
    path: str | Path = DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    *,
    game_id: str | None = None,
) -> Tuple[Dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        dict(row)
        for row in payload.get("results", []) or []
        if str(row.get("testability_status", "")) == TESTABLE
        and str(row.get("status", "")) == "UNRESOLVED"
    ]
    if game_id is not None:
        rows = [row for row in rows if str(row.get("game_id", "")) == str(game_id)]
    return tuple(rows)


def load_candidate_support_by_id(
    path: str | Path = DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result: Dict[str, Dict[str, Any]] = {}
    for index, candidate in enumerate(payload.get("candidates", []) or []):
        result[f"m1e{index:04d}"] = dict(candidate)
    return result


def load_observed_experiments_by_id(
    path: str | Path | None,
) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        str(experiment.get("candidate_id", "")): dict(experiment)
        for experiment in payload.get("experiments", []) or []
        if experiment.get("candidate_id")
    }


def estimate_experiment_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    support_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    observed_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    score_weights: Mapping[str, float] = DEFAULT_SCORE_WEIGHTS,
) -> Tuple[ExperimentalValueEstimate, ...]:
    support = support_by_id or {}
    observed = observed_by_id or {}
    type_counts = Counter(str(row.get("candidate_type", "")) for row in rows)
    action_types = _action_types(rows)
    max_action_type_count = max(
        (len(types) for types in action_types.values()),
        default=1,
    )
    delta_raw = [
        _expected_delta_magnitude(
            row,
            support.get(_candidate_prefix(row), {}),
            observed.get(str(row.get("candidate_id", "")), {}),
        )[0]
        for row in rows
    ]
    max_delta = max(delta_raw, default=1.0) or 1.0
    total = max(1, len(rows))

    estimates: List[ExperimentalValueEstimate] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        support_candidate = support.get(_candidate_prefix(row), {})
        observed_experiment = observed.get(candidate_id, {})
        raw_delta, basis = _expected_delta_magnitude(
            row,
            support_candidate,
            observed_experiment,
        )
        delta_score = _clamp(raw_delta / max_delta)
        candidate_type = str(row.get("candidate_type", ""))
        type_count = type_counts.get(candidate_type, 1)
        novelty_score = _clamp(1.0 - ((type_count - 1) / max(1, total - 1)))
        disambiguation = _disambiguation_power(
            row,
            action_types=action_types,
            max_action_type_count=max_action_type_count,
        )
        diversity_score = disambiguation
        score = (
            float(score_weights.get("delta_score", 0.4)) * delta_score
            + float(score_weights.get("novelty_score", 0.3)) * novelty_score
            + float(score_weights.get("diversity_score", 0.3)) * diversity_score
        )
        estimates.append(
            ExperimentalValueEstimate(
                candidate_id=candidate_id,
                game_id=str(row.get("game_id", "")),
                candidate_type=candidate_type,
                action=str(row.get("action", "")),
                required_observation=str(row.get("required_observation", "")),
                score=round(score, 6),
                expected_information_gain=round(score, 6),
                delta_score=round(delta_score, 6),
                novelty_score=round(novelty_score, 6),
                diversity_score=round(diversity_score, 6),
                expected_delta_magnitude=round(raw_delta, 6),
                expected_state_change=_expected_state_change(row, observed_experiment),
                expected_novelty=_expected_novelty_label(novelty_score),
                expected_disambiguation_power=round(disambiguation, 6),
                score_basis=basis,
            )
        )
    return tuple(
        sorted(
            estimates,
            key=lambda estimate: (
                estimate.score,
                estimate.delta_score,
                estimate.diversity_score,
                estimate.novelty_score,
                estimate.candidate_id,
            ),
            reverse=True,
        )
    )


def mark_recommended_estimates(
    estimates: Sequence[ExperimentalValueEstimate],
    *,
    max_recommended: int = 5,
) -> Tuple[ExperimentalValueEstimate, ...]:
    """Mark a high-value, type-diverse recommendation set."""
    selected_ids: set[str] = set()
    selected_types: set[str] = set()
    selected_actions: set[Tuple[str, str]] = set()
    limit = max(0, int(max_recommended))
    candidates = list(estimates)
    for candidate_type in sorted({estimate.candidate_type for estimate in candidates}):
        if len(selected_ids) >= limit:
            break
        type_candidates = [
            estimate
            for estimate in candidates
            if estimate.candidate_type == candidate_type
            and estimate.candidate_id not in selected_ids
        ]
        if not type_candidates:
            continue
        best_for_type = max(
            type_candidates,
            key=lambda estimate: (
                estimate.score,
                estimate.delta_score,
                estimate.diversity_score,
                estimate.candidate_id,
            ),
        )
        selected_ids.add(best_for_type.candidate_id)
        selected_types.add(best_for_type.candidate_type)
        selected_actions.add((best_for_type.game_id, best_for_type.action))

    while len(selected_ids) < limit and candidates:
        remaining = [
            estimate
            for estimate in candidates
            if estimate.candidate_id not in selected_ids
        ]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda estimate: (
                estimate.score
                + (0.08 if estimate.candidate_type not in selected_types else 0.0)
                + (0.04 if (estimate.game_id, estimate.action) not in selected_actions else 0.0),
                estimate.score,
                estimate.delta_score,
                estimate.candidate_id,
            ),
        )
        selected_ids.add(best.candidate_id)
        selected_types.add(best.candidate_type)
        selected_actions.add((best.game_id, best.action))
    return tuple(
        estimate.with_recommendation(estimate.candidate_id in selected_ids)
        for estimate in estimates
    )


def summarize_experiment_value_estimates(
    estimates: Sequence[ExperimentalValueEstimate],
) -> Dict[str, Any]:
    recommended = [estimate for estimate in estimates if estimate.recommended]
    mean_score = (
        sum(estimate.expected_information_gain for estimate in estimates)
        / len(estimates)
        if estimates
        else 0.0
    )
    all_types = {estimate.candidate_type for estimate in estimates}
    recommended_types = {estimate.candidate_type for estimate in recommended}
    return {
        "candidates_total": len(estimates),
        "recommended_experiments": len(recommended),
        "mean_information_score": round(mean_score, 4),
        "type_diversity": round(
            len(recommended_types) / max(1, min(len(all_types), len(recommended))),
            4,
        )
        if recommended
        else 0.0,
        "candidate_types_total": len(all_types),
        "recommended_candidate_types": len(recommended_types),
        "recommended_types": sorted(recommended_types),
        "recommended_by_game": dict(
            sorted(Counter(estimate.game_id for estimate in recommended).items())
        ),
        "recommended_by_type": dict(
            sorted(Counter(estimate.candidate_type for estimate in recommended).items())
        ),
        "score_basis_counts": dict(
            sorted(Counter(estimate.score_basis for estimate in estimates).items())
        ),
        "wrong_confirmations": 0,
    }


def write_experiment_value_estimates(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _expected_delta_magnitude(
    row: Mapping[str, Any],
    support_candidate: Mapping[str, Any],
    observed_experiment: Mapping[str, Any],
) -> Tuple[float, str]:
    if observed_experiment:
        delta = observed_experiment.get("measured_delta", {}) or {}
        changed_ratio = float(delta.get("changed_cell_ratio", 0.0) or 0.0)
        changed_pixels = float(observed_experiment.get("changed_pixels", 0.0) or 0.0)
        return max(changed_ratio * 1000.0, math.log1p(changed_pixels)), "observed_delta"

    candidate_type = str(row.get("candidate_type", ""))
    support_rate = float(support_candidate.get("support_rate", 0.0) or 0.0)
    evidence = support_candidate.get("evidence", {}) or {}
    if candidate_type == "object_motion_candidate":
        return (
            support_rate
            * (1.0 + float(evidence.get("mean_motion_vectors", 0.0) or 0.0)),
            "estimated_from_trace_motion",
        )
    if candidate_type == "contact_change_candidate":
        changed_pairs = evidence.get("dominant_changed_contact_pairs", []) or []
        pair_mass = sum(float(item.get("count", 0.0) or 0.0) for item in changed_pairs)
        return support_rate * (1.0 + math.log1p(pair_mass)), "estimated_from_trace_contact"
    if candidate_type == "object_lifecycle_candidate":
        created = float(evidence.get("mean_created_object_count", 0.0) or 0.0)
        removed = float(evidence.get("mean_removed_object_count", 0.0) or 0.0)
        return support_rate * (1.0 + created + removed), "estimated_from_trace_lifecycle"
    if candidate_type == "shape_zone_candidate":
        delta = abs(float(evidence.get("mean_object_count_delta", 0.0) or 0.0))
        zones = len(evidence.get("dominant_zones_after", []) or [])
        return support_rate * (1.0 + delta + 0.25 * zones), "estimated_from_trace_shape"
    if candidate_type == "position_effect_candidate":
        changed_ratio = float(evidence.get("mean_changed_cell_ratio", 0.0) or 0.0)
        arg_rate = float(evidence.get("action_arg_rate", 0.0) or 0.0)
        return support_rate * (1.0 + changed_ratio * 100.0 + arg_rate), "estimated_from_trace_position"
    affordance = row.get("available_live_affordance", {}) or {}
    return (
        float(affordance.get("live_non_background_object_count", 0.0) or 0.0),
        "estimated_from_live_affordance",
    )


def _expected_state_change(
    row: Mapping[str, Any],
    observed_experiment: Mapping[str, Any],
) -> str:
    if observed_experiment:
        delta = observed_experiment.get("measured_delta", {}) or {}
        metric = str(delta.get("metric", row.get("required_observation", "")))
        changed_pixels = int(observed_experiment.get("changed_pixels", 0) or 0)
        return f"{metric}: observed {changed_pixels} changed pixels"
    return f"{row.get('required_observation', '')}: estimated before/after delta"


def _expected_novelty_label(novelty_score: float) -> str:
    if novelty_score >= 0.8:
        return "high"
    if novelty_score >= 0.55:
        return "medium"
    return "low"


def _action_types(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], set[str]]:
    result: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        result[(str(row.get("game_id", "")), str(row.get("action", "")))].add(
            str(row.get("candidate_type", ""))
        )
    return result


def _disambiguation_power(
    row: Mapping[str, Any],
    *,
    action_types: Mapping[Tuple[str, str], set[str]],
    max_action_type_count: int,
) -> float:
    types = action_types.get(
        (str(row.get("game_id", "")), str(row.get("action", ""))),
        set(),
    )
    return _clamp(len(types) / max(1, max_action_type_count))


def _candidate_prefix(row: Mapping[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id", ""))
    return candidate_id.split(":", 1)[0]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3h experiment value estimation.",
    )
    parser.add_argument(
        "--pretest",
        type=Path,
        default=DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--observed",
        type=Path,
        default=DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH,
    )
    parser.add_argument("--game-id", default="")
    parser.add_argument("--max-recommended", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_experiment_value_estimation(
        pretest_path=args.pretest,
        candidates_path=args.candidates,
        observed_experiments_path=args.observed,
        game_id=args.game_id or None,
        max_recommended=args.max_recommended,
    )
    write_experiment_value_estimates(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
