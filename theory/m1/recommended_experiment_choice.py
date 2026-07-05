"""M1.3i recommended experiment to polymorphic A25 choice.

This bridge turns the best M1.3h recommendation into a structured polymorphic
A25 experimental choice, executes the single concrete action through M1.3g, and
records a mechanic observation. It performs no hypothesis revision and never
confirms anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _env_dir

from .experiment_value_estimator import (
    DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH,
)
from .polymorphic_a25_adapter import (
    PolymorphicMechanicExperiment,
    execute_polymorphic_candidate,
)
from .polymorphic_a25_pretest import DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH

DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "recommended_polymorphic_a25_choice.json"
)


@dataclass(frozen=True)
class MechanicHypothesisCandidate:
    """An unresolved mechanic hypothesis exposed to a polymorphic A25 layer."""

    candidate_id: str
    game_id: str
    mechanic_family: str
    action: str
    predicted_metric: str
    expected_outcome: str
    expected_information_gain: float = 0.0
    score: float = 0.0
    observed_delta: Dict[str, Any] | None = None
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def with_observed_delta(
        self,
        observed_delta: Mapping[str, Any],
    ) -> "MechanicHypothesisCandidate":
        return MechanicHypothesisCandidate(
            candidate_id=self.candidate_id,
            game_id=self.game_id,
            mechanic_family=self.mechanic_family,
            action=self.action,
            predicted_metric=self.predicted_metric,
            expected_outcome=self.expected_outcome,
            expected_information_gain=self.expected_information_gain,
            score=self.score,
            observed_delta=dict(observed_delta),
            status=self.status,
            trace_support_counted_as_proof=self.trace_support_counted_as_proof,
            prior_counted_as_proof=self.prior_counted_as_proof,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "mechanic_family": self.mechanic_family,
            "action": self.action,
            "predicted_metric": self.predicted_metric,
            "expected_outcome": self.expected_outcome,
            "expected_information_gain": round(float(self.expected_information_gain), 4),
            "score": round(float(self.score), 4),
            "observed_delta": self.observed_delta,
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


@dataclass(frozen=True)
class PolymorphicA25ExperimentalChoice:
    """A mechanic experiment choice a future opt-in A25 path can consume."""

    candidate_id: str
    game_id: str
    action: str
    mechanic_family: str
    predicted_metric: str
    expected_outcome: str
    expected_information_gain: float
    score: float
    selection_reason: str = "m1_3h_recommended_highest_information_gain"
    a25_choice_type: str = "polymorphic_mechanic_experiment"
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    @property
    def competing_keys(self) -> Tuple[str, ...]:
        return (self.candidate_id,)

    @property
    def prediction_families(self) -> Tuple[str, ...]:
        return (self.mechanic_family,)

    @property
    def predicted_outcomes(self) -> Tuple[str, ...]:
        return (self.expected_outcome,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "action": self.action,
            "mechanic_family": self.mechanic_family,
            "predicted_metric": self.predicted_metric,
            "expected_outcome": self.expected_outcome,
            "expected_information_gain": round(float(self.expected_information_gain), 4),
            "score": round(float(self.score), 4),
            "selection_reason": self.selection_reason,
            "a25_choice_type": self.a25_choice_type,
            "competing_keys": list(self.competing_keys),
            "prediction_families": list(self.prediction_families),
            "predicted_outcomes": list(self.predicted_outcomes),
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


@dataclass(frozen=True)
class MechanicObservation:
    """One measured mechanic observation from the recommended experiment."""

    candidate_id: str
    game_id: str
    mechanic_family: str
    action: str
    predicted_metric: str
    expected_outcome: str
    observed_delta: Dict[str, Any]
    env_actions: int
    changed_pixels: int
    mechanic_experiment_generated: bool
    revision_performed: bool = False
    wrong_confirmations: int = 0
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "mechanic_family": self.mechanic_family,
            "action": self.action,
            "predicted_metric": self.predicted_metric,
            "expected_outcome": self.expected_outcome,
            "observed_delta": dict(self.observed_delta),
            "env_actions": int(self.env_actions),
            "changed_pixels": int(self.changed_pixels),
            "mechanic_experiment_generated": self.mechanic_experiment_generated,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def run_recommended_polymorphic_a25_choice(
    *,
    estimates_path: str | Path = DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH,
    pretest_path: str | Path = DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    candidate_id: str | None = None,
    game_id: str | None = None,
) -> Dict[str, Any]:
    """Build and execute the top recommended polymorphic A25 choice."""
    estimates = load_recommended_estimates(estimates_path)
    recommendation = select_recommendation(
        estimates,
        candidate_id=candidate_id,
        game_id=game_id,
    )
    pretest_row = load_pretest_row_for_candidate(
        pretest_path,
        str(recommendation.get("candidate_id", "")),
    )
    hypothesis = mechanic_hypothesis_from_recommendation(recommendation)
    choice = polymorphic_choice_from_hypothesis(hypothesis)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    experiment = execute_polymorphic_candidate(pretest_row, environments_dir=env_dir)
    observation = mechanic_observation_from_experiment(
        hypothesis,
        experiment,
    )
    hypothesis = hypothesis.with_observed_delta(observation.observed_delta)
    return {
        "config": {
            "estimates_path": str(estimates_path),
            "pretest_path": str(pretest_path),
            "environments_dir": str(env_dir),
            "candidate_id": candidate_id,
            "game_id": game_id,
        },
        "summary": summarize_recommended_choice(
            choice=choice,
            observation=observation,
        ),
        "mechanic_hypothesis_candidate": hypothesis.to_dict(),
        "polymorphic_a25_choice": choice.to_dict(),
        "mechanic_observation": observation.to_dict(),
        "executed_experiment": experiment.to_dict(),
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def load_recommended_estimates(
    path: str | Path = DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH,
) -> Tuple[Dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("recommended", []) or [
        item for item in payload.get("estimates", []) or [] if item.get("recommended")
    ]
    return tuple(dict(row) for row in rows)


def select_recommendation(
    estimates: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str | None = None,
    game_id: str | None = None,
) -> Dict[str, Any]:
    candidates = [dict(row) for row in estimates]
    if candidate_id is not None:
        candidates = [
            row for row in candidates if str(row.get("candidate_id", "")) == candidate_id
        ]
    if game_id is not None:
        candidates = [row for row in candidates if str(row.get("game_id", "")) == game_id]
    if not candidates:
        raise ValueError("no recommended experiment matches the requested filters")
    return max(
        candidates,
        key=lambda row: (
            float(row.get("score", 0.0) or 0.0),
            float(row.get("expected_information_gain", 0.0) or 0.0),
            str(row.get("candidate_id", "")),
        ),
    )


def load_pretest_row_for_candidate(
    path: str | Path,
    candidate_id: str,
) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in payload.get("results", []) or []:
        if str(row.get("candidate_id", "")) == str(candidate_id):
            return dict(row)
    raise ValueError(f"pretest row not found for candidate_id={candidate_id!r}")


def mechanic_hypothesis_from_recommendation(
    recommendation: Mapping[str, Any],
) -> MechanicHypothesisCandidate:
    return MechanicHypothesisCandidate(
        candidate_id=str(recommendation.get("candidate_id", "")),
        game_id=str(recommendation.get("game_id", "")),
        mechanic_family=str(recommendation.get("candidate_type", "")),
        action=str(recommendation.get("action", "")),
        predicted_metric=str(recommendation.get("required_observation", "")),
        expected_outcome=str(recommendation.get("expected_state_change", "")),
        expected_information_gain=float(
            recommendation.get("expected_information_gain", 0.0) or 0.0
        ),
        score=float(recommendation.get("score", 0.0) or 0.0),
    )


def polymorphic_choice_from_hypothesis(
    hypothesis: MechanicHypothesisCandidate,
) -> PolymorphicA25ExperimentalChoice:
    return PolymorphicA25ExperimentalChoice(
        candidate_id=hypothesis.candidate_id,
        game_id=hypothesis.game_id,
        action=hypothesis.action,
        mechanic_family=hypothesis.mechanic_family,
        predicted_metric=hypothesis.predicted_metric,
        expected_outcome=hypothesis.expected_outcome,
        expected_information_gain=hypothesis.expected_information_gain,
        score=hypothesis.score,
    )


def mechanic_observation_from_experiment(
    hypothesis: MechanicHypothesisCandidate,
    experiment: PolymorphicMechanicExperiment,
) -> MechanicObservation:
    return MechanicObservation(
        candidate_id=hypothesis.candidate_id,
        game_id=hypothesis.game_id,
        mechanic_family=hypothesis.mechanic_family,
        action=hypothesis.action,
        predicted_metric=hypothesis.predicted_metric,
        expected_outcome=hypothesis.expected_outcome,
        observed_delta=dict(experiment.measured_delta),
        env_actions=int(experiment.env_actions),
        changed_pixels=int(experiment.changed_pixels),
        mechanic_experiment_generated=bool(experiment.mechanic_experiment_generated),
        revision_performed=bool(experiment.revision_performed),
        wrong_confirmations=int(experiment.wrong_confirmations),
    )


def summarize_recommended_choice(
    *,
    choice: PolymorphicA25ExperimentalChoice,
    observation: MechanicObservation,
) -> Dict[str, Any]:
    return {
        "recommended_choices": 1,
        "a25_polymorphic_choices": 1,
        "mechanic_observations": 1,
        "env_actions": int(observation.env_actions),
        "observable_delta": bool(observation.observed_delta.get("changed")),
        "selected_candidate_id": choice.candidate_id,
        "selected_game_id": choice.game_id,
        "selected_mechanic_family": choice.mechanic_family,
        "selected_action": choice.action,
        "predicted_metric": choice.predicted_metric,
        "expected_information_gain": round(float(choice.expected_information_gain), 4),
        "revision_performed": bool(observation.revision_performed),
        "wrong_confirmations": int(observation.wrong_confirmations),
    }


def write_recommended_polymorphic_a25_choice(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3i recommended experiment to polymorphic A25 choice.",
    )
    parser.add_argument(
        "--estimates",
        type=Path,
        default=DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--pretest",
        type=Path,
        default=DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--game-id", default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_recommended_polymorphic_a25_choice(
        estimates_path=args.estimates,
        pretest_path=args.pretest,
        environments_dir=args.environments_dir,
        candidate_id=args.candidate_id or None,
        game_id=args.game_id or None,
    )
    write_recommended_polymorphic_a25_choice(payload, args.out)
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
