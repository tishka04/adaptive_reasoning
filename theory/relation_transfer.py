"""A18 relation transfer as an epistemic testing prior.

Transfer is deliberately weaker than confirmation. If a relation predicate was
confirmed in one game, this module can bias matching unresolved relation
hypotheses in another game so the designer tests them earlier. It never marks
the target hypothesis confirmed and never counts the transferred prior as proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, List, Sequence, Tuple

from .epistemic_metrics import HypothesisRecord, HypothesisStatus
from .generic_discriminating_experiment_designer import DiscriminatingPrediction


@dataclass(frozen=True)
class RelationTransferPrior:
    """A confirmed source-game relation that can bias future tests."""

    source_game_id: str
    source_key: str
    predicate: str
    outcome: str
    weight: float = 1.0
    status: HypothesisStatus = HypothesisStatus.CONFIRMED
    counted_as_proof: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_game_id": self.source_game_id,
            "source_key": self.source_key,
            "predicate": self.predicate,
            "outcome": self.outcome,
            "weight": self.weight,
            "status": self.status.value,
            "counted_as_proof": self.counted_as_proof,
        }


@dataclass
class RelationTransferRun:
    """Result of applying source relation priors to target hypotheses."""

    source_game_id: str
    target_game_id: str
    priors: List[RelationTransferPrior] = field(default_factory=list)
    target_predictions: List[DiscriminatingPrediction] = field(default_factory=list)
    transferred_predictions: List[DiscriminatingPrediction] = field(default_factory=list)
    prior_counted_as_proof: bool = False

    @property
    def transferred_count(self) -> int:
        return len(self.transferred_predictions)

    @property
    def transferred_keys(self) -> List[str]:
        return [prediction.key for prediction in self.transferred_predictions]

    @property
    def confirmed_target_count(self) -> int:
        return sum(
            1
            for prediction in self.transferred_predictions
            if prediction.status == HypothesisStatus.CONFIRMED
        )

    @property
    def transferred_but_unconfirmed(self) -> bool:
        return self.transferred_count > 0 and self.confirmed_target_count == 0

    @property
    def records(self) -> List[HypothesisRecord]:
        return [
            HypothesisRecord(
                key=prediction.key,
                description=(
                    f"transferred relation prior for {prediction.predicate_name} "
                    f"->{prediction.outcome}"
                ),
                status=prediction.status,
                support=0,
                contradictions=0,
                experiments_spent=0,
            )
            for prediction in self.transferred_predictions
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_game_id": self.source_game_id,
            "target_game_id": self.target_game_id,
            "priors": [prior.to_dict() for prior in self.priors],
            "target_prediction_count": len(self.target_predictions),
            "transferred_count": self.transferred_count,
            "transferred_keys": self.transferred_keys,
            "confirmed_target_count": self.confirmed_target_count,
            "transferred_but_unconfirmed": self.transferred_but_unconfirmed,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def extract_relation_transfer_priors(
    revisions: Sequence[Any],
    *,
    source_game_id: str,
    prior_weight: float = 1.0,
) -> List[RelationTransferPrior]:
    """Extract transferable priors from confirmed source relation revisions."""
    priors: List[RelationTransferPrior] = []
    for revision in revisions:
        family = str(getattr(revision, "family", "") or "")
        status = _status_from_revision(revision)
        if family != "relation" or status != HypothesisStatus.CONFIRMED:
            continue
        predicate = str(getattr(revision, "predicate", "") or "")
        outcome = str(
            getattr(revision, "observed_outcome", "")
            or getattr(revision, "predicted_outcome", "")
            or ""
        )
        if not predicate or not outcome:
            continue
        priors.append(
            RelationTransferPrior(
                source_game_id=source_game_id,
                source_key=str(getattr(revision, "key", "")),
                predicate=predicate,
                outcome=outcome,
                weight=float(prior_weight),
            )
        )
    return _dedupe_priors(priors)


def relation_predictions_from_candidates(
    candidates: Sequence[Any],
    *,
    outcomes: Sequence[str] = ("preserved", "broken"),
    predicates: Sequence[str] = ("same_shape", "adjacent_to", "aligned_with", "paired_with"),
) -> List[DiscriminatingPrediction]:
    """Create unresolved relation predictions from trace-discovered candidates."""
    predictions: List[DiscriminatingPrediction] = []
    predicate_set = {str(predicate) for predicate in predicates}
    for candidate in candidates:
        candidate_predicates = {
            str(getattr(predicate, "name", "")): int(getattr(predicate, "support", 0) or 0)
            for predicate in getattr(candidate, "predicates", ())
        }
        for predicate, support in candidate_predicates.items():
            if predicate not in predicate_set:
                continue
            for outcome in outcomes:
                predictions.append(
                    DiscriminatingPrediction(
                        key=(
                            f"relation::{candidate.action}::{predicate}::"
                            f"colors{candidate.source_color}_{candidate.target_color}::"
                            f"{outcome}"
                        ),
                        action=candidate.action,
                        source_color=int(candidate.source_color),
                        target_color=int(candidate.target_color),
                        family="relation",
                        predicate=predicate,
                        predicted_outcome=str(outcome),
                        support=int(support),
                        transition_support=int(
                            getattr(candidate, "transition_support", 0) or 0
                        ),
                        status=HypothesisStatus.UNRESOLVED,
                    )
                )
    return _dedupe_predictions(predictions)


def apply_relation_transfer_priors(
    predictions: Sequence[DiscriminatingPrediction],
    priors: Sequence[RelationTransferPrior],
) -> List[DiscriminatingPrediction]:
    """Attach matching source priors while keeping target hypotheses unresolved."""
    transferred: List[DiscriminatingPrediction] = []
    for prediction in predictions:
        matching = [
            prior
            for prior in priors
            if prediction.normalized_family == "relation"
            and prediction.predicate_name == prior.predicate
            and prediction.outcome == prior.outcome
        ]
        if not matching:
            transferred.append(prediction)
            continue
        prior_weight = sum(prior.weight for prior in matching)
        prior_keys = tuple(
            _dedupe(
                list(prediction.prior_source_keys)
                + [prior.source_key for prior in matching]
            )
        )
        transferred.append(
            replace(
                prediction,
                status=HypothesisStatus.UNRESOLVED,
                epistemic_prior=prediction.epistemic_prior + prior_weight,
                prior_source_keys=prior_keys,
                prior_counted_as_proof=False,
            )
        )
    return transferred


def run_relation_transfer(
    *,
    source_game_id: str,
    source_revisions: Sequence[Any],
    target_game_id: str,
    target_candidates: Sequence[Any],
    prior_weight: float = 1.0,
) -> RelationTransferRun:
    """Apply source confirmed relation priors to target candidate predictions."""
    priors = extract_relation_transfer_priors(
        source_revisions,
        source_game_id=source_game_id,
        prior_weight=prior_weight,
    )
    target_predictions = relation_predictions_from_candidates(target_candidates)
    transferred = apply_relation_transfer_priors(target_predictions, priors)
    transferred_predictions = [
        prediction
        for prediction in transferred
        if prediction.epistemic_prior > 0.0 or prediction.prior_source_keys
    ]
    return RelationTransferRun(
        source_game_id=source_game_id,
        target_game_id=target_game_id,
        priors=priors,
        target_predictions=target_predictions,
        transferred_predictions=transferred_predictions,
        prior_counted_as_proof=False,
    )


def _status_from_revision(revision: Any) -> HypothesisStatus:
    status = getattr(revision, "status_after", getattr(revision, "status", None))
    if isinstance(status, HypothesisStatus):
        return status
    if status is None:
        return HypothesisStatus.UNRESOLVED
    return HypothesisStatus(str(status))


def _dedupe_priors(priors: Iterable[RelationTransferPrior]) -> List[RelationTransferPrior]:
    seen: set[Tuple[str, str, str]] = set()
    result: List[RelationTransferPrior] = []
    for prior in priors:
        key = (prior.source_key, prior.predicate, prior.outcome)
        if key in seen:
            continue
        seen.add(key)
        result.append(prior)
    return result


def _dedupe_predictions(
    predictions: Iterable[DiscriminatingPrediction],
) -> List[DiscriminatingPrediction]:
    seen: set[str] = set()
    result: List[DiscriminatingPrediction] = []
    for prediction in predictions:
        if prediction.key in seen:
            continue
        seen.add(prediction.key)
        result.append(prediction)
    return result


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
