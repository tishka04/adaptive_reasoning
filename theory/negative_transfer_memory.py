"""A20 negative transfer memory.

When a transferred relation prior leads to a target-game refutation, the source
relation remains intact. Only the matching target context is downweighted so the
same transfer is tested later in that context while remaining available
elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, List, Sequence, Tuple

from .epistemic_metrics import HypothesisStatus
from .generic_discriminating_experiment_designer import DiscriminatingPrediction


@dataclass(frozen=True)
class NegativeTransferRecord:
    """A target-context-specific warning against a transferred relation prior."""

    source_relation: str
    target_game: str
    target_context_signature: str
    tested_hypothesis: str
    observed_outcome: str
    effect: str = "downweight_same_transfer_context"
    weight_delta: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_relation": self.source_relation,
            "target_game": self.target_game,
            "target_context_signature": self.target_context_signature,
            "tested_hypothesis": self.tested_hypothesis,
            "observed_outcome": self.observed_outcome,
            "effect": self.effect,
            "weight_delta": self.weight_delta,
        }


@dataclass
class NegativeTransferMemory:
    """In-memory collection of context-specific negative transfer records."""

    records: List[NegativeTransferRecord] = field(default_factory=list)

    def add(self, record: NegativeTransferRecord) -> None:
        if record not in self.records:
            self.records.append(record)

    def downweight_for(self, prediction: DiscriminatingPrediction) -> float:
        signature = transfer_context_signature_from_prediction(prediction)
        return sum(
            record.weight_delta
            for record in self.records
            if record.target_context_signature == signature
            and record.source_relation in prediction.prior_source_keys
        )

    def to_dict(self) -> dict[str, object]:
        return {"records": [record.to_dict() for record in self.records]}


def transfer_context_signature_from_prediction(
    prediction: DiscriminatingPrediction,
    *,
    target_game: str = "",
) -> str:
    """Return a stable target-context signature for transfer downweighting."""
    pair = (
        f"colors{prediction.source_color}_{prediction.target_color}"
        if prediction.target_color is not None
        else f"source{prediction.source_color}"
    )
    prefix = f"{target_game}::" if target_game else ""
    return (
        f"{prefix}{prediction.action}::{prediction.normalized_family}::"
        f"{prediction.predicate_name}::{pair}"
    )


def build_negative_transfer_records(
    *,
    target_game: str,
    selected_predictions: Sequence[DiscriminatingPrediction],
    revisions: Sequence[Any],
) -> List[NegativeTransferRecord]:
    """Create negative-transfer records from target refutations."""
    predictions_by_key = {prediction.key: prediction for prediction in selected_predictions}
    records: List[NegativeTransferRecord] = []
    for revision in revisions:
        status = _status_from_revision(revision)
        if status != HypothesisStatus.REFUTED:
            continue
        prediction = predictions_by_key.get(str(getattr(revision, "key", "")))
        if prediction is None or not prediction.prior_source_keys:
            continue
        for source_relation in prediction.prior_source_keys:
            records.append(
                NegativeTransferRecord(
                    source_relation=source_relation,
                    target_game=target_game,
                    target_context_signature=transfer_context_signature_from_prediction(
                        prediction,
                        target_game=target_game,
                    ),
                    tested_hypothesis=prediction.key,
                    observed_outcome=str(getattr(revision, "observed_outcome", "")),
                )
            )
    return records


def apply_negative_transfer_memory(
    predictions: Sequence[DiscriminatingPrediction],
    memory: NegativeTransferMemory,
    *,
    target_game: str = "",
) -> List[DiscriminatingPrediction]:
    """Downweight matching transferred priors without changing their status."""
    adjusted: List[DiscriminatingPrediction] = []
    for prediction in predictions:
        downweight = _downweight_for(
            prediction,
            memory.records,
            target_game=target_game,
        )
        if downweight <= 0:
            adjusted.append(prediction)
            continue
        adjusted.append(
            replace(
                prediction,
                status=HypothesisStatus.UNRESOLVED,
                epistemic_prior=max(0.0, prediction.epistemic_prior - downweight),
                prior_counted_as_proof=False,
            )
        )
    return adjusted


def _downweight_for(
    prediction: DiscriminatingPrediction,
    records: Sequence[NegativeTransferRecord],
    *,
    target_game: str,
) -> float:
    signature = transfer_context_signature_from_prediction(
        prediction,
        target_game=target_game,
    )
    return sum(
        record.weight_delta
        for record in records
        if record.target_context_signature == signature
        and record.source_relation in prediction.prior_source_keys
    )


def _status_from_revision(revision: Any) -> HypothesisStatus:
    status = getattr(revision, "status_after", getattr(revision, "status", None))
    if isinstance(status, HypothesisStatus):
        return status
    if status is None:
        return HypothesisStatus.UNRESOLVED
    return HypothesisStatus(str(status))
