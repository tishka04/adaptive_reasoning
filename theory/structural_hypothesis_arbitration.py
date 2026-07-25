"""Generic disagreement-based arbitration between structural theories.

SAGE.9s originally implemented this decision inside the relational-stencil
learner.  SAGE.9x makes the decision rule explicit so the same arbiter can be
causally evaluated on procedural episodes without copying its scoring logic.
The arbiter sees candidate predictions only; it never receives the hidden
correct theory or the terminal outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class StructuralExperimentOption:
    """Predicted structural reduction for one available intervention."""

    action_key: str
    predicted_reductions: Tuple[Tuple[str, int], ...]
    hypothesis_support: Tuple[Tuple[str, int], ...] = ()
    tie_break: int = 0


@dataclass(frozen=True)
class StructuralArbitrationChoice:
    """The most discriminating intervention and its sponsoring theory."""

    action_key: str
    hypothesis_id: str
    compared_hypothesis_ids: Tuple[str, ...]
    predicted_reductions: Tuple[Tuple[str, int], ...]
    disagreement_score: float
    distinct_predictions: int
    polarity_pairs: int


def select_discriminating_structural_experiment(
    *,
    options: Sequence[StructuralExperimentOption],
    hypothesis_priority: Sequence[str] = (),
) -> StructuralArbitrationChoice | None:
    """Choose the action that maximally separates live theory predictions."""
    priority = {
        str(hypothesis_id): index
        for index, hypothesis_id in enumerate(hypothesis_priority)
    }
    best: tuple[
        tuple[int, int, int, int, int, int],
        StructuralExperimentOption,
        str,
        int,
        int,
    ] | None = None
    for option in options:
        predictions = tuple(
            (str(hypothesis_id), int(reduction))
            for hypothesis_id, reduction in option.predicted_reductions
        )
        if not predictions:
            continue
        positive = [
            hypothesis_id
            for hypothesis_id, reduction in predictions
            if reduction > 0
        ]
        if not positive:
            continue
        reductions = dict(predictions)
        values = tuple(reduction for _, reduction in predictions)
        disagreement = max(values) - min(values)
        distinct_predictions = len(set(values))
        polarity_pairs = (
            sum(value > 0 for value in values)
            * sum(value <= 0 for value in values)
        )
        sponsor = min(
            positive,
            key=lambda hypothesis_id: (
                -reductions[hypothesis_id],
                priority.get(hypothesis_id, len(priority)),
                hypothesis_id,
            ),
        )
        support = dict(option.hypothesis_support).get(sponsor, 0)
        key = (
            int(distinct_predictions > 1),
            int(disagreement),
            int(polarity_pairs),
            int(reductions[sponsor]),
            int(support),
            int(option.tie_break),
        )
        if best is None or key > best[0]:
            best = (
                key,
                option,
                sponsor,
                distinct_predictions,
                polarity_pairs,
            )
    if best is None:
        return None
    _, option, sponsor, distinct_predictions, polarity_pairs = best
    predictions = tuple(
        (str(hypothesis_id), int(reduction))
        for hypothesis_id, reduction in option.predicted_reductions
    )
    values = tuple(reduction for _, reduction in predictions)
    disagreement = max(values) - min(values)
    disagreement_score = float(
        disagreement
        + int(distinct_predictions > 1)
        + polarity_pairs
    )
    return StructuralArbitrationChoice(
        action_key=str(option.action_key),
        hypothesis_id=sponsor,
        compared_hypothesis_ids=tuple(
            sorted(hypothesis_id for hypothesis_id, _ in predictions)
        ),
        predicted_reductions=predictions,
        disagreement_score=disagreement_score,
        distinct_predictions=distinct_predictions,
        polarity_pairs=polarity_pairs,
    )


def sequential_structural_experiment(
    *,
    options: Sequence[StructuralExperimentOption],
    hypothesis_priority: Sequence[str],
) -> StructuralArbitrationChoice | None:
    """Ablation: optimize only the first live theory, ignoring disagreement."""
    if not hypothesis_priority:
        return None
    first = str(hypothesis_priority[0])
    ranked = []
    for option_index, option in enumerate(options):
        reductions = dict(option.predicted_reductions)
        reduction = int(reductions.get(first, 0))
        if reduction <= 0:
            continue
        support = int(dict(option.hypothesis_support).get(first, 0))
        ranked.append((
            reduction,
            support,
            int(option.tie_break),
            -option_index,
            option,
        ))
    if not ranked:
        return None
    _, _, _, _, option = max(ranked, key=lambda item: item[:-1])
    predictions = tuple(
        (str(hypothesis_id), int(reduction))
        for hypothesis_id, reduction in option.predicted_reductions
    )
    values = tuple(reduction for _, reduction in predictions)
    distinct_predictions = len(set(values))
    polarity_pairs = (
        sum(value > 0 for value in values)
        * sum(value <= 0 for value in values)
    )
    return StructuralArbitrationChoice(
        action_key=str(option.action_key),
        hypothesis_id=first,
        compared_hypothesis_ids=tuple(
            sorted(hypothesis_id for hypothesis_id, _ in predictions)
        ),
        predicted_reductions=predictions,
        disagreement_score=float(
            max(values)
            - min(values)
            + int(distinct_predictions > 1)
            + polarity_pairs
        ),
        distinct_predictions=distinct_predictions,
        polarity_pairs=polarity_pairs,
    )


def surviving_hypotheses(
    *,
    choice: StructuralArbitrationChoice,
    observed_reduction: int,
) -> Tuple[str, ...]:
    """Return hypotheses whose prediction matches the live observation."""
    return tuple(sorted(
        hypothesis_id
        for hypothesis_id, predicted in choice.predicted_reductions
        if int(predicted) == int(observed_reduction)
    ))


__all__ = [
    "StructuralArbitrationChoice",
    "StructuralExperimentOption",
    "select_discriminating_structural_experiment",
    "sequential_structural_experiment",
    "surviving_hypotheses",
]
