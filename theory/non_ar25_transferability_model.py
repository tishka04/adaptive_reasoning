"""A28 transferability model from negative functional-agenda memories.

The model aggregates repeated negative records by predicate, predicted outcome,
and context signature. Repeated failures downweight only that local context
before the next agenda run; source confirmations are left untouched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from .epistemic_metrics import HypothesisStatus
from .non_ar25_functional_negative_memory import (
    FunctionalAgendaNegativeRecord,
    build_functional_agenda_negative_memory,
)
from .non_ar25_functional_progress import (
    DEFAULT_GAME_ID,
    DEFAULT_TRACE_PATH,
    NonAr25FunctionalProgressResult,
    run_non_ar25_functional_progress,
)
from .non_ar25_multi_relation_agenda import (
    non_ar25_relation_context_signature_from_prediction,
)


@dataclass(frozen=True)
class TransferabilityGroup:
    """Aggregated negative evidence for one relation context."""

    predicate: str
    predicted_outcome: str
    context_signature: str
    count: int = 0
    observed_outcomes: Tuple[str, ...] = ()
    tested_hypotheses: Tuple[str, ...] = ()

    def downweighted(self, *, min_negative_count: int) -> bool:
        return self.count >= int(min_negative_count)

    def to_dict(self, *, min_negative_count: int) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "predicted_outcome": self.predicted_outcome,
            "context_signature": self.context_signature,
            "count": self.count,
            "observed_outcomes": list(self.observed_outcomes),
            "tested_hypotheses": list(self.tested_hypotheses),
            "downweighted": self.downweighted(
                min_negative_count=min_negative_count
            ),
        }


@dataclass
class NegativeTransferabilityModel:
    """Local transferability map built from repeated negative memories."""

    groups: List[TransferabilityGroup] = field(default_factory=list)
    min_negative_count: int = 2

    @property
    def downweighted_context_signatures(self) -> List[str]:
        return [
            group.context_signature
            for group in self.groups
            if group.downweighted(min_negative_count=self.min_negative_count)
        ]

    def priority_adjustment_for(self, context_signature: str) -> float:
        for group in self.groups:
            if group.context_signature == context_signature:
                return -float(group.count)
        return 0.0

    def context_remains_testable(self, context_signature: str) -> bool:
        return context_signature not in set(self.downweighted_context_signatures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_negative_count": self.min_negative_count,
            "downweighted_context_signatures": self.downweighted_context_signatures,
            "groups": [
                group.to_dict(min_negative_count=self.min_negative_count)
                for group in self.groups
            ],
        }


@dataclass
class TransferabilityModelRunResult:
    """A28 run: repeated negative memories, then adapted functional agenda."""

    game_id: str
    trace_path: Path
    repeated_attempts: List[NonAr25FunctionalProgressResult] = field(default_factory=list)
    negative_records: List[FunctionalAgendaNegativeRecord] = field(default_factory=list)
    transferability_model: NegativeTransferabilityModel = field(
        default_factory=NegativeTransferabilityModel
    )
    adapted_attempt: NonAr25FunctionalProgressResult | None = None
    source_confirmed_keys: List[str] = field(default_factory=list)
    unrelated_context_signature: str = "ACTION6::aligned_with::colors9_8"
    error: str = ""

    @property
    def repeated_negative_contexts_downweighted(self) -> bool:
        return "ACTION6::same_shape::colors9_8" in (
            self.transferability_model.downweighted_context_signatures
        )

    @property
    def source_confirmations_unchanged(self) -> bool:
        if not self.source_confirmed_keys:
            return False
        tested_negative_keys = {
            record.tested_hypothesis for record in self.negative_records
        }
        return all(key not in tested_negative_keys for key in self.source_confirmed_keys)

    @property
    def unrelated_contexts_remain_testable(self) -> bool:
        return bool(
            self.transferability_model.context_remains_testable(
                self.unrelated_context_signature
            )
            and self.adapted_attempt
            and self.unrelated_context_signature
            in _selected_relation_contexts(self.adapted_attempt)
        )

    @property
    def functional_progress_still_measurable(self) -> bool:
        return bool(
            self.adapted_attempt
            and self.adapted_attempt.functional_progress_non_ar25
        )

    @property
    def wrong_confirmations(self) -> int:
        total = sum(attempt.wrong_confirmations for attempt in self.repeated_attempts)
        if self.adapted_attempt is not None:
            total += self.adapted_attempt.wrong_confirmations
        return total

    @property
    def trace_support_counted_as_proof(self) -> bool:
        return bool(
            any(attempt.trace_support_counted_as_proof for attempt in self.repeated_attempts)
            or self.adapted_attempt
            and self.adapted_attempt.trace_support_counted_as_proof
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "negative_record_count": len(self.negative_records),
            "repeated_negative_contexts_downweighted": (
                self.repeated_negative_contexts_downweighted
            ),
            "source_confirmations_unchanged": self.source_confirmations_unchanged,
            "unrelated_contexts_remain_testable": (
                self.unrelated_contexts_remain_testable
            ),
            "functional_progress_still_measurable": (
                self.functional_progress_still_measurable
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "source_confirmed_keys": list(self.source_confirmed_keys),
            "unrelated_context_signature": self.unrelated_context_signature,
            "selected_contexts_before_model": [
                _selected_relation_contexts(attempt)
                for attempt in self.repeated_attempts
            ],
            "selected_contexts_after_model": _selected_relation_contexts(
                self.adapted_attempt
            ),
            "negative_records": [
                record.to_dict() for record in self.negative_records
            ],
            "transferability_model": self.transferability_model.to_dict(),
            "adapted_attempt": (
                self.adapted_attempt.to_dict() if self.adapted_attempt else None
            ),
            "error": self.error,
        }


def build_negative_transferability_model(
    records: Sequence[FunctionalAgendaNegativeRecord],
    *,
    min_negative_count: int = 2,
) -> NegativeTransferabilityModel:
    """Aggregate negative records by predicate, outcome, and context."""
    buckets: dict[Tuple[str, str, str], List[FunctionalAgendaNegativeRecord]] = {}
    for record in records:
        predicate = _predicate_from_context(record.failed_relation_context)
        predicted = _predicted_outcome_from_hypothesis(record.tested_hypothesis)
        key = (predicate, predicted, record.failed_relation_context)
        buckets.setdefault(key, []).append(record)

    groups: List[TransferabilityGroup] = []
    for (predicate, predicted, context), items in sorted(buckets.items()):
        groups.append(
            TransferabilityGroup(
                predicate=predicate,
                predicted_outcome=predicted,
                context_signature=context,
                count=len(items),
                observed_outcomes=tuple(
                    _dedupe(record.observed_outcome for record in items)
                ),
                tested_hypotheses=tuple(
                    _dedupe(record.tested_hypothesis for record in items)
                ),
            )
        )
    return NegativeTransferabilityModel(
        groups=groups,
        min_negative_count=max(1, int(min_negative_count)),
    )


def run_negative_transferability_model(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    repeated_observations: int = 2,
    min_negative_count: int = 2,
) -> TransferabilityModelRunResult:
    """Learn transferability from repeated negative memories, then adapt."""
    path = Path(trace_path)
    result = TransferabilityModelRunResult(game_id=game_id, trace_path=path)

    for _ in range(max(1, int(repeated_observations))):
        attempt = run_non_ar25_functional_progress(
            game_id=game_id,
            trace_path=path,
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
        )
        result.repeated_attempts.append(attempt)
        if attempt.error:
            result.error = f"repeated_attempt_failed:{attempt.error}"
            return result
        result.negative_records.extend(
            build_functional_agenda_negative_memory(attempt).records
        )

    result.source_confirmed_keys = _confirmed_relation_keys(result.repeated_attempts)
    result.transferability_model = build_negative_transferability_model(
        result.negative_records,
        min_negative_count=min_negative_count,
    )
    if not result.transferability_model.downweighted_context_signatures:
        result.error = "no_repeated_negative_contexts_downweighted"
        return result

    adapted = run_non_ar25_functional_progress(
        game_id=game_id,
        trace_path=path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        excluded_relation_context_signatures=(
            result.transferability_model.downweighted_context_signatures
        ),
    )
    result.adapted_attempt = adapted
    if adapted.error:
        result.error = f"adapted_attempt_failed:{adapted.error}"
    return result


def _confirmed_relation_keys(
    attempts: Sequence[NonAr25FunctionalProgressResult],
) -> List[str]:
    keys: List[str] = []
    for attempt in attempts:
        agenda = attempt.agenda_result
        if agenda is None:
            continue
        for revision in agenda.revisions:
            if revision.status_after != HypothesisStatus.CONFIRMED:
                continue
            if revision.key not in keys:
                keys.append(revision.key)
    return keys


def _selected_relation_contexts(
    result: NonAr25FunctionalProgressResult | None,
) -> List[str]:
    if result is None or result.agenda_result is None:
        return []
    contexts: List[str] = []
    for prediction in result.agenda_result.selected_predictions_before_observation:
        context = non_ar25_relation_context_signature_from_prediction(prediction)
        if context not in contexts:
            contexts.append(context)
    return contexts


def _predicate_from_context(context_signature: str) -> str:
    parts = str(context_signature).split("::")
    return parts[1] if len(parts) >= 2 else ""


def _predicted_outcome_from_hypothesis(hypothesis_key: str) -> str:
    parts = str(hypothesis_key).split("::")
    return parts[-1] if parts else ""


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A28 transferability model from negative memories."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--repeated-observations", type=int, default=2)
    parser.add_argument("--min-negative-count", type=int, default=2)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_negative_transferability_model(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        repeated_observations=args.repeated_observations,
        min_negative_count=args.min_negative_count,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
