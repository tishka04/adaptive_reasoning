"""A27 negative memory inside the functional non-ar25 agenda.

A26 measures useful non-ar25 progress after an agenda-gated relation test. A27
feeds a refuted relation context back into the next agenda so the runner avoids
that exact relation context and tests an alternative while keeping functional
progress measurable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

from .epistemic_metrics import HypothesisStatus
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
class FunctionalAgendaNegativeRecord:
    """Context-specific warning from a refuted functional-agenda relation."""

    target_game: str
    failed_relation_context: str
    tested_hypothesis: str
    observed_outcome: str
    effect: str = "avoid_same_functional_agenda_context"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_game": self.target_game,
            "failed_relation_context": self.failed_relation_context,
            "tested_hypothesis": self.tested_hypothesis,
            "observed_outcome": self.observed_outcome,
            "effect": self.effect,
        }


@dataclass
class FunctionalAgendaNegativeMemory:
    """Negative memory used by a functional non-ar25 agenda."""

    records: List[FunctionalAgendaNegativeRecord] = field(default_factory=list)

    @property
    def excluded_relation_context_signatures(self) -> List[str]:
        result: List[str] = []
        for record in self.records:
            if record.failed_relation_context not in result:
                result.append(record.failed_relation_context)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "excluded_relation_context_signatures": (
                self.excluded_relation_context_signatures
            ),
        }


@dataclass
class FunctionalNegativeMemoryAgendaResult:
    """Two-pass A27 run: useful action, memory, adapted useful action."""

    game_id: str
    trace_path: Path
    first_attempt: NonAr25FunctionalProgressResult | None = None
    negative_memory: FunctionalAgendaNegativeMemory = field(
        default_factory=FunctionalAgendaNegativeMemory
    )
    second_attempt: NonAr25FunctionalProgressResult | None = None
    error: str = ""

    @property
    def negative_memory_used_in_functional_agenda(self) -> bool:
        return bool(
            self.negative_memory.records
            and self.second_attempt
            and self.second_attempt.agenda_result
            and self.second_attempt.agenda_result.excluded_relation_context_signatures
        )

    @property
    def failed_relation_context_not_selected_again(self) -> bool:
        failed = set(self.negative_memory.excluded_relation_context_signatures)
        if not failed or self.second_attempt is None or self.second_attempt.agenda_result is None:
            return False
        selected = set(_selected_relation_contexts(self.second_attempt))
        return bool(selected and selected.isdisjoint(failed))

    @property
    def alternative_relation_context_selected(self) -> bool:
        if self.first_attempt is None or self.second_attempt is None:
            return False
        first_contexts = set(_selected_relation_contexts(self.first_attempt))
        second_contexts = set(_selected_relation_contexts(self.second_attempt))
        return bool(second_contexts and second_contexts != first_contexts)

    @property
    def functional_progress_non_ar25_remains_measurable(self) -> bool:
        return bool(
            self.first_attempt
            and self.first_attempt.functional_progress_non_ar25
            and self.second_attempt
            and self.second_attempt.functional_progress_non_ar25
        )

    @property
    def first_selected_relation_contexts(self) -> List[str]:
        return _selected_relation_contexts(self.first_attempt)

    @property
    def second_selected_relation_contexts(self) -> List[str]:
        return _selected_relation_contexts(self.second_attempt)

    @property
    def wrong_confirmations(self) -> int:
        return (
            (0 if self.first_attempt is None else self.first_attempt.wrong_confirmations)
            + (0 if self.second_attempt is None else self.second_attempt.wrong_confirmations)
        )

    @property
    def trace_support_counted_as_proof(self) -> bool:
        return bool(
            self.first_attempt
            and self.first_attempt.trace_support_counted_as_proof
            or self.second_attempt
            and self.second_attempt.trace_support_counted_as_proof
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "negative_memory_used_in_functional_agenda": (
                self.negative_memory_used_in_functional_agenda
            ),
            "failed_relation_context_not_selected_again": (
                self.failed_relation_context_not_selected_again
            ),
            "alternative_relation_context_selected": (
                self.alternative_relation_context_selected
            ),
            "functional_progress_non_ar25_remains_measurable": (
                self.functional_progress_non_ar25_remains_measurable
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "negative_memory": self.negative_memory.to_dict(),
            "first_selected_relation_contexts": self.first_selected_relation_contexts,
            "second_selected_relation_contexts": self.second_selected_relation_contexts,
            "first_attempt": (
                self.first_attempt.to_dict() if self.first_attempt else None
            ),
            "second_attempt": (
                self.second_attempt.to_dict() if self.second_attempt else None
            ),
            "error": self.error,
        }


def run_functional_negative_memory_agenda(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
) -> FunctionalNegativeMemoryAgendaResult:
    """Run a functional agenda, store a refutation, then adapt the next agenda."""
    path = Path(trace_path)
    result = FunctionalNegativeMemoryAgendaResult(game_id=game_id, trace_path=path)

    first = run_non_ar25_functional_progress(
        game_id=game_id,
        trace_path=path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
    )
    result.first_attempt = first
    if first.error:
        result.error = f"first_attempt_failed:{first.error}"
        return result

    result.negative_memory = build_functional_agenda_negative_memory(first)
    if not result.negative_memory.records:
        result.error = "no_functional_negative_memory_records"
        return result

    second = run_non_ar25_functional_progress(
        game_id=game_id,
        trace_path=path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        excluded_relation_context_signatures=(
            result.negative_memory.excluded_relation_context_signatures
        ),
    )
    result.second_attempt = second
    if second.error:
        result.error = f"second_attempt_failed:{second.error}"
    return result


def build_functional_agenda_negative_memory(
    result: NonAr25FunctionalProgressResult,
) -> FunctionalAgendaNegativeMemory:
    """Build context memory from refuted relation hypotheses."""
    agenda = result.agenda_result
    if agenda is None:
        return FunctionalAgendaNegativeMemory()
    predictions_by_key = {
        prediction.key: prediction
        for prediction in agenda.selected_predictions_before_observation
    }
    records: List[FunctionalAgendaNegativeRecord] = []
    for revision in agenda.revisions:
        if revision.status_after != HypothesisStatus.REFUTED:
            continue
        prediction = predictions_by_key.get(revision.key)
        if prediction is None:
            continue
        context = non_ar25_relation_context_signature_from_prediction(prediction)
        records.append(
            FunctionalAgendaNegativeRecord(
                target_game=result.game_id,
                failed_relation_context=context,
                tested_hypothesis=prediction.key,
                observed_outcome=revision.observed_outcome,
            )
        )
    return FunctionalAgendaNegativeMemory(records=_dedupe_records(records))


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


def _dedupe_records(
    records: Sequence[FunctionalAgendaNegativeRecord],
) -> List[FunctionalAgendaNegativeRecord]:
    seen: set[str] = set()
    result: List[FunctionalAgendaNegativeRecord] = []
    for record in records:
        if record.failed_relation_context in seen:
            continue
        seen.add(record.failed_relation_context)
        result.append(record)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A27 negative memory inside functional non-ar25 agenda."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_functional_negative_memory_agenda(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
