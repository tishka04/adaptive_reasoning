"""A29 cross-game transferability check.

The A28 model learns a local negative transferability map. A29 applies that
map to another non-ar25 game at the predicate-analogy level: contexts with the
same relation predicate are downweighted, while unrelated relation contexts
remain testable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

from .non_ar25_multi_relation_agenda import (
    DEFAULT_PREFERRED_PREDICATES,
    NonAr25MultiRelationAgendaResult,
    NonAr25RelationAgendaItem,
    non_ar25_relation_context_signature,
    run_non_ar25_multi_relation_agenda,
)
from .non_ar25_transferability_model import (
    DEFAULT_GAME_ID as DEFAULT_SOURCE_GAME_ID,
    DEFAULT_TRACE_PATH as DEFAULT_SOURCE_TRACE_PATH,
    TransferabilityModelRunResult,
    run_negative_transferability_model,
)

DEFAULT_TARGET_GAME_ID = "dc22-4c9bff3e"
DEFAULT_TARGET_TRACE_PATH = Path(
    "human_traces/dc22-4c9bff3e.20260616-150906.steps.jsonl"
)


@dataclass
class CrossGameTransferabilityCheckResult:
    """Result of applying one learned transferability map to another game."""

    source_game_id: str
    target_game_id: str
    source_trace_path: Path
    target_trace_path: Path
    source_model_run: TransferabilityModelRunResult | None = None
    target_baseline: NonAr25MultiRelationAgendaResult | None = None
    target_adapted: NonAr25MultiRelationAgendaResult | None = None
    analogous_context_signatures: List[str] = field(default_factory=list)
    analogy_level: str = "predicate"
    error: str = ""

    @property
    def negative_model_applied_cross_game(self) -> bool:
        return bool(
            self.source_game_id != self.target_game_id
            and self.source_model_run
            and self.source_model_run.transferability_model.downweighted_context_signatures
            and self.target_adapted
            and self.target_adapted.excluded_relation_context_signatures
        )

    @property
    def analogous_context_downweighted(self) -> bool:
        if self.target_adapted is None or not self.analogous_context_signatures:
            return False
        excluded = set(self.target_adapted.excluded_relation_context_signatures)
        adapted_contexts = set(_agenda_contexts(self.target_adapted.agenda_items))
        return bool(
            set(self.analogous_context_signatures) <= excluded
            and adapted_contexts.isdisjoint(set(self.analogous_context_signatures))
        )

    @property
    def unrelated_context_still_testable(self) -> bool:
        if self.target_adapted is None:
            return False
        adapted_contexts = set(_selected_relation_contexts(self.target_adapted))
        unrelated = [
            context
            for context in adapted_contexts
            if context not in set(self.analogous_context_signatures)
        ]
        return bool(
            unrelated
            and self.target_adapted.active_transition_produced
            and self.target_adapted.hypothesis_remains_unresolved_until_observation
        )

    @property
    def source_confirmations_unchanged(self) -> bool:
        return bool(
            self.source_model_run
            and self.source_model_run.source_confirmations_unchanged
        )

    @property
    def functional_progress_still_measurable(self) -> bool:
        return bool(
            self.source_model_run
            and self.source_model_run.functional_progress_still_measurable
        )

    @property
    def wrong_confirmations(self) -> int:
        total = 0
        if self.source_model_run is not None:
            total += self.source_model_run.wrong_confirmations
        if self.target_baseline is not None:
            total += self.target_baseline.wrong_confirmations
        if self.target_adapted is not None:
            total += self.target_adapted.wrong_confirmations
        return total

    @property
    def trace_support_counted_as_proof(self) -> bool:
        return bool(
            self.source_model_run
            and self.source_model_run.trace_support_counted_as_proof
            or self.target_baseline
            and self.target_baseline.trace_support_counted_as_proof
            or self.target_adapted
            and self.target_adapted.trace_support_counted_as_proof
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_game_id": self.source_game_id,
            "target_game_id": self.target_game_id,
            "source_trace_path": str(self.source_trace_path),
            "target_trace_path": str(self.target_trace_path),
            "negative_model_applied_cross_game": (
                self.negative_model_applied_cross_game
            ),
            "analogous_context_downweighted": self.analogous_context_downweighted,
            "unrelated_context_still_testable": self.unrelated_context_still_testable,
            "source_confirmations_unchanged": self.source_confirmations_unchanged,
            "functional_progress_still_measurable": (
                self.functional_progress_still_measurable
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "analogy_level": self.analogy_level,
            "analogous_context_signatures": list(self.analogous_context_signatures),
            "target_baseline_selected_contexts": (
                _selected_relation_contexts(self.target_baseline)
            ),
            "target_adapted_selected_contexts": (
                _selected_relation_contexts(self.target_adapted)
            ),
            "source_model_run": (
                self.source_model_run.to_dict() if self.source_model_run else None
            ),
            "target_baseline": (
                self.target_baseline.to_dict() if self.target_baseline else None
            ),
            "target_adapted": (
                self.target_adapted.to_dict() if self.target_adapted else None
            ),
            "error": self.error,
        }


def run_cross_game_transferability_check(
    *,
    source_game_id: str = DEFAULT_SOURCE_GAME_ID,
    source_trace_path: Path | str = DEFAULT_SOURCE_TRACE_PATH,
    target_game_id: str = DEFAULT_TARGET_GAME_ID,
    target_trace_path: Path | str = DEFAULT_TARGET_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    repeated_observations: int = 2,
    min_negative_count: int = 2,
    preferred_predicates: Sequence[str] = DEFAULT_PREFERRED_PREDICATES,
) -> CrossGameTransferabilityCheckResult:
    """Learn transferability on one game and apply it to another."""
    source_path = Path(source_trace_path)
    target_path = Path(target_trace_path)
    result = CrossGameTransferabilityCheckResult(
        source_game_id=source_game_id,
        target_game_id=target_game_id,
        source_trace_path=source_path,
        target_trace_path=target_path,
    )

    source_run = run_negative_transferability_model(
        game_id=source_game_id,
        trace_path=source_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        repeated_observations=repeated_observations,
        min_negative_count=min_negative_count,
    )
    result.source_model_run = source_run
    if source_run.error:
        result.error = f"source_model_failed:{source_run.error}"
        return result

    baseline = run_non_ar25_multi_relation_agenda(
        game_id=target_game_id,
        trace_path=target_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        preferred_predicates=preferred_predicates,
    )
    result.target_baseline = baseline
    if baseline.error:
        result.error = f"target_baseline_failed:{baseline.error}"
        return result

    result.analogous_context_signatures = analogous_target_contexts(
        source_run,
        baseline.agenda_items,
    )
    if not result.analogous_context_signatures:
        result.error = "no_analogous_target_contexts"
        return result

    adapted = run_non_ar25_multi_relation_agenda(
        game_id=target_game_id,
        trace_path=target_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        preferred_predicates=preferred_predicates,
        excluded_relation_context_signatures=result.analogous_context_signatures,
    )
    result.target_adapted = adapted
    if adapted.error:
        result.error = f"target_adapted_failed:{adapted.error}"
    return result


def analogous_target_contexts(
    source_run: TransferabilityModelRunResult,
    target_agenda_items: Sequence[NonAr25RelationAgendaItem],
) -> List[str]:
    """Map source negative predicates to target contexts with same predicate."""
    downweighted_predicates = {
        group.predicate
        for group in source_run.transferability_model.groups
        if group.downweighted(
            min_negative_count=source_run.transferability_model.min_negative_count
        )
    }
    contexts: List[str] = []
    for item in target_agenda_items:
        if item.predicate not in downweighted_predicates:
            continue
        context = non_ar25_relation_context_signature(
            action=item.action,
            predicate=item.predicate,
            pair_colors=item.pair_colors,
        )
        if context not in contexts:
            contexts.append(context)
    return contexts


def _agenda_contexts(
    items: Sequence[NonAr25RelationAgendaItem],
) -> List[str]:
    contexts: List[str] = []
    for item in items:
        context = non_ar25_relation_context_signature(
            action=item.action,
            predicate=item.predicate,
            pair_colors=item.pair_colors,
        )
        if context not in contexts:
            contexts.append(context)
    return contexts


def _selected_relation_contexts(
    result: NonAr25MultiRelationAgendaResult | None,
) -> List[str]:
    if result is None:
        return []
    contexts: List[str] = []
    for prediction in result.selected_predictions_before_observation:
        if prediction.pair_colors is None:
            continue
        context = non_ar25_relation_context_signature(
            action=prediction.action,
            predicate=prediction.predicate_name,
            pair_colors=prediction.pair_colors,
        )
        if context not in contexts:
            contexts.append(context)
    return contexts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A29 cross-game transferability check."
    )
    parser.add_argument("--source-game-id", default=DEFAULT_SOURCE_GAME_ID)
    parser.add_argument("--source-trace-path", type=Path, default=DEFAULT_SOURCE_TRACE_PATH)
    parser.add_argument("--target-game-id", default=DEFAULT_TARGET_GAME_ID)
    parser.add_argument("--target-trace-path", type=Path, default=DEFAULT_TARGET_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_cross_game_transferability_check(
        source_game_id=args.source_game_id,
        source_trace_path=args.source_trace_path,
        target_game_id=args.target_game_id,
        target_trace_path=args.target_trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
