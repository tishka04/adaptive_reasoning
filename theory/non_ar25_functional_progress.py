"""A26 non-ar25 functional progress.

A25 proves that a non-ar25 relation agenda can gate an active experiment. A26
adds a separate functional-progress observation on the same live transition:
did the action move the state toward a local objective, without counting trace
support as proof or using game score as an action scorer?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from .epistemic_metrics import HypothesisStatus
from .non_ar25_multi_relation_agenda import (
    DEFAULT_GAME_ID,
    DEFAULT_TRACE_PATH,
    NonAr25MultiRelationAgendaResult,
    run_non_ar25_multi_relation_agenda,
)


@dataclass(frozen=True)
class FunctionalProgressObservation:
    """Measured non-score progress on one live transition."""

    source_color: int
    target_color: int
    levels_before: int = 0
    levels_after: int = 0
    changed_cells: int = 0
    source_count_before: int = 0
    source_count_after: int = 0
    target_count_before: int = 0
    target_count_after: int = 0
    source_to_target_pixels: int = 0

    @property
    def level_progressed(self) -> bool:
        return self.levels_after > self.levels_before

    @property
    def source_reduction(self) -> int:
        return self.source_count_before - self.source_count_after

    @property
    def target_gain(self) -> int:
        return self.target_count_after - self.target_count_before

    @property
    def source_target_gap_reduced(self) -> bool:
        return bool(self.source_to_target_pixels > 0 and self.source_reduction > 0)

    @property
    def useful_new_state(self) -> bool:
        return bool(
            self.changed_cells > 0
            and (
                self.source_target_gap_reduced
                or self.target_gain > 0
                or self.level_progressed
            )
        )

    @property
    def functional_progress(self) -> bool:
        return bool(
            self.level_progressed
            or self.source_target_gap_reduced
            or self.useful_new_state
        )

    @property
    def progress_signals(self) -> Tuple[str, ...]:
        signals: list[str] = []
        if self.level_progressed:
            signals.append("level_progressed")
        if self.source_target_gap_reduced:
            signals.append("source_target_gap_reduced")
        if self.source_to_target_pixels > 0:
            signals.append("source_to_target_pixels")
        if self.target_gain > 0:
            signals.append("target_color_gain")
        if self.useful_new_state:
            signals.append("useful_new_state")
        return tuple(signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_color": self.source_color,
            "target_color": self.target_color,
            "levels_before": self.levels_before,
            "levels_after": self.levels_after,
            "level_progressed": self.level_progressed,
            "changed_cells": self.changed_cells,
            "source_count_before": self.source_count_before,
            "source_count_after": self.source_count_after,
            "source_reduction": self.source_reduction,
            "target_count_before": self.target_count_before,
            "target_count_after": self.target_count_after,
            "target_gain": self.target_gain,
            "source_to_target_pixels": self.source_to_target_pixels,
            "source_target_gap_reduced": self.source_target_gap_reduced,
            "useful_new_state": self.useful_new_state,
            "functional_progress": self.functional_progress,
            "progress_signals": list(self.progress_signals),
        }


@dataclass
class NonAr25FunctionalProgressResult:
    """A26 result: agenda-gated active test plus functional progress."""

    game_id: str
    trace_path: Path
    agenda_result: NonAr25MultiRelationAgendaResult | None = None
    progress: FunctionalProgressObservation | None = None
    error: str = ""

    @property
    def non_ar25_multi_relation_agenda(self) -> bool:
        return bool(
            self.agenda_result
            and self.agenda_result.non_ar25_multi_relation_agenda
        )

    @property
    def active_transition_non_ar25(self) -> bool:
        return bool(
            self.game_id != "ar25-e3c63847"
            and self.agenda_result
            and self.agenda_result.active_transition_produced
        )

    @property
    def relation_observation_active(self) -> bool:
        if self.agenda_result is None:
            return False
        return any(
            revision.family == "relation"
            and revision.observed_outcome != "unobservable"
            and revision.status_after
            in {HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED}
            for revision in self.agenda_result.revisions
        )

    @property
    def active_test_called_only_when_local_agenda_observed(self) -> bool:
        return bool(
            self.agenda_result
            and self.agenda_result.validation_test_called_only_when_local_agenda_observed
        )

    @property
    def functional_progress_non_ar25(self) -> bool:
        return bool(
            self.active_transition_non_ar25
            and self.relation_observation_active
            and self.progress
            and self.progress.functional_progress
        )

    @property
    def wrong_confirmations(self) -> int:
        return 0 if self.agenda_result is None else self.agenda_result.wrong_confirmations

    @property
    def trace_support_counted_as_proof(self) -> bool:
        if self.agenda_result is None:
            return False
        return bool(
            self.agenda_result.trace_support_counted_as_proof
            or any(
                revision.trace_support_counted_as_proof
                for revision in self.agenda_result.revisions
            )
        )

    @property
    def env_actions(self) -> int:
        return 0 if self.agenda_result is None else self.agenda_result.env_actions

    @property
    def transition_count(self) -> int:
        return 0 if self.agenda_result is None else self.agenda_result.transition_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "non_ar25_multi_relation_agenda": self.non_ar25_multi_relation_agenda,
            "active_transition_non_ar25": self.active_transition_non_ar25,
            "active_test_called_only_when_local_agenda_observed": (
                self.active_test_called_only_when_local_agenda_observed
            ),
            "relation_observation_active": self.relation_observation_active,
            "functional_progress_non_ar25": self.functional_progress_non_ar25,
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "progress": self.progress.to_dict() if self.progress else None,
            "agenda_result": (
                self.agenda_result.to_dict() if self.agenda_result else None
            ),
            "error": self.error,
        }


def run_non_ar25_functional_progress(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    preferred_predicates: Sequence[str] = ("same_shape", "aligned_with", "adjacent_to"),
    excluded_relation_context_signatures: Sequence[str] = (),
    min_pair_change_pixels: int = 4,
    local_radius: int = 8,
    local_fraction_threshold: float = 0.8,
) -> NonAr25FunctionalProgressResult:
    """Run the A25 agenda and measure functional progress from its transition."""
    path = Path(trace_path)
    result = NonAr25FunctionalProgressResult(game_id=game_id, trace_path=path)
    agenda = run_non_ar25_multi_relation_agenda(
        game_id=game_id,
        trace_path=path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        preferred_predicates=preferred_predicates,
        excluded_relation_context_signatures=excluded_relation_context_signatures,
        min_pair_change_pixels=min_pair_change_pixels,
        local_radius=local_radius,
        local_fraction_threshold=local_fraction_threshold,
    )
    result.agenda_result = agenda
    if agenda.error:
        result.error = f"agenda_failed:{agenda.error}"
        return result
    if agenda.transition_update is None or agenda.experiment is None:
        result.error = "missing_active_transition"
        return result

    pair = _selected_pair(agenda)
    if pair is None:
        result.error = "missing_source_target_pair"
        return result
    result.progress = observe_functional_progress(
        agenda.transition_update,
        pair_colors=pair,
    )
    if not result.functional_progress_non_ar25:
        result.error = "no_functional_progress_observed"
    return result


def observe_functional_progress(
    transition_update: Any,
    *,
    pair_colors: Tuple[int, int],
) -> FunctionalProgressObservation:
    """Measure local source-target progress from a live transition."""
    source, target = int(pair_colors[0]), int(pair_colors[1])
    record = transition_update.record
    before = np.asarray(record.obs_before.raw_grid, dtype=np.int32)
    after = np.asarray(record.obs_after.raw_grid, dtype=np.int32)
    return FunctionalProgressObservation(
        source_color=source,
        target_color=target,
        levels_before=int(getattr(record.obs_before, "levels_completed", 0) or 0),
        levels_after=int(getattr(record.obs_after, "levels_completed", 0) or 0),
        changed_cells=int(np.sum(before != after)),
        source_count_before=int(np.sum(before == source)),
        source_count_after=int(np.sum(after == source)),
        target_count_before=int(np.sum(before == target)),
        target_count_after=int(np.sum(after == target)),
        source_to_target_pixels=int(np.sum((before == source) & (after == target))),
    )


def _selected_pair(
    agenda: NonAr25MultiRelationAgendaResult,
) -> Tuple[int, int] | None:
    if agenda.experiment is not None and agenda.experiment.predicted_pairs:
        return tuple(int(value) for value in agenda.experiment.predicted_pairs[0])
    for prediction in agenda.selected_predictions_before_observation:
        if prediction.pair_colors is not None:
            return prediction.pair_colors
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A26 non-ar25 functional progress."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_non_ar25_functional_progress(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
