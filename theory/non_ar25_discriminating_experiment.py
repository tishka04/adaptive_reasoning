"""A14 discriminating non-ar25 correspondence experiment.

This harness turns A13 from "confirm one plausible hypothesis" into a
minimal active experiment that separates two live-compatible correspondence
hypotheses. Trace support is used only to propose competitors; the verdicts
come solely from the live TransitionRecord produced by the runner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from .cross_game_correspondence_discovery import (
    DiscoveredCorrespondenceCandidate,
    discover_cross_game_correspondences,
)
from .epistemic_metrics import (
    EpistemicScore,
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    score_beliefs,
)
from .live_transition_loop import LiveTransitionBeliefLoop, LiveTransitionUpdate
from .mechanic_hypothesis import GameTheory
from .generic_discriminating_experiment_designer import (
    GenericDiscriminatingExperimentDesigner,
)
from .non_ar25_active_micro_run import (
    DEFAULT_GAME_ID,
    DEFAULT_TRACE_PATH,
    ActiveExperimentAction,
    _active_hypothesis_from_candidate,
    _configure_offline_env,
    _env_dir,
    _observed_predicates,
    _pair_change_pixels,
    _step_env_with_action,
    _valid_actions,
)
from .real_env_option_adapter import snapshot_frame


@dataclass(frozen=True)
class DiscriminatingNonAr25Experiment:
    """One active experiment with divergent predictions."""

    action: ActiveExperimentAction
    competing_keys: Tuple[str, ...]
    predicted_pairs: Tuple[Tuple[int, int], ...]
    divergence_reason: str

    @property
    def has_divergent_predictions(self) -> bool:
        return len(set(self.predicted_pairs)) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "competing_keys": list(self.competing_keys),
            "predicted_pairs": [list(pair) for pair in self.predicted_pairs],
            "divergence_reason": self.divergence_reason,
            "has_divergent_predictions": self.has_divergent_predictions,
        }


@dataclass(frozen=True)
class CompetingHypothesisRevision:
    """Live revision of one competitor after the same transition."""

    key: str
    source_color: int
    target_color: int
    relation: str = "modifies"
    status_before: HypothesisStatus = HypothesisStatus.UNRESOLVED
    status_after: HypothesisStatus = HypothesisStatus.UNRESOLVED
    changed_pixels: int = 0
    pair_change_pixels: int = 0
    observed_predicates: Tuple[str, ...] = ()
    reason: str = ""
    trace_support: int = 0
    trace_transition_support: int = 0
    trace_support_counted_as_proof: bool = False

    @property
    def pair_colors(self) -> Tuple[int, int]:
        return (self.source_color, self.target_color)

    @property
    def status_changed(self) -> bool:
        return self.status_before != self.status_after

    def to_record(self, *, experiments_spent: int) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=(
                "active discriminating non-ar25 test of "
                f"{self.source_color}->{self.target_color}"
            ),
            status=self.status_after,
            support=1 if self.status_after == HypothesisStatus.CONFIRMED else 0,
            contradictions=1 if self.status_after == HypothesisStatus.REFUTED else 0,
            experiments_spent=int(experiments_spent),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source_color": self.source_color,
            "target_color": self.target_color,
            "relation": self.relation,
            "status_before": self.status_before.value,
            "status_after": self.status_after.value,
            "status_changed": self.status_changed,
            "changed_pixels": self.changed_pixels,
            "pair_change_pixels": self.pair_change_pixels,
            "observed_predicates": list(self.observed_predicates),
            "reason": self.reason,
            "trace_support": self.trace_support,
            "trace_transition_support": self.trace_transition_support,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
        }


@dataclass
class NonAr25DiscriminatingExperimentResult:
    """Summary of an A14 active discriminating experiment."""

    game_id: str
    trace_path: Path
    experiment: DiscriminatingNonAr25Experiment | None = None
    transition_update: LiveTransitionUpdate | None = None
    revisions: List[CompetingHypothesisRevision] = field(default_factory=list)
    score: EpistemicScore | None = None
    discovered_candidates: int = 0
    env_actions: int = 0
    trace_dependent: bool = False
    trace_support_counted_as_proof: bool = False
    error: str = ""

    @property
    def competing_hypothesis_count(self) -> int:
        return len(self.revisions)

    @property
    def selected_experiment_has_divergent_predictions(self) -> bool:
        return bool(self.experiment and self.experiment.has_divergent_predictions)

    @property
    def transition_count(self) -> int:
        return 0 if self.transition_update is None else 1

    @property
    def active_transition_non_ar25(self) -> bool:
        return self.game_id != "ar25-e3c63847" and self.transition_count > 0

    @property
    def revision_differs_across_hypotheses(self) -> bool:
        statuses = {revision.status_after for revision in self.revisions}
        return len(statuses) >= 2

    @property
    def wrong_confirmations(self) -> int:
        return sum(
            1
            for revision in self.revisions
            if revision.status_after == HypothesisStatus.CONFIRMED
            and revision.pair_change_pixels <= 0
        )

    @property
    def confirmed_keys(self) -> List[str]:
        return [
            revision.key
            for revision in self.revisions
            if revision.status_after == HypothesisStatus.CONFIRMED
        ]

    @property
    def refuted_keys(self) -> List[str]:
        return [
            revision.key
            for revision in self.revisions
            if revision.status_after == HypothesisStatus.REFUTED
        ]

    @property
    def records(self) -> List[HypothesisRecord]:
        return [
            revision.to_record(experiments_spent=self.env_actions)
            for revision in self.revisions
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "experiment": self.experiment.to_dict() if self.experiment else None,
            "active_transition_non_ar25": self.active_transition_non_ar25,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "trace_dependent": self.trace_dependent,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "discovered_candidates": self.discovered_candidates,
            "competing_hypothesis_count": self.competing_hypothesis_count,
            "selected_experiment_has_divergent_predictions": (
                self.selected_experiment_has_divergent_predictions
            ),
            "revision_differs_across_hypotheses": (
                self.revision_differs_across_hypotheses
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "confirmed_keys": self.confirmed_keys,
            "refuted_keys": self.refuted_keys,
            "score": self.score.to_dict() if self.score is not None else None,
            "revisions": [revision.to_dict() for revision in self.revisions],
            "error": self.error,
        }


def run_non_ar25_discriminating_experiment(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    min_pair_change_pixels: int = 4,
    preferred_source_color: int | None = None,
) -> NonAr25DiscriminatingExperimentResult:
    """Run one active experiment that separates competing non-ar25 hypotheses."""
    path = Path(trace_path)
    result = NonAr25DiscriminatingExperimentResult(game_id=game_id, trace_path=path)

    discovery = discover_cross_game_correspondences(
        path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=max_candidates,
    )
    result.discovered_candidates = len(discovery.candidates)
    if not discovery.candidates:
        result.error = "no_unresolved_candidates"
        return result

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(env_dir),
        )
        env = arc.make(game_id)
        current_frame = env.step(GameAction.RESET)
    except Exception as exc:  # pragma: no cover - integration failure path
        result.error = f"env_setup_failed: {exc}"
        return result

    before = snapshot_frame(current_frame)
    selection = _select_discriminating_experiment(
        discovery.candidates,
        before.grid,
        env,
        preferred_source_color=preferred_source_color,
    )
    if selection is None:
        result.error = "no_live_discriminating_experiment"
        return result
    experiment, competitors = selection
    result.experiment = experiment

    theory = GameTheory(game_id)
    theory.seed_actions(before.available_actions)
    theory.add_semantic_hypotheses(
        correspondence=[
            _active_hypothesis_from_candidate(candidate) for candidate in competitors
        ]
    )
    loop = LiveTransitionBeliefLoop(
        game_id,
        theory=theory,
        available_actions=before.available_actions,
        infer_players=False,
        correspondence_pair_colors=competitors[0].pair_colors,
    )

    try:
        after_frame = _step_env_with_action(env, experiment.action)
    except Exception as exc:  # pragma: no cover - integration failure path
        result.error = f"env_step_failed:{experiment.action.name}: {exc}"
        return result
    if after_frame is None:
        result.error = f"env_step_returned_none:{experiment.action.name}"
        return result

    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    result.transition_update = loop.observe_grids(
        action=experiment.action.name,
        action_args=experiment.action.action_args,
        grid_before=before.grid,
        grid_after=after.grid,
        available_actions=before.available_actions or after.available_actions,
        game_state_before=before.game_state,
        game_state_after=after.game_state,
        levels_completed_before=before.levels_completed,
        levels_completed_after=after.levels_completed,
        timestamp=0,
        was_experiment=True,
    )
    result.env_actions = 1

    changed_pixels = int(np.sum(before.grid != after.grid))
    result.revisions = [
        _revise_competitor(
            before.grid,
            after.grid,
            candidate,
            changed_pixels=changed_pixels,
            min_pair_change_pixels=min_pair_change_pixels,
        )
        for candidate in competitors
    ]
    result.score = score_beliefs(
        result.records,
        MechanicsOracle(game_id),
        experiment_actions=max(1, result.env_actions),
    )
    return result


def _select_discriminating_experiment(
    candidates: Sequence[DiscoveredCorrespondenceCandidate],
    grid: np.ndarray,
    env: Any,
    *,
    preferred_source_color: int | None,
) -> tuple[
    DiscriminatingNonAr25Experiment,
    Tuple[DiscoveredCorrespondenceCandidate, ...],
] | None:
    designer = GenericDiscriminatingExperimentDesigner(max_competing_hypotheses=2)
    choice = designer.design(
        hypotheses=candidates,
        live_grid=grid,
        available_actions=_valid_actions(env),
        preferred_source_color=preferred_source_color,
    )
    if choice is None:
        return None

    by_key = {candidate.key: candidate for candidate in candidates}
    competitors = tuple(
        by_key[key] for key in choice.competing_keys if key in by_key
    )
    if len(competitors) < 2:
        return None

    experiment_action = ActiveExperimentAction(
        name=choice.action.name,
        raw_action=choice.action.raw_action,
        action_args=dict(choice.action.action_args),
        selection_reason=choice.selection_reason,
    )
    experiment = DiscriminatingNonAr25Experiment(
        action=experiment_action,
        competing_keys=tuple(candidate.key for candidate in competitors),
        predicted_pairs=tuple(candidate.pair_colors for candidate in competitors),
        divergence_reason=choice.divergence_reason,
    )
    return experiment, competitors


def _revise_competitor(
    before: np.ndarray,
    after: np.ndarray,
    candidate: DiscoveredCorrespondenceCandidate,
    *,
    changed_pixels: int,
    min_pair_change_pixels: int,
) -> CompetingHypothesisRevision:
    pair_change_pixels = _pair_change_pixels(before, after, candidate.pair_colors)
    if pair_change_pixels >= min_pair_change_pixels:
        status_after = HypothesisStatus.CONFIRMED
        reason = "observed_predicted_pair"
    elif changed_pixels > 0:
        status_after = HypothesisStatus.REFUTED
        reason = "observed_divergent_pair"
    else:
        status_after = HypothesisStatus.UNRESOLVED
        reason = "no_observable_change"
    predicates = _observed_predicates(
        before,
        after,
        candidate,
        pair_change_pixels=pair_change_pixels,
        min_pair_change_pixels=min_pair_change_pixels,
    )
    return CompetingHypothesisRevision(
        key=candidate.key,
        source_color=candidate.source_color,
        target_color=candidate.target_color,
        relation=candidate.relation,
        status_after=status_after,
        changed_pixels=changed_pixels,
        pair_change_pixels=pair_change_pixels,
        observed_predicates=tuple(predicates),
        reason=reason,
        trace_support=candidate.support,
        trace_transition_support=candidate.transition_support,
        trace_support_counted_as_proof=False,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A14 active non-ar25 discriminating experiment."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=_env_dir())
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--preferred-source-color", type=int, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_non_ar25_discriminating_experiment(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        preferred_source_color=args.preferred_source_color,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
