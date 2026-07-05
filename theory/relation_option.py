"""A22 generic relation options.

Relation options promote unresolved relation predicates into explicit
prepare/avoid options. They still delegate experiment choice to the generic
discriminating designer and revision to the live transition observer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from .cross_game_correspondence_discovery import discover_cross_game_correspondences
from .epistemic_metrics import (
    EpistemicScore,
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    score_beliefs,
)
from .generic_discriminating_experiment_designer import (
    DiscriminatingPrediction,
    GenericDiscriminatingExperimentChoice,
    GenericDiscriminatingExperimentDesigner,
)
from .live_transition_loop import LiveTransitionBeliefLoop, LiveTransitionUpdate
from .mechanic_hypothesis import GameTheory
from .non_ar25_active_micro_run import (
    ActiveExperimentAction,
    _configure_offline_env,
    _env_dir,
    _step_env_with_action,
    _valid_actions,
)
from .non_ar25_multi_family_experiment import (
    MultiFamilyHypothesisRevision,
    _relation_predictions,
    _revise_prediction,
)
from .real_env_option_adapter import snapshot_frame

DEFAULT_RELATION_GAME_ID = "ft09-0d8bbf25"
DEFAULT_RELATION_TRACE_PATH = Path(
    "human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"
)


@dataclass(frozen=True)
class RelationOption:
    """A generic option that prepares or avoids one relation outcome."""

    name: str
    predicate: str
    desired_outcome: str
    mode: str = "prepare"
    target_pair_colors: Tuple[int, int] | None = None
    max_steps: int = 1

    @property
    def termination_outcomes(self) -> Tuple[str, ...]:
        return (self.desired_outcome,)

    def matching_predictions(
        self,
        predictions: Sequence[DiscriminatingPrediction],
    ) -> List[DiscriminatingPrediction]:
        """Return unresolved relation predictions relevant to this option."""
        candidates = [
            prediction
            for prediction in predictions
            if prediction.status == HypothesisStatus.UNRESOLVED
            and prediction.normalized_family == "relation"
            and prediction.predicate_name == self.predicate
            and self._pair_matches(prediction)
        ]
        desired_groups = {
            prediction.divergence_group
            for prediction in candidates
            if prediction.outcome == self.desired_outcome
        }
        return [
            prediction
            for prediction in candidates
            if prediction.divergence_group in desired_groups
        ]

    def can_initiate(
        self,
        predictions: Sequence[DiscriminatingPrediction],
        *,
        relation_needed: bool = True,
    ) -> bool:
        """Initiation depends on a needed unresolved relation hypothesis."""
        return bool(relation_needed and self.matching_predictions(predictions))

    def observe_termination(
        self,
        revisions: Sequence[MultiFamilyHypothesisRevision],
    ) -> str:
        """Classify termination using observed relation outcomes only."""
        if not revisions:
            return "not_observed"
        if any(revision.observed_outcome in self.termination_outcomes for revision in revisions):
            return "success"
        if any(
            revision.status_after in {HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED}
            for revision in revisions
        ):
            return "contradiction"
        return "unresolved"

    def _pair_matches(self, prediction: DiscriminatingPrediction) -> bool:
        if self.target_pair_colors is None:
            return True
        return prediction.pair_colors == self.target_pair_colors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "predicate": self.predicate,
            "desired_outcome": self.desired_outcome,
            "target_pair_colors": (
                list(self.target_pair_colors)
                if self.target_pair_colors is not None
                else None
            ),
            "termination_outcomes": list(self.termination_outcomes),
            "max_steps": self.max_steps,
        }


class PrepareRelationOption(RelationOption):
    """Option that tries to make or preserve a relation."""

    def __init__(
        self,
        predicate: str,
        *,
        desired_outcome: str = "preserved",
        target_pair_colors: Tuple[int, int] | None = None,
        name: str | None = None,
        max_steps: int = 1,
    ) -> None:
        super().__init__(
            name=name or f"prepare_relation_{predicate}_{desired_outcome}",
            predicate=predicate,
            desired_outcome=desired_outcome,
            mode="prepare",
            target_pair_colors=target_pair_colors,
            max_steps=max_steps,
        )


class AvoidRelationOption(RelationOption):
    """Option that tries to keep a relation absent or broken."""

    def __init__(
        self,
        predicate: str,
        *,
        desired_outcome: str = "absent",
        target_pair_colors: Tuple[int, int] | None = None,
        name: str | None = None,
        max_steps: int = 1,
    ) -> None:
        super().__init__(
            name=name or f"avoid_relation_{predicate}_{desired_outcome}",
            predicate=predicate,
            desired_outcome=desired_outcome,
            mode="avoid",
            target_pair_colors=target_pair_colors,
            max_steps=max_steps,
        )


@dataclass
class RelationOptionRunResult:
    """One active execution of a generic relation option."""

    game_id: str
    trace_path: Path
    option: RelationOption
    initiation_holds: bool = False
    experiment: GenericDiscriminatingExperimentChoice | None = None
    transition_update: LiveTransitionUpdate | None = None
    selected_predictions_before_observation: List[DiscriminatingPrediction] = (
        field(default_factory=list)
    )
    revisions: List[MultiFamilyHypothesisRevision] = field(default_factory=list)
    termination_status: str = "not_started"
    termination_outcome: str = ""
    score: EpistemicScore | None = None
    discovered_candidates: int = 0
    env_actions: int = 0
    trace_support_counted_as_proof: bool = False
    error: str = ""

    @property
    def transition_count(self) -> int:
        return 0 if self.transition_update is None else 1

    @property
    def policy_action_chosen(self) -> bool:
        return self.experiment is not None

    @property
    def hypothesis_remains_unresolved_until_observation(self) -> bool:
        return bool(self.selected_predictions_before_observation) and all(
            prediction.status == HypothesisStatus.UNRESOLVED
            and not prediction.prior_counted_as_proof
            for prediction in self.selected_predictions_before_observation
        )

    @property
    def local_revision_after_observation(self) -> bool:
        return bool(self.revisions) and any(
            revision.status_after
            in {HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED}
            for revision in self.revisions
        )

    @property
    def wrong_confirmations(self) -> int:
        return sum(
            1
            for revision in self.revisions
            if revision.status_after == HypothesisStatus.CONFIRMED
            and revision.predicted_outcome != revision.observed_outcome
        )

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
            "option": self.option.to_dict(),
            "initiation_holds": self.initiation_holds,
            "policy_action_chosen": self.policy_action_chosen,
            "termination_status": self.termination_status,
            "termination_outcome": self.termination_outcome,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "hypothesis_remains_unresolved_until_observation": (
                self.hypothesis_remains_unresolved_until_observation
            ),
            "local_revision_after_observation": self.local_revision_after_observation,
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "discovered_candidates": self.discovered_candidates,
            "experiment": self.experiment.to_dict() if self.experiment else None,
            "selected_predictions_before_observation": [
                _prediction_to_dict(prediction)
                for prediction in self.selected_predictions_before_observation
            ],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "score": self.score.to_dict() if self.score is not None else None,
            "error": self.error,
        }


def run_relation_option_micro_run(
    *,
    option: RelationOption | None = None,
    game_id: str = DEFAULT_RELATION_GAME_ID,
    trace_path: Path | str = DEFAULT_RELATION_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    relation_needed: bool = True,
    min_pair_change_pixels: int = 4,
    local_radius: int = 8,
    local_fraction_threshold: float = 0.8,
) -> RelationOptionRunResult:
    """Run one active prepare/avoid relation option against a live env."""
    selected_option = option or PrepareRelationOption("same_shape")
    path = Path(trace_path)
    result = RelationOptionRunResult(
        game_id=game_id,
        trace_path=path,
        option=selected_option,
    )

    discovery = discover_cross_game_correspondences(
        path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=max_candidates,
    )
    result.discovered_candidates = len(discovery.candidates)

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
    valid_actions = _valid_actions(env)
    relation_predictions = _relation_predictions(
        discovery.candidates,
        before.grid,
        valid_actions,
    )
    option_predictions = selected_option.matching_predictions(relation_predictions)
    result.initiation_holds = selected_option.can_initiate(
        option_predictions,
        relation_needed=relation_needed,
    )
    if not result.initiation_holds:
        result.error = "relation_option_initiation_false"
        return result

    choice = GenericDiscriminatingExperimentDesigner(
        max_competing_hypotheses=2
    ).design(
        hypotheses=option_predictions,
        live_grid=before.grid,
        available_actions=valid_actions,
        preferred_family="relation",
    )
    if choice is None:
        result.error = "no_relation_option_experiment"
        return result
    result.experiment = choice

    by_key = {prediction.key: prediction for prediction in option_predictions}
    result.selected_predictions_before_observation = [
        by_key[key] for key in choice.competing_keys if key in by_key
    ]
    if not result.hypothesis_remains_unresolved_until_observation:
        result.error = "relation_option_hypothesis_resolved_before_observation"
        return result

    selected_action = ActiveExperimentAction(
        name=choice.action.name,
        raw_action=choice.action.raw_action,
        action_args=dict(choice.action.action_args),
        selection_reason=choice.selection_reason,
    )
    try:
        after_frame = _step_env_with_action(env, selected_action)
    except Exception as exc:  # pragma: no cover - integration failure path
        result.error = f"env_step_failed:{selected_action.name}: {exc}"
        return result
    if after_frame is None:
        result.error = f"env_step_returned_none:{selected_action.name}"
        return result

    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    loop = LiveTransitionBeliefLoop(
        game_id,
        theory=GameTheory(game_id),
        available_actions=before.available_actions,
        infer_players=False,
    )
    result.transition_update = loop.observe_grids(
        action=selected_action.name,
        action_args=selected_action.action_args,
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
    result.revisions = [
        _revise_prediction(
            result.transition_update,
            prediction,
            min_pair_change_pixels=min_pair_change_pixels,
            local_radius=local_radius,
            local_fraction_threshold=local_fraction_threshold,
        )
        for prediction in result.selected_predictions_before_observation
    ]
    result.termination_status = selected_option.observe_termination(result.revisions)
    result.termination_outcome = _dominant_observed_outcome(result.revisions)
    result.score = score_beliefs(
        result.records,
        MechanicsOracle(game_id),
        experiment_actions=max(1, result.env_actions),
    )
    return result


def _dominant_observed_outcome(
    revisions: Sequence[MultiFamilyHypothesisRevision],
) -> str:
    for revision in revisions:
        if revision.status_after == HypothesisStatus.CONFIRMED:
            return revision.observed_outcome
    if revisions:
        return revisions[0].observed_outcome
    return ""


def _prediction_to_dict(prediction: DiscriminatingPrediction) -> dict[str, Any]:
    return {
        "key": prediction.key,
        "family": prediction.normalized_family,
        "predicate": prediction.predicate_name,
        "predicted_outcome": prediction.outcome,
        "pair_colors": list(prediction.pair_colors) if prediction.pair_colors else None,
        "status": prediction.status.value,
        "support": prediction.support,
        "epistemic_prior": prediction.epistemic_prior,
        "prior_source_keys": list(prediction.prior_source_keys),
        "prior_counted_as_proof": prediction.prior_counted_as_proof,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A22 generic relation option micro-run."
    )
    parser.add_argument("--game-id", default=DEFAULT_RELATION_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_RELATION_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=_env_dir())
    parser.add_argument("--predicate", default="same_shape")
    parser.add_argument("--desired-outcome", default="preserved")
    parser.add_argument("--mode", choices=("prepare", "avoid"), default="prepare")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    option_cls = AvoidRelationOption if args.mode == "avoid" else PrepareRelationOption
    result = run_relation_option_micro_run(
        option=option_cls(
            args.predicate,
            desired_outcome=args.desired_outcome,
        ),
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
