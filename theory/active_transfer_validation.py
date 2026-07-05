"""A19 active validation of transferred relation priors.

The transfer prior can make a target-game relation experiment happen earlier,
but the target hypothesis remains unresolved until a real target transition is
observed. This module closes that loop for one source/target pair.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

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
    NonAr25MultiFamilyExperimentResult,
    _revise_prediction,
    run_non_ar25_multi_family_experiment,
)
from .real_env_option_adapter import snapshot_frame
from .relation_transfer import (
    RelationTransferRun,
    apply_relation_transfer_priors,
    extract_relation_transfer_priors,
    relation_predictions_from_candidates,
    run_relation_transfer,
)

DEFAULT_SOURCE_GAME_ID = "ft09-0d8bbf25"
DEFAULT_TARGET_GAME_ID = "bp35-0a0ad940"
DEFAULT_SOURCE_TRACE_PATH = Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl")
DEFAULT_TARGET_TRACE_PATH = Path("human_traces/bp35-0a0ad940.20260615-174246.steps.jsonl")


@dataclass
class ActiveTransferValidationResult:
    """Summary of one target-game active validation of transferred priors."""

    source_game_id: str
    target_game_id: str
    source_trace_path: Path
    target_trace_path: Path
    source_result: NonAr25MultiFamilyExperimentResult | None = None
    transfer: RelationTransferRun | None = None
    baseline_experiment: GenericDiscriminatingExperimentChoice | None = None
    experiment: GenericDiscriminatingExperimentChoice | None = None
    transition_update: LiveTransitionUpdate | None = None
    selected_predictions_before_observation: List[DiscriminatingPrediction] = (
        field(default_factory=list)
    )
    revisions: List[MultiFamilyHypothesisRevision] = field(default_factory=list)
    score: EpistemicScore | None = None
    env_actions: int = 0
    error: str = ""

    @property
    def transition_count(self) -> int:
        return 0 if self.transition_update is None else 1

    @property
    def transferred_prior_used(self) -> bool:
        return bool(
            self.experiment
            and self.experiment.epistemic_prior > 0.0
            and self.experiment.prior_source_keys
        )

    @property
    def transfer_changed_experiment_order(self) -> bool:
        if self.baseline_experiment is None or self.experiment is None:
            return False
        return (
            self.baseline_experiment.competing_keys != self.experiment.competing_keys
            or self.baseline_experiment.action.name != self.experiment.action.name
            or self.baseline_experiment.action.action_args
            != self.experiment.action.action_args
        )

    @property
    def active_transition_target_game(self) -> bool:
        return self.target_game_id != self.source_game_id and self.transition_count > 0

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
            "source_game_id": self.source_game_id,
            "target_game_id": self.target_game_id,
            "source_trace_path": str(self.source_trace_path),
            "target_trace_path": str(self.target_trace_path),
            "transferred_prior_used": self.transferred_prior_used,
            "transfer_changed_experiment_order": self.transfer_changed_experiment_order,
            "active_transition_target_game": self.active_transition_target_game,
            "hypothesis_remains_unresolved_until_observation": (
                self.hypothesis_remains_unresolved_until_observation
            ),
            "local_revision_after_observation": self.local_revision_after_observation,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "wrong_confirmations": self.wrong_confirmations,
            "baseline_experiment": (
                self.baseline_experiment.to_dict()
                if self.baseline_experiment
                else None
            ),
            "experiment": self.experiment.to_dict() if self.experiment else None,
            "transfer": self.transfer.to_dict() if self.transfer else None,
            "selected_predictions_before_observation": [
                _prediction_to_dict(prediction)
                for prediction in self.selected_predictions_before_observation
            ],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "score": self.score.to_dict() if self.score is not None else None,
            "error": self.error,
        }


def run_active_transfer_validation(
    *,
    source_game_id: str = DEFAULT_SOURCE_GAME_ID,
    source_trace_path: Path | str = DEFAULT_SOURCE_TRACE_PATH,
    target_game_id: str = DEFAULT_TARGET_GAME_ID,
    target_trace_path: Path | str = DEFAULT_TARGET_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    prior_weight: float = 100.0,
    min_pair_change_pixels: int = 4,
    local_radius: int = 8,
    local_fraction_threshold: float = 0.8,
) -> ActiveTransferValidationResult:
    """Validate that a transferred relation prior accelerates a real test."""
    source_path = Path(source_trace_path)
    target_path = Path(target_trace_path)
    result = ActiveTransferValidationResult(
        source_game_id=source_game_id,
        target_game_id=target_game_id,
        source_trace_path=source_path,
        target_trace_path=target_path,
    )

    source_result = run_non_ar25_multi_family_experiment(
        game_id=source_game_id,
        trace_path=source_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        preferred_family="relation",
        min_pair_change_pixels=min_pair_change_pixels,
        local_radius=local_radius,
        local_fraction_threshold=local_fraction_threshold,
    )
    result.source_result = source_result
    if source_result.error:
        result.error = f"source_relation_failed:{source_result.error}"
        return result

    discovery = discover_cross_game_correspondences(
        target_path,
        game_id=target_game_id,
        min_pixel_support=min_pixel_support,
        top_k=max_candidates,
    )
    base_predictions = relation_predictions_from_candidates(discovery.candidates)
    priors = extract_relation_transfer_priors(
        source_result.revisions,
        source_game_id=source_game_id,
        prior_weight=prior_weight,
    )
    predictions = apply_relation_transfer_priors(base_predictions, priors)
    result.transfer = run_relation_transfer(
        source_game_id=source_game_id,
        source_revisions=source_result.revisions,
        target_game_id=target_game_id,
        target_candidates=discovery.candidates,
        prior_weight=prior_weight,
    )

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(env_dir),
        )
        env = arc.make(target_game_id)
        current_frame = env.step(GameAction.RESET)
    except Exception as exc:  # pragma: no cover - integration failure path
        result.error = f"target_env_setup_failed: {exc}"
        return result

    before = snapshot_frame(current_frame)
    valid_actions = _valid_actions(env)
    designer = GenericDiscriminatingExperimentDesigner(max_competing_hypotheses=2)
    result.baseline_experiment = designer.design(
        hypotheses=base_predictions,
        live_grid=before.grid,
        available_actions=valid_actions,
        preferred_family="relation",
    )
    choice = designer.design(
        hypotheses=predictions,
        live_grid=before.grid,
        available_actions=valid_actions,
        preferred_family="relation",
    )
    if choice is None:
        result.error = "no_target_relation_experiment"
        return result
    if choice.epistemic_prior <= 0.0 or not choice.prior_source_keys:
        result.error = "target_experiment_did_not_use_transferred_prior"
        return result
    result.experiment = choice

    by_key = {prediction.key: prediction for prediction in predictions}
    selected_predictions = [
        by_key[key] for key in choice.competing_keys if key in by_key
    ]
    result.selected_predictions_before_observation = selected_predictions
    if not result.hypothesis_remains_unresolved_until_observation:
        result.error = "target_hypothesis_resolved_before_observation"
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
        result.error = f"target_env_step_failed:{selected_action.name}: {exc}"
        return result
    if after_frame is None:
        result.error = f"target_env_step_returned_none:{selected_action.name}"
        return result

    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    loop = LiveTransitionBeliefLoop(
        target_game_id,
        theory=GameTheory(target_game_id),
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
        for prediction in selected_predictions
    ]
    result.score = score_beliefs(
        result.records,
        MechanicsOracle(target_game_id),
        experiment_actions=max(1, result.env_actions),
    )
    return result


def _prediction_to_dict(prediction: DiscriminatingPrediction) -> dict[str, Any]:
    return {
        "key": prediction.key,
        "family": prediction.normalized_family,
        "predicate": prediction.predicate_name,
        "predicted_outcome": prediction.outcome,
        "status": prediction.status.value,
        "epistemic_prior": prediction.epistemic_prior,
        "prior_source_keys": list(prediction.prior_source_keys),
        "prior_counted_as_proof": prediction.prior_counted_as_proof,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A19 active validation of transferred relation priors."
    )
    parser.add_argument("--source-game-id", default=DEFAULT_SOURCE_GAME_ID)
    parser.add_argument("--source-trace-path", type=Path, default=DEFAULT_SOURCE_TRACE_PATH)
    parser.add_argument("--target-game-id", default=DEFAULT_TARGET_GAME_ID)
    parser.add_argument("--target-trace-path", type=Path, default=DEFAULT_TARGET_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=_env_dir())
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--prior-weight", type=float, default=100.0)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_active_transfer_validation(
        source_game_id=args.source_game_id,
        source_trace_path=args.source_trace_path,
        target_game_id=args.target_game_id,
        target_trace_path=args.target_trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        prior_weight=args.prior_weight,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
