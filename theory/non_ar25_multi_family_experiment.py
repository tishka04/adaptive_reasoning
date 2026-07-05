"""A16/A17 active non-ar25 multi-family prediction experiment.

This module demonstrates that the generic designer can discriminate hypotheses
outside the source-color -> target-color family, including relation predicates.
It still runs only one real environment transition and still revises from
observation, not trace support or game score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from .cross_game_correspondence_discovery import (
    _adjacent_exists,
    _aligned_exists,
    _same_shape_exists,
    discover_cross_game_correspondences,
)
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
    DEFAULT_GAME_ID,
    DEFAULT_TRACE_PATH,
    ActiveExperimentAction,
    _configure_offline_env,
    _env_dir,
    _pair_change_pixels,
    _step_env_with_action,
    _valid_actions,
)
from .real_env_option_adapter import snapshot_frame


@dataclass(frozen=True)
class MultiFamilyHypothesisRevision:
    """Live revision of one prediction-family hypothesis."""

    key: str
    family: str
    predicate: str
    predicted_outcome: str
    observed_outcome: str
    status_before: HypothesisStatus = HypothesisStatus.UNRESOLVED
    status_after: HypothesisStatus = HypothesisStatus.UNRESOLVED
    reason: str = ""
    trace_support: int = 0
    trace_transition_support: int = 0
    trace_support_counted_as_proof: bool = False

    @property
    def status_changed(self) -> bool:
        return self.status_before != self.status_after

    def to_record(self, *, experiments_spent: int) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=(
                f"active multi-family test of {self.family}: "
                f"{self.predicate} -> {self.predicted_outcome}"
            ),
            status=self.status_after,
            support=1 if self.status_after == HypothesisStatus.CONFIRMED else 0,
            contradictions=1 if self.status_after == HypothesisStatus.REFUTED else 0,
            experiments_spent=int(experiments_spent),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "predicate": self.predicate,
            "predicted_outcome": self.predicted_outcome,
            "observed_outcome": self.observed_outcome,
            "status_before": self.status_before.value,
            "status_after": self.status_after.value,
            "status_changed": self.status_changed,
            "reason": self.reason,
            "trace_support": self.trace_support,
            "trace_transition_support": self.trace_transition_support,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
        }


@dataclass
class NonAr25MultiFamilyExperimentResult:
    """Summary of one A16 active multi-family experiment."""

    game_id: str
    trace_path: Path
    experiment: GenericDiscriminatingExperimentChoice | None = None
    transition_update: LiveTransitionUpdate | None = None
    revisions: List[MultiFamilyHypothesisRevision] = field(default_factory=list)
    supported_prediction_families: List[str] = field(default_factory=list)
    score: EpistemicScore | None = None
    discovered_candidates: int = 0
    env_actions: int = 0
    trace_dependent: bool = False
    trace_support_counted_as_proof: bool = False
    error: str = ""

    @property
    def transition_count(self) -> int:
        return 0 if self.transition_update is None else 1

    @property
    def active_transition_non_ar25(self) -> bool:
        return self.game_id != "ar25-e3c63847" and self.transition_count > 0

    @property
    def competing_hypothesis_count(self) -> int:
        return len(self.revisions)

    @property
    def selected_experiment_has_divergent_predictions(self) -> bool:
        return bool(self.experiment and self.experiment.has_divergent_predictions)

    @property
    def revision_differs_across_hypotheses(self) -> bool:
        return len({revision.status_after for revision in self.revisions}) >= 2

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
            "experiment": self.experiment.to_dict() if self.experiment else None,
            "active_transition_non_ar25": self.active_transition_non_ar25,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "trace_dependent": self.trace_dependent,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "supported_prediction_families": list(self.supported_prediction_families),
            "discovered_candidates": self.discovered_candidates,
            "competing_hypothesis_count": self.competing_hypothesis_count,
            "selected_experiment_has_divergent_predictions": (
                self.selected_experiment_has_divergent_predictions
            ),
            "revision_differs_across_hypotheses": (
                self.revision_differs_across_hypotheses
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "score": self.score.to_dict() if self.score is not None else None,
            "revisions": [revision.to_dict() for revision in self.revisions],
            "error": self.error,
        }


def run_non_ar25_multi_family_experiment(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    preferred_family: str | None = "effect_scope",
    preferred_source_color: int | None = None,
    local_radius: int = 8,
    local_fraction_threshold: float = 0.8,
    min_pair_change_pixels: int = 4,
) -> NonAr25MultiFamilyExperimentResult:
    """Run one real non-ar25 experiment over multi-family predictions."""
    path = Path(trace_path)
    result = NonAr25MultiFamilyExperimentResult(game_id=game_id, trace_path=path)

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
    predictions = _build_multi_family_predictions(
        discovery.candidates,
        before.grid,
        valid_actions,
    )
    result.supported_prediction_families = _dedupe(
        prediction.normalized_family for prediction in predictions
    )

    choice = GenericDiscriminatingExperimentDesigner(max_competing_hypotheses=2).design(
        hypotheses=predictions,
        live_grid=before.grid,
        available_actions=valid_actions,
        preferred_source_color=preferred_source_color,
        preferred_family=preferred_family,
    )
    if choice is None:
        result.error = "no_live_multi_family_discriminating_experiment"
        return result
    result.experiment = choice

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

    by_key = {prediction.key: prediction for prediction in predictions}
    selected_predictions = [
        by_key[key] for key in choice.competing_keys if key in by_key
    ]
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
        MechanicsOracle(game_id),
        experiment_actions=max(1, result.env_actions),
    )
    return result


def _build_multi_family_predictions(
    candidates: Sequence[Any],
    grid: np.ndarray,
    valid_actions: Sequence[Any],
) -> List[DiscriminatingPrediction]:
    predictions = [
        DiscriminatingPrediction.from_hypothesis(candidate)
        for candidate in candidates
    ]
    predictions.extend(_effect_scope_predictions(grid, valid_actions))
    predictions.extend(_object_count_predictions(grid, valid_actions))
    predictions.extend(_relation_predictions(candidates, grid, valid_actions))
    return _dedupe_predictions(predictions)


def _effect_scope_predictions(
    grid: np.ndarray,
    valid_actions: Sequence[Any],
) -> List[DiscriminatingPrediction]:
    predictions: List[DiscriminatingPrediction] = []
    for action in valid_actions:
        source = _action_pixel_color(grid, getattr(action, "action_args", {}))
        if source is None:
            continue
        for outcome in ("local", "global"):
            predictions.append(
                DiscriminatingPrediction(
                    key=f"effect_scope::{action.name}::source{source}::{outcome}",
                    action=action.name,
                    source_color=source,
                    family="effect_scope",
                    predicate="effect_scope",
                    predicted_outcome=outcome,
                )
            )
    return predictions


def _object_count_predictions(
    grid: np.ndarray,
    valid_actions: Sequence[Any],
) -> List[DiscriminatingPrediction]:
    predictions: List[DiscriminatingPrediction] = []
    for action in valid_actions:
        source = _action_pixel_color(grid, getattr(action, "action_args", {}))
        if source is None:
            continue
        for outcome in ("changed", "stable"):
            predictions.append(
                DiscriminatingPrediction(
                    key=f"object_count::{action.name}::source{source}::{outcome}",
                    action=action.name,
                    source_color=source,
                    family="object_count",
                    predicate="object_count",
                    predicted_outcome=outcome,
                )
            )
    return predictions


def _relation_predictions(
    candidates: Sequence[Any],
    grid: np.ndarray,
    valid_actions: Sequence[Any],
) -> List[DiscriminatingPrediction]:
    predictions: List[DiscriminatingPrediction] = []
    relation_predicates = ("same_shape", "adjacent_to", "aligned_with", "paired_with")
    for action in valid_actions:
        source = _action_pixel_color(grid, getattr(action, "action_args", {}))
        if source is None:
            continue
        for candidate in candidates:
            if getattr(candidate, "action", "") != action.name:
                continue
            if int(getattr(candidate, "source_color", -1)) != int(source):
                continue
            target = getattr(candidate, "target_color", None)
            if target is None:
                continue
            candidate_predicates = {
                getattr(predicate, "name", ""): int(getattr(predicate, "support", 0) or 0)
                for predicate in getattr(candidate, "predicates", ())
            }
            for predicate in relation_predicates:
                if predicate not in candidate_predicates:
                    continue
                base_support = candidate_predicates[predicate]
                before_holds = _relation_holds_for_colors(
                    grid,
                    predicate,
                    int(source),
                    int(target),
                )
                outcomes = ("preserved", "broken") if before_holds else (
                    "appears",
                    "absent",
                )
                for outcome in outcomes:
                    predictions.append(
                        DiscriminatingPrediction(
                            key=(
                                f"relation::{action.name}::{predicate}::"
                                f"colors{int(source)}_{int(target)}::{outcome}"
                            ),
                            action=action.name,
                            source_color=int(source),
                            target_color=int(target),
                            family="relation",
                            predicate=predicate,
                            predicted_outcome=outcome,
                            support=base_support,
                            transition_support=int(
                                getattr(candidate, "transition_support", 0) or 0
                            ),
                        )
                    )
    return predictions


def _revise_prediction(
    update: LiveTransitionUpdate,
    prediction: DiscriminatingPrediction,
    *,
    min_pair_change_pixels: int,
    local_radius: int,
    local_fraction_threshold: float,
) -> MultiFamilyHypothesisRevision:
    observed = _observe_prediction_outcome(
        update,
        prediction,
        min_pair_change_pixels=min_pair_change_pixels,
        local_radius=local_radius,
        local_fraction_threshold=local_fraction_threshold,
    )
    if observed == prediction.outcome:
        status_after = HypothesisStatus.CONFIRMED
        reason = f"observed_predicted_{prediction.normalized_family}"
    elif observed == "unobservable":
        status_after = HypothesisStatus.UNRESOLVED
        reason = "predicate_unobservable"
    else:
        status_after = HypothesisStatus.REFUTED
        reason = f"observed_divergent_{prediction.normalized_family}"
    return MultiFamilyHypothesisRevision(
        key=prediction.key,
        family=prediction.normalized_family,
        predicate=prediction.predicate_name,
        predicted_outcome=prediction.outcome,
        observed_outcome=observed,
        status_after=status_after,
        reason=reason,
        trace_support=prediction.support,
        trace_transition_support=prediction.transition_support,
        trace_support_counted_as_proof=False,
    )


def _observe_prediction_outcome(
    update: LiveTransitionUpdate,
    prediction: DiscriminatingPrediction,
    *,
    min_pair_change_pixels: int,
    local_radius: int,
    local_fraction_threshold: float,
) -> str:
    record = update.record
    before = record.obs_before.raw_grid
    after = record.obs_after.raw_grid
    family = prediction.normalized_family
    if family == "color_transform":
        return _observe_color_transform(
            before,
            after,
            prediction,
            min_pair_change_pixels=min_pair_change_pixels,
        )
    if family == "effect_scope":
        return _observe_effect_scope(
            update,
            local_radius=local_radius,
            local_fraction_threshold=local_fraction_threshold,
        )
    if family == "object_count":
        return _observe_object_count(update)
    if family == "relation":
        return _observe_relation(before, after, prediction)
    return "unobservable"


def _observe_color_transform(
    before: np.ndarray,
    after: np.ndarray,
    prediction: DiscriminatingPrediction,
    *,
    min_pair_change_pixels: int,
) -> str:
    if prediction.target_color is None:
        return "unobservable"
    if (
        _pair_change_pixels(before, after, (prediction.source_color, prediction.target_color))
        >= min_pair_change_pixels
    ):
        return prediction.outcome
    mask = (before != after) & (before == int(prediction.source_color))
    if not bool(mask.any()):
        return "none"
    values, counts = np.unique(after[mask], return_counts=True)
    target = int(values[int(np.argmax(counts))])
    return f"{prediction.source_color}->{target}"


def _observe_effect_scope(
    update: LiveTransitionUpdate,
    *,
    local_radius: int,
    local_fraction_threshold: float,
) -> str:
    record = update.record
    changed = record.diff.changed_cells
    if not changed:
        return "none"
    if record.action.x is None or record.action.y is None:
        return "global" if record.diff.num_changed >= 10 else "local"
    x = int(record.action.x)
    y = int(record.action.y)
    local = 0
    for row, col in changed:
        if max(abs(int(row) - y), abs(int(col) - x)) <= int(local_radius):
            local += 1
    fraction = local / max(1, len(changed))
    if fraction >= float(local_fraction_threshold):
        return "local"
    return "global"


def _observe_object_count(update: LiveTransitionUpdate) -> str:
    before_count = len(update.record.obs_before.objects)
    after_count = len(update.record.obs_after.objects)
    if (
        before_count != after_count
        or update.record.diff.created_objects
        or update.record.diff.removed_objects
    ):
        return "changed"
    return "stable"


def _observe_relation(
    before: np.ndarray,
    after: np.ndarray,
    prediction: DiscriminatingPrediction,
) -> str:
    if prediction.target_color is None:
        return "unobservable"
    before_holds = _relation_holds(before, prediction)
    after_holds = _relation_holds(after, prediction)
    if not before_holds and after_holds:
        return "appears"
    if before_holds and not after_holds:
        return "broken"
    if before_holds and after_holds:
        return "preserved"
    return "absent"


def _relation_holds(grid: np.ndarray, prediction: DiscriminatingPrediction) -> bool:
    if prediction.target_color is None:
        return False
    return _relation_holds_for_colors(
        grid,
        prediction.predicate_name,
        prediction.source_color,
        int(prediction.target_color),
    )


def _relation_holds_for_colors(
    grid: np.ndarray,
    predicate: str,
    source_color: int,
    target_color: int,
) -> bool:
    predicate = str(predicate)
    if predicate == "same_shape":
        return _same_shape_exists(
            grid,
            int(source_color),
            int(target_color),
        )
    if predicate == "aligned_with":
        return _aligned_exists(
            grid,
            int(source_color),
            int(target_color),
        )
    if predicate == "adjacent_to":
        return _adjacent_exists(
            grid,
            int(source_color),
            int(target_color),
        )
    if predicate == "paired_with":
        return bool(
            np.any(grid == int(source_color))
            and np.any(grid == int(target_color))
        )
    return False


def _action_pixel_color(
    grid: np.ndarray,
    action_args: dict[str, Any],
) -> int | None:
    if "x" not in action_args or "y" not in action_args:
        return None
    try:
        x = int(action_args["x"])
        y = int(action_args["y"])
    except (TypeError, ValueError):
        return None
    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
        return None
    return int(grid[y, x])


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A16 active non-ar25 multi-family experiment."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=_env_dir())
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--preferred-family", default="effect_scope")
    parser.add_argument("--preferred-source-color", type=int, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_non_ar25_multi_family_experiment(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        preferred_family=args.preferred_family or None,
        preferred_source_color=args.preferred_source_color,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
