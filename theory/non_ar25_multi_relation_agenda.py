"""A25 non-ar25 multi-relation agenda.

This module ports the A24 agenda idea outside ar25. Relation candidates come
from cross-game discovery and are grounded in the current non-ar25 live grid.
The final active relation test is executed only after the local agenda's
required relation predicates have been observed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

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
from .multi_relation_option_composition import missing_relation_preconditions
from .non_ar25_active_micro_run import (
    ActiveExperimentAction,
    _configure_offline_env,
    _env_dir,
    _step_env_with_action,
    _valid_actions,
)
from .non_ar25_multi_family_experiment import (
    MultiFamilyHypothesisRevision,
    _relation_holds_for_colors,
    _relation_predictions,
    _revise_prediction,
)
from .real_env_option_adapter import snapshot_frame
from .relation_option import AvoidRelationOption, PrepareRelationOption, RelationOption

DEFAULT_GAME_ID = "ft09-0d8bbf25"
DEFAULT_TRACE_PATH = Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl")
DEFAULT_PREFERRED_PREDICATES = ("same_shape", "aligned_with", "adjacent_to")
PredicateGenerator = Callable[..., Iterable[str]]
AnchorExpander = Callable[..., Iterable[Any]]
CandidateRanker = Callable[..., Sequence[Any]]


@dataclass(frozen=True)
class NonAr25RelationAgendaItem:
    """One non-ar25 relation option required before the active test."""

    option: RelationOption
    required_predicate: str
    predicate: str
    pair_colors: Tuple[int, int]
    action: str
    observed_initially: bool
    support: int = 0

    def observed_in(self, predicates: Iterable[str]) -> bool:
        present = {str(predicate) for predicate in predicates}
        return self.required_predicate in present

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option.to_dict(),
            "required_predicate": self.required_predicate,
            "predicate": self.predicate,
            "pair_colors": list(self.pair_colors),
            "action": self.action,
            "observed_initially": self.observed_initially,
            "support": self.support,
        }


@dataclass
class NonAr25MultiRelationAgendaResult:
    """Result of one non-ar25 relation agenda and active test."""

    game_id: str
    trace_path: Path
    agenda_items: List[NonAr25RelationAgendaItem] = field(default_factory=list)
    missing_preconditions: List[str] = field(default_factory=list)
    selected_option_order: List[str] = field(default_factory=list)
    observed_agenda_predicates: List[str] = field(default_factory=list)
    experiment: GenericDiscriminatingExperimentChoice | None = None
    transition_update: LiveTransitionUpdate | None = None
    selected_predictions_before_observation: List[DiscriminatingPrediction] = (
        field(default_factory=list)
    )
    revisions: List[MultiFamilyHypothesisRevision] = field(default_factory=list)
    score: EpistemicScore | None = None
    discovered_candidates: int = 0
    raw_discovered_candidates: int = 0
    candidate_prediction_count: int = 0
    env_actions: int = 0
    selection_reason: str = "missing_preconditions"
    trace_support_counted_as_proof: bool = False
    excluded_relation_context_signatures: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def non_ar25_multi_relation_agenda(self) -> bool:
        return self.game_id != "ar25-e3c63847" and self.relation_candidate_count >= 2

    @property
    def relation_candidate_count(self) -> int:
        return len(self.agenda_items)

    @property
    def includes_prepare_and_avoid_options(self) -> bool:
        modes = {item.option.mode for item in self.agenda_items}
        return "prepare" in modes and "avoid" in modes

    @property
    def order_chosen_by_missing_preconditions(self) -> bool:
        expected = [
            item.option.name
            for item in self.agenda_items
            if item.required_predicate in self.missing_preconditions
        ]
        return bool(
            self.selection_reason == "missing_preconditions"
            and expected == self.selected_option_order
        )

    @property
    def local_agenda_observed(self) -> bool:
        observed = set(self.observed_agenda_predicates)
        return bool(self.agenda_items) and all(
            item.required_predicate in observed for item in self.agenda_items
        )

    @property
    def validation_test_called(self) -> bool:
        return self.experiment is not None

    @property
    def validation_test_called_only_when_local_agenda_observed(self) -> bool:
        return bool(self.validation_test_called and self.local_agenda_observed)

    @property
    def active_transition_produced(self) -> bool:
        return self.transition_count > 0

    @property
    def transition_count(self) -> int:
        return 0 if self.transition_update is None else 1

    @property
    def hypothesis_remains_unresolved_until_observation(self) -> bool:
        return bool(self.selected_predictions_before_observation) and all(
            prediction.status == HypothesisStatus.UNRESOLVED
            and not prediction.prior_counted_as_proof
            for prediction in self.selected_predictions_before_observation
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
            "non_ar25_multi_relation_agenda": self.non_ar25_multi_relation_agenda,
            "relation_candidate_count": self.relation_candidate_count,
            "includes_prepare_and_avoid_options": self.includes_prepare_and_avoid_options,
            "selection_reason": self.selection_reason,
            "order_chosen_by_missing_preconditions": (
                self.order_chosen_by_missing_preconditions
            ),
            "missing_preconditions": list(self.missing_preconditions),
            "selected_option_order": list(self.selected_option_order),
            "observed_agenda_predicates": list(self.observed_agenda_predicates),
            "local_agenda_observed": self.local_agenda_observed,
            "validation_test_called": self.validation_test_called,
            "validation_test_called_only_when_local_agenda_observed": (
                self.validation_test_called_only_when_local_agenda_observed
            ),
            "active_transition_produced": self.active_transition_produced,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "hypothesis_remains_unresolved_until_observation": (
                self.hypothesis_remains_unresolved_until_observation
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "excluded_relation_context_signatures": (
                list(self.excluded_relation_context_signatures)
            ),
            "discovered_candidates": self.discovered_candidates,
            "raw_discovered_candidates": self.raw_discovered_candidates,
            "candidate_prediction_count": self.candidate_prediction_count,
            "agenda_items": [item.to_dict() for item in self.agenda_items],
            "experiment": self.experiment.to_dict() if self.experiment else None,
            "selected_predictions_before_observation": [
                _prediction_to_dict(prediction)
                for prediction in self.selected_predictions_before_observation
            ],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "score": self.score.to_dict() if self.score is not None else None,
            "error": self.error,
        }


def run_non_ar25_multi_relation_agenda(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    preferred_predicates: Sequence[str] = DEFAULT_PREFERRED_PREDICATES,
    excluded_relation_context_signatures: Sequence[str] = (),
    min_pair_change_pixels: int = 4,
    local_radius: int = 8,
    local_fraction_threshold: float = 0.8,
    predicate_generator: PredicateGenerator | None = None,
    anchor_expander: AnchorExpander | None = None,
    candidate_ranker: CandidateRanker | None = None,
    pre_rank_candidate_multiplier: int = 5,
) -> NonAr25MultiRelationAgendaResult:
    """Build a local non-ar25 relation agenda, then run one active test."""
    path = Path(trace_path)
    result = NonAr25MultiRelationAgendaResult(game_id=game_id, trace_path=path)
    result.excluded_relation_context_signatures = [
        str(signature) for signature in excluded_relation_context_signatures
    ]

    discovery = discover_cross_game_correspondences(
        path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=(
            max_candidates
            if candidate_ranker is None
            else max(max_candidates, int(max_candidates) * int(pre_rank_candidate_multiplier))
        ),
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
    )
    result.raw_discovered_candidates = len(discovery.candidates)

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
    candidates = list(discovery.candidates)
    if candidate_ranker is not None:
        candidates = list(
            candidate_ranker(
                candidates,
                live_grid=before.grid,
                valid_actions=valid_actions,
                max_candidates=max_candidates,
                preferred_predicates=preferred_predicates,
            )
        )
    result.discovered_candidates = len(candidates)
    relation_predictions = _relation_predictions(
        candidates,
        before.grid,
        valid_actions,
    )
    result.candidate_prediction_count = len(relation_predictions)

    agenda, agenda_predictions = _build_local_relation_agenda(
        relation_predictions,
        before.grid,
        preferred_predicates=preferred_predicates,
        excluded_relation_context_signatures=excluded_relation_context_signatures,
    )
    result.agenda_items = agenda
    if len(agenda) < 2:
        result.error = "not_enough_relation_candidates_for_agenda"
        return result

    missing = missing_relation_preconditions(agenda, ())
    result.missing_preconditions = [item.required_predicate for item in missing]
    result.selected_option_order = [item.option.name for item in missing]
    result.observed_agenda_predicates = [
        item.required_predicate for item in agenda if item.observed_initially
    ]
    if not result.local_agenda_observed:
        result.error = "local_relation_agenda_not_observed"
        return result

    choice = GenericDiscriminatingExperimentDesigner(
        max_competing_hypotheses=2
    ).design(
        hypotheses=agenda_predictions,
        live_grid=before.grid,
        available_actions=valid_actions,
        preferred_family="relation",
    )
    if choice is None:
        result.error = "no_non_ar25_relation_agenda_experiment"
        return result
    result.experiment = choice

    by_key = {prediction.key: prediction for prediction in agenda_predictions}
    result.selected_predictions_before_observation = [
        by_key[key] for key in choice.competing_keys if key in by_key
    ]
    if not result.hypothesis_remains_unresolved_until_observation:
        result.error = "non_ar25_agenda_hypothesis_resolved_before_observation"
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
    result.score = score_beliefs(
        result.records,
        MechanicsOracle(game_id),
        experiment_actions=max(1, result.env_actions),
    )
    return result


def _build_local_relation_agenda(
    predictions: Sequence[DiscriminatingPrediction],
    grid: Any,
    *,
    preferred_predicates: Sequence[str],
    excluded_relation_context_signatures: Sequence[str] = (),
) -> Tuple[List[NonAr25RelationAgendaItem], List[DiscriminatingPrediction]]:
    groups = _group_predictions_by_action_pair(predictions)
    excluded = {str(signature) for signature in excluded_relation_context_signatures}
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(_available_predicates(item[1], preferred_predicates)),
            -sum(prediction.support for prediction in item[1]),
            str(item[0]),
        ),
    )
    for (action, pair), group_predictions in ranked_groups:
        filtered_group = [
            prediction
            for prediction in group_predictions
            if non_ar25_relation_context_signature_from_prediction(prediction)
            not in excluded
        ]
        available = _available_predicates(filtered_group, preferred_predicates)
        if len(available) < 2:
            continue
        agenda: List[NonAr25RelationAgendaItem] = []
        source, target = pair
        for predicate in available:
            holds = _relation_holds_for_colors(grid, predicate, source, target)
            support = max(
                prediction.support
                for prediction in group_predictions
                if prediction.predicate_name == predicate
                and non_ar25_relation_context_signature_from_prediction(prediction)
                not in excluded
            )
            if holds:
                option = PrepareRelationOption(
                    predicate,
                    desired_outcome="preserved",
                    target_pair_colors=pair,
                    name=f"prepare_relation_{predicate}_colors{source}_{target}",
                )
                required = f"relation_present::{predicate}::colors{source}_{target}"
            else:
                option = AvoidRelationOption(
                    predicate,
                    desired_outcome="absent",
                    target_pair_colors=pair,
                    name=f"avoid_relation_{predicate}_absent_colors{source}_{target}",
                )
                required = f"relation_absent::{predicate}::colors{source}_{target}"
            agenda.append(
                NonAr25RelationAgendaItem(
                    option=option,
                    required_predicate=required,
                    predicate=predicate,
                    pair_colors=pair,
                    action=action,
                    observed_initially=True,
                    support=support,
                )
            )
        filtered = [
            prediction
            for prediction in filtered_group
            if prediction.predicate_name in set(available)
        ]
        return agenda, filtered
    return [], []


def non_ar25_relation_context_signature(
    *,
    action: str,
    predicate: str,
    pair_colors: Tuple[int, int],
) -> str:
    """Stable signature for one non-ar25 relation context."""
    source, target = int(pair_colors[0]), int(pair_colors[1])
    return f"{str(action)}::{str(predicate)}::colors{source}_{target}"


def non_ar25_relation_context_signature_from_prediction(
    prediction: DiscriminatingPrediction,
) -> str:
    if prediction.pair_colors is None:
        return f"{prediction.action}::{prediction.predicate_name}::source{prediction.source_color}"
    return non_ar25_relation_context_signature(
        action=prediction.action,
        predicate=prediction.predicate_name,
        pair_colors=prediction.pair_colors,
    )


def _group_predictions_by_action_pair(
    predictions: Sequence[DiscriminatingPrediction],
) -> dict[Tuple[str, Tuple[int, int]], List[DiscriminatingPrediction]]:
    groups: dict[Tuple[str, Tuple[int, int]], List[DiscriminatingPrediction]] = {}
    for prediction in predictions:
        if prediction.normalized_family != "relation" or prediction.pair_colors is None:
            continue
        key = (prediction.action, prediction.pair_colors)
        groups.setdefault(key, []).append(prediction)
    return groups


def _available_predicates(
    predictions: Sequence[DiscriminatingPrediction],
    preferred_predicates: Sequence[str],
) -> List[str]:
    available = {prediction.predicate_name for prediction in predictions}
    return [
        str(predicate)
        for predicate in preferred_predicates
        if str(predicate) in available
    ]


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
        description="Run A25 non-ar25 multi-relation agenda."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=_env_dir())
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_non_ar25_multi_relation_agenda(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
