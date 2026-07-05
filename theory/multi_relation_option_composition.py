"""A24 multi-relation option composition.

The composer builds a small causal agenda from missing relation predicates. It
does not rank actions by game score: relation options are selected because
their required predicates are absent, then validation is accepted only when all
required relation predicates are observed in the live validation context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from .ar25_live_option_micro_run import (
    DEFAULT_GAME_ID,
    LiveMicroRunEvent,
    LiveOptionMicroRunResult,
    run_ar25_live_option_micro_run,
)
from .relation_option import AvoidRelationOption, PrepareRelationOption, RelationOption


@dataclass(frozen=True)
class RelationPreconditionOption:
    """One relation option tied to the live predicate it must establish."""

    option: RelationOption
    required_predicate: str
    description: str = ""

    def observed_in(self, predicates: Iterable[str]) -> bool:
        present = {str(predicate) for predicate in predicates}
        return self.required_predicate in present

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option.to_dict(),
            "required_predicate": self.required_predicate,
            "description": self.description,
        }


@dataclass
class MultiRelationOptionCompositionResult:
    """Result of composing several relation options before validation."""

    game_id: str
    relation_preconditions: List[RelationPreconditionOption]
    base_run: LiveOptionMicroRunResult | None = None
    relation_event: LiveMicroRunEvent | None = None
    validation_event: LiveMicroRunEvent | None = None
    missing_preconditions: List[str] = field(default_factory=list)
    selected_option_order: List[str] = field(default_factory=list)
    observed_preconditions_before_validation: List[str] = field(default_factory=list)
    selection_reason: str = "missing_preconditions"
    error: str = ""

    @property
    def multiple_relations_candidate(self) -> bool:
        return len(self.relation_preconditions) >= 2

    @property
    def includes_prepare_and_avoid_options(self) -> bool:
        modes = {precondition.option.mode for precondition in self.relation_preconditions}
        return "prepare" in modes and "avoid" in modes

    @property
    def order_chosen_by_missing_preconditions(self) -> bool:
        expected = [
            precondition.option.name
            for precondition in self.relation_preconditions
            if precondition.required_predicate in self.missing_preconditions
        ]
        return bool(
            self.selection_reason == "missing_preconditions"
            and expected == self.selected_option_order
        )

    @property
    def relation_options_success(self) -> bool:
        return bool(
            self.relation_event
            and self.relation_event.termination == "success"
            and self.all_required_relation_preconditions_observed
        )

    @property
    def all_required_relation_preconditions_observed(self) -> bool:
        if self.validation_event is None:
            return False
        return all(
            precondition.observed_in(self.validation_event.predicates_present)
            for precondition in self.relation_preconditions
        )

    @property
    def validation_called(self) -> bool:
        return self.validation_event is not None

    @property
    def validation_called_only_when_all_required_relations_observed(self) -> bool:
        return bool(self.validation_called and self.all_required_relation_preconditions_observed)

    @property
    def full_composed_chain_attempted(self) -> bool:
        return bool(self.relation_event and self.validation_event)

    @property
    def wrong_confirmations(self) -> int:
        return 0 if self.base_run is None else self.base_run.wrong_confirmations

    @property
    def confirmation_precision(self) -> float:
        return 0.0 if self.base_run is None else self.base_run.confirmation_precision

    @property
    def trace_dependent(self) -> bool:
        return False if self.base_run is None else self.base_run.trace_dependent

    @property
    def env_actions(self) -> int:
        return 0 if self.base_run is None else self.base_run.env_actions

    @property
    def transitions(self) -> int:
        return 0 if self.base_run is None else self.base_run.transitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "multiple_relations_candidate": self.multiple_relations_candidate,
            "includes_prepare_and_avoid_options": self.includes_prepare_and_avoid_options,
            "selection_reason": self.selection_reason,
            "order_chosen_by_missing_preconditions": (
                self.order_chosen_by_missing_preconditions
            ),
            "missing_preconditions": list(self.missing_preconditions),
            "selected_option_order": list(self.selected_option_order),
            "observed_preconditions_before_validation": (
                list(self.observed_preconditions_before_validation)
            ),
            "all_required_relation_preconditions_observed": (
                self.all_required_relation_preconditions_observed
            ),
            "validation_called": self.validation_called,
            "validation_called_only_when_all_required_relations_observed": (
                self.validation_called_only_when_all_required_relations_observed
            ),
            "relation_options_success": self.relation_options_success,
            "full_composed_chain_attempted": self.full_composed_chain_attempted,
            "wrong_confirmations": self.wrong_confirmations,
            "confirmation_precision": round(self.confirmation_precision, 4),
            "trace_dependent": self.trace_dependent,
            "env_actions": self.env_actions,
            "transitions": self.transitions,
            "relation_preconditions": [
                precondition.to_dict()
                for precondition in self.relation_preconditions
            ],
            "relation_event": (
                self.relation_event.__dict__ if self.relation_event else None
            ),
            "validation_event": (
                self.validation_event.__dict__ if self.validation_event else None
            ),
            "base_run": self.base_run.to_dict() if self.base_run else None,
            "error": self.error,
        }


def default_ar25_relation_preconditions() -> Tuple[RelationPreconditionOption, ...]:
    """Default relation agenda for ar25 correspondence validation."""
    return (
        RelationPreconditionOption(
            option=PrepareRelationOption(
                "same_shape",
                desired_outcome="observed",
                name="prepare_relation_same_shape",
            ),
            required_predicate="selected_source_matches_target_shape",
            description="source and target shapes must match before validation",
        ),
        RelationPreconditionOption(
            option=PrepareRelationOption(
                "aligned_with",
                desired_outcome="observed",
                name="prepare_relation_aligned_with",
            ),
            required_predicate="source_target_aligned",
            description="source and target must be aligned before validation",
        ),
        RelationPreconditionOption(
            option=AvoidRelationOption(
                "failed_validation_context",
                desired_outcome="absent",
                name="avoid_relation_failed_validation_context_absent",
            ),
            required_predicate="no_recent_failed_validation_same_context",
            description="avoid replaying a recently failed validation context",
        ),
    )


def missing_relation_preconditions(
    relation_preconditions: Sequence[RelationPreconditionOption],
    predicates_present: Iterable[str],
) -> List[RelationPreconditionOption]:
    """Return relation options whose required predicates are absent."""
    present = {str(predicate) for predicate in predicates_present}
    return [
        precondition
        for precondition in relation_preconditions
        if precondition.required_predicate not in present
    ]


def run_multi_relation_option_composition(
    *,
    game_id: str = DEFAULT_GAME_ID,
    environments_dir: Path | str | None = None,
    max_actions: int = 50,
    max_option_attempts: int = 1,
    relation_preconditions: Sequence[RelationPreconditionOption] | None = None,
    bootstrap_probe_actions: Sequence[str] | None = None,
) -> MultiRelationOptionCompositionResult:
    """Run an ar25 relation-agenda composition before correspondence validation."""
    preconditions = list(
        relation_preconditions or default_ar25_relation_preconditions()
    )
    result = MultiRelationOptionCompositionResult(
        game_id=game_id,
        relation_preconditions=preconditions,
    )

    run = run_ar25_live_option_micro_run(
        game_id=game_id,
        environments_dir=environments_dir,
        max_actions=max_actions,
        max_option_attempts=max_option_attempts,
        bootstrap_probe_actions=(
            bootstrap_probe_actions
            if bootstrap_probe_actions is not None
            else ("ACTION4", "ACTION4", "ACTION5", "ACTION2")
        ),
        use_contextual_readiness=True,
        use_prepare_option=True,
    )
    result.base_run = run
    if run.error:
        result.error = f"base_run_failed:{run.error}"
        return result

    result.relation_event = _first_successful_prepare_event(run.events)
    result.validation_event = _first_validation_event_after_relation(
        run.events,
        relation_event=result.relation_event,
    )
    if result.relation_event is None:
        result.error = "relation_preparation_not_successful"
        return result

    missing = missing_relation_preconditions(
        preconditions,
        result.relation_event.predicates_present,
    )
    result.missing_preconditions = [
        precondition.required_predicate for precondition in missing
    ]
    result.selected_option_order = [
        precondition.option.name for precondition in missing
    ]

    if result.validation_event is None:
        result.error = "validation_option_not_called_after_relation_agenda"
        return result
    result.observed_preconditions_before_validation = [
        precondition.required_predicate
        for precondition in preconditions
        if precondition.observed_in(result.validation_event.predicates_present)
    ]
    if not result.validation_called_only_when_all_required_relations_observed:
        result.error = "validation_called_before_all_relation_preconditions"
    return result


def _first_successful_prepare_event(
    events: Sequence[LiveMicroRunEvent],
) -> LiveMicroRunEvent | None:
    for event in events:
        if event.kind == "prepare_option" and event.termination == "success":
            return event
    return None


def _first_validation_event_after_relation(
    events: Sequence[LiveMicroRunEvent],
    *,
    relation_event: LiveMicroRunEvent | None,
) -> LiveMicroRunEvent | None:
    if relation_event is None:
        return None
    relation_seen = False
    for event in events:
        if event is relation_event:
            relation_seen = True
            continue
        if relation_seen and event.kind == "option_attempt":
            return event
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A24 multi-relation option composition."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--max-option-attempts", type=int, default=1)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_multi_relation_option_composition(
        game_id=args.game_id,
        environments_dir=args.environments_dir,
        max_actions=args.max_actions,
        max_option_attempts=args.max_option_attempts,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
