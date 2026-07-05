"""A23 option composition.

This module composes a relation-preparation option with the existing
correspondence-validation option. The key invariant is deliberately narrow:
validation is called only after the relation predicate was observed in the
live state produced by the preparation option.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .ar25_live_option_micro_run import (
    DEFAULT_GAME_ID,
    LiveMicroRunEvent,
    LiveOptionMicroRunResult,
    run_ar25_live_option_micro_run,
)
from .relation_option import PrepareRelationOption

DEFAULT_RELATION_PREDICATE = "source_target_relation_satisfied"


@dataclass
class OptionCompositionResult:
    """Result of one relation-option -> correspondence-option composition."""

    game_id: str
    relation_option: PrepareRelationOption
    validation_option_name: str = "validate_correspondence_colors10_11"
    base_run: LiveOptionMicroRunResult | None = None
    relation_event: LiveMicroRunEvent | None = None
    validation_event: LiveMicroRunEvent | None = None
    error: str = ""

    @property
    def relation_observed(self) -> bool:
        return bool(
            self.relation_event
            and self.relation_option.predicate in self.relation_event.predicates_after
        )

    @property
    def relation_option_success(self) -> bool:
        return bool(
            self.relation_event
            and self.relation_event.termination == "success"
            and self.relation_observed
        )

    @property
    def correspondence_option_called(self) -> bool:
        return self.validation_event is not None

    @property
    def correspondence_option_called_only_if_relation_observed(self) -> bool:
        return bool(
            self.validation_event
            and self.relation_observed
            and self.relation_option.predicate in self.validation_event.predicates_present
            and "strong_ready_to_validate_correspondence"
            in self.validation_event.predicates_present
        )

    @property
    def full_composed_chain_attempted(self) -> bool:
        return self.relation_option_success and self.correspondence_option_called

    @property
    def wrong_confirmations(self) -> int:
        return 0 if self.base_run is None else self.base_run.wrong_confirmations

    @property
    def confirmation_precision(self) -> float:
        return 0.0 if self.base_run is None else self.base_run.confirmation_precision

    @property
    def transitions(self) -> int:
        return 0 if self.base_run is None else self.base_run.transitions

    @property
    def env_actions(self) -> int:
        return 0 if self.base_run is None else self.base_run.env_actions

    @property
    def trace_dependent(self) -> bool:
        return False if self.base_run is None else self.base_run.trace_dependent

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "relation_option": self.relation_option.to_dict(),
            "validation_option_name": self.validation_option_name,
            "relation_observed": self.relation_observed,
            "relation_option_success": self.relation_option_success,
            "correspondence_option_called": self.correspondence_option_called,
            "correspondence_option_called_only_if_relation_observed": (
                self.correspondence_option_called_only_if_relation_observed
            ),
            "full_composed_chain_attempted": self.full_composed_chain_attempted,
            "wrong_confirmations": self.wrong_confirmations,
            "confirmation_precision": round(self.confirmation_precision, 4),
            "transitions": self.transitions,
            "env_actions": self.env_actions,
            "trace_dependent": self.trace_dependent,
            "relation_event": (
                self.relation_event.__dict__ if self.relation_event else None
            ),
            "validation_event": (
                self.validation_event.__dict__ if self.validation_event else None
            ),
            "base_run": self.base_run.to_dict() if self.base_run else None,
            "error": self.error,
        }


def run_option_composition(
    *,
    game_id: str = DEFAULT_GAME_ID,
    environments_dir: Path | str | None = None,
    max_actions: int = 50,
    max_option_attempts: int = 1,
    relation_predicate: str = DEFAULT_RELATION_PREDICATE,
    bootstrap_probe_actions: Sequence[str] | None = None,
) -> OptionCompositionResult:
    """Compose relation preparation with validation, guarded by observation."""
    relation_option = PrepareRelationOption(
        relation_predicate,
        desired_outcome="observed",
        name=f"prepare_relation_{relation_predicate}",
    )
    result = OptionCompositionResult(
        game_id=game_id,
        relation_option=relation_option,
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

    result.relation_event = _first_successful_relation_event(
        run.events,
        relation_predicate=relation_predicate,
    )
    result.validation_event = _first_validation_event_after_relation(
        run.events,
        relation_event=result.relation_event,
    )
    if result.relation_event is None:
        result.error = "relation_precondition_not_observed"
        return result
    if result.validation_event is None:
        result.error = "validation_option_not_called_after_relation"
        return result
    if not result.correspondence_option_called_only_if_relation_observed:
        result.error = "validation_called_without_observed_relation"
    return result


def _first_successful_relation_event(
    events: Sequence[LiveMicroRunEvent],
    *,
    relation_predicate: str,
) -> LiveMicroRunEvent | None:
    for event in events:
        if event.kind != "prepare_option":
            continue
        if event.termination != "success":
            continue
        if relation_predicate not in event.predicates_after:
            continue
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
        description="Run A23 relation -> correspondence option composition."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--max-option-attempts", type=int, default=1)
    parser.add_argument("--relation-predicate", default=DEFAULT_RELATION_PREDICATE)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_option_composition(
        game_id=args.game_id,
        environments_dir=args.environments_dir,
        max_actions=args.max_actions,
        max_option_attempts=args.max_option_attempts,
        relation_predicate=args.relation_predicate,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
