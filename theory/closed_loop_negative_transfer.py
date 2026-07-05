"""A21 closed-loop negative transfer.

This module checks that negative transfer records do more than sit in a log:
they change the next experimental choice. After A19 produces a target-context
refutation, A21 applies the A20 memory and asks the designer for a second
target experiment in the same live context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

from .active_transfer_validation import (
    ActiveTransferValidationResult,
    DEFAULT_SOURCE_GAME_ID,
    DEFAULT_SOURCE_TRACE_PATH,
    DEFAULT_TARGET_GAME_ID,
    DEFAULT_TARGET_TRACE_PATH,
    run_active_transfer_validation,
)
from .cross_game_correspondence_discovery import discover_cross_game_correspondences
from .epistemic_metrics import HypothesisStatus
from .generic_discriminating_experiment_designer import (
    DiscriminatingPrediction,
    GenericDiscriminatingExperimentChoice,
    GenericDiscriminatingExperimentDesigner,
)
from .negative_transfer_memory import (
    NegativeTransferMemory,
    NegativeTransferRecord,
    apply_negative_transfer_memory,
    build_negative_transfer_records,
)
from .non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from .real_env_option_adapter import snapshot_frame
from .relation_transfer import (
    apply_relation_transfer_priors,
    extract_relation_transfer_priors,
    relation_predictions_from_candidates,
)


@dataclass
class ClosedLoopNegativeTransferResult:
    """Summary of applying negative memory to the next target experiment."""

    source_game_id: str
    target_game_id: str
    source_trace_path: Path
    target_trace_path: Path
    first_attempt: ActiveTransferValidationResult | None = None
    negative_memory: NegativeTransferMemory = field(default_factory=NegativeTransferMemory)
    repeated_context: str = ""
    second_experiment: GenericDiscriminatingExperimentChoice | None = None
    second_selected_predictions: List[DiscriminatingPrediction] = field(default_factory=list)
    source_relation_status: HypothesisStatus = HypothesisStatus.UNRESOLVED
    error: str = ""

    @property
    def negative_memory_used(self) -> bool:
        return bool(self.negative_memory.records and self.second_experiment)

    @property
    def repeated_failed_context_not_selected(self) -> bool:
        if not self.repeated_context or self.second_experiment is None:
            return False
        return self.repeated_context not in self.second_experiment.competing_keys

    @property
    def alternative_experiment_selected(self) -> bool:
        if self.first_attempt is None or self.first_attempt.experiment is None:
            return False
        if self.second_experiment is None:
            return False
        first = self.first_attempt.experiment
        second = self.second_experiment
        return (
            first.competing_keys != second.competing_keys
            or first.action.name != second.action.name
            or first.action.action_args != second.action.action_args
        )

    @property
    def source_relation_remains_confirmed(self) -> bool:
        return self.source_relation_status == HypothesisStatus.CONFIRMED

    @property
    def wrong_confirmations(self) -> int:
        if self.first_attempt is None:
            return 0
        return self.first_attempt.wrong_confirmations

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_game_id": self.source_game_id,
            "target_game_id": self.target_game_id,
            "source_trace_path": str(self.source_trace_path),
            "target_trace_path": str(self.target_trace_path),
            "negative_memory_used": self.negative_memory_used,
            "repeated_failed_context_not_selected": (
                self.repeated_failed_context_not_selected
            ),
            "alternative_experiment_selected": self.alternative_experiment_selected,
            "source_relation_status": self.source_relation_status.value,
            "source_relation_remains_confirmed": (
                self.source_relation_remains_confirmed
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "repeated_context": self.repeated_context,
            "negative_memory": self.negative_memory.to_dict(),
            "first_experiment": (
                self.first_attempt.experiment.to_dict()
                if self.first_attempt and self.first_attempt.experiment
                else None
            ),
            "second_experiment": (
                self.second_experiment.to_dict()
                if self.second_experiment
                else None
            ),
            "second_selected_predictions": [
                _prediction_to_dict(prediction)
                for prediction in self.second_selected_predictions
            ],
            "error": self.error,
        }


def run_closed_loop_negative_transfer(
    *,
    source_game_id: str = DEFAULT_SOURCE_GAME_ID,
    source_trace_path: Path | str = DEFAULT_SOURCE_TRACE_PATH,
    target_game_id: str = DEFAULT_TARGET_GAME_ID,
    target_trace_path: Path | str = DEFAULT_TARGET_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    prior_weight: float = 100.0,
    negative_weight_delta: float | None = None,
) -> ClosedLoopNegativeTransferResult:
    """Run A19, build negative memory, then select the next target test."""
    source_path = Path(source_trace_path)
    target_path = Path(target_trace_path)
    result = ClosedLoopNegativeTransferResult(
        source_game_id=source_game_id,
        target_game_id=target_game_id,
        source_trace_path=source_path,
        target_trace_path=target_path,
    )

    first_attempt = run_active_transfer_validation(
        source_game_id=source_game_id,
        source_trace_path=source_path,
        target_game_id=target_game_id,
        target_trace_path=target_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        prior_weight=prior_weight,
    )
    result.first_attempt = first_attempt
    if first_attempt.error:
        result.error = f"first_attempt_failed:{first_attempt.error}"
        return result
    if first_attempt.source_result is not None:
        result.source_relation_status = _source_relation_status(first_attempt)

    records = build_negative_transfer_records(
        target_game=target_game_id,
        selected_predictions=first_attempt.selected_predictions_before_observation,
        revisions=first_attempt.revisions,
    )
    if negative_weight_delta is not None:
        records = [
            NegativeTransferRecord(
                source_relation=record.source_relation,
                target_game=record.target_game,
                target_context_signature=record.target_context_signature,
                tested_hypothesis=record.tested_hypothesis,
                observed_outcome=record.observed_outcome,
                effect=record.effect,
                weight_delta=float(negative_weight_delta),
            )
            for record in records
        ]
    result.negative_memory = NegativeTransferMemory(records=list(records))
    if not result.negative_memory.records:
        result.error = "no_negative_transfer_records"
        return result
    result.repeated_context = result.negative_memory.records[0].tested_hypothesis

    discovery = discover_cross_game_correspondences(
        target_path,
        game_id=target_game_id,
        min_pixel_support=min_pixel_support,
        top_k=max_candidates,
    )
    base_predictions = relation_predictions_from_candidates(discovery.candidates)
    priors = extract_relation_transfer_priors(
        first_attempt.source_result.revisions if first_attempt.source_result else [],
        source_game_id=source_game_id,
        prior_weight=prior_weight,
    )
    transferred = apply_relation_transfer_priors(base_predictions, priors)
    adjusted = apply_negative_transfer_memory(
        transferred,
        result.negative_memory,
        target_game=target_game_id,
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
    choice = GenericDiscriminatingExperimentDesigner(
        max_competing_hypotheses=2
    ).design(
        hypotheses=adjusted,
        live_grid=before.grid,
        available_actions=valid_actions,
        preferred_family="relation",
    )
    if choice is None:
        result.error = "no_second_relation_experiment"
        return result
    result.second_experiment = choice
    by_key = {prediction.key: prediction for prediction in adjusted}
    result.second_selected_predictions = [
        by_key[key] for key in choice.competing_keys if key in by_key
    ]
    return result


def _source_relation_status(
    result: ActiveTransferValidationResult,
) -> HypothesisStatus:
    if result.source_result is None:
        return HypothesisStatus.UNRESOLVED
    for revision in result.source_result.revisions:
        if (
            revision.key == "relation::ACTION6::same_shape::colors9_8::preserved"
            and revision.status_after == HypothesisStatus.CONFIRMED
        ):
            return HypothesisStatus.CONFIRMED
    return HypothesisStatus.UNRESOLVED


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
        description="Run A21 closed-loop negative transfer selection."
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
    result = run_closed_loop_negative_transfer(
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
