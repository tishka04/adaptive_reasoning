"""A30 multi-game experimental evaluation.

A30 is an evaluation harness for the existing A1-A29 machinery. It discovers
trace-backed games, runs the current hypothesis/discriminating-experiment/
revision/agenda/memory stack, and reports aggregate epistemic and functional
metrics without adding new predicates, options, or planners.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from .active_transfer_validation import ActiveTransferValidationResult
from .active_transfer_validation import run_active_transfer_validation
from .ar25_live_option_micro_run import run_ar25_live_option_micro_run
from .ar25_replay import run_ar25_belief_loop
from .cross_game_correspondence_discovery import discover_cross_game_correspondences
from .cross_game_transferability_check import CrossGameTransferabilityCheckResult
from .cross_game_transferability_check import run_cross_game_transferability_check
from .epistemic_metrics import HypothesisStatus
from .non_ar25_functional_negative_memory import (
    FunctionalAgendaNegativeMemory,
    build_functional_agenda_negative_memory,
)
from .non_ar25_functional_progress import (
    NonAr25FunctionalProgressResult,
    observe_functional_progress,
)
from .non_ar25_multi_relation_agenda import (
    NonAr25MultiRelationAgendaResult,
    non_ar25_relation_context_signature_from_prediction,
    run_non_ar25_multi_relation_agenda,
)
from .non_ar25_transferability_model import (
    NegativeTransferabilityModel,
    build_negative_transferability_model,
)

DEFAULT_TRACES_DIR = Path("human_traces")
DEFAULT_MAX_GAMES = 10
PredicateGenerator = Callable[..., Iterable[str]]
AnchorExpander = Callable[..., Iterable[Any]]
CandidateRanker = Callable[..., Sequence[Any]]


@dataclass(frozen=True)
class EvaluationTrace:
    """One trace selected for a game evaluation."""

    game_id: str
    trace_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
        }


@dataclass
class EvaluationResult:
    """A30 per-game evaluation metrics."""

    game_id: str
    trace_path: Path | None = None
    experiments_run: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_refuted: int = 0
    transfer_priors_used: int = 0
    negative_memories_created: int = 0
    useful_new_states: int = 0
    functional_progress: float = 0.0
    wrong_confirmations: int = 0
    candidate_hypotheses: int = 0
    relation_agenda_items: int = 0
    negative_memory_contexts_avoided: int = 0
    transferability_groups: int = 0
    downweighted_transferability_contexts: int = 0
    trace_support_counted_as_proof: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def experimental_efficiency(self) -> float:
        if self.experiments_run <= 0:
            return 0.0
        return (
            float(self.hypotheses_confirmed + self.hypotheses_refuted)
            / float(self.experiments_run)
        )

    @property
    def negative_memory_value(self) -> int:
        return self.negative_memory_contexts_avoided

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "experimental_efficiency": round(self.experimental_efficiency, 4),
            "transfer_priors_used": self.transfer_priors_used,
            "negative_memories_created": self.negative_memories_created,
            "negative_memory_contexts_avoided": (
                self.negative_memory_contexts_avoided
            ),
            "negative_memory_value": self.negative_memory_value,
            "useful_new_states": self.useful_new_states,
            "functional_progress": round(self.functional_progress, 4),
            "wrong_confirmations": self.wrong_confirmations,
            "candidate_hypotheses": self.candidate_hypotheses,
            "relation_agenda_items": self.relation_agenda_items,
            "transferability_groups": self.transferability_groups,
            "downweighted_transferability_contexts": (
                self.downweighted_transferability_contexts
            ),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "errors": list(self.errors),
        }


@dataclass
class MultiGameEvaluationResult:
    """Aggregate A30 metrics across games and optional transfer checks."""

    game_results: List[EvaluationResult] = field(default_factory=list)
    active_transfer: ActiveTransferValidationResult | None = None
    cross_game_transferability: CrossGameTransferabilityCheckResult | None = None
    skipped_traces: List[EvaluationTrace] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def games_evaluated(self) -> int:
        return len(self.game_results)

    @property
    def experiments_run(self) -> int:
        return (
            sum(result.experiments_run for result in self.game_results)
            + _active_transfer_experiments(self.active_transfer)
            + _cross_game_experiments(self.cross_game_transferability)
        )

    @property
    def hypotheses_confirmed(self) -> int:
        return (
            sum(result.hypotheses_confirmed for result in self.game_results)
            + _confirmed_revisions(self.active_transfer)
            + _cross_game_confirmed(self.cross_game_transferability)
        )

    @property
    def hypotheses_refuted(self) -> int:
        return (
            sum(result.hypotheses_refuted for result in self.game_results)
            + _refuted_revisions(self.active_transfer)
            + _cross_game_refuted(self.cross_game_transferability)
        )

    @property
    def transfer_priors_used(self) -> int:
        total = sum(result.transfer_priors_used for result in self.game_results)
        if self.active_transfer and self.active_transfer.transferred_prior_used:
            total += 1
        return total

    @property
    def transfer_hypotheses_tested(self) -> int:
        if not self.active_transfer or not self.active_transfer.transferred_prior_used:
            return 0
        return len(self.active_transfer.selected_predictions_before_observation)

    @property
    def transfer_refutations_after_transfer(self) -> int:
        if not self.active_transfer or not self.active_transfer.transferred_prior_used:
            return 0
        return _refuted_revisions(self.active_transfer)

    @property
    def useful_transfer_rate(self) -> float:
        if self.transfer_hypotheses_tested <= 0:
            return 0.0
        useful = 0
        if self.active_transfer and self.active_transfer.local_revision_after_observation:
            useful += int(self.active_transfer.transferred_prior_used)
        return float(useful) / float(self.transfer_hypotheses_tested)

    @property
    def erroneous_transfer_rate(self) -> float:
        if self.transfer_hypotheses_tested <= 0:
            return 0.0
        return (
            float(self.transfer_refutations_after_transfer)
            / float(self.transfer_hypotheses_tested)
        )

    @property
    def negative_memories_created(self) -> int:
        return sum(result.negative_memories_created for result in self.game_results)

    @property
    def negative_memory_contexts_avoided(self) -> int:
        total = sum(
            result.negative_memory_contexts_avoided
            for result in self.game_results
        )
        if (
            self.cross_game_transferability
            and self.cross_game_transferability.analogous_context_downweighted
        ):
            total += len(
                self.cross_game_transferability.analogous_context_signatures
            )
        return total

    @property
    def useful_new_states(self) -> int:
        return sum(result.useful_new_states for result in self.game_results)

    @property
    def functional_progress(self) -> float:
        if self.experiments_run <= 0:
            return 0.0
        return float(self.useful_new_states) / float(self.experiments_run)

    @property
    def experimental_efficiency(self) -> float:
        if self.experiments_run <= 0:
            return 0.0
        return (
            float(self.hypotheses_confirmed + self.hypotheses_refuted)
            / float(self.experiments_run)
        )

    @property
    def wrong_confirmations(self) -> int:
        total = sum(result.wrong_confirmations for result in self.game_results)
        if self.active_transfer is not None:
            total += self.active_transfer.wrong_confirmations
        if self.cross_game_transferability is not None:
            total += self.cross_game_transferability.wrong_confirmations
        return total

    @property
    def trace_support_counted_as_proof(self) -> bool:
        return bool(
            any(result.trace_support_counted_as_proof for result in self.game_results)
            or self.active_transfer
            and any(
                prediction.prior_counted_as_proof
                for prediction in self.active_transfer.selected_predictions_before_observation
            )
            or self.cross_game_transferability
            and self.cross_game_transferability.trace_support_counted_as_proof
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "games_evaluated": self.games_evaluated,
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "experimental_efficiency": round(self.experimental_efficiency, 4),
            "transfer_priors_used": self.transfer_priors_used,
            "transfer_hypotheses_tested": self.transfer_hypotheses_tested,
            "useful_transfer_rate": round(self.useful_transfer_rate, 4),
            "erroneous_transfer_rate": round(self.erroneous_transfer_rate, 4),
            "transfer_refutations_after_transfer": (
                self.transfer_refutations_after_transfer
            ),
            "negative_memories_created": self.negative_memories_created,
            "negative_memory_contexts_avoided": (
                self.negative_memory_contexts_avoided
            ),
            "useful_new_states": self.useful_new_states,
            "functional_progress": round(self.functional_progress, 4),
            "wrong_confirmations": self.wrong_confirmations,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "game_results": [result.to_dict() for result in self.game_results],
            "active_transfer": (
                self.active_transfer.to_dict() if self.active_transfer else None
            ),
            "cross_game_transferability": (
                self.cross_game_transferability.to_dict()
                if self.cross_game_transferability
                else None
            ),
            "skipped_traces": [trace.to_dict() for trace in self.skipped_traces],
            "errors": list(self.errors),
        }


def run_multi_game_evaluation(
    *,
    traces_dir: Path | str = DEFAULT_TRACES_DIR,
    trace_paths: Sequence[Path | str] = (),
    max_games: int = DEFAULT_MAX_GAMES,
    include_ar25: bool = True,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    run_memory_comparison: bool = True,
    run_transfer_checks: bool = True,
    ar25_budget: int = 80,
    ar25_revise_every: int = 20,
    predicate_generator: PredicateGenerator | None = None,
    anchor_expander: AnchorExpander | None = None,
    candidate_ranker: CandidateRanker | None = None,
    preferred_predicates: Sequence[str] | None = None,
) -> MultiGameEvaluationResult:
    """Run the current theory stack across several trace-backed games."""
    traces = select_evaluation_traces(
        traces_dir=traces_dir,
        trace_paths=trace_paths,
        max_games=max_games,
        include_ar25=include_ar25,
    )
    result = MultiGameEvaluationResult()

    for trace in traces:
        game_result = evaluate_game(
            trace.game_id,
            trace.trace_path,
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
            run_memory_comparison=run_memory_comparison,
            ar25_budget=ar25_budget,
            ar25_revise_every=ar25_revise_every,
            predicate_generator=predicate_generator,
            anchor_expander=anchor_expander,
            candidate_ranker=candidate_ranker,
            preferred_predicates=preferred_predicates,
        )
        result.game_results.append(game_result)

    if run_transfer_checks:
        trace_by_game = {trace.game_id: trace.trace_path for trace in traces}
        _run_transfer_checks(
            result,
            trace_by_game,
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
        )
    return result


def evaluate_game(
    game_id: str,
    trace_path: Path | str,
    *,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    run_memory_comparison: bool = True,
    ar25_budget: int = 80,
    ar25_revise_every: int = 20,
    predicate_generator: PredicateGenerator | None = None,
    anchor_expander: AnchorExpander | None = None,
    candidate_ranker: CandidateRanker | None = None,
    preferred_predicates: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate one game using only existing theory runners."""
    path = Path(trace_path)
    if game_id.startswith("ar25"):
        return _evaluate_ar25_game(
            game_id,
            path,
            environments_dir=environments_dir,
            ar25_budget=ar25_budget,
            ar25_revise_every=ar25_revise_every,
        )
    return _evaluate_non_ar25_game(
        game_id,
        path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        run_memory_comparison=run_memory_comparison,
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
        candidate_ranker=candidate_ranker,
        preferred_predicates=preferred_predicates,
    )


def select_evaluation_traces(
    *,
    traces_dir: Path | str = DEFAULT_TRACES_DIR,
    trace_paths: Sequence[Path | str] = (),
    max_games: int = DEFAULT_MAX_GAMES,
    include_ar25: bool = True,
) -> List[EvaluationTrace]:
    """Select the latest steps trace for each game id."""
    candidates = (
        [Path(path) for path in trace_paths]
        if trace_paths
        else sorted(Path(traces_dir).glob("*.steps.jsonl"))
    )
    by_game: dict[str, Path] = {}
    for path in candidates:
        game_id = _game_id_from_trace(path)
        if not game_id:
            continue
        if not include_ar25 and game_id.startswith("ar25"):
            continue
        previous = by_game.get(game_id)
        if previous is None or path.name > previous.name:
            by_game[game_id] = path

    selected: List[EvaluationTrace] = []
    for game_id in sorted(by_game):
        selected.append(EvaluationTrace(game_id=game_id, trace_path=by_game[game_id]))
        if len(selected) >= max(1, int(max_games)):
            break
    return selected


def _evaluate_ar25_game(
    game_id: str,
    trace_path: Path,
    *,
    environments_dir: Path | str | None,
    ar25_budget: int,
    ar25_revise_every: int,
) -> EvaluationResult:
    result = EvaluationResult(game_id=game_id, trace_path=trace_path)
    try:
        _, score, stats = run_ar25_belief_loop(
            budget=ar25_budget,
            revise_every=ar25_revise_every,
        )
        result.hypotheses_confirmed += int(score.hypotheses_confirmed)
        result.hypotheses_refuted += int(score.hypotheses_refuted)
        result.wrong_confirmations += int(score.wrong_confirmations)
        result.candidate_hypotheses += (
            int(score.hypotheses_confirmed)
            + int(score.hypotheses_refuted)
            + int(score.unverifiable)
        )
        result.errors.extend(_prefixed_error("ar25_replay", stats.get("error")))
    except Exception as exc:  # pragma: no cover - integration failure path
        result.errors.append(f"ar25_replay_failed:{exc}")

    live = run_ar25_live_option_micro_run(
        game_id=game_id,
        environments_dir=environments_dir,
        max_actions=20,
        max_option_attempts=1,
        use_prepare_option=True,
    )
    result.experiments_run += int(live.env_actions)
    result.wrong_confirmations += int(live.wrong_confirmations)
    if live.option_successes or live.full_chain_successes:
        result.useful_new_states += 1
    if live.error:
        result.errors.append(f"ar25_live_failed:{live.error}")
    result.functional_progress = _safe_ratio(
        result.useful_new_states,
        result.experiments_run,
    )
    return result


def _evaluate_non_ar25_game(
    game_id: str,
    trace_path: Path,
    *,
    environments_dir: Path | str | None,
    max_candidates: int,
    min_pixel_support: int,
    run_memory_comparison: bool,
    predicate_generator: PredicateGenerator | None,
    anchor_expander: AnchorExpander | None,
    candidate_ranker: CandidateRanker | None,
    preferred_predicates: Sequence[str] | None,
) -> EvaluationResult:
    result = EvaluationResult(game_id=game_id, trace_path=trace_path)
    discovery = discover_cross_game_correspondences(
        trace_path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=max_candidates,
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
    )
    result.candidate_hypotheses = len(discovery.candidates)
    result.wrong_confirmations += discovery.wrong_confirmations

    agenda = run_non_ar25_multi_relation_agenda(
        game_id=game_id,
        trace_path=trace_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        preferred_predicates=preferred_predicates or ("same_shape", "aligned_with", "adjacent_to"),
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
        candidate_ranker=candidate_ranker,
    )
    _accumulate_agenda(result, agenda)
    functional = _functional_result_from_agenda(game_id, trace_path, agenda)
    _accumulate_functional_progress(result, functional)
    memory = build_functional_agenda_negative_memory(functional)
    result.negative_memories_created += len(memory.records)
    model = build_negative_transferability_model(memory.records, min_negative_count=2)
    _accumulate_transferability_model(result, model)

    if run_memory_comparison and memory.records:
        adapted = run_non_ar25_multi_relation_agenda(
            game_id=game_id,
            trace_path=trace_path,
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
            preferred_predicates=preferred_predicates
            or ("same_shape", "aligned_with", "adjacent_to"),
            predicate_generator=predicate_generator,
            anchor_expander=anchor_expander,
            candidate_ranker=candidate_ranker,
            excluded_relation_context_signatures=(
                memory.excluded_relation_context_signatures
            ),
        )
        _accumulate_agenda(result, adapted)
        adapted_functional = _functional_result_from_agenda(
            game_id,
            trace_path,
            adapted,
        )
        _accumulate_functional_progress(result, adapted_functional)
        result.negative_memory_contexts_avoided += _avoided_context_count(
            memory,
            adapted,
        )

    result.functional_progress = _safe_ratio(
        result.useful_new_states,
        result.experiments_run,
    )
    return result


def _run_transfer_checks(
    result: MultiGameEvaluationResult,
    trace_by_game: dict[str, Path],
    *,
    environments_dir: Path | str | None,
    max_candidates: int,
    min_pixel_support: int,
) -> None:
    ft09 = "ft09-0d8bbf25"
    bp35 = "bp35-0a0ad940"
    dc22 = "dc22-4c9bff3e"

    if ft09 in trace_by_game and bp35 in trace_by_game:
        active = run_active_transfer_validation(
            source_game_id=ft09,
            source_trace_path=trace_by_game[ft09],
            target_game_id=bp35,
            target_trace_path=trace_by_game[bp35],
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
        )
        result.active_transfer = active
        if active.error:
            result.errors.append(f"active_transfer_failed:{active.error}")

    if ft09 in trace_by_game and dc22 in trace_by_game:
        cross = run_cross_game_transferability_check(
            source_game_id=ft09,
            source_trace_path=trace_by_game[ft09],
            target_game_id=dc22,
            target_trace_path=trace_by_game[dc22],
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
        )
        result.cross_game_transferability = cross
        if cross.error:
            result.errors.append(f"cross_game_transferability_failed:{cross.error}")


def _functional_result_from_agenda(
    game_id: str,
    trace_path: Path,
    agenda: NonAr25MultiRelationAgendaResult,
) -> NonAr25FunctionalProgressResult:
    result = NonAr25FunctionalProgressResult(
        game_id=game_id,
        trace_path=trace_path,
        agenda_result=agenda,
    )
    if agenda.error:
        result.error = f"agenda_failed:{agenda.error}"
        return result
    if agenda.transition_update is None:
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


def _accumulate_agenda(
    result: EvaluationResult,
    agenda: NonAr25MultiRelationAgendaResult,
) -> None:
    result.experiments_run += int(agenda.env_actions)
    result.relation_agenda_items += len(agenda.agenda_items)
    result.wrong_confirmations += agenda.wrong_confirmations
    result.trace_support_counted_as_proof = bool(
        result.trace_support_counted_as_proof
        or agenda.trace_support_counted_as_proof
    )
    for revision in agenda.revisions:
        if revision.status_after == HypothesisStatus.CONFIRMED:
            result.hypotheses_confirmed += 1
        elif revision.status_after == HypothesisStatus.REFUTED:
            result.hypotheses_refuted += 1
    if agenda.error:
        result.errors.append(f"agenda_failed:{agenda.error}")


def _accumulate_functional_progress(
    result: EvaluationResult,
    functional: NonAr25FunctionalProgressResult,
) -> None:
    if functional.progress and functional.progress.useful_new_state:
        result.useful_new_states += 1
    result.trace_support_counted_as_proof = bool(
        result.trace_support_counted_as_proof
        or functional.trace_support_counted_as_proof
    )
    if functional.error and not functional.error.startswith("agenda_failed:"):
        result.errors.append(f"functional_progress:{functional.error}")


def _accumulate_transferability_model(
    result: EvaluationResult,
    model: NegativeTransferabilityModel,
) -> None:
    result.transferability_groups += len(model.groups)
    result.downweighted_transferability_contexts += len(
        model.downweighted_context_signatures
    )


def _selected_pair(
    agenda: NonAr25MultiRelationAgendaResult,
) -> Tuple[int, int] | None:
    if agenda.experiment is not None and agenda.experiment.predicted_pairs:
        pair = agenda.experiment.predicted_pairs[0]
        return (int(pair[0]), int(pair[1]))
    for prediction in agenda.selected_predictions_before_observation:
        if prediction.pair_colors is not None:
            return prediction.pair_colors
    return None


def _avoided_context_count(
    memory: FunctionalAgendaNegativeMemory,
    adapted: NonAr25MultiRelationAgendaResult,
) -> int:
    excluded = set(memory.excluded_relation_context_signatures)
    if not excluded:
        return 0
    selected = {
        non_ar25_relation_context_signature_from_prediction(prediction)
        for prediction in adapted.selected_predictions_before_observation
    }
    agenda_contexts = {
        f"{item.action}::{item.predicate}::colors{item.pair_colors[0]}_{item.pair_colors[1]}"
        for item in adapted.agenda_items
    }
    avoided = [
        context
        for context in excluded
        if context not in selected and context not in agenda_contexts
    ]
    return len(avoided)


def _active_transfer_experiments(
    result: ActiveTransferValidationResult | None,
) -> int:
    return 0 if result is None else int(result.env_actions)


def _cross_game_experiments(
    result: CrossGameTransferabilityCheckResult | None,
) -> int:
    if result is None:
        return 0
    total = 0
    source = result.source_model_run
    if source is not None:
        total += sum(attempt.env_actions for attempt in source.repeated_attempts)
        if source.adapted_attempt is not None:
            total += source.adapted_attempt.env_actions
    if result.target_baseline is not None:
        total += result.target_baseline.env_actions
    if result.target_adapted is not None:
        total += result.target_adapted.env_actions
    return total


def _confirmed_revisions(result: Any) -> int:
    revisions = getattr(result, "revisions", ())
    return sum(
        1
        for revision in revisions
        if revision.status_after == HypothesisStatus.CONFIRMED
    )


def _refuted_revisions(result: Any) -> int:
    revisions = getattr(result, "revisions", ())
    return sum(
        1
        for revision in revisions
        if revision.status_after == HypothesisStatus.REFUTED
    )


def _cross_game_confirmed(
    result: CrossGameTransferabilityCheckResult | None,
) -> int:
    return sum(
        _confirmed_revisions(source)
        for source in _cross_game_revision_sources(result)
    )


def _cross_game_refuted(
    result: CrossGameTransferabilityCheckResult | None,
) -> int:
    return sum(
        _refuted_revisions(source)
        for source in _cross_game_revision_sources(result)
    )


def _cross_game_revision_sources(
    result: CrossGameTransferabilityCheckResult | None,
) -> List[Any]:
    if result is None:
        return []
    sources: List[Any] = []
    source = result.source_model_run
    if source is not None:
        sources.extend(attempt.agenda_result for attempt in source.repeated_attempts)
        if source.adapted_attempt is not None:
            sources.append(source.adapted_attempt.agenda_result)
    sources.extend([result.target_baseline, result.target_adapted])
    return [source for source in sources if source is not None]


def _game_id_from_trace(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                game_id = str(item.get("game_id", "")).strip()
                if game_id:
                    return game_id
                break
    except OSError:
        return ""
    name = path.name.split(".")[0]
    return name


def _prefixed_error(prefix: str, error: Any) -> List[str]:
    return [] if not error else [f"{prefix}:{error}"]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _parse_paths(values: Sequence[str]) -> List[Path]:
    return [Path(value) for value in values if value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A30 multi-game experimental evaluation."
    )
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--trace-path", action="append", default=[])
    parser.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES)
    parser.add_argument("--exclude-ar25", action="store_true")
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--skip-memory-comparison", action="store_true")
    parser.add_argument("--skip-transfer-checks", action="store_true")
    parser.add_argument("--ar25-budget", type=int, default=80)
    parser.add_argument("--ar25-revise-every", type=int, default=20)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_multi_game_evaluation(
        traces_dir=args.traces_dir,
        trace_paths=_parse_paths(args.trace_path),
        max_games=args.max_games,
        include_ar25=not args.exclude_ar25,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        run_memory_comparison=not args.skip_memory_comparison,
        run_transfer_checks=not args.skip_transfer_checks,
        ar25_budget=args.ar25_budget,
        ar25_revise_every=args.ar25_revise_every,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
