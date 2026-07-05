"""A7b active ar25 micro-run for live option execution.

This module is an integration harness, not a planner. It may run a short,
explicit probe cycle to reach a state where the already-confirmed option is
applicable, but the option itself still executes only because its initiation
predicate is true.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from .ar25_oracle import build_ar25_oracle
from .correspondence_hypothesis import (
    CorrespondenceHypothesis,
    CorrespondenceObservation,
    correspondence_key,
)
from .correspondence_preparation_policy import PrepareCorrespondencePolicy
from .contextual_readiness import (
    ContextualReadinessDiscriminator,
    ReadinessAssessment,
)
from .epistemic_metrics import EpistemicScore
from .live_transition_loop import LiveTransitionBeliefLoop
from .mechanic_hypothesis import GameTheory
from .precondition_hypothesis import (
    PreconditionHypothesis,
    extract_precondition_predicates,
)
from .prepare_correspondence_option import (
    PrepareCorrespondenceOption,
    PrepareCorrespondenceOptionRunner,
)
from .real_env_option_adapter import TheoryOptionRunner, snapshot_frame
from .strong_preparation_policy import StrongPrepareCorrespondencePolicy
from .theory_option import build_options_from_theory

DEFAULT_GAME_ID = "ar25-e3c63847"
DEFAULT_PAIR_COLORS = (10, 11)
DEFAULT_BOOTSTRAP_PROBES = ("ACTION4", "ACTION4", "ACTION5", "ACTION2")


@dataclass(frozen=True)
class LiveMicroRunEvent:
    """One active env action made during the micro-run."""

    step: int
    kind: str
    action: str
    levels_before: int = 0
    levels_after: int = 0
    predicates_present: List[str] = field(default_factory=list)
    predicates_after: List[str] = field(default_factory=list)
    termination: str = ""
    reason: str = ""
    role: str = ""
    target_predicate: str = ""
    context_key: str = ""
    readiness_status: str = ""


@dataclass
class LiveOptionMicroRunResult:
    """A7b integration result."""

    game_id: str
    transitions: int = 0
    env_actions: int = 0
    option_attempts: int = 0
    option_invocations: int = 0
    option_successes: int = 0
    option_contradictions: int = 0
    prepare_option_invocations: int = 0
    prepare_option_successes: int = 0
    prepare_option_contradictions: int = 0
    prepare_option_max_steps: int = 0
    full_chain_successes: int = 0
    preparation_attempts: int = 0
    strong_preparation_attempts: int = 0
    ready_reached_by_prepare_policy: bool = False
    strong_ready_reached_by_agent: bool = False
    ready_observed: bool = False
    strong_ready_observed: bool = False
    option_attempts_from_strong_ready: int = 0
    contextual_blocks: int = 0
    contextual_refutations: int = 0
    contextual_hypotheses: List[dict[str, object]] = field(default_factory=list)
    trace_dependent: bool = False
    use_prepare_policy: bool = False
    use_contextual_readiness: bool = False
    use_strong_preparation: bool = False
    use_prepare_option: bool = False
    bootstrap_probe_actions: List[str] = field(default_factory=list)
    events: List[LiveMicroRunEvent] = field(default_factory=list)
    score: EpistemicScore | None = None
    error: str = ""

    @property
    def wrong_confirmations(self) -> int:
        return 0 if self.score is None else self.score.wrong_confirmations

    @property
    def confirmation_precision(self) -> float:
        return 0.0 if self.score is None else self.score.confirmation_precision

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "transitions": self.transitions,
            "env_actions": self.env_actions,
            "option_attempts": self.option_attempts,
            "option_invocations": self.option_invocations,
            "option_successes": self.option_successes,
            "option_contradictions": self.option_contradictions,
            "prepare_option_invocations": self.prepare_option_invocations,
            "prepare_option_successes": self.prepare_option_successes,
            "prepare_option_contradictions": self.prepare_option_contradictions,
            "prepare_option_max_steps": self.prepare_option_max_steps,
            "validate_option_successes": self.option_successes,
            "full_chain_successes": self.full_chain_successes,
            "preparation_attempts": self.preparation_attempts,
            "strong_preparation_attempts": self.strong_preparation_attempts,
            "ready_reached_by_prepare_policy": self.ready_reached_by_prepare_policy,
            "strong_ready_reached_by_agent": self.strong_ready_reached_by_agent,
            "ready_observed": self.ready_observed,
            "strong_ready_observed": self.strong_ready_observed,
            "option_attempts_from_strong_ready": (
                self.option_attempts_from_strong_ready
            ),
            "contextual_blocks": self.contextual_blocks,
            "contextual_refutations": self.contextual_refutations,
            "contextual_hypotheses": list(self.contextual_hypotheses),
            "trace_dependent": self.trace_dependent,
            "use_prepare_policy": self.use_prepare_policy,
            "use_contextual_readiness": self.use_contextual_readiness,
            "use_strong_preparation": self.use_strong_preparation,
            "use_prepare_option": self.use_prepare_option,
            "bootstrap_probe_actions": list(self.bootstrap_probe_actions),
            "wrong_confirmations": self.wrong_confirmations,
            "confirmation_precision": round(self.confirmation_precision, 4),
            "score": self.score.to_dict() if self.score is not None else None,
            "events": [event.__dict__ for event in self.events],
            "error": self.error,
        }


def run_ar25_live_option_micro_run(
    *,
    game_id: str = DEFAULT_GAME_ID,
    environments_dir: Path | str | None = None,
    max_actions: int = 50,
    max_option_attempts: int = 1,
    bootstrap_probe_actions: Sequence[str] = DEFAULT_BOOTSTRAP_PROBES,
    use_prepare_policy: bool = False,
    use_contextual_readiness: bool = False,
    use_strong_preparation: bool = False,
    use_prepare_option: bool = False,
    background_value: int | None = 9,
) -> LiveOptionMicroRunResult:
    """Run a short active offline ar25 option smoke test."""
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    probes = [_normalize_action_name(action) for action in bootstrap_probe_actions]
    result = LiveOptionMicroRunResult(
        game_id=game_id,
        use_prepare_policy=use_prepare_policy,
        use_contextual_readiness=use_contextual_readiness,
        use_strong_preparation=use_strong_preparation,
        use_prepare_option=use_prepare_option,
        bootstrap_probe_actions=list(probes),
    )

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

    theory = _bootstrap_ar25_option_theory(game_id)
    reset_snapshot = snapshot_frame(current_frame)
    loop = LiveTransitionBeliefLoop(
        game_id,
        theory=theory,
        available_actions=reset_snapshot.available_actions,
        background_value=background_value,
        infer_players=False,
        correspondence_pair_colors=DEFAULT_PAIR_COLORS,
    )
    options = build_options_from_theory(loop.theory)
    if not options:
        result.error = "no_confirmed_option_available"
        return result

    option = options[0]
    runner = TheoryOptionRunner(loop, option)
    prepare_policy = (
        PrepareCorrespondencePolicy()
        if use_prepare_policy and not use_prepare_option
        else None
    )
    strong_policy = (
        StrongPrepareCorrespondencePolicy()
        if use_strong_preparation and not use_prepare_option
        else None
    )
    prepare_option_runner = (
        PrepareCorrespondenceOptionRunner(
            loop,
            PrepareCorrespondenceOption.from_validation_option(option),
        )
        if use_prepare_option
        else None
    )
    readiness = (
        ContextualReadinessDiscriminator()
        if use_contextual_readiness
        else None
    )
    target_rule = option.target_rule
    previous_action = ""
    recent_actions: List[str] = []
    recent_successes: List[bool] = []
    probe_index = 0
    pending_successful_prepare_option = False

    while result.env_actions < max_actions:
        snapshot = snapshot_frame(current_frame)
        predicates = extract_precondition_predicates(
            snapshot.grid,
            target_rule=target_rule,
            pair_colors=option.pair_colors,
            previous_action=previous_action,
            recent_actions=recent_actions,
            recent_correspondence_successes=recent_successes,
        )
        assessment: ReadinessAssessment | None = None
        if readiness is not None:
            assessment = readiness.assess(predicates, target_rule=target_rule)
            predicates = set(assessment.predicates)
        result.ready_observed = result.ready_observed or (
            "ready_to_validate_correspondence" in predicates
        )
        result.strong_ready_observed = result.strong_ready_observed or (
            "strong_ready_to_validate_correspondence" in predicates
        )

        can_initiate = option.can_initiate(
            loop.theory,
            predicates,
            available_actions=snapshot.available_actions,
        )
        should_attempt_option = can_initiate
        if strong_policy is not None or prepare_option_runner is not None:
            if assessment is not None:
                should_attempt_option = can_initiate and assessment.strong_ready
            else:
                should_attempt_option = (
                    can_initiate
                    and "source_target_relation_satisfied" in predicates
                )

        if should_attempt_option:
            if (
                readiness is not None
                and assessment is not None
                and not readiness.can_attempt_validation(assessment)
            ):
                result.contextual_blocks += 1
                result.events.append(LiveMicroRunEvent(
                    step=result.env_actions,
                    kind="readiness_block",
                    action=option.policy_action,
                    levels_before=snapshot.levels_completed,
                    levels_after=snapshot.levels_completed,
                    predicates_present=sorted(predicates),
                    reason="contextual_readiness_refuted_same_context",
                    context_key=assessment.context.key,
                    readiness_status=assessment.hypothesis.status.value,
                ))
                break
            attempt = runner.run_once(
                env,
                current_frame,
                previous_action=previous_action,
                recent_actions=recent_actions,
                recent_correspondence_successes=recent_successes,
                step=result.env_actions,
            )
            result.option_attempts += 1
            if assessment is not None and assessment.strong_ready:
                result.option_attempts_from_strong_ready += 1
            if attempt.action_executed:
                result.env_actions += 1
            if attempt.invocation is not None:
                result.option_invocations += 1
                if attempt.invocation.success:
                    result.option_successes += 1
                    if pending_successful_prepare_option:
                        result.full_chain_successes += 1
                if attempt.invocation.contradiction:
                    result.option_contradictions += 1
            pending_successful_prepare_option = False
            contextual = None
            if readiness is not None and assessment is not None:
                contextual = readiness.observe_validation_result(
                    assessment,
                    attempt.invocation,
                )
                result.contextual_refutations = len(readiness.refuted())
                result.contextual_hypotheses = [
                    hypothesis.to_dict()
                    for hypothesis in readiness.hypotheses()
                ]
            result.events.append(LiveMicroRunEvent(
                step=result.env_actions,
                kind="option_attempt",
                action=option.policy_action,
                levels_before=snapshot.levels_completed,
                levels_after=_levels_after(attempt.transition_update),
                predicates_present=sorted(predicates),
                termination=attempt.termination,
                reason=attempt.reason,
                context_key=assessment.context.key if assessment is not None else "",
                readiness_status=(
                    contextual.status.value
                    if contextual is not None
                    else ""
                ),
            ))
            if result.option_attempts >= max_option_attempts:
                break
            if attempt.next_frame is not None:
                current_frame = attempt.next_frame
            previous_action = option.policy_action
            recent_actions = (recent_actions + [previous_action])[-4:]
            recent_successes.append(bool(attempt.success))
            recent_successes = recent_successes[-4:]
            continue

        if prepare_option_runner is not None:
            preparation = prepare_option_runner.run_once(
                env,
                current_frame,
                previous_action=previous_action,
                recent_actions=recent_actions,
                recent_correspondence_successes=recent_successes,
                step=result.env_actions,
            )
            if preparation.invocation is None:
                result.error = "prepare_option_not_initiated"
                break
            result.prepare_option_invocations += 1
            if preparation.success:
                result.prepare_option_successes += 1
                pending_successful_prepare_option = True
            if preparation.contradiction:
                result.prepare_option_contradictions += 1
            if preparation.maxed:
                result.prepare_option_max_steps += 1
            result.env_actions += preparation.env_actions
            result.strong_preparation_attempts += preparation.env_actions
            result.preparation_attempts += preparation.env_actions
            result.strong_ready_reached_by_agent = (
                result.strong_ready_reached_by_agent
                or preparation.success
            )
            result.ready_reached_by_prepare_policy = (
                result.ready_reached_by_prepare_policy
                or preparation.success
            )
            if preparation.next_frame is not None:
                current_frame = preparation.next_frame
            previous_action = preparation.previous_action
            recent_actions = list(preparation.recent_actions)
            recent_successes = list(preparation.recent_correspondence_successes)
            invocation = preparation.invocation
            attempted_predicates = sorted(
                {
                    step_result.plan.target_predicate
                    for step_result in preparation.steps
                }
            )
            result.events.append(LiveMicroRunEvent(
                step=result.env_actions,
                kind="prepare_option",
                action=",".join(invocation.actions),
                levels_before=snapshot.levels_completed,
                levels_after=snapshot_frame(
                    current_frame,
                    fallback_available_actions=snapshot.available_actions,
                ).levels_completed,
                predicates_present=invocation.predicates_initial,
                predicates_after=invocation.predicates_final,
                termination=invocation.termination,
                reason=invocation.reason,
                role="prepare_correspondence",
                target_predicate=",".join(attempted_predicates),
            ))
            if preparation.contradiction or preparation.maxed:
                break
            continue

        if strong_policy is not None:
            preparation = strong_policy.prepare_once(
                env=env,
                loop=loop,
                option=option,
                current_frame=current_frame,
                previous_action=previous_action,
                recent_actions=recent_actions,
                recent_correspondence_successes=recent_successes,
                step=result.env_actions,
            )
            if preparation is None:
                result.error = "strong_prepare_policy_no_applicable_hypothesis"
                break
            corr = CorrespondenceObservation.from_transition(
                preparation.update.record,
                pair_colors=option.pair_colors,
            )
            recent_successes.append(
                bool(corr.improves or preparation.update.effect.level_complete)
            )
            recent_successes = recent_successes[-4:]
            previous_action = preparation.action
            recent_actions.append(preparation.action)
            recent_actions = recent_actions[-4:]
            current_frame = preparation.next_frame
            result.env_actions += 1
            result.strong_preparation_attempts += 1
            result.preparation_attempts += 1
            result.strong_ready_reached_by_agent = (
                result.strong_ready_reached_by_agent
                or preparation.strong_ready_after
            )
            result.ready_reached_by_prepare_policy = (
                result.ready_reached_by_prepare_policy
                or preparation.strong_ready_after
            )
            result.events.append(LiveMicroRunEvent(
                step=result.env_actions,
                kind="strong_prepare_policy",
                action=preparation.action,
                levels_before=snapshot.levels_completed,
                levels_after=int(
                    getattr(
                        preparation.update.record.obs_after,
                        "levels_completed",
                        0,
                    )
                    or 0
                ),
                predicates_present=preparation.predicates_before,
                predicates_after=preparation.predicates_after,
                reason=preparation.plan.reason,
                role=preparation.plan.role,
                target_predicate=preparation.plan.target_predicate,
            ))
            continue

        if prepare_policy is not None:
            preparation = prepare_policy.prepare_once(
                env=env,
                loop=loop,
                option=option,
                current_frame=current_frame,
                previous_action=previous_action,
                recent_actions=recent_actions,
                recent_correspondence_successes=recent_successes,
                step=result.env_actions,
            )
            if preparation is None:
                result.error = "prepare_policy_no_applicable_hypothesis"
                break
            corr = CorrespondenceObservation.from_transition(
                preparation.update.record,
                pair_colors=option.pair_colors,
            )
            recent_successes.append(
                bool(corr.improves or preparation.update.effect.level_complete)
            )
            recent_successes = recent_successes[-4:]
            previous_action = preparation.action
            recent_actions.append(preparation.action)
            recent_actions = recent_actions[-4:]
            current_frame = preparation.next_frame
            result.env_actions += 1
            result.preparation_attempts += 1
            result.ready_reached_by_prepare_policy = (
                result.ready_reached_by_prepare_policy
                or preparation.ready_after
            )
            result.events.append(LiveMicroRunEvent(
                step=result.env_actions,
                kind="prepare_policy",
                action=preparation.action,
                levels_before=snapshot.levels_completed,
                levels_after=int(
                    getattr(
                        preparation.update.record.obs_after,
                        "levels_completed",
                        0,
                    )
                    or 0
                ),
                predicates_present=preparation.predicates_before,
                predicates_after=preparation.predicates_after,
                reason=preparation.plan.reason,
                role=preparation.plan.role,
                target_predicate=preparation.plan.target_predicate,
            ))
            continue

        if not probes:
            result.error = "ready_not_observed_no_probe_actions"
            break
        probe_action = _next_available_probe(
            probes,
            probe_index,
            snapshot.available_actions,
        )
        probe_index += 1
        if not probe_action:
            result.error = "no_probe_action_available"
            break
        try:
            next_frame = env.step(_to_game_action(probe_action))
        except Exception as exc:  # pragma: no cover - integration failure path
            result.error = f"env_step_failed:{probe_action}: {exc}"
            break
        if next_frame is None:
            result.error = f"env_step_returned_none:{probe_action}"
            break

        next_snapshot = snapshot_frame(
            next_frame,
            fallback_available_actions=snapshot.available_actions,
        )
        update = loop.observe_grids(
            action=probe_action,
            grid_before=snapshot.grid,
            grid_after=next_snapshot.grid,
            available_actions=snapshot.available_actions,
            game_state_before=snapshot.game_state,
            game_state_after=next_snapshot.game_state,
            levels_completed_before=snapshot.levels_completed,
            levels_completed_after=next_snapshot.levels_completed,
            timestamp=result.env_actions,
            was_experiment=True,
        )
        corr = CorrespondenceObservation.from_transition(
            update.record,
            pair_colors=option.pair_colors,
        )
        recent_successes.append(bool(corr.improves or update.effect.level_complete))
        recent_successes = recent_successes[-4:]
        previous_action = probe_action
        recent_actions.append(probe_action)
        recent_actions = recent_actions[-4:]
        current_frame = next_frame
        result.env_actions += 1
        result.events.append(LiveMicroRunEvent(
            step=result.env_actions,
            kind="bootstrap_probe",
            action=probe_action,
            levels_before=snapshot.levels_completed,
            levels_after=next_snapshot.levels_completed,
            predicates_present=sorted(predicates),
            reason="external_probe_not_option_policy",
        ))

    result.transitions = loop.transition_count
    if readiness is not None:
        result.contextual_refutations = len(readiness.refuted())
        result.contextual_hypotheses = [
            hypothesis.to_dict()
            for hypothesis in readiness.hypotheses()
        ]
    result.score = loop.score(
        build_ar25_oracle(),
        experiment_actions=max(1, loop.transition_count),
    )
    return result


def _bootstrap_ar25_option_theory(game_id: str) -> GameTheory:
    """Seed only the A6-confirmed option facts needed for A7b."""
    target_rule = correspondence_key("ACTION2", "validates", DEFAULT_PAIR_COLORS)
    theory = GameTheory(game_id)
    theory.seed_actions([f"ACTION{idx}" for idx in range(1, 8)])
    theory.add_semantic_hypotheses(
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION2",
                relation="validates",
                pair_colors=DEFAULT_PAIR_COLORS,
            )
        ],
        preconditions=[
            PreconditionHypothesis(
                target_rule=target_rule,
                predicate="ready_to_validate_correspondence",
                evidence_for=["a6_support_1", "a6_support_2", "a6_support_3"],
            )
        ],
    )
    for _ in range(2):
        theory.observe_correspondence(
            CorrespondenceObservation(
                action="ACTION2",
                pair_colors=DEFAULT_PAIR_COLORS,
                level_complete=True,
            ),
            was_experiment=True,
        )
    return theory


def _env_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "environment_files"


def _configure_offline_env(environments_dir: Path) -> None:
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = str(environments_dir)
    os.environ.setdefault("ARC_API_KEY", "test")
    os.environ.setdefault("RECORDINGS_DIR", "recordings")
    os.environ["TQDM_DISABLE"] = "1"


def _next_available_probe(
    probes: Sequence[str],
    start_index: int,
    available_actions: Sequence[str],
) -> str:
    available = set(available_actions)
    for offset in range(len(probes)):
        candidate = probes[(start_index + offset) % len(probes)]
        if candidate in available:
            return candidate
    return ""


def _to_game_action(action_name: str) -> Any:
    from arcengine import GameAction

    if hasattr(GameAction, "from_name"):
        return GameAction.from_name(action_name)
    return getattr(GameAction, action_name)


def _levels_after(update: Any) -> int:
    if update is None:
        return 0
    return int(getattr(update.record.obs_after, "levels_completed", 0) or 0)


def _normalize_action_name(action: Any) -> str:
    raw = str(action or "").strip().upper()
    if raw.isdigit():
        return "RESET" if int(raw) == 0 else f"ACTION{raw}"
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw


def _parse_probe_actions(raw: str) -> tuple[str, ...]:
    return tuple(
        _normalize_action_name(item)
        for item in raw.split(",")
        if item.strip()
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A7b active ar25 TheoryOptionRunner smoke test."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--max-option-attempts", type=int, default=1)
    parser.add_argument(
        "--environments-dir",
        type=Path,
        default=_env_dir(),
    )
    parser.add_argument(
        "--bootstrap-probes",
        default=",".join(DEFAULT_BOOTSTRAP_PROBES),
        help="Comma-separated external probe cycle; not an option policy.",
    )
    parser.add_argument(
        "--prepare-policy",
        action="store_true",
        help="Use the A8 symbolic preparation policy instead of probes.",
    )
    parser.add_argument(
        "--contextual-readiness",
        action="store_true",
        help="Use A9 contextual strong/weak readiness discrimination.",
    )
    parser.add_argument(
        "--strong-preparation",
        action="store_true",
        help="Use A10 discriminating predicates before option invocation.",
    )
    parser.add_argument(
        "--prepare-option",
        action="store_true",
        help="Use A11 PrepareCorrespondenceOption before validation.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_ar25_live_option_micro_run(
        game_id=args.game_id,
        environments_dir=args.environments_dir,
        max_actions=args.max_actions,
        max_option_attempts=args.max_option_attempts,
        bootstrap_probe_actions=_parse_probe_actions(args.bootstrap_probes),
        use_prepare_policy=bool(args.prepare_policy),
        use_contextual_readiness=bool(args.contextual_readiness),
        use_strong_preparation=bool(args.strong_preparation),
        use_prepare_option=bool(args.prepare_option),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
