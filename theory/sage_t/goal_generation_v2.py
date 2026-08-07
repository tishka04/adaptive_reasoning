"""T8.6h deterministic goal/progress bridge generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    Expression,
    GoalRule,
    ObservedTransition,
    ProgramFragment,
    ProgressRule,
    TerminalRule,
)
from .replay_gate import _programs_for as frozen_programs_for
from .synthesis import (
    AssembledProgram,
    DeterministicFragmentProposer,
    ProgramAssembler,
)


def goal_progress_bridge_fragment() -> ProgramFragment:
    """Return a support-free bundle coupling progress effects to completion."""

    game_over = TerminalRule(Expression.fact("game_over"), "game_over")
    win = TerminalRule(Expression.fact("level_complete"), "win")
    return ProgramFragment(
        fragment_id="goal_level_completion_progress_counter_bridge",
        kind="goal_bundle",
        payload=(
            ProgressRule(Expression(op="counter", value="progress")),
            GoalRule(
                Expression.fact("level_complete"),
                family="level_completion",
            ),
            (game_over, win),
        ),
        roles=("player", "target"),
        predicted_events=("level_complete", "progress"),
        provenance=("sage_t_deterministic_goal_progress_bridge",),
        prior_logprob=-0.05,
        support=0,
    )


def needs_goal_progress_bridge(
    transitions: Sequence[ObservedTransition],
) -> bool:
    """Activate only after an observed positive goal and progress transition."""

    for transition in transitions:
        observation = transition.observation
        if (
            observation.goal_probability is not None
            and observation.goal_probability >= 0.5
            and observation.progress_mean is not None
            and observation.progress_mean > 0.0
            and "level_complete" in transition.events
            and "progress" in transition.events
        ):
            return True
    return False


def programs_for_with_goal_progress_bridge(
    available_actions: Sequence[str],
    transitions: Sequence[ObservedTransition],
    manifest: Mapping[str, Any],
) -> tuple[AssembledProgram, ...]:
    """Generate the frozen grammar plus one observed-signal goal bundle."""

    if not needs_goal_progress_bridge(transitions):
        return frozen_programs_for(available_actions, transitions, manifest)
    generator = manifest["generator"]
    proposal = DeterministicFragmentProposer(
        maximum_operator_candidates_per_action=int(
            generator["maximum_operator_candidates_per_action"]
        )
    ).propose(
        available_actions=available_actions,
        transitions=transitions,
    )
    fragments = (*proposal.fragments, goal_progress_bridge_fragment())
    return ProgramAssembler(
        maximum_programs=int(generator["maximum_programs"]),
        maximum_dynamics_beam=int(generator["maximum_dynamics_beam"]),
    ).assemble(
        fragments,
        available_actions=available_actions,
    )


__all__ = [
    "goal_progress_bridge_fragment",
    "needs_goal_progress_bridge",
    "programs_for_with_goal_progress_bridge",
]
