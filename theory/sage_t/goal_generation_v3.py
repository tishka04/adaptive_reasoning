"""T8.6i structural-role-conditioned goal generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .contracts import (
    Expression,
    ObjectSchema,
    ObservedTransition,
    ProgramFragment,
)
from .goal_generation_v2 import goal_progress_bridge_fragment
from .replay_gate import _programs_for as frozen_programs_for
from .structural_roles import (
    STRUCTURAL_TARGET_ROLES,
    action_target_structural_role,
)
from .synthesis import (
    AssembledProgram,
    DeterministicFragmentProposer,
    ProgramAssembler,
)


def observed_goal_trigger_roles(
    transitions: Sequence[ObservedTransition],
) -> dict[str, str]:
    """Map actions to a consistent relative role observed to trigger a goal."""

    candidates: dict[str, set[str]] = {}
    for transition in transitions:
        observation = transition.observation
        if not (
            observation.goal_probability is not None
            and observation.goal_probability >= 0.5
            and observation.progress_mean is not None
            and observation.progress_mean > 0.0
            and "level_complete" in transition.events
            and "progress" in transition.events
        ):
            continue
        role = action_target_structural_role(
            transition.state_before,
            transition.action,
        )
        if role:
            candidates.setdefault(transition.action.action_name, set()).add(role)
    return {
        action: next(iter(roles))
        for action, roles in candidates.items()
        if len(roles) == 1
    }


def structural_goal_guard_fragments(
    proposal_fragments: Sequence[ProgramFragment],
    trigger_roles: Mapping[str, str],
) -> tuple[ProgramFragment, ...]:
    """Create support-free guarded dynamics for observed trigger roles."""

    output = []
    for fragment in proposal_fragments:
        if fragment.kind != "dynamics":
            continue
        try:
            binding, rule = fragment.payload
        except (TypeError, ValueError):
            continue
        role = trigger_roles.get(getattr(binding, "action_name", ""), "")
        events = set(fragment.predicted_events)
        if not role or not {"level_complete", "progress"}.issubset(events):
            continue
        guarded_rule = replace(
            rule,
            rule_id=f"{rule.rule_id}_structural_guard",
            condition=Expression.fact("role", "$target", role),
        )
        output.append(
            ProgramFragment(
                fragment_id=f"{fragment.fragment_id}_guard_{role}",
                kind="dynamics",
                payload=(binding, guarded_rule),
                roles=tuple(sorted(set(fragment.roles) | {role})),
                predicted_events=fragment.predicted_events,
                provenance=tuple(
                    sorted(
                        set(fragment.provenance)
                        | {"sage_t_structural_goal_guard"}
                    )
                ),
                prior_logprob=float(fragment.prior_logprob) - 0.10,
                support=0,
            )
        )
    return tuple(output)


def programs_for_with_structural_goal_guard(
    available_actions: Sequence[str],
    transitions: Sequence[ObservedTransition],
    manifest: Mapping[str, Any],
) -> tuple[AssembledProgram, ...]:
    """Generate guarded goal programs after a structurally grounded signal."""

    trigger_roles = observed_goal_trigger_roles(transitions)
    if not trigger_roles:
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
    schema_fragments = []
    for fragment in proposal.fragments:
        if fragment.kind == "schema" and isinstance(fragment.payload, ObjectSchema):
            schema_fragments.append(
                replace(
                    fragment,
                    payload=ObjectSchema(
                        tuple(
                            sorted(
                                set(fragment.payload.roles)
                                | set(STRUCTURAL_TARGET_ROLES)
                            )
                        )
                    ),
                )
            )
        else:
            schema_fragments.append(fragment)
    guards = structural_goal_guard_fragments(
        schema_fragments,
        trigger_roles,
    )
    fragments = (
        *schema_fragments,
        *guards,
        goal_progress_bridge_fragment(),
    )
    return ProgramAssembler(
        maximum_programs=int(generator["maximum_programs"]),
        maximum_dynamics_beam=int(generator["maximum_dynamics_beam"]),
    ).assemble(
        fragments,
        available_actions=available_actions,
    )


__all__ = [
    "observed_goal_trigger_roles",
    "programs_for_with_structural_goal_guard",
    "structural_goal_guard_fragments",
]
