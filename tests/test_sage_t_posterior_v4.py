from __future__ import annotations

import math
from dataclasses import replace

import pytest

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionBinding,
    Effect,
    Expression,
    GoalRule,
    GroundFact,
    JointProgramHypothesis,
    ObjectSchema,
    ProgressRule,
    TerminalRule,
    TransitionRule,
)
from theory.sage_t.posterior_v3 import (
    ChannelCalibratedProgramPosterior,
    ChannelPosteriorUpdatePolicy,
)
from theory.sage_t.posterior_v4 import (
    FamilyDiversityPolicy,
    FamilyDiversityProgramPosterior,
)


def _state() -> AbstractState:
    return AbstractState(
        entities=(AbstractEntity("target", ("object", "target")),),
        true_facts=frozenset({GroundFact("exists", ("target",))}),
        false_facts=frozenset({GroundFact("game_over")}),
    )


def _program(program_id: str, effect_kind: str) -> JointProgramHypothesis:
    return JointProgramHypothesis(
        program_id=program_id,
        object_schema=ObjectSchema(("object", "target")),
        action_bindings=(ActionBinding("ACTION1", "apply", target_role="target"),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{program_id}",
                action_operator="apply",
                condition=Expression.constant(True),
                effects=(Effect("assert", predicate=effect_kind),),
            ),
        ),
        progress_rule=ProgressRule(Expression.fact(effect_kind)),
        terminal_rules=(TerminalRule(Expression.fact("game_over"), "game_over"),),
        goal_rule=GoalRule(Expression.fact(effect_kind), family=effect_kind),
    )


def _force_collapsed_mass(posterior: FamilyDiversityProgramPosterior) -> None:
    particles = list(posterior.particles)
    probabilities = (1.0 - 1e-9, 1e-9)
    posterior._particles = [
        replace(
            particle,
            log_weight=math.log(probability),
            log_joint=math.log(probability),
        )
        for particle, probability in zip(particles, probabilities)
    ]


def test_zero_family_mixture_is_equivalent_to_channel_posterior() -> None:
    programs = (_program("left", "moved"), _program("right", "created"))
    channel = ChannelCalibratedProgramPosterior(
        update_policy=ChannelPosteriorUpdatePolicy.terminal_tempered(),
    )
    family = FamilyDiversityProgramPosterior(
        update_policy=FamilyDiversityPolicy.terminal_tempered(),
    )
    channel.seed(programs, initial_state=_state())
    family.seed(programs, initial_state=_state())

    assert {
        item.program.canonical_hash: item.probability
        for item in family.particles
    } == pytest.approx(
        {
            item.program.canonical_hash: item.probability
            for item in channel.particles
        },
        abs=1e-15,
    )


def test_family_projection_reserves_mass_without_creating_support() -> None:
    posterior = FamilyDiversityProgramPosterior(
        update_policy=FamilyDiversityPolicy.terminal_tempered_family_mix(0.05),
    )
    posterior.seed(
        (_program("left", "moved"), _program("right", "created")),
        initial_state=_state(),
    )
    hashes_before = {item.program.canonical_hash for item in posterior.particles}
    _force_collapsed_mass(posterior)
    entropy_before = posterior.normalized_entropy

    projection = posterior._project_if_needed(raw_surprise=4.0)

    probabilities = sorted(item.probability for item in posterior.particles)
    assert projection["triggered"]
    assert probabilities[0] >= 0.025 - 1e-9
    assert posterior.normalized_entropy > entropy_before
    assert {item.program.canonical_hash for item in posterior.particles} == hashes_before
    assert math.isclose(sum(probabilities), 1.0)


def test_projection_is_not_applied_without_raw_surprise() -> None:
    posterior = FamilyDiversityProgramPosterior(
        update_policy=FamilyDiversityPolicy.terminal_tempered_family_mix(0.10),
    )
    posterior.seed(
        (_program("left", "moved"), _program("right", "created")),
        initial_state=_state(),
    )
    _force_collapsed_mass(posterior)
    before = [item.probability for item in posterior.particles]

    projection = posterior._project_if_needed(raw_surprise=3.0)

    assert not projection["triggered"]
    assert [item.probability for item in posterior.particles] == before


def test_family_policy_preserves_repair_v2_budgets() -> None:
    policy = FamilyDiversityPolicy.terminal_tempered_family_mix(
        0.05
    ).with_repair_v2()

    assert policy.incremental_repair
    assert policy.repair_parent_limit == 2
    assert policy.repair_child_limit == 8
    assert policy.repair_survivor_limit == 4
    assert policy.family_mixture == 0.05
