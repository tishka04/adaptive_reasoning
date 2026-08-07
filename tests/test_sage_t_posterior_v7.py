from __future__ import annotations

import math
from dataclasses import replace

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
from theory.sage_t.posterior_v7 import (
    T8_6F_POLICIES,
    PosteriorConditionalFamilyFloorProgramPosterior,
    posterior_conditional_floor_policy,
)


def _state() -> AbstractState:
    return AbstractState(
        entities=(AbstractEntity("target", ("object", "target")),),
        true_facts=frozenset({GroundFact("exists", ("target",))}),
        false_facts=frozenset({GroundFact("game_over")}),
    )


def _program(
    program_id: str,
    effect_kind: str,
    *,
    goal_family: str,
) -> JointProgramHypothesis:
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
        goal_rule=GoalRule(Expression.fact(effect_kind), family=goal_family),
    )


def _posterior() -> PosteriorConditionalFamilyFloorProgramPosterior:
    posterior = PosteriorConditionalFamilyFloorProgramPosterior(
        update_policy=posterior_conditional_floor_policy(),
    )
    posterior.seed(
        (
            _program("a1", "moved", goal_family="family_a"),
            _program("a2", "created", goal_family="family_a"),
            _program("b1", "removed", goal_family="family_b"),
        ),
        initial_state=_state(),
    )
    probabilities = (0.999999, 1e-7, 9e-7)
    posterior._particles = [
        replace(
            particle,
            log_weight=math.log(probability),
            log_joint=math.log(probability),
        )
        for particle, probability in zip(posterior.particles, probabilities)
    ]
    return posterior


def test_t8_6f_has_one_pre_registered_challenger() -> None:
    assert tuple(T8_6F_POLICIES) == (
        "legacy",
        "terminal_tempered_20",
        "terminal_tempered_20_family_floor_0501_posterior_conditional",
    )
    challenger = T8_6F_POLICIES[
        "terminal_tempered_20_family_floor_0501_posterior_conditional"
    ]
    assert challenger.entropy_floor == 0.0501
    assert challenger.maximum_family_mixture == 0.02
    assert dict(challenger.channel_temperatures)["terminal"] == 0.20


def test_reference_is_uniform_across_families_and_posterior_inside_family() -> None:
    posterior = _posterior()
    reference = posterior._family_reference_probabilities()
    particles = list(posterior.particles)
    family_a = [
        particle
        for particle in particles
        if particle.program.semantic_family[1] == "family_a"
    ]
    family_b = [
        particle
        for particle in particles
        if particle.program.semantic_family[1] == "family_b"
    ]

    mass_a = sum(reference[p.program.canonical_hash] for p in family_a)
    mass_b = sum(reference[p.program.canonical_hash] for p in family_b)
    original_ratio = family_a[0].probability / family_a[1].probability
    reference_ratio = (
        reference[family_a[0].program.canonical_hash]
        / reference[family_a[1].program.canonical_hash]
    )

    assert math.isclose(mass_a, 0.5)
    assert math.isclose(mass_b, 0.5)
    assert math.isclose(reference_ratio, original_ratio)


def test_projection_preserves_within_family_ratios_exactly() -> None:
    posterior = _posterior()
    family_a = [
        particle
        for particle in posterior.particles
        if particle.program.semantic_family[1] == "family_a"
    ]
    ratio_before = family_a[0].probability / family_a[1].probability

    projection = posterior._project_if_needed(raw_surprise=4.0)

    family_a_after = [
        particle
        for particle in posterior.particles
        if particle.program.semantic_family[1] == "family_a"
    ]
    ratio_after = family_a_after[0].probability / family_a_after[1].probability
    assert projection["triggered"]
    assert projection["floor_reached"]
    assert math.isclose(ratio_after, ratio_before, rel_tol=1e-12)


def test_repair_v2_keeps_posterior_conditional_policy() -> None:
    repair = posterior_conditional_floor_policy().with_repair_v2()

    assert repair.incremental_repair
    assert repair.entropy_floor == 0.0501
    assert repair.name.endswith("_repair_v2")
