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
    PosteriorConditionalFamilyFloorProgramPosterior,
    posterior_conditional_floor_policy,
)
from theory.sage_t.posterior_v8 import (
    T8_6G_POLICIES,
    MinimumKLFamilyFloorPolicy,
    MinimumKLFamilyFloorProgramPosterior,
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


def _programs() -> tuple[JointProgramHypothesis, ...]:
    return (
        _program("a1", "moved", goal_family="family_a"),
        _program("a2", "created", goal_family="family_a"),
        _program("b1", "removed", goal_family="family_b"),
    )


def _set_collapsed_mass(posterior: object) -> dict[str, float]:
    probabilities = (0.999999, 1e-7, 9e-7)
    posterior._particles = [
        replace(
            particle,
            log_weight=math.log(probability),
            log_joint=math.log(probability),
        )
        for particle, probability in zip(posterior.particles, probabilities)
    ]
    return {
        particle.program.canonical_hash: probability
        for particle, probability in zip(posterior.particles, probabilities)
    }


def _kl(posterior: object, original: dict[str, float]) -> float:
    return sum(
        particle.probability
        * math.log(
            particle.probability / original[particle.program.canonical_hash]
        )
        for particle in posterior.particles
    )


def test_t8_6g_has_one_minimum_kl_challenger() -> None:
    assert tuple(T8_6G_POLICIES) == (
        "legacy",
        "terminal_tempered_20",
        "terminal_tempered_20_family_floor_0501_minimum_kl",
    )
    challenger = T8_6G_POLICIES[
        "terminal_tempered_20_family_floor_0501_minimum_kl"
    ]
    assert challenger.entropy_floor == 0.0501
    assert challenger.maximum_family_total_variation == 0.02
    assert dict(challenger.channel_temperatures)["terminal"] == 0.20


def test_minimum_kl_projection_reaches_floor_within_tv_cap() -> None:
    policy = MinimumKLFamilyFloorPolicy.minimum_kl_challenger()
    posterior = MinimumKLFamilyFloorProgramPosterior(update_policy=policy)
    posterior.seed(_programs(), initial_state=_state())
    _set_collapsed_mass(posterior)

    projection = posterior._project_if_needed(raw_surprise=4.0)

    assert projection["triggered"]
    assert projection["floor_reached"]
    assert posterior.normalized_entropy >= policy.entropy_floor - 1e-10
    assert 0.0 < projection["family_total_variation"] <= 0.02
    assert 0.0 < projection["projection_alpha"] < 1.0
    assert projection["projection_kl"] > 0.0


def test_minimum_kl_projection_preserves_within_family_ratios() -> None:
    posterior = MinimumKLFamilyFloorProgramPosterior(
        update_policy=MinimumKLFamilyFloorPolicy.minimum_kl_challenger(),
    )
    posterior.seed(_programs(), initial_state=_state())
    _set_collapsed_mass(posterior)
    family_a = [
        particle
        for particle in posterior.particles
        if particle.program.semantic_family[1] == "family_a"
    ]
    before = family_a[0].probability / family_a[1].probability

    posterior._project_if_needed(raw_surprise=4.0)

    family_a = [
        particle
        for particle in posterior.particles
        if particle.program.semantic_family[1] == "family_a"
    ]
    after = family_a[0].probability / family_a[1].probability
    assert math.isclose(after, before, rel_tol=1e-12)


def test_minimum_kl_cost_is_no_worse_than_uniform_family_path() -> None:
    programs = _programs()
    minimum = MinimumKLFamilyFloorProgramPosterior(
        update_policy=MinimumKLFamilyFloorPolicy.minimum_kl_challenger(),
    )
    uniform = PosteriorConditionalFamilyFloorProgramPosterior(
        update_policy=posterior_conditional_floor_policy(),
    )
    minimum.seed(programs, initial_state=_state())
    uniform.seed(programs, initial_state=_state())
    original = _set_collapsed_mass(minimum)
    _set_collapsed_mass(uniform)

    minimum._project_if_needed(raw_surprise=4.0)
    uniform._project_if_needed(raw_surprise=4.0)

    assert minimum.normalized_entropy >= 0.0501 - 1e-10
    assert uniform.normalized_entropy >= 0.0501 - 1e-10
    assert _kl(minimum, original) <= _kl(uniform, original) + 1e-12


def test_minimum_kl_projection_is_not_applied_without_surprise() -> None:
    posterior = MinimumKLFamilyFloorProgramPosterior(
        update_policy=MinimumKLFamilyFloorPolicy.minimum_kl_challenger(),
    )
    posterior.seed(_programs(), initial_state=_state())
    _set_collapsed_mass(posterior)
    before = [particle.log_joint for particle in posterior.particles]

    projection = posterior._project_if_needed(raw_surprise=3.0)

    assert not projection["triggered"]
    assert [particle.log_joint for particle in posterior.particles] == before
