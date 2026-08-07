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
from theory.sage_t.posterior_v3 import ChannelCalibratedProgramPosterior
from theory.sage_t.posterior_v5 import terminal_temperature_policy
from theory.sage_t.posterior_v6 import (
    T8_6E_POLICIES,
    AdaptiveFamilyFloorPolicy,
    AdaptiveFamilyFloorProgramPosterior,
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


def _programs() -> tuple[JointProgramHypothesis, ...]:
    return (_program("left", "moved"), _program("right", "created"))


def _force_collapsed_mass(posterior: AdaptiveFamilyFloorProgramPosterior) -> None:
    probabilities = (1.0 - 1e-9, 1e-9)
    posterior._particles = [
        replace(
            particle,
            log_weight=math.log(probability),
            log_joint=math.log(probability),
        )
        for particle, probability in zip(posterior.particles, probabilities)
    ]


def test_registered_policies_change_only_floor_after_terminal_20() -> None:
    assert tuple(T8_6E_POLICIES) == (
        "legacy",
        "terminal_tempered_20",
        "terminal_tempered_20_family_floor_0501",
        "terminal_tempered_20_family_floor_0525",
        "terminal_tempered_20_family_floor_0550",
    )
    for name, policy in T8_6E_POLICIES.items():
        temperatures = dict(policy.channel_temperatures)
        if name == "legacy":
            assert set(temperatures.values()) == {1.0}
        else:
            assert temperatures["terminal"] == 0.20
            assert all(
                value == 1.0
                for channel, value in temperatures.items()
                if channel != "terminal"
            )
        assert policy.maximum_family_mixture == 0.02


def test_zero_floor_is_bit_equivalent_to_terminal_20() -> None:
    programs = _programs()
    control = ChannelCalibratedProgramPosterior(
        update_policy=terminal_temperature_policy(0.20),
    )
    adaptive = AdaptiveFamilyFloorProgramPosterior(
        update_policy=AdaptiveFamilyFloorPolicy.terminal_tempered_20(),
    )
    control.seed(programs, initial_state=_state())
    adaptive.seed(programs, initial_state=_state())

    assert [particle.log_weight for particle in adaptive.particles] == [
        particle.log_weight for particle in control.particles
    ]
    assert [particle.log_joint for particle in adaptive.particles] == [
        particle.log_joint for particle in control.particles
    ]


def test_projection_uses_minimum_mixture_to_reach_floor() -> None:
    policy = AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.0501)
    posterior = AdaptiveFamilyFloorProgramPosterior(update_policy=policy)
    posterior.seed(_programs(), initial_state=_state())
    hashes = {particle.program.canonical_hash for particle in posterior.particles}
    _force_collapsed_mass(posterior)

    projection = posterior._project_if_needed(raw_surprise=4.0)

    assert projection["triggered"]
    assert projection["floor_reached"]
    assert posterior.normalized_entropy >= policy.entropy_floor - 1e-10
    assert 0.0 < projection["applied_mixture"] < 0.02
    assert {particle.program.canonical_hash for particle in posterior.particles} == hashes
    assert math.isclose(sum(p.probability for p in posterior.particles), 1.0)


def test_projection_is_not_applied_outside_registered_collapse() -> None:
    policy = AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.055)
    posterior = AdaptiveFamilyFloorProgramPosterior(update_policy=policy)
    posterior.seed(_programs(), initial_state=_state())
    _force_collapsed_mass(posterior)
    before = [particle.log_joint for particle in posterior.particles]

    projection = posterior._project_if_needed(raw_surprise=3.0)

    assert not projection["triggered"]
    assert projection["applied_mixture"] == 0.0
    assert [particle.log_joint for particle in posterior.particles] == before


def test_projection_is_deterministic_and_repair_budgets_are_preserved() -> None:
    policy = AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.0525)
    mixtures = []
    for _ in range(2):
        posterior = AdaptiveFamilyFloorProgramPosterior(update_policy=policy)
        posterior.seed(_programs(), initial_state=_state())
        _force_collapsed_mass(posterior)
        mixtures.append(
            posterior._project_if_needed(raw_surprise=4.0)["applied_mixture"]
        )

    repair = policy.with_repair_v2()
    assert mixtures[0] == mixtures[1]
    assert repair.incremental_repair
    assert repair.repair_parent_limit == 2
    assert repair.repair_child_limit == 8
    assert repair.repair_survivor_limit == 4
    assert repair.entropy_floor == pytest.approx(0.0525)


def test_invalid_adaptive_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceed collapse"):
        AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.05)
