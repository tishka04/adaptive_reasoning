from __future__ import annotations

import math

import pytest

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GoalRule,
    GroundFact,
    JointProgramHypothesis,
    ObjectSchema,
    ObservedTransition,
    PredictionPacket,
    ProgressRule,
    TerminalRule,
    TransitionRule,
)
from theory.sage_t.posterior import packet_log_likelihood
from theory.sage_t.posterior_v2 import (
    CalibratedProgramPosterior,
    PosteriorUpdatePolicy,
)
from theory.sage_t.posterior_v3 import (
    ChannelCalibratedProgramPosterior,
    ChannelPosteriorUpdatePolicy,
    packet_channel_log_likelihoods,
)


def _state() -> AbstractState:
    return AbstractState(
        entities=(AbstractEntity("target", ("object", "target")),),
        true_facts=frozenset(
            {
                GroundFact("exists", ("target",)),
                GroundFact("role", ("target", "target")),
            }
        ),
        false_facts=frozenset(
            {
                GroundFact("solved", ("target",)),
                GroundFact("game_over"),
                GroundFact("level_complete"),
            }
        ),
    )


def _program(program_id: str, *, solves: bool) -> JointProgramHypothesis:
    return JointProgramHypothesis(
        program_id=program_id,
        object_schema=ObjectSchema(("object", "target")),
        action_bindings=(ActionBinding("ACTION1", "apply", target_role="target"),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{program_id}",
                action_operator="apply",
                condition=Expression.constant(True),
                effects=(
                    Effect(
                        "assert",
                        predicate="solved" if solves else "no_effect",
                        terms=("$target",) if solves else (),
                    ),
                ),
            ),
        ),
        progress_rule=ProgressRule(Expression.fact("solved", "target")),
        terminal_rules=(TerminalRule(Expression.fact("game_over"), "game_over"),),
        goal_rule=GoalRule(Expression.fact("solved", "target"), family="solve"),
    )


def _evidence(*, positive: bool) -> ObservedTransition:
    before = _state()
    after = (
        before.with_updates(asserted=(GroundFact("solved", ("target",)),))
        if positive
        else before
    )
    return ObservedTransition(
        state_before=before,
        action=ActionCandidate("ACTION1", {"entity_id": "target"}),
        state_after=after,
        observation=PredictionPacket(
            object_deltas={"solved": float(positive)},
            progress_mean=float(positive),
            progress_distribution={
                f"value:{int(positive)}": 1.0,
            },
            terminal_probability=0.0,
            goal_probability=float(positive),
            known_channels=frozenset(
                {"objects", "progress", "terminal", "goal"}
            ),
            state_after=after,
        ),
        events=("progress",) if positive else ("no_effect",),
    )


def test_channel_terms_sum_to_the_frozen_joint_likelihood() -> None:
    predicted = PredictionPacket(
        object_deltas={"moved": 0.95},
        progress_mean=1.0,
        progress_distribution={"value:1": 0.95, "other": 0.05},
        terminal_probability=0.05,
        goal_probability=0.95,
        known_channels=frozenset(
            {"objects", "progress", "terminal", "goal"}
        ),
    )
    observed = PredictionPacket(
        object_deltas={"moved": 1.0},
        progress_mean=1.0,
        progress_distribution={"value:1": 1.0},
        terminal_probability=0.0,
        goal_probability=1.0,
        known_channels=frozenset(
            {"objects", "progress", "terminal", "goal"}
        ),
    )

    channels = packet_channel_log_likelihoods(predicted, observed)

    assert sum(channels.values()) == pytest.approx(
        packet_log_likelihood(predicted, observed),
        abs=1e-15,
    )


def test_registered_channel_policies_are_scoped_as_pre_registered() -> None:
    terminal = ChannelPosteriorUpdatePolicy.terminal_tempered()
    teleology = ChannelPosteriorUpdatePolicy.teleology_tempered()
    correlated = ChannelPosteriorUpdatePolicy.teleology_correlation_aware()

    assert terminal.channel_multiplier("objects", 4) == 1.0
    assert terminal.channel_multiplier("terminal", 4) == 0.25
    assert teleology.channel_multiplier("progress", 4) == 0.25
    assert teleology.channel_multiplier("goal", 4) == 0.25
    assert correlated.channel_multiplier("objects", 4) == 1.0
    assert correlated.channel_multiplier("terminal", 4) == 0.125


def test_channel_legacy_is_bit_equivalent_to_t8_6_legacy() -> None:
    programs = (_program("signal", solves=True), _program("neutral", solves=False))
    baseline = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.legacy(),
        repair_ess_threshold=1.0,
    )
    channel = ChannelCalibratedProgramPosterior(
        update_policy=ChannelPosteriorUpdatePolicy.legacy(),
        repair_ess_threshold=1.0,
    )
    for posterior in (baseline, channel):
        posterior.seed(programs, initial_state=_state())
        posterior.observe(_evidence(positive=False), allow_repair=False)
        posterior.observe(_evidence(positive=True), allow_repair=False)

    baseline_mass = {
        item.program.canonical_hash: item.probability
        for item in baseline.particles
    }
    channel_mass = {
        item.program.canonical_hash: item.probability
        for item in channel.particles
    }
    assert channel_mass == pytest.approx(baseline_mass, abs=1e-15)
    assert channel.normalized_entropy == pytest.approx(
        baseline.normalized_entropy,
        abs=1e-15,
    )


def test_raw_surprise_remains_untempered_and_channels_are_auditable() -> None:
    posterior = ChannelCalibratedProgramPosterior(
        update_policy=ChannelPosteriorUpdatePolicy.teleology_tempered(),
        repair_ess_threshold=1.0,
    )
    posterior.seed(
        (_program("signal", solves=True), _program("neutral", solves=False)),
        initial_state=_state(),
    )

    diagnostics = posterior.observe(_evidence(positive=True), allow_repair=False)

    assert diagnostics is not None
    assert math.isfinite(diagnostics.raw_mixture_surprise)
    assert diagnostics.channel_multipliers["objects"] == 1.0
    assert diagnostics.channel_multipliers["terminal"] == 0.25
    assert set(diagnostics.raw_channel_log_likelihood_mean) == {
        "objects",
        "relations",
        "topology",
        "progress",
        "terminal",
        "goal",
    }
