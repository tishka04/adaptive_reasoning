from __future__ import annotations

import math
from dataclasses import replace

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
from theory.sage_t.posterior import ProgramPosterior
from theory.sage_t.posterior_v2 import (
    CalibratedProgramParticle,
    CalibratedProgramPosterior,
    PosteriorUpdatePolicy,
)
from theory.sage_t.synthesis import AssembledProgram


def _state(*, regime_index: int = 0) -> AbstractState:
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
        regime_index=regime_index,
    )


def _program(program_id: str, *, solves: bool) -> JointProgramHypothesis:
    effect = (
        Effect("assert", predicate="solved", terms=("$target",))
        if solves
        else Effect("assert", predicate="no_effect")
    )
    return JointProgramHypothesis(
        program_id=program_id,
        object_schema=ObjectSchema(("object", "target")),
        action_bindings=(ActionBinding("ACTION1", "apply", target_role="target"),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{program_id}",
                action_operator="apply",
                condition=Expression.constant(True),
                effects=(effect,),
            ),
        ),
        progress_rule=ProgressRule(Expression.fact("solved", "target")),
        terminal_rules=(TerminalRule(Expression.fact("game_over"), "game_over"),),
        goal_rule=GoalRule(Expression.fact("solved", "target"), family="solve"),
    )


def _evidence(*, regime_index: int = 0, reset: bool = False) -> ObservedTransition:
    before = _state(regime_index=regime_index)
    after = before.with_updates(asserted=(GroundFact("solved", ("target",)),))
    return ObservedTransition(
        state_before=before,
        action=ActionCandidate("ACTION1", {"entity_id": "target"}),
        state_after=after,
        observation=PredictionPacket(
            progress_mean=1.0,
            terminal_probability=0.0,
            goal_probability=1.0,
            known_channels=frozenset({"objects", "progress", "terminal", "goal"}),
            state_after=after,
        ),
        events=("progress",),
        reset=reset,
    )


def _negative_evidence() -> ObservedTransition:
    state = _state()
    return ObservedTransition(
        state_before=state,
        action=ActionCandidate("ACTION1", {"entity_id": "target"}),
        state_after=state,
        observation=PredictionPacket(
            progress_mean=0.0,
            terminal_probability=0.0,
            goal_probability=0.0,
            known_channels=frozenset({"objects", "progress", "terminal", "goal"}),
            state_after=state,
        ),
        events=("no_effect",),
    )


def _probabilities(posterior: ProgramPosterior) -> dict[str, float]:
    return {
        particle.program.canonical_hash: particle.probability
        for particle in posterior.particles
    }


def test_registered_policy_multipliers() -> None:
    assert PosteriorUpdatePolicy.legacy().evidence_multiplier(4) == 1.0
    assert PosteriorUpdatePolicy.tempered().evidence_multiplier(4) == 0.25
    assert PosteriorUpdatePolicy.correlation_aware().evidence_multiplier(4) == 0.5
    assert PosteriorUpdatePolicy.combined().evidence_multiplier(4) == 0.125


def test_legacy_policy_matches_frozen_posterior() -> None:
    programs = (
        AssembledProgram(_program("right", solves=True), -1.0),
        AssembledProgram(_program("wrong", solves=False), -1.0),
    )
    baseline = ProgramPosterior(repair_ess_threshold=1.0)
    challenger = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.legacy(),
        repair_ess_threshold=1.0,
    )
    baseline.seed(programs, initial_state=_state())
    challenger.seed(programs, initial_state=_state())

    for _ in range(2):
        baseline.observe(_evidence(), allow_repair=False)
        challenger.observe(_evidence(), allow_repair=False)

    assert _probabilities(challenger) == pytest.approx(
        _probabilities(baseline),
        abs=1e-15,
    )
    assert challenger.normalized_entropy == pytest.approx(
        baseline.normalized_entropy,
        abs=1e-15,
    )
    assert [p.latest_log_likelihood for p in challenger.particles] == pytest.approx(
        [p.latest_log_likelihood for p in baseline.particles],
        abs=1e-15,
    )


def test_surprise_is_raw_preupdate_mixture_surprise() -> None:
    posterior = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.tempered(),
        repair_ess_threshold=1.0,
    )
    posterior.seed(
        (_program("right", solves=True), _program("wrong", solves=False)),
        initial_state=_state(),
    )
    before = tuple(posterior.particles)
    evidence = _evidence()
    raw = []
    for particle in before:
        prediction = posterior.executor.step(
            particle.program,
            evidence.state_before,
            evidence.action,
        )
        from theory.sage_t.posterior import packet_log_likelihood

        raw.append(packet_log_likelihood(prediction, evidence.observation))
    expected_probability = sum(
        particle.probability * math.exp(log_likelihood)
        for particle, log_likelihood in zip(before, raw)
    )

    diagnostics = posterior.observe(evidence, allow_repair=False)

    assert diagnostics is not None
    assert diagnostics.raw_mixture_surprise == pytest.approx(
        -math.log(expected_probability)
    )
    assert diagnostics.raw_best_program_surprise == pytest.approx(-max(raw))
    assert diagnostics.evidence_multiplier == 0.25


def test_context_discount_is_scoped_by_regime_and_ignores_reset() -> None:
    posterior = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.correlation_aware(),
        repair_ess_threshold=1.0,
    )
    posterior.seed((_program("right", solves=True),), initial_state=_state())

    first = posterior.observe(_evidence(), allow_repair=False)
    reset = posterior.observe(_evidence(reset=True), allow_repair=False)
    second = posterior.observe(_evidence(), allow_repair=False)
    new_regime = posterior.observe(_evidence(regime_index=1), allow_repair=False)

    assert first is not None and first.context_count == 1
    assert reset is None
    assert second is not None and second.context_count == 2
    assert second.evidence_multiplier == pytest.approx(1 / math.sqrt(2))
    assert new_regime is not None and new_regime.context_count == 1


def test_semantic_collapse_uses_raw_surprise() -> None:
    policy = replace(
        PosteriorUpdatePolicy.tempered(),
        semantic_collapse_entropy_maximum=1.0,
        semantic_collapse_surprise_minimum=0.0,
    )
    posterior = CalibratedProgramPosterior(
        update_policy=policy,
        repair_ess_threshold=1.0,
    )
    posterior.seed((_program("wrong", solves=False),), initial_state=_state())

    diagnostics = posterior.observe(_evidence(), allow_repair=False)

    assert diagnostics is not None
    assert diagnostics.raw_mixture_surprise > 0.0
    assert diagnostics.semantic_collapse


def test_incremental_log_joint_equals_full_replay() -> None:
    posterior = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.combined().with_repair_v2(),
        repair_ess_threshold=1.0,
    )
    program = _program("right", solves=True)
    posterior.seed((program,), initial_state=_state())
    posterior.observe(_evidence(), allow_repair=False)
    posterior.observe(_evidence(), allow_repair=False)
    particle = posterior.particles[0]
    replayed = posterior._replay_particle(
        CalibratedProgramParticle(
            program=program,
            log_prior=particle.log_prior,
            log_weight=particle.log_prior,
            log_joint=particle.log_prior,
        )
    )

    assert particle.log_joint == pytest.approx(replayed.log_joint, abs=1e-12)
    assert particle.observations == replayed.observations == 2


def test_repair_v2_budgets_children_and_survivors() -> None:
    posterior = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.combined().with_repair_v2(),
        repair_ess_threshold=1.0,
        repair_log_likelihood_threshold=-100.0,
    )
    posterior.seed((_program("wrong", solves=False),), initial_state=_state())
    evidence = _evidence()
    posterior.observe(evidence, allow_repair=False)

    children = posterior.repair(evidence)
    snapshot = posterior.snapshot()

    assert snapshot["repair_cycles"] == 1
    assert snapshot["repairs_evaluated"] <= 8
    assert snapshot["repairs_survived"] <= 4
    assert len(children) <= 4
    assert all(child.observations == 1 for child in children)


def test_combined_policy_retains_more_mass_for_a_late_positive_shock() -> None:
    programs = (_program("signal", solves=True), _program("no_effect", solves=False))
    legacy = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.legacy(),
        repair_ess_threshold=1.0,
    )
    combined = CalibratedProgramPosterior(
        update_policy=PosteriorUpdatePolicy.combined(),
        repair_ess_threshold=1.0,
    )
    for posterior in (legacy, combined):
        posterior.seed(programs, initial_state=_state())
        for _ in range(4):
            posterior.observe(_negative_evidence(), allow_repair=False)

    legacy_diagnostics = legacy.observe(_evidence(), allow_repair=False)
    combined_diagnostics = combined.observe(_evidence(), allow_repair=False)
    legacy_mass = next(
        particle.probability
        for particle in legacy.particles
        if particle.program.program_id == "signal"
    )
    combined_mass = next(
        particle.probability
        for particle in combined.particles
        if particle.program.program_id == "signal"
    )

    assert legacy_diagnostics is not None and combined_diagnostics is not None
    assert legacy_diagnostics.raw_mixture_surprise > 3.0
    assert combined_diagnostics.raw_mixture_surprise > 3.0
    assert combined_mass > legacy_mass
