from __future__ import annotations

import math
from dataclasses import replace

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
from theory.sage_t.executor import ProgramExecutor
from theory.sage_t.posterior import ProgramPosterior, packet_log_likelihood
from theory.sage_t.synthesis import AssembledProgram, ProgramMutator


def _state() -> AbstractState:
    target = AbstractEntity(
        "target",
        ("object", "target"),
        center=(1.0, 1.0),
    )
    return AbstractState(
        entities=(target,),
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


def _program(
    program_id: str,
    *,
    solves: bool,
    node_padding: int = 0,
) -> JointProgramHypothesis:
    effect = (
        Effect("assert", predicate="solved", terms=("$target",))
        if solves
        else Effect("assert", predicate="no_effect")
    )
    condition = Expression.constant(True)
    for _ in range(node_padding):
        condition = Expression(
            op="and",
            args=(condition, Expression.constant(True)),
        )
    solved_count = Expression(
        op="count",
        args=(Expression.fact("solved", "?item"),),
        variable="?item",
        role="target",
    )
    return JointProgramHypothesis(
        program_id=program_id,
        object_schema=ObjectSchema(("object", "target")),
        action_bindings=(ActionBinding("ACTION1", "apply", target_role="target"),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{program_id}",
                action_operator="apply",
                condition=condition,
                effects=(effect,),
            ),
        ),
        progress_rule=ProgressRule(solved_count),
        terminal_rules=(
            TerminalRule(
                Expression.fact("game_over"),
                outcome="game_over",
            ),
        ),
        goal_rule=GoalRule(
            Expression(
                op="forall",
                args=(Expression.fact("solved", "?goal"),),
                variable="?goal",
                role="target",
            ),
            family="solve_targets",
        ),
    )


def _evidence() -> ObservedTransition:
    before = _state()
    after = before.with_updates(asserted=(GroundFact("solved", ("target",)),))
    return ObservedTransition(
        state_before=before,
        action=ActionCandidate(
            "ACTION1",
            {"entity_id": "target"},
        ),
        state_after=after,
        observation=PredictionPacket(
            progress_mean=1.0,
            terminal_probability=0.0,
            goal_probability=1.0,
            known_channels=frozenset({"objects", "progress", "terminal", "goal"}),
            state_after=after,
        ),
        events=("progress", "level_complete"),
    )


def test_executor_is_pure_cached_and_rolls_out_only_requested_actions() -> None:
    state = _state()
    program = _program("solve", solves=True)
    executor = ProgramExecutor()
    action = ActionCandidate("ACTION1", {"entity_id": "target"})

    first = executor.step(program, state, action)
    second = executor.step(program, state, action)
    rollout = executor.rollout(program, state, (action, action))

    assert first == second
    assert executor.cache_hits >= 1
    assert state.truth(GroundFact("solved", ("target",))).value == "false"
    assert first.state_after is not None
    assert first.state_after.truth(GroundFact("solved", ("target",))).value == "true"
    assert first.progress_mean == 1.0
    assert first.progress_distribution == {
        "value:1": 0.95,
        "other": 0.05,
    }
    assert first.goal_probability == 0.95
    assert len(rollout.packets) == 2
    assert rollout.packets[1].progress_mean == 0.0


def test_executor_supports_explicit_progress_win_and_failure_effects() -> None:
    base = _program("base", solves=False)
    win_program = replace(
        base,
        transition_rules=(
            replace(
                base.transition_rules[0],
                effects=(Effect("win"),),
            ),
        ),
        progress_rule=ProgressRule(Expression.fact("level_complete")),
        goal_rule=GoalRule(
            Expression.fact("level_complete"),
            family="level_completion",
        ),
    )
    failure_program = replace(
        base,
        transition_rules=(
            replace(
                base.transition_rules[0],
                effects=(Effect("fail"),),
            ),
        ),
    )
    action = ActionCandidate("ACTION1", {"entity_id": "target"})
    executor = ProgramExecutor()

    victory = executor.step(win_program, _state(), action)
    failure = executor.step(failure_program, _state(), action)

    assert victory.progress_mean == 1.0
    assert victory.goal_probability == 0.95
    assert failure.terminal_probability == 0.95


def test_posterior_prefers_the_program_that_explains_joint_signals() -> None:
    true_program = _program("true_program", solves=True)
    false_program = _program("false_program", solves=False)
    posterior = ProgramPosterior(repair_ess_threshold=1.0)
    posterior.seed(
        (
            AssembledProgram(true_program, -1.0),
            AssembledProgram(false_program, -1.0),
        ),
        initial_state=_state(),
    )

    posterior.observe(_evidence(), allow_repair=False)

    probabilities = {
        particle.program.program_id: particle.probability
        for particle in posterior.particles
    }
    assert probabilities["true_program"] > 0.99
    assert math.isclose(sum(probabilities.values()), 1.0)


def test_mdl_prior_penalizes_larger_equally_supported_programs() -> None:
    compact = _program("compact", solves=True)
    verbose = _program("verbose", solves=True, node_padding=3)
    posterior = ProgramPosterior()

    posterior.seed((compact, verbose), initial_state=_state())

    weights = {
        particle.program.program_id: particle.probability
        for particle in posterior.particles
    }
    assert compact.node_count < verbose.node_count
    assert weights["compact"] > weights["verbose"]


def test_repair_children_replay_the_complete_branch_history() -> None:
    posterior = ProgramPosterior(
        repair_ess_threshold=1.0,
        repair_log_likelihood_threshold=-100.0,
    )
    posterior.seed(
        (_program("wrong", solves=False),),
        initial_state=_state(),
    )
    evidence = _evidence()
    posterior.observe(evidence, allow_repair=False)

    children = posterior.repair(evidence)

    assert children
    assert all(child.observations == 1 for child in children)
    assert math.isclose(
        sum(particle.probability for particle in posterior.particles),
        1.0,
    )


def test_unknown_coverage_cannot_beat_a_correct_specific_prediction() -> None:
    observed = PredictionPacket(
        terminal_probability=0.0,
        known_channels=frozenset({"terminal"}),
    )
    specific = PredictionPacket(
        terminal_probability=0.05,
        known_channels=frozenset({"terminal"}),
    )

    assert packet_log_likelihood(
        specific,
        observed,
    ) > packet_log_likelihood(PredictionPacket(), observed)


def test_semantically_equivalent_programs_merge_on_observed_contexts() -> None:
    posterior = ProgramPosterior(repair_ess_threshold=1.0)
    posterior.seed(
        (
            _program("compact", solves=True),
            _program("verbose", solves=True, node_padding=2),
        ),
        initial_state=_state(),
    )

    posterior.observe(_evidence(), allow_repair=False)

    assert len(posterior.particles) == 1
    assert posterior.particles[0].probability == 1.0


def test_local_repair_covers_each_declared_edit_family() -> None:
    children = ProgramMutator(maximum_children=8).mutate(
        _program("wrong", solves=False),
        _evidence(),
    )
    identifiers = {child.program_id for child in children}

    assert any("effects" in identifier for identifier in identifiers)
    assert any("precondition" in identifier for identifier in identifiers)
    assert any("action_semantics" in identifier for identifier in identifiers)
    assert any("target_role" in identifier for identifier in identifiers)
    assert any("goal_quantifier" in identifier for identifier in identifiers)
