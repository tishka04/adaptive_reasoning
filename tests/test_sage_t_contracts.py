from __future__ import annotations

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
    ProgramFragment,
    ProgressRule,
    TerminalRule,
    TransitionRule,
    TruthValue,
    program_from_dict,
    program_to_dict,
)


def _program(
    *,
    program_id: str = "program",
    variable: str = "?target",
) -> JointProgramHypothesis:
    return JointProgramHypothesis(
        program_id=program_id,
        object_schema=ObjectSchema(("object", "target")),
        action_bindings=(ActionBinding("ACTION1", "apply", target_role="target"),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{program_id}",
                action_operator="apply",
                condition=Expression(
                    op="exists",
                    args=(Expression.fact("exists", variable),),
                    variable=variable,
                    role="target",
                ),
                effects=(
                    Effect(
                        operation="assert",
                        predicate="solved",
                        terms=("$target",),
                    ),
                ),
            ),
        ),
        progress_rule=ProgressRule(
            Expression(
                op="count",
                args=(Expression.fact("solved", "?item"),),
                variable="?item",
                role="target",
            )
        ),
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
        provenance=("unit_test",),
    )


def test_abstract_state_distinguishes_false_unknown_and_absent() -> None:
    known_true = GroundFact("exists", ("target",))
    known_false = GroundFact("solved", ("target",))
    state = AbstractState(
        entities=(AbstractEntity("target", ("object", "target")),),
        true_facts=frozenset({known_true}),
        false_facts=frozenset({known_false}),
    )

    assert state.truth(known_true) is TruthValue.TRUE
    assert state.truth(known_false) is TruthValue.FALSE
    assert state.truth(GroundFact("selected", ("target",))) is TruthValue.UNKNOWN


def test_dsl_round_trip_and_alpha_hash_ignore_names() -> None:
    original = _program(program_id="first", variable="?alpha")
    renamed = _program(program_id="second", variable="?beta")

    restored = program_from_dict(program_to_dict(original))

    assert restored.canonical_hash == original.canonical_hash
    assert renamed.canonical_hash == original.canonical_hash


def test_program_rejects_grounded_constants_and_incomplete_fragments() -> None:
    with pytest.raises(ValueError, match="grounded"):
        Expression.fact("exists", "(4,7)")
    with pytest.raises(ValueError, match="forbidden"):
        ObjectSchema(("object", "color_7"))
    with pytest.raises(ValueError, match="forbidden"):
        Effect(
            "set_register",
            key="selected",
            value="pixel_4",
        )
    with pytest.raises(ValueError, match="support=0"):
        ProgramFragment(
            fragment_id="invalid",
            kind="goal_bundle",
            payload=GoalRule(Expression.constant(True)),
            support=1,
        )
    with pytest.raises(ValueError, match="terminal"):
        replace(_program(), terminal_rules=())


def test_alpha_hash_keeps_distinct_variable_linkage_distinct() -> None:
    base = _program()
    rule = base.transition_rules[0]
    linked = replace(
        base,
        transition_rules=(
            replace(
                rule,
                effects=(
                    Effect(
                        operation="assert",
                        predicate="solved",
                        terms=("?target",),
                    ),
                ),
            ),
        ),
    )
    unlinked = replace(
        base,
        transition_rules=(
            replace(
                rule,
                effects=(
                    Effect(
                        operation="assert",
                        predicate="solved",
                        terms=("?other",),
                    ),
                ),
            ),
        ),
    )

    assert linked.canonical_hash != unlinked.canonical_hash
