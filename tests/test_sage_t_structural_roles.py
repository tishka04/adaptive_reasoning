from __future__ import annotations

import json

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
    ProgressRule,
    TerminalRule,
    TransitionRule,
)
from theory.sage_t.structural_roles import (
    EASTMOST_TARGET,
    WESTMOST_TARGET,
    StructuralRoleProgramExecutor,
    action_target_structural_role,
    augment_structural_roles,
)


def _state(*, shift: float = 0.0, reverse: bool = False) -> AbstractState:
    entities = (
        AbstractEntity(
            "local_left",
            ("object", "target"),
            center=(29.0, 4.0 + shift),
        ),
        AbstractEntity(
            "local_right",
            ("object", "target"),
            center=(29.0, 56.0 + shift),
        ),
    )
    return AbstractState(
        entities=tuple(reversed(entities)) if reverse else entities,
        false_facts=frozenset(
            {GroundFact("level_complete"), GroundFact("game_over")}
        ),
    )


def _guarded_program() -> JointProgramHypothesis:
    return JointProgramHypothesis(
        program_id="ordinal_goal",
        object_schema=ObjectSchema(
            ("object", "target", WESTMOST_TARGET, EASTMOST_TARGET)
        ),
        action_bindings=(
            ActionBinding("ACTION6", "apply", target_role="target"),
        ),
        transition_rules=(
            TransitionRule(
                rule_id="win_on_westmost",
                action_operator="apply",
                condition=Expression.fact(
                    "role", "$target", WESTMOST_TARGET
                ),
                effects=(Effect("progress", value=1.0), Effect("win")),
            ),
        ),
        progress_rule=ProgressRule(Expression(op="counter", value="progress")),
        terminal_rules=(
            TerminalRule(Expression.fact("game_over"), outcome="game_over"),
            TerminalRule(Expression.fact("level_complete"), outcome="win"),
        ),
        goal_rule=GoalRule(
            Expression.fact("level_complete"), family="level_completion"
        ),
    )


def test_ordinal_roles_are_translation_and_order_invariant() -> None:
    baseline = augment_structural_roles(_state())
    shifted = augment_structural_roles(_state(shift=100.0, reverse=True))

    baseline_roles = {entity.entity_id: entity.roles for entity in baseline.entities}
    shifted_roles = {entity.entity_id: entity.roles for entity in shifted.entities}
    assert WESTMOST_TARGET in baseline_roles["local_left"]
    assert EASTMOST_TARGET in baseline_roles["local_right"]
    assert shifted_roles == baseline_roles


def test_ties_and_missing_positions_do_not_invent_extrema() -> None:
    tied = AbstractState(
        entities=(
            AbstractEntity("a", ("target",), center=(0.0, 4.0)),
            AbstractEntity("b", ("target",), center=(1.0, 4.0)),
            AbstractEntity("c", ("target",)),
        )
    )

    enriched = augment_structural_roles(tied)

    assert all(
        WESTMOST_TARGET not in entity.roles
        and EASTMOST_TARGET not in entity.roles
        for entity in enriched.entities
    )


def test_action_target_role_supports_grounded_id_and_nearest_position() -> None:
    state = _state()

    by_id = action_target_structural_role(
        state,
        ActionCandidate("ACTION6", {"entity_id": "local_left"}),
    )
    by_position = action_target_structural_role(
        state,
        ActionCandidate("ACTION6", {"x": 56, "y": 29}),
    )

    assert by_id == WESTMOST_TARGET
    assert by_position == EASTMOST_TARGET


def test_structural_executor_separates_neutral_and_winning_target() -> None:
    executor = StructuralRoleProgramExecutor()
    program = _guarded_program()
    state = _state()

    neutral = executor.step(
        program,
        state,
        ActionCandidate("ACTION6", {"entity_id": "local_right"}),
    )
    winning = executor.step(
        program,
        state,
        ActionCandidate("ACTION6", {"entity_id": "local_left"}),
    )

    assert neutral.object_deltas == {"no_effect": 0.95}
    assert neutral.progress_mean == 0.0
    assert neutral.goal_probability == 0.05
    assert winning.progress_mean == 1.0
    assert winning.goal_probability == 0.95


def test_transferable_program_contains_no_ids_or_coordinates() -> None:
    payload = json.dumps(
        _guarded_program().canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert WESTMOST_TARGET in payload
    assert "local_left" not in payload
    assert "local_right" not in payload
    assert "29.0" not in payload
    assert "56.0" not in payload
