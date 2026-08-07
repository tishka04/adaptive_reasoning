from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact
from theory.sage_t.mixed_automata_v10_2 import (
    MAX_ACTION_SCHEMAS,
    MAX_OPTION_HORIZON,
    MAX_OPTION_STATES,
    REGISTERED_TRUE_INITIATION,
    OptionAutomaton,
    OptionExecutionState,
    OptionInitiationCondition,
    OptionState,
    OptionTransition,
    alternate,
    ast_node_log_cost,
    distinct_noop_probe,
    follow_relation_then_apply,
    generate_mixed_grammar,
    prime_then_repeat,
    repeat,
    until_then,
)

GROUNDINGS = {
    "a": {"action_name": "ACTION1"},
    "b": {"action_name": "ACTION6", "action_data": {"x": 4, "y": 9}},
    "probe": {"action_name": "ACTION6", "action_data": {"x": 1, "y": 2}},
    "apply": ("ACTION5", {}),
}


def _renamed(option: OptionAutomaton) -> OptionAutomaton:
    mapping = {
        state.state_id: f"renamed_{index}" for index, state in enumerate(option.states)
    }
    states = tuple(
        OptionState(mapping[state.state_id], terminal=state.terminal)
        for state in reversed(option.states)
    )
    transitions = tuple(
        OptionTransition(
            mapping[item.source_state_id],
            mapping[item.target_state_id],
            item.action_schema,
            predicate=item.predicate,
            predicate_present=item.predicate_present,
            minimum_visits=item.minimum_visits,
            relation=item.relation,
            priority=item.priority,
        )
        for item in reversed(option.transitions)
    )
    return OptionAutomaton(
        schema=option.schema,
        states=states,
        transitions=transitions,
        initial_state_id=mapping[option.initial_state_id],
        maximum_horizon=option.maximum_horizon,
        initiation_condition=option.initiation_condition,
    )


def test_registered_grammar_preserves_repeat_and_adds_all_mixed_families() -> None:
    grammar = generate_mixed_grammar(
        ("a", "b"),
        predicates=("state_change",),
        prime_counts=(1,),
        noop_termination_predicates=("state_change", "level_complete"),
    )

    assert {
        "repeat",
        "alternate",
        "prime_then_repeat",
        "until_then",
        "follow_relation_then_apply",
        "distinct_noop_probe",
    } <= {option.schema for option in grammar}
    assert sum(option.schema == "repeat" for option in grammar) == 2
    assert len({option.canonical_hash for option in grammar}) == len(grammar)
    assert all(len(option.states) <= MAX_OPTION_STATES for option in grammar)
    assert all(len(option.action_schemas) <= MAX_ACTION_SCHEMAS for option in grammar)
    assert all(option.maximum_horizon <= MAX_OPTION_HORIZON for option in grammar)


def test_automaton_limits_immutability_and_ast_cost_are_fail_closed() -> None:
    option = repeat("a", maximum_horizon=4)
    assert option.initial_state_id == "active"
    assert option.initial_state == OptionState("active")
    assert option.node_count == 2 + len(option.states) + len(option.transitions)
    assert ast_node_log_cost(option.node_count) == pytest.approx(
        -0.05 * option.node_count
    )
    with pytest.raises(FrozenInstanceError):
        option.maximum_horizon = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="limited to 4 states"):
        OptionAutomaton(
            schema="too_many_states",
            states=tuple(
                OptionState(f"state_{index}", terminal=index == 4) for index in range(5)
            ),
            transitions=(),
            initial_state_id="state_0",
            initiation_condition=REGISTERED_TRUE_INITIATION,
        )
    with pytest.raises(ValueError, match="limited to 2 action schemas"):
        OptionAutomaton(
            schema="too_many_actions",
            states=(
                OptionState("first"),
                OptionState("second"),
                OptionState("third"),
                OptionState("done", terminal=True),
            ),
            transitions=(
                OptionTransition("first", "second", "a"),
                OptionTransition("second", "third", "b"),
                OptionTransition("third", "done", "c"),
            ),
            initial_state_id="first",
            initiation_condition=REGISTERED_TRUE_INITIATION,
        )
    with pytest.raises(ValueError, match="maximum_horizon"):
        repeat("a", maximum_horizon=17)
    with pytest.raises(TypeError, match="explicit initiation condition"):
        OptionAutomaton(
            schema=option.schema,
            states=option.states,
            transitions=option.transitions,
            initial_state_id=option.initial_state_id,
            maximum_horizon=option.maximum_horizon,
        )


def test_canonical_hash_is_state_identity_and_grounding_free() -> None:
    option = alternate("a", "b", maximum_horizon=6)
    renamed = _renamed(option)

    assert option.canonical_hash == renamed.canonical_hash
    assert "renamed" not in str(renamed.canonical_payload)
    first = option.execute_one(option.new_execution(), GROUNDINGS)
    shifted = option.execute_one(
        option.new_execution(),
        {
            "a": {"action_name": "ACTION9", "action_data": {"x": 99, "y": 0}},
            "b": "ACTION2",
        },
    )
    assert first.action_name != shifted.action_name
    assert option.canonical_hash == renamed.canonical_hash


def test_structural_initiation_is_hashed_and_required_for_execution() -> None:
    template = repeat("a", maximum_horizon=3)
    ready_condition = OptionInitiationCondition.fact("reachable", role="target")
    blocked_condition = OptionInitiationCondition.fact("solved", role="target")

    def with_condition(condition: OptionInitiationCondition) -> OptionAutomaton:
        return OptionAutomaton(
            schema=template.schema,
            states=template.states,
            transitions=template.transitions,
            initial_state_id=template.initial_state_id,
            maximum_horizon=template.maximum_horizon,
            initiation_condition=condition,
        )

    ready = with_condition(ready_condition)
    blocked = with_condition(blocked_condition)
    target = AbstractEntity("target_local", ("object", "target"))
    unsatisfied = AbstractState(entities=(target,))
    satisfied = AbstractState(
        entities=(target,),
        true_facts=frozenset({GroundFact("reachable", ("target_local",))}),
    )
    explicitly_unreachable = AbstractState(
        entities=(target,),
        false_facts=frozenset({GroundFact("reachable", ("target_local",))}),
    )

    assert ready.canonical_hash != blocked.canonical_hash
    assert ready.canonical_payload["initiation"] == ready_condition.canonical_payload
    with pytest.raises(ValueError, match="initiation condition"):
        ready.new_execution()
    with pytest.raises(ValueError, match="initiation condition"):
        ready.new_execution(unsatisfied)
    execution = ready.new_execution(satisfied)
    with pytest.raises(ValueError, match="initiation condition"):
        ready.execute_one(execution, GROUNDINGS)
    with pytest.raises(ValueError, match="initiation condition"):
        ready.execute_one(execution, GROUNDINGS, state=unsatisfied)
    assert ready.execute_one(
        execution,
        GROUNDINGS,
        state=satisfied,
    ).executed

    assert template.initiation_condition is REGISTERED_TRUE_INITIATION
    assert template.can_initiate()
    requires_absence = OptionInitiationCondition.fact(
        "reachable",
        role="target",
        present=False,
    )
    assert not requires_absence.satisfied_by(unsatisfied)
    assert requires_absence.satisfied_by(explicitly_unreachable)


def test_execution_state_enforces_counter_terminal_and_pending_invariants() -> None:
    option = repeat("a", maximum_horizon=2)
    digest = option.canonical_hash

    with pytest.raises(ValueError, match="state_visits <= steps <= horizon"):
        OptionExecutionState(
            digest,
            "active",
            steps=1,
            state_visits=2,
            maximum_horizon=2,
        )
    with pytest.raises(ValueError, match="state_visits <= steps <= horizon"):
        OptionExecutionState(
            digest,
            "active",
            steps=3,
            state_visits=1,
            maximum_horizon=2,
        )
    with pytest.raises(ValueError, match="terminated execution"):
        OptionExecutionState(
            digest,
            "active",
            steps=1,
            state_visits=1,
            maximum_horizon=2,
            awaiting_observation=True,
            pending_action_schema="a",
            terminated=True,
        )
    with pytest.raises(ValueError, match="positive state visit"):
        OptionExecutionState(
            digest,
            "active",
            steps=1,
            state_visits=0,
            maximum_horizon=2,
            awaiting_observation=True,
            pending_action_schema="a",
        )

    wrong_pending = OptionExecutionState(
        digest,
        "active",
        steps=1,
        state_visits=1,
        maximum_horizon=2,
        awaiting_observation=True,
        pending_action_schema="b",
    )
    with pytest.raises(ValueError, match="pending action schema"):
        option.stateful_signature(wrong_pending)

    premature_terminal = OptionExecutionState(
        digest,
        "active",
        maximum_horizon=2,
        terminated=True,
    )
    with pytest.raises(ValueError, match="terminated flag"):
        option.stateful_signature(premature_terminal)


def test_prime_then_repeat_executes_one_action_and_replans_after_each_observation() -> (
    None
):
    option = prime_then_repeat("a", 2, "b", maximum_horizon=6)
    execution = option.new_execution()

    first = option.execute_one(execution, GROUNDINGS)
    assert first.issued_actions == 1
    assert first.action_schema == "a"
    assert first.requires_observation is True
    assert first.requires_update is True
    assert first.requires_replan is True
    with pytest.raises(RuntimeError, match="observe/update/replan"):
        option.execute_one(first.execution_after_action, GROUNDINGS)

    execution = option.observe_update_replan(
        first.execution_after_action,
        events=("no_effect",),
    )
    assert execution.state_id == "prime"
    second = option.execute_one(execution, GROUNDINGS)
    assert second.action_schema == "a"
    execution = option.observe_update_replan(
        second.execution_after_action,
        events=("no_effect",),
    )
    assert execution.state_id == "repeated"
    third = option.execute_one(execution, GROUNDINGS)
    assert third.action_schema == "b"
    assert third.data == {"x": 4, "y": 9}


def test_alternate_until_and_relation_options_have_mixed_stateful_execution() -> None:
    alternating = alternate("a", "b", maximum_horizon=5)
    first = alternating.execute_one(alternating.new_execution(), GROUNDINGS)
    after_first = alternating.observe(
        first.execution_after_action,
        first,
        events=("no_effect",),
    )
    assert after_first.state_id == "second"
    assert alternating.allowed_action_schemas(after_first) == ("b",)
    assert (
        alternating.observe(
            alternating.initial_state,
            "a",
            events=("no_effect",),
        ).state_id
        == "second"
    )

    gated = until_then("a", "state_change", "b", maximum_horizon=5)
    first = gated.execute_one(gated.new_execution(), GROUNDINGS)
    unchanged = gated.observe_update_replan(
        first.execution_after_action,
        events=("no_effect",),
    )
    assert unchanged.state_id == "until"
    first = gated.execute_one(unchanged, GROUNDINGS)
    changed = gated.observe_update_replan(
        first.execution_after_action,
        predicates=("state_change",),
    )
    assert changed.state_id == "apply"
    assert gated.allowed_action_schemas(changed) == ("b",)

    relation = follow_relation_then_apply("a", "b")
    assert any(
        item.relation == "successor_toward_enclosure"
        for item in relation.transitions
        if item.source_state_id == "follow"
    )


def test_relation_option_requires_matching_grounding_provenance() -> None:
    option = follow_relation_then_apply("a", "b")
    with pytest.raises(ValueError, match="requires relation provenance"):
        option.execute_one(option.new_execution(), GROUNDINGS)

    mismatched = dict(GROUNDINGS)
    mismatched["a"] = {
        "action_name": "ACTION1",
        "relation": "unrelated",
    }
    with pytest.raises(ValueError, match="does not match"):
        option.execute_one(option.new_execution(), mismatched)

    certified = dict(GROUNDINGS)
    certified["a"] = {
        "action_name": "ACTION1",
        "relation": "successor_toward_enclosure",
    }
    issued = option.execute_one(option.new_execution(), certified)
    assert issued.executed is True
    assert issued.action_schema == "a"


def test_noop_prefix_dedup_preserves_options_that_diverge_at_termination() -> None:
    opened = distinct_noop_probe(
        "probe",
        "apply",
        termination_predicate="opened",
    )
    closed = distinct_noop_probe(
        "probe",
        "apply",
        termination_predicate="closed",
    )
    opened_first = opened.execute_one(opened.new_execution(), GROUNDINGS)
    closed_first = closed.execute_one(closed.new_execution(), GROUNDINGS)

    opened_state = opened.observe_update_replan(
        opened_first.execution_after_action,
        events=("no_effect",),
    )
    closed_state = closed.observe_update_replan(
        closed_first.execution_after_action,
        events=("no_effect",),
    )

    assert opened_state.state_id == closed_state.state_id == "followup"
    assert opened.canonical_hash != closed.canonical_hash
    assert opened.stateful_signature(opened_state) != closed.stateful_signature(
        closed_state
    )


def test_danger_veto_issues_no_action_and_requires_safe_replanning() -> None:
    option = repeat("a")
    execution = option.new_execution()
    vetoed = option.execute_one(execution, GROUNDINGS, danger=True)

    assert vetoed.issued_actions == 0
    assert vetoed.danger_vetoed is True
    assert vetoed.execution_after_action == execution
    assert vetoed.requires_observation is False
    assert vetoed.requires_update is False
    assert vetoed.requires_replan is True
    safe = option.execute_one(execution, GROUNDINGS)
    assert safe.issued_actions == 1


def test_horizon_terminates_only_after_the_last_action_is_observed() -> None:
    option = repeat("a", maximum_horizon=1)
    issued = option.execute_one(option.new_execution(), GROUNDINGS)

    assert issued.execution_after_action.awaiting_observation is True
    finished = option.observe_update_replan(
        issued.execution_after_action,
        events=("no_effect",),
    )
    assert finished.terminated is True
    assert option.allowed_action_schemas(finished) == ()
    stopped = option.execute_one(finished, GROUNDINGS)
    assert stopped.issued_actions == 0
    assert stopped.reason == "option_terminated"
