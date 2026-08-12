from __future__ import annotations

from theory.sage_t.causal.adapters import causal_state_from_abstract
from theory.sage_t.causal.contracts import (
    ActionInterventionSpec,
    BindingSpec,
    CausalProgram,
    CausalState,
    CausalVariableSpec,
    GoalSpec,
    GroundedAction,
    MechanismSpec,
    ParentRef,
    ValueDistribution,
)
from theory.sage_t.causal.executor import CausalExecutor
from theory.sage_t.causal.options import (
    CausalOptionCompiler,
    CompiledCausalOption,
    MinimalOptionExtractor,
    OptionMechanismRegistry,
)
from theory.sage_t.contracts import AbstractEntity, AbstractState


def parent_program() -> CausalProgram:
    return CausalProgram(
        program_id="parent",
        bindings=BindingSpec({}),
        variables=(CausalVariableSpec("world.ready", "boolean", (False, True)),),
        mechanisms=(
            MechanismSpec(
                mechanism_id="ready_identity",
                output_variable="world.ready",
                parent_variables=(ParentRef("world.ready"),),
                operator_type="identity",
            ),
        ),
        action_model=tuple(ActionInterventionSpec(f"ACTION{index}") for index in range(1, 4)),
        goal=GoalSpec("world.ready == true"),
        description_length=1.0,
    )


def test_option_extraction_minimizes_and_removes_absolute_binding() -> None:
    state = AbstractState(
        entities=(
            AbstractEntity(
                "e1",
                ("target",),
                (("color", "red"),),
                center=(5.0, 4.0),
            ),
        )
    )
    actions = (
        GroundedAction("ACTION1"),
        GroundedAction("ACTION2", {"x": 4, "y": 5}),
        GroundedAction("ACTION3"),
    )

    def progresses(candidate: tuple[GroundedAction, ...]) -> bool:
        names = tuple(action.action_name for action in candidate)
        return names[-2:] == ("ACTION2", "ACTION3")

    option = MinimalOptionExtractor().extract(
        initiation_state=state,
        initiation_exact_hash="exact-init",
        actions=actions,
        states_before=(state, state, state),
        replay_progress=progresses,
    )
    assert [step.action_name for step in option.steps] == ["ACTION2", "ACTION3"]
    assert "x" not in option.steps[0].static_action_data
    assert "y" not in option.steps[0].static_action_data
    assert option.materialize(state)[0].action_data == {"x": 4.0, "y": 5.0}


def test_option_compiles_into_complete_particles_and_executes() -> None:
    state = AbstractState()
    actions = (GroundedAction("ACTION1"), GroundedAction("ACTION2"))
    option = MinimalOptionExtractor().extract(
        initiation_state=state,
        initiation_exact_hash="exact-init",
        actions=actions,
        states_before=(state, state),
        replay_progress=lambda candidate: tuple(
            action.action_name for action in candidate
        )
        == ("ACTION1", "ACTION2"),
    )
    children, registry = CausalOptionCompiler().compile(option, (parent_program(),))
    restored = CompiledCausalOption.from_dict(registry.to_dict())
    assert restored.owner_program_hashes == (children[0].canonical_hash,)

    executor = CausalExecutor(mechanism_registry=OptionMechanismRegistry())
    executor.compile(children[0])
    current = causal_state_from_abstract(state)
    for variable in children[0].variables:
        if variable.variable_id not in current.variables:
            current = CausalState(
                variables={
                    **current.variables,
                    variable.variable_id: ValueDistribution.deterministic(
                        variable.domain[0]
                    ),
                }
            )
    for action in actions:
        current = executor.predict_step(children[0], current, action).state_after
    complete = next(
        variable.variable_id
        for variable in children[0].variables
        if variable.variable_id.endswith(".complete")
    )
    assert current.value(complete).mode is True
