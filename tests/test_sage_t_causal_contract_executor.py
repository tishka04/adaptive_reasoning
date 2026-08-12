from dataclasses import replace

import pytest

from theory.sage_t.causal.compiler import ProgramCompiler
from theory.sage_t.causal.contracts import (
    ActionInterventionSpec,
    ActionProgram,
    BindingSpec,
    CausalProgram,
    CausalState,
    CausalVariableSpec,
    GoalSpec,
    GroundedAction,
    Intervention,
    MechanismSpec,
    ParentRef,
    ValueDistribution,
    causal_program_from_json,
    causal_program_to_json,
)
from theory.sage_t.causal.executor import CausalExecutor
from theory.sage_t.causal.mechanisms import MechanismRegistry


def causal_program(*, program_id="color_alignment", color_operator="cycle_attribute"):
    variables = (
        CausalVariableSpec("object.color", "attribute", ("red", "blue")),
        CausalVariableSpec("target.color", "attribute", ("red", "blue")),
        CausalVariableSpec("pair.aligned", "relation", (False, True)),
        CausalVariableSpec("level.complete", "terminal", (False, True)),
        CausalVariableSpec("level.failed", "terminal", (False, True)),
    )
    color_parameters = {"action_name": "CLICK", "values": ("red", "blue")}
    if color_operator == "identity":
        color_parameters = {}
    mechanisms = (
        MechanismSpec(
            "object_color_transition",
            "object.color",
            (ParentRef("object.color"),),
            color_operator,
            color_parameters,
        ),
        MechanismSpec(
            "target_color_persistence",
            "target.color",
            (ParentRef("target.color"),),
            "identity",
        ),
        MechanismSpec(
            "alignment",
            "pair.aligned",
            (
                ParentRef("object.color", "next"),
                ParentRef("target.color", "next"),
            ),
            "align_relation",
        ),
        MechanismSpec(
            "completion",
            "level.complete",
            (ParentRef("pair.aligned", "next"),),
            "all_predicate",
        ),
        MechanismSpec(
            "failure_persistence",
            "level.failed",
            (ParentRef("level.failed"),),
            "identity",
        ),
    )
    return CausalProgram(
        program_id=program_id,
        bindings=BindingSpec({"object": "entity-1", "target": "entity-2"}),
        variables=variables,
        mechanisms=mechanisms,
        action_model=(ActionInterventionSpec("CLICK"),),
        goal=GoalSpec(
            "level.complete == true",
            ("pair.aligned == true",),
            "level.failed == true",
        ),
        description_length=5.0 if color_operator == "cycle_attribute" else 4.0,
    )


def initial_state():
    return CausalState(
        variables={
            "object.color": ValueDistribution.deterministic("red"),
            "target.color": ValueDistribution.deterministic("blue"),
            "pair.aligned": ValueDistribution.deterministic(False),
            "level.complete": ValueDistribution.deterministic(False),
            "level.failed": ValueDistribution.deterministic(False),
        },
        observation_hash="exact-prefix-hash",
    )


def test_program_roundtrip_compilation_and_dynamic_rollout():
    program = causal_program()
    reloaded = causal_program_from_json(causal_program_to_json(program))
    assert reloaded.canonical_hash == program.canonical_hash

    executor = CausalExecutor()
    compiled = executor.compile(reloaded, action_catalog=("CLICK",))
    assert compiled.topological_order.index("object.color") < compiled.topological_order.index("pair.aligned")
    assert compiled.topological_order.index("pair.aligned") < compiled.topological_order.index("level.complete")

    prediction = executor.predict_step(program, initial_state(), GroundedAction("CLICK"))
    assert prediction.state_after.value("object.color").mode == "blue"
    assert prediction.state_after.value("pair.aligned").mode is True
    assert prediction.goal_probability == 1.0
    assert prediction.terminal_probability == 1.0

    trace = executor.rollout(
        program,
        initial_state(),
        ActionProgram((GroundedAction("CLICK"),), source="causal_probe"),
        horizon=1,
    )
    assert trace.final_prediction.structured_signature == prediction.structured_signature


def test_do_operator_cuts_mechanism_and_preserves_non_descendants():
    executor = CausalExecutor()
    state = initial_state()
    prediction = executor.intervene(
        causal_program(),
        state,
        Intervention("object.color", ValueDistribution.deterministic("red")),
    )
    assert prediction.state_after.value("object.color").mode == "red"
    assert prediction.state_after.value("pair.aligned").mode is False
    assert prediction.state_after.value("level.complete").mode is False
    assert prediction.state_after.value("target.color").mode == "blue"
    assert prediction.state_after.value("level.failed").mode is False


def test_compiler_rejects_contemporaneous_cycle_and_missing_actions():
    program = causal_program()
    mechanisms = list(program.mechanisms)
    mechanisms[2] = replace(
        mechanisms[2],
        parent_variables=(ParentRef("level.complete", "next"),),
        operator_type="all_predicate",
    )
    cyclic = replace(program, program_id="cyclic", mechanisms=tuple(mechanisms))
    with pytest.raises(ValueError, match="cycle"):
        ProgramCompiler(MechanismRegistry()).compile(cyclic, action_catalog=("CLICK",))
    with pytest.raises(ValueError, match="unavailable actions"):
        ProgramCompiler(MechanismRegistry()).compile(program, action_catalog=("ACTION3",))


def test_neural_mechanism_requires_registered_module_or_explicit_fallback():
    program = causal_program()
    mechanisms = list(program.mechanisms)
    mechanisms[0] = replace(
        mechanisms[0],
        operator_type="neural_local_transition",
        neural_module_id="missing_module",
        symbolic_fallback=None,
    )
    unresolved = replace(program, program_id="unresolved", mechanisms=tuple(mechanisms))
    with pytest.raises(ValueError, match="unresolved mechanism"):
        CausalExecutor().compile(unresolved)

    mechanisms[0] = replace(mechanisms[0], symbolic_fallback="cycle_attribute")
    fallback = replace(program, program_id="fallback", mechanisms=tuple(mechanisms))
    assert CausalExecutor().predict_step(
        fallback, initial_state(), GroundedAction("CLICK")
    ).state_after.value("object.color").mode == "blue"


def test_action_position_mechanism_uses_complete_action_and_coordinates():
    registry = MechanismRegistry()
    spec = replace(
        causal_program().mechanisms[0],
        operator_type="action_position",
        parameters={
            "deltas_by_action": {"ACTION3": [0, -8], "ACTION4": [0, 6]},
            "ground_action": "ACTION6",
            "row_key": "y",
            "column_key": "x",
        },
    )
    current = ValueDistribution.deterministic([38, 22])
    left = registry.evaluate(
        spec,
        (current,),
        action=GroundedAction("ACTION3"),
        current_output=current,
    )
    click = registry.evaluate(
        spec,
        (current,),
        action=GroundedAction("ACTION6", {"x": 30, "y": 12}),
        current_output=current,
    )
    assert left.mode == [38.0, 14.0]
    assert click.mode == [12.0, 30.0]
