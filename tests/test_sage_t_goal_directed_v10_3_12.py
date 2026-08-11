from __future__ import annotations

from theory.sage_t.contracts import AbstractState, ActionCandidate
from theory.sage_t.goal_directed_v10_3_2 import (
    GoalDirectedOption,
    OptionStep,
    ProgressProgramRegistry,
)
from theory.sage_t.goal_directed_v10_3_12 import (
    OptionLocalAutomatonInducer,
    RELATIONAL_CAUSAL_ROLE,
    RelationalMechanismSageTController,
)
from theory.sage_t.relational_program_v10_3_12 import (
    ARMS,
    compile_candidate_registry,
)


def _relational_registry():
    return compile_candidate_registry(
        {
            "a": {
                "macro_schema": "repeat_target",
                "target_selector": "same_effect_distinct_target",
            },
            "b": {
                "macro_schema": "path_successor",
                "target_selector": "successor_toward_enclosure",
            },
        },
        repeat_projection_identified=True,
    )


def test_effect_alignment_is_option_local_for_both_mechanisms() -> None:
    for schema in ("repeat_target", "path_successor"):
        option = GoalDirectedOption(
            schema=schema,
            steps=tuple(OptionStep("ACTION6") for _ in range(3)),
        )
        hashes = set()
        for prefix in (0, 4, 17):
            effects = tuple(f"prefix_{index}" for index in range(prefix)) + (
                "local_a",
                "local_b",
                "local_c",
            )
            learned = OptionLocalAutomatonInducer.align_effects(
                option, effects, option_start=prefix
            )
            hashes.add(learned.option_id)
            assert tuple(step.expected_effect for step in learned.steps) == (
                "local_a",
                "local_b",
                "local_c",
            )
        assert len(hashes) == 1


def test_factorized_repeat_role_uses_unique_boundary_relation() -> None:
    controller = RelationalMechanismSageTController(
        phase="preflight",
        registry=ProgressProgramRegistry(),
        arm=ARMS[0],
        relational_registry=_relational_registry(),
        exploration_seed=3521,
        goal_conditioning_enabled=False,
    )
    candidates = (
        ActionCandidate("ACTION6", {"x": 56, "y": 29}),
        ActionCandidate("ACTION6", {"x": 4, "y": 29}),
    )
    option = controller._choose_option(AbstractState(), candidates)
    assert option is not None
    assert option.steps[0].binding_method == RELATIONAL_CAUSAL_ROLE
    controller._active_option = option
    controller._active_cursor = 0
    selected = controller._continue_active_option(AbstractState(), candidates)
    assert selected is not None
    assert dict(selected.action_data) == {"x": 4, "y": 29}


def test_arms_use_fresh_support_zero_registries() -> None:
    relational = _relational_registry()
    for arm in ARMS:
        controller = RelationalMechanismSageTController(
            phase="preflight",
            registry=ProgressProgramRegistry(),
            arm=arm,
            relational_registry=relational,
            exploration_seed=3522,
            goal_conditioning_enabled=False,
        )
        assert controller.registry.snapshot()["programs"] == []
        assert controller.summary()["cross_reset_memory"] is False
        assert controller.summary()["grounded_arguments_persisted"] is False

