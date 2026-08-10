from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from theory.sage_t.contracts import AbstractEntity, AbstractState, ActionCandidate
from theory.sage_t.goal_directed_v10_3_2 import (
    ExtendedOptionEvaluator,
    GoalDirectedOption,
    GoalDirectedSageTController,
    OptionStep,
    ProgressProgramRegistry,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: dict


def _promoted_registry() -> tuple[ProgressProgramRegistry, GoalDirectedOption]:
    option = GoalDirectedOption(
        schema="repeat_target",
        steps=tuple(OptionStep("ACTION1") for _ in range(8)),
        source="synthetic_reproduction",
    )
    registry = ProgressProgramRegistry()
    registry.note_success(option, "independent-reset-a")
    registry.note_success(option, "independent-reset-b")
    registry.note_controls(
        option.option_id,
        binding_swap=True,
        order_permutation=True,
        automaton_ablation=True,
    )
    return registry, option


def _controller(
    registry: ProgressProgramRegistry, checksum: str
) -> tuple[UnifiedCognitiveController, GoalDirectedSageTController]:
    sage_t = GoalDirectedSageTController(
        phase="confirmation",
        registry=registry,
        registry_checksum=checksum,
        attestation_scope="fresh-confirmation-reset",
    )
    unified = UnifiedCognitiveController(
        "synthetic-source",
        config=UnifiedCognitiveConfig(
            sage_t_authority_mode="active",
            sage_t_counterfactual_gate_passed=True,
            sage_t_active_gate_passed=True,
        ),
        sage_t_controller=sage_t,
    )
    unified.on_reset()
    return unified, sage_t


def test_registry_is_transfer_safe_and_enters_each_reset_with_zero_support() -> None:
    registry, option = _promoted_registry()
    snapshot = registry.snapshot(promoted_only=True)
    encoded = json.dumps(snapshot, sort_keys=True)
    for forbidden in (
        '"game_id"',
        '"seed"',
        '"x"',
        '"y"',
        '"color"',
        '"raw_grid"',
        '"entity_id"',
        '"object_id"',
    ):
        assert forbidden not in encoded

    transferred_a = ProgressProgramRegistry(snapshot)
    transferred_b = ProgressProgramRegistry(snapshot)
    assert transferred_a.local_support(option.option_id) == 0
    assert transferred_b.local_support(option.option_id) == 0
    assert transferred_a.transferred_options() == (option,)


def test_reproduction_requires_two_independent_reset_scopes() -> None:
    option = GoalDirectedOption(
        schema="repeat_target",
        steps=(OptionStep("ACTION1"), OptionStep("ACTION1")),
    )
    registry = ProgressProgramRegistry()
    registry.note_success(option, "event-a", scope="same-reset")
    registry.note_success(option, "event-b", scope="same-reset")
    registry.note_controls(
        option.option_id,
        binding_swap=True,
        order_permutation=True,
        automaton_ablation=True,
    )
    assert registry.transferred_options() == ()
    assert registry.reproduction_candidates() == (option,)

    controller = GoalDirectedSageTController(
        phase="discovery", registry=registry, warmup_actions=0
    )
    selected = controller._choose_option(
        AbstractState(), (ActionCandidate("ACTION1"), ActionCandidate("ACTION2"))
    )
    assert selected == option
    registry.note_success(option, "event-c", scope="independent-reset")
    assert registry.transferred_options() == (option,)


def test_confirmation_requires_checksum_and_loaded_registry_changes_action() -> None:
    registry, option = _promoted_registry()
    snapshot = registry.snapshot(promoted_only=True)
    transferred = ProgressProgramRegistry(snapshot)
    with pytest.raises(ValueError, match="requires a registry checksum"):
        GoalDirectedSageTController(phase="confirmation", registry=transferred)

    unified, sage_t = _controller(transferred, snapshot["registry_checksum"])
    legal = (_Action("ACTION1", {}), _Action("ACTION2", {}))
    grid = np.zeros((5, 5), dtype=np.int16)
    decision = unified.select_action(
        current_grid=grid,
        available_actions=("ACTION1", "ACTION2"),
        legacy_action="ACTION2",
        legacy_action_data={},
        available_action_candidates=legal,
        game_state="NOT_FINISHED",
        levels_completed=0,
    )
    assert decision.source == "sage_t_joint_program"
    assert decision.action_name == option.steps[0].action_name
    assert sage_t.last_decision_registry_checksum == snapshot["registry_checksum"]

    after = grid.copy()
    after[2, 2] = 1
    unified.observe_transition(
        action=decision.action_name,
        action_data=decision.action_data,
        grid_before=grid,
        grid_after=after,
        available_actions=("ACTION1", "ACTION2"),
        levels_completed_before=0,
        levels_completed_after=0,
    )
    assert unified.summary()["transitions_observed"] == 1
    assert sage_t.summary()["registry_used_in_decision"] is True


def test_four_sterile_effects_demote_transferred_option_locally() -> None:
    registry, option = _promoted_registry()
    snapshot = registry.snapshot(promoted_only=True)
    transferred = ProgressProgramRegistry(snapshot)
    unified, sage_t = _controller(transferred, snapshot["registry_checksum"])
    legal = (_Action("ACTION1", {}), _Action("ACTION2", {}))
    grid = np.zeros((5, 5), dtype=np.int16)
    executed = 0
    for _ in range(4):
        decision = unified.select_action(
            current_grid=grid,
            available_actions=("ACTION1", "ACTION2"),
            legacy_action="ACTION2",
            legacy_action_data={},
            available_action_candidates=legal,
            game_state="NOT_FINISHED",
            levels_completed=0,
        )
        if decision.source != "sage_t_joint_program":
            break
        executed += 1
        unified.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=grid,
            grid_after=grid.copy(),
            available_actions=("ACTION1", "ACTION2"),
            levels_completed_before=0,
            levels_completed_after=0,
        )
    assert 1 <= executed <= 4
    assert transferred.local_contradictions(option.option_id) == 1
    assert transferred.eligible_transferred_options() == ()
    assert sage_t.summary()["option_contradictions"] == 1


def test_terminal_transition_stops_and_demotes_active_option() -> None:
    registry, option = _promoted_registry()
    snapshot = registry.snapshot(promoted_only=True)
    transferred = ProgressProgramRegistry(snapshot)
    unified, sage_t = _controller(transferred, snapshot["registry_checksum"])
    grid = np.zeros((5, 5), dtype=np.int16)
    legal = (_Action("ACTION1", {}), _Action("ACTION2", {}))
    decision = unified.select_action(
        current_grid=grid,
        available_actions=("ACTION1", "ACTION2"),
        legacy_action="ACTION2",
        legacy_action_data={},
        available_action_candidates=legal,
        game_state="NOT_FINISHED",
        levels_completed=0,
    )
    unified.observe_transition(
        action=decision.action_name,
        action_data=decision.action_data,
        grid_before=grid,
        grid_after=grid.copy(),
        available_actions=("ACTION1", "ACTION2"),
        game_state_before="NOT_FINISHED",
        game_state_after="GAME_OVER",
        levels_completed_before=0,
        levels_completed_after=0,
    )
    assert transferred.local_contradictions(option.option_id) == 1
    assert sage_t.summary()["active_option_id"] is None


def test_extended_evaluator_preserves_full_32_action_mixed_option() -> None:
    option = GoalDirectedOption(
        schema="mixed_automaton",
        steps=tuple(
            OptionStep("ACTION1" if index % 2 == 0 else "ACTION2")
            for index in range(32)
        ),
    )
    assessments = ExtendedOptionEvaluator().assess(
        (option,), registry=ProgressProgramRegistry(), tried_option_ids=set()
    )
    assert assessments[0].option == option
    assert len(assessments[0].option.steps) == 32
    assert assessments[0].option.mixed is True


def test_parameterized_option_is_generated_from_transient_anchor_only() -> None:
    controller = GoalDirectedSageTController(
        phase="discovery", warmup_actions=0
    )
    state = AbstractState(
        entities=(AbstractEntity("branch-local", ("target",), center=(2, 3)),)
    )
    option = controller._choose_option(
        state,
        (
            ActionCandidate("ACTION1"),
            ActionCandidate("ACTION6", {"x": 3, "y": 2}),
        ),
    )
    assert option is not None
    encoded = json.dumps(option.safe_payload, sort_keys=True)
    assert '"x"' not in encoded
    assert '"y"' not in encoded
    assert "branch-local" not in encoded
