"""SAGE.10f transferable causal-schema and first-terminal bridge tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from theory.live_transition_loop import build_observation
from theory.online_transferable_causal_schema import (
    FrozenCausalSchemaLibrary,
    OnlineCausalSchemaExporter,
    OnlineCausalSchemaTransfer,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=("ACTION6",),
        game_state="NOT_FINISHED",
        levels_completed=0,
        infer_players=False,
    )


def _square(
    *,
    color: int,
    row: int,
    column: int,
) -> np.ndarray:
    grid = np.zeros((9, 11), dtype=np.int32)
    grid[row:row + 2, column:column + 2] = color
    return grid


def _one_step_library():
    exporter = OnlineCausalSchemaExporter(source_tag="frozen-source")
    exporter.start_branch()
    before = _square(color=2, row=1, column=1)
    after = np.zeros_like(before)
    exporter.observe_transition(
        observation_before=_observation(before),
        observation_after=_observation(after),
        action_name="ACTION6",
        action_data={"x": 1, "y": 1},
        terminal_success=True,
        game_over=False,
    )
    return exporter, exporter.freeze()


def test_frozen_schema_omits_game_palette_coordinates_and_state_hashes():
    exporter, library = _one_step_library()

    payload = library.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    schema = payload["schemas"][0]

    assert payload["frozen"] is True
    assert payload["schema_count"] == 1
    assert "game_id" not in serialized
    assert "grid_hash" not in serialized
    assert '"x"' not in serialized
    assert '"y"' not in serialized
    assert schema["steps"][0]["action_family"] == "point"
    assert schema["steps"][0]["next_subgoal"] == "terminal_progress"

    frozen_payload = library.to_dict()
    exporter.start_branch()
    moved = _square(color=7, row=4, column=6)
    exporter.observe_transition(
        observation_before=_observation(moved),
        observation_after=_observation(np.zeros_like(moved)),
        action_name="ACTION6",
        action_data={"x": 6, "y": 4},
        terminal_success=True,
        game_over=False,
    )
    assert library.to_dict() == frozen_payload
    assert (
        FrozenCausalSchemaLibrary.from_dict(frozen_payload).to_dict()
        == frozen_payload
    )
    tampered = json.loads(json.dumps(frozen_payload))
    tampered["schemas"][0]["steps"][0]["target_role"] = "tampered"
    with pytest.raises(ValueError, match="does not match"):
        FrozenCausalSchemaLibrary.from_dict(tampered)


def test_source_schema_stays_probe_only_until_local_terminal_promotion():
    _, library = _one_step_library()
    transfer = OnlineCausalSchemaTransfer(
        library,
        local_effect_confirmation_threshold=2,
    )
    empty = np.zeros((9, 11), dtype=np.int32)

    for branch, (color, row, column) in enumerate((
        (7, 4, 5),
        (9, 6, 7),
    )):
        transfer.start_branch()
        before = _square(color=color, row=row, column=column)
        action_data = {"x": column, "y": row}
        selection = transfer.select(
            observation=_observation(before),
            available_actions=("ACTION6",),
            available_action_candidates=(
                SimpleNamespace(
                    name="ACTION6",
                    action_args=action_data,
                ),
            ),
            experiment_eligible=True,
        )
        assert selection is not None
        assert selection.promoted is False
        transfer.observe_transition(
            observation_before=_observation(before),
            observation_after=_observation(empty),
            action_name=selection.action_name,
            action_data=selection.action_data,
            terminal_success=False,
            game_over=False,
            no_effect=False,
        )
        assert transfer.summary()["promotions"] == 0, branch

    assert transfer.summary()["locally_confirmed_steps"] == 1

    transfer.start_branch()
    before = _square(color=4, row=2, column=7)
    action_data = {"x": 7, "y": 2}
    terminal_probe = transfer.select(
        observation=_observation(before),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(name="ACTION6", action_args=action_data),
        ),
        experiment_eligible=True,
    )
    assert terminal_probe is not None
    assert terminal_probe.promoted is False
    outcome = transfer.observe_transition(
        observation_before=_observation(before),
        observation_after=_observation(empty),
        action_name=terminal_probe.action_name,
        action_data=terminal_probe.action_data,
        terminal_success=True,
        game_over=False,
        no_effect=False,
    )

    assert outcome["terminal_backcredited_schemas"]
    assert transfer.summary()["promoted_schemas"] == 1

    transfer.start_branch()
    promoted = transfer.select(
        observation=_observation(before),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(name="ACTION6", action_args=action_data),
        ),
        experiment_eligible=False,
    )
    assert promoted is not None
    assert promoted.promoted is True


def test_repeatable_effects_unlock_short_chain_and_terminal_backcredit():
    exporter = OnlineCausalSchemaExporter(
        max_steps_per_schema=3,
    )
    exporter.start_branch()
    source_before = _square(color=2, row=1, column=1)
    source_middle = _square(color=2, row=3, column=4)
    empty = np.zeros_like(source_before)
    exporter.observe_transition(
        observation_before=_observation(source_before),
        observation_after=_observation(source_middle),
        action_name="ACTION6",
        action_data={"x": 1, "y": 1},
        terminal_success=False,
        game_over=False,
    )
    exporter.observe_transition(
        observation_before=_observation(source_middle),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data={"x": 4, "y": 3},
        terminal_success=True,
        game_over=False,
    )
    library = exporter.freeze()
    assert len(library.schemas[0].steps) == 2

    transfer = OnlineCausalSchemaTransfer(
        library,
        local_effect_confirmation_threshold=1,
    )
    transfer.start_branch()
    target_before = _square(color=8, row=4, column=2)
    target_middle = _square(color=8, row=6, column=6)
    first = transfer.select(
        observation=_observation(target_before),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(
                name="ACTION6",
                action_args={"x": 2, "y": 4},
            ),
        ),
        experiment_eligible=True,
    )
    assert first is not None
    assert first.step_index == 0
    transfer.observe_transition(
        observation_before=_observation(target_before),
        observation_after=_observation(target_middle),
        action_name=first.action_name,
        action_data=first.action_data,
        terminal_success=False,
        game_over=False,
        no_effect=False,
    )

    second = transfer.select(
        observation=_observation(target_middle),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(
                name="ACTION6",
                action_args={"x": 6, "y": 6},
            ),
        ),
        experiment_eligible=True,
    )
    assert second is not None
    assert second.step_index == 1
    transfer.observe_transition(
        observation_before=_observation(target_middle),
        observation_after=_observation(empty),
        action_name=second.action_name,
        action_data=second.action_data,
        terminal_success=True,
        game_over=False,
        no_effect=False,
    )

    summary = transfer.summary()
    assert summary["chain_advances"] == 1
    assert summary["terminal_backcredits"] == 1
    assert summary["promotions"] == 1


def test_unproductive_schema_probe_is_demoted_per_context():
    _, library = _one_step_library()
    transfer = OnlineCausalSchemaTransfer(
        library,
        local_effect_confirmation_threshold=1,
        nonprogress_demotion_threshold=2,
    )
    before = _square(color=7, row=4, column=5)
    candidate = SimpleNamespace(
        name="ACTION6",
        action_args={"x": 5, "y": 4},
    )

    for _ in range(2):
        transfer.start_branch()
        selection = transfer.select(
            observation=_observation(before),
            available_actions=("ACTION6",),
            available_action_candidates=(candidate,),
            experiment_eligible=True,
        )
        assert selection is not None
        transfer.observe_transition(
            observation_before=_observation(before),
            observation_after=_observation(before.copy()),
            action_name=selection.action_name,
            action_data=selection.action_data,
            terminal_success=False,
            game_over=False,
            no_effect=True,
        )

    transfer.start_branch()
    assert transfer.select(
        observation=_observation(before),
        available_actions=("ACTION6",),
        available_action_candidates=(candidate,),
        experiment_eligible=True,
    ) is None
    summary = transfer.summary()
    assert summary["demotions"] == 1
    assert summary["demotion_blocks"] >= 1


def test_cross_family_adapter_probe_requires_target_effect_confirmation():
    _, library = _one_step_library()
    transfer = OnlineCausalSchemaTransfer(
        library,
        local_effect_confirmation_threshold=1,
    )
    transfer.start_branch()
    before = _square(color=7, row=4, column=5)
    selection = transfer.select(
        observation=_observation(before),
        available_actions=("ACTION1",),
        available_action_candidates=(
            SimpleNamespace(name="ACTION1", action_args={}),
        ),
        experiment_eligible=True,
    )

    assert selection is not None
    assert selection.source_action_family == "point"
    assert selection.action_family == "primitive:ACTION1"
    outcome = transfer.observe_transition(
        observation_before=_observation(before),
        observation_after=_observation(np.zeros_like(before)),
        action_name=selection.action_name,
        action_data=selection.action_data,
        terminal_success=False,
        game_over=False,
        no_effect=False,
    )
    assert outcome["effect_matched"] is True
    summary = transfer.summary()
    assert summary["cross_family_adapter_probes"] == 1
    assert summary["cross_family_adapter_confirmations"] == 1
    assert summary["promotions"] == 0


def test_unified_controller_uses_frozen_schema_as_bounded_probe(
    monkeypatch,
):
    _, library = _one_step_library()
    controller = UnifiedCognitiveController(
        "target-game-name-is-not-transfer-data",
        available_actions=["ACTION6"],
        frozen_causal_schema_library=library,
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=0,
            enable_active_goal_hypotheses=False,
            enable_operator_planning=False,
            enable_theory_planning=False,
            enable_terminal_negative_frontier_exploration=False,
            enable_terminal_relational_stencil_induction=False,
            enable_terminal_multiform_relational_induction=False,
            frontier_exploration_min_stagnant_steps=1,
            frontier_exploration_min_failed_branches=0,
        ),
    )
    monkeypatch.setattr(
        controller.progress,
        "branch_diagnostics",
        lambda: {
            "branch_id": 0,
            "branch_actions": 8,
            "actions_since_terminal_improvement": 8,
            "max_hash_repeat": 1,
            "max_diff_repeat": 1,
            "unique_states_in_window": 8,
            "window_actions": 8,
        },
    )
    before = _square(color=7, row=4, column=5)
    candidate = SimpleNamespace(
        name="ACTION6",
        action_args={"x": 5, "y": 4},
    )

    decision = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        available_action_candidates=(candidate,),
        legacy_action="ACTION6",
        legacy_action_data={"x": 0, "y": 0},
    )

    assert decision.source == "transfer_causal_schema_probe"
    assert decision.transfer_causal_schema is True
    assert decision.transfer_causal_schema_experiment is True
    assert decision.transfer_causal_schema_promoted is False
    controller.observe_transition(
        action=decision.action_name,
        action_data=decision.action_data,
        grid_before=before,
        grid_after=np.zeros_like(before),
        available_actions=["ACTION6"],
    )
    summary = controller.summary()
    assert (
        summary["transferable_causal_schema_transfer"][
            "effect_confirmations"
        ]
        == 1
    )
    assert (
        summary["transferable_causal_schema_transfer"][
            "promoted_schemas"
        ]
        == 0
    )
