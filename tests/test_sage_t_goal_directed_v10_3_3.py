from __future__ import annotations

import json

import numpy as np

from theory.sage_t.contracts import AbstractEntity, AbstractState, ActionCandidate
from theory.sage_t.goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from theory.sage_t.goal_directed_v10_3_3 import (
    BRANCH_PRODUCTIVE_ANCHOR,
    DYNAMIC_SUCCESSOR,
    RelationalGoalDirectedSageTController,
)
from v3.schemas import FrameDiff, GameObservation, PrimitiveAction, TransitionRecord


def _observation(grid: np.ndarray) -> GameObservation:
    return GameObservation(
        raw_grid=grid,
        grid_hash=hash(grid.tobytes()),
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=["ACTION6"],
    )


def _productive_record() -> TransitionRecord:
    before = np.zeros((5, 5), dtype=np.int16)
    after = before.copy()
    after[1, 1] = 2
    return TransitionRecord(
        action=PrimitiveAction("ACTION6", x=1, y=1),
        obs_before=_observation(before),
        obs_after=_observation(after),
        diff=FrameDiff(
            changed_cells=[(1, 1)],
            changed_values_before=[0],
            changed_values_after=[2],
            created_objects=[],
            removed_objects=[],
            moved_objects=[],
            num_changed=1,
        ),
        timestamp=0,
    )


def test_collision_is_not_claimed_unique_and_branch_anchor_reacquires_target() -> None:
    controller = RelationalGoalDirectedSageTController(
        phase="preflight", warmup_actions=0
    )
    controller.start_branch()
    state = AbstractState(
        entities=(AbstractEntity("ephemeral", ("target",), center=(1, 1)),)
    )
    candidates = (
        ActionCandidate("ACTION6", {"x": 1, "y": 1}),
        ActionCandidate("ACTION6", {"x": 2, "y": 1}),
    )
    assert controller._choose_option(state, candidates) is None
    assert controller.summary()["structural_collision_count"] >= 2

    controller.observe_transition(_productive_record())
    option = controller._choose_option(state, candidates)
    assert option is not None
    assert option.steps[0].binding_method == BRANCH_PRODUCTIVE_ANCHOR
    controller._active_option = option
    selected = controller._continue_active_option(state, candidates)
    assert selected is not None
    assert dict(selected.action_data) == {"x": 1, "y": 1}

    encoded = json.dumps(option.safe_payload, sort_keys=True)
    assert '"x"' not in encoded
    assert '"y"' not in encoded
    assert "ephemeral" not in encoded


def test_reset_drops_every_ephemeral_anchor() -> None:
    controller = RelationalGoalDirectedSageTController(phase="preflight")
    controller.start_branch()
    controller.observe_transition(_productive_record())
    assert controller.summary()["ephemeral_anchor_count"] == 1
    controller.start_branch()
    assert controller.summary()["ephemeral_anchor_count"] == 0
    assert controller.summary()["ephemeral_action_data_persisted"] is False


def test_stagnant_protected_route_and_asymmetric_veto_cannot_block_same_action() -> None:
    controller = RelationalGoalDirectedSageTController(
        phase="preflight", warmup_actions=0
    )
    controller.start_branch()
    controller._consecutive_sterile_transitions = 4
    controller._productive_anchor_by_action["ACTION6"] = {"x": 1, "y": 1}
    controller._active_option = GoalDirectedOption(
        schema="repeat_target",
        steps=(
            OptionStep(
                "ACTION6",
                binding_method=BRANCH_PRODUCTIVE_ANCHOR,
            ),
        ),
    )
    grid = np.zeros((5, 5), dtype=np.int16)
    arbitration = controller.decide(
        symbolic_action_name="ACTION6",
        symbolic_action_data={"x": 2, "y": 1},
        observation=_observation(grid),
        legal_actions=(
            ActionCandidate("ACTION6", {"x": 1, "y": 1}),
            ActionCandidate("ACTION6", {"x": 2, "y": 1}),
        ),
        protected_route=True,
        danger_veto=lambda _candidate: True,
    )
    assert arbitration.applied is True
    assert arbitration.action_data == {"x": 1, "y": 1}


def test_dynamic_successor_replans_from_live_candidates_without_persisting_path() -> None:
    chain = tuple(
        AbstractEntity(
            f"dot_{index}",
            ("object", "target"),
            attributes=(("area", "one"), ("aspect", "square")),
            center=(float(20 - 2 * index), float(2 * index)),
        )
        for index in range(8)
    )
    state = AbstractState(
        entities=(
            AbstractEntity(
                "source",
                ("object", "movable", "target"),
                attributes=(("area", "medium"),),
                center=(22.0, -2.0),
            ),
            AbstractEntity(
                "enclosure",
                ("object", "target"),
                attributes=(("area", "large"),),
                center=(6.0, 14.0),
            ),
            *chain,
        )
    )
    candidates = tuple(
        ActionCandidate("ACTION6", {"x": x, "y": y})
        for x in range(0, 17, 2)
        for y in range(4, 23, 2)
    )
    option = GoalDirectedOption(
        schema="path_successor",
        steps=(OptionStep("ACTION6", binding_method=DYNAMIC_SUCCESSOR),),
    )
    controller = RelationalGoalDirectedSageTController(phase="preflight")
    controller.start_branch()
    controller._active_option = option
    selected = controller._continue_active_option(state, candidates)
    assert selected in candidates
    encoded = json.dumps(option.safe_payload, sort_keys=True)
    assert '"x"' not in encoded
    assert '"y"' not in encoded
