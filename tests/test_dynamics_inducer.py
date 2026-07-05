import numpy as np

from v4_1_reasoning_system.arc_agi.dynamics_inducer import DynamicsInducer
from v4_1_reasoning_system.arc_agi.game_memory import ActionProfile, GameMemory
from v4_1_reasoning_system.arc_agi.grid_analyzer import GridAnalyzer
from v4_1_reasoning_system.arc_agi.state_describer import GameObservation


def _fake_observation() -> GameObservation:
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[1, 1] = 2
    grid[2, 2] = 3
    return GameObservation(
        grid_description="test",
        objects=[
            {"value": 2, "size": 1, "center_y": 1, "center_x": 1, "is_player": True},
            {"value": 3, "size": 1, "center_y": 2, "center_x": 2, "is_player": False},
        ],
        player_info={"y": 1, "x": 1, "value": 2, "confidence": 0.9},
        action_semantics={},
        memory_summary={},
        raw_grid=grid,
        level=0,
        game_state="NOT_FINISHED",
        action_counter=0,
    )


def test_dynamics_inducer_classifies_movement_switch_click_and_unknown():
    memory = GameMemory()
    movement = ActionProfile(action_name="ACTION1", times_tried=5, times_changed_grid=5, times_moved_player=5)
    movement.displacements = [(-1.0, 0.0), (-1.0, 0.0)]
    switch = ActionProfile(action_name="ACTION5", times_tried=4, times_changed_grid=3)
    click = ActionProfile(action_name="ACTION6", times_tried=3, times_changed_grid=2)
    memory.action_profiles.update({
        "ACTION1": movement,
        "ACTION5": switch,
        "ACTION6": click,
    })
    memory.effective_click_values.add(3)

    belief = DynamicsInducer().induce(
        memory=memory,
        observation=_fake_observation(),
        available_actions=["ACTION1", "ACTION5", "ACTION6", "ACTION7"],
    )
    by_action = belief.by_action()

    assert by_action["ACTION1"].kind == "movement"
    assert by_action["ACTION5"].kind == "control_switch"
    assert by_action["ACTION6"].kind == "click_activation"
    assert by_action["ACTION7"].kind == "unknown"
    assert by_action["ACTION7"].information_gain > by_action["ACTION1"].information_gain


def test_dynamics_inducer_adds_low_confidence_affordance_priors_for_untried_actions():
    belief = DynamicsInducer().induce(
        memory=GameMemory(),
        observation=_fake_observation(),
        available_actions=["ACTION5", "ACTION6", "ACTION7"],
    )
    by_action = belief.by_action()

    assert by_action["ACTION5"].kind == "control_switch"
    assert by_action["ACTION5"].confidence < 0.25
    assert by_action["ACTION5"].information_gain == 1.0
    assert by_action["ACTION5"].evidence["affordance_prior"] == "interaction_probe"
    assert by_action["ACTION6"].kind == "click_activation"
    assert by_action["ACTION7"].kind == "unknown"


def test_game_memory_records_richer_action_effect_features_and_level_change():
    memory = GameMemory()
    before = np.zeros((3, 3), dtype=np.int32)
    after = before.copy()
    before[1, 1] = 2
    after[0, 1] = 2
    diff = GridAnalyzer.compute_diff(before, after)

    effect = memory.record_action(
        "ACTION1",
        before,
        after,
        diff,
        game_state="NOT_FINISHED",
        levels_completed=1,
    )

    assert effect.level_changed is True
    assert effect.changed_values_before == [0, 2]
    assert effect.changed_values_after == [0, 2]
    assert effect.moved_values == [2]
    assert effect.candidate_control_value == 2
