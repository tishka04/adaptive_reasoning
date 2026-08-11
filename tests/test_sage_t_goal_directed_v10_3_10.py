from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.sage_t.goal_directed_v10_3_10 import (
    MAXIMUM_DIRECTIONAL_OPTION_HORIZON,
    MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN,
    DirectionalProgressAutomatonInducer,
    directional_milestone_descriptor,
    directional_milestone_signature,
)


def _record(action: str, index: int, *, progressed: bool = False):
    before = np.zeros((7, 7), dtype=np.int16)
    after = before.copy()
    after[1, 1] = index + 1
    return SimpleNamespace(
        action=SimpleNamespace(name=action),
        diff=SimpleNamespace(
            level_complete=progressed,
            game_over=False,
            is_noop=False,
            moved_objects=((index, (0, 0), (0, 1)),),
            created_objects=(),
            removed_objects=(),
            num_changed=1,
            player_displacement=(0, 1),
        ),
        obs_before=SimpleNamespace(
            levels_completed=0,
            grid_hash=index,
            raw_grid=before,
            objects=(object(),),
            available_actions=("ACTION1", "ACTION2", "ACTION3"),
        ),
        obs_after=SimpleNamespace(
            levels_completed=int(progressed),
            grid_hash=index + 1,
            raw_grid=after,
            objects=(object(),),
            available_actions=("ACTION1", "ACTION2", "ACTION3"),
        ),
    )


def _maximum_run(actions: tuple[str, ...]) -> int:
    maximum = 0
    previous = None
    run = 0
    for action in actions:
        if action == previous:
            run += 1
        else:
            previous = action
            run = 1
        maximum = max(maximum, run)
    return maximum


def test_directional_descriptor_ignores_raw_state_hash_and_values() -> None:
    first = _record("ACTION1", 1)
    second = _record("ACTION1", 99)

    assert directional_milestone_signature(first) == directional_milestone_signature(
        second
    )
    assert set(directional_milestone_descriptor(first)) == {
        "mode",
        "level_progress",
        "terminal",
        "noop",
        "actor_axis",
        "component_delta_sign",
        "action_space_delta_sign",
        "shape_changed",
        "object_set_changed",
        "actor_or_object_moved",
    }


def test_repeated_action_effect_becomes_stall_not_fresh_progress() -> None:
    inducer = DirectionalProgressAutomatonInducer()
    inducer.start_branch()
    for index in range(3):
        inducer.observe(
            _record("ACTION1", index),
            selected_step=None,
            active_option=None,
        )

    summary = inducer.summary()
    assert summary["directional_gain_events"] == 1
    assert summary["repeated_effect_events"] == 2
    assert inducer.last_transition_stalled is True


def test_frontier_updates_simulated_counts_and_bounds_identical_run() -> None:
    inducer = DirectionalProgressAutomatonInducer()
    inducer.start_branch()
    for index in range(3):
        inducer.observe(
            _record("ACTION1", index),
            selected_step=None,
            active_option=None,
        )

    option = inducer.compose_frontier(
        ("ACTION1", "ACTION2", "ACTION3"), rotation=0, horizon=32
    )

    assert option is not None and option.mixed
    assert len(option.steps) == MAXIMUM_DIRECTIONAL_OPTION_HORIZON
    assert (
        _maximum_run(option.action_schemas)
        <= MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN
    )
    assert option.source == "reset_local_directional_subgoal_frontier"


def test_only_level_transition_produces_success_option() -> None:
    inducer = DirectionalProgressAutomatonInducer()
    inducer.start_branch()
    assert (
        inducer.observe(
            _record("ACTION1", 0),
            selected_step=None,
            active_option=None,
        )
        is None
    )
    learned = inducer.observe(
        _record("ACTION2", 1, progressed=True),
        selected_step=None,
        active_option=None,
    )

    assert learned is not None
    assert learned.mixed
    serialized = str(learned.safe_payload)
    assert all(
        token not in serialized
        for token in ("game_id", "seed", "raw_grid", "entity_id", "color")
    )

