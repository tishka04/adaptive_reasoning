from __future__ import annotations

from types import SimpleNamespace

from theory.sage_t.contracts import ActionCandidate
from theory.sage_t.goal_directed_v10_3_2 import OptionStep
from theory.sage_t.goal_directed_v10_3_9 import (
    CausalSubgoalAutomatonInducer,
    CausalSubgoalSageTController,
    robust_effect_descriptor,
)


def _record(action: str, index: int, *, progressed: bool = False, displacement=(0, 1)):
    return SimpleNamespace(
        action=SimpleNamespace(name=action),
        diff=SimpleNamespace(
            level_complete=progressed,
            game_over=False,
            is_noop=False,
            moved_objects=(index,),
            created_objects=(),
            removed_objects=(),
            num_changed=4,
            player_displacement=displacement,
        ),
        obs_before=SimpleNamespace(levels_completed=0),
        obs_after=SimpleNamespace(levels_completed=int(progressed)),
    )


def test_effect_descriptor_accepts_extended_and_short_displacements() -> None:
    extended = robust_effect_descriptor(
        _record("ACTION1", 0, displacement=("actor", 3, -1))
    )
    short = robust_effect_descriptor(_record("ACTION1", 0, displacement=(1,)))

    assert extended["actor_axis"] == "vertical"
    assert short["actor_axis"] == "unknown"
    assert set(extended) == {
        "mode", "noop", "terminal", "level_progress", "moved_bucket",
        "created_bucket", "removed_bucket", "changed_bucket", "actor_axis",
    }


def test_causal_inducer_composes_effect_driven_mixed_frontier() -> None:
    inducer = CausalSubgoalAutomatonInducer()
    inducer.start_branch()
    for index, action in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION1")):
        assert (
            inducer.observe(
                _record(action, index),
                selected_step=OptionStep(action),
                active_option=None,
            )
            is None
        )

    option = inducer.compose_frontier(
        ("ACTION1", "ACTION2", "ACTION3"), rotation=1
    )

    assert option is not None
    assert option.mixed
    assert len(option.steps) >= 16
    assert option.source == "reset_local_causal_subgoal_frontier"
    assert inducer.summary()["effect_graph_edges"] >= 3
    assert inducer.summary()["observation_rejections"] == 0


def test_seed_changes_schema_probe_order_without_entering_program() -> None:
    candidates = tuple(
        ActionCandidate(f"ACTION{index}", {}) for index in range(1, 6)
    )
    left = CausalSubgoalSageTController(phase="preflight", exploration_seed=3361)
    right = CausalSubgoalSageTController(phase="preflight", exploration_seed=3362)

    left_option = left._probe_option(candidates)
    right_option = right._probe_option(candidates)

    assert left_option is not None and right_option is not None
    assert left_option.steps[0].action_name != right_option.steps[0].action_name
    assert "3361" not in str(left_option.safe_payload)
    assert "3362" not in str(right_option.safe_payload)


def test_level_progress_induces_mixed_option_from_branch_suffix() -> None:
    inducer = CausalSubgoalAutomatonInducer()
    inducer.start_branch()
    learned = None
    actions = ("ACTION1", "ACTION2", "ACTION3", "ACTION2")
    for index, action in enumerate(actions):
        learned = inducer.observe(
            _record(action, index, progressed=index == len(actions) - 1),
            selected_step=OptionStep(action),
            active_option=None,
        )

    assert learned is not None
    assert learned.mixed
    assert learned.action_schemas == actions
    serialized = str(learned.safe_payload)
    assert all(
        token not in serialized
        for token in ("game_id", "seed", "raw_grid", "entity_id", "color")
    )
