"""A7 real-env option adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from theory.ar25_oracle import build_ar25_oracle
from theory.correspondence_hypothesis import (
    CorrespondenceHypothesis,
    CorrespondenceObservation,
    correspondence_key,
)
from theory.live_transition_loop import LiveTransitionBeliefLoop
from theory.mechanic_hypothesis import GameTheory
from theory.precondition_hypothesis import PreconditionHypothesis
from theory.real_env_option_adapter import TheoryOptionRunner, snapshot_frame
from theory.theory_option import build_options_from_theory


@dataclass
class FakeFrame:
    frame: Any
    available_actions: List[Any]
    levels_completed: int = 0
    state: str = "NOT_FINISHED"


class FakeEnv:
    def __init__(self, next_frame: FakeFrame | None) -> None:
        self.next_frame = next_frame
        self.actions: List[str] = []

    def step(self, action: Any) -> FakeFrame | None:
        self.actions.append(_action_name(action))
        return self.next_frame


def test_runner_executes_policy_action_and_builds_live_transition():
    theory = _confirmed_theory()
    option = build_options_from_theory(theory)[0]
    loop = LiveTransitionBeliefLoop(
        "ar25-e3c63847",
        theory=theory,
        available_actions=["ACTION2"],
        background_value=0,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    before_grid = _ready_grid()
    before = FakeFrame(
        frame=[before_grid.tolist()],
        available_actions=[2],
        levels_completed=0,
    )
    after = FakeFrame(
        frame=[before_grid.tolist()],
        available_actions=[2],
        levels_completed=1,
    )
    env = FakeEnv(after)

    result = TheoryOptionRunner(loop, option).run_once(
        env,
        before,
        previous_action="ACTION2",
        recent_actions=["ACTION5", "ACTION2"],
        step=7,
    )

    assert result.initiated
    assert result.action_executed
    assert result.transition_produced
    assert env.actions == ["ACTION2"]
    assert loop.transition_count == 1
    assert result.invocation is not None
    assert result.invocation.success
    assert result.termination == "success"
    assert result.transition_update is not None
    assert result.transition_update.record.obs_before.levels_completed == 0
    assert result.transition_update.record.obs_after.levels_completed == 1

    score = loop.score(build_ar25_oracle(), experiment_actions=loop.transition_count)
    assert score.confirmation_precision >= 0.95
    assert score.wrong_confirmations == 0


def test_runner_does_not_step_when_initiation_is_false():
    theory = _confirmed_theory()
    option = build_options_from_theory(theory)[0]
    loop = LiveTransitionBeliefLoop(
        "ar25-e3c63847",
        theory=theory,
        available_actions=["ACTION2"],
        background_value=0,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    env = FakeEnv(FakeFrame(frame=[_ready_grid().tolist()], available_actions=[2]))

    result = TheoryOptionRunner(loop, option).run_once(
        env,
        FakeFrame(frame=[_ready_grid().tolist()], available_actions=[2]),
    )

    assert not result.initiated
    assert not result.action_executed
    assert not result.transition_produced
    assert result.reason == "initiation_false"
    assert env.actions == []
    assert loop.transition_count == 0


def test_snapshot_accepts_frame_data_grid_shapes_and_action_enums():
    grid = _ready_grid()

    one_plane = snapshot_frame(
        FakeFrame(frame=[grid.tolist()], available_actions=[0, 2], levels_completed=3)
    )
    direct_grid = snapshot_frame(
        FakeFrame(frame=grid.tolist(), available_actions=["ACTION2"])
    )

    assert one_plane.grid.shape == grid.shape
    assert direct_grid.grid.shape == grid.shape
    assert one_plane.available_actions == ["RESET", "ACTION2"]
    assert one_plane.levels_completed == 3


def _confirmed_theory() -> GameTheory:
    target_rule = correspondence_key("ACTION2", "validates", (10, 11))
    theory = GameTheory("ar25-e3c63847")
    theory.seed_actions(["ACTION2"])
    theory.add_semantic_hypotheses(
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION2",
                relation="validates",
                pair_colors=(10, 11),
            )
        ],
        preconditions=[
            PreconditionHypothesis(
                target_rule=target_rule,
                predicate="ready_to_validate_correspondence",
                evidence_for=["pre_a", "pre_b", "pre_c"],
            )
        ],
    )
    for _ in range(2):
        theory.observe_correspondence(
            CorrespondenceObservation(
                action="ACTION2",
                pair_colors=(10, 11),
                level_complete=True,
            ),
            was_experiment=True,
        )
    return theory


def _ready_grid() -> np.ndarray:
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[1:3, 1:3] = 10
    grid[1:3, 5:7] = 11
    grid[4, 4] = 4
    return grid


def _action_name(action: Any) -> str:
    name = getattr(action, "name", None)
    if name:
        return str(name).upper()
    raw = str(action).upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw
