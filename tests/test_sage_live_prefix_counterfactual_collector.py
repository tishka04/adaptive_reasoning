import numpy as np

from theory.sage.live_prefix_counterfactual_collector import (
    ACTION_UNAVAILABLE,
    REPLAY_DIVERGED,
    REPLAY_EXACT,
    LivePrefixAction,
    collect_live_prefix_counterfactual,
    state_signature_from_frame,
)


class FakeAction:
    def __init__(self, name, data=None):
        self.id = name
        self.data = dict(data or {})


class FakeGame:
    def __init__(self, env):
        self.env = env

    def _get_valid_actions(self):
        return [
            FakeAction("ACTION3"),
            FakeAction("ACTION4"),
            FakeAction("ACTION6", {"x": 12, "y": 0}),
        ]


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = np.asarray(grid, dtype=np.int32)
        self.available_actions = ["ACTION3", "ACTION4", "ACTION6"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame(self)
        self.grid = np.array([[0, 0], [0, 0]], dtype=np.int32)
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = np.array([[0, 0], [0, 0]], dtype=np.int32)
            self.step_count = 0
        elif name == "ACTION4":
            self.grid[0, 0] = 4
            self.step_count += 1
        elif name == "ACTION3":
            self.grid[0, 1] = 3
            self.step_count += 1
        elif name == "ACTION6":
            self.grid[1, 1] = int((data or {}).get("x", 6)) % 10
            self.step_count += 1
        return FakeFrame(self.grid.copy(), step=self.step_count)


def test_live_prefix_counterfactual_replays_prefix_then_collects_alternative():
    target_env = FakeEnv()
    target_env.step("RESET")
    target_frame = target_env.step("ACTION4")
    target_signature = state_signature_from_frame(target_frame)

    result = collect_live_prefix_counterfactual(
        game_id="bp35-0a0ad940",
        prefix_actions=[LivePrefixAction("ACTION4")],
        target_state_signature=target_signature,
        alternative_action="ACTION3",
        env_factory=lambda game_id: FakeEnv(),
    )

    assert result.status == REPLAY_EXACT
    assert result.live_prefix_replay_exact is True
    assert result.active_counterfactual_collection_attempted is True
    assert result.selected_action_legal is True
    assert result.invalid_action_selected is False
    assert result.state_changed is True
    assert result.env_actions == 2
    assert result.support == 0


def test_live_prefix_counterfactual_refuses_when_prefix_replay_diverges():
    result = collect_live_prefix_counterfactual(
        game_id="bp35-0a0ad940",
        prefix_actions=[LivePrefixAction("ACTION4")],
        target_state_signature="not-the-live-signature",
        alternative_action="ACTION3",
        env_factory=lambda game_id: FakeEnv(),
    )

    assert result.status == REPLAY_DIVERGED
    assert result.live_prefix_replay_exact is False
    assert result.active_counterfactual_collection_attempted is False
    assert result.selected_action_legal is False
    assert result.invalid_action_selected is False
    assert result.env_actions == 1


def test_live_prefix_counterfactual_marks_unavailable_alternative_illegal():
    target_env = FakeEnv()
    target_env.step("RESET")
    target_frame = target_env.step("ACTION4")
    target_signature = state_signature_from_frame(target_frame)

    result = collect_live_prefix_counterfactual(
        game_id="bp35-0a0ad940",
        prefix_actions=[LivePrefixAction("ACTION4")],
        target_state_signature=target_signature,
        alternative_action="ACTION7",
        env_factory=lambda game_id: FakeEnv(),
    )

    assert result.status == ACTION_UNAVAILABLE
    assert result.live_prefix_replay_exact is True
    assert result.active_counterfactual_collection_attempted is True
    assert result.selected_action_legal is False
    assert result.invalid_action_selected is True
