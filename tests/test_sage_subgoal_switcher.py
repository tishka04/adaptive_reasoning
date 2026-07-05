import json

from theory.sage.live_prefix_counterfactual_collector import LivePrefixAction
from theory.sage.subgoal_switcher import (
    SAGE3_TRUTH_STATUS,
    SUBGOAL_COUNTERFACTUAL,
    SUBGOAL_EXPLORE_TARGET,
    SUBGOAL_REPOSITION,
    SUBGOAL_RERUN,
    SUBGOAL_SAFE_HOLD,
    TRIGGER_REASON,
    decide_subgoal_switch,
    run_sage3_subgoal_switch_probe,
)


class FakeAction:
    def __init__(self, name, data=None):
        self.id = name
        self.data = dict(data or {})


class ViewAction:
    """Mirror of _ActionView (the shape decide_subgoal_switch consumes)."""

    def __init__(self, name, action_args=None):
        self.name = name
        self.action_args = dict(action_args or {})


_ACTION6_TARGETS = [
    {"x": 12, "y": 0},
    {"x": 24, "y": 0},
    {"x": 30, "y": 12},
    {"x": 5, "y": 5},
    {"x": 6, "y": 6},
    {"x": 7, "y": 7},
]


class FakeGame:
    def __init__(self, env):
        self.env = env

    def _get_valid_actions(self):
        actions = [
            FakeAction("ACTION3"),
            FakeAction("ACTION4"),
            FakeAction("ACTION7"),
        ]
        actions += [FakeAction("ACTION6", args) for args in _ACTION6_TARGETS]
        return actions


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = grid
        self.available_actions = ["ACTION3", "ACTION4", "ACTION6", "ACTION7"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame(self)
        self.grid = [[0] * 16 for _ in range(16)]
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = [[0] * 16 for _ in range(16)]
            self.step_count = 0
            return FakeFrame([list(row) for row in self.grid], step=0)
        self.step_count += 1
        if name == "ACTION4":
            self.grid[0][0] = 4
        elif name == "ACTION3":
            self.grid[0][1] = 3
        elif name == "ACTION7":
            self.grid[0][2] = 7
        elif name == "ACTION6":
            x = int((data or {}).get("x", 0)) % 16
            y = int((data or {}).get("y", 0)) % 16
            self.grid[y][x] = 9
        # clock cell guarantees a state change every executed step
        self.grid[15][15] = self.step_count % 16
        return FakeFrame([list(row) for row in self.grid], step=self.step_count)


def _m2_fused_requests():
    return {"summary": {"experiment_requests": 4, "ready_for_m3": 3, "support": 0}}


def _m3_fused_results():
    return {"summary": {"fused_requests_executed": 3, "support": 0}}


def _m3_counterfactual_feasibility():
    return {
        "counterfactual_probe_results": [
            {
                "recommended_frontier": {
                    "target_action": "ACTION3",
                    "target_action_args": None,
                }
            }
        ],
        "summary": {"frontier_recommendations": 1, "support": 0},
    }


def _p1_policy_probe():
    return {
        "candidate_policy_memory": {
            "enabled": True,
            "target_action": "ACTION6",
            "repositioning_action": "ACTION4",
            "known_success_args": [{"x": 12, "y": 0}, {"x": 24, "y": 0}],
            "support": 0,
        },
        "summary": {"candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"},
    }


def _p1_utility_handoff():
    return {"summary": {"support": 0}, "a33_ready": False}


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path):
    paths = {
        "m2": tmp_path / "m2.json",
        "m3": tmp_path / "m3.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1.json",
        "p1_handoff": tmp_path / "p1_handoff.json",
    }
    _write(paths["m2"], _m2_fused_requests())
    _write(paths["m3"], _m3_fused_results())
    _write(paths["m3_7f"], _m3_counterfactual_feasibility())
    _write(paths["p1"], _p1_policy_probe())
    _write(paths["p1_handoff"], _p1_utility_handoff())
    return paths


def _run(tmp_path, **kwargs):
    paths = _write_inputs(tmp_path)
    return run_sage3_subgoal_switch_probe(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        env_factory=lambda game_id: FakeEnv(),
        **kwargs,
    )


def test_decide_subgoal_switch_rotates_modes_legally():
    valid = [
        ViewAction("ACTION3"),
        ViewAction("ACTION4"),
        ViewAction("ACTION7"),
        ViewAction("ACTION6", {"x": 12, "y": 0}),
        ViewAction("ACTION6", {"x": 30, "y": 12}),
    ]
    memory = {"target_action": "ACTION6", "repositioning_action": "ACTION4"}
    frontier = {"target_action": "ACTION3", "target_action_args": None}
    prefix = [LivePrefixAction("ACTION6", {"x": 12, "y": 0})]

    modes = []
    for i in range(4):
        decision = decide_subgoal_switch(
            valid_actions=valid,
            prefix=prefix,
            policy_memory=memory,
            frontier=frontier,
            blocked_action="ACTION6",
            blocked_action_args={"x": 12, "y": 0},
            last_game_state="NOT_FINISHED",
            switch_index=i,
        )
        assert decision.subgoal_switch_triggered is True
        assert decision.trigger_reason == TRIGGER_REASON
        assert decision.selected_action_legal is True
        modes.append(decision.selected_subgoal)

    assert modes[0] == SUBGOAL_REPOSITION
    assert modes[1] == SUBGOAL_EXPLORE_TARGET
    assert modes[2] == SUBGOAL_COUNTERFACTUAL
    assert modes[3] == SUBGOAL_RERUN


def test_decide_subgoal_switch_safe_holds_on_terminal_risk():
    valid = [ViewAction("ACTION3"), ViewAction("ACTION4")]
    decision = decide_subgoal_switch(
        valid_actions=valid,
        prefix=[],
        policy_memory={"target_action": "ACTION6", "repositioning_action": "ACTION4"},
        frontier=None,
        last_game_state="GAME_OVER",
    )
    assert decision.selected_subgoal == SUBGOAL_SAFE_HOLD
    assert decision.safe_hold is True
    assert decision.selected_action_raw is None


def test_sage3_probe_switches_subgoal_after_exhaustion(tmp_path):
    out = tmp_path / "sage3.json"
    payload = _run(tmp_path, output_path=out, budget=12)

    assert out.exists()
    summary = payload["summary"]
    assert summary["subgoal_switches"] > 0
    assert summary["selected_action_always_legal"] is True
    assert summary["subgoal_switch_success_rate"] > 0.0
    assert len(summary["subgoals_used"]) >= 2
    assert summary["new_candidate_targets_discovered"] >= 1
    assert summary["active_counterfactuals_after_exhaustion"] >= 1
    assert summary["post_switch_repeat_rate"] == 0.0
    assert payload["subgoal_switch_triggered"] is True
    assert payload["trigger_reason"] == TRIGGER_REASON


def test_sage3_probe_keeps_candidate_only_discipline(tmp_path):
    payload = _run(tmp_path, budget=12)
    summary = payload["summary"]

    assert summary["support"] == 0
    assert summary["truth_status"] == SAGE3_TRUTH_STATUS
    assert summary["policy_result_counted_as_confirmation"] is False
    assert summary["outcome_status"].endswith("CANDIDATE_ONLY")
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False

    assert payload["support"] == 0
    assert payload["truth_status"] == SAGE3_TRUTH_STATUS
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload


def test_sage3_rerun_is_placeholder_not_effective(tmp_path):
    payload = _run(tmp_path, budget=12)
    summary = payload["summary"]
    # rerun_m2_m3 selects a legal placeholder action but never actually
    # re-derives M2/M3, so requested may be > 0 while effective stays 0.
    assert summary["rerun_m2_m3_effective_requests_generated"] == 0
    assert summary["placeholder_action_counted_as_subgoal_success"] is False

    rerun_steps = [
        s
        for s in payload["steps"]
        if s.get("selected_subgoal") == "rerun_m2_m3"
    ]
    for step in rerun_steps:
        assert step["rerun_m2_m3_requested"] is True
        assert step["rerun_m2_m3_executed"] is False
        assert step["placeholder_action_used"] is True
        assert step["placeholder_action_counted_as_subgoal_success"] is False


def test_sage3_switch_events_are_candidate_only(tmp_path):
    payload = _run(tmp_path, budget=12)
    events = payload["subgoal_switch_events"]
    assert events
    for event in events:
        assert event["trigger_reason"] == TRIGGER_REASON
        assert event["support"] == 0
        assert event["truth_status"] == SAGE3_TRUTH_STATUS
        assert event["policy_result_counted_as_confirmation"] is False
        assert event["a32_write_performed"] is False
        assert event["a33_write_performed"] is False
