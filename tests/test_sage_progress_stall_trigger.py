import json

from theory.sage.progress_stall_trigger import (
    PROGRESS_STALL_TRIGGER_REASON,
    SAGE4B_TRUTH_STATUS,
    ProgressStallTriggerConfig,
    build_sage4b_summary,
    evaluate_progress_stall_trigger,
    run_sage4b_progress_stall_trigger_probe,
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
        self.frame = grid
        self.available_actions = ["ACTION3", "ACTION4", "ACTION6"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame(self)
        self.grid = [[0] * 8 for _ in range(8)]
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = [[0] * 8 for _ in range(8)]
            self.step_count = 0
            return FakeFrame([list(row) for row in self.grid], step=0)
        self.step_count += 1
        if name == "ACTION3":
            self.grid[0][1] = self.step_count % 8
        elif name == "ACTION4":
            self.grid[0][0] = self.step_count % 8
        elif name == "ACTION6":
            x = int((data or {}).get("x", 0)) % 8
            y = int((data or {}).get("y", 0)) % 8
            self.grid[y][x] = self.step_count % 8
        return FakeFrame([list(row) for row in self.grid], step=self.step_count)


def _step(action, args=None, *, repeated=True, signature="sig", level=0, step=0):
    return {
        "step": step,
        "selected_action": action,
        "selected_action_args": dict(args or {}),
        "state_signature_after": signature,
        "levels_after": level,
        "repeated_previous_action": repeated,
        "env_actions": 1,
        "invalid_action_selected": False,
    }


def test_progress_stall_trigger_detects_repeated_same_action_args():
    steps = [
        _step("ACTION6", {"x": 1, "y": 1}, signature="sig-a", step=0),
        _step("ACTION6", {"x": 1, "y": 1}, signature="sig-b", step=1),
        _step("ACTION6", {"x": 1, "y": 1}, signature="sig-b", step=2),
    ]

    result = evaluate_progress_stall_trigger(
        steps=steps,
        proposed_action="ACTION6",
        proposed_action_args={"x": 1, "y": 1},
        config=ProgressStallTriggerConfig(
            window_size=4,
            same_action_arg_repeats=4,
            low_state_novelty_threshold=3,
            repeated_action_arg_rate_threshold=0.75,
        ),
    )

    assert result.switch_required is True
    assert result.trigger_reason == PROGRESS_STALL_TRIGGER_REASON
    assert result.same_action_args_repeated is True
    assert result.same_action_args_repeat_count == 4
    assert result.no_level_progress is True
    assert result.support == 0
    assert result.truth_status == SAGE4B_TRUTH_STATUS


def test_progress_stall_trigger_does_not_fire_when_level_progresses():
    steps = [
        _step("ACTION6", {"x": 1, "y": 1}, level=0, step=0),
        _step("ACTION6", {"x": 1, "y": 1}, level=1, step=1),
        _step("ACTION6", {"x": 1, "y": 1}, level=1, step=2),
    ]

    result = evaluate_progress_stall_trigger(
        steps=steps,
        proposed_action="ACTION6",
        proposed_action_args={"x": 1, "y": 1},
        config=ProgressStallTriggerConfig(window_size=4, same_action_arg_repeats=4),
    )

    assert result.no_level_progress is False
    assert result.switch_required is False


def test_sage4b_summary_fails_when_post_trigger_repetition_stays_high():
    run = {
        "summary": {
            "selected_action_always_legal": True,
            "subgoal_switches": 1,
            "active_counterfactuals_after_exhaustion": 1,
            "rerun_m2_m3_requested": 0,
            "levels_completed": 0,
            "unique_state_signatures": 2,
        },
        "steps": [
            _step("ACTION6", {"x": 1, "y": 1}, step=0),
            _step("ACTION6", {"x": 1, "y": 1}, step=1),
            {
                **_step("ACTION3", {}, repeated=False, step=2),
                "trigger_reason": PROGRESS_STALL_TRIGGER_REASON,
                "is_subgoal_switch": True,
            },
            *[
                _step("ACTION6", {"x": 1, "y": 1}, step=step)
                for step in range(3, 30)
            ],
        ],
    }

    summary = build_sage4b_summary(run)

    assert summary["progress_stall_detected"] is True
    assert summary["repeated_action_arg_rate_after_trigger"] > 0.9
    assert summary["repeat_collapse_interrupted"] is False
    assert summary["gate_passed"] is False
    assert summary["support"] == 0


def test_sage4b_runner_writes_candidate_only_artifact(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage4b.json"

    payload = run_sage4b_progress_stall_trigger_probe(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        output_path=out,
        game_id="fake-progress-stall",
        budget=12,
        same_action_arg_repeats=3,
        progress_stall_window=5,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE4B_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload
    assert payload["summary"]["support"] == 0


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
    _write(paths["m2"], {"summary": {"experiment_requests": 4, "ready_for_m3": 3, "support": 0}})
    _write(paths["m3"], {"summary": {"fused_requests_executed": 3, "support": 0}})
    _write(
        paths["m3_7f"],
        {
            "counterfactual_probe_results": [
                {"recommended_frontier": {"target_action": "ACTION3", "target_action_args": None}}
            ],
            "summary": {"frontier_recommendations": 1, "support": 0},
        },
    )
    _write(
        paths["p1"],
        {
            "candidate_policy_memory": {
                "enabled": True,
                "target_action": "ACTION6",
                "repositioning_action": "ACTION4",
                "known_success_args": [{"x": 12, "y": 0}],
                "support": 0,
            },
            "summary": {"candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"},
        },
    )
    _write(paths["p1_handoff"], {"summary": {"support": 0}, "a33_ready": False})
    return paths
