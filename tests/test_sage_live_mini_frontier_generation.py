import json

from theory.sage.live_mini_frontier_generation import (
    SAGE5C_TRUTH_STATUS,
    live_transition_diff,
    run_sage5c_live_mini_frontier_generation,
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
            FakeAction("ACTION6", {"x": 24, "y": 0}),
            FakeAction("ACTION6", {"x": 30, "y": 12}),
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
        elif name == "ACTION6":
            x = int((data or {}).get("x", 0)) % 16
            y = int((data or {}).get("y", 0)) % 16
            self.grid[y][x] = 9
        self.grid[15][15] = self.step_count % 16
        return FakeFrame([list(row) for row in self.grid], step=self.step_count)


def test_live_transition_diff_summarizes_patch_and_components():
    before = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    after = [[0, 2, 0], [0, 0, 0], [0, 0, 3]]

    diff = live_transition_diff(before, after, terminal_after=False, levels_delta=0)

    assert diff["changed_cells"] == 3
    assert diff["changed_bbox"] == {"x_min": 1, "y_min": 0, "x_max": 2, "y_max": 2}
    assert diff["color_transitions"] == {"0->2": 1, "0->3": 1, "1->0": 1}
    assert diff["colors_added"] == [2, 3]
    assert diff["colors_removed"] == [1]
    assert diff["components_created_total"] >= 1


def test_sage5c_generates_effective_requests_from_placeholders(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage5c.json"

    payload = run_sage5c_live_mini_frontier_generation(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        source_sage5_path=paths["sage5"],
        source_sage5b_path=paths["sage5b"],
        output_path=out,
        max_generated_requests=3,
        max_counterfactual_collections=4,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE5C_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["generated_requests_counted_as_support"] is False
    assert payload["mini_frontier_counted_as_evidence"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    comparison = payload["comparison"]
    assert comparison["rerun_m2_m3_requested"] > 0
    assert comparison["effective_requests_generated"] == 3
    assert comparison["effective_request_ratio"] > 0.0
    assert comparison["residual_placeholder_switch_ratio"] < comparison["source_placeholder_switch_ratio"]
    assert comparison["outcome_status"].endswith("CANDIDATE_ONLY")

    assert len(payload["mini_frontier_hypotheses"]) == 3
    assert len(payload["mini_frontier_m3_requests"]) == 3
    request = payload["mini_frontier_m3_requests"][0]
    assert request["status"] == "READY_FOR_M3"
    assert request["replayability"] == "LIVE_PREFIX_REPLAY_CONTEXT"
    assert request["context_state_origin"] == "sage5_live_prefix_frame_before"
    assert request["support"] == 0


def _write_inputs(tmp_path):
    paths = {
        "m2": tmp_path / "m2.json",
        "m3": tmp_path / "m3.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1.json",
        "p1_handoff": tmp_path / "p1_handoff.json",
        "sage5": tmp_path / "sage5.json",
        "sage5b": tmp_path / "sage5b.json",
    }
    _write(paths["m2"], {"summary": {"support": 0}})
    _write(paths["m3"], {"summary": {"support": 0}})
    _write(
        paths["m3_7f"],
        {
            "counterfactual_probe_results": [
                {"recommended_frontier": {"target_action": "ACTION3", "target_action_args": None}}
            ],
            "summary": {"support": 0},
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
    _write(
        paths["sage5"],
        {
            "config": {"game_id": "fake-unknown", "budgets": [12]},
            "outcome_status": "SAGE_UNKNOWN_BOUNDED_PROBE_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY",
            "comparison": {"support": 0},
            "support": 0,
        },
    )
    _write(
        paths["sage5b"],
        {
            "outcome_status": "SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_HIGH_CANDIDATE_ONLY",
            "comparison": {
                "rerun_m2_m3_requested": 9,
                "placeholder_switch_ratio": 0.75,
                "effective_request_ratio": 0.0,
                "support": 0,
            },
            "support": 0,
        },
    )
    return paths


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
