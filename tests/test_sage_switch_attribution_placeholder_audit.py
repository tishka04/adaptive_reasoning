import json

from theory.sage.switch_attribution_placeholder_audit import (
    SAGE5B_TRUTH_STATUS,
    run_sage5b_switch_attribution_placeholder_audit,
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


def test_sage5b_audits_switch_causes_and_placeholder_dependency(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage5b.json"

    payload = run_sage5b_switch_attribution_placeholder_audit(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        source_sage5_path=paths["sage5"],
        output_path=out,
        max_counterfactual_collections=4,
        placeholder_dependency_threshold=0.0,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE5B_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    comparison = payload["comparison"]
    assert comparison["switches_total"] > 0
    assert comparison["switches_due_to_success_like_targets_exhausted_or_loop_guard"] > 0
    assert comparison["switches_due_to_progress_stall_or_repeat_collapse"] == 0
    assert comparison["placeholder_rerun_m2_m3_switches"] > 0
    assert comparison["rerun_m2_m3_requested"] > 0
    assert comparison["rerun_m2_m3_effective_requests_generated"] == 0
    assert comparison["effective_request_ratio"] == 0.0
    assert comparison["placeholder_dependency_under_threshold"] is False
    assert comparison["outcome_status"].endswith("CANDIDATE_ONLY")

    row = payload["per_budget_results"][0]
    assert row["placeholder_audit"]["placeholder_dependency_high"] is True
    assert row["placeholder_audit"]["placeholder_action_counted_as_subgoal_success"] is False


def test_sage5b_uses_source_sage5_game_and_budgets(tmp_path):
    paths = _write_inputs(tmp_path, budgets=[7, 9])

    payload = run_sage5b_switch_attribution_placeholder_audit(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        source_sage5_path=paths["sage5"],
        env_factory=lambda game_id: FakeEnv(),
    )

    assert payload["config"]["game_id"] == "fake-unknown"
    assert payload["config"]["budgets"] == [7, 9]
    assert payload["comparison"]["budgets_evaluated"] == [7, 9]
    assert payload["support"] == 0


def _write_inputs(tmp_path, budgets=None):
    paths = {
        "m2": tmp_path / "m2.json",
        "m3": tmp_path / "m3.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1.json",
        "p1_handoff": tmp_path / "p1_handoff.json",
        "sage5": tmp_path / "sage5.json",
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
            "config": {"game_id": "fake-unknown", "budgets": list(budgets or [10])},
            "outcome_status": "SAGE_UNKNOWN_BOUNDED_PROBE_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY",
            "comparison": {"unknown_game": True, "budgets_gate_passed": 1, "support": 0},
            "support": 0,
        },
    )
    return paths


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
