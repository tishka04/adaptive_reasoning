import json

from theory.sage.long_horizon_progress_stall_transfer import (
    SAGE4C_TRUTH_STATUS,
    run_sage4c_long_horizon_progress_stall_transfer,
)


class FakeAction:
    def __init__(self, name, data=None):
        self.id = name
        self.data = dict(data or {})


_ACTION6_TARGETS = [
    {"x": 12, "y": 0},
    {"x": 24, "y": 0},
    {"x": 30, "y": 12},
    {"x": 5, "y": 5},
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
        self.grid[15][15] = self.step_count % 16
        return FakeFrame([list(row) for row in self.grid], step=self.step_count)


def test_sage4c_runs_budgets_with_progress_stall_gate(tmp_path):
    paths = _write_inputs(tmp_path)
    baseline = tmp_path / "sage4_baseline.json"
    out = tmp_path / "sage4c.json"
    _write(baseline, _baseline_sage4((10, 20, 30)))

    payload = run_sage4c_long_horizon_progress_stall_transfer(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        baseline_sage4_path=baseline,
        output_path=out,
        game_id="fake-longhorizon",
        budgets=(10, 20, 30),
        max_counterfactual_collections=4,
        same_action_arg_repeats=3,
        progress_stall_window=5,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE4C_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    comparison = payload["comparison"]
    assert comparison["any_budget_gate_passed"] is True
    assert comparison["outcome_status"].endswith("CANDIDATE_ONLY")
    assert comparison["support"] == 0
    assert comparison["budgets_with_progress_stall_detected"]
    assert comparison["budgets_with_subgoal_switches"]

    for row in payload["per_budget_results"]:
        metrics = row["metrics"]
        gate = row["gate"]
        assert metrics["progress_stall_detected"] is True
        assert metrics["subgoal_switches"] > 0
        assert metrics["repeated_action_arg_rate_lower_than_sage4"] is True
        assert gate["selected_action_always_legal"] is True
        assert gate["progress_stall_detected"] is True
        assert gate["active_counterfactual_or_rerun_requested"] is True


def test_sage4c_keeps_gate_failed_without_baseline_improvement(tmp_path):
    paths = _write_inputs(tmp_path)
    baseline = tmp_path / "sage4_baseline.json"
    _write(baseline, _baseline_sage4((10,), repeat_rate=0.0))

    payload = run_sage4c_long_horizon_progress_stall_transfer(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        baseline_sage4_path=baseline,
        game_id="fake-longhorizon",
        budgets=(10,),
        max_counterfactual_collections=4,
        same_action_arg_repeats=3,
        progress_stall_window=5,
        env_factory=lambda game_id: FakeEnv(),
    )

    row = payload["per_budget_results"][0]
    assert row["metrics"]["repeated_action_arg_rate_lower_than_sage4"] is False
    assert row["gate"]["gate_passed"] is False
    assert payload["comparison"]["any_budget_gate_passed"] is False
    assert payload["support"] == 0


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


def _baseline_sage4(budgets, repeat_rate=0.95):
    return {
        "outcome_status": "SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY",
        "support": 0,
        "per_budget_results": [
            {
                "budget": int(budget),
                "metrics": {
                    "repeated_action_arg_rate": float(repeat_rate),
                    "support": 0,
                },
            }
            for budget in budgets
        ],
    }
