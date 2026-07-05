import json

from theory.sage.long_horizon_transfer import (
    SAGE4_TRUTH_STATUS,
    run_sage4_long_horizon_transfer,
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
        self.grid[15][15] = self.step_count % 16
        return FakeFrame([list(row) for row in self.grid], step=self.step_count)


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
                "known_success_args": [{"x": 12, "y": 0}, {"x": 24, "y": 0}],
                "support": 0,
            },
            "summary": {"candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"},
        },
    )
    _write(paths["p1_handoff"], {"summary": {"support": 0}, "a33_ready": False})
    return paths


def _run(tmp_path, **kwargs):
    paths = _write_inputs(tmp_path)
    return run_sage4_long_horizon_transfer(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        game_id="fake-longhorizon",
        env_factory=lambda game_id: FakeEnv(),
        **kwargs,
    )


def test_sage4_runs_all_budgets_and_writes_output(tmp_path):
    out = tmp_path / "sage4.json"
    payload = _run(tmp_path, output_path=out, budgets=(10, 20, 30))

    assert out.exists()
    budgets = [row["budget"] for row in payload["per_budget_results"]]
    assert budgets == [10, 20, 30]
    for row in payload["per_budget_results"]:
        metrics = row["metrics"]
        for key in (
            "levels_completed",
            "terminal_rate",
            "subgoal_switches",
            "subgoal_switch_success_rate",
            "new_candidate_targets_discovered",
            "active_counterfactuals_after_exhaustion",
            "rerun_m2_m3_requested",
            "rerun_m2_m3_effective_requests_generated",
            "post_switch_repeat_rate",
            "unique_state_signatures",
        ):
            assert key in metrics
        assert metrics["support"] == 0
        assert metrics["policy_result_counted_as_confirmation"] is False


def test_sage4_gate_passes_on_disciplined_long_horizon(tmp_path):
    payload = _run(tmp_path, budgets=(10, 20, 30))
    comparison = payload["comparison"]

    assert comparison["all_budgets_gate_passed"] is True
    assert comparison["outcome_status"].endswith("CANDIDATE_ONLY")
    assert comparison["discovered_new_candidate_targets_beyond_bp35"] is True
    for row in payload["per_budget_results"]:
        gate = row["gate"]
        assert gate["selected_action_always_legal"] is True
        assert gate["no_repeat_collapse"] is True
        assert gate["subgoal_switches_happened"] is True
        assert gate["useful_subgoal_capability"] is True
        assert gate["gate_passed"] is True


def test_sage4_rerun_requests_not_counted_as_effective(tmp_path):
    payload = _run(tmp_path, budgets=(30,))
    metrics = payload["per_budget_results"][0]["metrics"]
    # rerun is a legal placeholder transition; it must never be reported as an
    # actually-generated M2/M3 re-derivation request.
    assert metrics["rerun_m2_m3_effective_requests_generated"] == 0


def test_sage4_keeps_candidate_only_discipline(tmp_path):
    payload = _run(tmp_path, budgets=(10, 20))
    assert payload["support"] == 0
    assert payload["truth_status"] == SAGE4_TRUTH_STATUS
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload
    assert payload["comparison"]["support"] == 0
    assert payload["comparison"]["a32_write_performed"] is False
