import json

from theory.sage.unknown_game_bounded_probe import (
    SAGE5_TRUTH_STATUS,
    run_sage5_unknown_game_bounded_probe,
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


def test_sage5_unknown_game_probe_passes_bounded_gate(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage5.json"

    payload = run_sage5_unknown_game_bounded_probe(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        source_sage4c_path=paths["sage4c"],
        m2_dataset_manifest_path=paths["manifest"],
        human_traces_dir=paths["human_traces"],
        output_path=out,
        game_id="fake-unknown",
        budgets=(10, 20),
        max_counterfactual_collections=4,
        same_action_arg_repeats=3,
        progress_stall_window=5,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE5_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    identity = payload["unknown_game_identity"]
    assert identity["unknown_game"] is True
    assert identity["no_human_trace_for_game"] is True
    assert identity["no_game_specific_prior"] is True

    comparison = payload["comparison"]
    assert comparison["any_budget_gate_passed"] is True
    assert comparison["source_sage4c_all_budgets_gate_passed"] is True
    assert comparison["offline_counterfactual_allowed"] is False
    assert comparison["terminal_rate_guarded"] is True
    assert comparison["outcome_status"].endswith("CANDIDATE_ONLY")

    for row in payload["per_budget_results"]:
        assert set(row["baselines"]) == {"random_legal", "neutral_legal_fallback"}
        gate = row["gate"]
        assert gate["unknown_game"] is True
        assert gate["selected_action_always_legal"] is True
        assert gate["progress_stall_detector_runs"] is True
        assert gate["subgoal_switches_happened"] is True
        assert gate["no_catastrophic_repeat_collapse"] is True
        assert gate["terminal_rate_under_threshold"] is True


def test_sage5_blocks_known_trace_or_m2_game_identity(tmp_path):
    paths = _write_inputs(tmp_path, manifest_games={"fake-known": 3})
    (paths["human_traces"] / "fake-20260705.steps.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )

    payload = run_sage5_unknown_game_bounded_probe(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        source_sage4c_path=paths["sage4c"],
        m2_dataset_manifest_path=paths["manifest"],
        human_traces_dir=paths["human_traces"],
        game_id="fake-known",
        budgets=(10,),
        env_factory=lambda game_id: FakeEnv(),
    )

    assert payload["unknown_game_identity"]["unknown_game"] is False
    assert payload["unknown_game_identity"]["no_human_trace_for_game"] is False
    assert payload["comparison"]["any_budget_gate_passed"] is False
    assert payload["comparison"]["outcome_status"].startswith(
        "SAGE_UNKNOWN_BOUNDED_PROBE_BLOCKED_KNOWN_GAME"
    )
    assert payload["support"] == 0


def _write_inputs(tmp_path, manifest_games=None):
    paths = {
        "m2": tmp_path / "m2.json",
        "m3": tmp_path / "m3.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1.json",
        "p1_handoff": tmp_path / "p1_handoff.json",
        "sage4c": tmp_path / "sage4c.json",
        "manifest": tmp_path / "manifest.json",
        "human_traces": tmp_path / "human_traces",
    }
    paths["human_traces"].mkdir()
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
    _write(
        paths["sage4c"],
        {
            "outcome_status": "SAGE_PROGRESS_STALL_LONG_HORIZON_ALL_BUDGETS_TRANSFER_CANDIDATE_ONLY",
            "comparison": {
                "all_budgets_gate_passed": True,
                "budgets_gate_passed": 3,
                "budgets_total": 3,
                "support": 0,
            },
            "support": 0,
        },
    )
    _write(paths["manifest"], {"per_game_counts": dict(manifest_games or {})})
    return paths


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
