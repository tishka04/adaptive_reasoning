import json

from theory.sage.known_game_policy_probe import (
    POLICY_RANDOM_LEGAL,
    POLICY_REPEAT_FALLBACK,
    POLICY_SAGE1_NO_GUARD,
    POLICY_SAGE1B_GUARD,
    SAGE2_TRUTH_STATUS,
    run_sage2_known_game_policy_probe,
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
        self.grid = [[0, 0], [0, 0]]
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = [[0, 0], [0, 0]]
            self.step_count = 0
        elif name == "ACTION4":
            self.grid[0][0] = 4
            self.step_count += 1
        elif name == "ACTION3":
            self.grid[0][1] = 3
            self.step_count += 1
        elif name == "ACTION6":
            self.grid[1][1] = int((data or {}).get("x", 6)) % 10
            self.step_count += 1
        return FakeFrame([list(row) for row in self.grid], step=self.step_count)


def _m2_fused_requests():
    return {
        "summary": {
            "experiment_requests": 4,
            "ready_for_m3": 3,
            "blocked_not_testable": 1,
            "llm_output_counted_as_evidence": False,
            "world_model_counted_as_evidence": False,
            "support": 0,
        }
    }


def _m3_fused_results():
    return {
        "summary": {
            "fused_requests_executed": 3,
            "fused_requests_skipped_blocked": 1,
            "fusion_hypothesis_routing_validated": True,
            "new_independent_terminal_risk_evidence": False,
            "support": 0,
        }
    }


def _m3_counterfactual_feasibility():
    return {
        "counterfactual_probe_results": [
            {
                "recommended_frontier": {
                    "target_action": "ACTION3",
                    "target_action_args": None,
                    "frontier_type": (
                        "NEED_ACTIVE_COUNTERFACTUAL_COLLECTION_FROM_TRACE_CONTEXT"
                    ),
                }
            }
        ],
        "summary": {
            "counterfactual_requests_seen": 1,
            "feasible_counterfactual_requests": 0,
            "frontier_recommendations": 1,
            "support": 0,
        },
    }


def _p1_policy_probe():
    return {
        "candidate_policy_counted_as_confirmation": False,
        "candidate_policy_memory": {
            "enabled": True,
            "ready_for_agent_policy_probe": True,
            "a33_ready": False,
            "target_action": "ACTION6",
            "repositioning_action": "ACTION4",
            "known_success_args": [{"x": 12, "y": 0}],
            "support": 0,
        },
        "summary": {
            "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
            "support": 0,
        },
    }


def _p1_utility_handoff():
    return {
        "summary": {
            "agentic_utility_status": "SUPPORTED_CANDIDATE_ONLY",
            "support": 0,
        },
        "a33_ready": False,
        "policy_result_counted_as_mechanistic_confirmation": False,
    }


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path):
    paths = {
        "m2": tmp_path / "m2_requests.json",
        "m3": tmp_path / "m3_fused.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1_probe.json",
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
    return run_sage2_known_game_policy_probe(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        env_factory=lambda game_id: FakeEnv(),
        **kwargs,
    )


def _summaries(payload):
    return {run["policy"]: run["summary"] for run in payload["policy_runs"]}


def test_sage2_probe_runs_all_four_policies_and_writes_output(tmp_path):
    out = tmp_path / "sage2.json"
    payload = _run(tmp_path, output_path=out, budget=8)

    assert out.exists()
    summaries = _summaries(payload)
    assert set(summaries) == {
        POLICY_RANDOM_LEGAL,
        POLICY_REPEAT_FALLBACK,
        POLICY_SAGE1_NO_GUARD,
        POLICY_SAGE1B_GUARD,
    }
    for summary in summaries.values():
        assert summary["support"] == 0
        assert summary["truth_status"] == SAGE2_TRUTH_STATUS
        assert summary["policy_result_counted_as_confirmation"] is False
        assert summary["a32_write_performed"] is False
        assert summary["a33_write_performed"] is False
        assert summary["selected_action_always_legal"] is True


def test_sage2_probe_metrics_present_for_each_policy(tmp_path):
    payload = _run(tmp_path, budget=8)
    summaries = _summaries(payload)
    required = {
        "levels_completed",
        "terminal_rate",
        "state_changed_rate",
        "unique_state_signatures",
        "repeated_action_arg_rate",
        "loop_guard_switches",
        "active_counterfactual_collections",
    }
    for summary in summaries.values():
        assert required.issubset(summary)


def test_sage2_loop_guard_reduces_repeated_action_arg_rate(tmp_path):
    payload = _run(tmp_path, budget=8, loop_guard_max_repeats=2)
    summaries = _summaries(payload)
    sage1 = summaries[POLICY_SAGE1_NO_GUARD]
    sage1b = summaries[POLICY_SAGE1B_GUARD]

    assert sage1b["loop_guard_switches"] > 0
    assert sage1["loop_guard_switches"] == 0
    assert sage1b["repeated_action_arg_rate"] < sage1["repeated_action_arg_rate"]
    assert sage1b["max_same_action_arg_repeats"] <= 2


def test_sage2_only_sage_policies_collect_counterfactuals(tmp_path):
    payload = _run(tmp_path, budget=8)
    summaries = _summaries(payload)
    assert summaries[POLICY_SAGE1B_GUARD]["active_counterfactual_collections"] >= 1
    assert summaries[POLICY_SAGE1_NO_GUARD]["active_counterfactual_collections"] >= 1
    assert summaries[POLICY_RANDOM_LEGAL]["active_counterfactual_collections"] == 0
    assert summaries[POLICY_REPEAT_FALLBACK]["active_counterfactual_collections"] == 0


def test_sage2_comparison_is_candidate_only(tmp_path):
    payload = _run(tmp_path, budget=8)
    comparison = payload["comparison"]

    assert comparison["outcome_status_is_candidate_only"] is True
    assert comparison["outcome_status"].endswith("CANDIDATE_ONLY")
    assert "verdict" not in comparison
    assert comparison["policy_result_counted_as_confirmation"] is False
    assert comparison["support"] == 0
    assert comparison["truth_status"] == SAGE2_TRUTH_STATUS
    assert comparison["a32_write_performed"] is False
    assert comparison["a33_write_performed"] is False

    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == SAGE2_TRUTH_STATUS
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
