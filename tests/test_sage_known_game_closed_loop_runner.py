import json

from theory.sage.known_game_closed_loop_runner import (
    SAGE1_TRUTH_STATUS,
    run_sage1_known_game_closed_loop,
)
from theory.sage.policy_loop_guard import SAGE1B_TRUTH_STATUS


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


def test_sage1_runner_uses_real_env_available_actions_and_legal_actions(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage1.json"

    payload = run_sage1_known_game_closed_loop(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        output_path=out,
        budget=2,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    summary = payload["summary"]
    assert summary["known_game_live_run_completed"] is True
    assert summary["benchmark_run"] is False
    assert summary["live_steps_executed"] == 2
    assert summary["available_actions_source"] == "real_env_live_api"
    assert summary["real_env_available_actions_used"] is True
    assert summary["synthetic_available_actions_used"] is False
    assert summary["selected_action_always_legal"] is True
    assert summary["invalid_action_selected"] is False
    assert summary["support"] == 0
    assert summary["truth_status"] == SAGE1_TRUTH_STATUS
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False

    assert payload["live_steps"][0]["selected_action"] == "ACTION4"
    assert payload["live_steps"][1]["selected_action"] == "ACTION6"
    assert payload["live_steps"][1]["selected_action_args"] == {"x": 12, "y": 0}


def test_sage1_runner_attempts_active_counterfactual_by_live_prefix(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = run_sage1_known_game_closed_loop(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        budget=2,
        env_factory=lambda game_id: FakeEnv(),
    )

    summary = payload["summary"]
    assert summary["active_counterfactual_collection_attempted"] is True
    assert summary["live_prefix_replay_exact"] is True
    assert summary["offline_counterfactual_allowed"] is False
    assert summary["policy_result_counted_as_confirmation"] is False
    collection = payload["active_counterfactual_collections"][0]
    assert collection["alternative_action"] == "ACTION3"
    assert collection["live_prefix_replay_exact"] is True
    assert collection["selected_action_legal"] is True
    assert collection["invalid_action_selected"] is False
    assert collection["support"] == 0


def test_sage1_runner_preserves_candidate_only_roles(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = run_sage1_known_game_closed_loop(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        budget=1,
        env_factory=lambda game_id: FakeEnv(),
    )

    logger = payload["logger_contract"]
    assert logger["policy_result_counted_as_confirmation"] is False
    assert logger["offline_counterfactual_allowed"] is False
    assert logger["active_counterfactual_collection_attempted"] is True
    assert payload["input_summaries"]["hypothesis_context"]["m2_ready_for_m3"] == 3
    assert payload["input_summaries"]["m3_tests"]["m3_fused_requests_executed"] == 3
    assert payload["input_summaries"]["policy_context"]["a33_ready"] is False
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"


def test_sage1b_loop_guard_interrupts_repeated_target_fallback(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage1b.json"

    payload = run_sage1_known_game_closed_loop(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        output_path=out,
        budget=5,
        env_factory=lambda game_id: FakeEnv(),
        enable_loop_guard=True,
        loop_guard_max_repeats=2,
    )

    assert out.exists()
    summary = payload["summary"]
    assert summary["loop_guard_enabled"] is True
    assert summary["repeated_same_action_args_detected"] is True
    assert summary["fallback_loop_interrupted"] is True
    assert summary["switch_action_selected_after_exhaustion"] is True
    assert summary["max_same_action_arg_repeats"] <= 2
    assert summary["selected_action_always_legal"] is True
    assert summary["invalid_action_selected"] is False
    assert summary["policy_result_counted_as_confirmation"] is False
    assert summary["support"] == 0
    assert summary["truth_status"] == SAGE1B_TRUTH_STATUS
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False

    switched = [
        step for step in payload["live_steps"] if step["fallback_loop_interrupted"]
    ]
    assert switched
    assert switched[0]["selected_action"] == "ACTION3"
    assert switched[0]["decision_reason"] == "policy_loop_guard_switch_after_exhaustion"
    assert switched[0]["blocked_action"] == "ACTION6"
    assert switched[0]["blocked_action_args"] == {"x": 12, "y": 0}
