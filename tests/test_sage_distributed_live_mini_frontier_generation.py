import json

import numpy as np

from theory.sage.distributed_live_mini_frontier_generation import (
    SAGE5E_TRUTH_STATUS,
    diff_signature,
    live_mini_frontier_dedup_key,
    run_sage5e_distributed_live_mini_frontier_generation,
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
            FakeAction("ACTION5"),
            FakeAction("ACTION6", {"x": 12, "y": 0}),
            FakeAction("ACTION6", {"x": 24, "y": 0}),
            FakeAction("ACTION6", {"x": 30, "y": 12}),
        ]


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = np.asarray(grid, dtype=np.int32)
        self.available_actions = ["ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame(self)
        self.grid = np.zeros((16, 16), dtype=np.int32)
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = np.zeros((16, 16), dtype=np.int32)
            self.step_count = 0
            return FakeFrame(self.grid.copy(), step=0)
        self.step_count += 1
        marker = self.step_count % 16
        if name == "ACTION4":
            self.grid[0, marker] = 4
        elif name == "ACTION3":
            self.grid[1, marker] = 3
        elif name == "ACTION5":
            self.grid[2, marker] = 5
            self.grid[3, marker] = 5
        elif name == "ACTION6":
            x = int((data or {}).get("x", 0)) % 16
            y = int((data or {}).get("y", 0)) % 16
            self.grid[y, x] = 6
        self.grid[15, 15] = marker
        return FakeFrame(self.grid.copy(), step=self.step_count)


def test_diff_signature_and_dedup_key_are_stable():
    diff = {
        "changed_cells": 2,
        "changed_bbox": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
        "color_transitions": {"0->5": 2},
        "component_delta_by_color": {"5": 1},
        "terminal_after": False,
        "levels_delta": 0,
        "ignored": "not part of signature",
    }
    hypothesis = {"diff_summary": diff}
    request = {
        "context_snapshot_hash": "abc",
        "target_action": "ACTION5",
        "target_action_args": None,
    }

    assert diff_signature(diff) == {
        "changed_cells": 2,
        "changed_bbox": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
        "color_transitions": {"0->5": 2},
        "component_delta_by_color": {"5": 1},
        "terminal_after": False,
        "levels_delta": 0,
    }
    assert live_mini_frontier_dedup_key(hypothesis, request) == (
        live_mini_frontier_dedup_key(hypothesis, dict(reversed(list(request.items()))))
    )


def test_sage5e_distributes_generation_and_execution_across_budgets(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage5e.json"

    payload = run_sage5e_distributed_live_mini_frontier_generation(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        source_sage5_path=paths["sage5"],
        source_sage5b_path=paths["sage5b"],
        source_sage5c_path=paths["sage5c"],
        output_path=out,
        requests_per_budget=2,
        execute_requests_per_budget=1,
        max_counterfactual_collections=4,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE5E_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["generated_requests_counted_as_support"] is False
    assert payload["mini_frontier_counted_as_evidence"] is False
    assert payload["mini_frontier_execution_counted_as_evidence"] is False
    assert payload["support_events_counted_as_support"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    summary = payload["summary"]
    assert summary["gate_passed"] is True
    assert summary["budgets_with_effective_generation"] == [12, 18, 24]
    assert summary["budgets_with_execution"] == [12, 18, 24]
    assert summary["effective_requests_generated"] == 6
    assert summary["mini_frontier_m3_requests"] == 6
    assert summary["selected_execution_requests"] == 3
    assert summary["requests_executed"] == 3
    assert summary["requests_blocked"] == 0
    assert summary["dedup_key_count"] == 6
    assert summary["outcome_status"].endswith("CANDIDATE_ONLY")

    budgets = [
        row["budget"]
        for row in payload["per_budget_results"]
        if row["generation_metrics"]["effective_requests_generated"] > 0
    ]
    assert budgets == [12, 18, 24]
    assert all(request["dedup_key"] for request in payload["mini_frontier_m3_requests"])
    assert all(
        request["generated_by"] == "SAGE.5e_distributed_live_mini_frontier"
        for request in payload["mini_frontier_m3_requests"]
    )


def _write_inputs(tmp_path):
    paths = {
        "m2": tmp_path / "m2.json",
        "m3": tmp_path / "m3.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1.json",
        "p1_handoff": tmp_path / "p1_handoff.json",
        "sage5": tmp_path / "sage5.json",
        "sage5b": tmp_path / "sage5b.json",
        "sage5c": tmp_path / "sage5c.json",
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
                "known_success_args": [{"x": 12, "y": 0}, {"x": 24, "y": 0}],
                "support": 0,
            },
            "summary": {"candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"},
        },
    )
    _write(paths["p1_handoff"], {"summary": {"support": 0}, "a33_ready": False})
    _write(
        paths["sage5"],
        {
            "config": {"game_id": "fake-unknown", "budgets": [12, 18, 24]},
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
    _write(
        paths["sage5c"],
        {
            "outcome_status": "SAGE_LIVE_MINI_FRONTIER_GENERATED_CANDIDATE_ONLY",
            "summary": {
                "effective_requests_generated": 2,
                "effective_request_ratio": 0.1,
                "support": 0,
            },
            "support": 0,
        },
    )
    return paths


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
