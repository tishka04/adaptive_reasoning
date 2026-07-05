import json

import numpy as np

from theory.sage.live_mini_frontier_m3_executor import (
    SAGE5D_TRUTH_STATUS,
    run_sage5d_live_mini_frontier_m3_execution,
    select_sage5c_mini_frontier_requests,
)
from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame


class FakeAction:
    def __init__(self, name, data=None):
        self.id = name
        self.data = dict(data or {})


class FakeGame:
    def __init__(self, env):
        self.env = env

    def _get_valid_actions(self):
        return [
            FakeAction("ACTION5"),
            FakeAction("ACTION6", {"x": 1, "y": 1}),
            FakeAction("ACTION6", {"x": 2, "y": 1}),
        ]


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = np.asarray(grid, dtype=np.int32)
        self.available_actions = ["ACTION5", "ACTION6"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame(self)
        self.grid = np.zeros((4, 4), dtype=np.int32)
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = np.zeros((4, 4), dtype=np.int32)
            self.step_count = 0
            return FakeFrame(self.grid.copy(), step=0)
        self.step_count += 1
        if name == "ACTION5":
            self.grid[0, :] = 5
        elif name == "ACTION6":
            x = int((data or {}).get("x", 1))
            y = int((data or {}).get("y", 1))
            self.grid[y, x] = 6
        return FakeFrame(self.grid.copy(), step=self.step_count)


def test_select_sage5c_requests_stratifies_family_and_action():
    source = _source_payload()

    selected = select_sage5c_mini_frontier_requests(
        source,
        min_requests=4,
        max_requests=4,
    )

    families = {row["hypothesis_family"] for row in selected}
    actions = {row["target_action"] for row in selected}
    assert len(selected) == 4
    assert "object_delta_candidate" in families
    assert "local_patch_change_candidate" in families
    assert "ACTION5" in actions
    assert "ACTION6" in actions


def test_sage5d_executes_live_prefix_requests_candidate_only(tmp_path):
    source_path = tmp_path / "sage5c.json"
    out = tmp_path / "sage5d.json"
    source_path.write_text(json.dumps(_source_payload()), encoding="utf-8")

    payload = run_sage5d_live_mini_frontier_m3_execution(
        source_sage5c_path=source_path,
        output_path=out,
        min_requests=4,
        max_requests=4,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["truth_status"] == SAGE5D_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["support_events_counted_as_support"] is False
    assert payload["mini_frontier_execution_counted_as_evidence"] is False
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    summary = payload["summary"]
    assert summary["gate_passed"] is True
    assert summary["requests_executed"] == 4
    assert summary["requests_blocked"] == 0
    assert summary["live_prefix_replay_exact_events"] == 4
    assert summary["context_snapshot_hash_verified_events"] == 4
    assert summary["object_delta_candidate_executed"] is True
    assert summary["local_patch_change_candidate_executed"] is True
    assert summary["action5_executed"] is True
    assert summary["action6_executed"] is True
    assert summary["outcome_status"].endswith("CANDIDATE_ONLY")

    experiment = payload["controlled_experiments"][0]
    assert experiment["execution_mode"] == "live_prefix_replay_context"
    assert experiment["live_prefix_replay_exact"] is True
    assert experiment["target_context_signature_verified"] is True
    assert experiment["control_context_signature_verified"] is True
    assert experiment["support"] == 0


def test_sage5d_blocks_when_context_hash_does_not_replay(tmp_path):
    source = _source_payload()
    source["mini_frontier_m3_requests"][0]["context_snapshot_hash"] = "bad-hash"
    source_path = tmp_path / "sage5c_bad.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    payload = run_sage5d_live_mini_frontier_m3_execution(
        source_sage5c_path=source_path,
        min_requests=1,
        max_requests=1,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert payload["summary"]["requests_executed"] == 0
    assert payload["summary"]["requests_blocked"] == 1
    assert payload["summary"]["blocked_replay_events"] == 1
    assert payload["summary"]["gate_passed"] is False
    assert payload["blocked_replay_events"][0]["blocked_reason"] == (
        "context_snapshot_hash_mismatch"
    )
    assert payload["support"] == 0


def _source_payload():
    return {
        "outcome_status": "SAGE_LIVE_MINI_FRONTIER_GENERATED_CANDIDATE_ONLY",
        "summary": {"effective_requests_generated": 4, "support": 0},
        "mini_frontier_m3_requests": [
            _request(1, ["ACTION6"], [{"x": 1, "y": 1}], "ACTION5", None, "object_delta_candidate"),
            _request(
                2,
                ["ACTION6", "ACTION5"],
                [{"x": 1, "y": 1}, {}],
                "ACTION6",
                {"x": 2, "y": 1},
                "local_patch_change_candidate",
            ),
            _request(
                3,
                ["ACTION6", "ACTION5", "ACTION6"],
                [{"x": 1, "y": 1}, {}, {"x": 2, "y": 1}],
                "ACTION5",
                None,
                "local_patch_change_candidate",
            ),
            _request(
                4,
                ["ACTION6", "ACTION5", "ACTION6", "ACTION5"],
                [{"x": 1, "y": 1}, {}, {"x": 2, "y": 1}, {}],
                "ACTION6",
                {"x": 1, "y": 1},
                "local_patch_change_candidate",
            ),
        ],
    }


def _request(step, prefix, prefix_args, target_action, target_args, family):
    context_hash = _signature_after_prefix(prefix, prefix_args)
    control = "ACTION6" if target_action == "ACTION5" else "ACTION5"
    return {
        "request_id": f"m2m3::sage5c::live_mini_frontier::050::{step:04d}",
        "source_hypothesis_id": f"sage5c::live_mini_frontier::050::{step:04d}",
        "source_transition_id": f"sage5c::fake::budget_050::step_{step:04d}",
        "source_step": step,
        "game_id": "fake-unknown",
        "status": "READY_FOR_M3",
        "replayability": "LIVE_PREFIX_REPLAY_CONTEXT",
        "context_state_origin": "sage5_live_prefix_frame_before",
        "context_replay": list(prefix),
        "context_replay_args": [dict(item) for item in prefix_args],
        "context_snapshot_hash": context_hash,
        "target_action": target_action,
        "target_action_args": dict(target_args) if target_args else None,
        "suggested_control_actions": [control],
        "metric": "local_patch_before_after",
        "hypothesis_family": family,
        "truth_status": "NOT_EVALUATED_BY_M2",
        "support": 0,
    }


def _signature_after_prefix(prefix, prefix_args):
    env = FakeEnv()
    frame = env.step("RESET")
    for name, args in zip(prefix, prefix_args):
        frame = env.step(name, data=dict(args))
    return state_signature_from_frame(frame)
