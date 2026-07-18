import copy

import numpy as np
import pytest

from theory.sage import controlled_followup_acquisition as acquisition
from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame


@pytest.fixture(scope="module")
def real_payload():
    return acquisition.run_sage5h_controlled_followup_acquisition()


def test_sage5h_resolves_real_followups_with_audited_partial_result(real_payload):
    summary = real_payload["summary"]

    assert summary["followup_requests_consumed"] == 4
    assert summary["followup_outcomes"] == 4
    assert summary["followups_completed"] == 2
    assert summary["followups_blocked"] == 2
    assert summary["control_diversity_followups_completed"] == 0
    assert summary["control_diversity_followups_blocked"] == 2
    assert summary["support_followups_completed"] == 1
    assert summary["cross_measurement_followups_completed"] == 1
    assert summary["control_surface_contexts_audited"] == 3
    assert summary["control_surface_exhausted_audits"] == 3
    assert summary["controlled_experiments_executed"] == 4
    assert summary["controlled_experiments_blocked"] == 0
    assert summary["comparable_support_events_acquired"] == 1
    assert summary["cross_measurement_divergences"] == 1
    assert summary["all_followups_resolved"] is True
    assert summary["all_requested_followups_completed"] is False
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == acquisition.SAGE5H_PARTIAL_CONTROL_SURFACE_LIMIT


def test_sage5h_real_cross_measurement_preserves_nonmerged_divergence(real_payload):
    outcome = next(
        row
        for row in real_payload["followup_outcomes"]
        if row["request_type"] == acquisition.CROSS_MEASUREMENT_REQUEST
    )

    assert outcome["completed"] is True
    assert (
        outcome["resolution_status"]
        == acquisition.FOLLOWUP_ACQUIRED_CROSS_MEASUREMENT_DIVERGENCE
    )
    assert outcome["clusters_remained_unmerged"] is True
    alignment = outcome["measurement_alignment"]
    assert alignment["local_patch_aligned"] is True
    assert alignment["object_delta_aligned"] is False
    assert alignment["all_measurements_aligned"] is False
    object_signatures = alignment["object_delta_signatures_by_cluster"]
    assert object_signatures["sage5f::candidate_mechanism_cluster::002"] == {
        "object_count_delta": 0,
        "object_count_delta_by_color": {"0": -1, "3": 1},
    }
    assert object_signatures["sage5f::candidate_mechanism_cluster::003"] == {
        "object_count_delta": -1,
        "object_count_delta_by_color": {"0": -1},
    }


def test_sage5h_real_candidate_updates_gain_support_but_remain_control_blocked(
    real_payload,
):
    action6, action5 = real_payload["updated_candidate_assessments"]

    assert action6["action"] == "ACTION6"
    assert action6["raw_support_events_after"] == 3
    assert action6["distinct_control_actions_after"] == 1
    assert action6["control_surface_exhausted"] is True
    assert action6["missing_revision_requirements"] == [
        "minimum_distinct_control_actions"
    ]

    assert action5["action"] == "ACTION5"
    assert action5["raw_support_events_before"] == 2
    assert action5["new_comparable_support_events"] == 1
    assert action5["raw_support_events_after"] == 3
    assert action5["independent_context_events_after"] == 3
    assert action5["distinct_control_actions_after"] == 1
    assert action5["control_surface_exhausted"] is True
    assert action5["missing_revision_requirements"] == [
        "minimum_distinct_control_actions"
    ]
    assert (
        action5["a32_intake_recommendation"]
        == acquisition.UPDATED_RECOMMENDATION_CONTROL_SURFACE_EXHAUSTED
    )
    assert action5["ready_for_A32_intake"] is False


def test_sage5h_keeps_all_results_candidate_only(real_payload):
    assert real_payload["support"] == 0
    assert real_payload["truth_status"] == acquisition.SAGE5H_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["execution_performed"] is True
    assert real_payload["revision_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert real_payload["followup_events_counted_as_scientific_support"] is False
    assert real_payload["control_surface_block_counted_as_refutation"] is False
    assert real_payload["cross_measurement_alignment_counted_as_confirmation"] is False
    assert real_payload["candidate_assessment_counted_as_revision"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False
    assert all(row["support"] == 0 for row in real_payload["controlled_experiments"])
    assert all(
        row["support_events_counted_as_support"] is False
        for row in real_payload["controlled_experiments"]
    )


def test_sage5h_rejects_source_that_counts_support():
    sage5g, sage5e, sage5f = _valid_sources()
    sage5g["support"] = 1

    with pytest.raises(ValueError, match="SAGE.5g support must remain 0"):
        acquisition.validate_sage5h_sources(sage5g, sage5e, sage5f)


def test_sage5h_rejects_source_that_writes_a32_or_a33():
    sage5g, sage5e, sage5f = _valid_sources()
    sage5f["a33_write_performed"] = True

    with pytest.raises(ValueError, match="SAGE.5f cannot write A32/A33"):
        acquisition.validate_sage5h_sources(sage5g, sage5e, sage5f)


def test_cross_measurement_alignment_distinguishes_alignment_and_divergence():
    aligned = {
        "a": {
            "local_patch": _measurement(changed=2),
            "object_delta": _measurement(changed=2, object_delta={"3": 1}),
        },
        "b": {
            "local_patch": _measurement(changed=2),
            "object_delta": _measurement(changed=2, object_delta={"3": 1}),
        },
    }
    divergent = copy.deepcopy(aligned)
    divergent["b"]["object_delta"] = _measurement(
        changed=2,
        object_delta={"0": -1},
    )

    assert acquisition.cross_measurement_alignment(aligned)[
        "all_measurements_aligned"
    ] is True
    result = acquisition.cross_measurement_alignment(divergent)
    assert result["local_patch_aligned"] is True
    assert result["object_delta_aligned"] is False
    assert result["all_measurements_aligned"] is False


def test_updated_assessment_counts_new_context_without_scientific_support():
    candidate = {
        "candidate_id": "candidate-1",
        "candidate_key": "mechanic_prediction::fake",
        "game_id": "fake",
        "action": "ACTION5",
        "action_args": None,
        "raw_support_events": 2,
        "independent_context_events": 2,
        "contradiction_events": 0,
        "contexts": [
            {"context_snapshot_hash": "c1"},
            {"context_snapshot_hash": "c2"},
        ],
        "control_interventions": [{"action": "ACTION6"}],
    }
    outcomes = [
        {
            "candidate_id": "candidate-1",
            "request_type": acquisition.SUPPORT_REQUEST,
            "completed": True,
            "acquired_raw_support_events": 1,
            "acquired_context_snapshot_hash": "c3",
        },
        {
            "candidate_id": "candidate-1",
            "request_type": acquisition.CONTROL_DIVERSITY_REQUEST,
            "completed": False,
            "resolution_status": acquisition.FOLLOWUP_BLOCKED_CONTROL_SURFACE,
        },
    ]

    updated = acquisition.update_candidate_assessments(
        candidates=[candidate],
        outcomes=outcomes,
    )[0]

    assert updated["raw_support_events_after"] == 3
    assert updated["independent_context_events_after"] == 3
    assert updated["distinct_control_actions_after"] == 1
    assert updated["missing_revision_requirements"] == [
        "minimum_distinct_control_actions"
    ]
    assert updated["support"] == 0
    assert updated["candidate_assessment_counted_as_revision"] is False


def test_control_surface_audit_reports_exhaustion_with_action5_action6_only():
    request = _fake_request()
    followup = {
        "action": "ACTION6",
        "excluded_control_actions": ["ACTION5"],
    }

    audit = acquisition.audit_context_control_surface(
        request=request,
        followup=followup,
        environments_dir=None,
        env_factory=lambda game_id: FakeEnv(include_action3=False),
    )

    assert audit["execution_status"] == "AUDITED"
    assert audit["live_prefix_replay_exact"] is True
    assert audit["available_action_names"] == ["ACTION5", "ACTION6"]
    assert audit["eligible_distinct_control_actions"] == []
    assert audit["control_surface_exhausted"] is True


def test_control_surface_audit_finds_new_distinct_action_when_available():
    request = _fake_request()
    followup = {
        "action": "ACTION6",
        "excluded_control_actions": ["ACTION5"],
    }

    audit = acquisition.audit_context_control_surface(
        request=request,
        followup=followup,
        environments_dir=None,
        env_factory=lambda game_id: FakeEnv(include_action3=True),
    )

    assert audit["available_action_names"] == ["ACTION3", "ACTION5", "ACTION6"]
    assert audit["eligible_distinct_control_actions"] == ["ACTION3"]
    assert audit["control_surface_exhausted"] is False


def test_followup_experiment_cache_reuses_same_replay_measurement():
    request = _fake_request(target_action="ACTION5", target_args=None)
    cache = {}
    experiments = []
    factory = lambda game_id: FakeEnv(include_action3=False)

    first = acquisition.execute_followup_experiment(
        source_request=request,
        metric="local_patch_before_after",
        control_action="ACTION6",
        followup_id="followup-1",
        purpose="support",
        environments_dir=None,
        env_factory=factory,
        execution_cache=cache,
        experiments=experiments,
    )
    second = acquisition.execute_followup_experiment(
        source_request=request,
        metric="local_patch_before_after",
        control_action="ACTION6",
        followup_id="followup-2",
        purpose="cross_measurement:local_patch",
        environments_dir=None,
        env_factory=factory,
        execution_cache=cache,
        experiments=experiments,
    )

    assert first is second
    assert len(experiments) == 1
    assert first["execution_status"] == "EXECUTED"
    assert first["live_prefix_replay_exact"] is True
    assert first["support_events"] == 1
    assert first["source_followup_ids"] == ["followup-1", "followup-2"]
    assert first["support"] == 0


def _valid_sources():
    base = {
        "summary": {"support": 0},
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    sage5g = {
        **copy.deepcopy(base),
        "candidate_review_item_counted_as_revision": False,
        "candidate_review_item_counted_as_scientific_verdict": False,
    }
    return sage5g, copy.deepcopy(base), copy.deepcopy(base)


def _measurement(*, changed, object_delta=None):
    return {
        "target_measurement": {
            "changed_pixels": changed,
            "local_patch_available": False,
            "local_changed_pixels": 0,
            "object_count_delta": sum((object_delta or {}).values()),
            "object_count_delta_by_color": dict(object_delta or {}),
        }
    }


def _fake_request(*, target_action="ACTION6", target_args=None):
    frame = FakeFrame(np.zeros((8, 8), dtype=np.int32), step=0)
    return {
        "request_id": "fake-request",
        "source_hypothesis_id": "fake-hypothesis",
        "source_transition_id": "sage5e::fake::budget_050::step_0000",
        "game_id": "fake",
        "hypothesis_family": "local_patch_change_candidate",
        "target_action": target_action,
        "target_action_args": copy.deepcopy(target_args),
        "metric": "local_patch_before_after",
        "context_replay": [],
        "context_replay_args": [],
        "context_snapshot_hash": state_signature_from_frame(frame),
        "suggested_control_actions": [
            "ACTION6" if target_action == "ACTION5" else "ACTION5"
        ],
        "truth_status": "NOT_EVALUATED_BY_M2",
        "support": 0,
    }


class FakeAction:
    def __init__(self, name, data=None):
        self.id = name
        self.data = dict(data or {})


class FakeGame:
    def __init__(self, env):
        self.env = env

    def _get_valid_actions(self):
        actions = [
            FakeAction("ACTION5"),
            FakeAction("ACTION6", {"x": 1, "y": 1}),
            FakeAction("ACTION6", {"x": 3, "y": 1}),
        ]
        if self.env.include_action3:
            actions.insert(0, FakeAction("ACTION3"))
        return actions


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = np.asarray(grid, dtype=np.int32)
        self.available_actions = ["ACTION3", "ACTION5", "ACTION6"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self, *, include_action3):
        self.include_action3 = include_action3
        self._game = FakeGame(self)
        self.grid = np.zeros((8, 8), dtype=np.int32)
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = np.zeros((8, 8), dtype=np.int32)
            self.step_count = 0
            return FakeFrame(self.grid.copy(), step=0)
        self.step_count += 1
        marker = self.step_count % 7
        if name == "ACTION5":
            self.grid[2, marker] = 5
            self.grid[2, marker + 1] = 5
        elif name == "ACTION6":
            args = data or getattr(action, "data", {}) or {}
            x = int(args.get("x", 1)) % 8
            y = int(args.get("y", 1)) % 8
            self.grid[y, x] = 6
        elif name == "ACTION3":
            pass
        return FakeFrame(self.grid.copy(), step=self.step_count)
