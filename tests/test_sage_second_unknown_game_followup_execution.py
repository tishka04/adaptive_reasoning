import copy
import json

import numpy as np
import pytest

import theory.sage as sage
from theory.sage import second_unknown_game_followup_execution as execution
from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        execution.DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH.read_text(encoding="utf-8")
    )


def test_sage6e_real_artifact_executes_all_pre_registered_protocols(real_payload):
    summary = real_payload["summary"]

    assert summary["game_id"] == "wa30-ee6fef47"
    assert summary["budgets"] == [50, 150, 300]
    assert summary["protocols_available"] == 4
    assert summary["protocols_selected"] == 4
    assert summary["protocols_executed"] == 4
    assert summary["protocols_blocked"] == 0
    assert summary["protocols_executed_by_budget"] == {
        "50": 2,
        "150": 1,
        "300": 1,
    }
    assert summary["control_diversity_protocols_executed"] == 3
    assert summary["neutral_replication_protocols_executed"] == 1
    assert summary["outcome_status"] == execution.SAGE6E_EXECUTION_COMPLETED


def test_sage6e_real_results_are_exact_and_unsubstituted(real_payload):
    results = real_payload["controlled_followup_results"]

    assert len(results) == 4
    assert [row["budget"] for row in results] == [50, 150, 300, 50]
    assert [row["source_step"] for row in results] == [48, 132, 24, 12]
    assert [row["target_action"] for row in results] == ["ACTION2"] * 4
    assert [row["control_action"] for row in results] == [
        "ACTION3",
        "ACTION3",
        "ACTION3",
        "ACTION1",
    ]
    assert [row["target_signal"] for row in results] == [33.0, 33.0, 32.0, 32.0]
    assert [row["control_signal"] for row in results] == [33.0, 33.0, 32.0, 32.0]
    assert [row["effect_size"] for row in results] == [0.0, 0.0, 0.0, 0.0]
    assert all(row["protocol_execution_exact"] for row in results)
    assert all(row["live_prefix_replay_exact"] for row in results)
    assert all(row["target_context_signature_verified"] for row in results)
    assert all(row["control_context_signature_verified"] for row in results)
    assert not any(row["protocol_substitution_performed"] for row in results)
    assert not any(row["context_substitution_performed"] for row in results)
    assert not any(row["budget_substitution_performed"] for row in results)
    assert not any(row["target_action_substitution_performed"] for row in results)
    assert not any(row["control_action_substitution_performed"] for row in results)


def test_sage6e_applies_pre_registered_interpretations_without_verdict(real_payload):
    results = real_payload["controlled_followup_results"]

    assert [row["pre_registered_result_status"] for row in results] == [
        execution.CONTROL_EFFECT_NONPOSITIVE,
        execution.CONTROL_EFFECT_NONPOSITIVE,
        execution.CONTROL_EFFECT_NONPOSITIVE,
        execution.NEUTRAL_REPLICATION_MATCHED,
    ]
    assert [row["pre_registered_condition_met"] for row in results] == [
        False,
        False,
        False,
        True,
    ]
    assert [row["pre_registered_deviation_detected"] for row in results] == [
        True,
        True,
        True,
        False,
    ]
    assert not any(row["result_counted_as_confirmation"] for row in results)
    assert not any(row["result_counted_as_scientific_verdict"] for row in results)


def test_sage6e_identifies_control_sensitive_pattern_candidate_only(real_payload):
    assessment = real_payload["pre_registered_outcome_assessment"]

    assert assessment["prior_control_action"] == "ACTION1"
    assert assessment["distinct_control_action"] == "ACTION3"
    assert assessment["prior_control_effect_sizes"] == [32.0, 32.0, 32.0]
    assert assessment["distinct_control_effect_sizes"] == [0.0, 0.0, 0.0]
    assert assessment["distinct_control_matches_target_across_all_budgets"] is True
    assert (
        assessment["prior_positive_effect_was_control_dependent_candidate_only"] is True
    )
    assert assessment["neutral_replication_effect_size"] == 0.0
    assert assessment["neutral_context_replication_matched"] is True
    assert assessment["candidate_status"] == execution.CONTROL_SENSITIVE_PATTERN
    assert assessment["ready_for_post_execution_consolidation"] is True
    assert assessment["ready_for_A32_review"] is False
    assert assessment["pre_registered_conditions_met"] == 1
    assert assessment["pre_registered_deviations"] == 3
    assert assessment["support"] == 0


def test_sage6e_summary_records_four_neutral_raw_events(real_payload):
    summary = real_payload["summary"]

    assert summary["target_actions_executed"] == {"ACTION2": 4}
    assert summary["control_actions_executed"] == {"ACTION1": 1, "ACTION3": 3}
    assert summary["target_signal_total"] == 130.0
    assert summary["control_signal_total"] == 130.0
    assert summary["controlled_effect_sizes"] == [0.0, 0.0, 0.0, 0.0]
    assert summary["positive_effect_events"] == 0
    assert summary["negative_effect_events"] == 0
    assert summary["zero_effect_events"] == 4
    assert summary["raw_support_events"] == 0
    assert summary["raw_contradiction_events"] == 0
    assert summary["raw_neutral_events"] == 4


def test_sage6e_gate_and_top_level_quarantine_are_complete(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == execution.SAGE6E_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["pre_registered_consistency_counted_as_confirmation"] is False
    assert real_payload["pre_registered_deviation_counted_as_refutation"] is False
    assert (
        real_payload["control_sensitive_pattern_counted_as_scientific_verdict"] is False
    )
    assert real_payload["source_scoped_mechanics_reused"] == 0
    assert real_payload["cross_game_mechanics_imported"] == 0
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage6e_execution_audit_never_reads_outcomes(real_payload):
    audit = real_payload["execution_audit"]

    assert [row["execution_rank"] for row in audit] == [1, 2, 3, 4]
    assert all(row["selected"] for row in audit)
    assert all(
        row["selection_reason"] == "PRE_REGISTERED_SOURCE_ORDER" for row in audit
    )
    assert all(row["outcome_metrics_read_for_selection"] is False for row in audit)
    assert all(row["substitution_allowed"] is False for row in audit)


def test_sage6e_runner_supports_exact_fake_execution(tmp_path):
    source = _fake_source()
    source_path = tmp_path / "sage6d_fake.json"
    output_path = tmp_path / "sage6e_fake.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    payload = execution.run_sage6e_second_unknown_game_followup_execution(
        source_sage6d_path=source_path,
        output_path=output_path,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert output_path.exists()
    assert payload["summary"]["protocols_executed"] == 4
    assert payload["summary"]["protocols_blocked"] == 0
    assert payload["summary"]["controlled_effect_sizes"] == [2.0, 2.0, 2.0, 0.0]
    assert payload["summary"]["pre_registered_conditions_met"] == 4
    assert payload["summary"]["pre_registered_deviations"] == 0
    assert payload["summary"]["gate_passed"] is True
    assert payload["outcome_status"] == execution.SAGE6E_EXECUTION_COMPLETED


def test_sage6e_blocks_mismatched_context_without_substitution(tmp_path):
    source = _fake_source()
    protocol = source["pre_registered_followup_protocols"][0]
    protocol["context_snapshot_hash"] = "bad-context-hash"
    manifest = source["handoff_items"][0]["context_cluster_manifest"]
    cluster = next(
        row
        for row in manifest
        if row["context_cluster_id"] == protocol["source_context_cluster_id"]
    )
    cluster["context_snapshot_hash"] = "bad-context-hash"
    source_path = tmp_path / "sage6d_bad_context.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    payload = execution.run_sage6e_second_unknown_game_followup_execution(
        source_sage6d_path=source_path,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert payload["summary"]["protocols_executed"] == 3
    assert payload["summary"]["protocols_blocked"] == 1
    assert payload["blocked_followup_results"][0]["blocked_reason"] == (
        "context_snapshot_hash_mismatch"
    )
    assert payload["summary"]["gate_passed"] is False
    assert payload["outcome_status"] == execution.SAGE6E_EXECUTION_INCOMPLETE
    assert payload["support"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source.__setitem__("support", 1), "support 0"),
        (
            lambda source: source.__setitem__("execution_performed", True),
            "execution, verdict, or registry writes",
        ),
        (
            lambda source: source.__setitem__("a32_write_performed", True),
            "execution, verdict, or registry writes",
        ),
        (
            lambda source: source["handoff_items"][0].__setitem__(
                "ready_for_A32_review", True
            ),
            "ready only for followup execution",
        ),
        (
            lambda source: source["pre_registered_followup_protocols"][0].__setitem__(
                "cross_context_substitution_allowed", True
            ),
            "replay and candidate state",
        ),
    ],
)
def test_sage6e_rejects_invalid_source_state(tmp_path, mutation, message):
    source = _source_payload()
    mutation(source)
    path = tmp_path / "invalid_source.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        execution.run_sage6e_second_unknown_game_followup_execution(
            source_sage6d_path=path,
            env_factory=lambda game_id: FakeEnv(),
        )


def test_sage6e_rejects_protocol_identity_or_control_drift(tmp_path):
    source = _source_payload()
    source["pre_registered_followup_protocols"][0]["protocol_id"] = "drifted"
    path = tmp_path / "identity_drift.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="identities and order"):
        execution.validate_sage6e_source(source)

    source = _source_payload()
    source["pre_registered_followup_protocols"][0]["control_action"] = "ACTION4"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="control-diversity protocols"):
        execution.validate_sage6e_source(source)


def test_sage6e_condition_classification_is_explicit():
    diversity = {"request_type": execution.CONTROL_DIVERSITY_REQUEST}
    neutral = {"request_type": execution.NEUTRAL_REPLICATION_REQUEST}

    assert execution.pre_registered_condition_met(diversity, 1.0) is True
    assert execution.pre_registered_condition_met(diversity, 0.0) is False
    assert execution.classify_pre_registered_result(diversity, 0.0) == (
        execution.CONTROL_EFFECT_NONPOSITIVE
    )
    assert execution.pre_registered_condition_met(neutral, 0.0) is True
    assert execution.pre_registered_condition_met(neutral, 1.0) is False
    assert execution.classify_pre_registered_result(neutral, 0.0) == (
        execution.NEUTRAL_REPLICATION_MATCHED
    )


def test_sage6e_runner_does_not_mutate_source_or_write_without_output(tmp_path):
    source = _fake_source()
    before = copy.deepcopy(source)
    source_path = tmp_path / "sage6d.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    execution.run_sage6e_second_unknown_game_followup_execution(
        source_sage6d_path=source_path,
        env_factory=lambda game_id: FakeEnv(),
    )

    assert json.loads(source_path.read_text(encoding="utf-8")) == before
    assert set(tmp_path.iterdir()) == {source_path}


def test_sage6e_writer_and_package_exports(real_payload, tmp_path):
    output_path = tmp_path / "nested" / "sage6e.json"

    execution.write_sage6e_second_unknown_game_followup_execution(
        real_payload, output_path
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload
    assert sage.DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH == (
        execution.DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH
    )
    assert (
        sage.run_sage6e_second_unknown_game_followup_execution
        is execution.run_sage6e_second_unknown_game_followup_execution
    )
    assert (
        sage.write_sage6e_second_unknown_game_followup_execution
        is execution.write_sage6e_second_unknown_game_followup_execution
    )


class FakeAction:
    def __init__(self, name):
        self.id = name
        self.data = {}


class FakeGame:
    def __init__(self, env):
        self.env = env

    def _get_valid_actions(self):
        return [FakeAction(f"ACTION{index}") for index in range(1, 5)]


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = np.asarray(grid, dtype=np.int32)
        self.available_actions = [f"ACTION{index}" for index in range(1, 5)]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame(self)
        self.grid = np.zeros((64, 64), dtype=np.int32)
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = np.zeros((64, 64), dtype=np.int32)
            self.step_count = 0
            return FakeFrame(self.grid.copy(), step=0)
        before_step = self.step_count
        self.step_count += 1
        row = 50 + before_step // 64
        column = before_step % 61
        if before_step == 12 and name in {"ACTION1", "ACTION2"}:
            self.grid[row, column] = 7
        elif name == "ACTION2":
            self.grid[row, column : column + 3] = 2
        elif name == "ACTION3":
            self.grid[row, column] = 3
        elif name == "ACTION4":
            self.grid[row, column] = 4
        else:
            self.grid[row, column] = 1
        return FakeFrame(self.grid.copy(), step=self.step_count)


def _fake_source():
    source = _source_payload()
    fake_game_id = "fake-wa30"
    source["summary"]["game_id"] = fake_game_id
    source["source_sage6c_context"]["game_id"] = fake_game_id
    source["source_sage6b_context"]["game_id"] = fake_game_id
    source["handoff_items"][0]["game_id"] = fake_game_id
    manifest = {
        row["context_cluster_id"]: row
        for row in source["handoff_items"][0]["context_cluster_manifest"]
    }
    for protocol in source["pre_registered_followup_protocols"]:
        protocol["game_id"] = fake_game_id
        signature = _signature_after_prefix(
            protocol["context_replay"], protocol["context_replay_args"]
        )
        protocol["context_snapshot_hash"] = signature
        manifest[protocol["source_context_cluster_id"]]["context_snapshot_hash"] = (
            signature
        )
    return source


def _signature_after_prefix(actions, action_args):
    env = FakeEnv()
    frame = env.step("RESET")
    for name, args in zip(actions, action_args):
        frame = env.step(name, data=dict(args))
    return state_signature_from_frame(frame)


def _source_payload():
    return json.loads(execution.DEFAULT_SAGE6D_HANDOFF_PATH.read_text(encoding="utf-8"))
