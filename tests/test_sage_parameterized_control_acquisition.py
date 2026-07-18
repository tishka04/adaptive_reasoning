import copy
import json

import pytest

from theory.sage import parameterized_control_acquisition as acquisition


@pytest.fixture(scope="module")
def real_payload():
    return acquisition.run_sage5j_parameterized_control_acquisition()


@pytest.fixture(scope="module")
def real_sources():
    a32 = json.loads(
        acquisition.DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )
    sage5e = json.loads(
        acquisition.DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH.read_text(
            encoding="utf-8"
        )
    )
    return a32, sage5e


def test_sage5j_executes_the_complete_real_preregistered_protocol(real_payload):
    summary = real_payload["summary"]

    assert summary["source_protocol_decisions"] == 2
    assert summary["pre_registered_experiments_consumed"] == 8
    assert summary["experiments_executed"] == 8
    assert summary["experiments_blocked"] == 0
    assert summary["live_prefix_replay_exact_experiments"] == 8
    assert summary["protocol_exact_match_experiments"] == 8
    assert summary["protocol_substitutions_detected"] == 0
    assert summary["target_control_pairs_executed"] == 8
    assert summary["parameterized_variants_executed"] == 4
    assert summary["variant_replications_completed"] == 4
    assert summary["candidate_protocols_complete"] == 2
    assert summary["candidates_ready_for_A32_protocol_result_review"] == 2
    assert summary["all_pre_registered_experiments_executed_exactly"] is True
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == acquisition.SAGE5J_MIXED_DISCRIMINATION


def test_sage5j_real_action6_controls_are_consistently_non_discriminating(
    real_payload,
):
    rows = [
        row
        for row in real_payload["executed_parameterized_control_experiments"]
        if row["target_action"] == "ACTION6"
    ]

    assert len(rows) == 4
    assert {row["target_signal"] for row in rows} == {5.0}
    assert {row["control_signal"] for row in rows} == {5.0}
    assert all(
        row["discrimination_status"] == acquisition.NON_DISCRIMINATING_EQUAL_EFFECT
        for row in rows
    )
    assert all(row["paired_effect_signatures_equal"] is True for row in rows)
    assert sum(row["neutral_events"] for row in rows) == 4
    assert sum(row["support_events"] for row in rows) == 0
    assert sum(row["contradiction_events"] for row in rows) == 0


def test_sage5j_real_action5_controls_are_consistently_discriminating(real_payload):
    rows = [
        row
        for row in real_payload["executed_parameterized_control_experiments"]
        if row["target_action"] == "ACTION5"
    ]

    assert len(rows) == 4
    assert {row["target_signal"] for row in rows} == {21.0}
    assert {row["control_signal"] for row in rows} == {4.0}
    assert all(
        row["discrimination_status"] == acquisition.DISCRIMINATING_TARGET_EFFECT
        for row in rows
    )
    assert all(row["paired_effect_signatures_equal"] is False for row in rows)
    assert sum(row["support_events"] for row in rows) == 4
    assert sum(row["neutral_events"] for row in rows) == 0
    assert sum(row["contradiction_events"] for row in rows) == 0


def test_sage5j_executes_exact_variants_contexts_and_measurements(real_payload):
    requested = {
        row["experiment_id"]: row for row in real_payload["pre_registered_experiments"]
    }

    for row in real_payload["executed_parameterized_control_experiments"]:
        protocol = requested[row["protocol_experiment_id"]]
        assert row["source_request_id"] == protocol["source_request_id"]
        assert row["context_snapshot_hash"] == protocol["context_snapshot_hash"]
        assert row["target_action"] == protocol["target_action"]
        assert row["target_action_args"] == protocol["target_action_args"]
        assert row["control_action"] == protocol["control_action"]
        assert row["control_action_args"] == protocol["control_action_args"]
        assert row["measurement"] == protocol["measurement"]
        assert row["live_prefix_replay_exact"] is True
        assert row["protocol_exact_match"] is True
        assert row["protocol_substitution_detected"] is False


def test_sage5j_real_candidate_assessments_preserve_mixed_result(real_payload):
    action6, action5 = real_payload["candidate_protocol_assessments"]

    assert action6["action"] == "ACTION6"
    assert action6["raw_support_events"] == 0
    assert action6["raw_neutral_events"] == 4
    assert action6["raw_contradiction_events"] == 0
    assert action6["parameterized_protocol_result"] == (
        acquisition.CANDIDATE_NON_DISCRIMINATING
    )

    assert action5["action"] == "ACTION5"
    assert action5["raw_support_events"] == 4
    assert action5["raw_neutral_events"] == 0
    assert action5["raw_contradiction_events"] == 0
    assert action5["parameterized_protocol_result"] == (
        acquisition.CANDIDATE_DISCRIMINATING
    )

    assert all(row["executed_experiments"] == 4 for row in (action6, action5))
    assert all(
        row["variant_replication_complete"] is True for row in (action6, action5)
    )
    assert all(row["protocol_execution_complete"] is True for row in (action6, action5))
    assert all(
        row["ready_for_A32_protocol_result_review"] is True
        for row in (action6, action5)
    )


def test_sage5j_keeps_execution_candidate_only(real_payload):
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == acquisition.SAGE5J_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["execution_performed"] is True
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert (
        real_payload["parameterized_control_events_counted_as_scientific_support"]
        is False
    )
    assert real_payload["parameterized_controls_counted_as_distinct_actions"] is False
    assert real_payload["neutral_events_counted_as_refutation"] is False
    assert real_payload["protocol_result_counted_as_a32_decision"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage5j_rejects_a32_source_support_confirmation_or_a33(real_sources):
    a32, _ = real_sources
    source = copy.deepcopy(a32)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        acquisition.validate_a32_4_protocol_source(source)

    source = copy.deepcopy(a32)
    source["confirmation_performed"] = True
    with pytest.raises(ValueError, match="cannot revise or confirm"):
        acquisition.validate_a32_4_protocol_source(source)

    source = copy.deepcopy(a32)
    source["a33_ready"] = True
    with pytest.raises(ValueError, match="cannot make the candidates A33-ready"):
        acquisition.validate_a32_4_protocol_source(source)


def test_sage5j_rejects_protocol_mutation_or_control_relabelling(real_sources):
    a32, _ = real_sources
    source = copy.deepcopy(a32)
    source["requested_followup_experiments"][0]["status"] = "EXECUTED"
    with pytest.raises(ValueError, match="flattened experiment plan must be immutable"):
        acquisition.validate_a32_4_protocol_source(source)

    source = copy.deepcopy(a32)
    source["requested_followup_experiments"][0][
        "parameterized_control_counted_as_distinct_action"
    ] = True
    source["protocol_decisions"][0]["preregistered_experiments"][0][
        "parameterized_control_counted_as_distinct_action"
    ] = True
    with pytest.raises(ValueError, match="cannot relabel its control"):
        acquisition.validate_a32_4_protocol_source(source)


def test_sage5j_rejects_cross_source_context_or_target_arg_substitution(real_sources):
    a32, sage5e = real_sources
    source = copy.deepcopy(a32)
    source["requested_followup_experiments"][0]["context_snapshot_hash"] = "wrong"
    source["protocol_decisions"][0]["preregistered_experiments"][0][
        "context_snapshot_hash"
    ] = "wrong"
    with pytest.raises(ValueError, match="context hash does not match"):
        acquisition.validate_sage5j_sources(source, sage5e)

    source = copy.deepcopy(a32)
    source["requested_followup_experiments"][0]["target_action_args"] = {
        "x": 99,
        "y": 57,
    }
    source["protocol_decisions"][0]["preregistered_experiments"][0][
        "target_action_args"
    ] = {"x": 99, "y": 57}
    with pytest.raises(ValueError, match="target args do not match"):
        acquisition.validate_sage5j_sources(source, sage5e)


def test_sage5j_rejects_mutated_sage5e_scientific_state(real_sources):
    _, sage5e = real_sources
    source = copy.deepcopy(sage5e)
    source["support"] = 1

    with pytest.raises(ValueError, match="SAGE.5e support must remain 0"):
        acquisition.validate_sage5e_execution_source(source)


def test_discrimination_and_candidate_result_classification():
    assert acquisition.discrimination_status_from_delta({"effect_size": 1}) == (
        acquisition.DISCRIMINATING_TARGET_EFFECT
    )
    assert acquisition.discrimination_status_from_delta({"effect_size": 0}) == (
        acquisition.NON_DISCRIMINATING_EQUAL_EFFECT
    )
    assert acquisition.discrimination_status_from_delta({"effect_size": -1}) == (
        acquisition.CONTROL_EXCEEDS_TARGET_EFFECT
    )
    assert (
        acquisition.candidate_protocol_result(
            complete=True,
            experiment_count=4,
            support_events=4,
            contradiction_events=0,
            neutral_events=0,
        )
        == acquisition.CANDIDATE_DISCRIMINATING
    )
    assert (
        acquisition.candidate_protocol_result(
            complete=True,
            experiment_count=4,
            support_events=0,
            contradiction_events=0,
            neutral_events=4,
        )
        == acquisition.CANDIDATE_NON_DISCRIMINATING
    )


def test_protocol_substitution_detection_is_argument_exact():
    reasons = acquisition.protocol_substitution_reasons(
        protocol={
            "context_snapshot_hash": "context",
            "measurement": "local_patch_before_after",
            "target_action": "ACTION6",
            "target_action_args": {"x": 26, "y": 57},
            "control_action": "ACTION6",
            "control_action_args": {"x": 18, "y": 57},
        },
        request={
            "context_snapshot_hash": "context",
            "metric": "local_patch_before_after",
        },
        target_arm={"action": "ACTION6", "action_args": {"x": 26, "y": 57}},
        control_arm={"action": "ACTION6", "action_args": {"x": 34, "y": 57}},
    )

    assert reasons == ("control_action_args",)


def test_sage5j_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "sage5j.json"

    acquisition.write_sage5j_parameterized_control_acquisition(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload
