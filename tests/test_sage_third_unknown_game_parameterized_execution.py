import copy
import json

import pytest

import theory.sage as sage
import theory.sage.third_unknown_game_parameterized_execution as execution


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        execution.DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        execution.DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_sage7b_executes_all_pre_registered_arms(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["game_id"] == "tn36-ab4f63cc"
    assert summary["budgets_evaluated"] == [50, 150, 300]
    assert summary["requests_available"] == 18
    assert summary["requests_executed"] == 18
    assert summary["requests_blocked"] == 0
    assert summary["target_arm_executions"] == 18
    assert summary["control_arm_executions"] == 36
    assert summary["total_arm_executions"] == 54
    assert summary["comparison_events"] == 36
    assert summary["requests_executed_by_budget"] == {
        "50": 6,
        "150": 6,
        "300": 6,
    }
    assert [row["target_arm_executions"] for row in rows] == [6, 6, 6]
    assert [row["control_arm_executions"] for row in rows] == [12, 12, 12]
    assert [row["comparison_events"] for row in rows] == [12, 12, 12]


def test_sage7b_preserves_exact_live_prefix_protocol(real_payload):
    experiments = real_payload["executed_parameterized_experiments"]

    assert len(experiments) == 18
    assert len({row["request_id"] for row in experiments}) == 18
    assert all(row["live_prefix_replay_exact"] is True for row in experiments)
    assert all(row["context_snapshot_hash_verified"] is True for row in experiments)
    assert all(row["protocol_exact_match"] is True for row in experiments)
    assert all(row["protocol_substitution_detected"] is False for row in experiments)
    assert all(
        row["target_arm"]["action_args"] == row["target_action_args"]
        for row in experiments
    )
    for experiment in experiments:
        expected = experiment["pre_registered_parameterized_control_variants"]
        observed = experiment["control_arms"]
        assert len(expected) == len(observed) == 2
        assert [row["action"] for row in observed] == [
            row["action"] for row in expected
        ]
        assert [row["action_args"] for row in observed] == [
            row["action_args"] for row in expected
        ]


def test_sage7b_keeps_one_action_family_for_targets_and_controls(real_payload):
    experiments = real_payload["executed_parameterized_experiments"]
    summary = real_payload["summary"]

    assert summary["action_families"] == ["ACTION6"]
    assert summary["distinct_action_families"] == 1
    assert summary["parameterized_variants_counted_as_distinct_actions"] is False
    for experiment in experiments:
        assert experiment["target_action"] == "ACTION6"
        assert experiment["parameterized_action_family"] == "ACTION6"
        assert all(row["action"] == "ACTION6" for row in experiment["control_arms"])
        assert experiment["parameterized_controls_counted_as_distinct_actions"] is False


def test_sage7b_observes_mixed_parameterized_deltas(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["metrics_executed"] == {
        "local_patch_before_after": 32,
        "terminal_state_after_rollout": 4,
    }
    assert summary["discrimination_statuses"] == {
        "DISCRIMINATING_TARGET_EFFECT_CANDIDATE_ONLY": 13,
        "NON_DISCRIMINATING_EQUAL_EFFECT_CANDIDATE_ONLY": 23,
    }
    assert summary["positive_delta_events"] == 13
    assert summary["negative_delta_events"] == 0
    assert summary["zero_delta_events"] == 23
    assert summary["distinct_effect_sizes"] == [0.0, 2.0]
    assert summary["min_effect_size"] == 0.0
    assert summary["max_effect_size"] == 2.0
    assert [row["positive_delta_events"] for row in rows] == [5, 4, 4]
    assert [row["negative_delta_events"] for row in rows] == [0, 0, 0]
    assert [row["zero_delta_events"] for row in rows] == [7, 8, 8]


def test_sage7b_effect_depends_on_parameterized_control(real_payload):
    comparisons = [
        comparison
        for experiment in real_payload["executed_parameterized_experiments"]
        for comparison in experiment["parameterized_comparisons"]
    ]
    local_x25 = [
        row
        for row in comparisons
        if row["metric"] == "local_patch_before_after"
        and row["target_action_args"] == {"x": 25, "y": 42}
    ]

    assert len(local_x25) == 26
    against_34 = [
        row for row in local_x25 if row["control_action_args"] == {"x": 34, "y": 51}
    ]
    against_41 = [
        row for row in local_x25 if row["control_action_args"] == {"x": 41, "y": 44}
    ]
    assert len(against_34) == len(against_41) == 13
    assert {row["target_signal"] for row in against_34 + against_41} == {2.0}
    assert {row["control_signal"] for row in against_34} == {0.0}
    assert {row["control_signal"] for row in against_41} == {2.0}
    assert {row["controlled_delta"]["effect_size"] for row in against_34} == {2.0}
    assert {row["controlled_delta"]["effect_size"] for row in against_41} == {0.0}


def test_sage7b_passes_gates_but_requires_consolidation(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["summary"]["ready_for_event_consolidation"] is True
    assert real_payload["summary"]["required_next_step"] == (
        execution.SAGE7B_CONSOLIDATION_REQUIRED
    )
    assert real_payload["outcome_status"] == execution.SAGE7B_EXECUTION_COMPLETED
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == execution.SAGE7B_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["raw_events_counted_as_scientific_support"] is False
    assert real_payload["positive_deltas_counted_as_support"] is False
    assert real_payload["negative_deltas_counted_as_refutation"] is False
    assert real_payload["zero_deltas_counted_as_non_identifiability"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage7b_runner_reproduces_the_complete_execution(tmp_path):
    output = tmp_path / "sage7b.json"

    payload = execution.run_sage7b_parameterized_execution(output_path=output)

    assert output.exists()
    assert payload["summary"]["requests_executed"] == 18
    assert payload["summary"]["total_arm_executions"] == 54
    assert payload["summary"]["positive_delta_events"] == 13
    assert payload["summary"]["zero_delta_events"] == 23
    assert all(payload["gate"].values())
    assert payload["outcome_status"] == execution.SAGE7B_EXECUTION_COMPLETED


def test_sage7b_rejects_mutated_source_contract(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        execution.validate_sage7b_source(source)

    source = copy.deepcopy(real_source)
    source["a32_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        execution.validate_sage7b_source(source)

    source = copy.deepcopy(real_source)
    source["cross_game_mechanics_imported"] = 1
    with pytest.raises(ValueError, match="cannot import or generalize"):
        execution.validate_sage7b_source(source)

    source = copy.deepcopy(real_source)
    source["mini_frontier_m3_requests"][0][
        "pre_registered_parameterized_control_variants"
    ][0]["action_args"] = source["mini_frontier_m3_requests"][0]["target_action_args"]
    with pytest.raises(ValueError, match="controls must remain exact"):
        execution.validate_sage7b_source(source)

    source = copy.deepcopy(real_source)
    controls = source["mini_frontier_m3_requests"][0][
        "pre_registered_parameterized_control_variants"
    ]
    controls[1]["action_args"] = controls[0]["action_args"]
    with pytest.raises(ValueError, match="controls must be unique"):
        execution.validate_sage7b_source(source)


def test_protocol_substitution_detection_is_argument_exact(real_source):
    request = real_source["mini_frontier_m3_requests"][0]
    target_arm = {
        "action": request["target_action"],
        "action_args": request["target_action_args"],
    }
    control_arms = [
        {"action": row["action"], "action_args": row["action_args"]}
        for row in request["pre_registered_parameterized_control_variants"]
    ]

    assert (
        execution.exact_protocol_substitution_reasons(
            request=request,
            target_arm=target_arm,
            control_arms=control_arms,
        )
        == []
    )
    mutated = copy.deepcopy(control_arms)
    mutated[1]["action_args"]["x"] += 1
    assert execution.exact_protocol_substitution_reasons(
        request=request,
        target_arm=target_arm,
        control_arms=mutated,
    ) == ["control_2_action_args"]


def test_sage7b_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    execution.write_sage7b_parameterized_execution(real_payload, output)

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage7b_api():
    assert sage.DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH == (
        execution.DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH
    )
    assert (
        sage.run_sage7b_parameterized_execution
        is execution.run_sage7b_parameterized_execution
    )
    assert (
        sage.write_sage7b_parameterized_execution
        is execution.write_sage7b_parameterized_execution
    )
