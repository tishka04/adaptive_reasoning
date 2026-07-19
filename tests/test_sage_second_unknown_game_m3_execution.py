import copy
import json

import pytest

import theory.sage as sage
import theory.sage.second_unknown_game_m3_execution as execution


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        execution.DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        execution.DEFAULT_SAGE6B_M3_EXECUTION_PATH.read_text(encoding="utf-8")
    )


def test_sage6b_selects_two_hash_diverse_requests_per_budget(real_source):
    selected = execution.select_sage6b_execution_requests(real_source)

    assert len(selected) == 6
    assert [execution.budget_from_sage6a_request(row) for row in selected] == [
        50,
        50,
        150,
        150,
        300,
        300,
    ]
    assert [row["source_step"] for row in selected] == [12, 48, 132, 144, 24, 36]
    assert len({row["context_snapshot_hash"] for row in selected}) == 6
    assert all(row["status"] == "READY_FOR_M3" for row in selected)
    assert all(
        row["context_state_origin"] == execution.EXPECTED_CONTEXT_STATE_ORIGIN
        for row in selected
    )


def test_sage6b_step_spread_is_deterministic():
    rows = [
        {"source_step": step, "request_id": f"request-{step}"}
        for step in [5, 1, 4, 2, 3]
    ]

    spread = execution.spread_requests_by_step(rows)

    assert [row["source_step"] for row in spread] == [1, 5, 2, 4, 3]


def test_sage6b_real_execution_covers_all_budgets(real_payload):
    summary = real_payload["summary"]

    assert summary["game_id"] == "wa30-ee6fef47"
    assert summary["budgets_available"] == [50, 150, 300]
    assert summary["requests_available"] == 20
    assert summary["requests_selected"] == 6
    assert summary["requests_selected_by_budget"] == {"50": 2, "150": 2, "300": 2}
    assert summary["unique_selected_context_snapshot_hashes"] == 6
    assert summary["selected_source_steps_by_budget"] == {
        "50": [12, 48],
        "150": [132, 144],
        "300": [24, 36],
    }
    assert summary["requests_executed"] == 6
    assert summary["requests_executed_by_budget"] == {"50": 2, "150": 2, "300": 2}
    assert summary["requests_blocked"] == 0


def test_sage6b_verifies_every_live_prefix_and_context_hash(real_payload):
    summary = real_payload["summary"]
    experiments = real_payload["controlled_experiments"]

    assert summary["live_prefix_replay_exact_events"] == 6
    assert summary["context_snapshot_hash_verified_events"] == 6
    assert len(experiments) == 6
    assert all(row["live_prefix_replay_exact"] is True for row in experiments)
    assert all(row["target_context_signature_verified"] is True for row in experiments)
    assert all(row["control_context_signature_verified"] is True for row in experiments)
    assert all(row["executed_by"].startswith("SAGE.6b") for row in experiments)


def test_sage6b_records_raw_controlled_events_without_scientific_verdict(real_payload):
    summary = real_payload["summary"]

    assert summary["target_actions_executed"] == {"ACTION2": 6}
    assert summary["control_actions_executed"] == {"ACTION1": 6}
    assert summary["hypothesis_families_executed"] == {
        "local_patch_change_candidate": 6
    }
    assert summary["target_signal_total"] == 194.0
    assert summary["control_signal_total"] == 34.0
    assert summary["controlled_effect_sizes"] == [0.0, 32.0, 32.0, 32.0, 32.0, 32.0]
    assert summary["positive_effect_events"] == 5
    assert summary["negative_effect_events"] == 0
    assert summary["zero_effect_events"] == 1
    assert summary["raw_support_events"] == 5
    assert summary["raw_contradiction_events"] == 0
    assert summary["raw_neutral_events"] == 1


def test_sage6b_passes_all_gates_but_remains_candidate_only(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["outcome_status"] == execution.SAGE6B_EXECUTION_COMPLETED
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == execution.SAGE6B_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["support_events_counted_as_support"] is False
    assert real_payload["contradiction_events_counted_as_refutation"] is False
    assert real_payload["mini_frontier_execution_counted_as_evidence"] is False
    assert real_payload["source_scoped_mechanics_reused"] == 0
    assert real_payload["cross_game_mechanics_imported"] == 0
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage6b_selection_never_reads_outcomes(real_payload):
    audit = real_payload["selection_audit"]
    selected = [row for row in audit if row["selected"]]

    assert len(audit) == 20
    assert len(selected) == 6
    assert sorted(row["selection_rank"] for row in selected) == [1, 2, 3, 4, 5, 6]
    assert all(row["outcome_metrics_read_for_selection"] is False for row in audit)


def test_sage6b_runner_supports_a_smaller_stratified_quota(tmp_path):
    output = tmp_path / "sage6b.json"

    payload = execution.run_sage6b_second_unknown_game_m3_execution(
        requests_per_budget=1,
        output_path=output,
    )

    assert output.exists()
    assert payload["summary"]["requests_selected"] == 3
    assert payload["summary"]["requests_executed"] == 3
    assert payload["summary"]["requests_selected_by_budget"] == {
        "50": 1,
        "150": 1,
        "300": 1,
    }
    assert payload["summary"]["unique_selected_context_snapshot_hashes"] == 3
    assert all(payload["gate"].values())
    assert payload["outcome_status"] == execution.SAGE6B_EXECUTION_COMPLETED


def test_sage6b_rejects_mutated_scientific_or_replay_state(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        execution.validate_sage6b_source(source)

    source = copy.deepcopy(real_source)
    source["a32_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        execution.validate_sage6b_source(source)

    source = copy.deepcopy(real_source)
    source["scope_generalization_performed"] = True
    with pytest.raises(ValueError, match="cannot generalize"):
        execution.validate_sage6b_source(source)

    source = copy.deepcopy(real_source)
    source["mini_frontier_m3_requests"][0]["context_state_origin"] = "offline_frame"
    with pytest.raises(ValueError, match="ready, replayable, and scoped"):
        execution.validate_sage6b_source(source)


def test_sage6b_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    execution.write_sage6b_second_unknown_game_m3_execution(real_payload, output)

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage6b_api():
    assert sage.DEFAULT_SAGE6B_M3_EXECUTION_PATH == (
        execution.DEFAULT_SAGE6B_M3_EXECUTION_PATH
    )
    assert (
        sage.run_sage6b_second_unknown_game_m3_execution
        is execution.run_sage6b_second_unknown_game_m3_execution
    )
    assert (
        sage.write_sage6b_second_unknown_game_m3_execution
        is execution.write_sage6b_second_unknown_game_m3_execution
    )
