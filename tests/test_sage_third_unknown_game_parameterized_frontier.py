import copy
import json

import pytest

import theory.sage as sage
import theory.sage.third_unknown_game_parameterized_frontier as frontier


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        frontier.DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        frontier.DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH.read_text(encoding="utf-8")
    )


def test_sage7a_reproduces_all_switches_and_placeholders(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["game_id"] == "tn36-ab4f63cc"
    assert summary["budgets_evaluated"] == [50, 150, 300]
    assert summary["source_switches_expected"] == 172
    assert summary["switches_reproduced"] == 172
    assert summary["source_switch_count_reproduced_exactly"] is True
    assert summary["total_switch_events_observed"] == 174
    assert summary["terminal_guard_events_outside_source_switch_count"] == 2
    assert summary["true_exploratory_switches"] == 27
    assert summary["new_candidate_target_switches"] == 27
    assert summary["placeholder_rerun_m2_m3_switches"] == 145
    assert [
        row["switch_attribution"]["placeholder_rerun_m2_m3_switches"] for row in rows
    ] == [41, 52, 52]
    assert all(
        row["rerun_reproducibility"]["switch_count_matches_source"] for row in rows
    )


def test_sage7a_distributes_a_stratified_frontier_per_budget(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["requests_per_budget_target"] == 6
    assert summary["mini_frontier_hypotheses_generated"] == 18
    assert summary["effective_requests_generated"] == 18
    assert summary["requests_by_budget"] == {"50": 6, "150": 6, "300": 6}
    assert summary["effective_request_ratio"] == 0.124138
    assert summary["unselected_placeholder_switches"] == 127
    assert [
        row["parameterized_frontier_generation"]["selected_placeholder_ordinals"]
        for row in rows
    ] == [
        [0, 8, 16, 24, 32, 40],
        [0, 10, 20, 30, 40, 51],
        [0, 10, 20, 30, 40, 51],
    ]
    assert all(
        row["parameterized_frontier_generation"]["effective_requests_generated"] == 6
        for row in rows
    )


def test_sage7a_pre_registers_same_action_parameterized_controls(real_payload):
    requests = real_payload["mini_frontier_m3_requests"]
    summary = real_payload["summary"]

    assert len(requests) == 18
    assert summary["action_families"] == ["ACTION6"]
    assert summary["distinct_action_families"] == 1
    assert summary["distinct_target_parameter_variants"] == 3
    assert summary["distinct_control_parameter_variants"] == 4
    assert summary["parameterized_control_variants_pre_registered"] == 36
    assert summary["parameterized_variants_counted_as_distinct_actions"] is False
    for request in requests:
        controls = request["pre_registered_parameterized_control_variants"]
        assert request["target_action"] == "ACTION6"
        assert request["target_action_args"]
        assert request["parameterized_action_family"] == "ACTION6"
        assert request["suggested_control_actions"] == ["ACTION6"]
        assert request["parameterized_control_policy"] == (
            frontier.PARAMETERIZED_CONTROL_POLICY
        )
        assert request["parameterized_controls_counted_as_distinct_actions"] is False
        assert len(controls) == 2
        assert all(row["action"] == "ACTION6" for row in controls)
        assert all(
            row["action_args"] != request["target_action_args"] for row in controls
        )
        assert all(row["live_legal_at_context"] is True for row in controls)
        assert all(row["counted_as_distinct_action"] is False for row in controls)


def test_sage7a_requests_are_ready_and_exactly_replayable(real_payload):
    hypotheses = real_payload["mini_frontier_hypotheses"]
    requests = real_payload["mini_frontier_m3_requests"]

    assert len({row["hypothesis_id"] for row in hypotheses}) == 18
    assert len({row["request_id"] for row in requests}) == 18
    assert all(row["hypothesis_id"].startswith("sage7a::") for row in hypotheses)
    assert all(
        row["source_request_id"] == "sage7_placeholder::rerun_m2_m3"
        for row in hypotheses
    )
    assert all(row["game_id"] == "tn36-ab4f63cc" for row in requests)
    assert all(row["status"] == "READY_FOR_M3" for row in requests)
    assert all(row["replayability"] == "LIVE_PREFIX_REPLAY_CONTEXT" for row in requests)
    assert all(
        row["context_state_origin"] == "sage7_third_game_live_prefix_frame_before"
        for row in requests
    )
    assert all(len(row["context_replay"]) == row["source_step"] for row in requests)
    assert all(
        len(row["context_replay_args"]) == len(row["context_replay"])
        for row in requests
    )
    assert all(row["context_snapshot_hash"] for row in requests)
    assert real_payload["summary"]["unique_context_snapshot_hashes"] == 10
    assert all(row["support"] == 0 for row in hypotheses + requests)


def test_sage7a_passes_gates_but_does_not_open_a32(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["summary"]["ready_for_parameterized_m3_execution"] is True
    assert real_payload["summary"]["required_next_step"] == (
        frontier.SAGE7A_PARAMETERIZED_EXECUTION_REQUIRED
    )
    assert real_payload["outcome_status"] == frontier.SAGE7A_FRONTIER_GENERATED
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == frontier.SAGE7A_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["execution_performed"] is False
    assert real_payload["generated_requests_counted_as_support"] is False
    assert real_payload["mini_frontier_counted_as_evidence"] is False
    assert real_payload["parameterized_controls_counted_as_distinct_actions"] is False
    assert real_payload["source_scoped_mechanics_reused"] == 0
    assert real_payload["cross_game_mechanics_imported"] == 0
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage7a_runner_supports_a_registered_budget_subset(tmp_path):
    output = tmp_path / "sage7a.json"

    payload = frontier.run_sage7a_parameterized_mini_frontier(
        budgets=(50,),
        requests_per_budget=2,
        output_path=output,
    )

    assert output.exists()
    assert payload["summary"]["source_switches_expected"] == 50
    assert payload["summary"]["switches_reproduced"] == 50
    assert payload["summary"]["effective_requests_generated"] == 2
    assert payload["summary"]["requests_by_budget"] == {"50": 2}
    assert all(payload["gate"].values())
    assert payload["outcome_status"] == frontier.SAGE7A_FRONTIER_GENERATED


def test_stratified_placeholder_selection_is_fixed_before_outcomes():
    assert frontier.stratified_placeholder_ordinals(41, 6) == [0, 8, 16, 24, 32, 40]
    assert frontier.stratified_placeholder_ordinals(52, 6) == [0, 10, 20, 30, 40, 51]
    assert frontier.stratified_placeholder_ordinals(3, 6) == [0, 1, 2]
    assert frontier.stratified_placeholder_ordinals(0, 6) == []


def test_sage7a_rejects_mutated_scientific_or_action_state(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        frontier.validate_sage7a_source(source)

    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        frontier.validate_sage7a_source(source)

    source = copy.deepcopy(real_source)
    source["cross_game_mechanics_imported"] = 1
    with pytest.raises(ValueError, match="cannot import or generalize"):
        frontier.validate_sage7a_source(source)

    source = copy.deepcopy(real_source)
    source["action_surface_audit"][
        "parameterized_action_variants_counted_as_distinct_actions"
    ] = True
    with pytest.raises(ValueError, match="cannot be relabelled"):
        frontier.validate_sage7a_source(source)

    source = copy.deepcopy(real_source)
    source["summary"]["ready_for_parameterized_mini_frontier"] = False
    with pytest.raises(ValueError, match="explicitly require"):
        frontier.validate_sage7a_source(source)


def test_sage7a_requires_two_control_variants():
    with pytest.raises(ValueError, match="at least two parameterized controls"):
        frontier.run_sage7a_parameterized_mini_frontier(
            budgets=(50,),
            requests_per_budget=1,
            controls_per_request=1,
        )


def test_sage7a_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    frontier.write_sage7a_parameterized_mini_frontier(real_payload, output)

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage7a_api():
    assert sage.DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH == (
        frontier.DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH
    )
    assert (
        sage.run_sage7a_parameterized_mini_frontier
        is frontier.run_sage7a_parameterized_mini_frontier
    )
    assert (
        sage.write_sage7a_parameterized_mini_frontier
        is frontier.write_sage7a_parameterized_mini_frontier
    )
