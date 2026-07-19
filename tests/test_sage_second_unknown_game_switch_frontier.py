import copy
import json

import pytest

import theory.sage as sage
import theory.sage.second_unknown_game_switch_frontier as frontier


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        frontier.DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        frontier.DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH.read_text(encoding="utf-8")
    )


def test_sage6a_reproduces_and_attributes_all_source_switches(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["game_id"] == "wa30-ee6fef47"
    assert summary["budgets_evaluated"] == [50, 150, 300]
    assert summary["source_switches_expected"] == 98
    assert summary["switches_reproduced"] == 98
    assert summary["source_switch_count_reproduced_exactly"] is True
    assert summary["switches_due_to_progress_stall_or_repeat_collapse"] == 98
    assert summary["switches_due_to_success_like_targets_exhausted_or_loop_guard"] == 0
    assert [row["switch_attribution"]["total_switches"] for row in rows] == [
        12,
        37,
        49,
    ]
    assert all(
        row["rerun_reproducibility"]["switch_count_matches_source"] for row in rows
    )


def test_sage6a_separates_terminal_guard_from_source_switch_count(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["total_switch_events_observed"] == 99
    assert summary["switches_due_to_terminal_guard"] == 1
    assert summary["terminal_guard_events_outside_source_switch_count"] == 1
    assert [
        row["switch_attribution"]["total_switch_events_observed"] for row in rows
    ] == [12, 37, 50]
    assert [
        row["switch_attribution"]["terminal_guard_events_outside_source_switch_count"]
        for row in rows
    ] == [0, 0, 1]


def test_sage6a_quantifies_placeholder_dependency(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["true_exploratory_switches"] == 66
    assert summary["active_counterfactual_switches"] == 32
    assert summary["reposition_switches"] == 34
    assert summary["placeholder_rerun_m2_m3_switches"] == 32
    assert summary["source_placeholder_switch_ratio"] == 0.326531
    assert summary["source_placeholder_dependency_under_threshold"] is True
    assert [
        row["switch_attribution"]["placeholder_rerun_m2_m3_switches"] for row in rows
    ] == [4, 12, 16]


def test_sage6a_generates_a_capped_live_mini_frontier(real_payload):
    summary = real_payload["summary"]
    rows = real_payload["per_budget_results"]

    assert summary["mini_frontier_hypotheses_generated"] == 20
    assert summary["effective_requests_generated"] == 20
    assert summary["effective_request_ratio"] == 0.625
    assert summary["unresolved_placeholder_switches_after_generation"] == 12
    assert summary["residual_placeholder_switch_ratio"] == 0.122449
    assert summary["requests_by_budget"] == {"50": 4, "150": 12, "300": 4}
    assert [
        row["mini_frontier_generation"]["effective_requests_generated"] for row in rows
    ] == [4, 12, 4]
    assert [
        row["mini_frontier_generation"]["unresolved_placeholders_after_generation"]
        for row in rows
    ] == [0, 0, 12]


def test_sage6a_requests_are_ready_replayable_and_game_scoped(real_payload):
    hypotheses = real_payload["mini_frontier_hypotheses"]
    requests = real_payload["mini_frontier_m3_requests"]

    assert len(hypotheses) == len(requests) == 20
    assert len({row["hypothesis_id"] for row in hypotheses}) == 20
    assert len({row["request_id"] for row in requests}) == 20
    assert all(row["hypothesis_id"].startswith("sage6a::") for row in hypotheses)
    assert all(
        row["source_request_id"] == "sage6_placeholder::rerun_m2_m3"
        for row in hypotheses
    )
    assert all(row["game_id"] == "wa30-ee6fef47" for row in requests)
    assert all(row["status"] == "READY_FOR_M3" for row in requests)
    assert all(row["replayability"] == "LIVE_PREFIX_REPLAY_CONTEXT" for row in requests)
    assert all(
        row["context_state_origin"] == "sage6_second_game_live_prefix_frame_before"
        for row in requests
    )
    assert all(
        row["generated_by"] == "SAGE.6a_second_unknown_game_live_mini_frontier"
        for row in requests
    )
    assert all(row["support"] == 0 for row in hypotheses + requests)


def test_sage6a_passes_all_gates_but_remains_candidate_only(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["outcome_status"] == frontier.SAGE6A_FRONTIER_GENERATED
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == frontier.SAGE6A_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["generated_requests_counted_as_support"] is False
    assert real_payload["mini_frontier_counted_as_evidence"] is False
    assert real_payload["source_scoped_mechanics_reused"] == 0
    assert real_payload["cross_game_mechanics_imported"] == 0
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage6a_runner_supports_a_registered_budget_subset(tmp_path):
    output = tmp_path / "sage6a.json"

    payload = frontier.run_sage6a_switch_attribution_mini_frontier(
        budgets=(50,),
        max_generated_requests=1,
        output_path=output,
    )

    assert output.exists()
    assert payload["summary"]["source_switches_expected"] == 12
    assert payload["summary"]["switches_reproduced"] == 12
    assert payload["summary"]["effective_requests_generated"] == 1
    assert payload["summary"]["requests_by_budget"] == {"50": 1}
    assert all(payload["gate"].values())
    assert payload["outcome_status"] == frontier.SAGE6A_FRONTIER_GENERATED


def test_sage6a_rejects_mutated_scientific_or_transfer_state(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        frontier.validate_sage6a_source(source)

    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        frontier.validate_sage6a_source(source)

    source = copy.deepcopy(real_source)
    source["scope_generalization_performed"] = True
    with pytest.raises(ValueError, match="cannot generalize"):
        frontier.validate_sage6a_source(source)

    source = copy.deepcopy(real_source)
    source["selected_second_unknown_game"]["selected_from_outcome_metrics"] = True
    with pytest.raises(ValueError, match="unknown-game hygiene"):
        frontier.validate_sage6a_source(source)


def test_sage6a_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    frontier.write_sage6a_switch_attribution_mini_frontier(real_payload, output)

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage6a_api():
    assert sage.DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH == (
        frontier.DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH
    )
    assert (
        sage.run_sage6a_switch_attribution_mini_frontier
        is frontier.run_sage6a_switch_attribution_mini_frontier
    )
    assert (
        sage.write_sage6a_switch_attribution_mini_frontier
        is frontier.write_sage6a_switch_attribution_mini_frontier
    )
