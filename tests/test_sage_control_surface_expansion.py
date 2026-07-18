import copy

import pytest

from theory.sage import control_surface_expansion as expansion


@pytest.fixture(scope="module")
def real_payload():
    return expansion.run_sage5i_control_surface_expansion()


def test_sage5i_exhaustively_audits_real_replayable_candidate_contexts(real_payload):
    summary = real_payload["summary"]

    assert summary["source_requests_available"] == 24
    assert summary["candidates_evaluated"] == 2
    assert summary["candidate_contexts_considered"] == 24
    assert summary["unique_candidate_contexts"] == 24
    assert summary["context_surface_audits"] == 24
    assert summary["replay_exact_context_surface_audits"] == 24
    assert summary["action_surface_signature_counts"] == {"ACTION5,ACTION6": 24}
    assert summary["candidate_context_counts"] == {"ACTION5": 13, "ACTION6": 11}
    assert summary["contexts_with_third_action_family"] == 0
    assert summary["candidates_with_action_distinct_surface_exhausted"] == 2
    assert summary["candidates_with_new_distinct_control"] == 0
    assert summary["control_expansion_experiments_executed"] == 0
    assert summary["bounded_action_distinct_exhaustion_proven"] is True
    assert summary["all_candidate_contexts_audited_exact"] is True
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == expansion.SAGE5I_ACTION_DISTINCT_EXHAUSTED


def test_sage5i_real_surface_retains_parameterized_options_without_relabelling(
    real_payload,
):
    by_action = {
        row["action"]: row for row in real_payload["candidate_control_surface_results"]
    }

    assert by_action["ACTION6"]["contexts_scanned"] == 11
    assert by_action["ACTION6"]["unique_contexts"] == 11
    assert by_action["ACTION6"]["parameterized_control_option_count"] == 3
    assert by_action["ACTION5"]["contexts_scanned"] == 13
    assert by_action["ACTION5"]["unique_contexts"] == 13
    assert by_action["ACTION5"]["parameterized_control_option_count"] == 4
    assert all(
        row["counted_as_action_distinct_control"] is False
        for surface in by_action.values()
        for row in surface["parameterized_control_options"]
    )


def test_sage5i_real_candidates_remain_blocked_on_control_diversity(real_payload):
    assessments = real_payload["updated_candidate_assessments"]

    assert [row["action"] for row in assessments] == ["ACTION6", "ACTION5"]
    assert all(row["raw_support_events_after"] == 3 for row in assessments)
    assert all(row["independent_context_events_after"] == 3 for row in assessments)
    assert all(row["distinct_control_actions_after"] == 1 for row in assessments)
    assert all(
        row["missing_revision_requirements"] == ["minimum_distinct_control_actions"]
        for row in assessments
    )
    assert all(
        row["a32_intake_recommendation"]
        == expansion.A32_PARAMETERIZED_PROTOCOL_REQUIRED
        for row in assessments
    )
    assert all(row["ready_for_A32_intake"] is False for row in assessments)


def test_sage5i_builds_explicit_a32_protocol_decision_proposal(real_payload):
    proposal = real_payload["a32_parameterized_control_protocol_proposal"]

    assert proposal["proposal_required"] is True
    assert proposal["proposal_status"] == (
        "A32_REVIEW_REQUIRED_DO_NOT_AUTO_RELAX_CRITERION"
    )
    assert proposal["historical_requirement"] == {
        "minimum_distinct_control_action_names": 2,
        "currently_satisfied": False,
    }
    assert proposal["parameterized_control_option_counts"] == {
        "sage5g::a32_review_candidate::001": 3,
        "sage5g::a32_review_candidate::002": 4,
    }
    assert proposal["recommendation"] == (
        "A32_DECISION_REQUIRED_NO_AUTOMATIC_CONFIRMATION"
    )
    assert proposal["protocol_proposal_counted_as_revision"] is False


def test_sage5i_keeps_audit_and_protocol_candidate_only(real_payload):
    assert real_payload["support"] == 0
    assert real_payload["truth_status"] == expansion.SAGE5I_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["execution_performed"] is True
    assert real_payload["revision_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert (
        real_payload["bounded_control_surface_exhaustion_counted_as_refutation"]
        is False
    )
    assert real_payload["parameterized_controls_counted_as_distinct_actions"] is False
    assert (
        real_payload["control_expansion_events_counted_as_scientific_support"] is False
    )
    assert real_payload["protocol_proposal_counted_as_revision"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage5i_rejects_source_that_counts_support():
    sage5e, sage5g, sage5h = _valid_sources()
    sage5e["support"] = 1

    with pytest.raises(ValueError, match="SAGE.5e support must remain 0"):
        expansion.validate_sage5i_sources(sage5e, sage5g, sage5h)


def test_sage5i_rejects_source_write_or_refutation_relabelling():
    sage5e, sage5g, sage5h = _valid_sources()
    sage5g["a32_write_performed"] = True

    with pytest.raises(ValueError, match="SAGE.5g cannot write A32/A33"):
        expansion.validate_sage5i_sources(sage5e, sage5g, sage5h)

    sage5e, sage5g, sage5h = _valid_sources()
    sage5h["control_surface_block_counted_as_refutation"] = True
    with pytest.raises(ValueError, match="cannot count as refutation"):
        expansion.validate_sage5i_sources(sage5e, sage5g, sage5h)


def test_candidate_context_selection_requires_the_full_effect_signature():
    candidate = _candidate()
    matching = _request("matching")
    wrong_action = {**_request("wrong-action"), "target_action": "ACTION5"}
    wrong_effect = copy.deepcopy(_request("wrong-effect"))
    wrong_effect["diff_signature"]["changed_cells"] = 2

    selected = expansion.select_candidate_context_requests(
        candidate,
        [wrong_effect, wrong_action, matching],
    )

    assert [row["request_id"] for row in selected] == ["matching"]


def test_parameterized_options_exclude_exact_target_and_preserve_roles():
    options = expansion.parameterized_control_options(
        candidate=_candidate(),
        existing_controls=["ACTION5"],
        available_variants=[
            {"action": "ACTION6", "action_args": {"x": 1, "y": 1}},
            {"action": "ACTION6", "action_args": {"x": 3, "y": 1}},
            {"action": "ACTION5", "action_args": {}},
            {"action": "ACTION5", "action_args": {"mode": 2}},
            {"action": "ACTION3", "action_args": {}},
        ],
    )

    assert options == [
        {
            "action": "ACTION5",
            "action_args": {"mode": 2},
            "parameterized_control_role": "existing_control_parameter_variant",
            "counted_as_action_distinct_control": False,
        },
        {
            "action": "ACTION6",
            "action_args": {"x": 3, "y": 1},
            "parameterized_control_role": "same_action_alternative_args",
            "counted_as_action_distinct_control": False,
        },
    ]


def test_action_distinct_expansion_executes_two_exact_eligible_contexts(monkeypatch):
    experiments = []

    def fake_execute(**kwargs):
        result = {
            "experiment_id": f"experiment-{len(experiments) + 1}",
            "execution_status": "EXECUTED",
            "live_prefix_replay_exact": True,
            "support_events": 1,
            "contradiction_events": 0,
            "truth_status": "NOT_EVALUATED_BY_SAGE_5H",
        }
        experiments.append(result)
        return result

    monkeypatch.setattr(expansion, "execute_followup_experiment", fake_execute)
    requests = [_request("context-1"), _request("context-2")]
    surface = {
        "eligible_contexts": [
            {
                "source_request_id": "context-1",
                "live_prefix_replay_exact": True,
                "eligible_distinct_control_actions": ["ACTION3"],
            },
            {
                "source_request_id": "context-2",
                "live_prefix_replay_exact": True,
                "eligible_distinct_control_actions": ["ACTION3"],
            },
        ]
    }

    result = expansion.execute_candidate_control_expansion(
        candidate=_candidate(),
        candidate_requests=requests,
        surface=surface,
        min_control_contexts=2,
        environments_dir=None,
        env_factory=None,
        execution_cache={},
        experiments=[],
    )

    assert result["expansion_status"] == expansion.EXPANSION_ACQUIRED
    assert result["new_control_action"] == "ACTION3"
    assert result["controlled_experiment_ids"] == ["experiment-1", "experiment-2"]
    assert result["raw_support_events"] == 2
    assert result["raw_contradiction_events"] == 0
    assert result["new_distinct_control_action_acquired"] is True
    assert result["support"] == 0


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
    sage5h = {
        **copy.deepcopy(base),
        "control_surface_block_counted_as_refutation": False,
        "candidate_assessment_counted_as_revision": False,
    }
    return copy.deepcopy(base), copy.deepcopy(base), sage5h


def _candidate():
    return {
        "candidate_id": "candidate-1",
        "candidate_key": "mechanic_prediction::fake",
        "game_id": "fake",
        "action": "ACTION6",
        "action_args": {"x": 1, "y": 1},
        "predicted_metric": "local_patch_before_after",
        "predicted_effect_signature": {
            "changed_cells": 1,
            "color_transitions": {"0->6": 1},
            "terminal_after": False,
            "levels_delta": 0,
        },
        "control_interventions": [{"action": "ACTION5"}],
    }


def _request(request_id):
    return {
        "request_id": request_id,
        "source_transition_id": "sage5e::fake::budget_050::step_0000",
        "source_step": 0,
        "game_id": "fake",
        "target_action": "ACTION6",
        "target_action_args": {"x": 1, "y": 1},
        "metric": "local_patch_before_after",
        "diff_signature": {
            "changed_cells": 1,
            "color_transitions": {"0->6": 1},
            "terminal_after": False,
            "levels_delta": 0,
        },
    }
