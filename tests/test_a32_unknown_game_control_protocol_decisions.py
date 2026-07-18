import copy
import json

import pytest

from theory.a32 import unknown_game_control_protocol_decisions as decisions


@pytest.fixture(scope="module")
def real_payload():
    return decisions.run_a32_unknown_game_control_protocol_decision_consumer()


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        decisions.DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_a32_4_authorizes_real_parameterized_protocol_without_verdict(real_payload):
    summary = real_payload["summary"]

    assert summary["source_candidates_consumed"] == 2
    assert summary["protocol_decisions"] == 2
    assert summary["parameterized_protocols_authorized"] == 2
    assert (
        summary["strict_action_distinct_requirements_retained_without_exception"] == 0
    )
    assert summary["candidates_rejected_as_unidentifiable"] == 0
    assert summary["authorized_parameter_variants"] == 4
    assert summary["preregistered_followup_experiments"] == 8
    assert summary["decision_records_unresolved"] == 2
    assert summary["decision_records_confirmed"] == 0
    assert summary["decision_records_refuted"] == 0
    assert summary["outcome_status"] == decisions.A32_4_PROTOCOL_AUTHORIZED


def test_a32_4_selects_extreme_variants_and_cross_budget_contexts(real_payload):
    action6, action5 = real_payload["protocol_decisions"]

    assert action6["action"] == "ACTION6"
    assert [row["action_args"] for row in action6["authorized_control_variants"]] == [
        {"x": 18, "y": 57},
        {"x": 42, "y": 57},
    ]
    assert [row["budget"] for row in action6["preregistered_experiments"]] == [
        50,
        300,
        50,
        300,
    ]

    assert action5["action"] == "ACTION5"
    assert [row["action_args"] for row in action5["authorized_control_variants"]] == [
        {"x": 21, "y": 28},
        {"x": 39, "y": 28},
    ]
    assert [row["budget"] for row in action5["preregistered_experiments"]] == [
        50,
        300,
        50,
        300,
    ]


def test_a32_4_preregistration_is_executable_and_non_relabelled(real_payload):
    experiments = real_payload["requested_followup_experiments"]

    assert len(experiments) == 8
    assert len({row["experiment_id"] for row in experiments}) == 8
    assert len({row["source_request_id"] for row in experiments}) == 8
    assert all(row["paired_target_control_required"] is True for row in experiments)
    assert all(row["exact_context_replay_required"] is True for row in experiments)
    assert all(
        row["same_measurement_for_target_and_control_required"] is True
        for row in experiments
    )
    assert all(row["measurement"] == "local_patch_before_after" for row in experiments)
    assert all(
        row["evaluation_rule"]
        == "compare_paired_target_and_control_effect_signatures_without_post_hoc_metric_change"
        for row in experiments
    )
    assert all(
        row["parameterized_control_counted_as_distinct_action"] is False
        for row in experiments
    )
    assert all(row["status"] == "PRE_REGISTERED_NOT_EXECUTED" for row in experiments)
    assert all(row["support"] == 0 for row in experiments)


def test_a32_4_keeps_both_hypotheses_unresolved(real_payload):
    assert all(
        row["decision"] == decisions.AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL
        for row in real_payload["protocol_decisions"]
    )
    assert all(
        row["input_record"]["status"] == "unresolved"
        for row in real_payload["protocol_decisions"]
    )
    assert all(
        row["decision_record"]["status"] == "unresolved"
        for row in real_payload["protocol_decisions"]
    )
    assert all(
        row["decision_record"]["support"] == 0
        for row in real_payload["protocol_decisions"]
    )
    assert all(
        row["revision_performed"] is False for row in real_payload["protocol_decisions"]
    )
    assert all(
        row["confirmation_performed"] is False
        for row in real_payload["protocol_decisions"]
    )
    assert all(
        row["refutation_performed"] is False
        for row in real_payload["protocol_decisions"]
    )
    assert all(row["a33_ready"] is False for row in real_payload["protocol_decisions"])


def test_a32_4_retains_global_guardrails(real_payload):
    assert real_payload["scientific_review_performed"] is True
    assert real_payload["protocol_decision_performed"] is True
    assert real_payload["experimental_protocol_authorized"] is True
    assert real_payload["execution_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_ready"] is False
    assert real_payload["a33_write_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert real_payload["support"] == 0
    assert real_payload["action_distinct_requirement_retained_as_default"] is True
    assert real_payload["parameterized_controls_counted_as_distinct_actions"] is False
    assert (
        real_payload["parameterized_controls_counted_as_evidence_before_execution"]
        is False
    )
    assert real_payload["bounded_exhaustion_counted_as_refutation"] is False


def test_a32_4_rejects_source_support_or_prior_a32_write(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        decisions.validate_sage5i_protocol_source(source)

    source = copy.deepcopy(real_source)
    source["a32_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        decisions.validate_sage5i_protocol_source(source)


def test_a32_4_rejects_relabelled_controls_or_refutation(real_source):
    source = copy.deepcopy(real_source)
    source["parameterized_controls_counted_as_distinct_actions"] = True
    with pytest.raises(ValueError, match="cannot relabel parameterized controls"):
        decisions.validate_sage5i_protocol_source(source)

    source = copy.deepcopy(real_source)
    source["bounded_control_surface_exhaustion_counted_as_refutation"] = True
    with pytest.raises(ValueError, match="cannot count as refutation"):
        decisions.validate_sage5i_protocol_source(source)


def test_a32_4_requires_exact_aligned_candidate_evidence(real_source):
    source = copy.deepcopy(real_source)
    source["context_surface_audits"][0]["live_prefix_replay_exact"] = False
    with pytest.raises(ValueError, match="must be replay-exact"):
        decisions.validate_sage5i_protocol_source(source)

    source = copy.deepcopy(real_source)
    source["updated_candidate_assessments"][0]["candidate_id"] = "misaligned"
    with pytest.raises(ValueError, match="candidate ids must align"):
        decisions.validate_sage5i_protocol_source(source)


def test_protocol_decision_retains_strict_rule_when_surface_is_not_exhausted(
    real_source,
):
    surface = copy.deepcopy(real_source["candidate_control_surface_results"][0])
    assessment = copy.deepcopy(real_source["updated_candidate_assessments"][0])
    surface["control_surface_exhausted_action_distinct"] = False

    reasons = decisions.protocol_decision_reasons(
        surface=surface,
        assessment=assessment,
    )

    assert "action_distinct_surface_not_exhausted" in reasons
    assert decisions.protocol_decision_label(reasons) == (
        decisions.RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT
    )


def test_protocol_decision_rejects_unidentifiable_surface_without_refutation(
    real_source,
):
    surface = copy.deepcopy(real_source["candidate_control_surface_results"][0])
    assessment = copy.deepcopy(real_source["updated_candidate_assessments"][0])
    surface["parameterized_control_option_count"] = 1

    reasons = decisions.protocol_decision_reasons(
        surface=surface,
        assessment=assessment,
    )

    assert "insufficient_parameterized_control_variants" in reasons
    assert decisions.protocol_decision_label(reasons) == (
        decisions.REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE
    )


def test_variant_selection_is_deterministic_and_excludes_relabelled_options():
    selected = decisions.select_preregistered_variants(
        [
            _option(34),
            _option(100),
            _option(18),
            {**_option(10), "counted_as_action_distinct_control": True},
        ]
    )

    assert [row["action_args"] for row in selected] == [
        {"x": 18, "y": 57},
        {"x": 100, "y": 57},
    ]
    assert all(row["counted_as_action_distinct_control"] is False for row in selected)


def test_a32_4_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "a32_4.json"

    decisions.write_a32_unknown_game_control_protocol_decisions(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload


def _option(x):
    return {
        "action": "ACTION6",
        "action_args": {"x": x, "y": 57},
        "parameterized_control_role": "same_action_alternative_args",
        "counted_as_action_distinct_control": False,
    }
