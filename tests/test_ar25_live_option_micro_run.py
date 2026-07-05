"""A7b active ar25 option micro-run tests."""

from __future__ import annotations

from theory.ar25_live_option_micro_run import run_ar25_live_option_micro_run


def test_ar25_live_option_micro_run_produces_real_option_attempt():
    result = run_ar25_live_option_micro_run(max_actions=50)

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.transitions >= 1
    assert result.option_attempts >= 1
    assert result.option_invocations >= 1
    assert result.ready_observed
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    option_events = [
        event for event in result.events
        if event.kind == "option_attempt"
    ]
    assert option_events
    assert option_events[0].action == "ACTION2"
    assert option_events[0].termination in {"success", "contradiction"}
    assert "ready_to_validate_correspondence" in option_events[0].predicates_present

    probe_events = [
        event for event in result.events
        if event.kind == "bootstrap_probe"
    ]
    assert probe_events
    assert all(
        event.reason == "external_probe_not_option_policy"
        for event in probe_events
    )


def test_ar25_prepare_policy_reaches_ready_before_option_attempt():
    result = run_ar25_live_option_micro_run(
        max_actions=50,
        use_prepare_policy=True,
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.use_prepare_policy
    assert result.preparation_attempts >= 1
    assert result.ready_reached_by_prepare_policy
    assert result.option_attempts >= 1
    assert result.option_invocations >= 1
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    prep_events = [
        event for event in result.events
        if event.kind == "prepare_policy"
    ]
    assert prep_events
    assert {event.role for event in prep_events} <= {"move", "control_switch"}
    assert all(event.action != "ACTION2" for event in prep_events)
    assert any(
        "ready_to_validate_correspondence" in event.predicates_after
        for event in prep_events
    )

    option_events = [
        event for event in result.events
        if event.kind == "option_attempt"
    ]
    assert option_events
    assert option_events[0].termination in {"success", "contradiction"}
    assert "ready_to_validate_correspondence" in option_events[0].predicates_present


def test_ar25_contextual_readiness_blocks_same_failed_context():
    result = run_ar25_live_option_micro_run(
        max_actions=50,
        max_option_attempts=2,
        use_prepare_policy=True,
        use_contextual_readiness=True,
    )

    assert result.error == ""
    assert result.ready_reached_by_prepare_policy
    assert result.option_attempts == 1
    assert result.option_invocations == 1
    assert result.option_contradictions == 1
    assert result.contextual_refutations >= 1
    assert result.contextual_blocks >= 1
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    option_events = [
        event for event in result.events
        if event.kind == "option_attempt"
    ]
    block_events = [
        event for event in result.events
        if event.kind == "readiness_block"
    ]
    assert option_events
    assert block_events
    assert block_events[0].context_key == option_events[0].context_key
    assert block_events[0].readiness_status == "refuted"
    assert any(
        item["status"] == "refuted"
        for item in result.contextual_hypotheses
    )


def test_ar25_strong_preparation_invokes_option_from_strong_ready():
    result = run_ar25_live_option_micro_run(
        max_actions=50,
        max_option_attempts=1,
        use_contextual_readiness=True,
        use_strong_preparation=True,
    )

    assert result.error == ""
    assert result.use_strong_preparation
    assert result.trace_dependent is False
    assert result.strong_preparation_attempts >= 1
    assert result.strong_ready_reached_by_agent
    assert result.strong_ready_observed
    assert result.option_attempts >= 1
    assert result.option_attempts_from_strong_ready >= 1
    assert result.option_successes >= 1
    assert result.option_contradictions == 0
    assert result.contextual_blocks == 0
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    strong_events = [
        event for event in result.events
        if event.kind == "strong_prepare_policy"
    ]
    option_events = [
        event for event in result.events
        if event.kind == "option_attempt"
    ]
    assert strong_events
    assert option_events
    assert any(
        event.target_predicate == "source_target_relation_satisfied"
        for event in strong_events
    )
    assert all(
        "strong_ready_to_validate_correspondence" in event.predicates_present
        for event in option_events
    )


def test_ar25_prepare_option_chains_into_validate_option_success():
    result = run_ar25_live_option_micro_run(
        max_actions=50,
        max_option_attempts=1,
        use_contextual_readiness=True,
        use_prepare_option=True,
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.use_prepare_option
    assert result.prepare_option_invocations == 1
    assert result.prepare_option_successes > 0
    assert result.prepare_option_contradictions == 0
    assert result.prepare_option_max_steps == 0
    assert result.option_successes > 0
    assert result.full_chain_successes > 0
    assert result.option_attempts_from_strong_ready > 0
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    prepare_events = [
        event for event in result.events
        if event.kind == "prepare_option"
    ]
    validate_events = [
        event for event in result.events
        if event.kind == "option_attempt"
    ]

    assert len(prepare_events) == 1
    assert validate_events
    assert prepare_events[0].termination == "success"
    assert "ACTION3" in prepare_events[0].action
    assert "ACTION2" in prepare_events[0].action
    assert "selected_source_matches_target_shape" in prepare_events[0].target_predicate
    assert "source_target_relation_satisfied" in prepare_events[0].target_predicate
    assert validate_events[0].termination == "success"
