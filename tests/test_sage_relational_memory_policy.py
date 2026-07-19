import copy
import json

import pytest

from theory.sage import relational_memory_policy as policy


@pytest.fixture(scope="module")
def real_payload():
    return policy.run_sage8a_relational_memory_policy_integration()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            policy.DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH,
            policy.DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH,
        )
    )


def _options(entry):
    return (
        policy.PolicyActionOption(
            entry["lower_effect_action"], entry["lower_effect_action_args"]
        ),
        policy.PolicyActionOption(
            entry["selected_action"], entry["selected_action_args"]
        ),
        policy.PolicyActionOption(
            entry["equivalent_action"], entry["equivalent_action_args"]
        ),
    )


def _decide(payload, entry, proposed, *, game_id=None, context_hash=None, options=None):
    return policy.apply_relational_memory_policy(
        payload["policy"],
        game_id=game_id or entry["game_id"],
        context_snapshot_hash=context_hash or entry["context_snapshot_hashes"][0],
        proposed_action_raw=proposed,
        valid_actions=options or _options(entry),
        metric=entry["metric"],
    )


def test_sage8a_compiles_both_registries_for_all_exact_contexts(real_payload):
    summary = real_payload["summary"]

    assert summary["registry_entries_consumed"] == 2
    assert summary["policy_entries_compiled"] == 2
    assert summary["games_scoped"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["exact_context_hashes_scoped"] == 11
    assert summary["exact_application_audits"] == 2
    assert summary["equivalent_comparators_preserved"] == 2
    assert summary["wrong_context_overrides"] == 0
    assert summary["wrong_game_overrides"] == 0
    assert summary["gate_passed"] is True
    assert summary["ready_for_comparative_evaluation"] is True
    assert summary["outcome_status"] == policy.SAGE8A_POLICY_READY


@pytest.mark.parametrize("entry_index", [0, 1])
def test_sage8a_replaces_exact_lower_effect_comparator(real_payload, entry_index):
    entry = real_payload["policy_entries"][entry_index]
    options = _options(entry)

    decision = _decide(real_payload, entry, options[0], options=options)

    assert decision["relational_memory_applied"] is True
    assert decision["decision_reason"] == policy.LOWER_EFFECT_COMPARATOR_MATCH
    assert decision["selected_action"] == entry["selected_action"]
    assert decision["selected_action_args"] == entry["selected_action_args"]
    assert decision["selected_action_raw"] is options[1]
    assert decision["scope_match"] is True
    assert decision["selected_action_live_legal"] is True


@pytest.mark.parametrize("entry_index", [0, 1])
def test_sage8a_preserves_equivalent_comparator(real_payload, entry_index):
    entry = real_payload["policy_entries"][entry_index]
    options = _options(entry)

    decision = _decide(real_payload, entry, options[2], options=options)

    assert decision["relational_memory_applied"] is False
    assert decision["equivalent_comparator_preserved"] is True
    assert decision["decision_reason"] == policy.EQUIVALENT_COMPARATOR_PRESERVED
    assert decision["selected_action_raw"] is options[2]


def test_sage8a_falls_back_outside_exact_context_and_game(real_payload):
    entry = real_payload["policy_entries"][0]
    options = _options(entry)

    wrong_context = _decide(
        real_payload, entry, options[0], context_hash="unknown-context", options=options
    )
    wrong_game = _decide(
        real_payload, entry, options[0], game_id="unknown-game", options=options
    )

    assert wrong_context["decision_reason"] == policy.CONTEXT_OUT_OF_SCOPE
    assert wrong_game["decision_reason"] == policy.GAME_OUT_OF_SCOPE
    assert wrong_context["selected_action_raw"] is options[0]
    assert wrong_game["selected_action_raw"] is options[0]
    assert wrong_context["relational_memory_applied"] is False
    assert wrong_game["relational_memory_applied"] is False


def test_sage8a_does_not_override_an_unregistered_proposed_action(real_payload):
    entry = real_payload["policy_entries"][0]
    options = _options(entry)
    proposed = policy.PolicyActionOption("ACTION4")

    decision = _decide(real_payload, entry, proposed, options=options + (proposed,))

    assert decision["decision_reason"] == policy.PROPOSED_ACTION_OUT_OF_SCOPE
    assert decision["selected_action_raw"] is proposed
    assert decision["relational_memory_applied"] is False


def test_sage8a_requires_the_registered_target_to_be_live_legal(real_payload):
    entry = real_payload["policy_entries"][1]
    options = _options(entry)

    decision = _decide(
        real_payload, entry, options[0], options=(options[0], options[2])
    )

    assert decision["decision_reason"] == policy.TARGET_UNAVAILABLE
    assert decision["selected_action_raw"] is options[0]
    assert decision["selected_action_live_legal"] is False
    assert decision["relational_memory_applied"] is False


def test_sage8a_does_not_reevaluate_truth_or_recount_support(real_payload):
    assert real_payload["truth_status"] == policy.SAGE8A_TRUTH_STATUS
    assert real_payload["policy_integration_performed"] is True
    assert real_payload["comparative_evaluation_performed"] is False
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8a_rejects_mutated_registry_sources(real_sources):
    a33_3, a33_4 = real_sources
    changed = copy.deepcopy(a33_3)
    changed["control_dependent_relational_contrasts"][0]["target_action"] = "ACTION3"
    with pytest.raises(ValueError, match="exact scope-locked A33.3"):
        policy.validate_relational_memory_registry_sources(changed, a33_4)

    changed = copy.deepcopy(a33_4)
    changed["parameterized_relational_contrasts"][0]["target_action_args"]["x"] = 26
    with pytest.raises(ValueError, match="exact scope-locked A33.4"):
        policy.validate_relational_memory_registry_sources(a33_3, changed)


def test_sage8a_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8a.json"
    policy.write_sage8a_relational_memory_policy(real_payload, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert sage.PolicyActionOption is policy.PolicyActionOption
    assert sage.RelationalMemoryPolicyEntry is policy.RelationalMemoryPolicyEntry
    assert sage.apply_relational_memory_policy is policy.apply_relational_memory_policy
    assert (
        sage.run_sage8a_relational_memory_policy_integration
        is policy.run_sage8a_relational_memory_policy_integration
    )
