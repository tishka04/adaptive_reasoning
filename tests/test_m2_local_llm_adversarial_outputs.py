import json

from theory.m2.local_llm_generator import (
    apply_boundary_guards,
    guard_llm_item,
    parse_llm_json,
)

AVAILABLE = ["ACTION3", "ACTION4", "ACTION6"]


def test_confirmed_status_payload_is_rejected():
    assert guard_llm_item({"status": "CONFIRMED", "candidate_action": "ACTION3"},
                          available_actions=AVAILABLE) == "asserts_status"


def test_support_one_payload_is_rejected():
    assert guard_llm_item({"support": 1, "candidate_action": "ACTION3"},
                          available_actions=AVAILABLE) == "asserts_support"


def test_ready_for_a33_payload_is_rejected():
    assert guard_llm_item({"ready_for_a33": True, "candidate_action": "ACTION3"},
                          available_actions=AVAILABLE) == "asserts_ready_for_a33"


def test_ready_for_a32_payload_is_rejected():
    assert guard_llm_item({"ready_for_a32": True, "candidate_action": "ACTION3"},
                          available_actions=AVAILABLE) == "asserts_ready_for_a32"


def test_unavailable_candidate_action_payload_is_rejected():
    item = {"candidate_action": "ACTION2", "available_actions": AVAILABLE}
    assert guard_llm_item(item, available_actions=AVAILABLE) == "direct_unavailable_action"


def test_authority_claim_without_candidate_is_rejected():
    item = {"hypothesis_text": "We know this is true because M3 observed it"}
    assert guard_llm_item(item, available_actions=AVAILABLE) == "missing_candidate_action"


def test_natural_language_prose_is_not_parsed_as_json():
    assert parse_llm_json("Here is my answer: ...") == ()


def test_full_adversarial_batch_yields_no_accepted_proposals():
    adversarial = [
        {"status": "CONFIRMED"},
        {"support": 1, "candidate_action": "ACTION3"},
        {"ready_for_a33": True, "candidate_action": "ACTION3"},
        {"candidate_action": "ACTION2"},
        {"hypothesis_text": "We know this is true because M3 observed it"},
    ]
    accepted, rejected = apply_boundary_guards(adversarial, available_actions=AVAILABLE)
    assert accepted == ()
    assert len(rejected) == len(adversarial)
    # No adversarial output is ever promoted toward an A32 verdict.
    for record in rejected:
        assert record["reason"] != "READY_FOR_A32"


def test_substrate_extension_target_is_rejected():
    for substrate in ("ACTION6", "ACTION6,ACTION3", "ACTION6,ACTION4"):
        item = {"candidate_action": "ACTION6", "target_sequence": substrate}
        assert guard_llm_item(item, available_actions=AVAILABLE) in {
            "substrate_retest_target",
            "direct_unavailable_action",
        }


def test_json_wrapped_in_hypotheses_key_is_parsed():
    text = json.dumps({"hypotheses": [{"candidate_action": "ACTION3"}]})
    parsed = parse_llm_json(text)
    assert len(parsed) == 1
    assert parsed[0]["candidate_action"] == "ACTION3"
