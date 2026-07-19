import copy
import json

import pytest

import theory.sage as sage
from theory.sage import second_unknown_game_handoff_compiler as compiler


def test_sage6d_real_artifacts_compile_expected_followups():
    payload = compiler.run_sage6d_second_unknown_game_handoff()

    assert payload["outcome_status"] == compiler.SAGE6D_HANDOFF_COMPILED
    assert payload["summary"] == {
        "source_sage6c_outcome_status": (
            "SAGE_SECOND_UNKNOWN_GAME_EVENTS_CONSOLIDATED_CANDIDATE_ONLY"
        ),
        "game_id": "wa30-ee6fef47",
        "budgets": [50, 150, 300],
        "source_candidate_handoff_frontiers": 1,
        "handoff_items": 1,
        "pre_registered_followup_protocols": 4,
        "control_diversity_protocols": 3,
        "control_diversity_budgets": [50, 150, 300],
        "neutral_context_replication_protocols": 1,
        "pre_registered_new_control_actions": ["ACTION3"],
        "executed_distinct_control_actions": 1,
        "projected_distinct_control_actions_after_execution": 2,
        "context_clusters_preserved": 6,
        "protocol_contexts": 4,
        "raw_support_events": 5,
        "raw_contradiction_events": 0,
        "raw_neutral_events": 1,
        "ready_for_followup_execution": 1,
        "ready_for_A32_review": 0,
        "execution_performed": False,
        "gate_passed": True,
        "outcome_status": compiler.SAGE6D_HANDOFF_COMPILED,
        "support": 0,
        "truth_status": compiler.SAGE6D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }

    protocols = payload["pre_registered_followup_protocols"]
    assert [row["request_type"] for row in protocols] == [
        compiler.CONTROL_DIVERSITY_REQUEST,
        compiler.CONTROL_DIVERSITY_REQUEST,
        compiler.CONTROL_DIVERSITY_REQUEST,
        compiler.NEUTRAL_REPLICATION_REQUEST,
    ]
    assert [row["budget"] for row in protocols] == [50, 150, 300, 50]
    assert [row["source_step"] for row in protocols] == [48, 132, 24, 12]
    assert [row["control_action"] for row in protocols] == [
        "ACTION3",
        "ACTION3",
        "ACTION3",
        "ACTION1",
    ]
    assert [row["expected_prior_effect_size"] for row in protocols] == [
        32.0,
        32.0,
        32.0,
        0.0,
    ]


def test_sage6d_handoff_preserves_all_cluster_boundaries():
    payload = compiler.run_sage6d_second_unknown_game_handoff()
    handoff = payload["handoff_items"][0]
    manifest = handoff["context_cluster_manifest"]

    assert handoff["handoff_status"] == (compiler.HANDOFF_READY_FOR_FOLLOWUP_EXECUTION)
    assert handoff["ready_for_followup_execution"] is True
    assert handoff["ready_for_A32_review"] is False
    assert handoff["a32_intake_requested"] is False
    assert handoff["executed_control_actions"] == ["ACTION1"]
    assert handoff["pre_registered_new_control_actions"] == ["ACTION3"]
    assert handoff["projected_control_actions_after_execution"] == [
        "ACTION1",
        "ACTION3",
    ]
    assert len(manifest) == 6
    assert sum(row["frontier_role"] == "STABLE_PATTERN" for row in manifest) == 5
    assert sum(row["frontier_role"] == "CONTEXTUAL_EXCEPTION" for row in manifest) == 1
    assert (
        sum(row["selected_followup_protocol_id"] is not None for row in manifest) == 4
    )
    assert all(row["context_preserved"] for row in manifest)
    assert not any(row["cross_context_merge_performed"] for row in manifest)


def test_sage6d_protocols_are_exact_pre_registrations_only():
    payload = compiler.run_sage6d_second_unknown_game_handoff()
    protocols = payload["pre_registered_followup_protocols"]

    assert all(row["target_action"] == "ACTION2" for row in protocols)
    assert all(row["exact_replay_required"] for row in protocols)
    assert all(row["context_hash_verification_required"] for row in protocols)
    assert all(
        row["target_and_control_must_start_from_same_context"] for row in protocols
    )
    assert not any(row["cross_context_substitution_allowed"] for row in protocols)
    assert not any(row["cross_context_merge_allowed"] for row in protocols)
    assert all(
        row["execution_status"] == "PRE_REGISTERED_NOT_EXECUTED" for row in protocols
    )
    assert not any(row["execution_performed"] for row in protocols)
    assert len({row["context_snapshot_hash"] for row in protocols}) == 4
    assert all(
        len(row["context_replay"]) == len(row["context_replay_args"])
        for row in protocols
    )
    assert protocols[-1]["pre_registered_interpretation"] == {
        "consistency_condition": "controlled_effect_size_equals_0",
        "deviation_condition": "controlled_effect_size_not_equal_to_0",
        "any_outcome_remains_raw_candidate_only": True,
        "outcome_cannot_trigger_automatic_A32_or_A33_write": True,
    }


def test_sage6d_gate_and_top_level_quarantine_are_complete():
    payload = compiler.run_sage6d_second_unknown_game_handoff()

    assert payload["gate"]
    assert all(payload["gate"].values())
    assert payload["support"] == 0
    assert payload["truth_status"] == compiler.SAGE6D_TRUTH_STATUS
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["execution_performed"] is False
    assert payload["revision_performed"] is False
    assert payload["confirmation_performed"] is False
    assert payload["refutation_performed"] is False
    assert payload["a32_intake_requested"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert payload["source_scoped_mechanics_reused"] == 0
    assert payload["cross_game_mechanics_imported"] == 0
    assert payload["scope_generalization_performed"] is False


def test_sage6d_runner_is_pure_without_output_and_does_not_mutate_sources(tmp_path):
    source6c = _load_default(compiler.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH)
    source6b = _load_default(compiler.DEFAULT_SAGE6B_M3_EXECUTION_PATH)
    source6c_before = copy.deepcopy(source6c)
    source6b_before = copy.deepcopy(source6b)
    source6c_path = tmp_path / "sage6c.json"
    source6b_path = tmp_path / "sage6b.json"
    source6c_path.write_text(json.dumps(source6c), encoding="utf-8")
    source6b_path.write_text(json.dumps(source6b), encoding="utf-8")

    compiler.run_sage6d_second_unknown_game_handoff(
        source_sage6c_path=source6c_path,
        source_sage6b_path=source6b_path,
    )

    assert json.loads(source6c_path.read_text(encoding="utf-8")) == source6c_before
    assert json.loads(source6b_path.read_text(encoding="utf-8")) == source6b_before
    assert set(tmp_path.iterdir()) == {source6c_path, source6b_path}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source.__setitem__("support", 1), "support 0"),
        (
            lambda source: source.__setitem__("a32_write_performed", True),
            "registry writes",
        ),
        (
            lambda source: source["candidate_handoff_frontiers"][0].__setitem__(
                "ready_for_A32_review", True
            ),
            "not eligible",
        ),
        (
            lambda source: source["context_clusters"][0].__setitem__(
                "cross_context_merge_performed", True
            ),
            "singleton and unmerged",
        ),
    ],
)
def test_sage6d_rejects_invalid_sage6c_sources(tmp_path, mutation, message):
    source6c = _load_default(compiler.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH)
    mutation(source6c)
    source6c_path = tmp_path / "invalid_sage6c.json"
    source6c_path.write_text(json.dumps(source6c), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        compiler.run_sage6d_second_unknown_game_handoff(
            source_sage6c_path=source6c_path
        )


def test_sage6d_rejects_cross_source_game_drift(tmp_path):
    source6b = _load_default(compiler.DEFAULT_SAGE6B_M3_EXECUTION_PATH)
    source6b["config"]["game_id"] = "other-game"
    source6b["summary"]["game_id"] = "other-game"
    source6b_path = tmp_path / "invalid_sage6b.json"
    source6b_path.write_text(json.dumps(source6b), encoding="utf-8")

    with pytest.raises(ValueError, match="game identities must align"):
        compiler.run_sage6d_second_unknown_game_handoff(
            source_sage6b_path=source6b_path
        )


def test_sage6d_rejects_cross_source_context_drift(tmp_path):
    source6c = _load_default(compiler.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH)
    source6c["event_records"][0]["context_snapshot_hash"] = "drifted-context"
    source6c_path = tmp_path / "drifted_sage6c.json"
    source6c_path.write_text(json.dumps(source6c), encoding="utf-8")

    with pytest.raises(ValueError, match="contexts must align exactly"):
        compiler.run_sage6d_second_unknown_game_handoff(
            source_sage6c_path=source6c_path
        )


def test_sage6d_rejects_missing_common_source_suggested_control(tmp_path):
    source6b = _load_default(compiler.DEFAULT_SAGE6B_M3_EXECUTION_PATH)
    for request in source6b["selected_execution_requests"]:
        if request["request_id"].endswith("::0132"):
            request["suggested_control_actions"] = ["ACTION1"]
    source6b_path = tmp_path / "no_new_control.json"
    source6b_path.write_text(json.dumps(source6b), encoding="utf-8")

    with pytest.raises(ValueError, match="no source-suggested distinct control"):
        compiler.run_sage6d_second_unknown_game_handoff(
            source_sage6b_path=source6b_path
        )


def test_sage6d_control_selection_is_deterministic_and_explicit():
    assert (
        compiler.choose_distinct_control(
            suggested_controls=["ACTION1", "ACTION3", "ACTION4"],
            target_action="ACTION2",
            excluded_controls=["ACTION1"],
        )
        == "ACTION3"
    )
    with pytest.raises(ValueError, match="no source-suggested distinct control"):
        compiler.choose_distinct_control(
            suggested_controls=["ACTION1", "ACTION2"],
            target_action="ACTION2",
            excluded_controls=["ACTION1"],
        )


def test_sage6d_writer_and_package_exports(tmp_path):
    payload = compiler.run_sage6d_second_unknown_game_handoff()
    output_path = tmp_path / "nested" / "sage6d.json"

    compiler.write_sage6d_second_unknown_game_handoff(payload, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert sage.DEFAULT_SAGE6D_HANDOFF_PATH == compiler.DEFAULT_SAGE6D_HANDOFF_PATH
    assert (
        sage.run_sage6d_second_unknown_game_handoff
        is compiler.run_sage6d_second_unknown_game_handoff
    )
    assert (
        sage.write_sage6d_second_unknown_game_handoff
        is compiler.write_sage6d_second_unknown_game_handoff
    )


def _load_default(path):
    return json.loads(path.read_text(encoding="utf-8"))
