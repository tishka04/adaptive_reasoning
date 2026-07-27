from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage12 import mechanic_induction as v4
from theory.sage12 import target_mechanic_replication as v42
from theory.sage12.target_mechanic_recovery import (
    FAILURE_FORMAT_VERSION,
    RULE_ANCHOR_CONDITIONS,
    _build_runtime_failure,
    compact_qwen_prompt,
    compact_qwen_schema,
    load_frozen_manifest,
    public_rule,
    restore_public_rule,
    run_source_rehearsal,
)
from theory.sage12.target_mechanic_recovery_collection import run_collection


def _internal_rule(anchor: str, *, kind: str = "exact") -> v4.MechanicRule:
    value = "ACTION4" if kind == "exact" else "move"
    return v4.MechanicRule(
        rule_id=v4._rule_id(
            kind,
            value,
            anchor,
            "target_moved",
        ),
        action_scope_kind=kind,
        action_scope_value=value,
        anchor_condition=anchor,
        effect="target_moved",
    )


@pytest.mark.parametrize("kind", ["exact", "family"])
def test_public_any_rule_round_trips(kind):
    rule = _internal_rule("any", kind=kind)

    payload = public_rule(rule)
    restored = restore_public_rule(payload)

    assert payload["anchor_condition"] == "any"
    assert restored.to_dict() == rule.to_dict()


@pytest.mark.parametrize(
    ("internal", "public"),
    [
        ("occupied_object", "occupied"),
        ("empty", "free"),
        ("targetless", "none"),
    ],
)
def test_public_concrete_rule_round_trips(internal, public):
    rule = _internal_rule(internal)

    payload = public_rule(rule)

    assert payload["anchor_condition"] == public
    assert restore_public_rule(payload).to_dict() == rule.to_dict()


def test_every_structured_query_rule_round_trips_and_covers_any():
    query = v42.TargetMechanicQuery("ACTION4", "move", "occupied")
    rules = [
        rule
        for effect in v42.TARGET_EFFECT_LABELS
        for rule in v4._rules_for_query(query.as_internal(), effect)
    ]

    assert all(
        restore_public_rule(public_rule(rule)).to_dict() == rule.to_dict()
        for rule in rules
    )
    assert any(
        rule.action_scope_kind == "exact"
        and rule.anchor_condition == "any"
        for rule in rules
    )
    assert any(
        rule.action_scope_kind == "family"
        and rule.anchor_condition == "any"
        for rule in rules
    )


def test_qwen_contract_is_unchanged_and_concrete_only():
    window = v42.TargetMechanicWindowRecord.from_dict(
        json.loads(
            Path(
                "training/sage12/mechanic_induction_v4_2/"
                "source_train_windows.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
    )

    assert compact_qwen_schema() == v42.compact_qwen_schema()
    assert compact_qwen_prompt(window) == v42.compact_qwen_prompt(window)
    anchors = compact_qwen_schema()["properties"]["h"]["items"][
        "properties"
    ]["a"]["enum"]

    assert anchors == ["occupied", "free", "none"]
    assert "any" not in anchors
    assert "any" not in compact_qwen_prompt(window)


def test_frozen_manifest_separates_state_and_rule_anchors():
    manifest = load_frozen_manifest()

    assert manifest["manifest_checksum"] == (
        "81f14c655dc6b824970b2ecd8638ca62360abedcc7f4dcf3abed2b86cdd3a3c8"
    )
    assert manifest["collection"]["policy_seeds"] == [661, 709, 757, 809]
    assert manifest["window"]["state_anchor_vocabulary"] == [
        "occupied",
        "free",
        "none",
    ]
    assert tuple(
        manifest["window"]["rule_anchor_vocabulary"]
    ) == RULE_ANCHOR_CONDITIONS
    assert manifest["firewall"]["v4_2_shards_reused"] is False
    assert manifest["world_model_fit_authorized"] is False


def test_collection_fails_closed_without_source_gates(tmp_path):
    with pytest.raises(RuntimeError, match="source gates are missing"):
        run_collection(output_dir=tmp_path)


def test_full_source_rehearsal_serializes_all_predictions(tmp_path):
    result = run_source_rehearsal(output_dir=tmp_path)
    prediction_path = tmp_path / "source_rehearsal_predictions.jsonl"

    assert result["status"] == "PASS_SOURCE_REHEARSAL"
    assert result["source_windows"] == 1911
    assert result["prediction_rows"] == 1911
    assert result["round_trip_rate"] == 1.0
    assert result["exact_any_rules"] > 0
    assert result["family_any_rules"] > 0
    assert result["prediction_evidence_any_rules"] > 0
    assert all(result["checks"].values())
    assert len(prediction_path.read_text(encoding="utf-8").splitlines()) == 1911


def test_runtime_failure_is_automatic_and_revokes_authority(tmp_path):
    result = _build_runtime_failure(
        tmp_path,
        {"manifest_checksum": "manifest"},
        {"preflight_checksum": "preflight"},
        {"stage": "prediction_serialization"},
        KeyError("any"),
    )

    assert result["format_version"] == FAILURE_FORMAT_VERSION
    assert result["status"] == "FAIL_RUNTIME_CLOSED"
    assert result["v5_protocol_authorized"] is False
    assert result["world_model_fit_authorized"] is False
    assert result["ebm_fit_authorized"] is False
    assert (tmp_path / "runtime_failure.json").exists()


def test_published_v4_2_runtime_failure_is_unchanged():
    payload = json.loads(
        Path(
            "training/sage12/mechanic_induction_v4_2/runtime_failure.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["result_checksum"] == (
        "17934d7b576ac11c36abcac6235e7bc259247f225f49edf5e05126971390be6a"
    )
