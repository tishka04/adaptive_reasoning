from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from theory.sage_t import t10_2_8_protocol as protocol
from theory.sage_t import t10_2_8_runtime as runtime


def _qa_gate() -> dict[str, object]:
    return {
        "minimum_confident_correspondence": 0.9,
        "maximum_fully_ambiguous_correspondence": 0.1,
        "minimum_predicate_prevalence": 0.1,
        "maximum_predicate_prevalence": 0.9,
        "minimum_predicate_support": 1,
        "minimum_predicate_games": 1,
        "minimum_evaluable_nonterminal_prefix_fraction": 0.8,
        "minimum_multiframe_coherent_prefix_fraction": 0.5,
    }


def _manifest() -> dict[str, Any]:
    return {
        "manifest_checksum": "a" * 64,
        "qa_gate": _qa_gate(),
        "handoff_receipt": {
            "receipt_checksum": "b" * 64,
            "scientific_environment_sha256": "c" * 64,
            "predecessor_collection": {
                "accepted_event_count": 2,
                "replayed_physical_actions": 0,
            },
        },
    }


def _qa_event(index: int, *, predicate: bool, coherent: bool = True) -> dict[str, Any]:
    complete = {"complete": True}
    projections = {"one": complete, "two": complete if coherent else {"complete": False}}
    return {
        "event_id": f"event-{index}",
        "game_id": "bp35-0a0ad940",
        "correspondence": {
            "fraction_denominator": 10,
            "confident_matches": 10,
            "fully_ambiguous_matches": 0,
        },
        "transport_certificates": [
            {
                "exact": True,
                "live_graph_exact_attested": True,
                "summary_commutative_exact": True,
                "certifies_gauge_equivalence": True,
                "round_trip_exact": True,
                "commutativity": {"exact": True},
            }
        ],
        "transport": {
            "entity_permutation_invariant": True,
            "summary_commutative_exact": True,
        },
        "learned_predicates": ["effect"],
        "labels": {"effect": predicate, "level_complete": False},
        "prefix": {"nonterminal": True, "evaluable": True},
        "projections": projections,
        "selection": {
            "controller": "learned" if index else "capacity_matched_independent",
            "decision_engine_used": True,
            "option_conditioned": bool(index),
        },
        "outcome": {"progression": 0.0, "goal": False, "terminal": False},
    }


def test_recovery_seed_binding_extends_and_restores_split_registry() -> None:
    original = runtime._science.CONFIRMATION_SEEDS

    with runtime._recovery_seed_binding([3_119_945]):
        assert 3_119_945 in runtime._science.CONFIRMATION_SEEDS
        assert runtime._science._expected_source_split(3_119_945) == (
            "leave_one_game_out_confirmation"
        )

    assert runtime._science.CONFIRMATION_SEEDS == original


def test_qa_report_passes_balanced_predicate_and_coherent_frames() -> None:
    report = runtime.build_qa_report(
        manifest=_manifest(),
        events=[_qa_event(0, predicate=False), _qa_event(1, predicate=True)],
    )

    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["metrics"]["predicates"]["effect"]["prevalence"] == 0.5
    assert report["metrics"]["multiframe_coherent_prefix_fraction"] == 1.0
    assert report["behavior_diagnostics"]["diagnosis"] == "ZERO_OBSERVED_PROGRESS"
    assert report["fit_authorized"] is False


def test_qa_report_fails_universal_predicate_and_missing_multiframe() -> None:
    events = [
        _qa_event(0, predicate=True, coherent=False),
        _qa_event(1, predicate=True, coherent=False),
    ]

    report = runtime.build_qa_report(manifest=_manifest(), events=events)

    assert report["passed"] is False
    assert "learned_predicate_prevalence_and_support" in report["failed_checks"]
    assert "multiframe_coherent_prefixes" in report["failed_checks"]
    assert report["metrics"]["predicates"]["effect"]["prevalence"] == 1.0
    assert report["metrics"]["multiframe_coherent_prefix_fraction"] == 0.0


def test_lineage_audit_splits_parent_and_recovery_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_lane = {
        "split": "discovery",
        "game_id": "bp35-0a0ad940",
        "seed": 101,
        "lane_id": "parent",
    }
    recovery_lane = {
        "split": "leave_one_game_out_confirmation",
        "game_id": "su15-4c352900",
        "seed": 3_119_945,
        "lane_id": "recovery",
    }
    manifest = _manifest()
    manifest["handoff_receipt"]["lineage_registry"] = [
        {
            "lineage": "t10_2_2_parent",
            "physical_lane": parent_lane,
            "logical_lane": parent_lane,
            "provenance_manifest_checksum": protocol.PARENT_KERNEL_MANIFEST_CHECKSUM,
            "expected_event_count": 1,
        },
        {
            "lineage": "t10_2_7_recovery",
            "physical_lane": recovery_lane,
            "logical_lane": {**recovery_lane, "seed": 111, "lane_id": "orphan"},
            "provenance_manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM,
            "expected_event_count": 1,
        },
    ]
    manifest["handoff_receipt"]["lineage_registry_sha256"] = runtime.canonical_sha256(
        manifest["handoff_receipt"]["lineage_registry"]
    )
    events = [
        {
            "event_id": "parent-event",
            **{key: parent_lane[key] for key in ("split", "game_id", "seed")},
            "provenance": {
                "manifest_checksum": protocol.PARENT_KERNEL_MANIFEST_CHECKSUM,
                "environment_sha256": "c" * 64,
            },
        },
        {
            "event_id": "recovery-event",
            **{key: recovery_lane[key] for key in ("split", "game_id", "seed")},
            "provenance": {
                "manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM,
                "environment_sha256": "c" * 64,
            },
        },
    ]
    monkeypatch.setattr(runtime._science, "validate_source_events", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime._predecessor_runtime,
        "build_execution_manifest",
        lambda **_: {"manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM},
    )

    audit = runtime.build_lineage_audit(
        manifest=manifest,
        predecessor={"manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM},
        kernel={"manifest_checksum": protocol.PARENT_KERNEL_MANIFEST_CHECKSUM},
        events=events,
    )

    assert audit["passed"] is True
    assert audit["lineage_event_counts"] == {
        "t10_2_2_parent": 1,
        "t10_2_7_recovery": 1,
    }


def test_lineage_audit_fails_an_unregistered_physical_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    manifest["handoff_receipt"]["lineage_registry"] = []
    manifest["handoff_receipt"]["lineage_registry_sha256"] = runtime.canonical_sha256([])
    monkeypatch.setattr(runtime._science, "validate_source_events", lambda *a, **k: None)

    audit = runtime.build_lineage_audit(
        manifest=manifest,
        predecessor={},
        kernel={},
        events=[{"event_id": "unknown", "split": "discovery", "game_id": "x", "seed": 1}],
    )

    assert audit["passed"] is False
    assert audit["checks"]["all_events_registered"] is False
    assert audit["firewall"]["model_fit_opened"] is False


def test_compile_cli_returns_three_on_registered_qa_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runtime,
        "compile_phase",
        lambda **_: {"status": "FAIL_T10_2_8_QA_STOP_BEFORE_FIT"},
    )

    code = runtime.main(["compile"])

    assert code == 3
    assert "FAIL_T10_2_8_QA_STOP_BEFORE_FIT" in capsys.readouterr().out


def test_write_once_terminal_artifact_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    first = runtime.signed_payload({"value": 1}, checksum_key="terminal_checksum")
    second = runtime.signed_payload({"value": 2}, checksum_key="terminal_checksum")

    runtime._write_once_payload(path, first, checksum_key="terminal_checksum")
    runtime._write_once_payload(path, first, checksum_key="terminal_checksum")
    with pytest.raises(runtime.JournalIntegrityError, match="immutable"):
        runtime._write_once_payload(path, second, checksum_key="terminal_checksum")
