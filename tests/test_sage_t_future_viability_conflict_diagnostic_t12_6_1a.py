from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from theory.sage_t.causal import (
    future_viability_conflict_diagnostic_protocol as protocol_module,
)
from theory.sage_t.causal.future_viability_conflict_diagnostic import (
    CONSOLIDATION_POLICIES,
    extract_conflict_sensitivities,
)
from theory.sage_t.causal.future_viability_conflict_diagnostic_cli import (
    build_parser,
)
from theory.sage_t.causal.future_viability_conflict_diagnostic_protocol import (
    FutureViabilityConflictDiagnosticProtocol,
    freeze_future_viability_conflict_diagnostic,
    load_future_viability_conflict_diagnostic_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "future_viability_hierarchy_t12_6_1_bp35"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state() -> dict[str, object]:
    return {
        "entities": [],
        "true_facts": [],
        "false_facts": [],
        "counters": [],
        "registers": [],
        "topology": [],
        "regime_index": 0,
    }


def _edge(
    edge_id: str,
    *,
    source: str,
    action: str,
    target: str,
    terminal: bool,
    novel: bool,
) -> dict[str, object]:
    return {
        "action": {"action_data": {}, "action_name": action},
        "changed": True,
        "edge_id": edge_id,
        "novel": novel,
        "source_cell_id": source,
        "source_exact_hash": f"hash-{source}",
        "target_cell_id": target,
        "target_exact_hash": f"hash-{target}",
        "terminal": terminal,
    }


def test_protocol_is_posthoc_only_and_cli_has_no_evaluation_or_collection() -> None:
    protocol = FutureViabilityConflictDiagnosticProtocol()
    assert protocol.consolidation_policies == CONSOLIDATION_POLICIES
    assert protocol.confirmatory_claim_authorized is False
    assert protocol.same_archive_reconfirmation_authorized is False
    assert protocol.maximum_sdk_calls == 0
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-evaluation-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["diagnose"]).phase == "diagnose"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])


def test_protocol_rejects_policy_or_expected_count_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityConflictDiagnosticProtocol(expected_parent_conflicts=36)
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityConflictDiagnosticProtocol(
            consolidation_policies=("parent_order",)
        )


def test_synthetic_conflict_audit_separates_future_and_immediate_labels(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "archive.json"
    payload = {
        "cells": [
            {"cell_id": cell_id, "state": _state()}
            for cell_id in ("source", "future", "terminal", "leaf")
        ],
        "edges": [
            _edge(
                "edge-first",
                source="source",
                action="ACTION3",
                target="future",
                terminal=False,
                novel=True,
            ),
            _edge(
                "edge-conflict",
                source="source",
                action="ACTION3",
                target="terminal",
                terminal=True,
                novel=False,
            ),
            _edge(
                "edge-other",
                source="source",
                action="ACTION4",
                target="leaf",
                terminal=False,
                novel=False,
            ),
            _edge(
                "edge-reach",
                source="future",
                action="ACTION3",
                target="leaf",
                terminal=False,
                novel=True,
            ),
        ],
    }
    archive_path.write_text(json.dumps(payload), encoding="utf-8")
    extracted = extract_conflict_sensitivities(
        archive_metas=(
            {
                "arm": "arm",
                "lineage_seed": 2,
                "path": str(archive_path),
                "search_seed": 1,
                "sha256": _sha(archive_path),
            },
        ),
        root=tmp_path,
        corpus="evaluation",
        expected_search_seeds=(1,),
        expected_lineages=(2,),
        expected_arms=("arm",),
        future_horizon=4,
        local_radius=7,
    )
    assert extracted.metrics["parent_duplicate_action_conflicts"] == 1
    assert extracted.metrics["future_label_conflicts"] == 1
    assert extracted.metrics["immediate_label_conflicts"] == 1
    assert extracted.metrics["conflict_difference_pattern_counts"] == {
        "terminal+novel+target_cell_id": 1
    }
    parent = {
        item.base.action_name: item.base.productive_reach
        for item in extracted.observations_by_policy["parent_order"]
    }
    last = {
        item.base.action_name: item.base.productive_reach
        for item in extracted.observations_by_policy["archive_last"]
    }
    assert parent["ACTION3"] == 1
    assert last["ACTION3"] == 0
    assert not extracted.observations_by_policy["drop_conflicted_groups"]


def test_freeze_binds_failed_parent_and_keeps_all_authority_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "f" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability_conflict_diagnostic(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_evaluation_receipt_path=(
            _parent() / "evaluation" / "evaluation_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_conflict_diagnostic_manifest(
        manifest_path, root=_repo()
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert len(loaded["inputs"]["evaluation_archives"]) == 18
    assert loaded["design"]["axes_are_posthoc_to_conflict_inspection"] is True
    assert loaded["scientific_claims_authorized"] is False
    assert loaded["firewall"]["diagnostic_authorized"] is True
    assert loaded["firewall"]["environment_collection_authorized"] is False
    assert loaded["firewall"]["t12_6_2_freeze_authorized"] is False
