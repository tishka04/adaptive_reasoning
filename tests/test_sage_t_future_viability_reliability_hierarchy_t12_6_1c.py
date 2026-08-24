from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from theory.sage_t.causal import (
    future_viability_reliability_hierarchy_experiment as experiment,
)
from theory.sage_t.causal import (
    future_viability_reliability_hierarchy_protocol as protocol_module,
)
from theory.sage_t.causal.future_viability import FutureViabilityObservation
from theory.sage_t.causal.future_viability_hierarchy import (
    HierarchicalExtractionResult,
    HierarchicalViabilityObservation,
)
from theory.sage_t.causal.future_viability_reliability_hierarchy import (
    ReliabilityGatedFutureViabilityModel,
    evaluate_reliability_candidates,
)
from theory.sage_t.causal.future_viability_reliability_hierarchy_cli import (
    build_parser,
)
from theory.sage_t.causal.future_viability_reliability_hierarchy_protocol import (
    FutureViabilityReliabilityProtocol,
    freeze_future_viability_reliability_hierarchy,
    load_future_viability_reliability_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _hierarchy() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "future_viability_hierarchy_t12_6_1_bp35"
    )


def _seed_shift() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "future_viability_seed_shift_diagnostic_t12_6_1b_bp35"
    )


def _row(
    *,
    seed: int,
    lineage: int,
    index: int,
    action_key: str,
    action_name: str,
    exact: str,
    composition: str,
    productive: int,
    immediate: int,
) -> HierarchicalViabilityObservation:
    group = f"group-{seed}-{lineage}-{index}"
    return HierarchicalViabilityObservation(
        base=FutureViabilityObservation(
            group_id=group,
            corpus="training",
            search_seed=seed,
            lineage_seed=lineage,
            arm="local_archive_control",
            source_exact_hash=f"hash-{group}",
            action_key=action_key,
            action_name=action_name,
            coordinate_grounded=True,
            local_signature=exact,
            productive_reach=productive,
            immediate_score=immediate,
            terminal=False,
            changed=True,
            novel=True,
        ),
        composition_signature=composition,
    )


def _observations() -> tuple[HierarchicalViabilityObservation, ...]:
    rows = []
    distractor_labels = {9_101: 0, 9_102: 1, 9_103: 2}
    for seed in (9_101, 9_102, 9_103):
        for lineage in (8_701, 8_705):
            for index in range(45):
                rows.extend(
                    (
                        _row(
                            seed=seed,
                            lineage=lineage,
                            index=index,
                            action_key='ACTION4:{"x":4,"y":4}',
                            action_name="ACTION4",
                            exact="exact-unstable",
                            composition="composition-distractor",
                            productive=distractor_labels[seed],
                            immediate=2,
                        ),
                        _row(
                            seed=seed,
                            lineage=lineage,
                            index=index,
                            action_key='ACTION6:{"x":0,"y":0}',
                            action_name="ACTION6",
                            exact="exact-goal",
                            composition="composition-goal",
                            productive=4,
                            immediate=1,
                        ),
                        _row(
                            seed=seed,
                            lineage=lineage,
                            index=index,
                            action_key='ACTION6:{"x":9,"y":9}',
                            action_name="ACTION6",
                            exact="exact-bad",
                            composition="composition-bad",
                            productive=0,
                            immediate=7,
                        ),
                    )
                )
    return tuple(rows)


def _extraction() -> HierarchicalExtractionResult:
    observations = _observations()
    return HierarchicalExtractionResult(
        observations=observations,
        metrics={
            "all_archive_conditions_present": True,
            "archive_condition_count": 12,
            "duplicate_action_conflicts": 0,
            "expected_archive_condition_count": 12,
            "label_variable_group_count": 270,
            "multi_action_group_count": 270,
            "observation_count": len(observations),
            "search_seeds": [9_101, 9_102, 9_103],
            "source_lineages": [8_701, 8_705],
            "total_archive_edges": len(observations),
        },
    )


def test_protocol_is_source_train_only_and_cli_has_no_evaluation() -> None:
    protocol = FutureViabilityReliabilityProtocol()
    assert protocol.evaluation_archive_payloads_authorized is False
    assert protocol.physical_collection_authorized is False
    assert protocol.maximum_sdk_calls == 0
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--hierarchy-manifest",
            "hierarchy.json",
            "--hierarchy-compile-receipt",
            "compile.json",
            "--seed-shift-diagnostic-receipt",
            "diagnostic.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["compile"]).phase == "compile"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])


def test_protocol_rejects_threshold_retuning() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityReliabilityProtocol(
            minimum_compile_top1_accuracy=0.74
        )
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityReliabilityProtocol(
            reliability_candidates=("exact_span2_range2",)
        )


def test_model_rejects_heterogeneous_exact_and_retains_stable_exact() -> None:
    observations = _observations()
    training = tuple(
        row for row in observations if row.base.search_seed != 9_103
    )
    evaluation = tuple(
        row for row in observations if row.base.search_seed == 9_103
    )
    model = ReliabilityGatedFutureViabilityModel.fit(
        training,
        target_field="productive_reach",
        radius=7,
        minimum_signature_support=2,
        minimum_exact_seed_span=2,
        maximum_exact_label_range=0,
    )
    distractor = next(
        row for row in evaluation if row.base.local_signature == "exact-unstable"
    )
    value, tier, audit = model.score_with_audit(distractor)
    assert tier == "local_composition_signature"
    assert audit["exact_candidate_present"] is True
    assert audit["exact_candidate_reliable"] is False
    assert "excessive_label_range" in audit["exact_rejection_reasons"]
    goal = next(row for row in evaluation if row.base.local_signature == "exact-goal")
    assert model.score(goal) == (4.0, "reliable_exact_local_signature")
    loaded = ReliabilityGatedFutureViabilityModel.from_dict(model.to_dict())
    assert loaded.score_with_audit(distractor) == model.score_with_audit(distractor)


def test_candidate_selection_is_training_only_and_prefers_strict_tie() -> None:
    result = evaluate_reliability_candidates(
        _observations(),
        search_seeds=(9_101, 9_102, 9_103),
        radius=7,
        minimum_signature_support=2,
        candidate_names=(
            "exact_span2_range0",
            "exact_span2_range1",
            "exact_span2_range2",
        ),
        binding_shift=1,
    )
    assert result["selected_candidate"] == "exact_span2_range0"
    selected = result["candidate_results"]["exact_span2_range0"]
    assert selected["micro_metrics"]["future_binding_top1_accuracy"] == 1.0
    assert selected["micro_metrics"]["future_gain_over_incumbent"] == 0.0
    assert selected["micro_metrics"]["exact_rejection_exercised_rate"] == 1.0


def test_freeze_imports_only_training_archives(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "f" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability_reliability_hierarchy(
        output_path=manifest_path,
        hierarchy_manifest_path=_hierarchy() / "manifest.json",
        hierarchy_compile_receipt_path=(
            _hierarchy() / "compile" / "compile_receipt.json"
        ),
        seed_shift_diagnostic_receipt_path=(
            _seed_shift() / "diagnostic" / "diagnostic_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_reliability_manifest(
        manifest_path, root=_repo()
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert set(loaded["inputs"]) == {"training_archives"}
    assert len(loaded["inputs"]["training_archives"]) == 12
    assert loaded["design"]["evaluation_archive_registry_imported"] is False
    assert loaded["firewall"]["evaluation_authorized"] is False
    assert loaded["firewall"]["environment_collection_authorized"] is False


def test_synthetic_compile_passes_but_authorizes_only_new_protocol(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = FutureViabilityReliabilityProtocol()
    manifest = {
        "claim_boundary": {
            "authorized": "source-train reliability development",
            "not_authorized": ["confirmation", "physical collection"],
        },
        "firewall": {"compile_authorized": True},
        "inputs": {"training_archives": []},
        "manifest_checksum": "m" * 64,
        "parents": {
            "hierarchy_compile_receipt": {"receipt_checksum": "h" * 64},
            "seed_shift_diagnostic_receipt": {"receipt_checksum": "s" * 64},
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    monkeypatch.setattr(
        experiment,
        "load_future_viability_reliability_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_extract",
        lambda manifest, *, protocol, root: _extraction(),
    )
    output = tmp_path / "compile"
    receipt = experiment.compile_future_viability_reliability_hierarchy(
        manifest_path="unused.json",
        output_dir=output,
        root=_repo(),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_6_1C_SOURCE_TRAIN_COMPILE_GATE"
    assert receipt["metrics"]["confirmatory_claim_authorized"] is False
    assert receipt["metrics"]["physical_collection_authorized"] is False
    assert receipt["metrics"]["new_archive_protocol_freeze_authorized"] is True
    assert receipt["metrics"]["evaluation_archive_payloads_loaded"] == 0
