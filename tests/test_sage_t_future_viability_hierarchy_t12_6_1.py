from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from theory.sage_t.causal import future_viability_hierarchy_experiment as experiment
from theory.sage_t.causal import future_viability_hierarchy_protocol as protocol_module
from theory.sage_t.causal.future_viability import FutureViabilityObservation
from theory.sage_t.causal.future_viability_hierarchy import (
    HierarchicalExtractionResult,
    HierarchicalFutureViabilityModel,
    HierarchicalViabilityObservation,
    crossfit_hierarchical_viability,
)
from theory.sage_t.causal.future_viability_hierarchy_cli import build_parser
from theory.sage_t.causal.future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    freeze_future_viability_hierarchy,
    load_future_viability_hierarchy_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "future_viability_t12_6_bp35"


def _diagnostic() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "future_viability_diagnostic_t12_6a_bp35"
    )


def _observations(
    *, evaluation: bool = False
) -> tuple[HierarchicalViabilityObservation, ...]:
    seeds = (9_201, 9_202, 9_203) if evaluation else (9_101, 9_102, 9_103)
    rows = []
    for seed in seeds:
        for lineage in (8_701, 8_705):
            for index in range(45):
                group = f"group-{seed}-{lineage}-{index}"
                rows.extend(
                    (
                        HierarchicalViabilityObservation(
                            base=FutureViabilityObservation(
                                group_id=group,
                                corpus="evaluation" if evaluation else "training",
                                search_seed=seed,
                                lineage_seed=lineage,
                                arm="local_archive_control",
                                source_exact_hash=f"hash-{group}",
                                action_key='ACTION6:{"x":0,"y":0}',
                                action_name="ACTION6",
                                coordinate_grounded=True,
                                local_signature=f"exact-goal-{seed}-{index}",
                                productive_reach=4,
                                immediate_score=1,
                                terminal=False,
                                changed=True,
                                novel=True,
                            ),
                            composition_signature="composition-goal",
                        ),
                        HierarchicalViabilityObservation(
                            base=FutureViabilityObservation(
                                group_id=group,
                                corpus="evaluation" if evaluation else "training",
                                search_seed=seed,
                                lineage_seed=lineage,
                                arm="local_archive_control",
                                source_exact_hash=f"hash-{group}",
                                action_key='ACTION6:{"x":9,"y":9}',
                                action_name="ACTION6",
                                coordinate_grounded=True,
                                local_signature=f"exact-bad-{seed}-{index}",
                                productive_reach=0,
                                immediate_score=7,
                                terminal=False,
                                changed=True,
                                novel=True,
                            ),
                            composition_signature="composition-bad",
                        ),
                    )
                )
    return tuple(rows)


def _extraction(*, evaluation: bool = False) -> HierarchicalExtractionResult:
    observations = _observations(evaluation=evaluation)
    return HierarchicalExtractionResult(
        observations=observations,
        metrics={
            "all_archive_conditions_present": True,
            "archive_condition_count": 18 if evaluation else 12,
            "duplicate_action_conflicts": 0,
            "expected_archive_condition_count": 18 if evaluation else 12,
            "label_variable_group_count": len(observations) // 2,
            "multi_action_group_count": len(observations) // 2,
            "observation_count": len(observations),
            "search_seeds": (
                [9_201, 9_202, 9_203]
                if evaluation
                else [9_101, 9_102, 9_103]
            ),
            "source_lineages": [8_701, 8_705],
            "total_archive_edges": len(observations),
        },
    )


def test_protocol_is_chronological_zero_sdk_and_rejects_physical_cli() -> None:
    protocol = FutureViabilityHierarchyProtocol()
    assert not set(protocol.training_search_seeds) & set(
        protocol.evaluation_search_seeds
    )
    assert protocol.maximum_sdk_calls == 0
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-compile-receipt",
            "compile.json",
            "--diagnostic-manifest",
            "diagnostic.json",
            "--diagnostic-receipt",
            "diagnostic-receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["compile"]).phase == "compile"
    assert parser.parse_args(["evaluate"]).phase == "evaluate"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_protocol_rejects_posthoc_gate_or_descriptor_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityHierarchyProtocol(minimum_compile_top1_accuracy=0.74)
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityHierarchyProtocol(local_radius=8)


def test_hierarchy_uses_composition_before_family_and_beats_controls() -> None:
    observations = _observations()
    training = tuple(
        item for item in observations if item.base.search_seed != 9_103
    )
    evaluation = tuple(
        item for item in observations if item.base.search_seed == 9_103
    )
    model = HierarchicalFutureViabilityModel.fit(
        training,
        target_field="productive_reach",
        radius=7,
        minimum_signature_support=2,
    )
    score, tier = model.score(evaluation[0])
    assert score == 4.0
    assert tier == "local_composition_signature"
    loaded = HierarchicalFutureViabilityModel.from_dict(model.to_dict())
    assert loaded.score(evaluation[0]) == model.score(evaluation[0])

    audit = crossfit_hierarchical_viability(
        observations,
        search_seeds=(9_101, 9_102, 9_103),
        radius=7,
        minimum_signature_support=2,
        binding_shift=1,
    )
    metrics = audit["micro_metrics"]
    assert metrics["future_binding_top1_accuracy"] == 1.0
    assert metrics["incumbent_binding_top1_accuracy"] == 0.0
    assert metrics["immediate_binding_top1_accuracy"] == 0.0
    assert metrics["binding_swap_top1_accuracy"] == 0.0
    assert metrics["hierarchy_coverage"] == 1.0
    assert metrics["unique_top_rate"] == 1.0


def test_freeze_binds_negative_parent_and_diagnostic_without_opening_evaluation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "e" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability_hierarchy(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_compile_receipt_path=(
            _parent() / "compile" / "compile_receipt.json"
        ),
        diagnostic_manifest_path=_diagnostic() / "manifest.json",
        diagnostic_receipt_path=(
            _diagnostic() / "diagnostic" / "diagnostic_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_hierarchy_manifest(
        manifest_path, root=_repo(), open_evaluation=False
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert len(loaded["inputs"]["training_archives"]) == 12
    assert len(loaded["inputs"]["evaluation_archives"]) == 18
    assert loaded["design"]["development_is_posthoc_to_t12_6"] is True
    assert loaded["firewall"]["compile_authorized"] is True
    assert loaded["firewall"]["evaluation_authorized"] is False
    assert loaded["firewall"]["environment_collection_authorized"] is False


def test_synthetic_compile_and_evaluation_pass_hierarchy_gate(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = FutureViabilityHierarchyProtocol()
    manifest = {
        "claim_boundary": {
            "authorized": "offline hierarchical viability",
            "not_authorized": ["physical control"],
        },
        "firewall": {"compile_authorized": True},
        "inputs": {"training_archives": [], "evaluation_archives": []},
        "manifest_checksum": "m" * 64,
        "parents": {
            "t12_6_compile_receipt": {"receipt_checksum": "p" * 64},
            "t12_6a_receipt": {"receipt_checksum": "d" * 64},
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    monkeypatch.setattr(
        experiment,
        "load_future_viability_hierarchy_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_extract",
        lambda manifest, *, protocol, corpus, root: _extraction(
            evaluation=corpus == "evaluation"
        ),
    )
    compile_dir = tmp_path / "compile"
    compile_receipt = experiment.compile_future_viability_hierarchy(
        manifest_path="unused.json",
        output_dir=compile_dir,
        root=_repo(),
    )
    assert compile_receipt["passed"] is True
    assert compile_receipt["status"] == "PASS_T12_6_1_COMPILE_GATE"

    evaluation_dir = tmp_path / "evaluation"
    evaluation_receipt = experiment.evaluate_future_viability_hierarchy(
        manifest_path="unused.json",
        compile_receipt_path=compile_dir / "compile_receipt.json",
        output_dir=evaluation_dir,
        root=_repo(),
    )
    assert evaluation_receipt["passed"] is True
    assert evaluation_receipt["status"] == (
        "PASS_T12_6_1_HIERARCHICAL_VIABILITY_GATE"
    )
    assert evaluation_receipt["metrics"]["t12_6_2_freeze_authorized"] is True
