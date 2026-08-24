from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage_t.causal import (
    future_viability_seed_shift_diagnostic_protocol as protocol_module,
)
from theory.sage_t.causal.future_viability import FutureViabilityObservation
from theory.sage_t.causal.future_viability_hierarchy import (
    HierarchicalFutureViabilityModel,
    HierarchicalViabilityObservation,
)
from theory.sage_t.causal.future_viability_seed_shift_diagnostic import (
    SEED_SHIFT_DIAGNOSTIC_AXES,
    diagnose_future_viability_seed_shift,
)
from theory.sage_t.causal.future_viability_seed_shift_diagnostic_cli import (
    build_parser,
)
from theory.sage_t.causal.future_viability_seed_shift_diagnostic_protocol import (
    FutureViabilitySeedShiftDiagnosticProtocol,
    freeze_future_viability_seed_shift_diagnostic,
    load_future_viability_seed_shift_diagnostic_manifest,
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


def _conflict() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "future_viability_conflict_diagnostic_t12_6_1a_bp35"
    )


def _item(
    *,
    seed: int,
    group: str,
    action: str,
    reach: int,
    composition: str,
    training: bool,
) -> HierarchicalViabilityObservation:
    return HierarchicalViabilityObservation(
        base=FutureViabilityObservation(
            group_id=group,
            corpus="training" if training else "evaluation",
            search_seed=seed,
            lineage_seed=8_701,
            arm="arm",
            source_exact_hash=f"hash-{group}",
            action_key=f'{action}:{{"x":0,"y":0}}',
            action_name=action,
            coordinate_grounded=True,
            local_signature=f"exact-{seed}-{group}-{action}",
            productive_reach=reach,
            immediate_score=0,
            terminal=False,
            changed=True,
            novel=True,
        ),
        composition_signature=composition,
    )


def _synthetic() -> tuple[
    tuple[HierarchicalViabilityObservation, ...],
    tuple[HierarchicalViabilityObservation, ...],
]:
    training = []
    for seed in (1, 2, 3):
        for index in range(2):
            group = f"train-{seed}-{index}"
            training.extend(
                (
                    _item(
                        seed=seed,
                        group=group,
                        action="ACTION3",
                        reach=4,
                        composition="composition-good",
                        training=True,
                    ),
                    _item(
                        seed=seed,
                        group=group,
                        action="ACTION4",
                        reach=0,
                        composition="composition-bad",
                        training=True,
                    ),
                )
            )
    evaluation = []
    for seed in (10, 20, 30):
        for index in range(2):
            group = f"eval-{seed}-{index}"
            reversed_labels = seed == 20
            evaluation.extend(
                (
                    _item(
                        seed=seed,
                        group=group,
                        action="ACTION3",
                        reach=0 if reversed_labels else 4,
                        composition="composition-good",
                        training=False,
                    ),
                    _item(
                        seed=seed,
                        group=group,
                        action="ACTION4",
                        reach=4 if reversed_labels else 0,
                        composition="composition-bad",
                        training=False,
                    ),
                )
            )
    return tuple(training), tuple(evaluation)


def test_protocol_is_posthoc_only_and_cli_has_no_collection_or_evaluate() -> None:
    protocol = FutureViabilitySeedShiftDiagnosticProtocol()
    assert protocol.diagnostic_axes == SEED_SHIFT_DIAGNOSTIC_AXES
    assert protocol.focal_search_seed == 9_202
    assert protocol.confirmatory_claim_authorized is False
    assert protocol.model_or_descriptor_change_authorized is False
    assert protocol.maximum_sdk_calls == 0
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--hierarchy-manifest",
            "hierarchy.json",
            "--hierarchy-evaluation-receipt",
            "evaluation.json",
            "--conflict-manifest",
            "conflict.json",
            "--conflict-diagnostic-receipt",
            "diagnostic.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["diagnose"]).phase == "diagnose"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])


def test_protocol_rejects_focal_seed_or_axis_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilitySeedShiftDiagnosticProtocol(focal_search_seed=9_201)
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilitySeedShiftDiagnosticProtocol(
            diagnostic_axes=("support_tier_attribution",)
        )


def test_synthetic_diagnostic_identifies_stable_composition_reversal() -> None:
    training, evaluation = _synthetic()
    model = HierarchicalFutureViabilityModel.fit(
        training,
        target_field="productive_reach",
        radius=7,
        minimum_signature_support=2,
    )
    diagnostic = diagnose_future_viability_seed_shift(
        training,
        evaluation,
        future_model=model,
        focal_search_seed=20,
        reference_search_seeds=(10, 30),
        training_search_seeds=(1, 2, 3),
        radius=7,
        minimum_signature_support=2,
    )
    assert diagnostic["focal_summary"]["eligible_groups"] == 2
    assert diagnostic["focal_summary"]["hits"] == 0
    assert diagnostic["reference_contrast"]["reference_summary"]["accuracy"] == 1.0
    assert diagnostic["classification"] == (
        "POSTHOC_20_STABLE_COMPOSITION_REVERSAL_DOMINANT"
    )
    assert diagnostic["focal_summary"]["error_mechanism_counts"] == {
        "stable_composition_reversal": 2
    }
    assert all(
        value["accuracy"] == 0.0
        for value in diagnostic["leave_one_training_seed_out"].values()
    )


def test_freeze_binds_both_parents_and_keeps_authority_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "a" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability_seed_shift_diagnostic(
        output_path=manifest_path,
        hierarchy_manifest_path=_hierarchy() / "manifest.json",
        hierarchy_evaluation_receipt_path=(
            _hierarchy() / "evaluation" / "evaluation_receipt.json"
        ),
        conflict_manifest_path=_conflict() / "manifest.json",
        conflict_diagnostic_receipt_path=(
            _conflict() / "diagnostic" / "diagnostic_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_seed_shift_diagnostic_manifest(
        manifest_path, root=_repo()
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert len(loaded["inputs"]["training_archives"]) == 12
    assert len(loaded["inputs"]["evaluation_archives"]) == 18
    assert loaded["design"][
        "axes_are_frozen_before_individual_9202_error_inspection"
    ] is True
    assert loaded["scientific_claims_authorized"] is False
    assert loaded["firewall"]["diagnostic_authorized"] is True
    assert loaded["firewall"]["environment_collection_authorized"] is False
    assert loaded["firewall"]["future_protocol_freeze_authorized"] is False
