from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage_t.causal import future_viability_diagnostic_protocol as protocol_module
from theory.sage_t.causal.future_viability import FutureViabilityObservation
from theory.sage_t.causal.future_viability_diagnostic import (
    diagnose_future_viability_fold,
)
from theory.sage_t.causal.future_viability_diagnostic_cli import build_parser
from theory.sage_t.causal.future_viability_diagnostic_protocol import (
    FutureViabilityDiagnosticProtocol,
    freeze_future_viability_diagnostic,
    load_future_viability_diagnostic_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "future_viability_t12_6_bp35"


def _row(
    *,
    group: str,
    seed: int,
    lineage: int,
    action: str,
    signature: str,
    reach: int,
) -> FutureViabilityObservation:
    return FutureViabilityObservation(
        group_id=group,
        corpus="training",
        search_seed=seed,
        lineage_seed=lineage,
        arm="local_archive_control",
        source_exact_hash=f"hash-{group}",
        action_key=f"{action}:{{}}",
        action_name=action,
        coordinate_grounded=False,
        local_signature=signature,
        productive_reach=reach,
        immediate_score=4 + 2 * int(reach > 0),
        terminal=False,
        changed=reach > 0,
        novel=False,
    )


def _synthetic_observations() -> tuple[FutureViabilityObservation, ...]:
    rows = []
    for seed in (9_101, 9_102):
        for lineage in (8_701, 8_705):
            rows.extend(
                (
                    _row(
                        group=f"train-correct-{seed}-{lineage}",
                        seed=seed,
                        lineage=lineage,
                        action="ACTION3",
                        signature="sig-goal",
                        reach=4,
                    ),
                    _row(
                        group=f"train-correct-{seed}-{lineage}",
                        seed=seed,
                        lineage=lineage,
                        action="ACTION4",
                        signature="sig-bad",
                        reach=0,
                    ),
                    _row(
                        group=f"train-flip-{seed}-{lineage}",
                        seed=seed,
                        lineage=lineage,
                        action="ACTION3",
                        signature="sig-flip-goal",
                        reach=0,
                    ),
                    _row(
                        group=f"train-flip-{seed}-{lineage}",
                        seed=seed,
                        lineage=lineage,
                        action="ACTION4",
                        signature="sig-flip-bad",
                        reach=4,
                    ),
                )
            )
    for index in range(43):
        flipped = index >= 30
        rows.extend(
            (
                _row(
                    group=f"focal-{index}",
                    seed=9_103,
                    lineage=8_701,
                    action="ACTION3",
                    signature="sig-flip-goal" if flipped else "sig-goal",
                    reach=4,
                ),
                _row(
                    group=f"focal-{index}",
                    seed=9_103,
                    lineage=8_701,
                    action="ACTION4",
                    signature="sig-flip-bad" if flipped else "sig-bad",
                    reach=0,
                ),
            )
        )
    return tuple(rows)


def test_protocol_is_posthoc_zero_sdk_and_has_no_evaluation_phase() -> None:
    protocol = FutureViabilityDiagnosticProtocol()
    assert protocol.maximum_sdk_calls == 0
    assert protocol.evaluation_archive_payloads_authorized is False
    assert protocol.confirmatory_claim_authorized is False
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "manifest.json",
            "--parent-compile-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["diagnose"]).phase == "diagnose"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_protocol_rejects_posthoc_axis_or_locus_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityDiagnosticProtocol(focal_lineage_seed=8_705)
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityDiagnosticProtocol(
            diagnostic_axes=("support_tier", "new_axis")
        )


def test_diagnostic_reproduces_focal_miss_and_classifies_stable_misranking() -> None:
    diagnostic = diagnose_future_viability_fold(
        _synthetic_observations(),
        holdout_search_seed=9_103,
        focal_lineage_seed=8_701,
        reference_lineage_seed=8_705,
        radius=7,
        minimum_signature_support=2,
        binding_shift=1,
    )

    assert diagnostic["focal_metrics"]["eligible_groups"] == 43
    assert diagnostic["focal_metrics"]["hits"] == 30
    assert diagnostic["error_summary"]["errors"] == 13
    assert diagnostic["error_summary"]["error_mechanism_counts"] == {
        "stable_exact_signature_misranking": 13
    }
    assert diagnostic["classification"] == (
        "POSTHOC_STABLE_EXACT_SIGNATURE_MISRANKING_DOMINANT"
    )
    assert diagnostic["counterfactual_sensitivities"]["same_lineage_only"][
        "eligible_groups"
    ] == 43


def test_freeze_binds_only_parent_training_archives_and_negative_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability_diagnostic(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_compile_receipt_path=(
            _parent() / "compile" / "compile_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_diagnostic_manifest(
        manifest_path, root=_repo()
    )

    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["compile_receipt"]["status"] == (
        "FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE"
    )
    assert set(loaded["inputs"]) == {"training_archives"}
    assert len(loaded["inputs"]["training_archives"]) == 12
    assert loaded["design"]["evaluation_archive_payloads_excluded"] is True
    assert loaded["firewall"]["evaluation_authorized"] is False
    assert loaded["firewall"]["environment_collection_authorized"] is False
