from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from theory.sage_t.causal import future_viability_experiment as experiment
from theory.sage_t.causal import future_viability_protocol as protocol_module
from theory.sage_t.causal.future_viability import (
    ExtractionResult,
    FutureViabilityModel,
    FutureViabilityObservation,
    crossfit_future_viability,
    evaluate_future_viability_ranking,
)
from theory.sage_t.causal.future_viability_cli import build_parser
from theory.sage_t.causal.future_viability_protocol import (
    FutureViabilityProtocol,
    freeze_future_viability,
    load_future_viability_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _authority() -> Path:
    return _repo() / "training" / "sage_t" / "goal_cursor_control_t12_5c_bp35"


def _training_parent() -> Path:
    return _repo() / "training" / "sage_t" / "target_regrounding_t12_4a_4d_bp35"


def _evaluation_parent() -> Path:
    return _repo() / "training" / "sage_t" / "hazard_diversity_t12_4a_4d_1_bp35"


def _observations(*, evaluation: bool = False) -> tuple[FutureViabilityObservation, ...]:
    seeds = (9_201, 9_202, 9_203) if evaluation else (9_101, 9_102, 9_103)
    output = []
    for seed in seeds:
        for lineage in (8_701, 8_705):
            for index in range(45):
                group = f"g-{seed}-{lineage}-{index}"
                output.extend(
                    (
                        FutureViabilityObservation(
                            group_id=group,
                            corpus="evaluation" if evaluation else "training",
                            search_seed=seed,
                            lineage_seed=lineage,
                            arm="local_archive_control",
                            source_exact_hash=f"h-{group}",
                            action_key="ACTION3:{}",
                            action_name="ACTION3",
                            coordinate_grounded=False,
                            local_signature="sig_goal",
                            productive_reach=4,
                            immediate_score=1,
                            terminal=False,
                            changed=True,
                            novel=True,
                        ),
                        FutureViabilityObservation(
                            group_id=group,
                            corpus="evaluation" if evaluation else "training",
                            search_seed=seed,
                            lineage_seed=lineage,
                            arm="local_archive_control",
                            source_exact_hash=f"h-{group}",
                            action_key="ACTION4:{}",
                            action_name="ACTION4",
                            coordinate_grounded=False,
                            local_signature="sig_immediate",
                            productive_reach=0,
                            immediate_score=7,
                            terminal=False,
                            changed=True,
                            novel=True,
                        ),
                    )
                )
    return tuple(output)


def _extraction(*, evaluation: bool = False) -> ExtractionResult:
    observations = _observations(evaluation=evaluation)
    seeds = [9_201, 9_202, 9_203] if evaluation else [9_101, 9_102, 9_103]
    return ExtractionResult(
        observations,
        {
            "all_archive_conditions_present": True,
            "archive_condition_count": 18 if evaluation else 12,
            "duplicate_action_conflicts": 0,
            "expected_archive_condition_count": 18 if evaluation else 12,
            "label_variable_group_count": len(observations) // 2,
            "multi_action_group_count": len(observations) // 2,
            "observation_count": len(observations),
            "search_seeds": seeds,
            "source_lineages": [8_701, 8_705],
            "total_archive_edges": len(observations),
        },
    )


def test_protocol_is_zero_sdk_chronological_and_has_no_physical_phase() -> None:
    protocol = FutureViabilityProtocol()
    assert not set(protocol.training_search_seeds) & set(
        protocol.evaluation_search_seeds
    )
    assert protocol.maximum_sdk_calls == 0
    assert protocol.future_horizon == 4
    assert protocol.binding_shift == 1
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--authority-manifest",
            "a.json",
            "--authority-receipt",
            "ar.json",
            "--training-manifest",
            "t.json",
            "--training-receipt",
            "tr.json",
            "--evaluation-manifest",
            "e.json",
            "--evaluation-receipt",
            "er.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["compile"]).phase == "compile"
    assert parser.parse_args(["evaluate"]).phase == "evaluate"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["holdout"])


def test_protocol_rejects_post_hoc_gate_or_seed_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityProtocol(minimum_evaluation_top1_accuracy=0.69)
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityProtocol(evaluation_search_seeds=(9_201, 9_202, 9_204))


def test_future_model_beats_immediate_and_binding_swap() -> None:
    observations = _observations()
    training = tuple(item for item in observations if item.search_seed != 9_103)
    evaluation = tuple(item for item in observations if item.search_seed == 9_103)
    future = FutureViabilityModel.fit(
        training,
        target_field="productive_reach",
        radius=7,
        minimum_signature_support=2,
    )
    immediate = FutureViabilityModel.fit(
        training,
        target_field="immediate_score",
        radius=7,
        minimum_signature_support=2,
    )
    loaded = FutureViabilityModel.from_dict(future.to_dict())
    assert loaded.score(evaluation[0]) == future.score(evaluation[0])
    audit = evaluate_future_viability_ranking(
        evaluation,
        future_model=future,
        immediate_model=immediate,
        binding_shift=1,
    )
    metrics = audit["metrics"]
    assert metrics["future_binding_top1_accuracy"] == 1.0
    assert metrics["immediate_binding_top1_accuracy"] == 0.0
    assert metrics["binding_swap_top1_accuracy"] == 0.0
    assert metrics["target_local_signature_coverage"] == 1.0


def test_model_checksum_rejects_tampering() -> None:
    model = FutureViabilityModel.fit(
        _observations(),
        target_field="productive_reach",
        radius=7,
        minimum_signature_support=2,
    )
    payload = model.to_dict()
    payload["global_mean"] = 99.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        FutureViabilityModel.from_dict(payload)


def test_crossfit_keeps_search_seed_out_and_is_binding_specific() -> None:
    audit = crossfit_future_viability(
        _observations(),
        search_seeds=(9_101, 9_102, 9_103),
        radius=7,
        minimum_signature_support=2,
        binding_shift=1,
    )
    assert [item["holdout_search_seed"] for item in audit["folds"]] == [
        9_101,
        9_102,
        9_103,
    ]
    assert audit["micro_metrics"]["eligible_groups"] == 270
    assert audit["micro_metrics"]["future_binding_top1_accuracy"] == 1.0
    assert audit["micro_metrics"]["future_gain_over_immediate"] == 1.0
    assert audit["micro_metrics"]["future_gain_over_binding_swap"] == 1.0


def test_freeze_binds_t12_5c_authority_and_both_negative_archive_corpora(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability(
        output_path=manifest_path,
        authority_manifest_path=_authority() / "manifest.json",
        authority_receipt_path=_authority() / "control" / "control_receipt.json",
        training_manifest_path=_training_parent() / "manifest.json",
        training_receipt_path=(
            _training_parent() / "paired" / "target_regrounding_receipt.json"
        ),
        evaluation_manifest_path=_evaluation_parent() / "manifest.json",
        evaluation_receipt_path=(
            _evaluation_parent() / "paired" / "hazard_diversity_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["authority_parent"]["receipt"]["status"] == (
        "PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE"
    )
    assert loaded["training_parent"]["receipt"]["status"] == (
        "FAIL_T12_4A_4D_TARGET_WITNESS_GATE"
    )
    assert loaded["evaluation_parent"]["receipt"]["status"] == (
        "FAIL_T12_4A_4D_1_HAZARD_DIVERSITY_GATE"
    )
    assert len(loaded["inputs"]["training_archives"]) == 12
    assert len(loaded["inputs"]["evaluation_archives"]) == 18
    assert loaded["firewall"]["compile_authorized"]
    assert loaded["firewall"]["evaluation_authorized"] is False
    assert loaded["firewall"]["environment_collection_authorized"] is False


def test_synthetic_compile_and_evaluation_pass_only_to_t12_6b_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = FutureViabilityProtocol()
    parent_receipt = (
        _evaluation_parent() / "paired" / "hazard_diversity_receipt.json"
    )
    manifest = {
        "claim_boundary": {
            "authorized": "offline target-local future viability",
            "not_authorized": ["level progress", "physical control"],
        },
        "firewall": {"compile_authorized": True},
        "inputs": {"training_archives": [], "evaluation_archives": []},
        "manifest_checksum": "m" * 64,
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
        "authority_parent": {"receipt": {"receipt_checksum": "a" * 64}},
        "evaluation_parent": {"receipt": {"path": str(parent_receipt)}},
    }
    monkeypatch.setattr(
        experiment,
        "load_future_viability_manifest",
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
    compile_receipt = experiment.compile_future_viability(
        manifest_path="unused.json",
        output_dir=compile_dir,
        root=_repo(),
    )
    assert compile_receipt["passed"] is True
    assert compile_receipt["status"] == "PASS_T12_6_COMPILE_GATE"
    before = experiment.future_viability_status(
        manifest_path="unused.json",
        compile_receipt_path=compile_dir / "compile_receipt.json",
        evaluation_receipt_path=tmp_path / "missing.json",
        root=_repo(),
    )
    assert before["firewall"]["evaluation_authorized"]
    assert before["firewall"]["environment_collection_authorized"] is False

    evaluation_dir = tmp_path / "evaluation"
    evaluation_receipt = experiment.evaluate_future_viability(
        manifest_path="unused.json",
        compile_receipt_path=compile_dir / "compile_receipt.json",
        output_dir=evaluation_dir,
        root=_repo(),
    )
    assert evaluation_receipt["passed"] is True
    assert evaluation_receipt["status"] == "PASS_T12_6_FUTURE_VIABILITY_GATE"
    status = experiment.future_viability_status(
        manifest_path="unused.json",
        compile_receipt_path=compile_dir / "compile_receipt.json",
        evaluation_receipt_path=evaluation_dir / "evaluation_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"]
    assert status["firewall"]["t12_6b_physical_freeze_authorized"]
    assert status["firewall"]["environment_collection_authorized"] is False
    assert status["firewall"]["controller_authority"] is False
