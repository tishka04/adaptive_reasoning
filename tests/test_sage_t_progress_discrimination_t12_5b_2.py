from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage_t.causal import progress_discrimination_protocol as protocol_module
from theory.sage_t.causal.progress import CausalProgressProgram
from theory.sage_t.causal.progress_discrimination import (
    audit_progress_discrimination,
)
from theory.sage_t.causal.progress_discrimination_cli import build_parser
from theory.sage_t.causal.progress_discrimination_experiment import (
    progress_discrimination_status,
    run_progress_discrimination_audit,
)
from theory.sage_t.causal.progress_discrimination_protocol import (
    ProgressDiscriminationProtocol,
    freeze_progress_discrimination,
    load_progress_discrimination_manifest,
)
from theory.sage_t.causal.progress_shadow import (
    posterior_from_snapshot,
    step_from_projection,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "progress_shadow_t12_5b_r1_bp35"


def _real_inputs():
    parent = _parent()
    manifest = json.loads((parent / "manifest.json").read_text())
    trials = json.loads((parent / "shadow" / "shadow_trials.json").read_text())[
        "trials"
    ]
    posterior = posterior_from_snapshot(
        json.loads(
            (_repo() / manifest["inputs"]["posterior"]["path"]).read_text()
        )
    )
    registry = json.loads(
        (_repo() / manifest["inputs"]["program_registry"]["path"]).read_text()
    )
    programs = tuple(
        CausalProgressProgram.from_dict(dict(item)) for item in registry["programs"]
    )
    milestones = next(
        item.milestones for item in programs if item.progress_kind == "ordered_effects"
    )
    return manifest, trials, posterior, milestones


def _freeze(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    path = tmp_path / "manifest.json"
    freeze_progress_discrimination(
        output_path=path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "shadow" / "shadow_receipt.json",
        root=_repo(),
    )
    return path


def test_protocol_is_offline_bounded_and_cli_has_no_collection_phase() -> None:
    protocol = ProgressDiscriminationProtocol()
    assert protocol.maximum_sdk_calls == 0
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    assert set(protocol.authorized_parent_failed_checks) == {
        "all_candidate_actions_available",
        "causal_ranking_beats_non_goal_baselines",
    }
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["audit"]).phase == "audit"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])
    with pytest.raises(SystemExit):
        parser.parse_args(["control"])


def test_freeze_binds_exact_negative_parent_without_opening_control(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = _freeze(monkeypatch, tmp_path)
    manifest = load_progress_discrimination_manifest(
        manifest_path, root=_repo()
    )
    assert manifest["parent"]["receipt"]["status"] == (
        "FAIL_T12_5B_PROGRESS_SHADOW_GATE"
    )
    assert set(manifest["parent"]["failed_checks"]) == {
        "all_candidate_actions_available",
        "causal_ranking_beats_non_goal_baselines",
    }
    assert manifest["design"]["unavailable_action_is_not_a_zero_effect"]
    assert manifest["design"]["action_name_is_not_an_affordance_binding_field"]
    assert manifest["firewall"]["environment_collection_authorized"] is False
    assert manifest["firewall"]["causal_progress_control_authorized"] is False
    assert manifest["firewall"]["production_authority"] is False


def test_real_sealed_trials_have_no_observed_discriminative_contrast() -> None:
    manifest, trials, posterior, milestones = _real_inputs()
    protocol = ProgressDiscriminationProtocol()
    audit = audit_progress_discrimination(
        trials=trials,
        features=manifest["protocol"]["allowed_effect_features"],
        posterior=posterior,
        milestones=milestones,
        lineage_seeds=protocol.lineage_seeds,
        stages=protocol.stages,
        expected_actions=protocol.expected_actions,
        repetitions_per_branch=protocol.repetitions_per_branch,
        induction_lineage_seed=protocol.induction_lineage_seed,
        confirmation_lineage_seed=protocol.confirmation_lineage_seed,
        minimum_distractor_magnitude_gap=(
            protocol.minimum_distractor_magnitude_gap
        ),
    )
    metrics = audit["metrics"]
    assert metrics["exact_prefix_rate"] == 1.0
    assert metrics["availability_is_deterministic"] is True
    assert metrics["affordance_binding_coverage"] == 1.0
    assert metrics["minimum_executable_actions_per_context"] == 2
    assert metrics["unavailable_affordance_count"] == 5
    assert {
        (item["lineage_seed"], item["action_name"])
        for item in metrics["unavailable_affordances"]
    } == {(8705, "ACTION6")}
    assert metrics["causal_top1_accuracy"] == 1.0
    assert metrics["magnitude_top1_accuracy"] == 1.0
    assert metrics["ranking_disagreement_count"] == 0
    assert metrics["hard_contrast_count"] == 0


def test_synthetic_larger_nonprogress_effect_is_a_hard_contrast() -> None:
    manifest, _, posterior, milestones = _real_inputs()
    features = tuple(manifest["protocol"]["allowed_effect_features"])
    expected = tuple(f"GO{stage}" for stage in range(5))
    trials = []
    for seed in (8701, 8705):
        for stage in range(5):
            prefix = tuple(
                milestone.as_step(position=index, action_name=expected[index])
                for index, milestone in enumerate(milestones[:stage])
            )
            candidates = {
                expected[stage]: milestones[stage].as_step(
                    position=stage, action_name=expected[stage]
                ),
                "DISTRACTOR": step_from_projection(
                    action_name="DISTRACTOR",
                    vector=(1000,) * len(features),
                    features=features,
                    position=stage,
                ),
            }
            for action, step in candidates.items():
                for repetition in range(2):
                    trials.append(
                        {
                            "action_name": action,
                            "branch_available": True,
                            "candidate_step": step,
                            "lineage_seed": seed,
                            "prefix_exact": True,
                            "prefix_steps": prefix,
                            "repetition": repetition,
                            "stage": stage,
                            "trial_id": (
                                f"synthetic_{seed}_{stage}_{action}_{repetition}"
                            ),
                        }
                    )
    audit = audit_progress_discrimination(
        trials=trials,
        features=features,
        posterior=posterior,
        milestones=milestones,
        lineage_seeds=(8701, 8705),
        stages=tuple(range(5)),
        expected_actions=expected,
        repetitions_per_branch=2,
        induction_lineage_seed=8701,
        confirmation_lineage_seed=8705,
        minimum_distractor_magnitude_gap=1.0,
    )
    assert audit["metrics"]["hard_contrast_count"] == 10
    assert audit["metrics"]["hard_contrast_lineage_count"] == 2
    assert audit["metrics"]["ranking_disagreement_count"] == 10
    assert all(
        binding["matching_fields"] == ["stage", "milestone_signature"]
        for binding in audit["affordance_registry"]["bindings"]
    )


def test_offline_audit_seals_insufficiency_and_only_authorizes_collection_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = _freeze(monkeypatch, tmp_path)
    output = tmp_path / "audit"
    receipt = run_progress_discrimination_audit(
        manifest_path=manifest_path,
        output_dir=output,
        root=_repo(),
    )
    assert receipt["passed"] is False
    assert receipt["status"] == (
        "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    )
    metrics = receipt["metrics"]
    assert metrics["classification"] == "INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    assert metrics["collection_freeze_authorized"] is True
    assert metrics["hard_contrast_count"] == 0
    assert metrics["sdk_calls_used"] == 0
    assert metrics["checks"]["hard_contrast_exists_in_every_lineage"] is False
    assert all(
        metrics["checks"][name]
        for name in (
            "all_prefixes_exact",
            "local_availability_is_deterministic",
            "progress_action_is_locally_executable",
            "progress_affordance_transports_semantically",
            "no_environment_calls",
        )
    )
    status = progress_discrimination_status(
        manifest_path=manifest_path,
        receipt_path=output / "discrimination_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"] is True
    assert status["firewall"]["t12_5b_3_collection_freeze_authorized"] is True
    assert status["firewall"]["environment_collection_authorized"] is False
    assert status["firewall"]["t12_5c_control_freeze_authorized"] is False

