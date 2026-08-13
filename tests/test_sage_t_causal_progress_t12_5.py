from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage_t.causal import progress_protocol as protocol_module
from theory.sage_t.causal.option_contracts import EffectAtom, StepEffectContract
from theory.sage_t.causal.progress import (
    CausalProgressActionEvaluator,
    CausalProgressExecutor,
    JointCausalProgressPosterior,
    ProgressEvidence,
    rival_progress_programs,
)
from theory.sage_t.causal.progress_experiment import (
    causal_progress_status,
    compile_causal_progress,
)
from theory.sage_t.causal.progress_experiment_cli import build_parser
from theory.sage_t.causal.progress_protocol import (
    CausalProgressProtocol,
    freeze_causal_progress,
    load_causal_progress_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _contract_root() -> Path:
    return _repo() / "training" / "sage_t" / "option_contract_t12_4a_4c_bp35"


def _ablation_receipt() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "option_minimization_t12_4a_3r1_bp35"
        / "ablation"
        / "option_ablation_receipt.json"
    )


def _contracts() -> tuple[StepEffectContract, ...]:
    return (
        StepEffectContract(0, "ACTION1", (EffectAtom("role_counts", "a", 1),)),
        StepEffectContract(
            1, "ACTION2", (EffectAtom("predicate_counts", "b", -1),)
        ),
    )


def _programs(owner: str = "owner"):
    return rival_progress_programs(
        owner_program_hash=owner,
        effect_contracts=_contracts(),
        goal_predicate="goal == true",
        failure_predicate=None,
        evidence_ids=("e1", "e2"),
    )


def test_progress_is_effect_grounded_ordered_and_action_label_invariant() -> None:
    programs = {item.progress_kind: item for item in _programs()}
    ordered = programs["ordered_effects"]
    unordered = programs["unordered_effects"]
    executor = CausalProgressExecutor()
    forward = tuple(
        milestone.as_step(position=index, action_name=f"ACTION{index}")
        for index, milestone in enumerate(ordered.milestones)
    )
    reverse = tuple(reversed(forward))

    evaluation = executor.evaluate_trace(ordered, forward)
    assert evaluation.potentials == pytest.approx((0.5, 1.0))
    assert evaluation.predicted_success is True
    assert executor.evaluate_trace(ordered, reverse).predicted_success is False
    assert executor.evaluate_trace(unordered, reverse).predicted_success is True

    relabeled = tuple(
        {**dict(step), "action_name": f"RENAMED_{index}"}
        for index, step in enumerate(forward)
    )
    assert executor.evaluate_trace(ordered, relabeled) == evaluation


def test_joint_posterior_preserves_world_mass_and_learns_order() -> None:
    progress_programs = (*_programs("owner_a"), *_programs("owner_b"))
    posterior = JointCausalProgressPosterior.from_factorized_prior(
        owner_probabilities={"owner_a": 0.7, "owner_b": 0.3},
        progress_programs=progress_programs,
    )
    before = posterior.mass_by_owner()
    ordered = next(
        item
        for item in progress_programs
        if item.owner_program_hash == "owner_a"
        and item.progress_kind == "ordered_effects"
    )
    forward = tuple(item.as_step() for item in ordered.milestones)
    reverse = tuple(reversed(forward))
    posterior.update_many(
        (
            ProgressEvidence("positive", "l1", forward, True, "typed"),
            ProgressEvidence("reverse", "l1", reverse, False, "intervention"),
        )
        * 4
    )
    assert posterior.mass_by_kind()["ordered_effects"] > 0.95
    assert posterior.mass_by_owner() == pytest.approx(before)

    ranking = CausalProgressActionEvaluator.rank(
        posterior,
        {
            "expected": (ordered.milestones[1].as_step(),),
            "irrelevant": ({"delta": {"mechanism": {}}},),
        },
        prefix=(ordered.milestones[0].as_step(),),
    )
    assert ranking[0][0] == "expected"


def _freeze(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    freeze_causal_progress(
        output_path=manifest_path,
        parent_manifest_path=_contract_root() / "manifest.json",
        parent_receipt_path=(
            _contract_root() / "contract" / "option_contract_receipt.json"
        ),
        ablation_receipt_path=_ablation_receipt(),
        root=_repo(),
    )
    return manifest_path


def test_protocol_is_offline_bounded_and_cli_has_no_active_phase() -> None:
    protocol = CausalProgressProtocol()
    assert protocol.maximum_sdk_calls == 0
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    assert protocol.maximum_joint_particles == 96
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
            "--ablation-receipt",
            "ablation.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["compile"]).phase == "compile"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_real_sealed_sources_compile_progress_posterior_offline(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = _freeze(monkeypatch, tmp_path)
    manifest = load_causal_progress_manifest(manifest_path, root=_repo())
    assert manifest["firewall"]["causal_progress_compile_authorized"] is True
    assert manifest["firewall"]["production_authority"] is False
    assert manifest["claim_boundary"]["ablation_order_evidence_has_no_observed_typed_deltas"]

    output = tmp_path / "compiled"
    receipt = compile_causal_progress(
        manifest_path=manifest_path, output_dir=output, root=_repo()
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_5_CAUSAL_PROGRESS_GATE"
    metrics = receipt["metrics"]
    assert metrics["joint_particle_count"] == 96
    assert metrics["sdk_calls_used"] == 0
    assert metrics["ordered_replication_accuracy"] == pytest.approx(1.0)
    assert metrics["posterior_replication_accuracy"] == pytest.approx(1.0)
    assert metrics["posterior_mass_by_kind"]["ordered_effects"] >= 0.95
    assert metrics["maximum_parent_mass_error"] <= 1e-12
    assert all(metrics["checks"].values())
    assert metrics["positive_prefix_scores"] == sorted(
        metrics["positive_prefix_scores"]
    )
    assert max(metrics["failed_prefix_scores"]) == pytest.approx(0.0)

    registry = json.loads((output / "progress_program_registry.json").read_text())
    assert registry["virtual_joint_particle_count"] == 96
    semantic_text = json.dumps(
        [
            {
                "goal": item["goal_predicate"],
                "milestones": item["milestones"],
            }
            for item in registry["programs"]
        ]
    ).lower()
    assert "exact_hash" not in semantic_text
    assert "entity_id" not in semantic_text
    assert "pixel" not in semantic_text

    status = causal_progress_status(
        manifest_path=manifest_path,
        receipt_path=output / "causal_progress_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"] is True
    assert status["firewall"]["causal_progress_shadow_experiment_authorized"]
    assert status["firewall"]["causal_progress_control_authorized"] is False
    assert status["firewall"]["holdout_opened"] is False
