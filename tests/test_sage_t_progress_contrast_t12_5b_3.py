from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from theory.sage_t.causal import progress_contrast_experiment as experiment
from theory.sage_t.causal import progress_contrast_protocol as protocol_module
from theory.sage_t.causal.option_minimization_experiment import (
    _load_contextual_option,
)
from theory.sage_t.causal.options import MinimalCausalOption
from theory.sage_t.causal.progress import CausalProgressProgram
from theory.sage_t.causal.progress_contrast import (
    audit_prospective_progress_contrasts,
)
from theory.sage_t.causal.progress_contrast_cli import build_parser
from theory.sage_t.causal.progress_contrast_protocol import (
    ProgressContrastProtocol,
    freeze_progress_contrast,
    load_progress_contrast_manifest,
)
from theory.sage_t.causal.progress_shadow import (
    posterior_from_snapshot,
    step_from_projection,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "progress_discrimination_t12_5b_2_bp35"
    )


def _source_inputs():
    shadow = (
        _repo() / "training" / "sage_t" / "progress_shadow_t12_5b_r1_bp35"
    )
    manifest = json.loads((shadow / "manifest.json").read_text())
    posterior_payload = json.loads(
        (_repo() / manifest["inputs"]["posterior"]["path"]).read_text()
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
    option_payload = _load_contextual_option(
        _repo()
        / "training"
        / "sage_t"
        / "option_minimization_t12_4a_3r1_bp35"
        / "ablation"
        / "minimal_option.json"
    )
    option = MinimalCausalOption.from_dict(dict(option_payload["option"]))
    return posterior_payload, programs, milestones, option


def _synthetic_trials(*, hard: bool) -> tuple[experiment.ProgressContrastTrial, ...]:
    protocol = ProgressContrastProtocol()
    _, _, milestones, _ = _source_inputs()
    trials = []
    for seed in protocol.lineage_seeds:
        progress_action = "ACTION3" if seed == 8701 else "ACTION6"
        distractor_action = "ACTION4" if seed == 8701 else "ACTION3"
        for depth, context_id in zip(protocol.detour_depths, protocol.context_ids):
            prefix = [
                milestone.as_step(position=index, action_name="PREFIX")
                for index, milestone in enumerate(milestones[: protocol.target_stage])
            ]
            prefix.extend(
                step_from_projection(
                    action_name=protocol.detour_action,
                    vector=(0,) * len(protocol.allowed_effect_features),
                    features=protocol.allowed_effect_features,
                    position=protocol.target_stage + index,
                )
                for index in range(depth)
            )
            for action in protocol.candidate_actions:
                if action == progress_action:
                    candidate = milestones[protocol.target_stage].as_step(
                        position=len(prefix), action_name=action
                    )
                elif action == distractor_action:
                    vector = [0] * len(protocol.allowed_effect_features)
                    vector[0] = 1_000 if hard else 0
                    candidate = step_from_projection(
                        action_name=action,
                        vector=vector,
                        features=protocol.allowed_effect_features,
                        position=len(prefix),
                    )
                else:
                    candidate = step_from_projection(
                        action_name=action,
                        vector=(0,) * len(protocol.allowed_effect_features),
                        features=protocol.allowed_effect_features,
                        position=len(prefix),
                    )
                for repetition in range(protocol.repetitions_per_branch):
                    trials.append(
                        experiment.ProgressContrastTrial(
                            trial_id=(
                                f"synthetic_{seed}_{context_id}_{action}_{repetition}"
                            ),
                            lineage_seed=seed,
                            stage=protocol.target_stage,
                            context_id=context_id,
                            detour_action=protocol.detour_action,
                            detour_depth=depth,
                            action_name=action,
                            repetition=repetition,
                            original_prefix_exact=True,
                            detour_available=True,
                            detour_neutral=True,
                            detour_terminal=False,
                            detour_context_hash=f"context-{seed}-{depth}",
                            prefix_exact=True,
                            branch_available=True,
                            prefix_steps=tuple(prefix),
                            candidate_step=candidate,
                            level_delta=0,
                            terminal=False,
                            terminal_failure=False,
                            sdk_calls_after=len(trials) + 1,
                        )
                    )
    return tuple(trials)


def _audit(trials: tuple[experiment.ProgressContrastTrial, ...]):
    protocol = ProgressContrastProtocol()
    posterior_payload, _, milestones, _ = _source_inputs()
    return audit_prospective_progress_contrasts(
        trials=[item.to_dict() for item in trials],
        features=protocol.allowed_effect_features,
        posterior=posterior_from_snapshot(posterior_payload),
        milestones=milestones,
        lineage_seeds=protocol.lineage_seeds,
        target_stage=protocol.target_stage,
        context_ids=protocol.context_ids,
        candidate_actions=protocol.candidate_actions,
        repetitions_per_branch=protocol.repetitions_per_branch,
        minimum_distractor_magnitude_gap=(
            protocol.minimum_distractor_magnitude_gap
        ),
    )


def test_protocol_is_prospective_bounded_and_has_no_control_phase() -> None:
    protocol = ProgressContrastProtocol()
    assert protocol.target_stage == 3
    assert protocol.detour_action == "ACTION4"
    assert protocol.detour_depths == (1, 2, 3)
    assert protocol.expected_trial_count == 36
    assert protocol.maximum_sdk_calls == 3_500
    assert protocol.maximum_wall_seconds == 7_200
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
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
    assert parser.parse_args(["collect"]).phase == "collect"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["control"])
    with pytest.raises(SystemExit):
        parser.parse_args(["holdout"])


def test_freeze_binds_exact_parent_and_unique_nearest_stage(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_progress_contrast(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "audit" / "discrimination_receipt.json",
        root=_repo(),
    )
    loaded = load_progress_contrast_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    )
    assert loaded["selection"]["unique_nearest_stage"] == 3
    assert loaded["selection"]["detour_action"] == "ACTION4"
    assert loaded["selection"]["parent_causal_score_used_for_selection"] is False
    assert loaded["firewall"]["prospective_contrast_collection_authorized"]
    assert loaded["firewall"]["causal_progress_control_authorized"] is False
    assert loaded["firewall"]["t12_5c_control_freeze_authorized"] is False
    assert loaded["firewall"]["holdout_opened"] is False


def test_synthetic_prospective_hard_contrasts_bind_without_action_identity() -> None:
    audit = _audit(_synthetic_trials(hard=True))
    metrics = audit["metrics"]
    assert metrics["valid_context_count"] == 6
    assert metrics["common_valid_context_count"] == 3
    assert metrics["minimum_executable_actions_per_valid_context"] == 3
    assert metrics["affordance_binding_count"] == 3
    assert metrics["hard_contrast_count"] == 6
    assert metrics["common_hard_contrast_context_count"] == 3
    assert metrics["causal_hard_contrast_accuracy"] == 1.0
    assert metrics["magnitude_hard_contrast_accuracy"] == 0.0
    assert metrics["hard_contrast_accuracy_gain"] == 1.0
    assert any(
        len(set(binding["action_names"].values())) == 2
        for binding in audit["affordance_registry"]["bindings"]
    )
    assert all(
        binding["matching_fields"] == ["stage", "milestone_signature"]
        for binding in audit["affordance_registry"]["bindings"]
    )


def test_prospective_collection_without_hard_contrast_remains_negative() -> None:
    audit = _audit(_synthetic_trials(hard=False))
    metrics = audit["metrics"]
    assert metrics["progress_affordance_lineage_count"] == 2
    assert metrics["hard_contrast_count"] == 0
    assert metrics["common_hard_contrast_context_count"] == 0


def test_synthetic_collection_passes_only_to_t12_5c_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = ProgressContrastProtocol()
    posterior_payload, programs, _, option = _source_inputs()
    manifest = {
        "claim_boundary": {
            "authorized": "source-train prospective contrast collection",
            "not_authorized": ["environment control", "holdout performance"],
        },
        "design": {"posterior_never_selects_executed_actions": True},
        "firewall": {"prospective_contrast_collection_authorized": True},
        "game_id": "bp35",
        "inputs": {"successful_prefix_lengths": {"8701": 0, "8705": 0}},
        "manifest_checksum": "m" * 64,
        "parent": {
            "receipt": {
                "receipt_checksum": "r" * 64,
                "status": "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS",
            }
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    witnesses = tuple(
        SimpleNamespace(source_seed=seed) for seed in protocol.lineage_seeds
    )
    by_seed = {
        seed: tuple(
            item
            for item in _synthetic_trials(hard=True)
            if item.lineage_seed == seed
        )
        for seed in protocol.lineage_seeds
    }
    monkeypatch.setattr(
        experiment, "load_progress_contrast_manifest", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(
        experiment,
        "_load_inputs",
        lambda *args, **kwargs: (
            witnesses,
            option,
            posterior_payload,
            programs,
            {},
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_expected_stage_hashes",
        lambda *args, **kwargs: {
            (seed, protocol.target_stage): "stage-3"
            for seed in protocol.lineage_seeds
        },
    )
    monkeypatch.setattr(
        experiment,
        "_collect_lineage",
        lambda **kwargs: by_seed[int(kwargs["witness"].source_seed)],
    )
    output = tmp_path / "collection"
    receipt = experiment.run_progress_contrast_collection(
        manifest_path="unused.json",
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_5B_3_PROSPECTIVE_CONTRAST_GATE"
    assert all(receipt["metrics"]["checks"].values())
    assert (output / "contrast_trials.json").is_file()
    assert (output / "affordance_registry.json").is_file()
    assert (output / "hard_contrast_registry.json").is_file()
    assert (output / "contrast_receipt.json").is_file()
    status = experiment.progress_contrast_status(
        manifest_path="unused.json",
        receipt_path=output / "contrast_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"] is True
    assert status["firewall"]["t12_5c_control_freeze_authorized"] is True
    assert status["firewall"]["environment_collection_authorized"] is False
    assert status["firewall"]["causal_progress_control_authorized"] is False
