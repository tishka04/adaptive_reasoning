from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from theory.sage_t.causal import progress_shadow_experiment as experiment
from theory.sage_t.causal import progress_shadow_protocol as protocol_module
from theory.sage_t.causal.option_minimization_experiment import _load_contextual_option
from theory.sage_t.causal.options import MinimalCausalOption
from theory.sage_t.causal.progress import CausalProgressProgram
from theory.sage_t.causal.progress_shadow import (
    EmpiricalActionEffectModel,
    build_shadow_ranking,
    posterior_from_snapshot,
    step_from_projection,
)
from theory.sage_t.causal.progress_shadow_cli import build_parser
from theory.sage_t.causal.progress_shadow_protocol import (
    ProgressShadowProtocol,
    freeze_progress_shadow,
    load_progress_shadow_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "causal_progress_t12_5_bp35"


def _parent_inputs():
    root = _parent()
    posterior = experiment._read_json(root / "compiled" / "joint_progress_posterior.json")
    registry = experiment._read_json(root / "compiled" / "progress_program_registry.json")
    programs = tuple(
        CausalProgressProgram.from_dict(dict(item))
        for item in registry["programs"]
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
    return posterior, programs, option


def _synthetic_trials(
    *,
    lineage_seed: int,
    protocol: ProgressShadowProtocol,
    programs: tuple[CausalProgressProgram, ...],
) -> tuple[experiment.ProgressShadowTrial, ...]:
    milestones = experiment._ordered_milestones(programs)
    trials = []
    for stage in protocol.stages:
        prefix = tuple(
            milestone.as_step(
                position=index, action_name=protocol.expected_actions[index]
            )
            for index, milestone in enumerate(milestones[:stage])
        )
        for action_index, action in enumerate(protocol.candidate_actions):
            if action == protocol.expected_actions[stage]:
                candidate = milestones[stage].as_step(
                    position=stage, action_name=action
                )
            else:
                vector = [0] * len(protocol.allowed_effect_features)
                vector[0] = 100 + stage * 10 + action_index
                candidate = step_from_projection(
                    action_name=action,
                    vector=vector,
                    features=protocol.allowed_effect_features,
                    position=stage,
                )
            for repetition in range(protocol.repetitions_per_branch):
                trials.append(
                    experiment.ProgressShadowTrial(
                        trial_id=(
                            f"lineage_{lineage_seed}_stage_{stage}_"
                            f"{action.lower()}_rep_{repetition}"
                        ),
                        lineage_seed=lineage_seed,
                        stage=stage,
                        action_name=action,
                        repetition=repetition,
                        prefix_exact=True,
                        branch_available=True,
                        expected_stage_hash=f"stage-{stage}",
                        observed_stage_hash=f"stage-{stage}",
                        prefix_steps=prefix,
                        candidate_step=candidate,
                        level_delta=int(
                            stage == 4 and action == protocol.expected_actions[stage]
                        ),
                        terminal=False,
                        terminal_failure=False,
                        sdk_calls_after=len(trials) + 1,
                    )
                )
    return tuple(trials)


def test_protocol_is_paired_bounded_and_shadow_only() -> None:
    protocol = ProgressShadowProtocol()
    assert protocol.expected_trial_count == 60
    assert protocol.candidate_actions == ("ACTION3", "ACTION4", "ACTION6")
    assert protocol.excluded_non_executable_actions == ("ACTION7",)
    assert protocol.maximum_sdk_calls == 5_000
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    assert protocol.induction_lineage_seed != protocol.confirmation_lineage_seed
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "manifest.json",
            "--parent-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["control"])


def test_freeze_binds_passed_t12_5_and_keeps_control_closed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_progress_shadow(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "compiled" / "causal_progress_receipt.json",
        root=_repo(),
    )
    loaded = load_progress_shadow_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "PASS_T12_5_CAUSAL_PROGRESS_GATE"
    )
    assert loaded["firewall"]["causal_progress_shadow_collection_authorized"]
    assert loaded["firewall"]["causal_progress_control_authorized"] is False
    assert loaded["firewall"]["production_authority"] is False
    assert loaded["storage"]["maximum_sdk_calls"] == 5_000


def test_effect_model_uses_effects_and_not_action_names_for_progress() -> None:
    protocol = ProgressShadowProtocol()
    posterior_payload, programs, _ = _parent_inputs()
    posterior = posterior_from_snapshot(posterior_payload)
    trials = _synthetic_trials(
        lineage_seed=protocol.induction_lineage_seed,
        protocol=protocol,
        programs=programs,
    )
    model = EmpiricalActionEffectModel.fit(
        [item.to_dict() for item in trials],
        features=protocol.allowed_effect_features,
        induction_lineage_seed=protocol.induction_lineage_seed,
        expected_stages=protocol.stages,
        candidate_actions=protocol.candidate_actions,
    )
    for stage in protocol.stages:
        ranking = build_shadow_ranking(
            posterior=posterior,
            model=model,
            stage=stage,
            candidate_actions=protocol.candidate_actions,
            expected_actions=protocol.expected_actions,
        )
        assert ranking["rankings"]["causal_progress"][0] == (
            protocol.expected_actions[stage]
        )
        assert ranking["causal_margin"] > 0.0


def test_synthetic_paired_shadow_run_passes_without_policy_authority(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = ProgressShadowProtocol()
    posterior, programs, option = _parent_inputs()
    manifest = {
        "claim_boundary": {
            "authorized": "source-train observed-effect shadow ranking",
            "not_authorized": ["environment control"],
        },
        "design": {"rankings_never_select_executed_actions": True},
        "firewall": {"causal_progress_shadow_collection_authorized": True},
        "game_id": "bp35",
        "inputs": {
            "successful_prefix_lengths": {"8701": 0, "8705": 0},
        },
        "manifest_checksum": "m" * 64,
        "parent": {"receipt": {"receipt_checksum": "r" * 64}},
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    witnesses = tuple(
        SimpleNamespace(source_seed=seed) for seed in protocol.lineage_seeds
    )
    by_seed = {
        seed: _synthetic_trials(
            lineage_seed=seed, protocol=protocol, programs=programs
        )
        for seed in protocol.lineage_seeds
    }
    monkeypatch.setattr(
        experiment,
        "load_progress_shadow_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_load_inputs",
        lambda *args, **kwargs: (witnesses, option, posterior, programs, {}),
    )
    monkeypatch.setattr(
        experiment,
        "_expected_stage_hashes",
        lambda *args, **kwargs: {
            (seed, stage): f"stage-{stage}"
            for seed in protocol.lineage_seeds
            for stage in protocol.stages
        },
    )
    monkeypatch.setattr(
        experiment,
        "_collect_lineage",
        lambda **kwargs: by_seed[int(kwargs["witness"].source_seed)],
    )
    output = tmp_path / "shadow"
    receipt = experiment.run_progress_shadow(
        manifest_path="unused.json",
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_5B_PROGRESS_SHADOW_GATE"
    metrics = receipt["metrics"]
    assert metrics["trial_count"] == 60
    assert metrics["exact_prefix_rate"] == 1.0
    assert metrics["effect_transport"]["exact_projection_transport_rate"] == 1.0
    assert metrics["ranking"]["per_method"]["causal_progress"][
        "top1_accuracy"
    ] == 1.0
    assert metrics["observed_confirmation"]["top1_accuracy"] == 1.0
    assert all(metrics["checks"].values())
    assert (output / "shadow_trials.json").is_file()
    assert (output / "effect_model.json").is_file()
    assert (output / "shadow_rankings.json").is_file()
    assert (output / "shadow_receipt.json").is_file()
