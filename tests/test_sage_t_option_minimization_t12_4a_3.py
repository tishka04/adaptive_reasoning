from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import option_minimization_experiment as experiment
from theory.sage_t.causal import option_minimization_protocol as protocol_module
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.option_minimization_cli import build_parser
from theory.sage_t.causal.option_minimization_protocol import (
    OptionMinimizationProtocol,
    freeze_option_minimization,
    load_option_minimization_manifest,
    load_option_minimization_receipt,
)
from theory.sage_t.causal.options import MinimalCausalOption, MinimalOptionStep
from theory.sage_t.causal.witness_protocol import WitnessStep
from theory.sage_t.contracts import AbstractState


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class FakeAction:
    id: int
    data: dict | None = None


@dataclass
class FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (3, 4)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(3), FakeAction(4)]


class ExactSequenceEnv:
    sequence = (3, 4, 4, 4, 3, 3)

    def __init__(self) -> None:
        self._game = FakeGame()
        self.actions: list[int] = []

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.actions = []
        else:
            self.actions.append(value)
        progress = tuple(self.actions) == self.sequence
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = 99 if progress else len(self.actions)
        return FakeFrame(
            grid,
            state="WIN" if progress else "NOT_FINISHED",
            levels_completed=int(progress),
        )


def _candidate_steps() -> tuple[WitnessStep, ...]:
    return tuple(
        WitnessStep(
            expected_source_hash=f"source-{index}",
            action=GroundedAction(f"ACTION{action}"),
            expected_target_hash=f"target-{index}",
        )
        for index, action in enumerate(ExactSequenceEnv.sequence)
    )


def test_protocol_is_exhaustive_bounded_and_shadow_only() -> None:
    selected = OptionMinimizationProtocol()
    assert selected.exhaustive_subsequence_count == 64
    assert selected.repetitions_per_candidate_context == 3
    assert selected.maximum_sdk_calls == 24_000
    assert selected.maximum_artifact_bytes_per_run == 3 * 1024**3
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
            "--program-registry",
            "programs.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["ablate"]).phase == "ablate"
    assert parser.parse_args(["compile-shadow"]).phase == "compile-shadow"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_context_loader_reads_candidate_length_from_frozen_protocol(
    monkeypatch,
) -> None:
    protocol = OptionMinimizationProtocol()
    steps = tuple(
        WitnessStep(
            expected_source_hash=f"source-{index}",
            action=GroundedAction(f"ACTION{action}"),
            expected_target_hash=f"target-{index}",
        )
        for index, action in enumerate((1, 2, *ExactSequenceEnv.sequence))
    )
    witnesses = tuple(
        SimpleNamespace(
            source_seed=seed,
            witness_id=f"witness-{seed}",
            steps=steps,
            initial_exact_hash="initial",
            target_exact_hash="target",
            target_level=1,
        )
        for seed in protocol.source_seeds
    )
    manifest = {
        "parent": {"witness_registry": {"path": "registry.json"}},
        "protocol": protocol_module.asdict(protocol),
        "source_archives": [
            {"path": f"archive-{seed}.json", "seed": seed, "sha256": str(seed)}
            for seed in protocol.source_seeds
        ],
    }
    monkeypatch.setattr(
        experiment,
        "load_reconfirmation_registry",
        lambda path: ({}, witnesses),
    )
    monkeypatch.setattr(experiment, "_read_json", lambda path: {})
    monkeypatch.setattr(
        experiment.GoExploreArchive,
        "from_dict",
        lambda payload: SimpleNamespace(),
    )
    monkeypatch.setattr(
        experiment,
        "_archive_state_for_exact_hash",
        lambda archive, exact_hash: AbstractState(),
    )
    contexts = experiment._load_contexts(manifest)
    assert len(contexts) == 2
    assert [len(context.candidate) for context, _ in contexts] == [6, 6]
    assert [len(context.prefix) for context, _ in contexts] == [2, 2]
    assert [
        step.action.action_name for step in contexts[0][0].candidate
    ] == list(OptionMinimizationProtocol().expected_common_suffix)


def test_freeze_binds_passed_witnesses_and_keeps_control_closed(
    monkeypatch,
    tmp_path,
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "witness_reconfirmation_t12_4a_2_bp35"
    programs = repo / "training" / "sage_t" / "causal_inputs" / "programs.sealed.json"
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_option_minimization(
        output_path=manifest_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "confirmation" / "witness_receipt.json",
        program_registry_path=programs,
        root=repo,
    )
    loaded = load_option_minimization_manifest(
        manifest_path,
        root=repo,
        verify_code=False,
    )
    assert manifest["manifest_checksum"] == loaded["manifest_checksum"]
    assert [item["seed"] for item in loaded["source_archives"]] == [8701, 8705]
    assert loaded["firewall"]["option_ablation_authorized"] is True
    assert loaded["firewall"]["option_compilation_authorized"] is False
    assert loaded["firewall"]["option_active_authority"] is False
    assert loaded["firewall"]["t12_4a_4_transfer_freeze_authorized"] is False


def test_exhaustive_run_extracts_unique_six_step_option(
    monkeypatch,
    tmp_path,
) -> None:
    initial_frame = ExactSequenceEnv().step(0)
    initial_hash = state_signature_from_frame(initial_frame)
    target_env = ExactSequenceEnv()
    target_frame = target_env.step(0)
    for action in ExactSequenceEnv.sequence:
        target_frame = target_env.step(action)
    target_hash = state_signature_from_frame(target_frame)
    candidate = _candidate_steps()
    contexts = tuple(
        (
            experiment.AblationContext(
                seed=seed,
                witness_id=f"witness-{seed}",
                prefix=(),
                candidate=candidate,
                initial_exact_hash=initial_hash,
                initiation_exact_hash=initial_hash,
                initiation_signature=f"context-{seed}",
                target_exact_hash=target_hash,
                target_level=1,
                archive_checksum=str(seed) * 8,
            ),
            None,
        )
        for seed in (8701, 8705)
    )
    protocol = OptionMinimizationProtocol()
    manifest = {
        "firewall": {"option_ablation_authorized": True},
        "game_id": "bp35",
        "manifest_checksum": "m" * 64,
        "parent": {"receipt": {"receipt_checksum": "r" * 64}},
        "protocol": protocol_module.asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    monkeypatch.setattr(
        experiment,
        "load_option_minimization_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(experiment, "_load_contexts", lambda value: contexts)
    monkeypatch.setattr(
        experiment,
        "_make_env",
        lambda game_id, environments_dir, env_factory: ExactSequenceEnv(),
    )
    monkeypatch.setattr(experiment, "_reset_env", lambda env: env.step(0))
    output = tmp_path / "ablation"
    receipt = experiment.run_option_ablation(
        Path("unused.json"),
        output_dir=output,
        environments_dir=Path("unused"),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_4A_3_OPTION_ABLATION_GATE"
    assert receipt["metrics"]["trial_count"] == 390
    assert receipt["metrics"]["sdk_calls_used"] == 1_578
    assert receipt["metrics"]["selected_indices"] == [0, 1, 2, 3, 4, 5]
    assert receipt["metrics"]["minimum_successful_length"] == 6
    loaded = load_option_minimization_receipt(output / "option_ablation_receipt.json")
    assert loaded["receipt_checksum"] == receipt["receipt_checksum"]
    assert (output / "minimal_option.json").is_file()
    assert receipt["metrics"]["storage"]["within_budget"] is True


def test_shadow_compile_owns_complete_posterior_without_control(
    monkeypatch,
    tmp_path,
) -> None:
    repo = _repo()
    protocol = OptionMinimizationProtocol()
    manifest = {
        "firewall": {"option_ablation_authorized": True},
        "game_id": "bp35",
        "manifest_checksum": "m" * 64,
        "parent": {
            "receipt": {
                "receipt_checksum": "r" * 64,
                "status": "PASS_T12_4A_2_WITNESS_GATE",
            }
        },
        "program_registry": {
            "path": str(
                repo / "training" / "sage_t" / "causal_inputs" / "programs.sealed.json"
            )
        },
        "protocol": protocol_module.asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    option = MinimalCausalOption(
        initiation_signature=AbstractState().signature,
        initiation_exact_hash="exact-init",
        steps=tuple(
            MinimalOptionStep(f"ACTION{action}", {})
            for action in ExactSequenceEnv.sequence
        ),
        source_evidence_ids=("witness-8701", "witness-8705"),
        reproduction_count=6,
        minimization_evaluations=384,
        source="t12_4a_3_test",
    )
    contextual = protocol_module._signed(
        {
            "context_bindings": [{"seed": 8701}, {"seed": 8705}],
            "format_version": protocol_module.CONTEXTUAL_OPTION_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "minimality": {"exhaustive": True, "unique_minimal": True},
            "option": option.safe_payload,
            "option_checksum": option.checksum,
            "parent_t12_4a_2_receipt_checksum": "r" * 64,
            "protocol_checksum": protocol.checksum,
        },
        "contextual_option_checksum",
    )
    option_path = tmp_path / "minimal_option.json"
    option_path.write_text(json.dumps(contextual), encoding="utf-8")
    ablation_receipt = {
        "artifacts": {"minimal_option": {"path": str(option_path)}},
        "manifest_checksum": manifest["manifest_checksum"],
        "passed": True,
        "phase": "option_ablation",
        "receipt_checksum": "a" * 64,
    }
    monkeypatch.setattr(
        experiment,
        "load_option_minimization_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "load_option_minimization_receipt",
        lambda *args, **kwargs: ablation_receipt,
    )
    monkeypatch.setattr(
        experiment,
        "_load_contexts",
        lambda value: (
            (
                SimpleNamespace(initiation_exact_hash="exact-init"),
                SimpleNamespace(),
            ),
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_archive_state_for_exact_hash",
        lambda archive, exact_hash: AbstractState(),
    )
    output = tmp_path / "shadow"
    receipt = experiment.compile_option_shadow(
        Path("unused.json"),
        ablation_receipt_path=Path("unused-receipt.json"),
        output_dir=output,
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_4A_3_SHADOW_COMPILE_GATE"
    assert receipt["metrics"]["posterior_owner_mass"] == pytest.approx(1.0)
    assert receipt["metrics"]["compiled_program_count"] == 4
    assert receipt["metrics"]["checks"]["shadow_only_no_environment_calls"] is True
    assert (output / "posterior_snapshot.json").is_file()
