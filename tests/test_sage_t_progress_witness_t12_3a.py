from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import witness_experiment, witness_protocol
from theory.sage_t.causal.archive import GoExploreArchive
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.witness_experiment import (
    ReplayTrial,
    SdkCallBudget,
    _intervention_bundles,
    _metrics,
    run_witness_confirmation,
)
from theory.sage_t.causal.witness_experiment_cli import build_parser
from theory.sage_t.causal.witness_protocol import (
    ProgressWitness,
    WitnessConfirmProtocol,
    WitnessStep,
    common_action_suffix,
    extract_progress_witnesses,
    freeze_witness_experiment,
    load_witness_manifest,
    load_witness_receipt,
)
from theory.sage_t.contracts import AbstractState, GroundFact


@dataclass
class FakeAction:
    id: int
    data: dict | None = None


@dataclass
class FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1, 2, 3)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1), FakeAction(2), FakeAction(3)]


class CommonSuffixProgressEnv:
    """Two distinct first actions converge only after a third ACTION3."""

    def __init__(self) -> None:
        self._game = FakeGame()
        self.flavour = 0
        self.suffix_count = 0

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.flavour = 0
            self.suffix_count = 0
            return self._frame(0)
        if value in {1, 2} and self.suffix_count == 0:
            self.flavour = value
        elif value == 3 and self.flavour in {1, 2}:
            self.suffix_count += 1
        marker = 99 if self.suffix_count >= 3 else self.flavour * 10 + self.suffix_count
        return self._frame(marker)

    def _frame(self, marker: int) -> FakeFrame:
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = marker
        progress = self.suffix_count >= 3
        return FakeFrame(
            grid,
            state="WIN" if progress else "NOT_FINISHED",
            levels_completed=int(progress),
        )


class DivergentReplayEnv(CommonSuffixProgressEnv):
    def step(self, action, data=None):
        frame = super().step(action, data=data)
        value = int(getattr(action, "value", action))
        if value == 3:
            frame.frame[0, 0] = 7
        return frame


def _record_witness(first_action: int, *, seed: int, arm: str) -> ProgressWitness:
    env = CommonSuffixProgressEnv()
    frame = env.step(0)
    steps = []
    actions = (first_action, 3, 3, 3)
    for index, action_id in enumerate(actions):
        source_hash = state_signature_from_frame(frame)
        before_level = frame.levels_completed
        frame = env.step(action_id)
        steps.append(
            WitnessStep(
                expected_source_hash=source_hash,
                action=GroundedAction(f"ACTION{action_id}"),
                expected_target_hash=state_signature_from_frame(frame),
                level_delta=frame.levels_completed - before_level,
                terminal=False,
                success=frame.levels_completed > before_level,
            )
        )
    return ProgressWitness(
        witness_id=f"witness-{seed}",
        game_id="bp35",
        source_seed=seed,
        source_arm=arm,
        source_archive_sha256=str(seed) * 8,
        source_progress_edge_id=f"edge-{seed}",
        initial_exact_hash=steps[0].expected_source_hash,
        initial_level=0,
        target_exact_hash=steps[-1].expected_target_hash,
        target_level=1,
        steps=tuple(steps),
    )


def _state(index: int) -> AbstractState:
    return AbstractState(
        true_facts=frozenset({GroundFact("changed", (f"e{index}",))}),
        counters=(("progress", float(index)),),
    )


def _write_archive(
    path: Path,
    *,
    first_action: int,
    seed: int,
) -> dict[str, object]:
    archive = GoExploreArchive(maximum_cells=16, seed=seed)
    legal = tuple(GroundedAction(f"ACTION{value}") for value in (1, 2, 3))
    root, _ = archive.observe_state(
        state=_state(0),
        exact_hash="common-initial",
        level=0,
        legal_actions=legal,
    )
    source_cell = root
    source_hash = "common-initial"
    for index, action_id in enumerate((first_action, 3, 3, 3), start=1):
        target_hash = "common-target" if index == 4 else f"{seed}-state-{index}"
        target_state = _state(99 if index == 4 else seed + index)
        edge = archive.add_transition(
            source_cell_id=source_cell.cell_id,
            source_exact_hash=source_hash,
            action=GroundedAction(f"ACTION{action_id}"),
            target_state=target_state,
            target_exact_hash=target_hash,
            target_level=int(index == 4),
            target_legal_actions=legal,
            terminal=False,
            success=index == 4,
            changed=True,
        )
        source_cell = archive.cells[edge.target_cell_id]
        source_hash = target_hash
    path.write_text(
        witness_protocol._canonical(archive.to_dict()),
        encoding="utf-8",
    )
    return {
        "arm": "one_step_archive" if first_action == 1 else "burst_archive",
        "game_id": "bp35",
        "path": str(path),
        "seed": seed,
        "sha256": witness_protocol._file_sha256(path),
    }


def test_extracts_two_joint_witnesses_and_exact_common_suffix(tmp_path) -> None:
    artifacts = (
        _write_archive(tmp_path / "one.json", first_action=1, seed=6502),
        _write_archive(tmp_path / "burst.json", first_action=2, seed=6503),
    )
    witnesses = extract_progress_witnesses(artifacts)
    assert len(witnesses) == 2
    assert {len(item.steps) for item in witnesses} == {4}
    assert len({item.initial_exact_hash for item in witnesses}) == 1
    assert len({item.target_exact_hash for item in witnesses}) == 1
    assert [action.action_name for action in common_action_suffix(witnesses)] == [
        "ACTION3",
        "ACTION3",
        "ACTION3",
    ]


def test_sdk_budget_fails_before_overspend() -> None:
    budget = SdkCallBudget(2)
    budget.consume(2)
    with pytest.raises(RuntimeError, match="SDK call budget exceeded"):
        budget.consume()
    assert budget.used_calls == 2


def test_gate_requires_success_in_the_same_paired_repetitions() -> None:
    witness = _record_witness(1, seed=6502, arm="one_step_archive")
    expected_prefix = witness.steps[0].expected_target_hash

    def trial(trial_type: str, repetition: int, *, exact: bool) -> ReplayTrial:
        expected_progress = trial_type != "delete_last_suffix_action"
        return ReplayTrial(
            witness_id=witness.witness_id,
            trial_type=trial_type,
            repetition=repetition,
            exact=exact,
            initial_exact=True,
            observed_progress=expected_progress,
            expected_progress=expected_progress,
            final_exact_hash=witness.target_exact_hash,
            final_level=int(expected_progress),
            first_divergence="" if exact else "synthetic:branch",
            events=(
                {
                    "kind": "reset",
                    "observed_hash": witness.initial_exact_hash,
                    "exact": True,
                },
                {
                    "kind": "transition",
                    "phase": "prefix",
                    "observed_target_hash": expected_prefix,
                    "exact": True,
                },
            ),
        )

    trials = [trial("full_route", repetition, exact=True) for repetition in range(3)]
    trials.extend(
        trial("common_suffix", repetition, exact=repetition in {0, 2})
        for repetition in range(3)
    )
    trials.extend(
        trial(
            "delete_last_suffix_action",
            repetition,
            exact=repetition in {1, 2},
        )
        for repetition in range(3)
    )
    bundles = _intervention_bundles(
        witnesses=(witness,),
        trials=trials,
        suffix_length=3,
    )
    passed, metrics = _metrics(
        witnesses=(witness,),
        trials=trials,
        protocol=WitnessConfirmProtocol(),
        budget=SdkCallBudget(2_048),
        intervention_bundles=bundles,
    )
    item = metrics["per_witness"][0]
    assert item["suffix_confirmations"] == 2
    assert item["deletion_control_exact_no_progress"] == 2
    assert item["paired_contrast_confirmations"] == 1
    assert not passed


def test_frozen_confirmation_is_repeated_paired_and_bounded(
    monkeypatch, tmp_path
) -> None:
    repo = Path(__file__).resolve().parents[1]
    parent_root = repo / "training" / "sage_t" / "burst_go_explore_t12_2_bp35"
    witnesses = (
        _record_witness(1, seed=6502, arm="one_step_archive"),
        _record_witness(2, seed=6503, arm="burst_archive"),
    )
    monkeypatch.setattr(
        witness_protocol,
        "_git_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "dirty_entries": 0},
    )
    monkeypatch.setattr(
        witness_protocol,
        "extract_progress_witnesses",
        lambda artifacts: witnesses,
    )
    monkeypatch.setattr(witness_experiment, "_reset_env", lambda env: env.step(0))
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "witnesses.sealed.json"
    manifest = freeze_witness_experiment(
        output_path=manifest_path,
        witness_registry_path=registry_path,
        parent_manifest_path=parent_root / "manifest.json",
        parent_receipt_path=parent_root / "paired" / "burst_receipt.json",
        root=repo,
    )
    loaded = load_witness_manifest(manifest_path, root=repo, verify_code=False)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert loaded["firewall"]["terminal_shield_authorized"] is False

    output = tmp_path / "confirmation"
    report = run_witness_confirmation(
        manifest_path=manifest_path,
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: CommonSuffixProgressEnv(),
    )
    receipt = load_witness_receipt(
        output / "witness_receipt.json",
        manifest=loaded,
    )
    assert report["passed"]
    assert report["status"] == "PASS_T12_3A_WITNESS_GATE"
    assert report["storage"]["within_budget"]
    assert receipt["passed"]
    assert (output / "intervention_bundles.json").is_file()
    assert receipt["artifacts"]["intervention_bundles"]["sha256"]
    assert len(report["metrics"]["per_witness"]) == 2
    for item in report["metrics"]["per_witness"]:
        assert item["route_confirmations"] == 3
        assert item["suffix_confirmations"] == 3
        assert item["deletion_control_exact_no_progress"] == 3
        assert item["deletion_control_progresses"] == 0
        assert item["paired_prefix_exact"] == 3
        assert item["paired_contrast_confirmations"] == 3

    negative = run_witness_confirmation(
        manifest_path=manifest_path,
        output_dir=tmp_path / "divergent_confirmation",
        environments_dir="unused",
        env_factory=lambda game_id: DivergentReplayEnv(),
    )
    assert not negative["passed"]
    assert negative["status"] == "FAIL_T12_3A_WITNESS_GATE"
    assert negative["metrics"]["divergences"]

    (output / "intervention_bundles.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        load_witness_receipt(output / "witness_receipt.json", manifest=loaded)


def test_cli_exposes_only_freeze_run_and_status() -> None:
    parser = build_parser()
    assert (
        parser.parse_args(
            [
                "freeze",
                "--parent-manifest",
                "parent.json",
                "--parent-receipt",
                "receipt.json",
            ]
        ).phase
        == "freeze"
    )
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
