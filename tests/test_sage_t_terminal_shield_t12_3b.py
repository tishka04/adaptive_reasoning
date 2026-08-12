from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage_t.causal import shield_protocol
from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal.archive import ArchiveEdge
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.shield_experiment import (
    TerminalConfirmation,
    WitnessShieldTrial,
    _aggregate_gate,
    build_progress_protected_shield,
    confirm_terminal_candidate,
    run_shielded_burst_arm,
)
from theory.sage_t.causal.shield_experiment_cli import build_parser
from theory.sage_t.causal.shield_model import ProgressProtectedTerminalShield
from theory.sage_t.causal.shield_protocol import (
    ProtectedActionSpec,
    TerminalShieldProtocol,
    TerminalTraceCandidate,
    extract_protected_actions,
    extract_terminal_candidates,
)
from theory.sage_t.causal.terminal_shield import MultiStepTerminalShield
from theory.sage_t.causal.witness_protocol import load_witness_registry


@dataclass
class FakeAction:
    id: int
    data: dict | None = None


@dataclass
class FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1,)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1)]


class ThreeStepTerminalEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.count = 0

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.count = 0
        else:
            self.count += 1
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = self.count
        return FakeFrame(
            grid,
            state="GAME_OVER" if self.count >= 3 else "NOT_FINISHED",
        )


class DenyAllShield:
    def __init__(self) -> None:
        self.vetoes = 0

    def allows(self, cell_id, action):
        del cell_id, action
        self.vetoes += 1
        return False


def _edge(
    index: int,
    *,
    source_hash: str,
    target_hash: str,
    terminal: bool = False,
) -> ArchiveEdge:
    return ArchiveEdge(
        edge_id=f"edge-{index}",
        ordinal=index,
        source_cell_id=f"cell-{index}",
        source_exact_hash=source_hash,
        action=GroundedAction("ACTION1"),
        target_cell_id=f"cell-{index + 1}",
        target_exact_hash=target_hash,
        level_delta=0,
        terminal=terminal,
        success=False,
        changed=True,
        novel=True,
        prefix_id=f"prefix-{index}",
    )


def _terminal_candidate() -> TerminalTraceCandidate:
    env = ThreeStepTerminalEnv()
    frame = env.step(0)
    edges = []
    initial_hash = state_signature_from_frame(frame)
    for index in range(3):
        source_hash = state_signature_from_frame(frame)
        frame = env.step(1)
        edges.append(
            _edge(
                index,
                source_hash=source_hash,
                target_hash=state_signature_from_frame(frame),
                terminal=index == 2,
            )
        )
    return TerminalTraceCandidate(
        candidate_id="terminal-test",
        game_id="bp35",
        source_seed=6501,
        source_arm="one_step_archive",
        source_archive_sha256="a" * 64,
        terminal_edge_id=edges[-1].edge_id,
        initial_exact_hash=initial_hash,
        initial_level=0,
        terminal_source_level=0,
        edges=tuple(edges),
    )


def test_real_t12_2_artifacts_produce_balanced_terminal_and_progress_sets() -> None:
    repo = Path(__file__).resolve().parents[1]
    t12_2 = repo / "training" / "sage_t" / "burst_go_explore_t12_2_bp35"
    t12_3a = repo / "training" / "sage_t" / "progress_witness_t12_3a_bp35"
    receipt = json.loads(
        (t12_2 / "paired" / "burst_receipt.json").read_text(encoding="utf-8")
    )
    _, witnesses = load_witness_registry(t12_3a / "witnesses.sealed.json")
    protocol = TerminalShieldProtocol()
    candidates = extract_terminal_candidates(
        receipt["artifacts"]["archives"],
        protocol=protocol,
    )
    protected = extract_protected_actions(
        witnesses,
        receipt["artifacts"]["archives"],
    )
    assert len(candidates) == 12
    assert len({(item.source_seed, item.source_arm) for item in candidates}) == 6
    assert all(len(item.edges) <= 64 for item in candidates)
    assert len(protected) == 99
    assert {witness_id for item in protected for witness_id in item.witness_ids} == {
        item.witness_id for item in witnesses
    }


def test_terminal_confirmation_checks_every_exact_hash(monkeypatch) -> None:
    candidate = _terminal_candidate()
    monkeypatch.setattr(
        "theory.sage_t.causal.shield_experiment._reset_env",
        lambda env: env.step(0),
    )
    confirmation = confirm_terminal_candidate(
        candidate=candidate,
        repetition=0,
        environments_dir="unused",
        env_factory=lambda game_id: ThreeStepTerminalEnv(),
    )
    assert confirmation.confirmed
    assert confirmation.calls == 4
    assert len(confirmation.events) == 4


def test_progress_protection_overrides_confirmed_terminal_conflict() -> None:
    candidate = _terminal_candidate()
    confirmation = TerminalConfirmation(
        candidate_id=candidate.candidate_id,
        repetition=0,
        exact=True,
        terminal_failure=True,
        confirmed=True,
        calls=4,
        final_exact_hash=candidate.edges[-1].target_exact_hash,
        first_divergence="",
        events=(),
    )
    protected = ProtectedActionSpec(
        cell_id=candidate.edges[0].source_cell_id,
        action_key=candidate.edges[0].action.key,
        witness_ids=("witness-progress",),
    )
    shield, confirmed_ids = build_progress_protected_shield(
        candidates=(candidate,),
        confirmations=(confirmation,),
        protected_actions=(protected,),
        protocol=TerminalShieldProtocol(),
    )
    assert confirmed_ids == (candidate.candidate_id,)
    assert shield.base.support(
        candidate.edges[0].source_cell_id,
        candidate.edges[0].action,
    ).confirmed_unsafe
    assert shield.allows(
        candidate.edges[0].source_cell_id,
        candidate.edges[0].action,
    )
    assert not shield.allows(
        candidate.edges[1].source_cell_id,
        candidate.edges[1].action,
    )
    assert shield.metrics()["protected_conflict_overrides"] == 1


def test_shielded_burst_runner_applies_veto_before_real_action(monkeypatch) -> None:
    monkeypatch.setattr(
        "theory.sage_t.causal.shield_experiment._reset_env",
        lambda env: env.step(0),
    )
    control, _ = run_shielded_burst_arm(
        game_id="bp35",
        seed=6801,
        sdk_call_budget=20,
        burst_schedule=(4, 8, 16),
        environments_dir="unused",
        env_factory=lambda game_id: ThreeStepTerminalEnv(),
    )
    deny = DenyAllShield()
    treatment, used = run_shielded_burst_arm(
        game_id="bp35",
        seed=6801,
        sdk_call_budget=20,
        burst_schedule=(4, 8, 16),
        environments_dir="unused",
        env_factory=lambda game_id: ThreeStepTerminalEnv(),
        shield=deny,  # type: ignore[arg-type]
    )
    assert control.archive.metrics()["terminal_edges"] >= 1
    assert treatment.archive.metrics()["terminal_edges"] == 0
    assert treatment.archive.metrics()["edges"] == 0
    assert used is deny
    assert deny.vetoes >= 1


def _arm(*, terminal: int, edges: int, cells: int, progress: int, exact: float):
    return {
        "metrics": {
            "terminal_edges": terminal,
            "exploration_actions": edges,
            "symbolic_cells": cells,
            "sdk_calls": 100,
            "progress_edges": progress,
            "replay_exact_rate": exact,
            "symbolic_cells_per_1000_sdk_calls": cells * 10.0,
        },
        "shield_metrics": {"vetoes": 3},
    }


def test_gate_requires_terminal_gain_coverage_and_witness_non_regression() -> None:
    protocol = TerminalShieldProtocol()
    candidate = _terminal_candidate()
    candidates = tuple(
        TerminalTraceCandidate(
            candidate_id=f"terminal-{index}",
            game_id=candidate.game_id,
            source_seed=protocol.source_seeds[index % 3],
            source_arm=protocol.source_arms[(index // 3) % 2],
            source_archive_sha256=candidate.source_archive_sha256,
            terminal_edge_id=candidate.terminal_edge_id,
            initial_exact_hash=candidate.initial_exact_hash,
            initial_level=0,
            terminal_source_level=0,
            edges=candidate.edges,
        )
        for index in range(12)
    )
    confirmations = tuple(
        TerminalConfirmation(
            candidate_id=item.candidate_id,
            repetition=0,
            exact=True,
            terminal_failure=True,
            confirmed=True,
            calls=4,
            final_exact_hash=item.edges[-1].target_exact_hash,
            first_divergence="",
            events=(),
        )
        for item in candidates
    )
    base = MultiStepTerminalShield(horizon=64, minimum_support=2)
    base.record_terminal_trace(candidate.edges, exact_replay_confirmed=True)
    shield = ProgressProtectedTerminalShield(base=base)
    witness_trials = tuple(
        WitnessShieldTrial(
            witness_id=f"witness-{index // 3}",
            repetition=index % 3,
            exact=True,
            progressed=True,
            all_actions_protected=True,
            vetoed_actions=0,
            calls=4,
            final_exact_hash="target",
            first_divergence="",
            events=(),
        )
        for index in range(6)
    )
    conditions = tuple(
        {
            "seed": seed,
            "arms": {
                "burst_control": _arm(
                    terminal=20,
                    edges=100,
                    cells=100,
                    progress=1,
                    exact=1.0,
                ),
                "burst_terminal_shield": _arm(
                    terminal=10,
                    edges=100,
                    cells=90,
                    progress=1,
                    exact=1.0,
                ),
            },
        }
        for seed in protocol.evaluation_seeds
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        confirmations=confirmations,
        candidates=candidates,
        shield=shield,
        witness_trials=witness_trials,
        conditions=conditions,
        sdk_calls=1_000,
    )
    assert passed
    assert metrics["terminal_rate_ratio"] == 0.5
    assert metrics["coverage_ratio"] == 0.9

    regressed = list(conditions)
    regressed[0] = {
        **regressed[0],
        "arms": {
            **regressed[0]["arms"],
            "burst_terminal_shield": _arm(
                terminal=10,
                edges=100,
                cells=90,
                progress=0,
                exact=1.0,
            ),
        },
    }
    assert not _aggregate_gate(
        protocol=protocol,
        confirmations=confirmations,
        candidates=candidates,
        shield=shield,
        witness_trials=witness_trials,
        conditions=regressed,
        sdk_calls=1_000,
    )[0]


def test_cli_exposes_freeze_run_and_status_only() -> None:
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
    with pytest.raises(SystemExit):
        parser.parse_args(["neural"])


def test_freeze_is_bound_to_passed_t12_3a_and_three_gib(monkeypatch, tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    parent = repo / "training" / "sage_t" / "progress_witness_t12_3a_bp35"
    monkeypatch.setattr(
        shield_protocol,
        "_git_state",
        lambda root: {"commit": "e" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "terminal_candidates.sealed.json"
    manifest = shield_protocol.freeze_shield_experiment(
        output_path=manifest_path,
        terminal_registry_path=registry_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "confirmation" / "witness_receipt.json",
        root=repo,
    )
    loaded = shield_protocol.load_shield_manifest(
        manifest_path,
        root=repo,
        verify_code=False,
    )
    _, candidates, protected = shield_protocol.load_shield_registry(registry_path)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["passed"] is True
    assert loaded["parent"]["receipt"]["status"] == "PASS_T12_3A_WITNESS_GATE"
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert loaded["firewall"]["neural_training_authorized"] is False
    assert len(candidates) == 12
    assert len(protected) == 99

    archive_path = tmp_path / "archive.json"
    excursion_path = tmp_path / "excursions.json"
    archive_path.write_text("{}\n", encoding="utf-8")
    excursion_path.write_text("{}\n", encoding="utf-8")
    paired_path = tmp_path / "paired_evaluation.json"
    paired_path.write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "arms": {
                            "burst_control": {
                                "archive": {
                                    "path": str(archive_path),
                                    "sha256": shield_protocol._file_sha256(
                                        archive_path
                                    ),
                                },
                                "excursions": {
                                    "path": str(excursion_path),
                                    "sha256": shield_protocol._file_sha256(
                                        excursion_path
                                    ),
                                },
                            }
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    receipt = shield_protocol.shield_phase_receipt(
        manifest=loaded,
        phase="terminal_shield",
        passed=False,
        status="FAIL_T12_3B_TERMINAL_SHIELD_GATE",
        metrics={},
        artifacts={
            "paired_evaluation": {
                "path": str(paired_path),
                "sha256": shield_protocol._file_sha256(paired_path),
            }
        },
    )
    receipt_path = tmp_path / "shield_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    shield_protocol.load_shield_receipt(receipt_path, manifest=loaded, root=repo)
    archive_path.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="paired artifact checksum mismatch"):
        shield_protocol.load_shield_receipt(
            receipt_path,
            manifest=loaded,
            root=repo,
        )
