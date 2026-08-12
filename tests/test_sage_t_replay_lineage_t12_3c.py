from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage_t.causal import lineage_protocol
from theory.sage_t.causal.archive import ROOT_PREFIX_ID, GoExploreArchive
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.lineage_archive import LineagePreservingArchive
from theory.sage_t.causal.lineage_experiment import (
    ReplayAuditTrial,
    _aggregate_gate,
    run_lineage_burst_arm,
)
from theory.sage_t.causal.lineage_experiment_cli import build_parser
from theory.sage_t.causal.lineage_protocol import (
    ReplayLineageProtocol,
    extract_replay_audit_cases,
)
from theory.sage_t.contracts import AbstractState, GroundFact


def state(index: int) -> AbstractState:
    return AbstractState(
        true_facts=frozenset({GroundFact("changed", (str(index),))}),
        counters=(("index", float(index)),),
    )


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


class FourStepProgressEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.count = 0

    def step(self, action, data=None):
        del data
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.count = 0
            return FakeFrame(np.zeros((3, 3), dtype=np.int32))
        self.count += 1
        level = int(self.count >= 4)
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = self.count
        return FakeFrame(
            grid,
            state="WIN" if level else "NOT_FINISHED",
            levels_completed=level,
        )


def _build_alias_archive(*, preserve_lineage: bool):
    actions = tuple(GroundedAction(name) for name in ("A", "B", "C", "D"))
    archive = (
        LineagePreservingArchive(maximum_cells=16)
        if preserve_lineage
        else GoExploreArchive(maximum_cells=16)
    )
    root, _ = archive.observe_state(
        state=state(0),
        exact_hash="h0",
        level=0,
        legal_actions=actions,
    )

    def add(
        source_cell_id,
        source_hash,
        action,
        target_state,
        target_hash,
        *,
        prefix_id=None,
        path=(),
    ):
        common = {
            "source_cell_id": source_cell_id,
            "source_exact_hash": source_hash,
            "action": action,
            "target_state": target_state,
            "target_exact_hash": target_hash,
            "target_level": 0,
            "target_legal_actions": actions,
            "terminal": False,
            "success": False,
            "changed": source_hash != target_hash,
        }
        if preserve_lineage:
            assert isinstance(archive, LineagePreservingArchive)
            return archive.add_lineage_transition(
                **common,
                source_prefix_id=prefix_id or ROOT_PREFIX_ID,
                source_path_edge_ids=path,
            )
        return archive.add_transition(**common)

    short = add(root.cell_id, "h0", actions[0], state(1), "h1")
    branch = add(root.cell_id, "h0", actions[1], state(2), "h2")
    state2 = archive.cells[branch.target_cell_id]
    alias = add(
        state2.cell_id,
        "h2",
        actions[2],
        state(1),
        "h1",
        prefix_id=branch.prefix_id,
        path=(branch.edge_id,),
    )
    state1 = archive.cells[short.target_cell_id]
    target = add(
        state1.cell_id,
        "h1",
        actions[3],
        state(3),
        "h3",
        prefix_id=alias.prefix_id,
        path=(branch.edge_id, alias.edge_id),
    )
    target_variant = archive.cells[target.target_cell_id].variants["h3"]
    return archive, tuple(
        item.action_name for item in archive.prefixes.actions(target_variant.prefix_id)
    )


def test_lineage_archive_does_not_rebase_an_observed_transition() -> None:
    control, control_actions = _build_alias_archive(preserve_lineage=False)
    treatment, treatment_actions = _build_alias_archive(preserve_lineage=True)
    assert control_actions == ("A", "D")
    assert treatment_actions == ("B", "C", "D")
    assert treatment.metrics()["shortest_prefix_rebases_avoided"] == 1
    assert treatment.metrics()["lineage_rebased_transitions"] == 0
    assert control.metrics()["edges"] == treatment.metrics()["edges"] == 4


def test_lineage_runner_executes_a_complete_synthetic_burst() -> None:
    run = run_lineage_burst_arm(
        game_id="bp35",
        seed=7101,
        sdk_call_budget=12,
        burst_schedule=(4, 8, 16),
        preserve_lineage=True,
        environments_dir="unused",
        env_factory=lambda game_id: FourStepProgressEnv(),
        maximum_cells=32,
    )
    metrics = run.metrics()
    assert metrics["progress_edges"] >= 1
    assert metrics["replay_exact_rate"] == 1.0
    assert metrics["lineage_attached_transitions"] >= 4
    assert metrics["lineage_rebased_transitions"] == 0


def test_real_failed_parent_archives_form_a_bounded_balanced_audit() -> None:
    repo = Path(__file__).resolve().parents[1]
    evaluation = lineage_protocol._read_json(
        repo
        / "training"
        / "sage_t"
        / "terminal_shield_t12_3b_bp35"
        / "paired"
        / "paired_evaluation.json"
    )
    protocol = ReplayLineageProtocol()
    cases = extract_replay_audit_cases(evaluation, protocol=protocol)
    failed = [item for item in cases if item.case_kind == "failed"]
    controls = [item for item in cases if item.case_kind == "matched_control"]
    assert len(cases) <= protocol.maximum_audit_cases
    assert len(failed) >= protocol.minimum_failed_audit_cases
    assert controls
    assert {item.source_arm for item in failed} == {
        "burst_control",
        "burst_terminal_shield",
    }
    assert max(item.depth for item in cases) <= protocol.maximum_audit_depth


def _arm(*, exact: float, cells: float, progress: int, treatment: bool):
    metrics = {
        "replay_exact_rate": exact,
        "symbolic_cells_per_1000_sdk_calls": cells,
        "progress_edges": progress,
        "sdk_calls": 100,
    }
    if treatment:
        metrics.update(
            {
                "lineage_attached_transitions": 50,
                "shortest_prefix_rebases_avoided": 2,
                "lineage_rebased_transitions": 0,
            }
        )
    return {"metrics": metrics}


def test_gate_requires_reproduced_failure_and_prospective_exact_replay() -> None:
    protocol = ReplayLineageProtocol()
    audit = (
        ReplayAuditTrial(
            case_id="failed",
            case_kind="failed",
            repetition=0,
            exact=False,
            calls=10,
            first_divergence_step=8,
            first_divergence_kind="state_hash",
            expected_hash="expected",
            observed_hash="observed",
            events=(),
        ),
        ReplayAuditTrial(
            case_id="control",
            case_kind="matched_control",
            repetition=0,
            exact=True,
            calls=10,
            first_divergence_step=None,
            first_divergence_kind="",
            expected_hash="expected",
            observed_hash="expected",
            events=(),
        ),
    )
    conditions = tuple(
        {
            "seed": seed,
            "arms": {
                "shortest_prefix_control": _arm(
                    exact=0.90 if seed == 6803 else 1.0,
                    cells=100.0,
                    progress=0,
                    treatment=False,
                ),
                "lineage_preserving": _arm(
                    exact=1.0,
                    cells=90.0,
                    progress=0,
                    treatment=True,
                ),
            },
        }
        for seed in protocol.evaluation_seeds
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        audit_trials=audit,
        conditions=conditions,
        sdk_calls=1_000,
    )
    assert passed
    assert metrics["calibration_seed_replay_gain"] == pytest.approx(0.1)

    broken = list(conditions)
    broken[-1] = {
        **broken[-1],
        "arms": {
            **broken[-1]["arms"],
            "lineage_preserving": _arm(
                exact=0.94,
                cells=90.0,
                progress=0,
                treatment=True,
            ),
        },
    }
    assert not _aggregate_gate(
        protocol=protocol,
        audit_trials=audit,
        conditions=broken,
        sdk_calls=1_000,
    )[0]


def test_cli_exposes_only_freeze_run_and_status() -> None:
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
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["neural"])


def test_freeze_accepts_only_the_replay_only_failed_t12_3b_parent(
    monkeypatch, tmp_path
) -> None:
    repo = Path(__file__).resolve().parents[1]
    parent = repo / "training" / "sage_t" / "terminal_shield_t12_3b_bp35"
    monkeypatch.setattr(
        lineage_protocol,
        "_git_state",
        lambda root: {"commit": "f" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "replay_audit.sealed.json"
    manifest = lineage_protocol.freeze_lineage_experiment(
        output_path=manifest_path,
        audit_registry_path=registry_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "paired" / "shield_receipt.json",
        root=repo,
    )
    loaded = lineage_protocol.load_lineage_manifest(
        manifest_path, root=repo, verify_code=False
    )
    _, cases = lineage_protocol.load_lineage_registry(registry_path)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["failure_class"] == "REPLAY_EXACT_ONLY"
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert loaded["firewall"]["neural_training_authorized"] is False
    assert len(cases) <= ReplayLineageProtocol().maximum_audit_cases
