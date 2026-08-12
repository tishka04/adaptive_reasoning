from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import theory.sage_t.causal.experiment as experiment
from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal.contracts import (
    ActionInterventionSpec,
    BindingSpec,
    CausalProgram,
    CausalVariableSpec,
    GoalSpec,
    MechanismSpec,
    ParentRef,
)
from theory.sage_t.causal.experiment import (
    ArtifactBudgetExceeded,
    RunStorageBudget,
    freeze_experiment,
    load_experiment_manifest,
    load_receipt,
    run_experiment,
    run_replay,
    seal_bundle_plan,
    seal_program_registry,
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
    available_actions: tuple[int, ...] = (1, 2)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1), FakeAction(2)]


class FakeEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.levels = 0
        self.grid = np.zeros((7, 7), dtype=np.int32)
        self.grid[3, 3] = 2

    def step(self, action, data=None):
        del data
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.levels = 0
            self.grid.fill(0)
            self.grid[3, 3] = 2
            return FakeFrame(self.grid.copy())
        if value == 1:
            self.levels += 1
            self.grid[3, 3] = 3
            return FakeFrame(
                self.grid.copy(),
                state="WIN",
                levels_completed=self.levels,
            )
        self.grid[3, 4] = 4
        return FakeFrame(self.grid.copy(), levels_completed=self.levels)


def rival_program(program_id: str, *, learns_progress: bool) -> CausalProgram:
    mechanism = MechanismSpec(
        mechanism_id=f"{program_id}_level",
        output_variable="counter.levels_completed",
        parent_variables=(ParentRef("counter.levels_completed"),),
        operator_type="set" if learns_progress else "identity",
        parameters=(
            {"value": 1, "action_name": "ACTION1"}
            if learns_progress
            else {}
        ),
    )
    return CausalProgram(
        program_id=program_id,
        bindings=BindingSpec({}),
        variables=(
            CausalVariableSpec("counter.levels_completed", "counter", (0, 1)),
        ),
        mechanisms=(mechanism,),
        action_model=(
            ActionInterventionSpec("ACTION1"),
            ActionInterventionSpec("ACTION2"),
        ),
        goal=GoalSpec(
            "counter.levels_completed >= 1",
            ("counter.levels_completed >= 1",),
        ),
        description_length=1.0,
        provenance=("test:source",),
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def prepare_inputs(tmp_path: Path, *, games=("bp35",)):
    raw_registry = tmp_path / "raw_programs.json"
    registry = tmp_path / "programs.json"
    write_json(
        raw_registry,
        {
            "games": {
                game: {
                    "action_catalog": ["ACTION1", "ACTION2"],
                    "programs": [
                        rival_program(f"{game}_progress", learns_progress=True).to_dict(),
                        rival_program(f"{game}_identity", learns_progress=False).to_dict(),
                    ],
                }
                for game in games
            }
        },
    )
    seal_program_registry(raw_registry, registry)

    env = FakeEnv()
    env.step(0)
    prefix_frame = env.step(2)
    prefix_hash = state_signature_from_frame(prefix_frame)
    raw_plan = tmp_path / "raw_bundles.json"
    plan = tmp_path / "bundles.json"
    write_json(
        raw_plan,
        {
            "bundles": [
                {
                    "bundle_id": f"{game}-bundle",
                    "game_id": game,
                    "prefix_hash": prefix_hash,
                    "prefix": [{"action_name": "ACTION2"}],
                    "branches": [
                        {"action_name": "ACTION1"},
                        {"action_name": "ACTION2"},
                    ],
                }
                for game in games
            ]
        },
    )
    seal_bundle_plan(raw_plan, registry, plan)
    return registry, plan


def freeze_clean(monkeypatch, tmp_path, *, games=("bp35",), authority="shadow"):
    registry, plan = prepare_inputs(tmp_path, games=games)
    monkeypatch.setattr(
        experiment,
        "_git_state",
        lambda root: {"commit": "a" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest = tmp_path / "manifest.json"
    freeze_experiment(
        program_registry_path=registry,
        bundle_plan_path=plan,
        output_path=manifest,
        stage="source_train",
        game_ids=games,
        seeds=(7,),
        resets=2,
        action_budget_per_reset=1,
        authority=authority,
        root=Path(__file__).resolve().parents[1],
    )
    return manifest


def test_seal_and_freeze_bind_programs_bundles_code_and_protocol(monkeypatch, tmp_path):
    manifest_path = freeze_clean(monkeypatch, tmp_path)
    manifest = load_experiment_manifest(
        manifest_path,
        root=Path(__file__).resolve().parents[1],
    )
    assert manifest["stage"] == "source_train"
    assert manifest["scientific_claims_authorized"] is True
    assert manifest["program_registry"]["registry_checksum"]
    assert manifest["bundle_plan"]["plan_checksum"]
    assert manifest["code_sha256"]["theory/sage_t/causal/experiment_cli.py"]
    assert manifest["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert manifest["storage"]["hard_fail_before_write"] is True


def test_run_storage_budget_fails_before_crossing_limit(tmp_path):
    budget = RunStorageBudget(tmp_path, 8)
    budget.reserve(8)
    (tmp_path / "used.bin").write_bytes(b"12345678")
    with pytest.raises(ArtifactBudgetExceeded, match="would be exceeded"):
        budget.reserve(1)


def test_replay_preregisters_predictions_and_emits_bound_receipt(monkeypatch, tmp_path):
    manifest_path = freeze_clean(monkeypatch, tmp_path)
    output = tmp_path / "replay"
    report = run_replay(
        manifest_path=manifest_path,
        output_dir=output,
        env_factory=lambda game_id: FakeEnv(),
        root=Path(__file__).resolve().parents[1],
    )
    receipt = load_receipt(output / "replay_receipt.json")
    assert report["passed"] is True
    assert report["bundles"][0]["status"] == "BUNDLE_COMPLETE"
    assert report["bundles"][0]["predictions_registered_before_execution"] is True
    assert len(report["bundles"][0]["branches"]) == 2
    assert receipt["experiment_manifest_checksum"] == report[
        "experiment_manifest_checksum"
    ]
    assert receipt["report_checksum"] == report["report_checksum"]


def test_paired_runner_builds_fresh_rivals_and_a40_ablation(monkeypatch, tmp_path):
    manifest_path = freeze_clean(monkeypatch, tmp_path, authority="bounded")
    replay_output = tmp_path / "replay"
    run_replay(
        manifest_path=manifest_path,
        output_dir=replay_output,
        env_factory=lambda game_id: FakeEnv(),
        root=Path(__file__).resolve().parents[1],
    )
    paired_output = tmp_path / "paired"
    report = run_experiment(
        manifest_path=manifest_path,
        replay_receipt_path=replay_output / "replay_receipt.json",
        output_dir=paired_output,
        env_factory=lambda game_id: FakeEnv(),
        root=Path(__file__).resolve().parents[1],
    )
    condition = report["conditions"][0]
    assert condition["strict_prestate_pairing"] is True
    assert set(condition["arms"]) == set(experiment.DEFAULT_ARMS)
    for name, arm in condition["arms"].items():
        if name != "baseline":
            assert min(arm["metrics"]["initial_particle_counts"]) >= 2
    assert condition["arms"]["posterior_full"]["metrics"]["memory_records"] > 0
    assert condition["arms"]["no_a40_memory"]["metrics"]["memory_records"] == 0
    assert condition["arms"]["posterior_full"]["metrics"]["controller_errors"] == 0
    assert condition["arms"]["posterior_full"]["metrics"][
        "causal_pipeline_fallbacks"
    ] == 0
    receipt = load_receipt(paired_output / "gate_receipt.json")
    assert receipt["experiment_manifest_checksum"] == report[
        "experiment_manifest_checksum"
    ]
    assert report["holdout_opened"] is False
    assert report["production_authority"] is False


def test_source_validation_freeze_requires_passing_parent_receipt(tmp_path):
    registry, plan = prepare_inputs(tmp_path, games=("re86",))
    with pytest.raises(ValueError, match="source-train receipt"):
        freeze_experiment(
            program_registry_path=registry,
            bundle_plan_path=plan,
            output_path=tmp_path / "manifest.json",
            stage="source_validation",
            game_ids=("re86",),
            seeds=(0,),
            resets=1,
            action_budget_per_reset=1,
            allow_dirty=True,
        )


def test_freeze_refuses_dirty_tree_and_registry_tampering(monkeypatch, tmp_path):
    registry, plan = prepare_inputs(tmp_path)
    monkeypatch.setattr(
        experiment,
        "_git_state",
        lambda root: {"commit": "a" * 40, "dirty": True, "dirty_entries": 1},
    )
    with pytest.raises(ValueError, match="dirty tree"):
        freeze_experiment(
            program_registry_path=registry,
            bundle_plan_path=plan,
            output_path=tmp_path / "manifest.json",
            stage="source_train",
            game_ids=("bp35",),
            seeds=(0,),
            resets=1,
            action_budget_per_reset=1,
        )

    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["games"]["bp35"]["action_catalog"].append("ACTION9")
    registry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registry_checksum mismatch"):
        experiment.load_program_registry(registry)


def test_source_validation_manifest_binds_passing_source_receipt(monkeypatch, tmp_path):
    registry, plan = prepare_inputs(tmp_path, games=("re86",))
    parent = experiment._signed(
        {
            "format_version": experiment.RECEIPT_FORMAT,
            "kind": "paired_run",
            "stage": "source_train",
            "passed": True,
            "protocol_checksum": experiment.CausalProtocol().checksum,
            "experiment_manifest_checksum": "source-manifest",
            "report_checksum": "source-report",
            "metrics": {
                "games_with_progress": 1,
                "safety_regressions": 0,
                "posterior_ablation_advantage": True,
            },
            "reason": "PASS_CAUSAL_PAIRED_GATE",
        },
        "receipt_checksum",
    )
    parent_path = tmp_path / "source_receipt.json"
    write_json(parent_path, parent)
    monkeypatch.setattr(
        experiment,
        "_git_state",
        lambda root: {"commit": "b" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "validation_manifest.json"
    frozen = freeze_experiment(
        program_registry_path=registry,
        bundle_plan_path=plan,
        output_path=manifest_path,
        stage="source_validation",
        game_ids=("re86",),
        seeds=(2101,),
        resets=1,
        action_budget_per_reset=1,
        parent_receipt_path=parent_path,
        root=Path(__file__).resolve().parents[1],
    )
    loaded = load_experiment_manifest(
        manifest_path,
        root=Path(__file__).resolve().parents[1],
    )
    assert loaded["parent_receipt"]["receipt_checksum"] == parent[
        "receipt_checksum"
    ]
    assert frozen["stage"] == "source_validation"
