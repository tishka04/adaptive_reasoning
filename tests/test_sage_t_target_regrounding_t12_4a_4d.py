from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import target_regrounding_experiment as experiment
from theory.sage_t.causal import target_regrounding_protocol as protocol_module
from theory.sage_t.causal.contracts import GroundedAction, causal_program_from_dict
from theory.sage_t.causal.option_contracts import ContractedCausalOption
from theory.sage_t.causal.shield_model import ProgressProtectedTerminalShield
from theory.sage_t.causal.target_regrounding_cli import build_parser
from theory.sage_t.causal.target_regrounding_protocol import (
    TargetRegroundingProtocol,
    freeze_target_regrounding,
    load_target_regrounding_manifest,
)
from theory.sage_t.causal.witness_protocol import ProgressWitness, WitnessStep
from theory.sage_t.contracts import AbstractEntity, AbstractState


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "option_contract_t12_4a_4c_bp35"


def _registry_and_posterior() -> tuple[ContractedCausalOption, object]:
    root = _parent() / "contract"
    registry = ContractedCausalOption.from_dict(
        __import__("json").loads(
            (root / "contracted_option_registry.json").read_text(encoding="utf-8")
        )
    )
    programs_payload = __import__("json").loads(
        (root / "contracted_option_programs.json").read_text(encoding="utf-8")
    )
    programs = {
        item.canonical_hash: item
        for item in (
            causal_program_from_dict(dict(row))
            for row in programs_payload["programs"]
        )
    }
    snapshot = __import__("json").loads(
        (root / "contracted_posterior_snapshot.json").read_text(encoding="utf-8")
    )["posterior"]
    posterior = SimpleNamespace(
        particles=tuple(
            SimpleNamespace(
                probability=float(row["probability"]),
                program=programs[str(row["program_hash"])],
            )
            for row in snapshot["particles"]
        )
    )
    return registry, posterior


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
    def __init__(self, env: "RegroundingEnv") -> None:
        self.env = env

    def _get_valid_actions(self):
        values = (1, 2) if self.env.level == 0 else (3, 4, 6, 7)
        return [FakeAction(value) for value in values]


class RegroundingEnv:
    """Two exact routes share level 1; ACTION6/7 in either order reaches level 2."""

    def __init__(self) -> None:
        self.level = 0
        self.history: list[int] = []
        self._game = FakeGame(self)

    def step(self, action, data=None):
        del data
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.level = 0
            self.history = []
        elif self.level == 0 and value in {1, 2}:
            self.level = 1
            self.history = []
        elif self.level == 1:
            self.history.append(value)
            if len(self.history) >= 2 and set(self.history[-2:]) == {6, 7}:
                self.level = 2
                self.history = []
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[0, 0] = self.level
        for index, item in enumerate(self.history[-4:]):
            grid[1, index] = item
        available = (1, 2) if self.level == 0 else (3, 4, 6, 7)
        return FakeFrame(
            grid,
            levels_completed=self.level,
            available_actions=available,
        )


def _witness(action: int, seed: int) -> ProgressWitness:
    env = RegroundingEnv()
    initial = env.step(0)
    source_hash = state_signature_from_frame(initial)
    target = env.step(action)
    target_hash = state_signature_from_frame(target)
    step = WitnessStep(
        expected_source_hash=source_hash,
        action=GroundedAction(f"ACTION{action}"),
        expected_target_hash=target_hash,
        level_delta=1,
        success=True,
    )
    return ProgressWitness(
        witness_id=f"synthetic-{seed}",
        game_id="bp35",
        source_seed=seed,
        source_arm="synthetic_exact_route",
        source_archive_sha256=str(seed) * 8,
        source_progress_edge_id=f"edge-{seed}",
        initial_exact_hash=source_hash,
        initial_level=0,
        target_exact_hash=target_hash,
        target_level=1,
        steps=(step,),
    )


def test_protocol_is_strictly_paired_bounded_and_has_no_activation_phase() -> None:
    protocol = TargetRegroundingProtocol()
    assert protocol.search_seeds == (9101, 9102, 9103)
    assert protocol.source_lineages == (8701, 8705)
    assert protocol.search_arms == (
        "local_archive_control",
        "contract_regrounded",
    )
    assert protocol.sdk_calls_per_search_arm == 2048
    assert protocol.maximum_total_sdk_calls == 26_000
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
            "--witness-registry",
            "witnesses.json",
            "--shield-manifest",
            "shield-manifest.json",
            "--shield-receipt",
            "shield-receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_freeze_binds_passed_contract_two_routes_and_terminal_shield(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_target_regrounding(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "contract" / "option_contract_receipt.json",
        witness_registry_path=(
            _repo()
            / "training"
            / "sage_t"
            / "witness_reconfirmation_t12_4a_2_bp35"
            / "witnesses.sealed.json"
        ),
        shield_manifest_path=(
            _repo()
            / "training"
            / "sage_t"
            / "lineage_shield_t12_3e_bp35"
            / "manifest.json"
        ),
        shield_receipt_path=(
            _repo()
            / "training"
            / "sage_t"
            / "lineage_shield_t12_3e_bp35"
            / "paired"
            / "lineage_shield_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_target_regrounding_manifest(
        manifest_path,
        root=_repo(),
        verify_code=False,
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["inputs"]["route_lengths"] == [64, 61]
    assert loaded["inputs"]["entry_level"] == 1
    assert loaded["safety_parent"]["receipt"]["status"] == (
        "PASS_T12_3E_LINEAGE_SHIELD_GATE"
    )
    assert loaded["firewall"]["target_regrounding_experiment_authorized"] is True
    assert loaded["firewall"]["option_control_authorized"] is False
    assert loaded["firewall"][
        "t12_4a_4e_option_extraction_freeze_authorized"
    ] is False


def test_contract_scorer_prefers_unseen_role_grounded_probe() -> None:
    registry, _ = _registry_and_posterior()
    scorer = experiment.ContractRegroundingScorer(registry)
    state = AbstractState(
        entities=(
            AbstractEntity("e1", ("object", "movable"), center=(4.0, 5.0)),
        )
    )
    old = scorer.score(state, GroundedAction("ACTION4"))
    local = scorer.score(
        state,
        GroundedAction("ACTION6", {"x": 4, "y": 5}),
    )
    assert local[0] > old[0]
    assert local[1] > old[1]


def test_paired_search_confirms_new_cross_lineage_witness_and_stays_shadow(
    monkeypatch,
    tmp_path,
) -> None:
    protocol = TargetRegroundingProtocol()
    registry, posterior = _registry_and_posterior()
    witnesses = (_witness(1, 8701), _witness(2, 8705))
    manifest = {
        "firewall": {"target_regrounding_experiment_authorized": True},
        "game_id": "bp35",
        "manifest_checksum": "m" * 64,
        "parent": {
            "receipt": {
                "receipt_checksum": "r" * 64,
                "status": "PASS_T12_4A_4C_OPTION_CONTRACT_GATE",
            }
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
        "scientific_claims_authorized": True,
    }
    monkeypatch.setattr(
        experiment,
        "load_target_regrounding_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_load_frozen_inputs",
        lambda *args, **kwargs: (
            witnesses,
            registry,
            posterior,
            ProgressProtectedTerminalShield(),
        ),
    )
    receipt = experiment.run_target_regrounding_experiment(
        manifest_path="unused.json",
        output_dir=tmp_path / "paired",
        environments_dir="unused",
        env_factory=lambda game_id: RegroundingEnv(),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_4A_4D_TARGET_WITNESS_GATE"
    metrics = receipt["metrics"]
    assert metrics["discovered_witness_count"] == 2
    assert metrics["confirmation_count"] == 4
    assert metrics["confirmation_exact_rate"] == 1.0
    assert metrics["final_hash_count"] == 1
    assert metrics["sdk_calls_used"] <= 26_000
    assert metrics["storage"]["within_budget"] is True
    assert all(metrics["checks"].values())
    assert (tmp_path / "paired" / "progress_witnesses.sealed.json").is_file()
    assert (tmp_path / "paired" / "intervention_bundles.json").is_file()
