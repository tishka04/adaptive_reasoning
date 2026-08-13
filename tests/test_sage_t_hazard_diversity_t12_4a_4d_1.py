from __future__ import annotations

from types import SimpleNamespace

import pytest

from theory.sage_t.causal.archive import SymbolicArchiveCell
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.hazard_diversity_cli import build_parser
from theory.sage_t.causal import hazard_diversity_protocol as protocol_module
from theory.sage_t.causal.hazard_diversity_experiment import (
    _active_gate,
    _cross_fit,
    compile_hazard_diversity,
    hazard_diversity_status,
    run_hazard_diversity_arm,
)
from theory.sage_t.causal.hazard_diversity_model import (
    AbstractHazardModel,
    HazardObservation,
    StructuralActionDiversityPolicy,
    local_hazard_signature,
)
from theory.sage_t.causal.hazard_diversity_protocol import (
    HazardDiversityProtocol,
    freeze_hazard_diversity,
)
from theory.sage_t.causal.shield_model import ProgressProtectedTerminalShield
from theory.sage_t.contracts import AbstractEntity, AbstractState


def _state(*, row: float = 10.0, column: float = 20.0) -> AbstractState:
    return AbstractState(
        entities=(
            AbstractEntity(
                "local-a",
                ("object", "target", "unknown", "clickable"),
                attributes=(("area", "one"), ("aspect", "square")),
                center=(row, column),
            ),
            AbstractEntity(
                "local-b",
                ("object", "target", "unknown"),
                attributes=(("area", "large"), ("aspect", "wide")),
                center=(row + 3.0, column - 2.0),
            ),
        )
    )


def _observation(
    seed: int,
    *,
    terminal: bool,
    action_name: str = "ACTION6",
    x: int = 20,
    y: int = 10,
) -> HazardObservation:
    return HazardObservation(
        search_seed=seed,
        lineage_seed=8701 + seed % 2,
        source_exact_hash=f"state-{seed}-{terminal}-{action_name}",
        state=_state(),
        action=GroundedAction(action_name, {"x": x, "y": y}),
        terminal=terminal,
    )


def test_protocol_is_three_arm_bounded_and_uses_fresh_active_seeds() -> None:
    protocol = HazardDiversityProtocol()
    assert protocol.compile_search_seeds == (9101, 9102, 9103)
    assert protocol.active_search_seeds == (9201, 9202, 9203)
    assert not set(protocol.compile_search_seeds) & set(protocol.active_search_seeds)
    assert protocol.search_arms == (
        "local_archive_control",
        "diversity_control",
        "abstract_hazard_diversity",
    )
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    assert protocol.maximum_total_sdk_calls == 38_000
    with pytest.raises(ValueError, match="preregistered value changed"):
        HazardDiversityProtocol(local_hazard_radius=8)


def test_cli_has_freeze_compile_run_status_but_no_activation() -> None:
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
    assert parser.parse_args(["compile"]).phase == "compile"
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_local_hazard_signature_is_translation_and_identity_invariant() -> None:
    first = local_hazard_signature(
        _state(row=10.0, column=20.0),
        GroundedAction("ACTION6", {"x": 20, "y": 10}),
    )
    translated = AbstractState(
        entities=(
            AbstractEntity(
                "different-id-a",
                ("clickable", "target", "object", "unknown"),
                attributes=(("aspect", "square"), ("area", "one")),
                center=(30.0, 45.0),
            ),
            AbstractEntity(
                "different-id-b",
                ("target", "object", "unknown"),
                attributes=(("aspect", "wide"), ("area", "large")),
                center=(33.0, 43.0),
            ),
        )
    )
    second = local_hazard_signature(
        translated,
        GroundedAction("ACTION6", {"x": 45, "y": 30}),
    )
    assert first == second


def test_hazard_model_round_trip_and_veto() -> None:
    observations = tuple(
        _observation(seed, terminal=True)
        for seed in (9101, 9102, 9103)
        for _ in range(2)
    ) + tuple(
        _observation(seed, terminal=False, action_name="ACTION4")
        for seed in (9101, 9102, 9103)
    )
    model = AbstractHazardModel.fit(
        observations,
        radius=7,
        minimum_support=2,
        unsafe_rate_threshold=0.75,
    )
    loaded = AbstractHazardModel.from_dict(model.to_dict())
    state = _state()
    assert loaded.is_unsafe(
        state,
        GroundedAction("ACTION6", {"x": 20, "y": 10}),
    )
    assert not loaded.is_unsafe(
        state,
        GroundedAction("ACTION4", {"x": 20, "y": 10}),
    )


def test_cross_fit_is_seed_held_out() -> None:
    observations = []
    for seed in (9101, 9102, 9103):
        for index in range(4):
            observations.append(
                HazardObservation(
                    search_seed=seed,
                    lineage_seed=8701 + index % 2,
                    source_exact_hash=f"unsafe-{seed}-{index}",
                    state=_state(),
                    action=GroundedAction("ACTION6", {"x": 20, "y": 10}),
                    terminal=True,
                )
            )
            observations.append(
                HazardObservation(
                    search_seed=seed,
                    lineage_seed=8701 + index % 2,
                    source_exact_hash=f"safe-{seed}-{index}",
                    state=_state(),
                    action=GroundedAction("ACTION4", {"x": 20, "y": 10}),
                    terminal=False,
                )
            )
    result = _cross_fit(observations, protocol=HazardDiversityProtocol())
    assert [item["holdout_search_seed"] for item in result["folds"]] == [
        9101,
        9102,
        9103,
    ]
    assert result["micro_metrics"]["recall"] == 1.0
    assert result["micro_metrics"]["precision"] == 1.0


def test_real_parent_freeze_and_offline_compile_pass_before_active_authority(
    monkeypatch,
    tmp_path,
) -> None:
    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    parent = repo / "training" / "sage_t" / "target_regrounding_t12_4a_4d_bp35"
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "f" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    freeze_hazard_diversity(
        output_path=manifest_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "paired" / "target_regrounding_receipt.json",
        root=repo,
    )
    receipt = compile_hazard_diversity(
        manifest_path=manifest_path,
        output_dir=tmp_path / "compile",
    )
    assert receipt["status"] == "PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE"
    assert receipt["metrics"]["crossfit_micro"]["precision"] > 0.96
    assert receipt["metrics"]["crossfit_micro"]["recall"] > 0.51
    status = hazard_diversity_status(
        manifest_path=manifest_path,
        compile_receipt_path=tmp_path / "compile" / "compile_receipt.json",
    )
    assert status["firewall"]["hazard_diversity_active_run_authorized"] is True
    assert status["next_phase_authorized"] is False
    assert status["firewall"][
        "t12_4a_4e_option_extraction_freeze_authorized"
    ] is False


def test_diversity_policy_balances_action_families_and_applies_hazard() -> None:
    state = _state()
    candidates = (
        GroundedAction("ACTION3"),
        GroundedAction("ACTION4"),
        GroundedAction("ACTION6", {"x": 20, "y": 10}),
        GroundedAction("ACTION6", {"x": 23, "y": 13}),
    )
    cell = SymbolicArchiveCell(
        cell_id="cell-test",
        symbolic_signature=state.signature,
        level=1,
        legal_action_keys=tuple(item.key for item in candidates),
        state=state,
        action_attempts={item.key: 0 for item in candidates},
    )
    safe = AbstractHazardModel.fit(
        (), radius=7, minimum_support=2, unsafe_rate_threshold=0.75
    )
    policy = StructuralActionDiversityPolicy(seed=7)
    selected = []
    for _ in range(9):
        action = policy.choose(
            cell,
            candidates,
            static_shield=ProgressProtectedTerminalShield(),
            hazard_model=safe,
            novelty_scorer=None,
        )
        assert action is not None
        selected.append(action.action_name)
        cell.action_attempts[action.key] += 1
    counts = {name: selected.count(name) for name in set(selected)}
    assert set(counts) == {"ACTION3", "ACTION4", "ACTION6"}
    assert max(counts.values()) / len(selected) <= 0.5


def test_active_gate_requires_exercised_hazard_and_safe_diverse_treatment() -> None:
    protocol = HazardDiversityProtocol()

    def metrics(arm: str) -> dict:
        return {
            "abstract_hazard_vetoes": 2 if arm == "abstract_hazard_diversity" else 0,
            "action_family_counts": {"ACTION3": 3, "ACTION4": 3, "ACTION6": 4},
            "candidate_catalog_checksum": "same",
            "entry_exact": True,
            "exploration_actions": 10,
            "materialized_option_actions": 0,
            "option_applicable_mass": 0.0,
            "progress_edges": 1 if arm == "abstract_hazard_diversity" else 0,
            "replay_exact_rate": 1.0,
            "sdk_calls": 100,
            "terminal_edges": 1 if arm == "abstract_hazard_diversity" else 2,
        }

    conditions = [
        {
            "arms": {
                arm: {"metrics": metrics(arm)} for arm in protocol.search_arms
            }
        }
    ]
    confirmations = [
        {
            "available": True,
            "final_exact_hash": "same-final",
            "prefix_exact": True,
            "progressed": True,
            "terminal_failure": False,
        }
        for _ in range(4)
    ]
    candidate = SimpleNamespace(progress_suffix=(GroundedAction("ACTION6"),))
    passed, guidance, result = _active_gate(
        protocol=protocol,
        conditions=conditions,
        confirmation_trials=confirmations,
        candidate=candidate,
        total_sdk_calls=300,
    )
    assert passed is True
    assert guidance is True
    assert all(result["checks"].values())


def test_all_three_runtime_arms_reach_synthetic_progress() -> None:
    from tests.test_sage_t_target_regrounding_t12_4a_4d import (
        RegroundingEnv,
        _registry_and_posterior,
        _witness,
    )

    registry, posterior = _registry_and_posterior()
    hazard_model = AbstractHazardModel.fit(
        (), radius=7, minimum_support=2, unsafe_rate_threshold=0.75
    )
    shares = {}
    for arm in HazardDiversityProtocol().search_arms:
        run = run_hazard_diversity_arm(
            game_id="bp35",
            witness=_witness(1, 8701),
            registry=registry,
            posterior=posterior,
            shield=ProgressProtectedTerminalShield(),
            hazard_model=hazard_model,
            arm=arm,
            search_seed=9201,
            sdk_call_budget=128,
            maximum_excursions=64,
            maximum_cells=1_000,
            burst_schedule=(4, 8, 16),
            environments_dir="unused",
            env_factory=lambda game_id: RegroundingEnv(),
        )
        assert run.progress_edge_id is not None
        shares[arm] = run.metrics()["maximum_action_family_share"]
    assert shares["diversity_control"] <= 0.5
    assert shares["abstract_hazard_diversity"] <= 0.5
