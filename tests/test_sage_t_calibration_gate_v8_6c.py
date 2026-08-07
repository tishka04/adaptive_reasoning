from __future__ import annotations

import json
from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6c import (
    CHALLENGERS,
    DEFAULT_ACTION_SCHEDULES,
    _family_runner,
    _new_family_posterior,
    freeze_manifest,
    load_manifest,
    select_challenger,
)
from theory.sage_t.posterior_v4 import T8_6C_POLICIES


def test_manifest_binds_failed_t8_6b_and_projection_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    assert loaded["frozen_invariants"]["mixtures"] == [0.02, 0.05, 0.10]
    assert loaded["frozen_invariants"]["only_inference_change"] == (
        "collapse_triggered_family_projection"
    )
    assert loaded["firewall"]["source_validation_opened"] is False


def test_t8_6c_reuses_the_exact_64_root_schedule() -> None:
    schedules = json.loads(
        Path(DEFAULT_ACTION_SCHEDULES).read_text(encoding="utf-8")
    )

    assert len(schedules) == 64
    assert all(1 <= len(actions) <= 5 for actions in schedules.values())


def test_family_runner_restores_frozen_factory() -> None:
    original = v86._new_posterior

    with _family_runner():
        assert v86._new_posterior is _new_family_posterior

    assert v86._new_posterior is original


def test_selection_keeps_original_calibration_and_likelihood_gates(
    tmp_path: Path,
) -> None:
    manifest = freeze_manifest(output_path=tmp_path / "manifest.json")
    rows = []
    winner_name = "terminal_tempered_family_mix_05"
    for game in ("lp85", "su15"):
        for root_index in range(4):
            root = f"{game}:{root_index}"
            for condition in T8_6C_POLICIES:
                winning = condition == winner_name
                rows.append(
                    {
                        "episode_id": root,
                        "game": game,
                        "condition": condition,
                        "checkpoint": 5,
                        "terminal_brier": 0.1 if winning else 0.4,
                        "terminal_log_loss": 0.2 if winning else 0.8,
                        "hidden_log_likelihood": -1.0 if winning else -2.0,
                        "decision_latency_ms": 1.0,
                    }
                )
    updates = [
        {
            "condition": condition,
            "semantic_collapse": condition == "legacy" and index < 4,
        }
        for condition in T8_6C_POLICIES
        for index in range(10)
    ]

    winner, evaluations = select_challenger(
        rows,
        updates,
        manifest=manifest,
    )

    assert winner == winner_name
    assert evaluations[winner_name]["passed"]
    assert all(name in evaluations for name in CHALLENGERS)
