from __future__ import annotations

import json
from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6g import (
    CHALLENGERS,
    DEFAULT_ACTION_SCHEDULES,
    _minimum_kl_runner,
    _new_minimum_kl_posterior,
    freeze_manifest,
    load_manifest,
    select_challenger,
)
from theory.sage_t.posterior_v8 import T8_6G_POLICIES


def test_manifest_binds_t8_6f_and_freezes_minimum_kl_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    invariants = loaded["frozen_invariants"]
    assert invariants["terminal_temperature"] == 0.20
    assert invariants["entropy_floor"] == 0.0501
    assert invariants["maximum_family_total_variation"] == 0.02
    assert invariants["objective"] == "minimize_KL_q_parallel_p"
    assert invariants["optimization_domain"] == "semantic_family_masses"
    assert invariants["within_family_ratio_invariant"] is True
    assert loaded["firewall"]["source_validation_opened"] is False
    assert loaded["firewall"]["authority"] == "shadow"


def test_t8_6g_reuses_exact_parent_schedule() -> None:
    schedules = json.loads(
        Path(DEFAULT_ACTION_SCHEDULES).read_text(encoding="utf-8")
    )

    assert len(schedules) == 64
    assert all(1 <= len(actions) <= 5 for actions in schedules.values())


def test_minimum_kl_runner_restores_frozen_factory() -> None:
    original = v86._new_posterior

    with _minimum_kl_runner():
        assert v86._new_posterior is _new_minimum_kl_posterior

    assert v86._new_posterior is original


def test_selection_preserves_every_t8_6_gate(tmp_path: Path) -> None:
    manifest = freeze_manifest(output_path=tmp_path / "manifest.json")
    winner_name = CHALLENGERS[0]
    rows = []
    for game in ("lp85", "su15"):
        for root_index in range(4):
            root = f"{game}:{root_index}"
            for condition in T8_6G_POLICIES:
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
        for condition in T8_6G_POLICIES
        for index in range(10)
    ]

    winner, evaluations = select_challenger(
        rows,
        updates,
        manifest=manifest,
    )

    assert winner == winner_name
    assert evaluations[winner]["passed"]
