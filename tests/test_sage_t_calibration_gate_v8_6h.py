from __future__ import annotations

import json
from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6h import (
    DEFAULT_ACTION_SCHEDULES,
    SELECTED_POLICY,
    T8_6H_POLICIES,
    _goal_generation_runner,
    _new_minimum_kl_posterior,
    freeze_manifest,
    load_manifest,
    select_challenger,
)
from theory.sage_t.goal_generation_v2 import (
    programs_for_with_goal_progress_bridge,
)


def test_manifest_binds_t8_6g_and_changes_only_goal_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    invariants = loaded["frozen_invariants"]
    assert invariants["selected_posterior"] == SELECTED_POLICY
    assert invariants["posterior_implementation"] == "unchanged_from_t8_6g"
    assert invariants["goal_bridge_support"] == 0
    assert invariants["goal_bridge_prior_logprob"] == -0.05
    assert invariants["maximum_new_goal_bundles"] == 1
    assert invariants["repair_requires_full_history_replay"] is True
    assert loaded["firewall"]["source_validation_opened"] is False
    assert loaded["firewall"]["authority"] == "shadow"


def test_t8_6h_reuses_exact_parent_schedule() -> None:
    schedules = json.loads(
        Path(DEFAULT_ACTION_SCHEDULES).read_text(encoding="utf-8")
    )

    assert len(schedules) == 64
    assert all(1 <= len(actions) <= 5 for actions in schedules.values())


def test_goal_generation_runner_restores_both_frozen_factories() -> None:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for

    with _goal_generation_runner():
        assert v86._new_posterior is _new_minimum_kl_posterior
        assert v86._programs_for is programs_for_with_goal_progress_bridge

    assert v86._new_posterior is original_posterior
    assert v86._programs_for is original_generator


def test_selection_keeps_the_t8_6g_policy_and_all_gates(tmp_path: Path) -> None:
    manifest = freeze_manifest(output_path=tmp_path / "manifest.json")
    rows = []
    for game in ("lp85", "su15"):
        for root_index in range(4):
            root = f"{game}:{root_index}"
            for condition in T8_6H_POLICIES:
                winning = condition == SELECTED_POLICY
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
        for condition in T8_6H_POLICIES
        for index in range(10)
    ]

    winner, evaluations = select_challenger(
        rows,
        updates,
        manifest=manifest,
    )

    assert winner == SELECTED_POLICY
    assert evaluations[SELECTED_POLICY]["passed"]
