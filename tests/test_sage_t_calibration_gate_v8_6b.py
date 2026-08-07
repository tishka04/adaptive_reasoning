from __future__ import annotations

import json
from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6b import (
    CHALLENGERS,
    DEFAULT_ACTION_SCHEDULES,
    _channel_runner,
    _new_channel_posterior,
    freeze_manifest,
    load_manifest,
    select_challenger,
)
from theory.sage_t.posterior_v3 import T8_6B_POLICIES


def test_manifest_binds_parent_result_and_exact_action_schedules(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    assert loaded["frozen_invariants"]["only_inference_change"] == (
        "channel_specific_evidence_multiplier"
    )
    assert loaded["frozen_invariants"]["actions"] == "exact_t8_6_schedules"
    assert loaded["firewall"]["source_validation_opened"] is False
    assert "ar25" in loaded["forbidden_games"]


def test_frozen_schedules_cover_all_64_roots_without_extra_reveals() -> None:
    schedules = json.loads(
        Path(DEFAULT_ACTION_SCHEDULES).read_text(encoding="utf-8")
    )

    assert len(schedules) == 64
    assert all(1 <= len(actions) <= 5 for actions in schedules.values())
    assert all(
        action.startswith("ACTION")
        for actions in schedules.values()
        for action in actions
    )


def test_channel_runner_restores_the_frozen_t8_6_factory() -> None:
    original = v86._new_posterior

    with _channel_runner():
        assert v86._new_posterior is _new_channel_posterior

    assert v86._new_posterior is original


def test_preregistered_selection_uses_all_original_gates(
    tmp_path: Path,
) -> None:
    manifest = freeze_manifest(output_path=tmp_path / "manifest.json")
    rows = []
    for game in ("lp85", "su15"):
        for root_index in range(4):
            root = f"{game}:{root_index}"
            for condition in ("legacy", *CHALLENGERS):
                winning = condition == "terminal_tempered"
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
        for condition in T8_6B_POLICIES
        for index in range(10)
    ]

    winner, evaluations = select_challenger(
        rows,
        updates,
        manifest=manifest,
    )

    assert winner == "terminal_tempered"
    assert evaluations[winner]["passed"]
    assert not evaluations["teleology_tempered"]["passed"]
