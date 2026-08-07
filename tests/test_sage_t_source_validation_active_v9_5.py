from __future__ import annotations

import json

from theory.sage_t import calibration_gate_v8_6c as checksums
from theory.sage_t import source_validation_active_v9_5 as validation


def test_t9_5_manifest_uses_full_frozen_validation_protocol() -> None:
    manifest = validation.load_manifest()

    assert len(manifest["source_validation_games"]) == 3
    assert len(manifest["seeds"]) == 5
    assert manifest["resets_per_pair"] == 14
    assert manifest["configured_actions_per_pair"] >= 1_000
    assert manifest["configured_pairs"] == 15
    assert manifest["policy"]["retouched_after_t9_4"] is False
    assert manifest["firewall"]["source_validation_opened"] is True
    assert manifest["firewall"]["holdout_opened"] is False


def test_condition_checkpoints_are_atomic_and_manifest_bound(tmp_path) -> None:
    manifest = validation.load_manifest()
    path = validation._condition_path(
        tmp_path,
        game_id="re86-4e57566e",
        seed=1061,
    )
    row = {
        "format_version": validation.FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "game": "re86-4e57566e",
        "seed": 1061,
    }
    row["condition_checksum"] = checksums._checksum(row)
    checksums._write_json(path, row)

    loaded = validation._load_condition(
        path,
        manifest_checksum=manifest["manifest_checksum"],
    )

    assert loaded == row
    tampered = dict(row)
    tampered["seed"] = 1062
    path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        validation._load_condition(
            path,
            manifest_checksum=manifest["manifest_checksum"],
        )
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("tampered condition was accepted")


def test_report_gate_requires_positive_progress_top8() -> None:
    manifest = validation.load_manifest()
    rows = []
    for game in manifest["source_validation_games"]:
        for seed in manifest["seeds"]:
            positive = game == manifest["source_validation_games"][0]
            rows.append(
                {
                    "game": game,
                    "seed": seed,
                    "configured_actions": manifest["configured_actions_per_pair"],
                    "active": {
                        "actions": 1000,
                        "levels_completed": int(positive),
                        "game_over_actions": 0,
                    },
                    "off": {
                        "actions": 1000,
                        "levels_completed": 0,
                        "game_over_actions": 0,
                    },
                    "level_rate_delta": 0.001 if positive else 0.0,
                    "game_over_rate_delta": 0.0,
                    "intervention": {
                        "interventions": 10,
                        "useful_interventions": int(positive),
                    },
                    "progress_action_top8": {
                        "positive_actions": int(positive),
                        "top8": int(positive),
                    },
                    "false_high_terminal": {
                        "false_high": 0,
                        "observations": 1000,
                    },
                    "decision_latencies_ms": [1.0],
                    "observation_latencies_ms": [1.0],
                    "illegal_actions": 0,
                    "controller_errors": 0,
                    "environment_errors": 0,
                    "effective_mode": "active",
                }
            )

    report = validation.build_report(
        rows,
        manifest=manifest,
        runtime={"ready": True},
        wall_seconds=1.0,
    )

    assert report["metrics"]["progress_action_top8_rate"] == 1.0
    assert report["checks"]["progress_action_top8"] is True
    assert report["holdout_opened"] is False
