from __future__ import annotations

from theory.sage_t.live_shadow_checkpoint import (
    _write_timeout_checkpoint,
    aggregate_checkpoints,
    load_frozen_manifest,
)
from theory.sage_t.live_shadow_pilot import (
    load_frozen_manifest as load_t8_manifest,
)


def test_checkpoint_retry_changes_execution_only() -> None:
    retry = load_frozen_manifest()
    base = load_t8_manifest()

    assert retry["base_t8_manifest_checksum"] == base["manifest_checksum"]
    assert retry["scientific_protocol_changes"] == []
    assert retry["condition_timeout_seconds"] == 180.0


def test_timeout_checkpoints_produce_a_fail_closed_partial_report(tmp_path) -> None:
    base = load_t8_manifest()
    for game_id in base["source_train_games"]:
        for seed in base["seeds"]:
            _write_timeout_checkpoint(
                game_id=game_id,
                seed=seed,
                timeout_seconds=180.0,
                output_dir=tmp_path,
                stderr="synthetic timeout",
            )

    report = aggregate_checkpoints(output_dir=tmp_path)

    assert report["completed_conditions"] == 0
    assert report["timed_out_conditions"] == 3
    assert report["rows"] == 0
    assert report["integration_gate_passed"] is False
    assert report["active_authority_authorized"] is False
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "rows.jsonl").exists()
