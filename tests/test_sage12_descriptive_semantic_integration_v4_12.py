from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage12.descriptive_semantic_integration_v4_12 import (
    ACTIVE_EFFECTS,
    _effect_pair_rows,
    evaluate_integration,
    freeze_manifest,
)
from theory.sage12.semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    _checksum,
    _read_json,
    _write_json,
)


def _record(*, game: str, positive: bool) -> SimpleNamespace:
    return SimpleNamespace(
        game_id=game,
        labels={effect: bool(positive) for effect in SEMANTIC_EFFECTS},
        applicable={effect: True for effect in SEMANTIC_EFFECTS},
    )


def test_manifest_reuses_v411_and_freezes_conditional_global_gate(tmp_path) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)

    assert tuple(manifest["active_effects"]) == ACTIVE_EFFECTS
    assert manifest["teacher_contract"]["scalar_progress_target_used"] is False
    assert manifest["training"]["progress_pair_weight"] == 0.0
    assert manifest["integration_gate"]["runs_only_if_semantic_gate_passes"]
    assert manifest["integration_gate"]["trajectory_depth"] == 3
    assert manifest["authority_promoted"] is False


def test_effect_pair_loss_gives_partial_credit_per_observed_effect() -> None:
    records = (_record(game="bp35", positive=True), _record(game="bp35", positive=False))
    comparison = SimpleNamespace(
        panel_id="panel",
        game_id="bp35",
        left=0,
        right=1,
        fresh=True,
    )
    strong = np.full((2, len(SEMANTIC_EFFECTS)), 0.5, dtype=np.float64)
    weak = strong.copy()
    for effect in ACTIVE_EFFECTS:
        column = SEMANTIC_EFFECTS.index(effect)
        strong[:, column] = (0.9, 0.1)
        weak[:, column] = (0.55, 0.45)

    rows = _effect_pair_rows(
        records,
        (comparison,),
        {"strong": strong, "weak": weak},
    )

    assert len(rows) == len(ACTIVE_EFFECTS)
    assert all(row["strong"] < row["weak"] for row in rows.values())


def test_effect_pair_rows_ignore_equal_and_legacy_effects() -> None:
    left = _record(game="bp35", positive=False)
    right = _record(game="bp35", positive=False)
    matrix = np.full((2, len(SEMANTIC_EFFECTS)), 0.5, dtype=np.float64)

    rows = _effect_pair_rows(
        (left, right),
        (
            SimpleNamespace(
                panel_id="fresh",
                game_id="bp35",
                left=0,
                right=1,
                fresh=True,
            ),
            SimpleNamespace(
                panel_id="legacy",
                game_id="bp35",
                left=0,
                right=1,
                fresh=False,
            ),
        ),
        {"model": matrix},
    )

    assert rows == {}


def test_failed_semantic_gate_skips_world_model_and_ebm(tmp_path) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)
    semantic = {
        "semantic_gate_passed": False,
        "result_checksum": "semantic-unit",
    }
    _write_json(tmp_path / "semantic_result.json", semantic)

    result = evaluate_integration(output_dir=tmp_path)

    assert result["verdict"] == "SKIPPED_SEMANTIC_GATE_FAILED"
    assert not result["world_model_fitted"]
    assert not result["ebm_fitted"]
    written = _read_json(tmp_path / "integration_result.json")
    checksum = written.pop("result_checksum")
    assert checksum == _checksum(written)


def test_manifest_rejects_effect_capacity_drift(tmp_path) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)
    path = tmp_path / "frozen_manifest.json"
    payload = _read_json(path)
    payload["active_effects"] = list(reversed(ACTIVE_EFFECTS))
    payload.pop("manifest_checksum")
    payload["manifest_checksum"] = _checksum(payload)
    _write_json(path, payload)

    with pytest.raises(ValueError, match="active-effect drift"):
        from theory.sage12.descriptive_semantic_integration_v4_12 import (
            load_manifest,
        )

        load_manifest(tmp_path)
