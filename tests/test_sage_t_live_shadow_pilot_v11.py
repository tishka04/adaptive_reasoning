from __future__ import annotations

from pathlib import Path

from theory.sage_t.live_shadow_pilot_v11 import (
    MAXIMUM_ACTIONS,
    VALIDATION_GAMES,
    freeze_action_manifest,
    freeze_confirmation_manifest,
    load_action_manifest,
    load_confirmation_manifest,
)


def test_t8_7_action_manifest_opens_only_source_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.json"

    frozen = freeze_action_manifest(output_path=path)
    loaded = load_action_manifest(path)

    assert loaded == frozen
    assert tuple(loaded["source_train_games"]) == VALIDATION_GAMES
    assert loaded["split"] == "source_validation"
    assert loaded["challenger"]["model_changes"] == []
    assert loaded["authority"]["mode"] == "shadow"
    assert "ar25-e3c63847" in loaded["forbidden_games"]


def test_t8_7_confirmation_binds_unchanged_parent(
    tmp_path: Path,
) -> None:
    actions = tmp_path / "actions.json"
    confirmation = tmp_path / "confirmation.json"
    freeze_action_manifest(output_path=actions)

    frozen = freeze_confirmation_manifest(
        output_path=confirmation,
        action_manifest_path=actions,
    )
    loaded = load_confirmation_manifest(
        confirmation,
        action_manifest_path=actions,
    )

    assert loaded == frozen
    assert loaded["model"] == "unchanged_t8_6j_r3"
    assert loaded["maximum_actions"] == MAXIMUM_ACTIONS == 150
    assert loaded["firewall"]["source_validation_opened"] is True
    assert loaded["firewall"]["holdout_opened"] is False
    assert loaded["firewall"]["bounded_authority"] is False
