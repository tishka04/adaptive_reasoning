from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage12.bound_mechanic_pilot import BindingSignature, BoundEvent
from theory.sage12.pairwise_causal_pilot import (
    AUTHORITATIVE_EFFECTS,
    DEFAULT_FROZEN_MANIFEST_PATH,
    AntisymmetricLinearModel,
    PairEffectExample,
    _capacity,
    _fit_model,
    load_frozen_manifest,
    run_validation_collection,
    run_validation_evaluation,
    validate_model_view,
)


def _binding(
    *,
    kind: str,
    occupied: bool,
    direction: str,
    relation: str,
) -> BindingSignature:
    return BindingSignature(
        kind=kind,
        action_family="click",
        requested_direction=direction,
        occupied=occupied,
        path_status="blocked" if occupied else "open",
        actor_relation=relation,
        actor_relative_direction=direction,
        target_area_bucket="small" if occupied else "none",
        target_aspect_bucket="square" if occupied else "none",
        target_affordance="movable" if occupied else "none",
    )


def _event(binding: BindingSignature, *, removed: bool = False) -> BoundEvent:
    return BoundEvent(
        action_name="ACTION6",
        action_family="click",
        binding=binding,
        effects={
            "target_created": False,
            "target_removed": removed,
            "target_moved": False,
        },
        applicable={
            "target_created": True,
            "target_removed": binding.occupied,
            "target_moved": binding.occupied,
        },
    )


def _example(
    *,
    pair_id: str = "pair",
    game_id: str = "bp35",
    created: tuple[bool, bool] = (True, False),
    removed: tuple[bool, bool] = (False, True),
    moved: tuple[bool, bool] = (False, False),
) -> PairEffectExample:
    occupied = _binding(
        kind="occupied_object",
        occupied=True,
        direction="right",
        relation="aligned_row",
    )
    free = _binding(
        kind="free_slot",
        occupied=False,
        direction="left",
        relation="near",
    )
    return PairEffectExample(
        pair_id=pair_id,
        game_id=game_id,
        source_split="source_train",
        context=tuple(_event(occupied, removed=index % 2 == 0) for index in range(8)),
        left_action_name="ACTION6",
        left_action_family="click",
        left_binding=free,
        right_action_name="ACTION6",
        right_action_family="click",
        right_binding=occupied,
        outcomes={
            "target_created": created,
            "target_removed": removed,
            "target_moved": moved,
        },
        applicable={
            "target_created": True,
            "target_removed": True,
            "target_moved": True,
        },
    )


def test_manifest_freezes_source_only_pairwise_protocol() -> None:
    manifest = load_frozen_manifest()
    assert manifest["status"] == "FROZEN_BEFORE_SOURCE_PREFLIGHT"
    assert manifest["authoritative_effects"] == [
        "target_created",
        "target_removed",
    ]
    assert manifest["diagnostic_effects"] == ["target_moved"]
    assert manifest["firewall"]["source_validation_opened"] is False
    assert manifest["model"]["fit_intercept"] is False


def test_direction_uses_only_discordant_applicable_pairs() -> None:
    example = _example()
    assert example.is_discordant("target_created")
    assert example.direction("target_created") == 1
    assert example.direction("target_removed") == 0
    assert not example.is_discordant("target_moved")
    with pytest.raises(ValueError, match="discordant"):
        example.direction("target_moved")


def test_pair_model_view_is_difference_only_and_identity_free() -> None:
    example = _example()
    for projection in ("minimal", "relational", "typed"):
        for mode in (
            "structured",
            "history_no_binding",
            "action_only",
            "binding_only",
        ):
            validate_model_view(example, projection, mode)
            rendered = json.dumps(example.model_view(projection, mode))
            assert "bp35" not in rendered
            assert "pair_id" not in rendered
            assert "outcome" not in rendered


def test_antisymmetric_model_exactly_inverts_complete_arm_swap() -> None:
    rows = [
        {"binding:free": 1.0, "binding:occupied": -1.0},
        {"binding:free": -0.5, "binding:occupied": 0.5},
    ]
    model = _fit_model(rows, [1, 0])
    row = {"binding:free": 0.75, "binding:occupied": -0.75}
    swapped = {key: -value for key, value in row.items()}
    assert model.predict(swapped) == pytest.approx(1.0 - model.predict(row), abs=1e-15)
    assert AntisymmetricLinearModel.from_dict(model.to_dict()) == model
    assert model.to_dict()["fit_intercept"] is False


def test_binding_swap_preserves_action_and_context_but_changes_view() -> None:
    example = _example()
    ordinary = example.model_view("typed", "structured")
    swapped = example.binding_swapped_view("typed", "structured")
    assert ordinary != swapped
    assert example.context == _example().context
    assert example.left_action_name == example.right_action_name == "ACTION6"


def test_capacity_excludes_movement_from_authority() -> None:
    examples = [
        _example(pair_id=f"pair-{index}", game_id="bp35") for index in range(12)
    ] + [_example(pair_id=f"other-{index}", game_id="tu93") for index in range(12)]
    capacity = _capacity(examples)
    assert capacity["effects"]["target_created"]["discordant_pairs"] == 24
    assert capacity["effects"]["target_removed"]["games_with_at_least_10"] == 2
    assert capacity["effects"]["target_moved"]["discordant_pairs"] == 0
    assert capacity["effects"]["target_moved"]["authority"] is False


def test_validation_collection_is_blocked_by_source_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "source_preflight.json").write_text(
        json.dumps(
            {
                "status": "FAIL_CLOSED",
                "validation_collection_authorized": False,
                "preflight_checksum": "failure",
            }
        ),
        encoding="utf-8",
    )
    result = run_validation_collection(
        frozen_manifest_path=DEFAULT_FROZEN_MANIFEST_PATH,
        output_dir=tmp_path,
    )
    assert result["status"] == "SKIPPED_SOURCE_PREFLIGHT"
    assert result["validation_opened"] is False
    assert not (tmp_path / "validation_shards").exists()


def test_validation_evaluation_records_source_failure(tmp_path: Path) -> None:
    (tmp_path / "source_preflight.json").write_text(
        json.dumps(
            {
                "status": "FAIL_CLOSED",
                "preflight_checksum": "failure",
            }
        ),
        encoding="utf-8",
    )
    result = run_validation_evaluation(
        frozen_manifest_path=DEFAULT_FROZEN_MANIFEST_PATH,
        output_dir=tmp_path,
    )
    assert result["status"] == "SKIPPED_SOURCE_PREFLIGHT"
    assert result["validation_opened"] is False
    assert result["world_model_protocol_authorized"] is False


def test_frozen_authoritative_effect_tuple_matches_manifest() -> None:
    assert AUTHORITATIVE_EFFECTS == ("target_created", "target_removed")
