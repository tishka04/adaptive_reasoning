from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.live_transition_loop import build_observation
from theory.sage12.action_target_data import (
    ActionTargetAnchor,
    ActionTargetTrace,
    ObservedActionTargetEffects,
)
from theory.sage12.bound_mechanic_pilot import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    TARGET_EFFECTS,
    ActionSpec,
    BindingPairRecord,
    BindingSignature,
    BoundEvent,
    BoundWindow,
    BranchArm,
    CalibrationBundle,
    beam_rollout,
    binding_swap_control,
    fit_priors,
    load_frozen_manifest,
    pair_windows,
    replay_prefix,
    run_world_model_evaluation,
    score_window,
    select_branch_actions,
    validate_model_view,
)


def _signature(
    *,
    kind: str = "occupied_object",
    direction: str = "right",
    occupied: bool = True,
    relation: str = "aligned_row",
    relative: str = "right",
) -> BindingSignature:
    return BindingSignature(
        kind=kind,
        action_family="click",
        requested_direction=direction,
        occupied=occupied,
        path_status="blocked" if occupied else "open",
        actor_relation=relation,
        actor_relative_direction=relative,
        target_area_bucket="small",
        target_aspect_bucket="square",
        target_affordance="movable",
    )


def _event(
    *,
    binding: BindingSignature | None = None,
    created: bool = False,
    removed: bool = False,
    moved: bool = False,
) -> BoundEvent:
    binding = binding or _signature()
    return BoundEvent(
        action_name="ACTION6",
        action_family="click",
        binding=binding,
        effects={
            "target_created": created,
            "target_removed": removed,
            "target_moved": moved,
        },
        applicable={
            "target_created": binding.kind != "targetless",
            "target_removed": binding.occupied,
            "target_moved": binding.occupied,
        },
    )


def _trace(
    *,
    split: str,
    action_args: dict[str, int],
    anchor: ActionTargetAnchor,
    removed: bool,
) -> ActionTargetTrace:
    effects = ObservedActionTargetEffects(
        labels={
            "actor_displaced": False,
            "target_created": False,
            "target_removed": removed,
            "target_moved": False,
        },
        applicable={
            "actor_displaced": True,
            "target_created": True,
            "target_removed": True,
            "target_moved": True,
        },
    )
    return ActionTargetTrace(
        game_id="sc25",
        source_split=split,
        policy_seed=857,
        reset_index=0,
        step_index=8,
        collection_phase="test",
        available_action_names=("ACTION6",),
        selected_action_name="ACTION6",
        selected_action_data=action_args,
        anchor=anchor,
        effects=effects,
        frame_before=np.zeros((3, 3), dtype=int),
        frame_after=np.zeros((3, 3), dtype=int),
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _pair() -> BindingPairRecord:
    left_anchor = ActionTargetAnchor(
        kind="clicked_object",
        action_family="click",
        row=1,
        col=1,
        in_bounds=True,
        occupied=True,
        target_object_id=4,
        target_area_bucket="small",
        target_aspect_bucket="square",
        target_affordance="movable",
        actor_relation="aligned_row",
        actor_relative_direction="right",
        path_status="blocked",
    )
    right_anchor = ActionTargetAnchor(
        kind="clicked_empty",
        action_family="click",
        row=1,
        col=2,
        in_bounds=True,
        occupied=False,
        target_area_bucket="none",
        target_aspect_bucket="none",
        target_affordance="none",
        actor_relation="aligned_row",
        actor_relative_direction="right",
        path_status="open",
    )
    pre_hash = "a" * 64
    left_trace = _trace(
        split="source_train",
        action_args={"x": 1, "y": 1},
        anchor=left_anchor,
        removed=True,
    )
    right_trace = _trace(
        split="source_train",
        action_args={"x": 2, "y": 1},
        anchor=right_anchor,
        removed=False,
    )
    return BindingPairRecord(
        game_id="sc25",
        source_split="source_train",
        policy_seed=857,
        reset_index=0,
        root_index=0,
        path="",
        depth=0,
        context=tuple(_event() for _ in range(8)),
        expected_pre_state_sha256=pre_hash,
        replay_pre_state_sha256=pre_hash,
        left=BranchArm(
            arm="left",
            action=ActionSpec("ACTION6", {"x": 1, "y": 1}),
            trace=left_trace,
            replay_pre_state_sha256=pre_hash,
            post_state_sha256="b" * 64,
        ),
        right=BranchArm(
            arm="right",
            action=ActionSpec("ACTION6", {"x": 2, "y": 1}),
            trace=right_trace,
            replay_pre_state_sha256=pre_hash,
            post_state_sha256="c" * 64,
        ),
    )


def test_manifest_is_frozen_and_content_addressed() -> None:
    manifest = load_frozen_manifest()
    assert manifest["status"] == "FROZEN_BEFORE_SOURCE_COLLECTION"
    assert manifest["collection"]["source_seeds"] == [857, 907, 953, 1009]
    assert manifest["collection"]["validation_seeds"] == [
        1061,
        1103,
        1151,
        1201,
    ]
    assert manifest["firewall"]["v4_2_1_shards_reused"] is False


def test_binding_projection_ladder_excludes_identity_and_coordinates() -> None:
    signature = _signature()
    assert set(signature.model_view("minimal")) == {
        "kind",
        "occupied",
        "path_status",
    }
    assert "actor_relation" in signature.model_view("relational")
    assert "target_affordance" in signature.model_view("typed")
    view = signature.model_view("typed")
    assert "row" not in view
    assert "col" not in view
    assert "object_id" not in view


def test_pair_round_trip_digest_and_model_firewall() -> None:
    pair = _pair()
    restored = BindingPairRecord.from_dict(pair.to_dict())
    assert restored.to_dict() == pair.to_dict()
    assert restored.pair_digest == pair.pair_digest
    for window in pair_windows(restored):
        validate_model_view(window, "typed")
        rendered = json.dumps(window.model_view("typed"))
        assert "sc25" not in rendered
        assert '"x"' not in rendered and '"y"' not in rendered


def test_pair_rejects_nonidentical_arm_prestate() -> None:
    payload = _pair().to_dict()
    payload["right"]["replay_pre_state_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="identical pre-state"):
        BindingPairRecord.from_dict(payload)


def test_replay_mismatch_fails_closed_before_execution() -> None:
    frame = np.zeros((2, 2), dtype=int)
    with pytest.raises(RuntimeError, match="replay mismatch"):
        replay_prefix(
            object(),
            frame,
            (),
            expected_pre_state_sha256="0" * 64,
        )


def test_replay_accepts_duplicate_byte_identical_legal_candidates() -> None:
    from theory.sage12 import bound_mechanic_pilot as pilot

    duplicate = _Action("ACTION6", {"x": 1, "y": 1})

    class _Environment:
        pass

    original = pilot._legal_actions
    pilot._legal_actions = lambda _env: (duplicate, duplicate)
    try:
        assert (
            pilot._find_action(_Environment(), ActionSpec("ACTION6", {"x": 1, "y": 1}))
            is duplicate
        )
    finally:
        pilot._legal_actions = original


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: dict[str, int]


def test_same_action_different_argument_is_preferred() -> None:
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 2
    grid[3, 3] = 3
    observation = build_observation(
        grid,
        available_actions=("ACTION1", "ACTION6"),
        game_state="NOT_FINISHED",
        levels_completed=0,
        infer_players=True,
    )
    actions = (
        _Action("ACTION1", {}),
        _Action("ACTION6", {"x": 1, "y": 1}),
        _Action("ACTION6", {"x": 3, "y": 3}),
    )
    left, right = select_branch_actions(actions, observation, {}, salt="test")
    assert left.name == right.name == "ACTION6"
    assert left.action_args != right.action_args


def test_binding_swap_changes_only_query_binding() -> None:
    windows = list(pair_windows(_pair()))
    swapped, changed_rate = binding_swap_control(windows)
    assert changed_rate == 1.0
    for original, changed in zip(windows, swapped):
        assert original.context == changed.context
        assert original.labels == changed.labels
        assert original.query_action_name == changed.query_action_name
        assert original.query_binding != changed.query_binding


def test_rules_keep_support_zero_and_observations_as_evidence() -> None:
    occupied = _signature()
    context = tuple(
        _event(binding=occupied, removed=index % 2 == 0) for index in range(8)
    )
    window = BoundWindow(
        pair_id="pair",
        arm="left",
        game_id="bp35",
        source_split="source_train",
        root_key="root",
        path="",
        context=context,
        query_action_name="ACTION6",
        query_action_family="click",
        query_binding=occupied,
        labels={label: False for label in TARGET_EFFECTS},
        applicable={label: True for label in TARGET_EFFECTS},
    )
    priors = fit_priors([window], "typed")
    _, evidence = score_window(window, priors, projection="typed", mode="structured")
    assert all(item.rule.support == 0 for item in evidence)
    assert any(
        item.observed_support + item.observed_refutations >= 2 for item in evidence
    )


def test_world_model_constraints_forbid_creation_on_occupied_target() -> None:
    occupied = _signature()
    context = tuple(_event(binding=occupied) for _ in range(8))
    query = _event(binding=occupied)
    window = BoundWindow(
        pair_id="pair",
        arm="left",
        game_id="bp35",
        source_split="source_train",
        root_key="root",
        path="",
        context=context,
        query_action_name=query.action_name,
        query_action_family=query.action_family,
        query_binding=query.binding,
        labels=query.effects,
        applicable=query.applicable,
    )
    priors = fit_priors([window], "typed")
    parameters = {
        mode: {label: {"slope": 1.0, "intercept": 0.0} for label in TARGET_EFFECTS}
        for mode in (
            "structured",
            "no_binding",
            "action_only",
            "binding_only",
            "template",
        )
    }
    thresholds = {mode: {label: 0.5 for label in TARGET_EFFECTS} for mode in parameters}
    calibration = CalibrationBundle("typed", parameters, thresholds)
    result = beam_rollout(
        initial_context=context,
        queries=(query, query, query),
        priors=priors,
        calibration=calibration,
        projection="typed",
        mode="structured",
        beam_width=8,
    )
    created_index = TARGET_EFFECTS.index("target_created")
    assert all(
        not step[created_index] for sequence in result["sequences"] for step in sequence
    )


def test_world_model_refuses_failed_binding_result(tmp_path: Path) -> None:
    (tmp_path / "binding_result.json").write_text(
        json.dumps(
            {
                "status": "FAIL_CLOSED",
                "world_model_fit_authorized": False,
                "result_checksum": "failure",
            }
        ),
        encoding="utf-8",
    )
    result = run_world_model_evaluation(
        frozen_manifest_path=DEFAULT_FROZEN_MANIFEST_PATH,
        output_dir=tmp_path,
    )
    assert result["status"] == "SKIPPED_FAIL_CLOSED"
    assert result["world_model_fitted"] is False


def test_v4_2_1_result_is_unchanged() -> None:
    path = (
        Path("training") / "sage12" / "mechanic_induction_v4_2_1" / "pilot_result.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        payload["result_checksum"]
        == "27861c650c1cd51f5ee96c03e3ae297497a4d04e39f49391b1631840b43757ff"
    )
