from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage12.action_target_data import (
    build_action_target_trace,
    build_observation,
    grid_sha256,
)
from theory.sage12.compiler import SLOT_EFFECTS
from theory.sage12.counterfactual_panel_collection_v4_11 import (
    select_panel_actions,
)
from theory.sage12.counterfactual_semantic_panels_v4_11 import (
    CounterfactualPanel,
    PanelArm,
    SemanticPanelPrediction,
    _bootstrap_panel_difference,
    _centered_residual,
    _horizon_return,
    _sigmoid,
    evaluate_student,
    freeze_manifest,
)
from theory.sage12.semantic_teacher_v4_9 import _write_json


def _trace(
    *,
    game: str = "bp35",
    action: str = "ACTION1",
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
    step: int = 0,
):
    source = (
        np.asarray([[0, 1], [0, 0]], dtype=np.int16)
        if before is None
        else before
    )
    destination = source.copy() if after is None else after
    return build_action_target_trace(
        game_id=game,
        source_split="source_train",
        policy_seed=1,
        reset_index=0,
        step_index=step,
        collection_phase="unit",
        available_action_names=("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
        selected_action_name=action,
        selected_action_data={},
        frame_before=source,
        frame_after=destination,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _arm(index: int, action: str) -> PanelArm:
    return PanelArm(
        arm_index=index,
        replay_pre_state_sha256="pre",
        immediate_trace=_trace(action=action, step=index),
        continuations=((), ()),
    )


def test_manifest_freezes_medium_panel_budget_and_no_downstream_authority(
    tmp_path,
) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)

    assert manifest["collection"]["target_panels_per_game"] == 96
    assert manifest["collection"]["minimum_panels_per_game"] == 80
    assert manifest["collection"]["continuation_rollouts"] == 2
    assert not manifest["evaluation"]["can_fit_world_model_in_this_iteration"]
    assert not manifest["evaluation"]["can_fit_ebm_in_this_iteration"]


def test_counterfactual_panel_requires_distinct_actions_and_matching_prestate() -> None:
    first = _arm(0, "ACTION1")
    duplicate = replace(_arm(1, "ACTION2"), immediate_trace=first.immediate_trace)

    with pytest.raises(ValueError, match="distinct"):
        CounterfactualPanel(
            game_id="bp35",
            policy_seed=1,
            reset_index=0,
            panel_index=0,
            expected_pre_state_sha256="pre",
            pre_grid_sha256=grid_sha256(first.immediate_trace.frame_before),
            arms=(first, duplicate),
        )

    with pytest.raises(ValueError, match="replay hashes"):
        CounterfactualPanel(
            game_id="bp35",
            policy_seed=1,
            reset_index=0,
            panel_index=0,
            expected_pre_state_sha256="other",
            pre_grid_sha256=grid_sha256(first.immediate_trace.frame_before),
            arms=(first, _arm(1, "ACTION2")),
        )


def test_panel_action_selection_is_deterministic_and_bounded() -> None:
    frame = SimpleNamespace(
        frame=np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int16),
        game_state="NOT_FINISHED",
        levels_completed=0,
    )
    observation = build_observation(
        frame.frame,
        available_actions=("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
        game_state=frame.game_state,
        levels_completed=frame.levels_completed,
        infer_players=True,
    )
    legal = tuple(
        SimpleNamespace(name=f"ACTION{index}", action_args={})
        for index in range(1, 5)
    )

    first = select_panel_actions(
        legal,
        observation=observation,
        pre_frame=frame,
        used_repeat_keys=set(),
        selection_counts={},
        maximum_arms=3,
        salt="unit",
    )
    second = select_panel_actions(
        legal,
        observation=observation,
        pre_frame=frame,
        used_repeat_keys=set(),
        selection_counts={},
        maximum_arms=3,
        salt="unit",
    )

    assert [row.name for row in first] == [row.name for row in second]
    assert len(first) == 3
    assert len({row.name for row in first}) == 3


def test_horizon_teacher_uses_discounted_continuations() -> None:
    before = np.asarray([[0, 1], [0, 0]], dtype=np.int16)
    moved = np.asarray([[0, 0], [0, 1]], dtype=np.int16)
    immediate = _trace(before=before, after=moved, step=0)
    continuation = _trace(before=moved, after=before, step=1)
    arm = PanelArm(
        arm_index=0,
        replay_pre_state_sha256="pre",
        immediate_trace=immediate,
        continuations=((continuation,), (continuation,)),
    )

    value, uncertainty = _horizon_return(arm)

    assert np.isfinite(value)
    assert uncertainty == pytest.approx(0.0)


def test_centered_residual_is_zero_mean_per_panel_and_zero_for_singletons() -> None:
    root = np.zeros((3, 2), dtype=np.float64)
    full = np.asarray([[1.0, 3.0], [3.0, 7.0], [8.0, 9.0]])
    indices = np.asarray([10, 11, 12], dtype=np.int64)

    centered = _centered_residual(
        full,
        root,
        indices,
        {"panel": (10, 11), "singleton": (12,)},
    )

    assert np.max(np.abs(centered[:2].mean(axis=0))) <= 1e-12
    assert np.array_equal(centered[2], np.zeros(2))


def test_pair_probability_is_exactly_antisymmetric() -> None:
    forward = float(_sigmoid(np.asarray([2.75]))[0])
    reverse = float(_sigmoid(np.asarray([-2.75]))[0])

    assert forward + reverse == pytest.approx(1.0, abs=1e-12)


def test_panel_prediction_adapts_without_changing_slot_annotation_contract() -> None:
    probabilities = {effect: 0.25 for effect in SLOT_EFFECTS}
    panel = SemanticPanelPrediction(
        panel_id="panel",
        effect_probabilities={"slot": probabilities},
        progress_scores={"slot": 0.5},
        preference_probabilities={"slot": {"slot": 0.5}},
    )

    annotations = panel.to_slot_annotations()

    assert len(annotations) == 1
    assert annotations[0].support == 0
    assert set(annotations[0].effect_probabilities) == set(SLOT_EFFECTS)


def test_bootstrap_uses_equal_game_blocks() -> None:
    rows = {
        "a": {"game_id": "bp35", "baseline": 2.0, "full": 1.0},
        "b": {"game_id": "bp35", "baseline": 2.0, "full": 1.0},
        "c": {"game_id": "cd82", "baseline": 3.0, "full": 1.0},
    }

    result = _bootstrap_panel_difference(
        rows,
        left_key="baseline",
        right_key="full",
        samples=100,
        seed=1,
    )

    assert result["ci_lower"] > 0.0


def test_capacity_failure_stops_before_model_or_downstream_fit(tmp_path) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)
    _write_json(
        tmp_path / "teacher_qa.json",
        {
            "manifest_checksum": manifest["manifest_checksum"],
            "qa_checksum": "qa-unit",
            "teacher_ready": False,
        },
    )

    result = evaluate_student(output_dir=tmp_path, device="cpu")

    assert result["verdict"] == "COMPARATIVE_CAUSAL_TEACHER_CAPACITY_FAILED"
    assert not result["world_model_fitted"]
    assert not result["ebm_fitted"]
