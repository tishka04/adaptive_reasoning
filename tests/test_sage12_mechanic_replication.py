from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from theory.sage12.action_target_data import EFFECT_LABELS, build_action_target_trace
from theory.sage12.mechanic_induction import MechanicQuery
from theory.sage12.mechanic_replication import (
    ActorRoleState,
    CalibrationBundle,
    CausalRoleTracker,
    MechanicWindowRecord,
    SemanticTransitionEvent,
    _select_threshold,
    apply_calibration,
    build_mechanic_windows,
    compact_qwen_prompt,
    compact_qwen_schema,
    compile_compact_rule,
    load_frozen_manifest,
    multilabel_metrics,
    validate_model_view,
)


def _grid(player=(2, 2), target=None, *, shape=(12, 16)):
    grid = np.zeros(shape, dtype=np.int32)
    if player is not None:
        grid[player] = 1
    if target is not None:
        grid[target] = 2
    return grid


def _trace(
    *,
    step=0,
    before=None,
    after=None,
    action="ACTION4",
    reset=0,
):
    before = _grid(player=(2, 2 + step)) if before is None else before
    after = _grid(player=(2, 3 + step)) if after is None else after
    return build_action_target_trace(
        game_id="bp35",
        source_split="source_train",
        policy_seed=307,
        reset_index=reset,
        step_index=step,
        collection_phase="test_v4_1",
        available_action_names=(
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION6",
        ),
        selected_action_name=action,
        selected_action_data={},
        frame_before=before,
        frame_after=after,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _event(
    *,
    moved=True,
    state=ActorRoleState.TRANSLATIONAL.value,
    action="ACTION4",
):
    return SemanticTransitionEvent(
        action_name=action,
        action_family="move",
        anchor_condition="open",
        effects={
            "actor_displaced": moved,
            "target_created": False,
            "target_removed": False,
            "target_moved": False,
        },
        applicable={label: True for label in EFFECT_LABELS},
        actor_role_known=state != ActorRoleState.AMBIGUOUS.value,
        actor_role_state=state,
    )


def _window(*, context=None, moved=True):
    context = tuple(context or [_event() for _ in range(8)])
    return MechanicWindowRecord(
        game_id="bp35",
        source_split="source_train",
        policy_seed=307,
        reset_index=0,
        query_step_index=8,
        context=context,
        query=MechanicQuery("ACTION4", "move", "open"),
        labels={
            "actor_displaced": moved,
            "target_created": False,
            "target_removed": False,
            "target_moved": False,
        },
        applicable={label: True for label in EFFECT_LABELS},
        actor_role_known=True,
        actor_role_state=ActorRoleState.TRANSLATIONAL.value,
    )


def test_causal_tracker_resolves_translation_without_unit_step_assumption():
    before = _grid(player=(2, 2))
    after = _grid(player=(2, 4))
    event = CausalRoleTracker().observe(
        _trace(before=before, after=after, action="ACTION4")
    )

    assert event.actor_role_state == ActorRoleState.TRANSLATIONAL.value
    assert event.applicable["actor_displaced"]
    assert event.effects["actor_displaced"]


def test_causal_tracker_marks_repeated_non_translation_explicitly():
    tracker = CausalRoleTracker()
    event = None
    for step in range(8):
        before = np.zeros((12, 16), dtype=np.int32)
        after = before.copy()
        before[3:8, 3:8] = 2 + step % 2
        after[3:8, 3:8] = 3 - step % 2
        event = tracker.observe(
            _trace(step=step, before=before, after=after, action="ACTION1")
        )

    assert event is not None
    assert event.actor_role_state == ActorRoleState.NON_TRANSLATIONAL.value
    assert not event.applicable["actor_displaced"]


def test_tracker_is_causal_when_future_frames_change():
    first = _trace(step=0)
    future_a = _trace(step=1)
    future_b = _trace(
        step=1,
        after=_grid(player=(9, 9)),
    )
    tracker_a = CausalRoleTracker()
    tracker_b = CausalRoleTracker()

    event_a = tracker_a.observe(first)
    tracker_a.observe(future_a)
    event_b = tracker_b.observe(first)
    tracker_b.observe(future_b)

    assert event_a == event_b


def test_v4_1_window_builder_preserves_reset_and_continuity_firewalls():
    traces = [_trace(step=index) for index in range(10)]
    windows = build_mechanic_windows(traces)

    assert len(windows) == 2
    assert all(len(window.context) == 8 for window in windows)

    broken = list(traces)
    broken[9] = replace(broken[9], reset_index=1)
    assert len(build_mechanic_windows(broken)) == 1


def test_role_state_is_audit_only_and_absent_from_model_view():
    window = _window()

    validate_model_view(window)
    rendered = str(window.model_view())

    assert "actor_role_state" not in rendered
    assert "actor_role_known" not in rendered
    assert window.actor_role_state in str(window.to_dict())


def test_window_round_trip_preserves_audit_and_model_views():
    window = _window()
    restored = MechanicWindowRecord.from_dict(window.to_dict())

    assert restored.to_dict() == window.to_dict()
    assert restored.model_view() == window.model_view()


def test_calibration_bundle_round_trip_and_checksum():
    parameters = {
        mode: {
            label: {"slope": 1.0, "intercept": 0.0}
            for label in EFFECT_LABELS
        }
        for mode in (
            "structured",
            "context_ablation",
            "local_action",
            "global_action",
            "template",
        )
    }
    thresholds = {
        mode: {label: 0.5 for label in EFFECT_LABELS}
        for mode in parameters
    }
    bundle = CalibrationBundle(
        parameters=parameters,
        thresholds=thresholds,
        source_oof_metrics={},
    )

    restored = CalibrationBundle.from_dict(bundle.to_dict())

    assert restored.calibration_checksum == bundle.calibration_checksum
    assert np.allclose(
        apply_calibration(np.full((2, 4), 0.25), restored, "structured"),
        0.25,
    )
    corrupted = bundle.to_dict()
    corrupted["calibration_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        CalibrationBundle.from_dict(corrupted)


def test_threshold_selection_is_deterministic_and_uses_source_scores():
    probabilities = np.asarray([0.1, 0.2, 0.6, 0.7])
    targets = np.asarray([0, 1, 1, 1])

    first = _select_threshold(probabilities, targets)
    second = _select_threshold(probabilities, targets)

    assert first == second
    assert first in {0.1, 0.2, 0.5, 0.6, 0.7}


def test_metrics_use_frozen_per_label_thresholds():
    targets = np.asarray([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    masks = np.ones_like(targets)
    probabilities = np.asarray(
        [[0.4, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1]]
    )
    thresholds = {label: 0.5 for label in EFFECT_LABELS}
    thresholds["actor_displaced"] = 0.3

    metrics = multilabel_metrics(
        targets,
        masks,
        probabilities,
        thresholds=thresholds,
    )

    assert metrics["per_label"]["actor_displaced"]["f1"] == 1.0


def test_compact_qwen_prompt_is_outcome_blind_for_query():
    window = _window(moved=True)
    prompt = compact_qwen_prompt(window)

    assert "labels" not in prompt
    assert "query_step_index" not in prompt
    assert "Q=ACTION4/move/open" in prompt
    assert len(prompt) < 900


def test_compact_qwen_rule_compiles_only_when_grounded():
    query = MechanicQuery("ACTION4", "move", "open")
    payload = {
        "s": "e",
        "v": "ACTION4",
        "a": "open",
        "e": "actor_displaced",
        "z": 0,
    }

    rule = compile_compact_rule(payload, query)

    assert rule.support == 0
    assert rule.matches_query(query)
    with pytest.raises(ValueError, match="grounded"):
        compile_compact_rule({**payload, "v": "ACTION1"}, query)


def test_compact_qwen_schema_is_closed_and_bounded():
    schema = compact_qwen_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["h"]["maxItems"] == 8
    assert (
        schema["properties"]["h"]["items"]["properties"]["z"]["const"]
        == 0
    )


def test_frozen_v4_1_manifest_checksum_is_valid():
    manifest = load_frozen_manifest()

    assert manifest["collection"]["policy_seeds"] == [307, 347, 389, 433]
    assert manifest["qwen"]["preflight_maximum_input_tokens"] == 384
    assert manifest["world_model_fit_authorized"] is False
