from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from theory.sage12 import mechanic_replication as v41
from theory.sage12.target_mechanic_replication import (
    TARGET_EFFECT_LABELS,
    TargetCalibrationBundle,
    TargetMechanicQuery,
    TargetMechanicWindowRecord,
    TargetTransitionEvent,
    _chat_token_count,
    _identity_probe,
    _window_quality,
    apply_calibration,
    coarsen_anchor,
    compact_qwen_prompt,
    compact_qwen_schema,
    compile_compact_rule,
    fit_source_calibration,
    load_frozen_manifest,
    multilabel_metrics,
    validate_model_view,
)
from theory.sage12.target_mechanic_replication_collection import (
    run_collection,
)


def _event(
    *,
    created: bool = False,
    removed: bool = False,
    moved: bool = True,
) -> TargetTransitionEvent:
    return TargetTransitionEvent(
        action_name="ACTION4",
        action_family="move",
        anchor_condition="occupied",
        effects={
            "target_created": created,
            "target_removed": removed,
            "target_moved": moved,
        },
        applicable={label: True for label in TARGET_EFFECT_LABELS},
        actor_role_known=True,
        actor_role_state="translational",
    )


def _window() -> TargetMechanicWindowRecord:
    return TargetMechanicWindowRecord(
        game_id="bp35",
        source_split="source_train",
        policy_seed=479,
        reset_index=0,
        query_step_index=8,
        context=tuple(_event() for _ in range(8)),
        query=TargetMechanicQuery("ACTION4", "move", "occupied"),
        labels={
            "target_created": False,
            "target_removed": False,
            "target_moved": True,
        },
        applicable={label: True for label in TARGET_EFFECT_LABELS},
        actor_role_known=True,
        actor_role_state="translational",
        excluded_actor_displaced=True,
        excluded_actor_applicable=True,
    )


def _source_windows() -> list[TargetMechanicWindowRecord]:
    path = Path(
        "training/sage12/mechanic_induction_v4_1/source_train_windows.jsonl"
    )
    return [
        TargetMechanicWindowRecord.from_v41(
            v41.MechanicWindowRecord.from_dict(json.loads(line))
        )
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_coarse_anchor_projection_is_closed_and_causal():
    assert coarsen_anchor("occupied_actor") == "occupied"
    assert coarsen_anchor("occupied_object") == "occupied"
    assert coarsen_anchor("empty") == "free"
    assert coarsen_anchor("open") == "free"
    assert coarsen_anchor("targetless") == "none"
    assert coarsen_anchor("unknown") == "none"
    with pytest.raises(ValueError, match="unsupported"):
        coarsen_anchor("game_specific")


def test_target_event_rejects_actor_effect():
    payload = _event().to_dict()
    payload["effects"]["actor_displaced"] = True

    with pytest.raises(ValueError, match="only target effects"):
        TargetTransitionEvent.from_dict(payload)


def test_window_model_view_excludes_actor_and_provenance():
    window = _window()

    validate_model_view(window)
    rendered = json.dumps(window.model_view(), sort_keys=True)

    assert "actor" not in rendered
    assert "game_id" not in rendered
    assert window.to_dict()["excluded_effect_audit"]["actor_displaced"]


def test_internal_adapter_pads_actor_as_inapplicable():
    internal = _window().as_internal()

    assert not internal.applicable["actor_displaced"]
    assert not internal.labels["actor_displaced"]
    assert all(
        not event.applicable["actor_displaced"] for event in internal.context
    )
    assert internal.query.anchor_condition == "occupied_object"


def test_window_round_trip_preserves_public_and_audit_views():
    window = _window()
    restored = TargetMechanicWindowRecord.from_dict(window.to_dict())

    assert restored.to_dict() == window.to_dict()
    assert restored.model_view() == window.model_view()


def test_target_calibration_bundle_is_three_label_only():
    parameters = {
        mode: {
            label: {"slope": 1.0, "intercept": 0.0}
            for label in TARGET_EFFECT_LABELS
        }
        for mode in v41.MODEL_MODES
    }
    thresholds = {
        mode: {label: 0.5 for label in TARGET_EFFECT_LABELS}
        for mode in v41.MODEL_MODES
    }
    bundle = TargetCalibrationBundle(
        parameters=parameters,
        thresholds=thresholds,
        source_oof_metrics={},
    )

    restored = TargetCalibrationBundle.from_dict(bundle.to_dict())

    assert restored.calibration_checksum == bundle.calibration_checksum
    assert apply_calibration(
        np.full((2, 3), 0.25),
        restored,
        "structured",
    ).shape == (2, 3)


def test_target_metrics_have_no_actor_dimension():
    targets = np.asarray([[1, 0, 1], [0, 0, 0]], dtype=np.int8)
    masks = np.ones_like(targets)
    probabilities = np.asarray([[0.9, 0.1, 0.8], [0.1, 0.1, 0.2]])

    metrics = multilabel_metrics(targets, masks, probabilities)

    assert tuple(metrics["per_label"]) == TARGET_EFFECT_LABELS
    assert "actor_displaced" not in metrics["per_label"]


def test_compact_qwen_contract_excludes_actor():
    window = _window()
    prompt = compact_qwen_prompt(window)
    schema = compact_qwen_schema()

    assert "actor" not in prompt
    assert "actor_displaced" not in json.dumps(schema)
    assert schema["properties"]["h"]["maxItems"] == 8
    assert schema["properties"]["h"]["items"]["properties"]["a"]["enum"] == [
        "occupied",
        "free",
        "none",
    ]


def test_compact_qwen_rule_is_target_only_and_grounded():
    query = TargetMechanicQuery("ACTION4", "move", "occupied")
    payload = {
        "s": "e",
        "v": "ACTION4",
        "a": "occupied",
        "e": "target_moved",
        "z": 0,
    }

    rule = compile_compact_rule(payload, query)

    assert rule.effect == "target_moved"
    assert rule.anchor_condition == "occupied_object"
    with pytest.raises(ValueError, match="invalid effect"):
        compile_compact_rule(
            {**payload, "e": "actor_displaced"},
            query,
        )


def test_token_counter_accepts_transformers_batch_encoding():
    class FakeTensor:
        shape = (1, 300)

    class FakeTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return {"input_ids": FakeTensor()}

    assert _chat_token_count(FakeTokenizer(), _window()) == 300


def test_source_projection_matches_preregistered_diagnostics():
    windows = _source_windows()
    identity = _identity_probe(windows)
    quality = _window_quality(windows)
    calibration = fit_source_calibration(windows)
    structured = calibration.source_oof_metrics["calibrated"]["structured"]

    assert len(windows) == 1911
    assert identity["gain"] == pytest.approx(0.03872318158032445)
    assert structured["macro_brier"] == pytest.approx(0.04791919560534961)
    assert structured["macro_f1"] == pytest.approx(0.6890683251585507)
    assert quality["per_label"]["target_created"]["positives"] == 87
    assert quality["excluded_effect_audit"]["positives"] == 35


def test_collection_fails_closed_without_passing_preflight(tmp_path):
    with pytest.raises(RuntimeError, match="preflight is missing"):
        run_collection(output_dir=tmp_path)


def test_frozen_v4_2_manifest_checksum_and_authority():
    manifest = load_frozen_manifest()

    assert manifest["collection"]["policy_seeds"] == [479, 523, 569, 617]
    assert tuple(manifest["effects"]["authoritative"]) == TARGET_EFFECT_LABELS
    assert manifest["effects"]["diagnostic_only"] == ["actor_displaced"]
    assert manifest["world_model_fit_authorized"] is False


def test_v4_1_published_preflight_checksum_is_unchanged():
    payload = json.loads(
        Path(
            "training/sage12/mechanic_induction_v4_1/"
            "source_train_preflight.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["preflight_checksum"] == (
        "cffa41e2ae980f64dfc76cbe40076809b301da4e8f98dffbc02122eb2bfa147c"
    )
    assert v41.load_frozen_manifest()["manifest_checksum"] == (
        "86b3d3b38ba41d0f860169928f6cc5afd6765ccdbf83078e3a09d60da0e07abc"
    )
