from __future__ import annotations

from theory.sage12.compiler import (
    SLOT_EFFECTS,
    SemanticActionSlot,
    SlotAnnotation,
)
from theory.sage12.descriptive_semantic_integration_v4_12 import ACTIVE_EFFECTS
from theory.sage12.integration_pilot_v4_7 import SlotExample
from theory.sage12.semantic_bottleneck_curve_v4_13 import (
    CORRUPTED_EFFECTS,
    NOISE_LEVELS,
    _corrupt_oracle,
    _spearman,
    freeze_manifest,
)


def _oracle_example(index: int = 0) -> tuple[SlotExample, SlotAnnotation]:
    labels = {
        effect: bool((offset + index) % 2)
        for offset, effect in enumerate(SLOT_EFFECTS)
    }
    slot = SemanticActionSlot(
        slot_id=f"slot-{index}",
        action_name="ACTION1",
        action_data={},
        semantic_signature={
            **{"binding_kind": "targetless"},
            **{
                f"v412.effect.{effect}": (
                    float(labels[effect])
                    if effect in labels
                    else float((offset + index) % 2)
                )
                for offset, effect in enumerate(ACTIVE_EFFECTS)
            },
        },
    )
    example = SlotExample(
        example_id=f"example-{index}",
        node_id=f"node-{index}",
        root_key=f"root-{index}",
        game_id="bp35",
        path="",
        side="L",
        context=(),
        slot=slot,
        labels=labels,
        applicable={effect: True for effect in SLOT_EFFECTS},
        utility=0.0,
    )
    annotation = SlotAnnotation(
        slot_id=slot.slot_id,
        effect_probabilities={
            effect: float(labels[effect]) for effect in SLOT_EFFECTS
        },
        source="oracle",
        support=0,
    )
    return example, annotation


def test_manifest_runs_complete_curve_without_semantic_gate(tmp_path) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)

    assert manifest["evaluation"]["all_conditions_run_unconditionally"]
    assert tuple(manifest["source_games"])
    assert manifest["evaluation"]["trajectory_depth"] == 3
    assert manifest["evaluation"]["live_win_rate_claimed"] is False
    assert set(NOISE_LEVELS) == {
        "oracle_100",
        "oracle_90",
        "oracle_75",
        "oracle_50",
    }


def test_oracle_corruption_is_exact_at_zero_and_complete_at_one() -> None:
    example, annotation = _oracle_example()
    annotations = {example.slot.slot_id: annotation}

    exact, exact_annotations, exact_summary = _corrupt_oracle(
        (example,),
        annotations,
        flip_probability=0.0,
        seed=1,
        source="exact",
    )
    flipped, flipped_annotations, flipped_summary = _corrupt_oracle(
        (example,),
        annotations,
        flip_probability=1.0,
        seed=1,
        source="flipped",
    )

    assert exact_summary["observed_bit_accuracy"] == 1.0
    assert flipped_summary["observed_bit_accuracy"] == 0.0
    for effect in ACTIVE_EFFECTS:
        key = f"v412.effect.{effect}"
        assert exact[0].slot.semantic_signature[key] == (
            example.slot.semantic_signature[key]
        )
        assert flipped[0].slot.semantic_signature[key] == (
            1.0 - example.slot.semantic_signature[key]
        )
    for effect in SLOT_EFFECTS:
        assert exact_annotations[example.slot.slot_id].effect_probabilities[
            effect
        ] == annotation.effect_probabilities[effect]
        assert flipped_annotations[example.slot.slot_id].effect_probabilities[
            effect
        ] == 1.0 - annotation.effect_probabilities[effect]


def test_oracle_corruption_is_deterministic_and_nested() -> None:
    examples = tuple(_oracle_example(index)[0] for index in range(50))
    annotations = {
        example.slot.slot_id: _oracle_example(index)[1]
        for index, example in enumerate(examples)
    }
    first = _corrupt_oracle(
        examples,
        annotations,
        flip_probability=0.10,
        seed=5_130,
        source="ten",
    )
    repeated = _corrupt_oracle(
        examples,
        annotations,
        flip_probability=0.10,
        seed=5_130,
        source="ten",
    )
    twenty_five = _corrupt_oracle(
        examples,
        annotations,
        flip_probability=0.25,
        seed=5_130,
        source="twenty-five",
    )

    assert first[2] == repeated[2]
    assert [item.slot.semantic_signature for item in first[0]] == [
        item.slot.semantic_signature for item in repeated[0]
    ]
    assert (
        twenty_five[2]["observed_bit_accuracy"]
        <= first[2]["observed_bit_accuracy"]
    )
    assert first[2]["bits"] == len(examples) * len(CORRUPTED_EFFECTS)


def test_spearman_reports_direction_of_semantic_curve() -> None:
    assert _spearman([1.0, 0.9, 0.75, 0.5], [8.0, 7.0, 6.0, 5.0]) == 1.0
    assert _spearman([1.0, 0.9, 0.75, 0.5], [5.0, 6.0, 7.0, 8.0]) == -1.0
