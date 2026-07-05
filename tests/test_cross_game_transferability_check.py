"""A29 cross-game transferability tests."""

from __future__ import annotations

from pathlib import Path

from theory.cross_game_transferability_check import (
    analogous_target_contexts,
    run_cross_game_transferability_check,
)
from theory.non_ar25_multi_relation_agenda import NonAr25RelationAgendaItem
from theory.non_ar25_transferability_model import (
    NegativeTransferabilityModel,
    TransferabilityGroup,
    TransferabilityModelRunResult,
)
from theory.relation_option import AvoidRelationOption, PrepareRelationOption


def test_analogous_target_contexts_map_by_predicate_without_proof():
    source_run = TransferabilityModelRunResult(
        game_id="ft09-0d8bbf25",
        trace_path=Path("human_traces/ft09.steps.jsonl"),
        transferability_model=NegativeTransferabilityModel(
            groups=[
                TransferabilityGroup(
                    predicate="same_shape",
                    predicted_outcome="broken",
                    context_signature="ACTION6::same_shape::colors9_8",
                    count=2,
                )
            ],
            min_negative_count=2,
        ),
    )
    target_items = [
        NonAr25RelationAgendaItem(
            option=AvoidRelationOption(
                "same_shape",
                desired_outcome="absent",
                target_pair_colors=(8, 0),
                name="avoid_relation_same_shape_absent_colors8_0",
            ),
            required_predicate="relation_absent::same_shape::colors8_0",
            predicate="same_shape",
            pair_colors=(8, 0),
            action="ACTION6",
            observed_initially=True,
        ),
        NonAr25RelationAgendaItem(
            option=PrepareRelationOption(
                "aligned_with",
                desired_outcome="preserved",
                target_pair_colors=(8, 0),
                name="prepare_relation_aligned_with_colors8_0",
            ),
            required_predicate="relation_present::aligned_with::colors8_0",
            predicate="aligned_with",
            pair_colors=(8, 0),
            action="ACTION6",
            observed_initially=True,
        ),
    ]

    assert analogous_target_contexts(source_run, target_items) == [
        "ACTION6::same_shape::colors8_0"
    ]


def test_cross_game_transferability_downweights_analogous_context_only():
    result = run_cross_game_transferability_check(
        source_trace_path=Path(
            "human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"
        ),
        target_game_id="dc22-4c9bff3e",
        target_trace_path=Path(
            "human_traces/dc22-4c9bff3e.20260616-150906.steps.jsonl"
        ),
        max_candidates=20,
        min_pixel_support=1,
        repeated_observations=2,
        min_negative_count=2,
    )

    assert result.error == ""
    assert result.negative_model_applied_cross_game
    assert result.analogous_context_downweighted
    assert result.unrelated_context_still_testable
    assert result.source_confirmations_unchanged
    assert result.functional_progress_still_measurable
    assert result.wrong_confirmations == 0
    assert result.trace_support_counted_as_proof is False
    assert result.analogous_context_signatures == [
        "ACTION6::same_shape::colors8_0"
    ]

    assert result.target_baseline is not None
    assert "avoid_relation_same_shape_absent_colors8_0" in (
        result.target_baseline.selected_option_order
    )

    assert result.target_adapted is not None
    assert result.target_adapted.excluded_relation_context_signatures == [
        "ACTION6::same_shape::colors8_0"
    ]
    assert "avoid_relation_same_shape_absent_colors8_0" not in (
        result.target_adapted.selected_option_order
    )
    assert result.target_adapted.active_transition_produced
    assert result.target_adapted.hypothesis_remains_unresolved_until_observation
