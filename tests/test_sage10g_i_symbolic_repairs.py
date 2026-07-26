"""SAGE.10g-i multi-source, chain-economics, and effect-bridge tests."""

from __future__ import annotations

from theory.online_causal_subgoal_graph import OnlineCausalSubgoalGraph
from theory.online_transferable_causal_schema import (
    CausalEffectTemplate,
    CausalSchemaProvenance,
    FrozenCausalSchemaLibrary,
    TransferableCausalSchema,
    TransferableCausalSchemaStep,
    merge_frozen_causal_schema_libraries,
)
from theory.sage11.curriculum import FrozenSchemaCurriculum


def _library(source: str, support: int = 1) -> FrozenCausalSchemaLibrary:
    effect = CausalEffectTemplate(
        family="object_count",
        predicate="decreases",
        direction="down",
    )
    step = TransferableCausalSchemaStep(
        action_family="directional",
        target_role="global",
        effects=(effect,),
        next_subgoal="reduce_count",
    )
    schema = TransferableCausalSchema(
        schema_id="",
        steps=(step,),
        terminal_support=support,
        provenance=(
            CausalSchemaProvenance(
                source_tag=source,
                terminal_context=f"{source}-terminal",
            ),
        ),
    )
    payload = schema.to_dict()
    payload.pop("schema_id")
    normalized = TransferableCausalSchema.from_dict(payload)
    return FrozenCausalSchemaLibrary((normalized,))


def test_multi_source_library_coalesces_schema_and_preserves_provenance():
    merged = merge_frozen_causal_schema_libraries(
        [_library("bp35"), _library("cd82", support=2)],
        allowed_source_tags=("bp35", "cd82"),
        forbidden_source_tags=("s5i5",),
    )
    assert len(merged.schemas) == 1
    assert merged.source_tags == ("bp35", "cd82")
    assert merged.schemas[0].terminal_support == 3
    assert len(merged.content_checksum) == 64


def test_multi_source_library_rejects_holdout_contamination():
    try:
        merge_frozen_causal_schema_libraries(
            [_library("s5i5")],
            allowed_source_tags=("bp35",),
            forbidden_source_tags=("s5i5",),
        )
    except ValueError as error:
        assert "forbidden sources" in str(error)
    else:
        raise AssertionError("holdout provenance must fail closed")


def test_curriculum_freezes_only_registered_training_sources():
    curriculum = FrozenSchemaCurriculum.build({
        "bp35": _library("bp35"),
        "cd82": _library("cd82"),
    })
    manifest = curriculum.to_dict()
    assert manifest["frozen"] is True
    assert manifest["source_games"] == ["bp35", "cd82"]
    assert manifest["holdout_sources_present"] == []
    assert len(manifest["checksum"]) == 64


def test_confirmed_transfer_effect_enters_graph_without_edge_or_terminal_credit():
    graph = OnlineCausalSubgoalGraph()
    confirmed = graph.note_confirmed_external_effect(
        ("('object_count', 'decreases', 'down')",),
        intervention_signature="ACTION1",
        context_signature="branch-1",
    )
    assert confirmed
    utilities = graph.confirmed_external_effect_utilities()
    assert utilities[confirmed[0]] == 0.5
    summary = graph.summary()
    assert summary["confirmed_external_effects"] == 1
    assert summary["confirmed_edges"] == 0
