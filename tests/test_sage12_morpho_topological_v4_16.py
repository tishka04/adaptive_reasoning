from __future__ import annotations

import json

import numpy as np
import pytest

from theory.live_transition_loop import build_observation, build_transition_record
from theory.sage12 import (
    Sage12Config,
    Sage12Mode,
    SemanticActionCandidate,
    SemanticPlanningController,
)
from theory.sage12.morpho_topological_v4_16 import (
    FORMAT_VERSION,
    HUMAN_TRAIN_GAMES,
    TRANSFER_GAMES,
    _checksum,
    cluster_embeddings,
    compile_corpus,
    evaluate,
    freeze_manifest,
    load_manifest,
    load_records,
    prepare_shadow,
    train_model,
)
from theory.sage12.mt import (
    ClusterRegistry,
    MorphoTopologicalAnalogyAdvisor,
    MTModelConfig,
    SageMTConfig,
    SageMTMode,
    TransformationEmbedding,
    TransformationPrototype,
    TransformationPrototypeMemory,
    build_mt_graph,
    compile_mt_transition,
    encode_transitions,
    fit_cluster_registry,
    fit_mt_model,
    predict_graphs,
)


def _grid(*cells: tuple[int, int], value: int = 1, size: int = 9) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    for row, col in cells:
        grid[row, col] = value
    return grid


def test_graph_is_palette_translation_and_node_order_invariant() -> None:
    first = _grid((2, 2), (2, 3), (3, 2), value=3)
    second = _grid((4, 5), (4, 6), (5, 5), value=17)

    left = build_mt_graph(
        first,
        action_name="ACTION6",
        action_data={"row": 2, "col": 2},
    )
    right = build_mt_graph(
        second,
        action_name="ACTION6",
        action_data={"row": 4, "col": 5},
    )
    permuted = left.permuted(tuple(reversed(range(len(left.nodes)))))

    assert left.signature == right.signature
    assert left.signature == permuted.signature
    encoded = json.dumps(left.model_view(), sort_keys=True).lower()
    for forbidden in ("color", "value", "row", "col", "object_id"):
        assert f'"{forbidden}"' not in encoded


def test_transition_compiler_discovers_merge_split_growth_and_hole_change() -> None:
    separated = _grid((4, 2), (4, 4))
    merged = _grid((4, 2), (4, 3), (4, 4))
    fusion = compile_mt_transition(
        separated,
        "ACTION6",
        merged,
        action_data={"row": 4, "col": 3},
    )
    separation = compile_mt_transition(
        merged,
        "ACTION6",
        separated,
        action_data={"row": 4, "col": 3},
    )

    assert any(row.kind == "merge" for row in fusion.correspondences)
    assert any(event.startswith("merge#") for event in fusion.events)
    assert fusion.invariant_deltas["object_components"] == -1
    assert any(row.kind == "split" for row in separation.correspondences)
    assert separation.invariant_deltas["object_components"] == 1

    small = _grid((3, 3))
    larger = _grid((3, 3), (3, 4))
    growth = compile_mt_transition(small, "ACTION6", larger)
    assert any(event.startswith("growth#") for event in growth.events)

    ring = np.zeros((7, 7), dtype=np.int64)
    ring[2:5, 2:5] = 1
    ring[3, 3] = 0
    filled = ring.copy()
    filled[3, 3] = 1
    hole = compile_mt_transition(ring, "ACTION6", filled)
    assert hole.invariant_deltas["holes"] == -1
    assert any(
        event.startswith("morphology_changed#") for event in hole.events
    )


def test_transition_round_trip_keeps_student_teacher_and_audit_separate() -> None:
    before = _grid((2, 2))
    after = _grid((2, 2), (2, 3))
    record = compile_mt_transition(
        before,
        "ACTION6",
        after,
        source_game_id="fixture-secret",
        action_data={"row": 2, "col": 3},
        audit={"frame_before_sha256": "audit-only"},
    )
    payload = record.to_dict()
    restored = type(record).from_dict(payload)

    assert restored.delta_signature == record.delta_signature
    assert restored.graph_before.model_view() == record.graph_before.model_view()
    student = json.dumps(payload["student_view"], sort_keys=True).lower()
    assert "fixture" not in student
    assert "frame_before" not in student
    assert "row" not in student
    assert payload["audit"]["source_game_id"] == "fixture"


@pytest.mark.parametrize("device", ["cpu"])
def test_teacher_and_causal_encoder_train_and_ignore_node_order(device: str) -> None:
    pytest.importorskip("torch")
    records = []
    for game_index, game in enumerate(("g1", "g2", "g3")):
        for offset in range(2):
            before = _grid((2 + offset, 2))
            after = _grid((2 + offset, 2), (2 + offset, 3))
            records.append(
                compile_mt_transition(
                    before,
                    "ACTION6",
                    after,
                    source_game_id=game,
                )
            )
            before_motion = _grid((5, 2 + offset))
            after_motion = _grid((5, 3 + offset))
            records.append(
                compile_mt_transition(
                    before_motion,
                    "ACTION4",
                    after_motion,
                    source_game_id=game,
                )
            )
    config = MTModelConfig(
        embedding_width=8,
        hidden_width=24,
        latent_width=8,
        message_passing_layers=2,
        maximum_nodes=16,
        maximum_edges=64,
        epochs=2,
        batch_size=6,
        seed=17,
    )
    model, metadata = fit_mt_model(records, config=config, device=device)
    embeddings = encode_transitions(
        model,
        records,
        config=config,
        device=device,
    )
    original = records[0].graph_before
    permuted = original.permuted(tuple(reversed(range(len(original.nodes)))))
    predictions = predict_graphs(
        model,
        (original, permuted),
        config=config,
        device=device,
    )

    assert len(embeddings) == len(records)
    assert metadata["records"] == len(records)
    assert np.isfinite(metadata["final_loss"])
    assert np.dot(predictions[0][0], predictions[1][0]) > 0.999


def test_clustering_builds_multigame_prototypes_and_rejects_unknowns() -> None:
    rng = np.random.default_rng(5)
    embeddings = []
    records = []
    for family, center in (
        ("growth", np.asarray((1.0, 0.0))),
        ("motion", np.asarray((0.0, 1.0))),
    ):
        for index in range(12):
            game = f"g{index % 3}"
            if family == "growth":
                before = _grid((2, 2))
                after = _grid((2, 2), (2, 3))
                action = "ACTION6"
            else:
                before = _grid((5, 2))
                after = _grid((5, 3))
                action = "ACTION4"
            record = compile_mt_transition(
                before,
                action,
                after,
                source_game_id=game,
                productive=family == "growth",
                risk=False,
            )
            records.append(record)
            vector = center + rng.normal(0.0, 0.02, size=2)
            vector /= np.linalg.norm(vector)
            embeddings.append(
                TransformationEmbedding(
                    transition_id=record.transition_id + f":{family}:{index}",
                    vector=tuple(vector),
                    predicted_vector=tuple(vector),
                    uncertainty=0.1,
                    delta_signature=record.delta_signature,
                    source_game_id=game,
                )
            )
            # Clustering aligns records by transition id.
            object.__setattr__(
                records[-1],
                "transition_id",
                embeddings[-1].transition_id,
            )
    registry = fit_cluster_registry(
        embeddings,
        records,
        minimum_support=4,
        minimum_games=3,
        bootstrap_samples=3,
        parameter_grid=((4, 2),),
        seed=11,
    )
    memory = TransformationPrototypeMemory(registry)
    matches = memory.retrieve((1.0, 0.0), action_family="interact")

    assert len(registry.prototypes) >= 2
    assert registry.eligible_coverage == 1.0
    assert matches
    assert matches[0].candidate_alias in {"croissance", "transformation_mixte"}
    assert memory.assign((-1.0, 0.0), action_family="interact") == "unknown"
    before_support = memory.snapshot()["evidence"][matches[0].prototype_id][
        "support"
    ]
    memory.observe(matches[0].prototype_id, productive=True, risk=False)
    assert (
        memory.snapshot()["evidence"][matches[0].prototype_id]["support"]
        == before_support + 1
    )


class _StaticMTPredictor:
    def predict_graphs(self, graphs):
        return tuple(
            (
                (1.0, 0.0)
                if graph.action_name == "ACTION1"
                else (0.0, 1.0),
                0.1,
            )
            for graph in graphs
        )

    def encode_transition(self, record):
        return (0.0, 1.0)


def _registry() -> ClusterRegistry:
    return ClusterRegistry(
        prototypes=(
            TransformationPrototype(
                prototype_id="mt::productive",
                centroid=(1.0, 0.0),
                medoid_transition_id="p",
                assignment_threshold=0.8,
                dispersion=0.01,
                support=30,
                games=("g1", "g2", "g3"),
                action_families=("move",),
                dominant_delta_signatures=("growth:one",),
                enriched_events=("growth#1",),
                candidate_alias="croissance",
                productive_observations=25,
                risk_observations=1,
            ),
            TransformationPrototype(
                prototype_id="mt::risky",
                centroid=(0.0, 1.0),
                medoid_transition_id="r",
                assignment_threshold=0.8,
                dispersion=0.01,
                support=30,
                games=("g1", "g2", "g3"),
                action_families=("move",),
                dominant_delta_signatures=("death:two",),
                enriched_events=("death#1",),
                candidate_alias="disparition",
                productive_observations=1,
                risk_observations=25,
            ),
        ),
        labels_by_transition={},
        noise_transition_ids=(),
        selected_parameters={"min_cluster_size": 4, "min_samples": 2},
        stability_ari=1.0,
        eligible_coverage=1.0,
    )


def test_controller_runs_mt_in_shadow_without_changing_symbolic_action() -> None:
    memory = TransformationPrototypeMemory(_registry())
    advisor = MorphoTopologicalAnalogyAdvisor(
        model=_StaticMTPredictor(),
        model_config=MTModelConfig(latent_width=2),
        memory=memory,
        config=SageMTConfig(mode=SageMTMode.SHADOW),
    )
    controller = SemanticPlanningController(
        config=Sage12Config(mode=Sage12Mode.OFF),
        transformation_advisor=advisor,
    )
    grid = _grid((4, 4))
    observation = build_observation(
        grid,
        available_actions=("ACTION1", "ACTION2"),
        infer_players=False,
    )
    decision = controller.arbitrate(
        symbolic_action_name="ACTION2",
        symbolic_action_data={},
        symbolic_source="symbolic",
        observation=observation,
        candidates=(
            SemanticActionCandidate("ACTION1"),
            SemanticActionCandidate("ACTION2"),
        ),
        protected_competence_available=False,
        danger_veto=lambda *_: False,
    )

    assert decision.action_name == "ACTION2"
    assert decision.applied is False
    assert decision.mt_suggested_action == "ACTION1"
    assert decision.mt_advisory_id

    after = grid.copy()
    after[4, 5] = 1
    transition = build_transition_record(
        action="ACTION2",
        grid_before=grid,
        grid_after=after,
        available_actions=("ACTION1", "ACTION2"),
        infer_players=False,
    )
    controller.observe_transition(transition)
    summary = controller.summary()["sage_mt"]
    assert summary["observations"] == 1
    assert summary["memory"]["evidence"]["mt::risky"]["support"] == 31


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_v416_freeze_and_compile_write_versioned_firewalled_artifacts(
    tmp_path,
) -> None:
    human = tmp_path / "human"
    panels = tmp_path / "v411"
    for game in HUMAN_TRAIN_GAMES:
        before = _grid((2, 2)).tolist()
        after = _grid((2, 2), (2, 3)).tolist()
        _write_jsonl(
            human / f"{game}-fixture.steps.jsonl",
            (
                {
                    "game_id": f"{game}-fixture",
                    "episode_id": "episode",
                    "step": 1,
                    "frame_before": before,
                    "available_actions": ["ACTION6"],
                    "action": "ACTION6",
                    "action_args": {"x": 2, "y": 3},
                    "frame_after": after,
                    "game_state_after": "NOT_FINISHED",
                    "levels_completed_after": 0,
                },
            ),
        )
    (panels / "source_train_shards").mkdir(parents=True)
    for game in TRANSFER_GAMES:
        before = _grid((5, 2)).tolist()
        after = _grid((5, 3)).tolist()
        _write_jsonl(
            panels / "source_train_shards" / f"{game}.jsonl",
            (
                {
                    "format_version": "fixture",
                    "game_id": game,
                    "panel_id": f"{game}-panel",
                    "arms": [
                        {
                            "arm_index": 0,
                            "immediate_trace": {
                                "frame_before": before,
                                "frame_after": after,
                                "selected_action_name": "ACTION4",
                                "selected_action_data": {},
                                "game_state_after": "NOT_FINISHED",
                                "levels_completed_before": 0,
                                "levels_completed_after": 0,
                                "effects": {},
                            },
                        }
                    ],
                },
            ),
        )
    (panels / "frozen_manifest.json").write_text(
        json.dumps({"fixture": True}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    manifest = freeze_manifest(
        output_dir=output,
        human_traces_dir=human,
        v411_dir=panels,
    )
    qa = compile_corpus(
        output_dir=output,
        human_traces_dir=human,
        v411_dir=panels,
        maximum_train_records=2,
        maximum_transfer_records=2,
    )

    assert load_manifest(output)["manifest_checksum"] == manifest[
        "manifest_checksum"
    ]
    assert qa["student_view_safe"] is True
    assert len(load_records(output / "train_transitions.jsonl")) == 2
    assert len(load_records(output / "transfer_transitions.jsonl")) == 2
    assert qa["format_version"] == FORMAT_VERSION
    assert qa["active_validation_opened"] is False
    assert qa["holdout_opened"] is False

    # Exercise the complete artifact chain with a deliberately tiny frozen
    # model. The scientific defaults remain covered by the manifest test
    # above; this lane verifies orchestration and fail-closed output.
    frozen_path = output / "frozen_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["model"].update(
        {
            "embedding_width": 8,
            "hidden_width": 16,
            "latent_width": 8,
            "message_passing_layers": 1,
            "maximum_nodes": 16,
            "maximum_edges": 64,
            "epochs": 1,
            "batch_size": 4,
        }
    )
    frozen["clustering"].update(
        {
            "parameter_grid": [[2, 1]],
            "bootstrap_samples": 2,
            "minimum_support": 2,
            "minimum_games": 2,
        }
    )
    frozen.pop("manifest_checksum")
    frozen["manifest_checksum"] = _checksum(frozen)
    frozen_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    trained = train_model(output_dir=output, device="cpu")
    clusters = cluster_embeddings(output_dir=output)
    result = evaluate(output_dir=output, device="cpu")
    activation = prepare_shadow(output_dir=output)

    assert trained["embedding_rows"] == 2
    assert clusters["format_version"] == "sage12-mt-clusters-v4.16"
    assert result["format_version"] == "sage12-morpho-topological-result-v4.16"
    assert result["boundaries"]["controller_authority_promoted"] is False
    assert activation["controller_authority"] is False
