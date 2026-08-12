from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from theory.sage_t.causal import representation_experiment, representation_protocol
from theory.sage_t.causal.archive import abstract_state_to_payload
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.lineage_archive import LineagePreservingArchive
from theory.sage_t.causal.relational_novelty import (
    ARCHIVE_CONTEXT_DIM,
    RELATIONAL_DIM,
    RELATIONAL_FEATURE_DIM,
    ArchiveContext,
    encode_action_entity_relations,
    encode_relational_state_action,
)
from theory.sage_t.causal.representation_experiment import (
    compile_archive_examples,
    train_representation_experiment,
)
from theory.sage_t.causal.representation_experiment_cli import build_parser
from theory.sage_t.causal.representation_protocol import (
    _parent_failure_is_representation_only,
    load_representation_manifest,
)
from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _state(index: int) -> AbstractState:
    return AbstractState(
        entities=(
            AbstractEntity(
                "player",
                ("object", "player"),
                (("area", "one"),),
                (float(index), float(index)),
            ),
            AbstractEntity(
                "target",
                ("object", "target"),
                (("area", "small"),),
                (5.0, 7.0),
            ),
        ),
        true_facts=frozenset({GroundFact("changed", (f"e{index}",))}),
    )


def test_relational_encoder_uses_action_binding_and_pre_action_context() -> None:
    state = _state(0)
    on_target = GroundedAction("ACTION1", {"x": 7, "y": 5})
    elsewhere = GroundedAction("ACTION1", {"x": 40, "y": 40})
    first = ArchiveContext(
        cell_visits=1,
        action_attempts=0,
        legal_actions=4,
        archive_cells=1,
    )
    repeated = ArchiveContext(
        cell_visits=8,
        action_attempts=4,
        cell_expansions=7,
        unique_tried_actions=3,
        legal_actions=4,
        archive_cells=30,
        global_edges=100,
        global_action_trials=10,
        global_action_changed=8,
        global_action_novel=2,
        cell_action_trials=4,
        cell_action_changed=3,
        cell_action_novel=1,
    )
    target_relations = encode_action_entity_relations(state, on_target)
    elsewhere_relations = encode_action_entity_relations(state, elsewhere)
    assert len(target_relations) == RELATIONAL_DIM
    assert target_relations != elsewhere_relations
    assert target_relations[15] == 1.0
    assert target_relations[20] == 1.0
    assert len(first.encode()) == ARCHIVE_CONTEXT_DIM
    assert first.encode() != repeated.encode()
    encoded = encode_relational_state_action(state, on_target, first)
    assert len(encoded) == RELATIONAL_FEATURE_DIM
    assert encoded != encode_relational_state_action(state, on_target, repeated)


def test_archive_examples_reconstruct_strictly_pre_action_context() -> None:
    archive = LineagePreservingArchive(maximum_cells=8, seed=1)
    action = GroundedAction("ACTION1", {"x": 7, "y": 5})
    root, _ = archive.observe_state(
        state=_state(0),
        exact_hash="raw-0",
        level=0,
        legal_actions=(action,),
    )
    first = archive.add_lineage_transition(
        source_cell_id=root.cell_id,
        source_exact_hash="raw-0",
        source_prefix_id=root.variants["raw-0"].prefix_id,
        source_path_edge_ids=(),
        action=action,
        target_state=_state(1),
        target_exact_hash="raw-1",
        target_level=0,
        target_legal_actions=(action,),
        terminal=False,
        success=False,
        changed=True,
    )
    target = archive.cells[first.target_cell_id]
    archive.add_lineage_transition(
        source_cell_id=target.cell_id,
        source_exact_hash="raw-1",
        source_prefix_id=first.prefix_id,
        source_path_edge_ids=(first.edge_id,),
        action=action,
        target_state=_state(0),
        target_exact_hash="raw-0",
        target_level=0,
        target_legal_actions=(action,),
        terminal=False,
        success=False,
        changed=True,
    )
    _, examples = compile_archive_examples(archive, seed=8401, split="train")
    assert len(examples) == 2
    assert examples[0]["novel"] is True
    assert examples[0]["archive_context"]["global_edges"] == 0
    assert examples[0]["archive_context"]["global_action_trials"] == 0
    assert examples[1]["novel"] is False
    assert examples[1]["archive_context"]["global_edges"] == 1
    assert examples[1]["archive_context"]["global_action_trials"] == 1


def test_real_t12_4_failure_is_exactly_representation_and_calibration() -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "neural_novelty_t12_4_bp35"
    manifest = representation_protocol.load_neural_novelty_manifest(
        parent / "manifest.json",
        root=repo,
    )
    receipt = representation_protocol.load_neural_novelty_receipt(
        parent / "training" / "training_receipt.json",
        manifest=manifest,
        root=repo,
    )
    assert _parent_failure_is_representation_only(manifest, receipt)
    assert receipt["metrics"]["brier_gain"] >= manifest["protocol"][
        "minimum_brier_gain"
    ]
    assert receipt["metrics"]["state_shuffle_degradation"] < 0.0
    assert receipt["metrics"]["maximum_ece"] > manifest["protocol"][
        "maximum_ece"
    ]


def test_freeze_excludes_opened_seed_and_keeps_all_authority_closed(
    monkeypatch, tmp_path
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "neural_novelty_t12_4_bp35"
    monkeypatch.setattr(
        representation_protocol,
        "_git_state",
        lambda root: {"commit": "a" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = representation_protocol.freeze_representation_experiment(
        output_path=manifest_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "training" / "training_receipt.json",
        root=repo,
    )
    loaded = load_representation_manifest(
        manifest_path,
        root=repo,
        verify_code=False,
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["diagnostic_exclusions"]["opened_seed"] == 7703
    assert 7703 not in loaded["protocol"]["collection_seeds"]
    assert loaded["parent"]["receipt"]["failure_class"] == (
        "STATE_AND_ARCHIVE_CONTEXT_REPRESENTATION"
    )
    assert loaded["firewall"]["representation_collection_authorized"] is True
    assert loaded["firewall"]["representation_training_authorized"] is False
    assert loaded["firewall"]["neural_active_evaluation_authorized"] is False
    assert loaded["firewall"]["option_extraction_authorized"] is False
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3


def test_training_phase_compares_relational_legacy_and_action_controls(
    monkeypatch, tmp_path
) -> None:
    states = {
        "s0": abstract_state_to_payload(_state(0)),
        "s1": abstract_state_to_payload(_state(1)),
    }
    examples = []
    for index in range(96):
        action = GroundedAction(
            f"ACTION{1 + index % 2}",
            {"x": 7 if index % 2 else 40, "y": 5 if index % 2 else 40},
        )
        examples.append(
            {
                "split": "train" if index < 64 else "validation",
                "source_state_id": f"s{index % 2}",
                "action": {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                },
                "action_key": action.key,
                "archive_context": ArchiveContext(
                    cell_visits=1 + index % 8,
                    action_attempts=index % 4,
                    cell_expansions=index % 9,
                    unique_tried_actions=index % 2,
                    legal_actions=2,
                    archive_cells=1 + index,
                    global_edges=index,
                    global_action_trials=index % 12,
                    global_action_changed=index % 6,
                    global_action_novel=index % 4,
                    cell_action_trials=index % 4,
                    cell_action_changed=index % 3,
                    cell_action_novel=index % 2,
                ).to_dict(),
                "semantic_changed": bool(index % 2),
                "novel": bool((index // 2) % 2),
                "example_id": f"example-{index}",
            }
        )
    dataset = {
        "states": states,
        "examples": examples,
        "dataset_checksum": "d" * 64,
    }
    manifest = {
        "manifest_checksum": "m" * 64,
        "protocol_checksum": "p" * 64,
        "protocol": {},
        "parent": {"receipt": {"receipt_checksum": "r" * 64}},
    }
    collection_receipt = {
        "receipt_checksum": "c" * 64,
        "status": "PASS_T12_4A_COLLECTION_GATE",
        "artifacts": {
            "dataset": {
                "path": "unused.json",
                "sha256": "d" * 64,
                "dataset_checksum": "d" * 64,
            }
        },
    }
    protocol = SimpleNamespace(
        hidden_dim=8,
        batch_size=16,
        training_epochs=1,
        learning_rate=1e-3,
        torch_seed=1,
        minimum_training_examples=1,
        minimum_validation_examples=1,
        maximum_parameters=15_000,
        minimum_change_brier_gain=-1.0,
        minimum_novelty_brier_gain=-1.0,
        minimum_legacy_mean_brier_improvement=-1.0,
        minimum_state_shuffle_change_degradation=-1.0,
        minimum_context_shuffle_novelty_degradation=-1.0,
        minimum_relation_ablation_degradation=-1.0,
        maximum_ece=1.0,
        maximum_artifact_bytes_per_run=3 * 1024**3,
    )
    monkeypatch.setattr(
        representation_experiment,
        "load_representation_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        representation_experiment,
        "load_representation_receipt",
        lambda *args, **kwargs: collection_receipt,
    )
    monkeypatch.setattr(
        representation_experiment,
        "load_representation_dataset",
        lambda path, protocol: dataset,
    )
    monkeypatch.setattr(
        representation_experiment,
        "RepresentationProtocol",
        lambda **kwargs: protocol,
    )
    report = train_representation_experiment(
        manifest_path="unused.json",
        collection_receipt_path="unused-receipt.json",
        output_dir=tmp_path / "training",
    )
    assert report["passed"] is True
    assert "legacy_mean_brier_improvement" in report["metrics"]
    assert "state_shuffle_change_degradation" in report["metrics"]
    assert "context_shuffle_novelty_degradation" in report["metrics"]
    assert "relation_ablation_degradation" in report["metrics"]
    assert (tmp_path / "training" / "representation_receipt.json").is_file()


def test_cli_exposes_no_active_or_option_phase() -> None:
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["collect"]).phase == "collect"
    assert parser.parse_args(["train"]).phase == "train"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["extract-option"])
