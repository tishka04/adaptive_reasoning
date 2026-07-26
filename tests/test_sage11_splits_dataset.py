"""SAGE.11 split firewall and dataset-policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage11.dataset import (
    MINIMUM_STRONG_TERMINAL_EVENTS,
    MixturePolicy,
    NeuroTransition,
    ProgressLabels,
    Sage11ControllerCollector,
    Sage11DatasetBuilder,
    verify_manifest,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)
from theory.sage11.splits import (
    ArtifactPurpose,
    NEURO_HOLDOUT_V1,
    SAGE11_SPLITS,
)


def _transition(
    *,
    game_id: str = "bp35",
    step: int = 0,
    action_name: str = "ACTION1",
    action_data: dict | None = None,
    terminal: bool = False,
    weak: bool = False,
) -> NeuroTransition:
    return NeuroTransition(
        game_id=game_id,
        seed=0,
        reset_index=0,
        step_index=step,
        policy_arm="active_controller",
        action_name=action_name,
        action_data=dict(action_data or {}),
        atoms_before=(f"state:step({step})",),
        atoms_after=(f"state:step({step + 1})",),
        effect_atoms=("effect:changed_cells(one)",),
        changed=True,
        noop=False,
        unsafe=False,
        labels=ProgressLabels(
            terminal_event=terminal,
            level_completed=terminal,
            frontier_credit=weak,
        ),
    )


def test_split_registry_is_disjoint_immutable_and_content_addressed():
    groups = SAGE11_SPLITS.groups()
    flat = [game for games in groups.values() for game in games]
    assert len(flat) == 25
    assert len(set(flat)) == 25
    assert len(SAGE11_SPLITS.checksum) == 64
    assert set(NEURO_HOLDOUT_V1).isdisjoint(
        groups["historical_benchmark"]
    )


def test_training_artifact_touching_holdout_is_rejected():
    with pytest.raises(ValueError, match="leakage firewall"):
        SAGE11_SPLITS.assert_authorized(
            ["s5i5"],
            purpose=ArtifactPurpose.TRAIN,
        )
    SAGE11_SPLITS.assert_authorized(
        ["s5i5", "vc33"],
        purpose=ArtifactPurpose.HOLDOUT_CONFIRMATION,
    )


def test_mixture_policy_is_deterministic_and_exercises_every_arm():
    policy = MixturePolicy()
    first = [
        policy.arm_for(
            game_id="bp35",
            seed=0,
            reset_index=0,
            step_index=step,
        )
        for step in range(1000)
    ]
    second = [
        policy.arm_for(
            game_id="bp35",
            seed=0,
            reset_index=0,
            step_index=step,
        )
        for step in range(1000)
    ]
    assert first == second
    assert set(first) == {
        "active_controller",
        "uniform_legal",
        "frontier_stall_probe",
    }


def test_dataset_enforces_dedup_caps_and_action6_coverage(tmp_path: Path):
    builder = Sage11DatasetBuilder(
        target_transitions=2,
        per_game_cap=2,
    )
    click = _transition(
        step=0,
        action_name="ACTION6",
        action_data={"x": 2, "y": 3},
    )
    assert builder.add(click)
    assert not builder.add(click)
    assert builder.add(_transition(step=1))
    assert not builder.add(_transition(step=2))
    shard = builder.write_jsonl_shard(tmp_path / "train.jsonl")
    manifest = builder.manifest([shard])
    assert manifest.total_transitions == 2
    assert manifest.action6_argument_coverage["keys:x,y"] == 1
    verify_manifest(manifest)
    summary = builder.summary()
    assert summary["rejected_duplicates"] == 1
    assert summary["rejected_per_game_cap"] == 1


def test_dataset_manifest_records_and_enforces_amended_game_cap(
    tmp_path: Path,
):
    builder = Sage11DatasetBuilder(
        target_transitions=3,
        per_game_cap=2,
        game_caps={"bp35": 3},
    )
    for step in range(3):
        assert builder.add(_transition(step=step))
    assert not builder.add(_transition(step=3))

    shard = builder.write_jsonl_shard(tmp_path / "amended.jsonl")
    manifest = builder.manifest([shard])
    assert manifest.game_caps == {"bp35": 3}
    assert manifest.to_dict()["overflow_transitions"] == 1
    verify_manifest(manifest)


def test_terminal_head_counts_only_strong_labels():
    builder = Sage11DatasetBuilder(
        target_transitions=MINIMUM_STRONG_TERMINAL_EVENTS + 5,
        per_game_cap=MINIMUM_STRONG_TERMINAL_EVENTS + 5,
    )
    for step in range(MINIMUM_STRONG_TERMINAL_EVENTS - 1):
        assert builder.add(_transition(step=step, terminal=True))
    assert builder.add(_transition(
        step=MINIMUM_STRONG_TERMINAL_EVENTS,
        weak=True,
    ))
    manifest = builder.manifest()
    assert manifest.strong_terminal_events == (
        MINIMUM_STRONG_TERMINAL_EVENTS - 1
    )
    assert manifest.weak_progress_events == 1
    assert not manifest.terminal_head_enabled


def test_live_controller_transitions_archive_in_sage11_format():
    builder = Sage11DatasetBuilder(target_transitions=10)
    collector = Sage11ControllerCollector(
        builder,
        game_id="bp35",
        seed=0,
    )
    controller = UnifiedCognitiveController(
        "bp35",
        config=UnifiedCognitiveConfig(
            enable_frontier_oriented_exploration=False,
            enable_transferable_causal_schema_priors=False,
        ),
        neuro_transition_collector=collector,
    )
    import numpy as np

    grid = np.zeros((4, 4), dtype=np.int32)
    decision = controller.select_action(
        current_grid=grid,
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )
    controller.observe_transition(
        action=decision.action_name,
        action_data=decision.action_data,
        grid_before=grid,
        grid_after=grid.copy(),
        available_actions=["ACTION1"],
    )
    assert len(builder.records) == 1
    assert builder.records[0].format_version == "sage11-transition-v2"
    assert builder.records[0].source_split == "source_train"
    assert len(builder.records[0].state_digest_before) == 64
    assert builder.records[0].labels.strength == "negative"
