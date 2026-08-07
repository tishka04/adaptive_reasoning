from __future__ import annotations

import math
from types import SimpleNamespace

from theory.sage_t.contracts import PredictionPacket
from theory.sage_t.posterior import DEFAULT_CHANNEL_WEIGHTS, packet_log_likelihood
from theory.sage_t.replay_gate import (
    episodes_from_binding_pairs,
)
from theory.sage_t.replay_gate import (
    load_frozen_manifest as load_t7_manifest,
)
from theory.sage_t.selection_autopsy import (
    SelectionAutopsyRow,
    build_report,
    count_source_signals,
    load_frozen_manifest,
    run_episode,
)


def _trace(
    action: str,
    *,
    moved: bool = False,
    created: bool = False,
    level_complete: bool = False,
):
    return SimpleNamespace(
        selected_action_name=action,
        selected_action_data={},
        available_action_names=("ACTION1", "ACTION2"),
        levels_completed_before=0,
        levels_completed_after=int(level_complete),
        anchor=SimpleNamespace(
            target_affordance="movable",
            action_family="move",
            kind="move_destination",
            occupied=False,
            path_status="open",
            actor_relation="adjacent",
            in_bounds=True,
            row=1,
            col=2,
        ),
        effects=SimpleNamespace(
            labels={
                "actor_displaced": moved,
                "target_created": created,
                "target_moved": False,
                "target_removed": False,
            },
            noop=not (moved or created or level_complete),
            level_complete=level_complete,
            game_over=False,
        ),
    )


def _pair(index: int):
    return SimpleNamespace(
        source_split="source_train",
        game_id="game",
        pair_digest=f"pair_{index}",
        root_key="game:1:0:0",
        depth=0 if index == 0 else 1 if index < 3 else 2,
        path="" if index == 0 else str(index),
        left=SimpleNamespace(trace=_trace("ACTION1", moved=True)),
        right=SimpleNamespace(trace=_trace("ACTION2", created=True)),
    )


def test_manifest_is_bound_to_frozen_t7_baseline() -> None:
    manifest = load_frozen_manifest()
    base = load_t7_manifest()

    assert manifest["base_t7_manifest_checksum"] == base["manifest_checksum"]
    assert manifest["common_evaluation_weights"] == dict(
        DEFAULT_CHANNEL_WEIGHTS
    )
    assert manifest["firewall"]["holdout_opened"] is False


def test_native_ablation_rewards_omitting_a_correct_channel() -> None:
    predicted = PredictionPacket(
        progress_mean=0.0,
        progress_distribution={"value:0": 0.95, "other": 0.05},
        known_channels=frozenset({"progress"}),
    )
    observed = PredictionPacket(
        progress_mean=0.0,
        progress_distribution={"value:0": 1.0},
        known_channels=frozenset({"progress"}),
    )
    dynamics_weights = {**DEFAULT_CHANNEL_WEIGHTS, "progress": 0.0, "goal": 0.0}

    joint_score = packet_log_likelihood(
        predicted,
        observed,
        channel_weights=DEFAULT_CHANNEL_WEIGHTS,
    )
    dynamics_score = packet_log_likelihood(
        predicted,
        observed,
        channel_weights=dynamics_weights,
    )

    assert joint_score == 2.0 * math.log(0.95)
    assert dynamics_score == 0.0
    assert joint_score < dynamics_score


def test_episode_autopsy_emits_common_and_native_scores_at_all_checkpoints() -> None:
    episode = episodes_from_binding_pairs(tuple(_pair(index) for index in range(5)))[
        0
    ]

    rows = run_episode(
        episode,
        base_manifest=load_t7_manifest(),
        autopsy_manifest=load_frozen_manifest(),
        seed=857,
    )

    assert [row.observations for row in rows] == [1, 3, 5]
    assert all(math.isfinite(row.joint_common_log_likelihood) for row in rows)
    assert all(math.isfinite(row.dynamics_common_log_likelihood) for row in rows)
    assert all(row.best_family_rank is not None for row in rows)
    assert all(row.generated_programs >= row.posterior_programs for row in rows)


def _autopsy_row(
    *,
    episode: str,
    game: str,
    observations: int,
) -> SelectionAutopsyRow:
    return SelectionAutopsyRow(
        episode_id=episode,
        source_game=game,
        observations=observations,
        joint_native_log_likelihood=-1.1,
        dynamics_native_log_likelihood=-1.0,
        joint_common_log_likelihood=-1.0,
        dynamics_common_log_likelihood=-2.0,
        joint_common_minus_dynamics=1.0,
        native_joint_minus_dynamics=-0.1,
        joint_channel_scores={"progress": -0.05},
        dynamics_channel_scores={"progress": -1.0},
        best_generated_log_likelihood=-0.9,
        joint_selection_regret=0.1,
        best_exact_rank=2,
        best_exact_probability=0.2,
        best_family_rank=1,
        best_family_probability_mass=0.5,
        best_exact_pruned=False,
        best_family_pruned=False,
        top_prior_minus_best_prior=0.3,
        top_evidence_minus_best_evidence=-0.2,
        generated_programs=64,
        posterior_programs=16,
        revealed_teleological_positives=0,
        hidden_teleological_positives=0,
        repairs_attempted=0,
        repairs_admitted=0,
    )


def test_report_detects_scoring_bias_and_blocks_underidentified_validation() -> None:
    manifest = dict(load_frozen_manifest())
    manifest["source_train_games"] = ["game"]
    rows = tuple(
        _autopsy_row(
            episode=f"root_{root}",
            game="game",
            observations=checkpoint,
        )
        for root in range(4)
        for checkpoint in (1, 3, 5)
    )

    report = build_report(
        rows,
        manifest=manifest,
        signal_counts={
            "arms": 100,
            "progress_positive": 3,
            "goal_positive": 3,
            "terminal_positive": 5,
        },
    )

    assert report["diagnosis"] == (
        "ablation_scoring_bias_plus_teleological_underidentification"
    )
    assert report["checks"]["common_score_noninferior"] is True
    assert report["checks"]["native_ablation_rejects_joint"] is True
    assert report["checks"]["teleological_support_sufficient"] is False
    assert report["source_validation_authorized"] is False
    assert report["status"] == "DIAGNOSIS_COMPLETE_FAIL_CLOSED"


def test_signal_count_uses_all_counterfactual_arms() -> None:
    pair = _pair(0)
    pair.right.trace.effects.level_complete = True
    pair.right.trace.levels_completed_after = 1

    counts = count_source_signals((pair,))

    assert counts["arms"] == 2
    assert counts["progress_positive"] == 1
    assert counts["goal_positive"] == 1
