from __future__ import annotations

import json
from types import SimpleNamespace

from theory.sage_t.replay_gate import (
    StageReplayResult,
    build_replay_report,
    episodes_from_binding_pairs,
    fast_panel_from_binding_pair,
    load_frozen_manifest,
    paired_bootstrap_interval,
    run_all,
    run_replay_episodes,
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


def _pair(
    *,
    digest: str = "pair",
    root: str = "game:1:0:0",
    depth: int = 0,
    path: str = "",
):
    return SimpleNamespace(
        source_split="source_train",
        game_id="game",
        pair_digest=digest,
        root_key=root,
        depth=depth,
        path=path,
        left=SimpleNamespace(trace=_trace("ACTION1", moved=True)),
        right=SimpleNamespace(trace=_trace("ACTION2", created=True)),
    )


def test_frozen_manifest_verifies_checksum_coefficients_and_code() -> None:
    manifest = load_frozen_manifest()

    assert manifest["observation_checkpoints"] == [1, 3, 5]
    assert manifest["generator"]["maximum_programs"] == 64
    assert manifest["firewall"]["holdout_opened"] is False


def test_fast_v43_adapter_preserves_same_prestate_and_unknown_masks() -> None:
    panel = fast_panel_from_binding_pair(_pair())

    assert len(panel.arms) == 2
    assert {arm.action.action_name for arm in panel.arms} == {
        "ACTION1",
        "ACTION2",
    }
    assert all(arm.state_before.signature == panel.state.signature for arm in panel.arms)
    assert all("relations" not in arm.observation.known_channels for arm in panel.arms)
    assert all("topology" not in arm.observation.known_channels for arm in panel.arms)
    assert panel.arms[0].events == ("moved",)
    assert panel.arms[1].events == ("created",)


def test_root_episode_uses_first_five_depth_then_path_panels() -> None:
    pairs = (
        _pair(digest="d2b", depth=2, path="11"),
        _pair(digest="d1b", depth=1, path="1"),
        _pair(digest="d2a", depth=2, path="00"),
        _pair(digest="d0", depth=0, path=""),
        _pair(digest="d1a", depth=1, path="0"),
        _pair(digest="d2c", depth=2, path="01"),
    )

    episodes = episodes_from_binding_pairs(pairs)

    assert len(episodes) == 1
    assert [panel.panel_id for panel in episodes[0].panels] == [
        "d0",
        "d1a",
        "d1b",
        "d2a",
        "d2c",
    ]


def test_replay_emits_all_conditions_at_one_three_and_five_observations() -> None:
    pairs = tuple(
        _pair(
            digest=f"pair_{index}",
            depth=0 if index == 0 else 1 if index < 3 else 2,
            path="" if index == 0 else str(index),
        )
        for index in range(5)
    )
    episode = episodes_from_binding_pairs(pairs)[0]

    rows = run_replay_episodes((episode,), manifest=load_frozen_manifest())

    assert len(rows) == 12
    assert {(row.condition, row.observations) for row in rows} == {
        (condition, checkpoint)
        for condition in ("joint", "dynamics_only", "random", "action_only")
        for checkpoint in (1, 3, 5)
    }
    joint = [row for row in rows if row.condition == "joint"]
    assert all(row.correct_family_generated is not None for row in joint)
    assert all(row.decision_time_ms >= 0.0 for row in rows)


def test_paired_bootstrap_uses_intersection_and_preserves_pairing() -> None:
    interval = paired_bootstrap_interval(
        {"a": 3.0, "b": 5.0, "unpaired": 100.0},
        {"a": 1.0, "b": 3.0},
        samples=200,
        seed=857,
    )

    assert interval.n == 2
    assert interval.mean == 2.0
    assert interval.lower == 2.0
    assert interval.upper == 2.0


def _stage(
    episode: str,
    condition: str,
    *,
    log_likelihood: float,
    entropy: float,
    quality: float,
) -> StageReplayResult:
    return StageReplayResult(
        episode_id=episode,
        source_game="game",
        source_split="source_validation",
        condition=condition,
        observations=5,
        held_out_log_likelihood=log_likelihood,
        entropy_reduction=entropy,
        discriminative_action_quality=quality,
        decision_time_ms=1.0,
        repairs_attempted=0,
        repairs_admitted=0,
        generated_programs=64,
        posterior_programs=16,
        correct_family_generated=(False if condition == "joint" else None),
        correct_family_top1=(False if condition == "joint" else None),
        correct_family_top5=(False if condition == "joint" else None),
        correct_family_top16=(False if condition == "joint" else None),
        generation_failure=condition == "joint",
    )


def test_report_separates_generator_coverage_from_posterior_selection() -> None:
    rows = []
    for episode in ("root_a", "root_b"):
        rows.extend(
            (
                _stage(
                    episode,
                    "joint",
                    log_likelihood=-1.0,
                    entropy=0.8,
                    quality=0.9,
                ),
                _stage(
                    episode,
                    "dynamics_only",
                    log_likelihood=-2.0,
                    entropy=0.7,
                    quality=0.8,
                ),
                _stage(
                    episode,
                    "random",
                    log_likelihood=-1.5,
                    entropy=0.1,
                    quality=0.2,
                ),
                _stage(
                    episode,
                    "action_only",
                    log_likelihood=-3.0,
                    entropy=0.0,
                    quality=0.0,
                ),
            )
        )

    report = build_replay_report(
        rows,
        manifest=load_frozen_manifest(),
        expected_games=("game",),
    )

    assert report["status"] == "PASS"
    assert report["diagnosis"]["primary_bottleneck"] == "generator_coverage"
    assert report["diagnosis"]["generation_failures_at_5"] == 2
    assert report["diagnosis"]["selection_failures_at_5"] == 0


def test_all_runner_is_fail_closed_when_source_shards_are_missing(tmp_path) -> None:
    report = run_all(
        v43_dir=tmp_path / "missing_v43",
        output_dir=tmp_path / "output",
    )

    assert report["status"] == "FAIL_CLOSED"
    assert report["source_validation"]["authorized"] is False
    assert report["holdout_opened"] is False
    persisted = json.loads(
        (tmp_path / "output" / "gate_report.json").read_text(encoding="utf-8")
    )
    assert persisted["active_authority_authorized"] is False
