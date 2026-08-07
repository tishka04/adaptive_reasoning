from __future__ import annotations

import json
from pathlib import Path

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t.calibration_gate_v8_6 import (
    DEFAULT_SHARD_DIR,
    EXPECTED_CORPUS,
    _all_root_episodes,
    _evaluate_episode,
    _load_episode_checkpoint,
    _save_episode_checkpoint,
    _signal_sequences,
    corpus_inventory,
    freeze_selection_manifest,
    load_selection_manifest,
    select_challenger,
)
from theory.sage_t.posterior_v2 import T8_6_POLICIES
from theory.sage_t.replay_gate import load_frozen_manifest as load_t7_manifest


def _pairs():  # type: ignore[no-untyped-def]
    return load_pairs(str(DEFAULT_SHARD_DIR), ("lp85", "su15"))


def test_v43_corpus_and_signal_roots_match_the_frozen_contract() -> None:
    pairs = _pairs()

    assert corpus_inventory(pairs) == EXPECTED_CORPUS
    shocks = _signal_sequences(pairs)
    assert len(shocks) == 25
    assert sum(item["positive_kind"] == "goal" for item in shocks) == 3


def test_selection_manifest_freezes_code_data_and_firewall(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    frozen = freeze_selection_manifest(output_path=path)

    loaded = load_selection_manifest(path)

    assert loaded == frozen
    assert loaded["firewall"]["authority"] == "shadow"
    assert loaded["firewall"]["source_validation_opened"] is False
    assert set(loaded["source_train_games"]) == {"lp85", "su15"}
    assert "ar25" in loaded["forbidden_games"]


def test_all_policies_receive_the_exact_same_revealed_actions() -> None:
    episode = _all_root_episodes(_pairs())[0]
    t7 = load_t7_manifest(verify_code=True)
    _, legacy_updates, keys, _ = _evaluate_episode(
        episode,
        policy=T8_6_POLICIES["legacy"],
        manifest=t7,
    )
    _, challenger_updates, challenger_keys, _ = _evaluate_episode(
        episode,
        policy=T8_6_POLICIES["tempered"],
        manifest=t7,
        forced_keys=keys,
    )

    assert challenger_keys == keys
    assert [row["action_key"] for row in challenger_updates] == [
        row["action_key"] for row in legacy_updates
    ]


def test_pre_registered_selection_rule_chooses_only_a_full_gate_survivor() -> None:
    manifest = json.loads(
        Path(
            "theory/sage_t/sage_t8_6_selection_manifest.json"
        ).read_text(encoding="utf-8")
    )
    rows = []
    for game in ("lp85", "su15"):
        for index in range(4):
            root = f"{game}:{index}"
            for condition, brier, loss, likelihood in (
                ("legacy", 0.4, 0.8, -2.0),
                ("tempered", 0.1, 0.2, -1.0),
                ("correlation_aware", 0.5, 0.9, -2.1),
                ("combined", 0.5, 0.9, -2.1),
            ):
                rows.append(
                    {
                        "episode_id": root,
                        "game": game,
                        "condition": condition,
                        "checkpoint": 5,
                        "terminal_brier": brier,
                        "terminal_log_loss": loss,
                        "hidden_log_likelihood": likelihood,
                        "decision_latency_ms": 1.0,
                    }
                )
    updates = []
    for condition in T8_6_POLICIES:
        for index in range(10):
            updates.append(
                {
                    "condition": condition,
                    "semantic_collapse": condition == "legacy" and index < 4,
                }
            )

    winner, diagnostics = select_challenger(rows, updates, manifest=manifest)

    assert winner == "tempered"
    assert diagnostics["tempered"]["passed"]
    assert not diagnostics["combined"]["passed"]


def test_atomic_episode_checkpoint_round_trips_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    _save_episode_checkpoint(
        path,
        manifest_checksum="manifest",
        episode_id="root",
        condition="combined",
        rows=({"checkpoint": 1},),
        updates=({"observation": 1},),
        keys=("ACTION1:{}",),
        arm_rows=({"arm": "hidden"},),
    )

    restored = _load_episode_checkpoint(
        path,
        manifest_checksum="manifest",
        episode_id="root",
        condition="combined",
    )
    assert restored == (
        [{"checkpoint": 1}],
        [{"observation": 1}],
        ["ACTION1:{}"],
        [{"arm": "hidden"}],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["keys"] = ["ACTION2:{}"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_episode_checkpoint(
        path,
        manifest_checksum="manifest",
        episode_id="root",
        condition="combined",
    ) is None
