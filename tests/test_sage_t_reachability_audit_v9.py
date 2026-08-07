from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import reachability_audit_v9 as v9


def test_winner_paths_are_the_three_frozen_rrr_trajectories() -> None:
    pairs = load_pairs(v9.DEFAULT_SHARD_DIR, v9.SOURCE_GAMES)
    paths = v9.winner_paths(pairs)

    assert len(pairs) == 380
    assert len(paths) == 3
    assert set(paths.values()) == {("RRR",)}
    assert all(root.startswith("lp85:") for root in paths)


def test_failure_taxonomy_is_exclusive_and_precedenced() -> None:
    common = {
        "ground_truth_covered": True,
        "goal_sequence_generated": True,
        "unpruned_static": 1,
        "unpruned_executable": 1,
        "production_executable": 1,
        "goal_program_rank": 1,
        "selected_goal_action": True,
        "terminal_blocked": False,
        "execution_errors": 0,
    }

    assert v9.classify_failure(**common)[0] == "REACHABLE"
    assert v9.classify_failure(
        **{**common, "goal_sequence_generated": False}
    )[0] == "NO_GOAL_CANDIDATE"
    assert v9.classify_failure(
        **{**common, "unpruned_executable": 0}
    )[0] == "EXECUTION_MODEL_MISS"
    assert v9.classify_failure(
        **{**common, "production_executable": 0}
    )[0] == "GOAL_CANDIDATE_PRUNED"
    assert v9.classify_failure(
        **{**common, "goal_program_rank": None}
    )[0] == "GOAL_CANDIDATE_UNDERVALUED"
    assert v9.classify_failure(
        **{**common, "selected_goal_action": False, "terminal_blocked": True}
    )[0] == "GOAL_CANDIDATE_VETOED"


def test_manifest_is_source_train_shadow_only() -> None:
    manifest = v9.load_manifest()

    assert manifest["source_train_games"] == ["lp85", "su15"]
    assert manifest["controller_caps"] == v9.CURRENT_CAPS
    assert manifest["firewall"]["authority"] == "shadow"
    assert manifest["firewall"]["source_validation_games_executed"] is False
    assert manifest["firewall"]["holdout_opened"] is False
