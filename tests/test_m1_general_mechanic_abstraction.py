import numpy as np

from theory.m1.general_mechanic_abstraction import (
    TRUTH_STATUS,
    build_relation_graphs_by_frame,
    build_general_mechanic_candidate_ledger,
    detect_dynamic_invariants,
    emit_dynamic_invariant_mechanic_hypotheses,
    emit_relation_change_mechanic_hypotheses,
    generate_role_hypotheses,
    infer_action_effect_abstractions,
    infer_relation_deltas,
    track_entities,
)


def _moving_actor_grids():
    grids = []
    for x in [1, 2, 3, 4]:
        grid = np.zeros((6, 8), dtype=np.int32)
        grid[3, x] = 7
        grid[1, 6] = 4
        grids.append(grid)
    return grids


def _elapsed_hud_grids():
    grids = []
    for filled in [1, 2, 3, 4]:
        grid = np.zeros((6, 8), dtype=np.int32)
        grid[-1, :filled] = 9
        grid[2, 4] = 5
        grids.append(grid)
    return grids


def _moving_actor_with_elapsed_hud_grids():
    grids = []
    for index, x in enumerate([1, 2, 3, 4], start=1):
        grid = np.zeros((6, 8), dtype=np.int32)
        grid[3, x] = 7
        grid[-1, :index] = 9
        grids.append(grid)
    return grids


def _create_delete_grids():
    grid0 = np.zeros((5, 5), dtype=np.int32)
    grid1 = grid0.copy()
    grid1[2, 2] = 4
    grid2 = grid0.copy()
    return [grid0, grid1, grid2]


def _contact_created_grids():
    grid0 = np.zeros((5, 6), dtype=np.int32)
    grid0[2, 1] = 7
    grid0[2, 4] = 4
    grid1 = np.zeros((5, 6), dtype=np.int32)
    grid1[2, 3] = 7
    grid1[2, 4] = 4
    return [grid0, grid1]


def _downward_drift_grids():
    grids = []
    for y in [1, 2, 3, 4]:
        grid = np.zeros((7, 7), dtype=np.int32)
        grid[y, 3] = 6
        grid[0, : y + 1] = 9
        grids.append(grid)
    return grids


def test_track_entities_preserves_candidate_only_entity_summaries():
    tracks = track_entities(
        _moving_actor_grids(),
        ["ACTION3", "ACTION3", "ACTION3"],
        background=0,
    )

    moving = max(tracks, key=lambda track: track.summary()["motion_score"])
    summary = moving.summary()

    assert summary["frames_seen"] == 4
    assert summary["motion_score"] > 0
    assert summary["support"] == 0
    assert summary["truth_status"] == TRUTH_STATUS
    assert summary["observation_counted_as_confirmation"] is False


def test_role_generator_emits_controllable_actor_candidate_without_confirmation():
    tracks = track_entities(
        _moving_actor_grids(),
        ["ACTION3", "ACTION3", "ACTION3"],
        background=0,
    )
    moving = max(tracks, key=lambda track: track.summary()["motion_score"])

    roles = generate_role_hypotheses(moving)
    role_names = {role.role for role in roles}
    actor = next(role for role in roles if role.role == "controllable_actor")

    assert "controllable_actor" in role_names
    assert actor.score > 0
    assert actor.support == 0
    assert actor.truth_status == TRUTH_STATUS


def test_role_generator_emits_timer_or_hud_for_monotone_edge_entity():
    tracks = track_entities(
        _elapsed_hud_grids(),
        ["ACTION6", "ACTION6", "ACTION6"],
        background=0,
    )

    hud_roles = [
        role
        for track in tracks
        for role in generate_role_hypotheses(track)
        if role.role == "timer_or_hud"
    ]

    assert hud_roles
    assert hud_roles[0].support == 0
    assert "monotone_size_sequence" in hud_roles[0].evidence


def test_action_effect_abstraction_detects_move_and_hud_tick():
    grids = _moving_actor_with_elapsed_hud_grids()
    actions = ["ACTION3", "ACTION3", "ACTION3"]
    tracks = track_entities(grids, actions, background=0)
    roles = [role for track in tracks for role in generate_role_hypotheses(track)]

    effects = infer_action_effect_abstractions(
        grid_history=grids,
        action_history=actions,
        tracks=tracks,
        role_hypotheses=roles,
    )
    action3 = next(row for row in effects if row.action == "ACTION3")

    assert "move_entity" in action3.effect_families
    assert "tick_latent" in action3.effect_families
    assert action3.affected_entities
    assert action3.support == 0
    assert action3.truth_status == TRUTH_STATUS


def test_action_effect_abstraction_detects_create_delete_and_no_effect():
    grids = _create_delete_grids() + [_create_delete_grids()[-1]]
    actions = ["ACTION1", "ACTION2", "ACTION5"]
    tracks = track_entities(grids, actions, background=0)
    roles = [role for track in tracks for role in generate_role_hypotheses(track)]

    effects = infer_action_effect_abstractions(
        grid_history=grids,
        action_history=actions,
        tracks=tracks,
        role_hypotheses=roles,
    )
    by_action = {row.action: row for row in effects}

    assert "create_entity" in by_action["ACTION1"].effect_families
    assert "delete_entity" in by_action["ACTION2"].effect_families
    assert "no_effect" in by_action["ACTION5"].effect_families
    assert all(row.support == 0 for row in effects)


def test_relation_graph_detects_distance_and_contact_delta():
    grids = _contact_created_grids()
    actions = ["ACTION3"]
    tracks = track_entities(grids, actions, background=0)
    roles = [role for track in tracks for role in generate_role_hypotheses(track)]

    graphs = build_relation_graphs_by_frame(tracks, roles)
    deltas = infer_relation_deltas(graphs, actions)

    delta_types = {delta.relation_delta_type for delta in deltas}
    assert "distance_decreases" in delta_types
    assert "contact_created" in delta_types
    assert all(delta.support == 0 for delta in deltas)
    assert all(delta.truth_status == TRUTH_STATUS for delta in deltas)


def test_relation_change_hypotheses_are_candidate_only():
    grids = _contact_created_grids()
    actions = ["ACTION3"]
    tracks = track_entities(grids, actions, background=0)
    roles = [role for track in tracks for role in generate_role_hypotheses(track)]
    graphs = build_relation_graphs_by_frame(tracks, roles)
    deltas = infer_relation_deltas(graphs, actions)

    hypotheses = emit_relation_change_mechanic_hypotheses(deltas)

    assert hypotheses
    assert any(h.mechanic_family == "relation_change" for h in hypotheses)
    assert all(h.support == 0 for h in hypotheses)
    assert all(h.status == "UNRESOLVED" for h in hypotheses)
    assert all(h.truth_status == TRUTH_STATUS for h in hypotheses)


def test_dynamic_invariant_detector_finds_monotone_counter_and_drift():
    grids = _downward_drift_grids()
    actions = ["ACTION3", "ACTION4", "ACTION6"]
    tracks = track_entities(grids, actions, background=0)
    roles = [role for track in tracks for role in generate_role_hypotheses(track)]

    invariants = detect_dynamic_invariants(
        grid_history=grids,
        action_history=actions,
        tracks=tracks,
        role_hypotheses=roles,
    )
    families = {row.invariant_family for row in invariants}

    assert "monotone_counter" in families
    assert "exogenous_motion" in families
    assert all(row.support == 0 for row in invariants)
    assert all(row.truth_status == TRUTH_STATUS for row in invariants)


def test_dynamic_invariant_hypotheses_are_candidate_only():
    grids = _downward_drift_grids()
    actions = ["ACTION3", "ACTION4", "ACTION6"]
    tracks = track_entities(grids, actions, background=0)
    roles = [role for track in tracks for role in generate_role_hypotheses(track)]
    invariants = detect_dynamic_invariants(
        grid_history=grids,
        action_history=actions,
        tracks=tracks,
        role_hypotheses=roles,
    )

    hypotheses = emit_dynamic_invariant_mechanic_hypotheses(invariants)

    assert hypotheses
    assert all(row.mechanic_family == "dynamic_invariant" for row in hypotheses)
    assert all(row.support == 0 for row in hypotheses)
    assert all(row.status == "UNRESOLVED" for row in hypotheses)
    assert all(row.truth_status == TRUTH_STATUS for row in hypotheses)


def test_general_mechanic_ledger_has_entity_role_hypotheses_and_guards():
    grids = _moving_actor_with_elapsed_hud_grids()
    actions = ["ACTION3"] * (len(grids) - 1)

    payload = build_general_mechanic_candidate_ledger(
        grid_history=grids,
        action_history=actions,
        game_id="zz00-test",
        source_label="synthetic_grid_history",
        policy_label="unit_test_policy",
    )

    summary = payload["summary"]
    hypotheses = payload["mechanic_hypotheses"]

    assert summary["mechanic_hypotheses_generated"] == len(hypotheses)
    assert summary["action_effect_rows"] >= 1
    assert summary["action_effect_hypotheses_generated"] >= 1
    assert summary["relation_graph_frames"] >= 1
    assert summary["relation_delta_rows"] >= 1
    assert summary["relation_change_hypotheses_generated"] >= 1
    assert summary["dynamic_invariant_candidates"] >= 1
    assert summary["dynamic_invariant_hypotheses_generated"] >= 1
    assert summary["support"] == 0
    assert summary["truth_status"] == TRUTH_STATUS
    assert summary["ready_for_m3_g0"] is True
    assert {row["mechanic_family"] for row in hypotheses} >= {
        "entity_role",
        "action_effect",
        "relation_change",
        "dynamic_invariant",
    }
    assert all(row["status"] == "UNRESOLVED" for row in hypotheses)
    assert all(row["support"] == 0 for row in hypotheses)
    assert all(row["truth_status"] == TRUTH_STATUS for row in hypotheses)
