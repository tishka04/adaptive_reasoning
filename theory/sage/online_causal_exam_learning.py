"""SAGE.8j bounded online causal learning during a fresh ARC examination."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .live_mini_frontier_m3_executor import EnvFactory, _make_real_env
from .live_prefix_counterfactual_collector import select_live_action
from .relational_memory_closed_loop_evaluation import action_identity


DEFAULT_SAGE8J_ONLINE_CAUSAL_EXAM_LEARNING_PATH = (
    Path("diagnostics") / "sage" / "sage8j_online_causal_exam_learning.json"
)

SAGE8J_SCHEMA_VERSION = "sage.online_causal_exam_learning.v1"
SAGE8J_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8J"
SAGE8J_ARC_GAIN = "SAGE_ONLINE_CAUSAL_EXAM_LEARNING_ARC_GAIN_OBSERVED"
SAGE8J_SUCCESS_NO_GAIN = (
    "SAGE_ONLINE_CAUSAL_EXAM_LEARNING_SUCCESS_WITHOUT_ABLATION_GAIN"
)
SAGE8J_ACTIVE_NO_GAIN = "SAGE_ONLINE_CAUSAL_EXAM_LEARNING_ACTIVE_NO_ARC_GAIN"

DEFAULT_FRESH_TARGET_GAMES = (
    "lf52-271a04aa",
    "lp85-305b61c3",
)
DEFAULT_MAX_ACTION_EXECUTIONS = 512
DEFAULT_MAX_TRIALS = 256
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_DISCOVERED_STATES = 128
TERMINAL_WIN_STATES = {"WIN", "WON", "VICTORY"}
NON_TERMINAL_STATES = {"", "NOT_FINISHED", "PLAYING", "IN_PROGRESS"}


def run_sage8j_online_causal_exam_learning(
    *,
    target_games: Sequence[str] = DEFAULT_FRESH_TARGET_GAMES,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    max_action_executions: int = DEFAULT_MAX_ACTION_EXECUTIONS,
    max_trials: int = DEFAULT_MAX_TRIALS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_discovered_states: int = DEFAULT_MAX_DISCOVERED_STATES,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run fixed-schedule and online-learning arms on fresh target games."""
    games = tuple(str(game_id) for game_id in target_games)
    if not games or len(games) != len(set(games)):
        raise ValueError("SAGE.8j requires unique non-empty fresh target games")
    if any(game_id in {"tn36-ab4f63cc", "wa30-ee6fef47"} for game_id in games):
        raise ValueError("SAGE.8j fresh targets must exclude SAGE.8i games")
    if min(max_action_executions, max_trials, max_depth, max_discovered_states) <= 0:
        raise ValueError("SAGE.8j bounds must all be positive")
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    episodes = []
    for game_id in games:
        control = execute_static_exam_arm(
            game_id,
            environments_dir=env_dir,
            max_action_executions=max_action_executions,
            max_trials=max_trials,
            max_depth=max_depth,
            env_factory=env_factory,
        )
        adaptive = execute_online_causal_exam_arm(
            game_id,
            environments_dir=env_dir,
            max_action_executions=max_action_executions,
            max_trials=max_trials,
            max_depth=max_depth,
            max_discovered_states=max_discovered_states,
            env_factory=env_factory,
        )
        episodes.append(compare_sage8j_exam_arms(game_id, control, adaptive))

    metrics = summarize_sage8j_metrics(episodes)
    gate = build_sage8j_gate(
        games,
        episodes,
        metrics,
        max_action_executions=max_action_executions,
        max_trials=max_trials,
        max_depth=max_depth,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8j online causal examination gate did not pass")
    summary = summarize_sage8j(episodes, metrics, gate)
    payload = {
        "config": {
            "schema_version": SAGE8J_SCHEMA_VERSION,
            "target_games": list(games),
            "environments_dir": str(env_dir),
            "bounds_per_game_per_arm": {
                "max_action_executions": int(max_action_executions),
                "max_trials": int(max_trials),
                "max_sequence_depth": int(max_depth),
                "max_discovered_states": int(max_discovered_states),
            },
            "exam_design": {
                "learning_algorithm_frozen_before_exam": True,
                "action_schedule_frozen_before_exam": False,
                "beliefs_update_after_each_observed_candidate_effect": True,
                "learned_state_graph_updates_during_exam": True,
                "observed_outcomes_become_legal_next_step_evidence": True,
                "unexecuted_action_outcomes_available": False,
                "independent_trial_resets_allowed_and_counted": True,
                "all_replayed_prefix_actions_counted": True,
                "same_action_and_trial_bounds_between_arms": True,
            },
            "control_arm": {
                "schedule": "PREDECLARED_LEXICOGRAPHIC_SEQUENCE_ENUMERATION",
                "outcomes_used_to_reorder_or_prune": False,
                "stops_after_observed_positive": True,
            },
            "adaptive_arm": {
                "schedule": "OBSERVED_EFFECT_CONDITIONED_CAUSAL_STATE_GRAPH_DFS",
                "new_state_nodes_created_from_observed_effects": True,
                "duplicate_and_no_effect_states_pruned_online": True,
                "action_family_hypotheses_revised_online": True,
                "successful_sequence_retained_after_observation": True,
            },
            "structural_representation": {
                "connected_components_extracted_online": True,
                "component_shape_area_color_and_position_encoded": True,
                "pairwise_spatial_relations_encoded": True,
                "volatile_bottom_status_row_excluded_when_present": True,
                "exact_visual_digest_retained_for_replay_audit": True,
            },
            "freshness_contract": {
                "sage8i_games_excluded": True,
                "sage8i_action_traces_loaded": False,
                "prior_target_action_traces_loaded": [],
                "target_games_selected_from_metadata_only": True,
                "game_source_files_inspected_by_agent": [],
                "protocol_frozen_before_first_target_action": True,
            },
        },
        "paired_exam_episodes": episodes,
        "online_learning_metrics": metrics,
        "gate": gate,
        "summary": summary,
        "status": "EVALUATED",
        "outcome_status": summary["outcome_status"],
        "truth_status": SAGE8J_TRUTH_STATUS,
        "online_learning_during_exam_performed": True,
        "learning_algorithm_frozen_before_exam": True,
        "agent_policy_state_frozen_during_exam": False,
        "sage8i_action_traces_loaded": False,
        "evaluation_outcomes_used_after_exam_for_algorithm_tuning": False,
        "future_outcomes_used_for_action_selection": False,
        "game_source_files_inspected_by_agent": [],
        "scientific_review_performed": False,
        "confirmation_performed": False,
        "revision_performed": False,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "support": 0,
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_sage8j_online_causal_exam_learning(payload, output_path)
    return payload


def execute_static_exam_arm(
    game_id: str,
    *,
    environments_dir: str | Path,
    max_action_executions: int,
    max_trials: int,
    max_depth: int,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    """Execute a schedule declared from RESET actions before outcomes are seen."""
    env = _new_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    reset_snapshot = snapshot_frame(frame)
    root_commands = legal_commands(env)
    experiments = []
    action_executions = 0
    trials = 0
    positive_trial: Dict[str, Any] | None = None
    for sequence in static_sequence_schedule(root_commands, max_depth=max_depth):
        if (
            trials >= max_trials
            or action_executions + len(sequence) > max_action_executions
        ):
            break
        trial = execute_exam_trial(env, sequence)
        trials += 1
        action_executions += int(trial.get("actions_executed", 0) or 0)
        experiments.append(public_trial_record(trial, trial_index=trials))
        if bool(trial.get("positive_observed", False)):
            positive_trial = trial
            break
    return {
        "status": "EXECUTED",
        "arm": "static_predeclared_schedule",
        "game_id": game_id,
        "reset_visual_digest": visual_digest(reset_snapshot.grid),
        "reset_structural_key": extract_structural_state(reset_snapshot.grid)[
            "structural_key"
        ],
        "root_legal_action_count": len(root_commands),
        "schedule_predeclared_before_outcomes": True,
        "outcomes_used_to_reorder_or_prune": False,
        "trials_executed": trials,
        "resets_executed": trials + 1,
        "action_executions": action_executions,
        "max_action_executions": int(max_action_executions),
        "max_trials": int(max_trials),
        "max_depth": int(max_depth),
        "positive_observed": positive_trial is not None,
        "levels_completed_delta_best": int(
            positive_trial.get("levels_completed_delta", 0) if positive_trial else 0
        ),
        "win_observed": bool(positive_trial and positive_trial.get("win", False)),
        "successful_sequence": list(
            positive_trial.get("sequence", []) if positive_trial else []
        ),
        "answer_learned_during_exam": False,
        "belief_updates": 0,
        "hypothesis_revisions": 0,
        "discovered_structural_states": 0,
        "duplicate_states_pruned": 0,
        "no_effect_states_pruned": 0,
        "experiments": experiments,
        "future_outcomes_used_for_action_selection": False,
        "evaluation_outcomes_used_after_exam_for_algorithm_tuning": False,
        "all_replayed_actions_counted": True,
        "all_selected_actions_legal": all(
            bool(row.get("all_selected_actions_legal", False)) for row in experiments
        ),
    }


def execute_online_causal_exam_arm(
    game_id: str,
    *,
    environments_dir: str | Path,
    max_action_executions: int,
    max_trials: int,
    max_depth: int,
    max_discovered_states: int,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    """Learn a causal state graph from the outcomes observed during the exam."""
    env = _new_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    reset_snapshot = snapshot_frame(frame)
    reset_structural = extract_structural_state(reset_snapshot.grid)
    root_commands = legal_commands(env)
    root = make_state_node(
        sequence=[],
        structural_key=str(reset_structural["structural_key"]),
        exact_digest=visual_digest(reset_snapshot.grid),
        legal_action_commands=root_commands,
    )
    stack = [root]
    known_states = {str(root["structural_key"])}
    causal_beliefs: Dict[str, Dict[str, Any]] = {}
    experiments = []
    action_executions = 0
    trials = 0
    belief_updates = 0
    hypothesis_revisions = 0
    duplicate_states_pruned = 0
    no_effect_states_pruned = 0
    nondeterministic_prefixes = 0
    positive_trial: Dict[str, Any] | None = None

    while stack and trials < max_trials and action_executions < max_action_executions:
        node = stack[-1]
        if not node["untried_actions"]:
            stack.pop()
            continue
        command, selection = select_online_experiment_action(
            node["untried_actions"], causal_beliefs
        )
        node["untried_actions"].remove(command)
        sequence = [*node["sequence"], command]
        if len(sequence) > max_depth:
            continue
        if action_executions + len(sequence) > max_action_executions:
            break
        trial = execute_exam_trial(env, sequence)
        trials += 1
        action_executions += int(trial.get("actions_executed", 0) or 0)
        last = dict(trial.get("last_transition", {}) or {})
        before_key = str(last.get("before_structural_key", ""))
        if before_key and before_key != str(node["structural_key"]):
            nondeterministic_prefixes += 1
        family = str(command.get("action", ""))
        revision = update_online_causal_belief(
            causal_beliefs,
            action_family=family,
            transition=last,
        )
        belief_updates += 1
        hypothesis_revisions += int(revision["hypothesis_revised"])
        public = public_trial_record(trial, trial_index=trials)
        public["online_selection"] = selection
        public["belief_revision"] = revision
        experiments.append(public)

        if bool(trial.get("positive_observed", False)):
            positive_trial = trial
            break
        after_key = str(last.get("after_structural_key", ""))
        state_changed = bool(last.get("structural_state_changed", False))
        if not state_changed:
            no_effect_states_pruned += 1
            continue
        if after_key in known_states:
            duplicate_states_pruned += 1
            continue
        if len(known_states) >= max_discovered_states or len(sequence) >= max_depth:
            continue
        known_states.add(after_key)
        stack.append(
            make_state_node(
                sequence=sequence,
                structural_key=after_key,
                exact_digest=str(last.get("after_visual_digest", "")),
                legal_action_commands=list(trial.get("final_legal_commands", []) or []),
            )
        )

    replay = (
        verify_learned_answer(
            game_id,
            list(positive_trial.get("sequence", []) or []),
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        if positive_trial is not None
        else empty_learned_answer_replay()
    )
    return {
        "status": "EXECUTED",
        "arm": "online_causal_state_graph_learning",
        "game_id": game_id,
        "reset_visual_digest": visual_digest(reset_snapshot.grid),
        "reset_structural_key": str(reset_structural["structural_key"]),
        "root_legal_action_count": len(root_commands),
        "learning_algorithm_frozen_before_exam": True,
        "policy_state_frozen_during_exam": False,
        "outcomes_used_to_reorder_or_prune": True,
        "trials_executed": trials,
        "resets_executed": trials + 1,
        "action_executions": action_executions,
        "max_action_executions": int(max_action_executions),
        "max_trials": int(max_trials),
        "max_depth": int(max_depth),
        "max_discovered_states": int(max_discovered_states),
        "positive_observed": positive_trial is not None,
        "levels_completed_delta_best": int(
            positive_trial.get("levels_completed_delta", 0) if positive_trial else 0
        ),
        "win_observed": bool(positive_trial and positive_trial.get("win", False)),
        "successful_sequence": list(
            positive_trial.get("sequence", []) if positive_trial else []
        ),
        "answer_learned_during_exam": positive_trial is not None,
        "learned_answer_exact_replay": replay,
        "belief_updates": belief_updates,
        "hypothesis_revisions": hypothesis_revisions,
        "causal_action_family_beliefs": dict(sorted(causal_beliefs.items())),
        "discovered_structural_states": len(known_states),
        "duplicate_states_pruned": duplicate_states_pruned,
        "no_effect_states_pruned": no_effect_states_pruned,
        "nondeterministic_prefixes": nondeterministic_prefixes,
        "experiments": experiments,
        "future_outcomes_used_for_action_selection": False,
        "evaluation_outcomes_used_after_exam_for_algorithm_tuning": False,
        "all_replayed_actions_counted": True,
        "all_selected_actions_legal": all(
            bool(row.get("all_selected_actions_legal", False)) for row in experiments
        ),
    }


def execute_exam_trial(
    env: Any, sequence: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Reset, execute one hypothesized sequence, and observe each consequence."""
    try:
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {
            "status": "BLOCKED",
            "reason": f"reset_failed:{exc}",
            "sequence": [],
            "actions_executed": 0,
            "positive_observed": False,
            "all_selected_actions_legal": False,
        }
    reset = snapshot_frame(frame)
    initial_levels = int(reset.levels_completed)
    transitions = []
    executed = []
    all_legal = True
    for step, raw in enumerate(sequence):
        action = str(raw.get("action", ""))
        args = dict(raw.get("action_args", {}) or {})
        selected = select_live_action(env, action, action_args=args)
        if selected is None:
            all_legal = False
            break
        before = snapshot_frame(frame)
        frame = _step_env_action(env, selected)
        after = snapshot_frame(
            frame, fallback_available_actions=before.available_actions
        )
        transition = observe_causal_transition(before, after, action=action, args=args)
        transition["trial_step"] = step
        transitions.append(transition)
        executed.append({"action": action, "action_args": args})
        if bool(transition["positive_observed"]):
            break
    final = snapshot_frame(frame, fallback_available_actions=reset.available_actions)
    positive = any(bool(row.get("positive_observed", False)) for row in transitions)
    return {
        "status": "POSITIVE_OBSERVED" if positive else "EXECUTED",
        "sequence": executed,
        "sequence_exactly_executed": len(executed) == len(sequence),
        "actions_executed": len(executed),
        "all_selected_actions_legal": all_legal,
        "reset_visual_digest": visual_digest(reset.grid),
        "initial_levels_completed": initial_levels,
        "final_levels_completed": int(final.levels_completed),
        "levels_completed_delta": int(final.levels_completed) - initial_levels,
        "game_state_after": str(final.game_state),
        "win": str(final.game_state).upper() in TERMINAL_WIN_STATES,
        "positive_observed": positive,
        "transitions": transitions,
        "last_transition": transitions[-1] if transitions else {},
        "final_legal_commands": legal_commands(env),
    }


def observe_causal_transition(
    before: Any,
    after: Any,
    *,
    action: str,
    args: Mapping[str, Any],
) -> Dict[str, Any]:
    before_grid = np.asarray(before.grid, dtype=np.int32)
    after_grid = np.asarray(after.grid, dtype=np.int32)
    before_structure = extract_structural_state(before_grid)
    after_structure = extract_structural_state(after_grid)
    comparable = before_grid.shape == after_grid.shape
    changed = int(np.count_nonzero(before_grid != after_grid)) if comparable else 0
    stable_before = stable_grid(before_grid)
    stable_after = stable_grid(after_grid)
    stable_changed = (
        int(np.count_nonzero(stable_before != stable_after))
        if stable_before.shape == stable_after.shape
        else 0
    )
    level_delta = int(after.levels_completed) - int(before.levels_completed)
    terminal_win = str(after.game_state).upper() in TERMINAL_WIN_STATES
    effect_payload = {
        "action_family": action,
        "stable_changed_pixels": stable_changed,
        "object_count_delta": int(after_structure["object_count"])
        - int(before_structure["object_count"]),
        "relation_signature_changed": (
            before_structure["relation_signature"]
            != after_structure["relation_signature"]
        ),
        "level_delta": level_delta,
        "terminal_win": terminal_win,
    }
    return {
        "action": action,
        "action_args": dict(args),
        "action_identity": action_identity(action, args),
        "before_visual_digest": visual_digest(before_grid),
        "after_visual_digest": visual_digest(after_grid),
        "before_structural_key": str(before_structure["structural_key"]),
        "after_structural_key": str(after_structure["structural_key"]),
        "visual_changed_pixels": changed,
        "stable_changed_pixels": stable_changed,
        "structural_state_changed": (
            before_structure["structural_key"] != after_structure["structural_key"]
        ),
        "object_count_before": int(before_structure["object_count"]),
        "object_count_after": int(after_structure["object_count"]),
        "relation_signature_changed": effect_payload["relation_signature_changed"],
        "effect_signature": digest_json(effect_payload),
        "levels_completed_before": int(before.levels_completed),
        "levels_completed_after": int(after.levels_completed),
        "level_delta": level_delta,
        "game_state_after": str(after.game_state),
        "terminal_win": terminal_win,
        "positive_observed": bool(level_delta > 0 or terminal_win),
        "outcome_observed_only_after_action": True,
    }


def extract_structural_state(grid: Any) -> Dict[str, Any]:
    """Extract object descriptors and pairwise relations from the current frame."""
    array = stable_grid(np.asarray(grid, dtype=np.int32))
    if array.size == 0:
        return {
            "structural_key": digest_json({"shape": list(array.shape)}),
            "object_count": 0,
            "component_signature": digest_json([]),
            "relation_signature": digest_json([]),
        }
    colors, counts = np.unique(array, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    components = connected_components(array, excluded_color=background)
    objects = []
    for component in components:
        min_y, min_x, max_y, max_x = component["bbox"]
        objects.append(
            {
                "color": int(component["color"]),
                "area": int(component["area"]),
                "height": int(max_y - min_y + 1),
                "width": int(max_x - min_x + 1),
                "center_y_bucket": int(round(component["center_y"] / 4.0)),
                "center_x_bucket": int(round(component["center_x"] / 4.0)),
            }
        )
    objects.sort(
        key=lambda row: (
            row["color"],
            row["area"],
            row["height"],
            row["width"],
            row["center_y_bucket"],
            row["center_x_bucket"],
        )
    )
    relations = []
    for left_index, left in enumerate(objects):
        for right in objects[left_index + 1 :]:
            relations.append(
                {
                    "color_pair": [left["color"], right["color"]],
                    "vertical_relation": compare_bucket(
                        left["center_y_bucket"], right["center_y_bucket"]
                    ),
                    "horizontal_relation": compare_bucket(
                        left["center_x_bucket"], right["center_x_bucket"]
                    ),
                    "distance_bucket": min(
                        15,
                        abs(left["center_y_bucket"] - right["center_y_bucket"])
                        + abs(left["center_x_bucket"] - right["center_x_bucket"]),
                    ),
                }
            )
    component_signature = digest_json(objects)
    relation_signature = digest_json(relations)
    structural_key = digest_json(
        {
            "shape": list(array.shape),
            "background": background,
            "component_signature": component_signature,
            "relation_signature": relation_signature,
        }
    )
    return {
        "structural_key": structural_key,
        "object_count": len(objects),
        "component_signature": component_signature,
        "relation_signature": relation_signature,
    }


def connected_components(
    array: np.ndarray,
    *,
    excluded_color: int,
) -> list[Dict[str, Any]]:
    visited: set[tuple[int, int]] = set()
    components = []
    height, width = array.shape
    for y in range(height):
        for x in range(width):
            color = int(array[y, x])
            if color == excluded_color or (y, x) in visited:
                continue
            stack = [(y, x)]
            visited.add((y, x))
            points = []
            while stack:
                point_y, point_x = stack.pop()
                points.append((point_y, point_x))
                for delta_y, delta_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_y = point_y + delta_y
                    next_x = point_x + delta_x
                    next_point = (next_y, next_x)
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and next_point not in visited
                        and int(array[next_y, next_x]) == color
                    ):
                        visited.add(next_point)
                        stack.append(next_point)
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            components.append(
                {
                    "color": color,
                    "area": len(points),
                    "bbox": [min(ys), min(xs), max(ys), max(xs)],
                    "center_y": sum(ys) / len(ys),
                    "center_x": sum(xs) / len(xs),
                }
            )
    return components


def update_online_causal_belief(
    beliefs: MutableMapping[str, Dict[str, Any]],
    *,
    action_family: str,
    transition: Mapping[str, Any],
) -> Dict[str, Any]:
    previous = dict(beliefs.get(action_family, {}) or {})
    previous_status = str(previous.get("hypothesis_status", "UNKNOWN"))
    trials = int(previous.get("trials", 0) or 0) + 1
    effectful = int(previous.get("effectful_observations", 0) or 0) + int(
        bool(transition.get("structural_state_changed", False))
    )
    positives = int(previous.get("positive_observations", 0) or 0) + int(
        bool(transition.get("positive_observed", False))
    )
    signatures = set(previous.get("effect_signatures", []) or [])
    signature = str(transition.get("effect_signature", ""))
    if signature:
        signatures.add(signature)
    if positives:
        status = "OBSERVED_GOAL_CAUSAL"
    elif effectful == 0:
        status = "NO_STRUCTURAL_EFFECT_OBSERVED"
    elif effectful == trials:
        status = "OBSERVED_STRUCTURAL_EFFECT"
    else:
        status = "CONTEXT_DEPENDENT_EFFECT"
    updated = {
        "action_family": action_family,
        "trials": trials,
        "effectful_observations": effectful,
        "no_effect_observations": trials - effectful,
        "positive_observations": positives,
        "effect_signatures": sorted(signatures),
        "hypothesis_status": status,
        "effect_rate": effectful / trials,
        "updated_from_observed_outcomes_only": True,
    }
    beliefs[action_family] = updated
    return {
        "action_family": action_family,
        "previous_hypothesis_status": previous_status,
        "updated_hypothesis_status": status,
        "hypothesis_revised": status != previous_status,
        "belief_update_index": trials,
        "outcome_was_observed_before_update": True,
    }


def select_online_experiment_action(
    untried_actions: Sequence[Mapping[str, Any]],
    beliefs: Mapping[str, Mapping[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    def score(command: Mapping[str, Any]) -> tuple[Any, ...]:
        family = str(command.get("action", ""))
        belief = dict(beliefs.get(family, {}) or {})
        trials = int(belief.get("trials", 0) or 0)
        effect_rate = float(belief.get("effect_rate", 0.0) or 0.0)
        return (
            int(trials == 0),
            effect_rate,
            -trials,
            action_identity(family, dict(command.get("action_args", {}) or {})),
        )

    selected = dict(max(untried_actions, key=score))
    family = str(selected.get("action", ""))
    prior = dict(beliefs.get(family, {}) or {})
    return selected, {
        "selection_policy": "ONLINE_EPISTEMIC_THEN_OBSERVED_EFFECT_RATE",
        "action_identity": action_identity(
            family, dict(selected.get("action_args", {}) or {})
        ),
        "action_family_prior_trials": int(prior.get("trials", 0) or 0),
        "action_family_prior_effect_rate": float(prior.get("effect_rate", 0.0) or 0.0),
        "current_observation_used": True,
        "past_observed_effects_used": bool(prior),
        "future_outcomes_used": False,
    }


def verify_learned_answer(
    game_id: str,
    sequence: Sequence[Mapping[str, Any]],
    *,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    env = _new_env(game_id, environments_dir, env_factory)
    replay = execute_exam_trial(env, sequence)
    verified = bool(
        replay.get("positive_observed", False)
        and replay.get("sequence_exactly_executed", False)
        and list(replay.get("sequence", []) or []) == list(sequence)
    )
    return {
        "performed_after_exam_learning": True,
        "exact_replay_verified": verified,
        "actions_executed": int(replay.get("actions_executed", 0) or 0),
        "level_delta": int(replay.get("levels_completed_delta", 0) or 0),
        "win": bool(replay.get("win", False)),
        "sequence_match": list(replay.get("sequence", []) or []) == list(sequence),
        "used_to_reorder_exam_actions": False,
    }


def empty_learned_answer_replay() -> Dict[str, Any]:
    return {
        "performed_after_exam_learning": False,
        "exact_replay_verified": False,
        "actions_executed": 0,
        "level_delta": 0,
        "win": False,
        "sequence_match": False,
        "used_to_reorder_exam_actions": False,
    }


def compare_sage8j_exam_arms(
    game_id: str,
    control: Mapping[str, Any],
    adaptive: Mapping[str, Any],
) -> Dict[str, Any]:
    control_delta = int(control.get("levels_completed_delta_best", 0) or 0)
    adaptive_delta = int(adaptive.get("levels_completed_delta_best", 0) or 0)
    return {
        "evaluation_id": f"sage8j::{game_id}::fresh_online_exam",
        "game_id": game_id,
        "fresh_relative_to_sage8i": True,
        "same_reset_between_arms": (
            str(control.get("reset_visual_digest", ""))
            == str(adaptive.get("reset_visual_digest", ""))
        ),
        "same_structural_reset_between_arms": (
            str(control.get("reset_structural_key", ""))
            == str(adaptive.get("reset_structural_key", ""))
        ),
        "same_action_budget_between_arms": (
            int(control.get("max_action_executions", 0) or 0)
            == int(adaptive.get("max_action_executions", 0) or 0)
        ),
        "same_trial_budget_between_arms": (
            int(control.get("max_trials", 0) or 0)
            == int(adaptive.get("max_trials", 0) or 0)
        ),
        "control_levels_completed_delta": control_delta,
        "adaptive_levels_completed_delta": adaptive_delta,
        "levels_completed_absolute_gain": adaptive_delta - control_delta,
        "control_win": bool(control.get("win_observed", False)),
        "adaptive_win": bool(adaptive.get("win_observed", False)),
        "control_arm": dict(control),
        "adaptive_arm": dict(adaptive),
        "future_outcomes_used_for_action_selection": False,
        "support_counted": 0,
        "wrong_confirmations": 0,
    }


def summarize_sage8j_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    controls = [dict(row.get("control_arm", {}) or {}) for row in episodes]
    adaptives = [dict(row.get("adaptive_arm", {}) or {}) for row in episodes]
    control_levels = sum(
        int(row.get("levels_completed_delta_best", 0) or 0) for row in controls
    )
    adaptive_levels = sum(
        int(row.get("levels_completed_delta_best", 0) or 0) for row in adaptives
    )
    return {
        "fresh_games_evaluated": len(episodes),
        "control_action_executions": sum(
            int(row.get("action_executions", 0) or 0) for row in controls
        ),
        "adaptive_action_executions": sum(
            int(row.get("action_executions", 0) or 0) for row in adaptives
        ),
        "control_trials_executed": sum(
            int(row.get("trials_executed", 0) or 0) for row in controls
        ),
        "adaptive_trials_executed": sum(
            int(row.get("trials_executed", 0) or 0) for row in adaptives
        ),
        "adaptive_belief_updates": sum(
            int(row.get("belief_updates", 0) or 0) for row in adaptives
        ),
        "adaptive_hypothesis_revisions": sum(
            int(row.get("hypothesis_revisions", 0) or 0) for row in adaptives
        ),
        "adaptive_discovered_structural_states": sum(
            int(row.get("discovered_structural_states", 0) or 0) for row in adaptives
        ),
        "adaptive_duplicate_states_pruned": sum(
            int(row.get("duplicate_states_pruned", 0) or 0) for row in adaptives
        ),
        "adaptive_no_effect_states_pruned": sum(
            int(row.get("no_effect_states_pruned", 0) or 0) for row in adaptives
        ),
        "control_levels_completed_delta_total": control_levels,
        "adaptive_levels_completed_delta_total": adaptive_levels,
        "levels_completed_absolute_gain": adaptive_levels - control_levels,
        "control_wins": sum(bool(row.get("win_observed", False)) for row in controls),
        "adaptive_wins": sum(bool(row.get("win_observed", False)) for row in adaptives),
        "adaptive_answers_learned_during_exam": sum(
            bool(row.get("answer_learned_during_exam", False)) for row in adaptives
        ),
        "adaptive_learned_answers_exactly_replayed": sum(
            bool(
                row.get("learned_answer_exact_replay", {}).get(
                    "exact_replay_verified", False
                )
            )
            for row in adaptives
        ),
    }


def build_sage8j_gate(
    games: Sequence[str],
    episodes: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    *,
    max_action_executions: int,
    max_trials: int,
    max_depth: int,
) -> Dict[str, bool]:
    return {
        "fresh_targets_exclude_sage8i_games": bool(games)
        and not ({"tn36-ab4f63cc", "wa30-ee6fef47"} & set(games)),
        "every_fresh_target_evaluated_once": len(episodes) == len(games)
        and {str(row.get("game_id", "")) for row in episodes} == set(games),
        "paired_arms_share_reset_and_bounds": all(
            bool(row.get("same_reset_between_arms", False))
            and bool(row.get("same_structural_reset_between_arms", False))
            and bool(row.get("same_action_budget_between_arms", False))
            and bool(row.get("same_trial_budget_between_arms", False))
            for row in episodes
        ),
        "control_schedule_does_not_adapt_to_outcomes": all(
            bool(
                row.get("control_arm", {}).get(
                    "schedule_predeclared_before_outcomes", False
                )
            )
            and not bool(
                row.get("control_arm", {}).get(
                    "outcomes_used_to_reorder_or_prune", True
                )
            )
            for row in episodes
        ),
        "adaptive_policy_state_updates_during_exam": all(
            bool(
                row.get("adaptive_arm", {}).get(
                    "learning_algorithm_frozen_before_exam", False
                )
            )
            and not bool(
                row.get("adaptive_arm", {}).get("policy_state_frozen_during_exam", True)
            )
            and bool(
                row.get("adaptive_arm", {}).get(
                    "outcomes_used_to_reorder_or_prune", False
                )
            )
            and int(row.get("adaptive_arm", {}).get("belief_updates", 0) or 0) > 0
            for row in episodes
        ),
        "all_interactions_respect_predeclared_bounds": all(
            int(row.get(arm, {}).get("action_executions", 0) or 0)
            <= max_action_executions
            and int(row.get(arm, {}).get("trials_executed", 0) or 0) <= max_trials
            and int(row.get(arm, {}).get("max_depth", 0) or 0) == max_depth
            for row in episodes
            for arm in ("control_arm", "adaptive_arm")
        ),
        "all_selected_actions_were_live_legal": all(
            bool(row.get("control_arm", {}).get("all_selected_actions_legal", False))
            and bool(
                row.get("adaptive_arm", {}).get("all_selected_actions_legal", False)
            )
            for row in episodes
        ),
        "no_future_outcomes_or_post_exam_tuning_used": all(
            not bool(row.get("future_outcomes_used_for_action_selection", True))
            and not bool(
                row.get("control_arm", {}).get(
                    "evaluation_outcomes_used_after_exam_for_algorithm_tuning", True
                )
            )
            and not bool(
                row.get("adaptive_arm", {}).get(
                    "evaluation_outcomes_used_after_exam_for_algorithm_tuning", True
                )
            )
            for row in episodes
        ),
        "learned_answers_replay_exactly_when_observed": all(
            not bool(
                row.get("adaptive_arm", {}).get("answer_learned_during_exam", False)
            )
            or bool(
                row.get("adaptive_arm", {})
                .get("learned_answer_exact_replay", {})
                .get("exact_replay_verified", False)
            )
            for row in episodes
        ),
        "online_metrics_account_for_every_episode": int(
            metrics.get("fresh_games_evaluated", 0) or 0
        )
        == len(episodes),
        "no_truth_or_registry_support_mutation": all(
            int(row.get("support_counted", -1) or 0) == 0
            and int(row.get("wrong_confirmations", -1) or 0) == 0
            for row in episodes
        ),
    }


def summarize_sage8j(
    episodes: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    level_gain = int(metrics.get("levels_completed_absolute_gain", 0) or 0)
    adaptive_levels = int(metrics.get("adaptive_levels_completed_delta_total", 0) or 0)
    control_wins = int(metrics.get("control_wins", 0) or 0)
    adaptive_wins = int(metrics.get("adaptive_wins", 0) or 0)
    if level_gain > 0 or adaptive_wins > control_wins:
        outcome = SAGE8J_ARC_GAIN
    elif adaptive_levels > 0 or adaptive_wins > 0:
        outcome = SAGE8J_SUCCESS_NO_GAIN
    else:
        outcome = SAGE8J_ACTIVE_NO_GAIN
    return {
        **dict(metrics),
        "games_evaluated": sorted({str(row.get("game_id", "")) for row in episodes}),
        "online_learning_during_exam_performed": True,
        "learning_algorithm_frozen_before_exam": True,
        "agent_policy_state_frozen_during_exam": False,
        "observed_outcomes_used_for_next_action_selection": True,
        "future_outcomes_used_for_action_selection": False,
        "sage8i_action_traces_loaded": False,
        "primary_arc_progress_improved": level_gain > 0 or adaptive_wins > control_wins,
        "primary_arc_progress_regressed": level_gain < 0
        or adaptive_wins < control_wins,
        "outcome_status": outcome,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(gate.values()),
    }


def make_state_node(
    *,
    sequence: Sequence[Mapping[str, Any]],
    structural_key: str,
    exact_digest: str,
    legal_action_commands: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "sequence": [dict(row) for row in sequence],
        "structural_key": structural_key,
        "exact_digest": exact_digest,
        "untried_actions": [dict(row) for row in legal_action_commands],
    }


def legal_commands(env: Any) -> list[Dict[str, Any]]:
    commands = [
        {
            "action": str(getattr(action, "name", "")),
            "action_args": dict(getattr(action, "action_args", {}) or {}),
        }
        for action in _valid_actions(env)
    ]
    return sorted(
        commands,
        key=lambda row: action_identity(
            str(row.get("action", "")), dict(row.get("action_args", {}) or {})
        ),
    )


def static_sequence_schedule(
    commands: Sequence[Mapping[str, Any]],
    *,
    max_depth: int,
) -> Iterator[list[Dict[str, Any]]]:
    frozen = [dict(command) for command in commands]
    for depth in range(1, max_depth + 1):
        for sequence in itertools.product(frozen, repeat=depth):
            yield [dict(command) for command in sequence]


def public_trial_record(
    trial: Mapping[str, Any],
    *,
    trial_index: int,
) -> Dict[str, Any]:
    last = dict(trial.get("last_transition", {}) or {})
    return {
        "trial_index": int(trial_index),
        "status": str(trial.get("status", "")),
        "sequence": list(trial.get("sequence", []) or []),
        "actions_executed": int(trial.get("actions_executed", 0) or 0),
        "all_selected_actions_legal": bool(
            trial.get("all_selected_actions_legal", False)
        ),
        "positive_observed": bool(trial.get("positive_observed", False)),
        "levels_completed_delta": int(trial.get("levels_completed_delta", 0) or 0),
        "win": bool(trial.get("win", False)),
        "last_transition": last,
    }


def stable_grid(grid: np.ndarray) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int32)
    return array[:-1, :] if array.ndim == 2 and array.shape[0] > 1 else array


def visual_digest(grid: Any) -> str:
    array = np.asarray(grid, dtype=np.int32)
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def compare_bucket(left: int, right: int) -> str:
    return "BEFORE" if left < right else "AFTER" if left > right else "ALIGNED"


def _new_env(
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Any:
    return (
        env_factory(game_id)
        if env_factory is not None
        else _make_real_env(game_id, environments_dir)
    )


def write_sage8j_online_causal_exam_learning(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE8J_ONLINE_CAUSAL_EXAM_LEARNING_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments-dir")
    parser.add_argument(
        "--output", default=str(DEFAULT_SAGE8J_ONLINE_CAUSAL_EXAM_LEARNING_PATH)
    )
    parser.add_argument(
        "--max-actions", type=int, default=DEFAULT_MAX_ACTION_EXECUTIONS
    )
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_DISCOVERED_STATES)
    args = parser.parse_args(argv)
    payload = run_sage8j_online_causal_exam_learning(
        environments_dir=args.environments_dir,
        output_path=args.output,
        max_action_executions=args.max_actions,
        max_trials=args.max_trials,
        max_depth=args.max_depth,
        max_discovered_states=args.max_states,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
