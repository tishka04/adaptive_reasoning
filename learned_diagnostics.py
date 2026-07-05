"""PASSIVE learned diagnostics: the models score branches but never decide.

The agent is driven by the handmade heuristic teacher. At every step we expand
all available actions, score each branch with BOTH the handmade teacher and the
learned scorer, and log:

    - agreement rate of argmax(handmade) vs argmax(learned)
    - mean rank correlation between the two score vectors
    - the branches where the learned model most disagrees

This is the gate before letting the learned score steer the search: if the
learned best is systematically absurd (prefers death/no-op), we see it here
WITHOUT breaking the agent.

Run with the bundled env interpreter:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe learned_diagnostics.py \\
        --games public_seen --action-effect models\\action_effect.joblib \\
        --value models\\value.joblib
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from level7_frontier_recovery import (
    Arcade,
    OperationMode,
    ENV_DIR,
    PROJECT_ROOT,
    _action_enum,
    _available_names_from_raw,
    _primary_grid,
    _resolve_full_game_id,
    _state_name,
)

import game_splits
from abstraction_dataset_io import load_dataset, make_history_features, state_matrix
from build_break_event_dataset import break_class
from extract_state_abstractions import (
    FEATURE_SCHEMA,
    extract_state_features,
    largest_component_local_features,
)
from learned_scoring import LearnedScorer

ACTIONS = [f"ACTION{i}" for i in range(1, 8)]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "learned_diagnostics"


def _spearman(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-9 or np.std(rb) < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _action_data(action: str, shape) -> Optional[Dict[str, int]]:
    if action != "ACTION6":
        return None
    h, w = shape
    return {"x": int(w) // 2, "y": int(h) // 2}


def _step(env, full_game_id, action, action_data):
    data: Dict[str, Any] = {"game_id": full_game_id}
    if action_data:
        data.update(action_data)
    try:
        return env.step(_action_enum(action), data=data)
    except TypeError:
        return env.step(_action_enum(action))


def _changed(a, b) -> int:
    left = np.array(a, dtype=np.int32)
    right = np.array(b, dtype=np.int32)
    if left.shape != right.shape:
        return int(max(left.size, right.size))
    return int(np.count_nonzero(left != right))


def _local_features_by_action(grid, actions: List[str], shape) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for action in actions:
        action_data = _action_data(action, shape)
        cursor = None
        if action_data and "x" in action_data and "y" in action_data:
            cursor = (float(action_data["y"]), float(action_data["x"]))
        out[action] = largest_component_local_features(grid, cursor=cursor)
    return out


def _branch_outcome(env, full_game_id: str, action: str, action_data, grid, feats) -> Dict[str, Any]:
    probe = copy.deepcopy(env)
    raw = _step(probe, full_game_id, action, action_data)
    state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
    next_grid = _primary_grid(raw) if raw is not None else grid
    next_feats = extract_state_features(next_grid)
    delta_largest = float(next_feats.get("largest_component_size", 0.0)) - float(
        feats.get("largest_component_size", 0.0)
    )
    changed = _changed(grid, next_grid)
    return {
        "action": action,
        "state": state,
        "changed_cells": changed,
        "is_no_op": changed == 0,
        "is_game_over": state in {"GAME_OVER", "ERROR"},
        "delta_largest_component_size": delta_largest,
        "break_class": break_class(delta_largest),
        "is_big_break": break_class(delta_largest) == "BIG_BREAK",
    }


def _top_action_record(
    scores,
    outcomes: Dict[str, Dict[str, Any]],
    *,
    score_attr: str,
) -> Dict[str, Any]:
    if not scores:
        return {}
    best = max(scores, key=lambda item: float(getattr(item, score_attr)))
    outcome = outcomes[best.action]
    return {
        "action": best.action,
        "score": round(float(getattr(best, score_attr)), 6),
        "break_probability": round(float(best.predicted_break_probability), 6),
        "break_bonus": round(float(best.break_bonus), 6),
        "break_bonus_applied": bool(best.break_bonus_applied),
        "predicted_danger": round(float(best.predicted_danger), 6),
        "predicted_no_op": round(float(best.predicted_macro_scores.get("explore", 0.0)), 6),
        "is_big_break": bool(outcome["is_big_break"]),
        "is_no_op": bool(outcome["is_no_op"]),
        "is_game_over": bool(outcome["is_game_over"]),
        "true_delta_largest_component_size": round(
            float(outcome["delta_largest_component_size"]),
            4,
        ),
    }


def diagnose_game(
    arc: Arcade,
    scorer: LearnedScorer,
    full_game_id: str,
    *,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    env = arc.make(full_game_id)
    raw = getattr(env, "observation_space", None)
    if raw is None:
        return {"game_id": full_game_id, "error": "no observation_space"}
    grid = _primary_grid(raw)
    feats = extract_state_features(grid)

    agreements = 0
    comparisons = 0
    spearmans: List[float] = []
    learned_prefers_worse = 0  # learned best != handmade best AND lower handmade
    worst_disagreements: List[Dict[str, Any]] = []
    branch_records: List[Dict[str, Any]] = []
    top_learned_records: List[Dict[str, Any]] = []
    top_general_records: List[Dict[str, Any]] = []
    top_break_records: List[Dict[str, Any]] = []
    action_history: List[str] = []
    action_repeat_count = 0
    steps_since_state_change = 0

    for step_index in range(max_steps):
        available = [a for a in (_available_names_from_raw(raw) or ACTIONS) if a in ACTIONS]
        if not available:
            break
        shape = np.array(grid, dtype=np.int32).shape
        shape = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (64, 64)

        history_features = make_history_features(
            action_history,
            action_repeat_count,
            steps_since_state_change,
        )
        local_by_action = _local_features_by_action(grid, available, shape)
        scores = scorer.score_actions(
            feats,
            available,
            history_features,
            local_by_action,
        )
        outcomes = {
            action: _branch_outcome(env, full_game_id, action, _action_data(action, shape), grid, feats)
            for action in available
        }
        score_by_action = {score.action: score for score in scores}
        for action in available:
            score = score_by_action[action]
            outcome = outcomes[action]
            branch_records.append(
                {
                    "game_id": full_game_id,
                    "step": step_index,
                    "action": action,
                    "learned_score": round(float(score.learned_score), 6),
                    "general_score": round(float(score.general_score), 6),
                    "break_probability": round(float(score.predicted_break_probability), 6),
                    "break_bonus": round(float(score.break_bonus), 6),
                    "break_bonus_applied": bool(score.break_bonus_applied),
                    "predicted_danger": round(float(score.predicted_danger), 6),
                    "predicted_no_op": round(float(score.predicted_macro_scores.get("explore", 0.0)), 6),
                    "is_big_break": bool(outcome["is_big_break"]),
                    "is_no_op": bool(outcome["is_no_op"]),
                    "is_game_over": bool(outcome["is_game_over"]),
                    "true_delta_largest_component_size": round(
                        float(outcome["delta_largest_component_size"]),
                        4,
                    ),
                }
            )
        top_learned_records.append(_top_action_record(scores, outcomes, score_attr="learned_score"))
        top_general_records.append(_top_action_record(scores, outcomes, score_attr="general_score"))
        top_break_records.append(_top_action_record(scores, outcomes, score_attr="predicted_break_probability"))

        handmade = [s.handmade_score for s in scores]
        learned = [s.learned_score for s in scores]
        hm_best = int(np.argmax(handmade))
        ln_best = int(np.argmax(learned))
        comparisons += 1
        if available[hm_best] == available[ln_best]:
            agreements += 1
        else:
            gap = handmade[hm_best] - handmade[ln_best]
            if gap > 1e-6:
                learned_prefers_worse += 1
            worst_disagreements.append(
                {
                    "game_id": full_game_id,
                    "step": step_index,
                    "handmade_best": available[hm_best],
                    "learned_best": available[ln_best],
                    "handmade_gap": round(float(gap), 4),
                    "learned_pick_progress": round(scores[ln_best].predicted_progress, 4),
                    "learned_pick_danger": round(scores[ln_best].predicted_danger, 4),
                    "learned_pick_break_probability": round(
                        scores[ln_best].predicted_break_probability,
                        4,
                    ),
                    "learned_pick_break_bonus": round(scores[ln_best].break_bonus, 4),
                    "learned_pick_true_big_break": bool(outcomes[available[ln_best]]["is_big_break"]),
                    "learned_pick_true_no_op": bool(outcomes[available[ln_best]]["is_no_op"]),
                    "learned_pick_true_game_over": bool(outcomes[available[ln_best]]["is_game_over"]),
                }
            )
        spearmans.append(_spearman(handmade, learned))

        # PASSIVE: the handmade teacher drives the agent.
        chosen = available[hm_best]
        next_raw = _step(env, full_game_id, chosen, _action_data(chosen, shape))
        state = _state_name(getattr(next_raw, "state", "ERROR")) if next_raw is not None else "ERROR"
        chosen_outcome = outcomes[chosen]
        action_repeat_count = (
            action_repeat_count + 1
            if action_history and action_history[-1] == chosen
            else 1
        )
        action_history.append(chosen)
        steps_since_state_change = (
            0 if int(chosen_outcome["changed_cells"]) > 0 else steps_since_state_change + 1
        )
        if next_raw is None or state in {"GAME_OVER", "WIN", "ERROR"}:
            break
        raw = next_raw
        grid = _primary_grid(raw)
        feats = extract_state_features(grid)

    worst_disagreements.sort(key=lambda d: -d["handmade_gap"])
    def _rate(records: List[Dict[str, Any]], key: str) -> float:
        return round(float(np.mean([1.0 if r.get(key) else 0.0 for r in records])), 4) if records else 0.0

    def _top_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(records),
            "big_break_rate": _rate(records, "is_big_break"),
            "no_op_rate": _rate(records, "is_no_op"),
            "game_over_rate": _rate(records, "is_game_over"),
            "mean_break_probability": round(
                float(np.mean([r.get("break_probability", 0.0) for r in records])),
                4,
            )
            if records
            else 0.0,
            "break_bonus_applied_rate": _rate(records, "break_bonus_applied"),
        }

    return {
        "game_id": full_game_id,
        "steps": comparisons,
        "agreement_rate": round(agreements / comparisons, 4) if comparisons else 0.0,
        "mean_rank_correlation": round(float(np.mean(spearmans)), 4) if spearmans else 0.0,
        "learned_prefers_worse_rate": round(learned_prefers_worse / comparisons, 4) if comparisons else 0.0,
        "branch_count": len(branch_records),
        "branch_big_break_rate": _rate(branch_records, "is_big_break"),
        "top_learned": _top_summary(top_learned_records),
        "top_general": _top_summary(top_general_records),
        "top_break_probability": _top_summary(top_break_records),
        "top_learned_records": top_learned_records,
        "top_disagreements": worst_disagreements[:5],
        "branch_records": branch_records,
    }


def _records_from_reports(reports: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for report in reports:
        out.extend(report.get(key, []))
    return out


def _boolean_rate(records: List[Dict[str, Any]], key: str) -> float:
    return round(float(np.mean([1.0 if r.get(key) else 0.0 for r in records])), 4) if records else 0.0


def _top_records_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(records),
        "big_break_rate": _boolean_rate(records, "is_big_break"),
        "no_op_rate": _boolean_rate(records, "is_no_op"),
        "game_over_rate": _boolean_rate(records, "is_game_over"),
        "break_bonus_applied_rate": _boolean_rate(records, "break_bonus_applied"),
    }


def _ranking_summary(
    branch_records: List[Dict[str, Any]],
    *,
    score_key: str,
    top_ks: List[int],
) -> Dict[str, Any]:
    if not branch_records:
        return {}
    sorted_records = sorted(
        branch_records,
        key=lambda row: float(row.get(score_key, 0.0)),
        reverse=True,
    )
    positives = sum(1 for row in branch_records if row.get("is_big_break"))
    out: Dict[str, Any] = {
        "score_key": score_key,
        "branch_count": len(branch_records),
        "positive_count": positives,
        "base_rate": round(float(positives / len(branch_records)), 4),
        "top_k": {},
    }
    for raw_k in top_ks:
        k = min(max(1, raw_k), len(sorted_records))
        top = sorted_records[:k]
        hits = sum(1 for row in top if row.get("is_big_break"))
        out["top_k"][str(raw_k)] = {
            "k_effective": k,
            "hits": int(hits),
            "precision_at_k": round(float(hits / k), 4),
            "recall_at_k": round(float(hits / positives), 4) if positives else None,
            "no_op_rate": _boolean_rate(top, "is_no_op"),
            "game_over_rate": _boolean_rate(top, "is_game_over"),
            "mean_break_probability": round(
                float(np.mean([row.get("break_probability", 0.0) for row in top])),
                4,
            ),
            "break_bonus_applied_rate": _boolean_rate(top, "break_bonus_applied"),
        }
    return out


def diagnose_dataset(
    scorer: LearnedScorer,
    *,
    dataset_path: str,
    games: str,
    sources: Optional[List[str]],
    top_ks: List[int],
) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    if games and games.lower() != "all":
        data = data.filter_games(game_splits.resolve(games, full_ids=True))
    data = data.filter_sources(sources)
    branch_records: List[Dict[str, Any]] = []
    rows = data.rows
    if not rows:
        return {
            "dataset": dataset_path,
            "games": games,
            "sources": sources,
            "branch_summary": {},
            "ranking_by_final_score": {},
            "ranking_by_general_score": {},
            "ranking_by_break_probability": {},
        }

    actions = [str(row.get("action", "")) for row in rows]
    x_effect = np.stack(
        [
            scorer._input_for_names(
                scorer.effect_input_names,
                row.get("state_features", {}),
                actions[i],
                row.get("history_features", {}),
                row.get("largest_component_features", {}),
            )
            for i, row in enumerate(rows)
        ]
    )
    pred_delta = scorer.effect_model.predict(x_effect)
    state = state_matrix(rows)
    next_state = state.copy()
    feature_pos = {name: i for i, name in enumerate(FEATURE_SCHEMA)}
    for target_index, target in enumerate(scorer.effect_targets):
        if not target.startswith("delta_"):
            continue
        base = target[len("delta_") :]
        if base in feature_pos:
            next_state[:, feature_pos[base]] += pred_delta[:, target_index]
    progress = scorer.progress_reg.predict(next_state)
    if isinstance(scorer.danger_clf, dict):
        danger = np.full(len(rows), float(scorer.danger_clf.get("constant", 0.0)), dtype=np.float32)
    else:
        danger = scorer.danger_clf.predict_proba(next_state)[:, 1]
    corr_index = scorer.effect_targets.index("delta_top_pair_0_global_correspondence")
    general = (
        scorer.progress_weight * progress
        + scorer.correspondence_weight * pred_delta[:, corr_index]
        - scorer.danger_weight * danger
    )

    macro_predictions: List[Dict[str, float]] = [{} for _ in rows]
    macro_bonus = np.zeros(len(rows), dtype=np.float32)
    if scorer.macro_model is not None and scorer.macro_input_names:
        x_macro = np.stack(
            [
                scorer._input_for_names(
                    scorer.macro_input_names,
                    row.get("state_features", {}),
                    actions[i],
                    row.get("history_features", {}),
                    row.get("largest_component_features", {}),
                )
                for i, row in enumerate(rows)
            ]
        )
        macro_raw = scorer.macro_model.predict(x_macro)
        for i in range(len(rows)):
            scores = {
                name: max(0.0, float(macro_raw[i, j]))
                for j, name in enumerate(scorer.macro_targets)
            }
            macro_predictions[i] = scores
            macro_bonus[i] = scorer._macro_bonus(scores)

    break_probability = np.zeros(len(rows), dtype=np.float32)
    if scorer.break_model is not None and scorer.break_input_names:
        x_break = np.stack(
            [
                scorer._input_for_names(
                    scorer.break_input_names,
                    row.get("state_features", {}),
                    actions[i],
                    row.get("history_features", {}),
                    row.get("largest_component_features", {}),
                )
                for i, row in enumerate(rows)
            ]
        )
        classes = list(getattr(scorer.break_model, "classes_", []))
        if 1 in classes:
            break_probability = scorer.break_model.predict_proba(x_break)[:, classes.index(1)]

    break_bonus = np.zeros(len(rows), dtype=np.float32)
    break_applied = np.zeros(len(rows), dtype=bool)
    for i, action in enumerate(actions):
        bonus, applied = scorer._break_bonus(
            action=action,
            break_probability=float(break_probability[i]),
            predicted_danger=float(danger[i]),
            predicted_macro_scores=macro_predictions[i],
        )
        break_bonus[i] = bonus
        break_applied[i] = applied
    learned = general + macro_bonus + break_bonus

    for i, row in enumerate(rows):
        action = actions[i]
        changed_cells = int(row.get("changed_cells", 0))
        branch_records.append(
            {
                "game_id": row.get("game_id", ""),
                "episode_source": row.get("episode_source", ""),
                "action": action,
                "learned_score": round(float(learned[i]), 6),
                "general_score": round(float(general[i]), 6),
                "break_probability": round(float(break_probability[i]), 6),
                "break_bonus": round(float(break_bonus[i]), 6),
                "break_bonus_applied": bool(break_applied[i]),
                "predicted_danger": round(float(danger[i]), 6),
                "predicted_no_op": round(float(macro_predictions[i].get("explore", 0.0)), 6),
                "is_big_break": bool(row.get("is_big_break", False)),
                "is_no_op": changed_cells == 0,
                "is_game_over": bool(row.get("game_over", False)),
                "true_delta_largest_component_size": round(
                    float(row.get("delta_largest_component_size", 0.0)),
                    4,
                ),
            }
        )
    return {
        "dataset": dataset_path,
        "games": games,
        "sources": sources,
        "branch_summary": _top_records_summary(branch_records),
        "ranking_by_final_score": _ranking_summary(branch_records, score_key="learned_score", top_ks=top_ks),
        "ranking_by_general_score": _ranking_summary(branch_records, score_key="general_score", top_ks=top_ks),
        "ranking_by_break_probability": _ranking_summary(branch_records, score_key="break_probability", top_ks=top_ks),
        "ranking_by_break_bonus": _ranking_summary(branch_records, score_key="break_bonus", top_ks=top_ks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Passive learned diagnostics (models never decide).")
    parser.add_argument("--games", default="public_seen")
    parser.add_argument("--action-effect", default="models/action_effect.joblib")
    parser.add_argument("--value", default="models/value.joblib")
    parser.add_argument("--macro-scores", default="models/macro_scores_history.joblib")
    parser.add_argument("--with-break-expert", action="store_true")
    parser.add_argument("--break-classifier", default="models/break_classifier.joblib")
    parser.add_argument("--break-bonus-weight", type=float, default=2.0)
    parser.add_argument("--break-noop-threshold", type=float, default=0.7)
    parser.add_argument("--break-danger-threshold", type=float, default=0.25)
    parser.add_argument("--action6-break-threshold", type=float, default=0.85)
    parser.add_argument("--macro-score-weight", type=float, default=0.05)
    parser.add_argument("--top-k", default="10,25,50,100")
    parser.add_argument("--dataset-eval", default=None, help="Optional natural branch JSONL to rank offline.")
    parser.add_argument("--dataset-only", action="store_true", help="Run only --dataset-eval, skipping env stepping.")
    parser.add_argument("--dataset-sources", default=None, help="Comma list of dataset sources to keep.")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR / "report.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    macro_scores_path = args.macro_scores if Path(args.macro_scores).exists() else None
    break_classifier_path = (
        args.break_classifier
        if args.with_break_expert and Path(args.break_classifier).exists()
        else None
    )
    scorer = LearnedScorer(
        args.action_effect,
        args.value,
        macro_scores_path=macro_scores_path,
        break_classifier_path=break_classifier_path,
        break_bonus_weight=args.break_bonus_weight,
        break_noop_threshold=args.break_noop_threshold,
        break_danger_threshold=args.break_danger_threshold,
        action6_break_threshold=args.action6_break_threshold,
        macro_score_weight=args.macro_score_weight if macro_scores_path else 0.0,
    )
    games = game_splits.resolve(args.games, full_ids=True)
    top_ks = [int(k.strip()) for k in args.top_k.split(",") if k.strip()]

    reports = []
    if not args.dataset_only:
        for game in games:
            full_game_id = _resolve_full_game_id(arc, game)
            report = diagnose_game(arc, scorer, full_game_id, max_steps=args.max_steps, seed=args.seed)
            reports.append(report)
            if not args.quiet:
                print(
                    f"[{full_game_id}] agree={report.get('agreement_rate')} "
                    f"rankcorr={report.get('mean_rank_correlation')} "
                    f"worse={report.get('learned_prefers_worse_rate')} "
                    f"top_big={report.get('top_learned', {}).get('big_break_rate')} "
                    f"top_noop={report.get('top_learned', {}).get('no_op_rate')} "
                    f"steps={report.get('steps')}"
                )

    branch_records = _records_from_reports(reports, "branch_records")
    top_learned_records = _records_from_reports(reports, "top_learned_records")
    # Keep per-game reports compact-ish by retaining summaries and examples, not every branch twice.
    per_game = []
    for report in reports:
        compact = dict(report)
        compact.pop("branch_records", None)
        per_game.append(compact)

    agg = {
        "games": len(reports),
        "mean_agreement_rate": round(float(np.mean([r["agreement_rate"] for r in reports])), 4) if reports else 0.0,
        "mean_rank_correlation": round(float(np.mean([r["mean_rank_correlation"] for r in reports])), 4) if reports else 0.0,
        "mean_learned_prefers_worse": round(float(np.mean([r["learned_prefers_worse_rate"] for r in reports])), 4) if reports else 0.0,
        "branch_summary": _top_records_summary(branch_records),
        "top_learned_summary": _top_records_summary(top_learned_records),
        "ranking_by_final_score": _ranking_summary(branch_records, score_key="learned_score", top_ks=top_ks),
        "ranking_by_general_score": _ranking_summary(branch_records, score_key="general_score", top_ks=top_ks),
        "ranking_by_break_probability": _ranking_summary(branch_records, score_key="break_probability", top_ks=top_ks),
        "ranking_by_break_bonus": _ranking_summary(branch_records, score_key="break_bonus", top_ks=top_ks),
    }
    config = {
        "action_effect": args.action_effect,
        "value": args.value,
        "macro_scores": macro_scores_path,
        "with_break_expert": bool(args.with_break_expert),
        "break_classifier": break_classifier_path,
        "break_bonus_weight": args.break_bonus_weight,
        "break_noop_threshold": args.break_noop_threshold,
        "break_danger_threshold": args.break_danger_threshold,
        "action6_break_threshold": args.action6_break_threshold,
        "macro_score_weight": args.macro_score_weight if macro_scores_path else 0.0,
    }
    dataset_report = None
    if args.dataset_eval:
        sources = [s.strip() for s in args.dataset_sources.split(",")] if args.dataset_sources else None
        dataset_report = diagnose_dataset(
            scorer,
            dataset_path=args.dataset_eval,
            games=args.games,
            sources=sources,
            top_ks=top_ks,
        )
        if not args.quiet:
            rank = dataset_report["ranking_by_final_score"]
            print(
                "\nDATASET-EVAL: "
                f"branches={rank.get('branch_count')} positives={rank.get('positive_count')} "
                f"base={rank.get('base_rate')}"
            )
            for k, item in rank.get("top_k", {}).items():
                print(
                    f"  final top@{k}: hits={item['hits']} "
                    f"P@k={item['precision_at_k']} R@k={item['recall_at_k']} "
                    f"noop={item['no_op_rate']} gameover={item['game_over_rate']}"
                )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": config,
                "aggregate": agg,
                "dataset_eval": dataset_report,
                "per_game": per_game,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"\nAGGREGATE: {agg}")
        print(f"Gate hint: advance to active reinjection only if mean_learned_prefers_worse is low.")
        print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
