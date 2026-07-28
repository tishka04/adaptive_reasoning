"""Targeted source-only functional intervention collector for SAGE12 V4.10."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.sage11.splits import SOURCE_TRAIN
from theory.unified_cognition_ab_benchmark import (
    _available_action_names,
    _is_terminal,
    _make_real_env,
)

from .action_aligned_semantics_v4_10 import (
    DEFAULT_OUTPUT_DIR,
    load_manifest,
)
from .action_target_data import (
    ActionTargetTrace,
    build_action_target_trace,
    build_observation,
    resolve_action_target,
)
from .semantic_teacher_v4_9 import (
    SemanticTeacherRecord,
    _checksum,
    _file_sha256,
    _json_safe,
    _read_jsonl,
    _write_json,
    compile_semantics,
    load_teacher_records,
)

COLLECTION_VERSION = "sage12-functional-intervention-collection-v4.10"

_TARGET_FLOORS = {
    "local_change": 80,
    "path_opened": 12,
    "path_closed": 12,
    "actor_approached_root": 12,
    "contact_gained": 10,
    "reachable_area_increased": 12,
    "reachable_area_decreased": 12,
    "target_created": 24,
    "target_removed": 24,
    "target_moved": 20,
    "productive": 60,
    "risk": 20,
}


def _record_stratum(record: SemanticTeacherRecord) -> str:
    root = record.graph.root
    return "|".join(
        (
            str(root.get("action_name", "unknown")),
            str(root.get("root_kind", "unknown")),
            str(root.get("actor_relation", "unknown")),
            str(root.get("path_status", "unknown")),
            str(root.get("root_affordance", "none")),
        )
    )


def _candidate_stratum(action: Any, observation: Any) -> str:
    anchor = resolve_action_target(
        observation,
        str(action.name),
        dict(action.action_args),
    )
    if anchor.occupied:
        root_kind = "occupied_object"
    elif anchor.in_bounds:
        root_kind = "virtual_cell"
    elif observation.best_player is not None:
        root_kind = "actor"
    else:
        root_kind = "action_root"
    return "|".join(
        (
            str(action.name),
            root_kind,
            str(anchor.actor_relation),
            str(anchor.path_status),
            str(anchor.target_affordance),
        )
    )


def _initial_statistics(
    records: Sequence[SemanticTeacherRecord],
    *,
    game: str,
) -> tuple[
    Counter[str],
    Counter[str],
    dict[str, Counter[str]],
    Counter[str],
]:
    action_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    yields: dict[str, Counter[str]] = defaultdict(Counter)
    trials: Counter[str] = Counter()
    for record in records:
        if record.game_id != game:
            continue
        root = record.graph.root
        action_counts[str(root.get("action_name", "unknown"))] += 1
        stratum = _record_stratum(record)
        trials[stratum] += 1
        for effect in _TARGET_FLOORS:
            if record.applicable[effect] and record.labels[effect]:
                positive_counts[effect] += 1
                yields[stratum][effect] += 1
    return action_counts, positive_counts, yields, trials


def select_functional_action(
    legal: Sequence[Any],
    *,
    observation: Any,
    action_counts: Mapping[str, int],
    positive_counts: Mapping[str, int],
    yield_counts: Mapping[str, Mapping[str, int]],
    trial_counts: Mapping[str, int],
    exploration_fraction: float,
    salt: str,
) -> Any:
    """Select an outcome-blind action using past source-only functional yield."""

    candidates = [(action, _candidate_stratum(action, observation)) for action in legal]
    rng = random.Random(f"sage12-v4.10:{salt}")
    if rng.random() < float(exploration_fraction):
        minimum_action = min(
            int(action_counts.get(str(action.name), 0))
            for action, _stratum in candidates
        )
        candidates = [
            item
            for item in candidates
            if int(action_counts.get(str(item[0].name), 0)) == minimum_action
        ]
        minimum_trials = min(
            int(trial_counts.get(stratum, 0)) for _action, stratum in candidates
        )
        candidates = [
            item
            for item in candidates
            if int(trial_counts.get(item[1], 0)) == minimum_trials
        ]
        return candidates[rng.randrange(len(candidates))][0]

    total_trials = sum(int(value) for value in trial_counts.values())

    def score(item: tuple[Any, str]) -> tuple[float, float, str]:
        action, stratum = item
        trials = int(trial_counts.get(stratum, 0))
        value = 0.0
        deficit_total = 0.0
        for effect, floor in _TARGET_FLOORS.items():
            deficit = max(0.0, float(floor - int(positive_counts.get(effect, 0))))
            deficit_total += deficit
            positives = int(yield_counts.get(stratum, {}).get(effect, 0))
            rate = (positives + 1.0) / (trials + 2.0)
            value += deficit * rate
        uncertainty = math.sqrt(math.log(total_trials + 2.0) / (trials + 1.0))
        value += 0.02 * deficit_total * uncertainty
        coverage = -float(action_counts.get(str(action.name), 0))
        tie = hashlib.sha256(f"{salt}:{stratum}".encode()).hexdigest()
        return value, coverage, tie

    return max(candidates, key=score)[0]


def _load_shard(path: Path) -> list[ActionTargetTrace]:
    if not path.exists():
        return []
    return [ActionTargetTrace.from_dict(row) for row in _read_jsonl(path)]


def _write_trace_shard(path: Path, records: Sequence[ActionTargetTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _json_safe(record.to_dict()),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def _collect_game(
    *,
    game: str,
    target_rows: int,
    base_records: Sequence[SemanticTeacherRecord],
    existing: Sequence[ActionTargetTrace],
    environment_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[ActionTargetTrace], dict[str, Any]]:
    records = list(existing)
    base_repeat_keys = {
        record.exact_repeat_key for record in base_records if record.game_id == game
    }
    repeat_keys = base_repeat_keys | {record.exact_repeat_key() for record in records}
    action_counts, positive_counts, yields, trials = _initial_statistics(
        base_records,
        game=game,
    )
    for trace in records:
        labels, applicable, _score, _evidence = compile_semantics(trace)
        observation = build_observation(
            trace.frame_before,
            available_actions=trace.available_action_names,
            game_state=trace.game_state_before,
            levels_completed=trace.levels_completed_before,
        )
        preview = type(
            "_ActionPreview",
            (),
            {
                "name": trace.selected_action_name,
                "action_args": dict(trace.selected_action_data),
            },
        )()
        stratum = _candidate_stratum(preview, observation)
        action_counts[trace.selected_action_name] += 1
        trials[stratum] += 1
        for effect in _TARGET_FLOORS:
            if applicable[effect] and labels[effect]:
                positive_counts[effect] += 1
                yields[stratum][effect] += 1

    collection = manifest["collection"]
    seeds = tuple(int(value) for value in collection["policy_seeds"])
    maximum_resets = int(collection["maximum_resets_per_game"])
    action_budget = int(collection["action_budget_per_reset"])
    raw_steps = 0
    duplicate_rejections = 0
    resets_used = 0
    terminal_events = 0
    for reset_index in range(maximum_resets):
        if len(records) >= target_rows:
            break
        seed = seeds[reset_index % len(seeds)]
        env = _make_real_env(game, environment_root)
        try:
            frame = _reset_env(env)
        except ModuleNotFoundError as exc:
            if exc.name != "arcengine":
                raise
            frame = env.step(0)
        resets_used += 1
        for step_index in range(action_budget):
            before = snapshot_frame(frame)
            if _is_terminal(before.game_state):
                break
            legal = tuple(
                action
                for action in _valid_actions(env)
                if str(action.name) not in {"", "RESET"}
            )
            if not legal:
                break
            available = tuple(sorted(set(_available_action_names(legal))))
            observation = build_observation(
                before.grid,
                available_actions=available,
                game_state=before.game_state,
                levels_completed=before.levels_completed,
                infer_players=True,
            )
            selected = select_functional_action(
                legal,
                observation=observation,
                action_counts=action_counts,
                positive_counts=positive_counts,
                yield_counts=yields,
                trial_counts=trials,
                exploration_fraction=float(collection["exploration_fraction"]),
                salt=f"{game}:{seed}:{reset_index}:{step_index}:{raw_steps}",
            )
            stratum = _candidate_stratum(selected, observation)
            after_frame = _step_env_action(env, selected)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
            trace = build_action_target_trace(
                game_id=game,
                source_split="source_train",
                policy_seed=seed,
                reset_index=reset_index,
                step_index=step_index,
                collection_phase="v4_10_functional_targeted",
                available_action_names=available,
                selected_action_name=str(selected.name),
                selected_action_data=dict(selected.action_args),
                frame_before=before.grid,
                frame_after=after.grid,
                game_state_before=before.game_state,
                game_state_after=after.game_state,
                levels_completed_before=before.levels_completed,
                levels_completed_after=after.levels_completed,
            )
            raw_steps += 1
            action_counts[trace.selected_action_name] += 1
            trials[stratum] += 1
            labels, applicable, _score, _evidence = compile_semantics(trace)
            for effect in _TARGET_FLOORS:
                if applicable[effect] and labels[effect]:
                    positive_counts[effect] += 1
                    yields[stratum][effect] += 1
            key = trace.exact_repeat_key()
            if key in repeat_keys:
                duplicate_rejections += 1
            else:
                repeat_keys.add(key)
                records.append(trace)
                terminal_events += int(labels["level_complete"])
            frame = after_frame
            if len(records) >= target_rows or _is_terminal(after.game_state):
                break
    return records, {
        "game_id": game,
        "target_rows": target_rows,
        "rows": len(records),
        "row_ratio": len(records) / max(target_rows, 1),
        "raw_steps": raw_steps,
        "resets_used": resets_used,
        "duplicate_rejections": duplicate_rejections,
        "new_completion_events": terminal_events,
        "positive_counts_after_historical_and_fresh": {
            effect: int(positive_counts[effect]) for effect in _TARGET_FLOORS
        },
        "action_counts_after_historical_and_fresh": dict(sorted(action_counts.items())),
    }


def run_collection(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    base_records = load_teacher_records()
    environment_root = (
        Path(environments_dir) if environments_dir is not None else _env_dir()
    )
    shard_dir = destination / "source_train_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for game in SOURCE_TRAIN:
        path = shard_dir / f"{game}.jsonl"
        existing = _load_shard(path)
        target = int(manifest["collection"]["rows_per_game"][game])
        if len(existing) < target:
            existing, report = _collect_game(
                game=game,
                target_rows=target,
                base_records=base_records,
                existing=existing,
                environment_root=environment_root,
                manifest=manifest,
            )
            _write_trace_shard(path, existing)
            reports[game] = report
        else:
            reports[game] = {
                "game_id": game,
                "target_rows": target,
                "rows": len(existing),
                "row_ratio": len(existing) / max(target, 1),
                "resumed_complete_shard": True,
            }
    shards = [
        {
            "game_id": game,
            "path": (shard_dir / f"{game}.jsonl").as_posix(),
            "rows": len(_load_shard(shard_dir / f"{game}.jsonl")),
            "sha256": _file_sha256(shard_dir / f"{game}.jsonl"),
        }
        for game in SOURCE_TRAIN
    ]
    threshold = float(manifest["collection"]["minimum_rows_ratio_per_game"])
    result: dict[str, Any] = {
        "format_version": COLLECTION_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(SOURCE_TRAIN),
        "target_rows": int(manifest["collection"]["target_rows"]),
        "rows": sum(int(row["rows"]) for row in shards),
        "reports": reports,
        "shards": shards,
        "minimum_row_ratio": min(
            int(row["rows"])
            / int(manifest["collection"]["rows_per_game"][row["game_id"]])
            for row in shards
        ),
        "offline_environment_opened": True,
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
    }
    result["checks"] = {
        "minimum_rows_ratio_per_game": (result["minimum_row_ratio"] >= threshold),
        "all_games_source_train": set(result["games"]) == set(SOURCE_TRAIN),
    }
    result["collection_ready"] = all(result["checks"].values())
    result["collection_checksum"] = _checksum(result)
    _write_json(destination / "collection_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--environments-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_collection(
        output_dir=args.output_dir,
        environments_dir=args.environments_dir,
    )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_collection",
    "select_functional_action",
]
