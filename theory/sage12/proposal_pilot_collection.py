"""Collect the frozen source-only SAGE12 proposal-pilot traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from theory.live_transition_loop import (
    build_observation,
    build_transition_record,
)
from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.unified_cognition_ab_benchmark import (
    _available_action_names,
    _is_terminal,
    _make_real_env,
)

from .proposal_pilot_data import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    ProposalPilotTrace,
    graph_to_mapping,
    load_frozen_manifest,
    shard_metadata,
)
from .scene_graph import build_scene_graph
from .world_model import observed_semantic_effects


COLLECTION_FORMAT_VERSION = "sage12-proposal-collection-v1"


def run_collection(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    shard_dir = destination / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    environment_root = (
        Path(environments_dir)
        if environments_dir is not None
        else _env_dir()
    )
    quotas = {
        str(game): int(quota)
        for game, quota in dict(frozen["game_quotas"]).items()
    }
    pending = {
        game: quota
        for game, quota in quotas.items()
        if not _completed_shard(
            shard_dir / f"{game}.jsonl",
            game=game,
            quota=quota,
        )
    }
    worker_count = max(1, min(int(workers), len(pending) or 1))
    reports: dict[str, Any] = {}
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _collect_game,
                    game=game,
                    quota=quota,
                    shard_path=shard_dir / f"{game}.jsonl",
                    environment_root=environment_root,
                    frozen=frozen,
                ): game
                for game, quota in pending.items()
            }
            for future in as_completed(futures):
                game = futures[future]
                reports[game] = future.result()
    else:
        for game, quota in pending.items():
            reports[game] = _collect_game(
                game=game,
                quota=quota,
                shard_path=shard_dir / f"{game}.jsonl",
                environment_root=environment_root,
                frozen=frozen,
            )
    metadata = [
        shard_metadata(
            shard_dir / f"{game}.jsonl",
            expected_game=game,
            expected_rows=quota,
        )
        for game, quota in quotas.items()
    ]
    combined = hashlib.sha256(
        "".join(item["sha256"] for item in metadata).encode("ascii")
    ).hexdigest()
    manifest = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "status": "COMPLETE",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "target_transitions": frozen["target_transitions"],
        "total_transitions": sum(item["rows"] for item in metadata),
        "shards": metadata,
        "combined_shard_checksum": combined,
        "source_train_rows": sum(
            item["rows"]
            for item in metadata
            if item["source_split"] == "source_train"
        ),
        "source_validation_rows": sum(
            item["rows"]
            for item in metadata
            if item["source_split"] == "source_validation"
        ),
        "source_games": list(quotas),
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
        "game_reports": reports,
    }
    _write_json_atomic(destination / "collection_manifest.json", manifest)
    return manifest


def _collect_game(
    *,
    game: str,
    quota: int,
    shard_path: Path,
    environment_root: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    split = str(dict(frozen["game_splits"])[game])
    seeds = tuple(int(seed) for seed in frozen["collection"]["policy_seeds"])
    action_budget = int(frozen["collection"]["action_budget_per_reset"])
    maximum_resets = int(frozen["collection"]["maximum_resets_per_game"])
    repeat_cap = int(frozen["collection"]["state_action_repeat_cap"])
    records: list[ProposalPilotTrace] = []
    repeats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    raw_steps = 0
    resets = 0
    while len(records) < quota and resets < maximum_resets:
        seed = seeds[resets % len(seeds)]
        env = _make_real_env(game, environment_root)
        try:
            frame = _reset_env(env)
        except ModuleNotFoundError as exc:
            if exc.name != "arcengine":
                raise
            frame = env.step(0)
        for step_index in range(action_budget):
            before = snapshot_frame(frame)
            if _is_terminal(before.game_state):
                break
            legal = tuple(
                action
                for action in _valid_actions(env)
                if action.name not in {"", "RESET"}
            )
            if not legal:
                break
            selected = _balanced_action(
                legal,
                counts=action_counts,
                game=game,
                seed=seed,
                reset_index=resets,
                raw_step=raw_steps,
            )
            action_names = tuple(
                sorted(set(_available_action_names(legal)))
            )
            observation = build_observation(
                before.grid,
                available_actions=action_names,
                game_state=before.game_state,
                levels_completed=before.levels_completed,
                infer_players=True,
            )
            graph = build_scene_graph(observation)
            frame_after = _step_env_action(env, selected)
            after = snapshot_frame(
                frame_after,
                fallback_available_actions=before.available_actions,
            )
            transition = build_transition_record(
                action=selected.name,
                action_args=dict(selected.action_args),
                grid_before=before.grid,
                grid_after=after.grid,
                available_actions=action_names,
                game_state_before=before.game_state,
                game_state_after=after.game_state,
                levels_completed_before=before.levels_completed,
                levels_completed_after=after.levels_completed,
                infer_players=True,
            )
            observed = tuple(sorted(observed_semantic_effects(transition)))
            repeat_key = _repeat_key(
                graph.signature,
                selected.name,
                selected.action_args,
            )
            repeat_index = repeats[repeat_key]
            repeats[repeat_key] += 1
            raw_steps += 1
            action_counts[selected.name] += 1
            if repeat_index < repeat_cap:
                records.append(
                    ProposalPilotTrace(
                        game_id=game,
                        source_split=split,
                        policy_seed=seed,
                        reset_index=resets,
                        step_index=step_index,
                        scene_graph=graph_to_mapping(graph),
                        available_action_names=action_names,
                        selected_action_name=selected.name,
                        selected_action_data=dict(selected.action_args),
                        observed_effects=observed,
                        changed=not transition.diff.is_noop,
                        noop=transition.diff.is_noop,
                        player_moved=(
                            transition.diff.player_displacement is not None
                        ),
                        level_complete=transition.diff.level_complete,
                        game_over=transition.diff.game_over,
                        productive=bool(
                            not transition.diff.is_noop
                            or transition.diff.level_complete
                        ),
                        repeat_index=repeat_index,
                    )
                )
            frame = frame_after
            if len(records) >= quota or _is_terminal(after.game_state):
                break
        resets += 1
    if len(records) != quota:
        raise RuntimeError(
            f"SAGE12 collection incomplete for {game}: "
            f"{len(records)}/{quota} after {raw_steps} actions"
        )
    _write_jsonl_atomic(shard_path, records)
    return {
        "game_id": game,
        "quota": quota,
        "raw_steps": raw_steps,
        "resets": resets,
        "rejected_repeat_rows": raw_steps - len(records),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _balanced_action(
    legal: Sequence[Any],
    *,
    counts: Mapping[str, int],
    game: str,
    seed: int,
    reset_index: int,
    raw_step: int,
) -> Any:
    by_name: dict[str, list[Any]] = {}
    for action in legal:
        by_name.setdefault(str(action.name), []).append(action)
    minimum = min(int(counts.get(name, 0)) for name in by_name)
    families = sorted(
        name
        for name in by_name
        if int(counts.get(name, 0)) == minimum
    )
    rng = random.Random(
        f"sage12-balanced:{game}:{seed}:{reset_index}:{raw_step}"
    )
    family = families[rng.randrange(len(families))]
    candidates = by_name[family]
    return candidates[rng.randrange(len(candidates))]


def _repeat_key(
    scene_signature: str,
    action_name: str,
    action_data: Mapping[str, Any],
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "scene": scene_signature,
                "action": action_name,
                "arguments": {
                    str(key): _json_safe(value)
                    for key, value in action_data.items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _completed_shard(path: Path, *, game: str, quota: int) -> bool:
    if not path.exists():
        return False
    try:
        shard_metadata(
            path,
            expected_game=game,
            expected_rows=quota,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _write_jsonl_atomic(
    path: Path,
    records: Sequence[ProposalPilotTrace],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect the frozen SAGE12 source-only proposal pilot."
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST_PATH,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    result = run_collection(
        frozen_manifest_path=args.frozen_manifest,
        output_dir=args.out_dir,
        environments_dir=args.environments_dir,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
