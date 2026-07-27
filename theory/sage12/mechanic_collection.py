"""Prospective chronological collector for the SAGE12 V4 mechanic pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.sage11.splits import SOURCE_VALIDATION
from theory.unified_cognition_ab_benchmark import (
    _available_action_names,
    _is_terminal,
    _make_real_env,
)

from .action_target_collection import select_collection_action
from .action_target_data import (
    EFFECT_LABELS,
    ActionTargetTrace,
    build_action_target_trace,
    build_observation,
)
from .mechanic_induction import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    load_frozen_manifest,
)


COLLECTION_FORMAT_VERSION = "sage12-mechanic-collection-v4"


def run_collection(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    shard_dir = destination / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    environment_root = (
        Path(environments_dir) if environments_dir is not None else _env_dir()
    )
    quota = int(frozen["collection"]["prospective_rows_per_game"])
    reports = {}
    for game in SOURCE_VALIDATION:
        path = shard_dir / f"{game}.jsonl"
        existing = _load_shard(path)
        if len(existing) < quota:
            records, report = _collect_game(
                game=game,
                target_rows=quota,
                existing=existing,
                environment_root=environment_root,
                frozen=frozen,
            )
            _write_jsonl_atomic(path, records)
            reports[game] = report
    payload = _collection_report(
        frozen=frozen,
        shard_dir=shard_dir,
        reports=reports,
    )
    _write_json_atomic(destination / "collection_manifest.json", payload)
    return payload


def _collect_game(
    *,
    game: str,
    target_rows: int,
    existing: Sequence[ActionTargetTrace],
    environment_root: Path,
    frozen: Mapping[str, Any],
) -> tuple[list[ActionTargetTrace], dict[str, Any]]:
    records = list(existing)
    action_counts = Counter(row.selected_action_name for row in records)
    stratum_counts = Counter(_stratum(row) for row in records)
    yield_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trial_counts = Counter()
    seeds = tuple(int(value) for value in frozen["collection"]["policy_seeds"])
    action_budget = int(frozen["collection"]["action_budget_per_reset"])
    maximum_resets = int(frozen["collection"]["maximum_resets_per_game"])
    reset_offset = (
        max((row.reset_index for row in records), default=-1) + 1
    )
    raw_steps = 0
    resets_used = 0
    chronological_repeats = 0
    seen_repeat_keys = {row.exact_repeat_key() for row in records}
    while len(records) < target_rows and resets_used < maximum_resets:
        reset_index = reset_offset + resets_used
        seed = seeds[reset_index % len(seeds)]
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
            available = tuple(sorted(set(_available_action_names(legal))))
            observation = build_observation(
                before.grid,
                available_actions=available,
                game_state=before.game_state,
                levels_completed=before.levels_completed,
                infer_players=True,
            )
            selected = select_collection_action(
                legal,
                observation=observation,
                action_counts=action_counts,
                stratum_counts=stratum_counts,
                yield_counts=yield_counts,
                trial_counts=trial_counts,
                records=records,
                adaptive=False,
                exploration_fraction=1.0,
                salt=f"v4:{game}:{seed}:{reset_index}:{step_index}",
            )
            after_frame = _step_env_action(env, selected)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
            trace = build_action_target_trace(
                game_id=game,
                source_split="source_validation",
                policy_seed=seed,
                reset_index=reset_index,
                step_index=step_index,
                collection_phase="v4_prospective_fixed",
                available_action_names=available,
                selected_action_name=selected.name,
                selected_action_data=dict(selected.action_args),
                frame_before=before.grid,
                frame_after=after.grid,
                game_state_before=before.game_state,
                game_state_after=after.game_state,
                levels_completed_before=before.levels_completed,
                levels_completed_after=after.levels_completed,
            )
            raw_steps += 1
            repeat_key = trace.exact_repeat_key()
            chronological_repeats += int(repeat_key in seen_repeat_keys)
            seen_repeat_keys.add(repeat_key)
            records.append(trace)
            action_counts[trace.selected_action_name] += 1
            stratum_counts[_stratum(trace)] += 1
            for label in EFFECT_LABELS:
                if trace.effects.applicable[label]:
                    trial_counts[_stratum(trace)] += 1
                    if trace.effects.labels[label]:
                        yield_counts[_stratum(trace)][label] += 1
            frame = after_frame
            if len(records) >= target_rows or _is_terminal(after.game_state):
                break
        resets_used += 1
    if len(records) != target_rows:
        raise RuntimeError(
            f"SAGE12 V4 collection incomplete for {game}: "
            f"{len(records)}/{target_rows}"
        )
    return records, {
        "game_id": game,
        "rows": len(records),
        "target_rows": target_rows,
        "new_raw_steps": raw_steps,
        "new_resets": resets_used,
        "chronological_exact_repeats_retained": chronological_repeats,
        "action_counts": dict(sorted(action_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
    }


def _stratum(trace: ActionTargetTrace) -> str:
    return "|".join(
        (
            trace.selected_action_name,
            trace.anchor.action_family,
            trace.anchor.kind,
            trace.anchor.path_status,
        )
    )


def _collection_report(
    *,
    frozen: Mapping[str, Any],
    shard_dir: Path,
    reports: Mapping[str, Any],
) -> dict[str, Any]:
    shards = []
    for game in SOURCE_VALIDATION:
        path = shard_dir / f"{game}.jsonl"
        rows = _load_shard(path)
        shards.append(
            {
                "game_id": game,
                "path": path.as_posix(),
                "rows": len(rows),
                "sha256": _file_sha256(path),
                "chronological_repeat_rows": len(rows)
                - len({row.exact_repeat_key() for row in rows}),
                "positive_counts": {
                    label: sum(
                        int(row.effects.applicable[label])
                        * int(row.effects.labels[label])
                        for row in rows
                    )
                    for label in EFFECT_LABELS
                },
            }
        )
    combined = hashlib.sha256(
        "".join(item["sha256"] for item in shards).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "status": "COMPLETE",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "split": "prospective_source_validation",
        "rows": sum(item["rows"] for item in shards),
        "games": list(SOURCE_VALIDATION),
        "shards": shards,
        "combined_shard_checksum": combined,
        "game_reports": dict(reports),
        "outcome_adaptive": False,
        "chronological_repeats_retained": True,
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
    }
    payload["report_checksum"] = _payload_checksum(payload)
    return payload


def _load_shard(path: Path) -> list[ActionTargetTrace]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(ActionTargetTrace.from_dict(json.loads(line)))
    return rows


def _write_jsonl_atomic(
    path: Path, records: Sequence[ActionTargetTrace]
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
    _replace_with_retry(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(12):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.1 * 2**attempt, 2.0))
    if last_error is not None:
        raise last_error


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-manifest", default=str(DEFAULT_FROZEN_MANIFEST_PATH)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir")
    args = parser.parse_args(argv)
    result = run_collection(
        frozen_manifest_path=args.frozen_manifest,
        output_dir=args.output_dir,
        environments_dir=args.environments_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["COLLECTION_FORMAT_VERSION", "run_collection"]
