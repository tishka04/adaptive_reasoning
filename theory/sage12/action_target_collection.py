"""Fresh source-only collector for the SAGE12 action-target V3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION
from theory.unified_cognition_ab_benchmark import (
    _available_action_names,
    _is_terminal,
    _make_real_env,
)

from .action_target_data import (
    EFFECT_LABELS,
    ActionTargetTrace,
    build_action_target_trace,
    build_observation,
    resolve_action_target,
)


COLLECTION_FORMAT_VERSION = "sage12-action-target-collection-v3"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "action_target_pilot_v3"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(payload.get("manifest_checksum", ""))
    check = dict(payload)
    check.pop("manifest_checksum", None)
    actual = _payload_checksum(check)
    if expected != actual:
        raise ValueError(f"V3 frozen-manifest checksum mismatch: {actual} != {expected}")
    if payload.get("format_version") != "sage12-action-target-pilot-v3":
        raise ValueError("unsupported SAGE12 V3 frozen manifest")
    return payload


def run_collection(
    *,
    split: str,
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
    normalized = str(split).strip().lower().replace("-", "_")
    if normalized == "source_train":
        report = _collect_source_train(
            frozen=frozen,
            shard_dir=shard_dir,
            environment_root=environment_root,
        )
        path = destination / "source_train_collection_manifest.json"
    elif normalized == "source_validation":
        report = _collect_source_validation(
            frozen=frozen,
            shard_dir=shard_dir,
            environment_root=environment_root,
        )
        path = destination / "collection_manifest.json"
    else:
        raise ValueError("split must be source_train or source_validation")
    _write_json_atomic(path, report)
    return report


def _collect_source_train(
    *,
    frozen: Mapping[str, Any],
    shard_dir: Path,
    environment_root: Path,
) -> dict[str, Any]:
    base_quotas = {
        str(game): int(quota)
        for game, quota in frozen["collection"]["source_train_base_quotas"].items()
    }
    all_records: dict[str, list[ActionTargetTrace]] = {}
    reports: dict[str, Any] = {}
    for game in SOURCE_TRAIN:
        path = shard_dir / f"{game}.jsonl"
        existing = _load_shard(path)
        if len(existing) < base_quotas[game]:
            existing, game_report = _collect_game(
                game=game,
                split="source_train",
                target_rows=base_quotas[game],
                existing=existing,
                environment_root=environment_root,
                frozen=frozen,
                phase="base",
            )
            _write_jsonl_atomic(path, existing)
            reports[game] = game_report
        all_records[game] = existing

    extra_total = int(frozen["collection"]["source_train_adaptive_rows"])
    allocations = allocate_adaptive_game_quotas(
        all_records,
        extra_total=extra_total,
        maximum_extra_per_game=int(
            frozen["collection"]["maximum_adaptive_rows_per_game"]
        ),
    )
    for game in SOURCE_TRAIN:
        target = base_quotas[game] + allocations.get(game, 0)
        path = shard_dir / f"{game}.jsonl"
        existing = all_records[game]
        if len(existing) < target:
            existing, game_report = _collect_game(
                game=game,
                split="source_train",
                target_rows=target,
                existing=existing,
                environment_root=environment_root,
                frozen=frozen,
                phase="adaptive",
            )
            _write_jsonl_atomic(path, existing)
            reports[game] = game_report
        all_records[game] = existing

    total = sum(len(rows) for rows in all_records.values())
    expected = int(frozen["collection"]["source_train_rows"])
    if total != expected:
        raise RuntimeError(f"V3 source-training row mismatch: {total} != {expected}")
    return _collection_report(
        frozen=frozen,
        split="source_train",
        shard_dir=shard_dir,
        games=SOURCE_TRAIN,
        reports=reports,
        adaptive_allocations=allocations,
    )


def _collect_source_validation(
    *,
    frozen: Mapping[str, Any],
    shard_dir: Path,
    environment_root: Path,
) -> dict[str, Any]:
    quota = int(frozen["collection"]["source_validation_rows_per_game"])
    reports: dict[str, Any] = {}
    for game in SOURCE_VALIDATION:
        path = shard_dir / f"{game}.jsonl"
        existing = _load_shard(path)
        if len(existing) < quota:
            records, report = _collect_game(
                game=game,
                split="source_validation",
                target_rows=quota,
                existing=existing,
                environment_root=environment_root,
                frozen=frozen,
                phase="validation_fixed",
            )
            _write_jsonl_atomic(path, records)
            reports[game] = report
    train_manifest_path = shard_dir.parent / "source_train_collection_manifest.json"
    train_manifest = (
        json.loads(train_manifest_path.read_text(encoding="utf-8"))
        if train_manifest_path.exists()
        else None
    )
    report = _collection_report(
        frozen=frozen,
        split="source_validation",
        shard_dir=shard_dir,
        games=SOURCE_VALIDATION,
        reports=reports,
        adaptive_allocations={},
    )
    report["source_train_manifest_checksum"] = (
        train_manifest.get("report_checksum") if train_manifest else None
    )
    report["total_transitions"] = int(
        report["split_rows"]
        + (train_manifest.get("split_rows", 0) if train_manifest else 0)
    )
    report["source_train_rows"] = (
        int(train_manifest.get("split_rows", 0)) if train_manifest else 0
    )
    report["source_validation_rows"] = int(report["split_rows"])
    report["status"] = (
        "COMPLETE"
        if report["total_transitions"]
        == int(frozen["collection"]["total_rows"])
        else "INCOMPLETE"
    )
    without = dict(report)
    without.pop("report_checksum", None)
    report["report_checksum"] = _payload_checksum(without)
    return report


def allocate_adaptive_game_quotas(
    records_by_game: Mapping[str, Sequence[ActionTargetTrace]],
    *,
    extra_total: int,
    maximum_extra_per_game: int,
) -> dict[str, int]:
    """Allocate the fixed training top-up by preregistered event deficits."""
    games = [
        game
        for game in SOURCE_TRAIN
        if game != "lp85" and game in records_by_game
    ]
    targets = {
        "actor_displaced": 200.0,
        "target_created": 100.0,
        "target_removed": 100.0,
        "target_moved": 100.0,
    }
    global_positive = Counter()
    rates: dict[str, dict[str, float]] = {}
    for game in games:
        rows = records_by_game[game]
        rate: dict[str, float] = {}
        for label in EFFECT_LABELS:
            eligible = [
                row for row in rows if bool(row.effects.applicable[label])
            ]
            positive = sum(bool(row.effects.labels[label]) for row in eligible)
            global_positive[label] += positive
            # Beta(1,1) smoothing makes empty strata explore rather than vanish.
            rate[label] = (positive + 1.0) / (len(eligible) + 2.0)
        rates[game] = rate
    expected_positive = {
        label: float(global_positive[label]) for label in EFFECT_LABELS
    }
    allocations = {game: 0 for game in games}
    for slot in range(int(extra_total)):
        candidates = [
            game
            for game in games
            if allocations[game] < int(maximum_extra_per_game)
        ]
        if not candidates:
            raise RuntimeError("adaptive per-game caps cannot hold requested rows")

        def score(game: str) -> tuple[float, float, str]:
            deficit_score = sum(
                max(0.0, targets[label] - expected_positive[label])
                * rates[game][label]
                for label in EFFECT_LABELS
            )
            balance = -float(allocations[game])
            tie = hashlib.sha256(f"v3-adaptive:{slot}:{game}".encode()).hexdigest()
            return deficit_score, balance, tie

        selected = max(candidates, key=score)
        allocations[selected] += 1
        for label in EFFECT_LABELS:
            expected_positive[label] += rates[selected][label]
    return allocations


def _collect_game(
    *,
    game: str,
    split: str,
    target_rows: int,
    existing: Sequence[ActionTargetTrace],
    environment_root: Path,
    frozen: Mapping[str, Any],
    phase: str,
) -> tuple[list[ActionTargetTrace], dict[str, Any]]:
    records = list(existing)
    repeat_keys = {row.exact_repeat_key() for row in records}
    action_counts = Counter(row.selected_action_name for row in records)
    stratum_counts = Counter(_trace_stratum(row) for row in records)
    yield_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trial_counts = Counter()
    for row in records:
        stratum = _trace_stratum(row)
        trial_counts[stratum] += 1
        for label in EFFECT_LABELS:
            if row.effects.applicable[label] and row.effects.labels[label]:
                yield_counts[stratum][label] += 1

    seeds = tuple(int(value) for value in frozen["collection"]["policy_seeds"])
    action_budget = int(frozen["collection"]["action_budget_per_reset"])
    maximum_resets = int(frozen["collection"]["maximum_resets_per_game"])
    reset_offset = 0
    if records:
        reset_offset = max(row.reset_index for row in records) + 1
    raw_steps = 0
    rejected_duplicates = 0
    resets_used = 0
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
                adaptive=phase == "adaptive",
                exploration_fraction=float(
                    frozen["collection"]["adaptive_exploration_fraction"]
                ),
                salt=f"{game}:{seed}:{reset_index}:{raw_steps}",
            )
            after_frame = _step_env_action(env, selected)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
            trace = build_action_target_trace(
                game_id=game,
                source_split=split,
                policy_seed=seed,
                reset_index=reset_index,
                step_index=step_index,
                collection_phase=phase,
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
            key = trace.exact_repeat_key()
            if key in repeat_keys:
                rejected_duplicates += 1
            else:
                repeat_keys.add(key)
                records.append(trace)
                stratum = _trace_stratum(trace)
                action_counts[trace.selected_action_name] += 1
                stratum_counts[stratum] += 1
                trial_counts[stratum] += 1
                for label in EFFECT_LABELS:
                    if trace.effects.applicable[label] and trace.effects.labels[label]:
                        yield_counts[stratum][label] += 1
            frame = after_frame
            if len(records) >= target_rows or _is_terminal(after.game_state):
                break
        resets_used += 1
    if len(records) != target_rows:
        raise RuntimeError(
            f"SAGE12 V3 collection incomplete for {game}: "
            f"{len(records)}/{target_rows} after {raw_steps} actions"
        )
    return records, {
        "game_id": game,
        "source_split": split,
        "target_rows": target_rows,
        "rows": len(records),
        "new_raw_steps": raw_steps,
        "new_resets": resets_used,
        "rejected_exact_duplicates": rejected_duplicates,
        "action_counts": dict(sorted(action_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
    }


def select_collection_action(
    legal: Sequence[Any],
    *,
    observation: Any,
    action_counts: Mapping[str, int],
    stratum_counts: Mapping[str, int],
    yield_counts: Mapping[str, Mapping[str, int]],
    trial_counts: Mapping[str, int],
    records: Sequence[ActionTargetTrace],
    adaptive: bool,
    exploration_fraction: float,
    salt: str,
) -> Any:
    candidates = []
    for action in legal:
        anchor = resolve_action_target(
            observation, action.name, dict(action.action_args)
        )
        stratum = _anchor_stratum(action.name, anchor)
        candidates.append((action, stratum))
    rng = random.Random(f"sage12-v3:{salt}")
    exploration = (not adaptive) or rng.random() < exploration_fraction
    if exploration:
        minimum_action = min(
            int(action_counts.get(action.name, 0)) for action, _ in candidates
        )
        candidates = [
            item
            for item in candidates
            if int(action_counts.get(item[0].name, 0)) == minimum_action
        ]
        minimum_stratum = min(
            int(stratum_counts.get(stratum, 0)) for _, stratum in candidates
        )
        candidates = [
            item
            for item in candidates
            if int(stratum_counts.get(item[1], 0)) == minimum_stratum
        ]
        return candidates[rng.randrange(len(candidates))][0]

    positives = Counter()
    for row in records:
        for label in EFFECT_LABELS:
            if row.effects.applicable[label] and row.effects.labels[label]:
                positives[label] += 1
    targets = {
        "actor_displaced": 200,
        "target_created": 100,
        "target_removed": 100,
        "target_moved": 100,
    }

    def adaptive_score(item: tuple[Any, str]) -> tuple[float, float, str]:
        _action, stratum = item
        trials = int(trial_counts.get(stratum, 0))
        score = 0.0
        for label in EFFECT_LABELS:
            rate = (
                int(yield_counts.get(stratum, {}).get(label, 0)) + 1.0
            ) / (trials + 2.0)
            score += max(0, targets[label] - positives[label]) * rate
        coverage = -float(stratum_counts.get(stratum, 0))
        tie = hashlib.sha256(f"{salt}:{stratum}".encode()).hexdigest()
        return score, coverage, tie

    return max(candidates, key=adaptive_score)[0]


def _anchor_stratum(action_name: str, anchor: Any) -> str:
    return "|".join(
        (
            str(action_name),
            str(anchor.kind),
            str(anchor.actor_relation),
            str(anchor.path_status),
            str(anchor.target_affordance),
        )
    )


def _trace_stratum(trace: ActionTargetTrace) -> str:
    return _anchor_stratum(trace.selected_action_name, trace.anchor)


def _collection_report(
    *,
    frozen: Mapping[str, Any],
    split: str,
    shard_dir: Path,
    games: Sequence[str],
    reports: Mapping[str, Any],
    adaptive_allocations: Mapping[str, int],
) -> dict[str, Any]:
    shards = []
    for game in games:
        path = shard_dir / f"{game}.jsonl"
        rows = _load_shard(path)
        shards.append(
            {
                "game_id": game,
                "source_split": split,
                "path": str(path).replace("\\", "/"),
                "rows": len(rows),
                "sha256": _file_sha256(path),
                "positive_counts": {
                    label: sum(
                        row.effects.applicable[label]
                        and row.effects.labels[label]
                        for row in rows
                    )
                    for label in EFFECT_LABELS
                },
                "applicable_counts": {
                    label: sum(row.effects.applicable[label] for row in rows)
                    for label in EFFECT_LABELS
                },
            }
        )
    combined = hashlib.sha256(
        "".join(item["sha256"] for item in shards).encode("ascii")
    ).hexdigest()
    payload: dict[str, Any] = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "status": "SOURCE_TRAIN_COMPLETE"
        if split == "source_train"
        else "SOURCE_VALIDATION_COMPLETE",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "split": split,
        "split_rows": sum(item["rows"] for item in shards),
        "games": list(games),
        "shards": shards,
        "combined_shard_checksum": combined,
        "adaptive_allocations": dict(sorted(adaptive_allocations.items())),
        "game_reports": dict(reports),
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
    }
    payload["report_checksum"] = _payload_checksum(payload)
    return payload


def _load_shard(path: Path) -> list[ActionTargetTrace]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(ActionTargetTrace.from_dict(json.loads(line)))
    return records


def _write_jsonl_atomic(path: Path, records: Sequence[ActionTargetTrace]) -> None:
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
    parser.add_argument("split", choices=("source_train", "source_validation"))
    parser.add_argument(
        "--frozen-manifest",
        default=str(DEFAULT_FROZEN_MANIFEST_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir")
    args = parser.parse_args(argv)
    payload = run_collection(
        split=args.split,
        frozen_manifest_path=args.frozen_manifest,
        output_dir=args.output_dir,
        environments_dir=args.environments_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTION_FORMAT_VERSION",
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "allocate_adaptive_game_quotas",
    "load_frozen_manifest",
    "run_collection",
    "select_collection_action",
]
