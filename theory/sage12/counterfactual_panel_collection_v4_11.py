"""Source-only counterfactual panel collector for SAGE12 V4.11."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir
from theory.real_env_option_adapter import snapshot_frame
from theory.sage11.splits import SOURCE_TRAIN
from theory.unified_cognition_ab_benchmark import _is_terminal, _make_real_env

from .action_target_data import (
    ActionTargetTrace,
    build_action_target_trace,
    grid_sha256,
)
from .bound_mechanic_pilot import (
    ActionSpec,
    _available,
    _candidate_signature,
    _find_action,
    _frame_hash,
    _legal_actions,
    _observation,
    replay_prefix,
)
from .counterfactual_semantic_panels_v4_11 import (
    CONTINUATION_ROLLOUTS,
    DEFAULT_OUTPUT_DIR,
    PROGRESS_HORIZON,
    CounterfactualPanel,
    PanelArm,
    load_manifest,
)
from .semantic_teacher_v4_9 import (
    _checksum,
    _file_sha256,
    _json_safe,
    _read_jsonl,
    _write_json,
)
from .semantic_teacher_v4_9 import (
    load_teacher_records as load_v49_records,
)
from .action_aligned_semantics_v4_10 import (
    load_teacher_records as load_v410_records,
)

COLLECTION_VERSION = "sage12-counterfactual-panel-collection-v4.11"


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _exact_repeat_key(frame: Any, action: Any) -> str:
    snapshot = snapshot_frame(frame)
    payload = {
        "frame_before_sha256": grid_sha256(snapshot.grid),
        "action_name": str(action.name),
        "action_data": dict(action.action_args),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _signature_fields(action: Any, observation: Any) -> dict[str, Any]:
    signature = _candidate_signature(action, observation)
    return {
        "action_name": str(action.name),
        **asdict(signature),
    }


def _hamming(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    keys = set(left) | set(right)
    return sum(left.get(key) != right.get(key) for key in keys)


def select_panel_actions(
    legal: Sequence[Any],
    *,
    observation: Any,
    pre_frame: Any,
    used_repeat_keys: set[str],
    selection_counts: Mapping[str, int],
    maximum_arms: int,
    salt: str,
) -> tuple[Any, ...]:
    """Greedily maximize pre-action semantic diversity; never inspect outcomes."""

    unique = {}
    for action in legal:
        spec = ActionSpec.from_action(action)
        repeat = _exact_repeat_key(pre_frame, action)
        if repeat in used_repeat_keys:
            continue
        unique.setdefault(spec.key, action)
    candidates = []
    for action in unique.values():
        fields = _signature_fields(action, observation)
        stratum = _canonical(fields)
        tie = hashlib.sha256(f"{salt}:{stratum}".encode()).hexdigest()
        candidates.append(
            {
                "action": action,
                "fields": fields,
                "stratum": stratum,
                "count": int(selection_counts.get(stratum, 0)),
                "tie": tie,
            }
        )
    if len(candidates) < 2:
        return ()
    candidates.sort(key=lambda row: (row["count"], row["tie"]))
    selected = [candidates.pop(0)]
    while candidates and len(selected) < maximum_arms:
        def score(row: Mapping[str, Any]) -> tuple[int, int, str]:
            diversity = min(
                _hamming(row["fields"], item["fields"]) for item in selected
            )
            return diversity, -int(row["count"]), str(row["tie"])

        best = max(candidates, key=score)
        selected.append(best)
        candidates.remove(best)
    return tuple(row["action"] for row in selected)


def _execute_trace(
    *,
    env: Any,
    frame: Any,
    action: Any,
    game: str,
    seed: int,
    reset_index: int,
    step_index: int,
    phase: str,
) -> tuple[ActionTargetTrace, Any]:
    legal = _legal_actions(env)
    available = _available(legal)
    before = snapshot_frame(frame)
    selected = _find_action(env, ActionSpec.from_action(action))
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
        collection_phase=phase,
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
    return trace, after_frame


def _continuation_action(
    legal: Sequence[Any],
    *,
    observation: Any,
    salt: str,
) -> Any:
    rows = []
    for action in legal:
        fields = _signature_fields(action, observation)
        tie = hashlib.sha256(
            f"{salt}:{_canonical(fields)}:{ActionSpec.from_action(action).key}".encode()
        ).hexdigest()
        rows.append((tie, ActionSpec.from_action(action).key, action))
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows[0][2]


def _collect_arm(
    *,
    node_env: Any,
    node_frame: Any,
    action: Any,
    arm_index: int,
    panel_index: int,
    game: str,
    seed: int,
    reset_index: int,
    pre_state_sha256: str,
) -> PanelArm:
    arm_env = copy.deepcopy(node_env)
    arm_frame = copy.deepcopy(node_frame)
    immediate, after_frame = _execute_trace(
        env=arm_env,
        frame=arm_frame,
        action=action,
        game=game,
        seed=seed,
        reset_index=reset_index,
        step_index=1_000_000 + panel_index * 10_000 + arm_index * 100,
        phase="v4_11_panel_immediate",
    )
    continuations = []
    for rollout_index in range(CONTINUATION_ROLLOUTS):
        rollout_env = copy.deepcopy(arm_env)
        rollout_frame = copy.deepcopy(after_frame)
        traces = []
        for offset in range(1, PROGRESS_HORIZON):
            before = snapshot_frame(rollout_frame)
            if _is_terminal(before.game_state):
                break
            legal = _legal_actions(rollout_env)
            if not legal:
                break
            observation = _observation(rollout_frame, legal)
            selected = _continuation_action(
                legal,
                observation=observation,
                salt=(
                    f"v4.11:{game}:{seed}:{reset_index}:{panel_index}:"
                    f"{arm_index}:{rollout_index}:{offset}"
                ),
            )
            trace, rollout_frame = _execute_trace(
                env=rollout_env,
                frame=rollout_frame,
                action=selected,
                game=game,
                seed=seed,
                reset_index=reset_index,
                step_index=(
                    1_000_000
                    + panel_index * 10_000
                    + arm_index * 100
                    + rollout_index * 10
                    + offset
                ),
                phase="v4_11_panel_continuation",
            )
            traces.append(trace)
        continuations.append(tuple(traces))
    return PanelArm(
        arm_index=arm_index,
        replay_pre_state_sha256=pre_state_sha256,
        immediate_trace=immediate,
        continuations=tuple(continuations),
    )


def _advance_action(
    legal: Sequence[Any],
    *,
    observation: Any,
    selection_counts: Mapping[str, int],
    salt: str,
) -> Any:
    rows = []
    for action in legal:
        fields = _signature_fields(action, observation)
        stratum = _canonical(fields)
        tie = hashlib.sha256(f"{salt}:{stratum}".encode()).hexdigest()
        rows.append((int(selection_counts.get(stratum, 0)), tie, action))
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows[0][2]


def _collect_game(
    *,
    game: str,
    existing: Sequence[CounterfactualPanel],
    manifest: Mapping[str, Any],
    environment_root: Path,
    historical_repeat_keys: set[str],
) -> tuple[list[CounterfactualPanel], dict[str, Any]]:
    panels = list(existing)
    used_repeat_keys = set(historical_repeat_keys)
    used_repeat_keys.update(
        arm.immediate_trace.exact_repeat_key()
        for panel in panels
        for arm in panel.arms
    )
    collection = manifest["collection"]
    target = int(collection["target_panels_per_game"])
    maximum_arms = int(collection["maximum_arms_per_panel"])
    action_budget = int(collection["action_budget_per_reset"])
    maximum_resets = int(collection["maximum_resets_per_game"])
    seeds = tuple(int(value) for value in collection["policy_seeds"])
    selection_counts: Counter[str] = Counter()
    for panel in panels:
        for arm in panel.arms:
            observation = _observation_from_trace(arm.immediate_trace)
            preview = type(
                "_Preview",
                (),
                {
                    "name": arm.immediate_trace.selected_action_name,
                    "action_args": dict(arm.immediate_trace.selected_action_data),
                },
            )()
            selection_counts[_canonical(_signature_fields(preview, observation))] += 1

    resets_used = 0
    replay_failures = 0
    duplicate_rejections = 0
    candidate_shortfalls = 0
    base_steps = 0
    continuation_steps = 0
    for reset_index in range(maximum_resets):
        if len(panels) >= target:
            break
        seed = seeds[reset_index % len(seeds)]
        env = _make_real_env(game, environment_root)
        try:
            frame = _reset_env(env)
        except ModuleNotFoundError as exc:
            if exc.name != "arcengine":
                raise
            frame = env.step(0)
        reset_template = copy.deepcopy(env)
        reset_frame = copy.deepcopy(frame)
        prefix: list[ActionSpec] = []
        resets_used += 1
        for step_index in range(action_budget):
            if len(panels) >= target:
                break
            before = snapshot_frame(frame)
            if _is_terminal(before.game_state):
                break
            legal = _legal_actions(env)
            if not legal:
                break
            observation = _observation(frame, legal)
            pre_hash = _frame_hash(frame)
            selected = select_panel_actions(
                legal,
                observation=observation,
                pre_frame=frame,
                used_repeat_keys=used_repeat_keys,
                selection_counts=selection_counts,
                maximum_arms=maximum_arms,
                salt=f"{game}:{seed}:{reset_index}:{step_index}:{len(panels)}",
            )
            if len(selected) >= 2:
                try:
                    node_env, node_frame = replay_prefix(
                        reset_template,
                        reset_frame,
                        tuple(prefix),
                        expected_pre_state_sha256=pre_hash,
                    )
                except RuntimeError:
                    replay_failures += 1
                    raise
                arms = tuple(
                    _collect_arm(
                        node_env=node_env,
                        node_frame=node_frame,
                        action=action,
                        arm_index=arm_index,
                        panel_index=len(panels),
                        game=game,
                        seed=seed,
                        reset_index=reset_index,
                        pre_state_sha256=pre_hash,
                    )
                    for arm_index, action in enumerate(selected)
                )
                panel = CounterfactualPanel(
                    game_id=game,
                    policy_seed=seed,
                    reset_index=reset_index,
                    panel_index=len(panels),
                    expected_pre_state_sha256=pre_hash,
                    pre_grid_sha256=grid_sha256(before.grid),
                    arms=arms,
                )
                panels.append(panel)
                for action, arm in zip(selected, arms):
                    repeat = arm.immediate_trace.exact_repeat_key()
                    if repeat in used_repeat_keys:
                        duplicate_rejections += 1
                        raise RuntimeError("V4.11 collector admitted an exact repeat")
                    used_repeat_keys.add(repeat)
                    selection_counts[
                        _canonical(_signature_fields(action, observation))
                    ] += 1
                    continuation_steps += sum(len(row) for row in arm.continuations)
            else:
                candidate_shortfalls += 1

            advance = (
                selected[step_index % len(selected)]
                if selected
                else _advance_action(
                    legal,
                    observation=observation,
                    selection_counts=selection_counts,
                    salt=f"advance:{game}:{seed}:{reset_index}:{step_index}",
                )
            )
            prefix.append(ActionSpec.from_action(advance))
            frame = _step_env_action(env, _find_action(env, ActionSpec.from_action(advance)))
            base_steps += 1
    return panels, {
        "game_id": game,
        "target_panels": target,
        "minimum_panels": int(collection["minimum_panels_per_game"]),
        "panels": len(panels),
        "arms": sum(len(panel.arms) for panel in panels),
        "base_steps": base_steps,
        "continuation_steps": continuation_steps,
        "resets_used": resets_used,
        "replay_failures": replay_failures,
        "duplicate_rejections": duplicate_rejections,
        "candidate_shortfalls": candidate_shortfalls,
    }


def _observation_from_trace(trace: ActionTargetTrace) -> Any:
    from .action_target_data import build_observation

    return build_observation(
        trace.frame_before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
        infer_players=True,
    )


def _load_shard(path: Path) -> list[CounterfactualPanel]:
    if not path.exists():
        return []
    return [CounterfactualPanel.from_dict(row) for row in _read_jsonl(path)]


def _write_shard(path: Path, panels: Sequence[CounterfactualPanel]) -> None:
    from .semantic_teacher_v4_9 import _write_jsonl

    _write_jsonl(path, (panel.to_dict() for panel in panels))


def run_collection(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    environment_root = (
        Path(environments_dir) if environments_dir is not None else _env_dir()
    )
    v49_records = load_v49_records()
    v410_records = load_v410_records()
    historical = {
        record.exact_repeat_key for record in itertools_chain(v49_records, v410_records)
    }
    shard_dir = destination / "source_train_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for game in SOURCE_TRAIN:
        path = shard_dir / f"{game}.jsonl"
        existing = _load_shard(path)
        if len(existing) < int(manifest["collection"]["target_panels_per_game"]):
            panels, report = _collect_game(
                game=game,
                existing=existing,
                manifest=manifest,
                environment_root=environment_root,
                historical_repeat_keys=historical,
            )
            _write_shard(path, panels)
            reports[game] = report
        else:
            reports[game] = {
                "game_id": game,
                "target_panels": int(
                    manifest["collection"]["target_panels_per_game"]
                ),
                "minimum_panels": int(
                    manifest["collection"]["minimum_panels_per_game"]
                ),
                "panels": len(existing),
                "arms": sum(len(panel.arms) for panel in existing),
                "resumed_complete_shard": True,
            }
    shards = []
    all_panels = []
    for game in SOURCE_TRAIN:
        path = shard_dir / f"{game}.jsonl"
        panels = _load_shard(path)
        all_panels.extend(panels)
        shards.append(
            {
                "game_id": game,
                "path": path.as_posix(),
                "panels": len(panels),
                "arms": sum(len(panel.arms) for panel in panels),
                "sha256": _file_sha256(path),
            }
        )
    exact_keys = [
        arm.immediate_trace.exact_repeat_key()
        for panel in all_panels
        for arm in panel.arms
    ]
    minimum = int(manifest["collection"]["minimum_panels_per_game"])
    result: dict[str, Any] = {
        "format_version": COLLECTION_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(SOURCE_TRAIN),
        "panels": len(all_panels),
        "arms": len(exact_keys),
        "reports": reports,
        "shards": shards,
        "checks": {
            "minimum_panels_each_game": all(
                int(row["panels"]) >= minimum for row in shards
            ),
            "all_source_train": all(
                panel.game_id in SOURCE_TRAIN for panel in all_panels
            ),
            "unique_panel_ids": (
                len({panel.panel_id for panel in all_panels}) == len(all_panels)
            ),
            "unique_immediate_repeat_keys": (
                len(set(exact_keys)) == len(exact_keys)
            ),
            "no_v49_v410_exact_repeats": not bool(set(exact_keys) & historical),
            "replay_verified": all(
                all(
                    arm.replay_pre_state_sha256
                    == panel.expected_pre_state_sha256
                    for arm in panel.arms
                )
                for panel in all_panels
            ),
        },
        "offline_environment_opened": True,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
    }
    result["collection_ready"] = all(result["checks"].values())
    result["collection_checksum"] = _checksum(result)
    _write_json(destination / "collection_manifest.json", result)
    return result


def itertools_chain(*rows: Iterable[Any]) -> Iterable[Any]:
    for group in rows:
        yield from group


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--environments-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_collection(
        output_dir=args.output_dir,
        environments_dir=args.environments_dir,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_collection", "select_panel_actions"]
