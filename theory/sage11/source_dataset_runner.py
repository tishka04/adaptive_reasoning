"""Resumable real-environment SAGE.10g-i/SAGE.11 source collection.

The default run collects each of the 14 source games under the amended base
cap plus one 1,292-row source-training overflow pool, stops finite-state games
after a pre-registered duplicate-saturation window, then publishes a
deterministic 100,000-row corpus. Validation rows remain in separate shards
and are firewalled from training consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from theory.online_transferable_causal_schema import (
    FrozenCausalSchemaLibrary,
    merge_frozen_causal_schema_libraries,
)
from theory.real_env_option_adapter import snapshot_frame
from theory.unified_cognition_ab_benchmark import (
    SharedLegacyProposalPolicy,
    _available_action_names,
    _is_terminal,
    _make_real_env,
    _materialize_decision,
)
from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions

from ..unified_cognitive_controller import UnifiedCognitiveController
from .curriculum import FrozenSchemaCurriculum
from .dataset import (
    DEFAULT_PER_GAME_CAP,
    DEFAULT_TARGET_TRANSITIONS,
    DatasetManifest,
    DatasetShard,
    NeuroTransition,
    Sage11ControllerCollector,
    Sage11DatasetBuilder,
    verify_manifest,
)
from .splits import (
    ArtifactPurpose,
    SAGE11_SPLITS,
    SOURCE_TRAIN,
    SOURCE_VALIDATION,
    short_game_id,
)


DEFAULT_OUTPUT_DIR = Path("training") / "sage11" / "source_dataset_v2"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_RUN_REPORT_PATH = DEFAULT_OUTPUT_DIR / "collection_report.json"
DEFAULT_CAPACITY_REPORT_PATH = DEFAULT_OUTPUT_DIR / "capacity_report.json"
DEFAULT_CURRICULUM_PATH = (
    Path("training")
    / "sage11"
    / "curriculum"
    / "frozen_schema_curriculum.json"
)
SOURCE_TRAIN_QUOTA = DEFAULT_PER_GAME_CAP
SOURCE_VALIDATION_QUOTA = DEFAULT_PER_GAME_CAP
COLLECTION_FORMAT_VERSION = "sage11-source-collection-v2"
DEFAULT_DUPLICATE_SATURATION_PATIENCE = 4_000
DEFAULT_SEED_WINDOW_RESETS = 200
DEFAULT_OVERFLOW_TRANSITIONS = 1_292
DEFAULT_OVERFLOW_GAMES = ("cd82", "dc22", "g50t", "ka59", "tr87")

ControllerFactory = Callable[
    [str, Sage11ControllerCollector],
    UnifiedCognitiveController,
]
EnvFactory = Callable[[str], Any]


def run_source_dataset_collection(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    curriculum_path: str | Path = DEFAULT_CURRICULUM_PATH,
    environments_dir: str | Path | None = None,
    game_ids: Sequence[str] | None = None,
    source_train_quota: int = SOURCE_TRAIN_QUOTA,
    source_validation_quota: int = SOURCE_VALIDATION_QUOTA,
    seed: int | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    action_budget_per_reset: int = 400,
    max_raw_multiplier: int = 50,
    checkpoint_every_resets: int = 10,
    env_factory: EnvFactory | None = None,
    controller_factory: ControllerFactory | None = None,
    require_full_curriculum: bool = True,
    workers: int = 1,
    target_transitions: int | None = None,
    duplicate_saturation_patience: int = (
        DEFAULT_DUPLICATE_SATURATION_PATIENCE
    ),
    overflow_transitions: int = DEFAULT_OVERFLOW_TRANSITIONS,
    overflow_games: Sequence[str] = DEFAULT_OVERFLOW_GAMES,
) -> Dict[str, Any]:
    """Collect or resume checksummed per-game shards from real environments."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shard_dir = destination / "shards"
    work_shard_dir = destination / "work_shards"
    schema_dir = destination / "source_schemas"
    shard_dir.mkdir(parents=True, exist_ok=True)
    work_shard_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)

    train_quota = max(1, int(source_train_quota))
    validation_quota = max(1, int(source_validation_quota))
    seed_values = (
        (int(seed),)
        if seed is not None
        else tuple(int(item) for item in seeds)
    )
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("SAGE.11 collection seeds must be unique")
    selected = tuple(
        short_game_id(game)
        for game in (
            game_ids
            if game_ids is not None
            else (*SOURCE_TRAIN, *SOURCE_VALIDATION)
        )
    )
    if len(selected) != len(set(selected)):
        raise ValueError("SAGE.11 source collection games must be unique")
    SAGE11_SPLITS.assert_authorized(
        selected,
        purpose=ArtifactPurpose.VALIDATE_SOURCE,
    )
    if require_full_curriculum and set(SOURCE_TRAIN).difference(selected):
        missing = sorted(set(SOURCE_TRAIN).difference(selected))
        raise ValueError(
            "full SAGE.10g curriculum requires all source games: "
            + ", ".join(missing)
        )

    full_protocol = set(selected) == set(
        (*SOURCE_TRAIN, *SOURCE_VALIDATION)
    )
    overflow_pool = (
        max(0, int(overflow_transitions))
        if full_protocol
        else 0
    )
    overflow_allocation = _overflow_allocation(
        overflow_pool,
        games=overflow_games,
        selected=selected,
    )
    quotas = {
        game: (
            train_quota
            if SAGE11_SPLITS.split_for(game) == "source_train"
            else validation_quota
        )
        + overflow_allocation.get(game, 0)
        for game in selected
    }
    requested_target = (
        int(target_transitions)
        if target_transitions is not None
        else (
            DEFAULT_TARGET_TRANSITIONS
            if full_protocol
            else sum(quotas.values())
        )
    )
    if requested_target <= 0 or requested_target % 10:
        raise ValueError(
            "SAGE.11 target transitions must be a positive multiple of 10"
        )
    environment_root = (
        Path(environments_dir)
        if environments_dir is not None
        else _env_dir()
    )
    game_reports: Dict[str, Any] = {}
    shards: list[DatasetShard] = []
    source_libraries: Dict[str, FrozenCausalSchemaLibrary] = {}
    pending: Dict[str, Dict[str, Any]] = {}
    verified_capacity = 0
    partial_publication_ready = True
    for game in selected:
        quota = quotas[game]
        shard_path = work_shard_dir / f"{game}.jsonl"
        metadata_path = work_shard_dir / f"{game}.checkpoint.json"
        schema_path = schema_dir / f"{game}.json"
        usable = _completed_game_checkpoint(
            metadata_path,
            shard_path=shard_path,
            expected_transitions=quota,
            expected_seeds=seed_values,
            expected_schema_path=(
                schema_path if game in SOURCE_TRAIN else None
            ),
            allow_in_progress=True,
        )
        if not usable:
            partial_publication_ready = False
            break
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        verified_capacity += int(payload["accepted_transitions"])
    partial_publication_ready = bool(
        partial_publication_ready
        and verified_capacity >= requested_target
    )
    for game in selected:
        quota = quotas[game]
        shard_path = work_shard_dir / f"{game}.jsonl"
        metadata_path = work_shard_dir / f"{game}.checkpoint.json"
        schema_path = schema_dir / f"{game}.json"
        resumed = _completed_game_checkpoint(
            metadata_path,
            shard_path=shard_path,
            expected_transitions=quota,
            expected_seeds=seed_values,
            expected_schema_path=(
                schema_path if game in SOURCE_TRAIN else None
            ),
            allow_in_progress=partial_publication_ready,
        )
        if resumed:
            report = json.loads(metadata_path.read_text(encoding="utf-8"))
            if report.get("status") == "IN_PROGRESS":
                report["checkpoint_status"] = "IN_PROGRESS"
                report["status"] = "TARGET_CAPACITY_REACHED"
            report["resumed_completed_game"] = True
            game_reports[game] = report
        else:
            pending[game] = {
                "game_id": game,
                "quota": quota,
                "shard_path": shard_path,
                "metadata_path": metadata_path,
                "schema_path": (
                    schema_path
                    if game in SOURCE_TRAIN
                    else None
                ),
                "environment_root": environment_root,
                "seeds": seed_values,
                "action_budget_per_reset": max(
                    1, int(action_budget_per_reset)
                ),
                "max_raw_multiplier": max(1, int(max_raw_multiplier)),
                "checkpoint_every_resets": max(
                    1, int(checkpoint_every_resets)
                ),
                "duplicate_saturation_patience": max(
                    1, int(duplicate_saturation_patience)
                ),
                "env_factory": env_factory,
                "controller_factory": controller_factory,
            }

    worker_count = max(1, int(workers))
    if worker_count > 1 and (env_factory is not None or controller_factory is not None):
        raise ValueError(
            "parallel collection does not accept injected factories"
        )
    if worker_count > 1 and len(pending) > 1:
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(pending))
        ) as executor:
            futures = {
                executor.submit(_collect_game_job, arguments): game
                for game, arguments in pending.items()
            }
            for future in as_completed(futures):
                game = futures[future]
                report = future.result()
                report["resumed_completed_game"] = False
                game_reports[game] = report
    else:
        for game, arguments in pending.items():
            report = _collect_game(**arguments)
            report["resumed_completed_game"] = False
            game_reports[game] = report

    for game in selected:
        report = game_reports[game]
        schema_path = schema_dir / f"{game}.json"
        if game in SOURCE_TRAIN:
            source_libraries[game] = _load_source_schema(
                schema_path,
                expected_checksum=str(report["schema_checksum"]),
            )

    allocation = _publication_allocation(
        game_reports,
        target_transitions=requested_target,
    )
    for game in selected:
        work_path = work_shard_dir / f"{game}.jsonl"
        published_path = shard_dir / f"{game}.jsonl"
        shard = _publish_selected_shard(
            work_path,
            published_path,
            transitions=allocation.get(game, 0),
        )
        if shard.transitions:
            shards.append(shard)
        game_reports[game]["published_transitions"] = shard.transitions

    builder = Sage11DatasetBuilder(
        purpose=ArtifactPurpose.VALIDATE_SOURCE,
        target_transitions=requested_target,
        per_game_cap=DEFAULT_PER_GAME_CAP,
        game_caps=quotas,
    )
    for shard in shards:
        builder.load_jsonl_shard(shard.path)
    manifest = builder.manifest(shards)
    verify_manifest(manifest)
    if set(selected) == set((*SOURCE_TRAIN, *SOURCE_VALIDATION)):
        if manifest.total_transitions != DEFAULT_TARGET_TRANSITIONS:
            raise ValueError(
                "default SAGE.11 source corpus must contain exactly "
                f"{DEFAULT_TARGET_TRANSITIONS} transitions"
            )
        if (
            int(manifest.to_dict()["overflow_transitions"])
            != overflow_pool
        ):
            raise ValueError(
                "default SAGE.11 source corpus must use exactly the approved "
                f"{overflow_pool}-row overflow pool"
            )

    manifest_payload = manifest.to_dict()
    manifest_payload["manifest_checksum"] = manifest.checksum
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest_payload)
    verify_source_dataset(manifest_path)

    curriculum_payload: Dict[str, Any] | None = None
    if set(SOURCE_TRAIN).issubset(source_libraries):
        curriculum = FrozenSchemaCurriculum.build(
            {
                game: source_libraries[game]
                for game in SOURCE_TRAIN
            }
        )
        curriculum_payload = {
            **curriculum.to_dict(),
            "library": curriculum.library.to_dict(),
        }
        _write_json(Path(curriculum_path), curriculum_payload)
        verify_frozen_curriculum(Path(curriculum_path))

    report_payload = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "status": "COMPLETE",
        "seeds": list(seed_values),
        "games": list(selected),
        "quotas": dict(sorted(quotas.items())),
        "publication_allocation": dict(sorted(allocation.items())),
        "overflow_transitions": overflow_pool,
        "overflow_allocation": dict(sorted(overflow_allocation.items())),
        "target_transitions": requested_target,
        "action_budget_per_reset": int(action_budget_per_reset),
        "duplicate_saturation_patience": int(
            duplicate_saturation_patience
        ),
        "workers_requested": worker_count,
        "workers": min(worker_count, max(1, len(pending))),
        "resume_scope": "verified_game_checkpoint",
        "game_reports": game_reports,
        "manifest_path": _repo_path(manifest_path),
        "manifest_checksum": manifest.checksum,
        "curriculum_path": (
            _repo_path(Path(curriculum_path))
            if curriculum_payload is not None
            else None
        ),
        "curriculum_checksum": (
            curriculum_payload["checksum"]
            if curriculum_payload is not None
            else None
        ),
    }
    _write_json(destination / "collection_report.json", report_payload)
    return {
        "manifest": manifest_payload,
        "curriculum": curriculum_payload,
        "report": report_payload,
    }


def _collect_game_job(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Pickle-safe worker entrypoint for one independent source game."""
    return _collect_game(**dict(arguments))


def _collect_game(
    *,
    game_id: str,
    quota: int,
    shard_path: Path,
    metadata_path: Path,
    schema_path: Path | None,
    environment_root: Path,
    seeds: Sequence[int],
    action_budget_per_reset: int,
    max_raw_multiplier: int,
    checkpoint_every_resets: int,
    duplicate_saturation_patience: int,
    env_factory: EnvFactory | None,
    controller_factory: ControllerFactory | None,
) -> Dict[str, Any]:
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("SAGE.11 collection requires at least one seed")
    purpose = (
        ArtifactPurpose.TRAIN
        if game_id in SOURCE_TRAIN
        else ArtifactPurpose.VALIDATE_SOURCE
    )
    builder = Sage11DatasetBuilder(
        purpose=purpose,
        target_transitions=quota,
        per_game_cap=quota,
    )
    partial = _partial_game_checkpoint(
        metadata_path,
        shard_path=shard_path,
        expected_transitions=quota,
    )
    resumed_partial_game = partial is not None
    if partial is not None:
        builder.load_jsonl_shard(shard_path)
        raw_steps = int(partial["raw_transitions"])
        resets = int(partial["resets"])
        decision_sources: Counter[str] = Counter(
            {
                str(source): int(count)
                for source, count in dict(
                    partial.get("decision_sources", {}) or {}
                ).items()
            }
        )
        attempted_seeds = [
            int(seed)
            for seed in tuple(partial.get("seeds_attempted", ()) or ())
            if int(seed) in seed_values
        ]
        duplicate_streaks = {
            int(seed): int(streak)
            for seed, streak in dict(
                partial.get("trailing_duplicate_streaks", {}) or {}
            ).items()
            if int(seed) in seed_values
        }
        if not duplicate_streaks and attempted_seeds:
            duplicate_streaks[attempted_seeds[-1]] = int(
                partial.get("trailing_duplicate_streak", 0)
            )
        saturated_seeds = {
            int(seed)
            for seed in tuple(partial.get("saturated_seeds", ()) or ())
            if int(seed) in seed_values
        }
        if "next_seed_position" in partial:
            seed_position = int(partial["next_seed_position"]) % len(
                seed_values
            )
        elif attempted_seeds:
            seed_position = (
                seed_values.index(attempted_seeds[-1]) + 1
            ) % len(seed_values)
        else:
            seed_position = 0
        seed_window_resets_used = int(
            partial.get("seed_window_resets_used", 0)
        )
    else:
        raw_steps = 0
        resets = 0
        decision_sources = Counter()
        attempted_seeds = []
        duplicate_streaks: Dict[int, int] = {}
        saturated_seeds: set[int] = set()
        seed_position = 0
        seed_window_resets_used = 0
    current_seed = seed_values[seed_position]
    def new_controller(
        run_seed: int,
    ) -> tuple[Sage11ControllerCollector, UnifiedCognitiveController]:
        run_collector = Sage11ControllerCollector(
            builder,
            game_id=game_id,
            seed=run_seed,
        )
        run_controller = (
            controller_factory(game_id, run_collector)
            if controller_factory is not None
            else UnifiedCognitiveController(
                game_id,
                neuro_transition_collector=run_collector,
            )
        )
        return run_collector, run_controller

    collector, controller = new_controller(current_seed)
    seed_libraries: list[FrozenCausalSchemaLibrary] = []
    if (
        partial is not None
        and schema_path is not None
        and schema_path.exists()
        and str(partial.get("schema_checksum", ""))
    ):
        restored_library = _load_source_schema(
            schema_path,
            expected_checksum=str(partial["schema_checksum"]),
        )
        seed_libraries.append(restored_library)
        seed_position = (seed_position + 1) % len(seed_values)
        seed_window_resets_used = 0
        current_seed = seed_values[seed_position]
        collector, controller = new_controller(current_seed)
    mixture = builder.mixture_policy
    max_raw_steps = quota * max_raw_multiplier * len(seed_values)
    max_resets = max(100, quota * 2) * len(seed_values)

    while len(builder.records) < quota and raw_steps < max_raw_steps:
        if resets >= max_resets or len(saturated_seeds) == len(seed_values):
            break
        if (
            current_seed in saturated_seeds
            or seed_window_resets_used >= DEFAULT_SEED_WINDOW_RESETS
        ):
            if schema_path is not None:
                seed_libraries.append(
                    controller.freeze_transferable_causal_schemas()
                )
            seed_window_resets_used = 0
            for _ in range(len(seed_values)):
                seed_position = (seed_position + 1) % len(seed_values)
                current_seed = seed_values[seed_position]
                if current_seed not in saturated_seeds:
                    break
            collector, controller = new_controller(current_seed)
        else:
            current_seed = seed_values[seed_position]
        if current_seed not in attempted_seeds:
            attempted_seeds.append(current_seed)
        collector.seed = current_seed
        controller.on_reset()
        active_policy = SharedLegacyProposalPolicy(
            game_id=game_id,
            seed=current_seed,
            reset_index=resets,
        )
        probe_policy = SharedLegacyProposalPolicy(
            game_id=game_id,
            seed=current_seed + 1_000_003,
            reset_index=resets,
        )
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, environment_root)
        )
        frame = _reset_collection_env(env)
        resets += 1
        seed_window_resets_used += 1
        for _ in range(action_budget_per_reset):
            before = snapshot_frame(frame)
            if _is_terminal(before.game_state):
                break
            legal_actions = tuple(_valid_actions(env))
            legal_actions = tuple(
                item
                for item in legal_actions
                if str(getattr(item, "name", "")) not in {"", "RESET"}
            )
            if not legal_actions:
                break

            accepted_index = len(builder.records)
            arm = mixture.arm_for(
                game_id=game_id,
                # The accepted-row schedule must remain one exact 70/20/10
                # sequence even when exploration rotates across run seeds.
                seed=seed_values[0],
                reset_index=0,
                step_index=accepted_index,
            )
            collector.set_policy_arm(arm)
            if arm == "uniform_legal":
                selected = _uniform_action(
                    legal_actions,
                    game_id=game_id,
                    seed=current_seed,
                    raw_step=raw_steps,
                )
                source = "uniform_legal"
            else:
                proposal_policy = (
                    active_policy
                    if arm == "active_controller"
                    else probe_policy
                )
                proposal = proposal_policy.select(legal_actions)
                if proposal is None:
                    break
                decision = controller.select_action(
                    current_grid=before.grid,
                    available_actions=_available_action_names(legal_actions),
                    legacy_action=str(getattr(proposal, "name", "")),
                    legacy_action_data=dict(
                        getattr(proposal, "action_args", {}) or {}
                    ),
                    available_action_candidates=legal_actions,
                    game_state=before.game_state,
                    levels_completed=before.levels_completed,
                )
                materialized = _materialize_decision(
                    legal_actions,
                    decision,
                )
                selected = materialized or proposal
                source = str(decision.source)
            decision_sources[source] += 1

            frame_after = _step_env_action(env, selected)
            after = snapshot_frame(
                frame_after,
                fallback_available_actions=before.available_actions,
            )
            raw_steps += 1
            previous_accepted = len(builder.records)
            controller.observe_transition(
                action=str(getattr(selected, "name", "")),
                action_data=dict(
                    getattr(selected, "action_args", {}) or {}
                ),
                grid_before=before.grid,
                grid_after=after.grid,
                available_actions=_available_action_names(legal_actions),
                game_state_before=before.game_state,
                game_state_after=after.game_state,
                levels_completed_before=before.levels_completed,
                levels_completed_after=after.levels_completed,
            )
            if (
                int(after.levels_completed) > int(before.levels_completed)
                and not _is_terminal(after.game_state)
            ):
                controller.on_level_change()
            frame = frame_after
            if len(builder.records) == previous_accepted:
                # The next deterministic mixture slot is intentionally
                # retried until a distinct behavioral transition is accepted.
                duplicate_streaks[current_seed] = (
                    duplicate_streaks.get(current_seed, 0) + 1
                )
                if (
                    duplicate_streaks[current_seed]
                    >= duplicate_saturation_patience
                ):
                    saturated_seeds.add(current_seed)
                    break
            else:
                duplicate_streaks[current_seed] = 0
            if len(builder.records) >= quota or _is_terminal(after.game_state):
                break

        if resets % checkpoint_every_resets == 0:
            partial_shard = builder.write_jsonl_shard(shard_path)
            partial_schema_checksum = ""
            if schema_path is not None:
                partial_library = merge_frozen_causal_schema_libraries(
                    [
                        *seed_libraries,
                        controller.freeze_transferable_causal_schemas(),
                    ],
                    allowed_source_tags=(game_id,),
                    max_schemas=64,
                )
                _write_json(schema_path, partial_library.to_dict())
                partial_schema_checksum = partial_library.content_checksum
            _write_json(
                metadata_path,
                _game_checkpoint_payload(
                    game_id=game_id,
                    status="IN_PROGRESS",
                    quota=quota,
                    raw_steps=raw_steps,
                    resets=resets,
                    shard=partial_shard,
                    builder=builder,
                    decision_sources=decision_sources,
                    schema_path=schema_path,
                    schema_checksum=partial_schema_checksum,
                    duplicate_streaks=duplicate_streaks,
                    saturated_seeds=sorted(saturated_seeds),
                    attempted_seeds=attempted_seeds,
                    next_seed_position=seed_position,
                    seed_window_resets_used=seed_window_resets_used,
                    resumed_partial_game=resumed_partial_game,
                ),
            )

    saturated = len(saturated_seeds) == len(seed_values)
    if len(builder.records) != quota and not saturated:
        partial_shard = builder.write_jsonl_shard(shard_path)
        payload = _game_checkpoint_payload(
            game_id=game_id,
            status="INCOMPLETE",
            quota=quota,
            raw_steps=raw_steps,
            resets=resets,
            shard=partial_shard,
            builder=builder,
            decision_sources=decision_sources,
            duplicate_streaks=duplicate_streaks,
            saturated_seeds=sorted(saturated_seeds),
            attempted_seeds=attempted_seeds,
            next_seed_position=seed_position,
            seed_window_resets_used=seed_window_resets_used,
            resumed_partial_game=resumed_partial_game,
        )
        _write_json(metadata_path, payload)
        raise RuntimeError(
            f"SAGE.11 collection incomplete for {game_id}: "
            f"{len(builder.records)}/{quota} accepted after "
            f"{raw_steps} raw transitions"
        )

    shard = builder.write_jsonl_shard(shard_path)
    schema_checksum = ""
    if schema_path is not None:
        seed_libraries.append(
            controller.freeze_transferable_causal_schemas()
        )
        library = merge_frozen_causal_schema_libraries(
            seed_libraries,
            allowed_source_tags=(game_id,),
            max_schemas=64,
        )
        _write_json(schema_path, library.to_dict())
        schema_checksum = library.content_checksum
    payload = _game_checkpoint_payload(
        game_id=game_id,
        status=("COMPLETE" if len(builder.records) == quota else "SATURATED"),
        quota=quota,
        raw_steps=raw_steps,
        resets=resets,
        shard=shard,
        builder=builder,
        decision_sources=decision_sources,
        schema_path=schema_path,
        schema_checksum=schema_checksum,
        duplicate_streaks=duplicate_streaks,
        saturated_seeds=sorted(saturated_seeds),
        attempted_seeds=attempted_seeds,
        next_seed_position=seed_position,
        seed_window_resets_used=seed_window_resets_used,
        resumed_partial_game=resumed_partial_game,
    )
    _write_json(metadata_path, payload)
    return payload


def _uniform_action(
    legal_actions: Sequence[Any],
    *,
    game_id: str,
    seed: int,
    raw_step: int,
) -> Any:
    rng = random.Random(
        f"sage11-uniform:{game_id}:{int(seed)}:{int(raw_step)}"
    )
    return legal_actions[rng.randrange(len(legal_actions))]


def _reset_collection_env(env: Any) -> Any:
    """Reset a real Arcade env, with an integer fallback for injected tests."""
    try:
        return _reset_env(env)
    except ModuleNotFoundError as exc:
        if exc.name != "arcengine":
            raise
        return env.step(0)


def _game_checkpoint_payload(
    *,
    game_id: str,
    status: str,
    quota: int,
    raw_steps: int,
    resets: int,
    shard: DatasetShard,
    builder: Sage11DatasetBuilder,
    decision_sources: Mapping[str, int],
    schema_path: Path | None = None,
    schema_checksum: str = "",
    duplicate_streaks: Mapping[int, int] | None = None,
    saturated_seeds: Sequence[int] = (),
    attempted_seeds: Sequence[int] = (),
    next_seed_position: int = 0,
    seed_window_resets_used: int = 0,
    resumed_partial_game: bool = False,
) -> Dict[str, Any]:
    streaks = {
        int(seed): int(streak)
        for seed, streak in dict(duplicate_streaks or {}).items()
    }
    return {
        "format_version": COLLECTION_FORMAT_VERSION,
        "game_id": game_id,
        "source_split": SAGE11_SPLITS.split_for(game_id),
        "status": status,
        "quota": int(quota),
        "accepted_transitions": int(shard.transitions),
        "raw_transitions": int(raw_steps),
        "resets": int(resets),
        "shard_path": _repo_path(Path(shard.path)),
        "shard_sha256": shard.sha256,
        "policy_counts": dict(
            sorted(builder.manifest().policy_counts.items())
        ),
        "decision_sources": dict(sorted(decision_sources.items())),
        "rejected_duplicates": max(
            int(builder.summary()["rejected_duplicates"]),
            int(raw_steps) - int(shard.transitions),
        ),
        "trailing_duplicate_streak": max(streaks.values(), default=0),
        "trailing_duplicate_streaks": {
            str(seed): streak for seed, streak in sorted(streaks.items())
        },
        "saturated_seeds": [int(seed) for seed in saturated_seeds],
        "seeds_attempted": [int(seed) for seed in attempted_seeds],
        "next_seed_position": int(next_seed_position),
        "seed_window_resets": DEFAULT_SEED_WINDOW_RESETS,
        "seed_window_resets_used": int(seed_window_resets_used),
        "resumed_partial_game": bool(resumed_partial_game),
        "schema_path": (
            _repo_path(schema_path) if schema_path is not None else None
        ),
        "schema_checksum": schema_checksum,
    }


def _partial_game_checkpoint(
    metadata_path: Path,
    *,
    shard_path: Path,
    expected_transitions: int,
) -> Dict[str, Any] | None:
    """Return a verified resumable checkpoint without trusting stale rows."""
    if not metadata_path.exists() or not shard_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != COLLECTION_FORMAT_VERSION:
        return None
    if payload.get("status") != "IN_PROGRESS":
        return None
    accepted = int(payload.get("accepted_transitions", -1))
    if int(payload.get("quota", -1)) != int(expected_transitions):
        return None
    if not 0 < accepted < int(expected_transitions):
        return None
    if hashlib.sha256(shard_path.read_bytes()).hexdigest() != str(
        payload.get("shard_sha256", "")
    ):
        return None
    return payload


def _completed_game_checkpoint(
    metadata_path: Path,
    *,
    shard_path: Path,
    expected_transitions: int,
    expected_seeds: Sequence[int],
    expected_schema_path: Path | None = None,
    allow_in_progress: bool = False,
) -> bool:
    if not metadata_path.exists() or not shard_path.exists():
        return False
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != COLLECTION_FORMAT_VERSION:
        return False
    allowed_statuses = {"COMPLETE", "SATURATED"}
    if allow_in_progress:
        allowed_statuses.add("IN_PROGRESS")
    if payload.get("status") not in allowed_statuses:
        return False
    if int(payload.get("quota", -1)) != int(expected_transitions):
        return False
    accepted = int(payload.get("accepted_transitions", -1))
    if payload.get("status") == "COMPLETE" and accepted != int(
        expected_transitions
    ):
        return False
    if payload.get("status") == "SATURATED" and not (
        0 <= accepted < int(expected_transitions)
    ):
        return False
    if payload.get("status") == "IN_PROGRESS" and not (
        0 < accepted < int(expected_transitions)
    ):
        return False
    if payload.get("status") == "SATURATED" and tuple(
        int(seed) for seed in payload.get("seeds_attempted", ())
    ) != tuple(int(seed) for seed in expected_seeds):
        return False
    if hashlib.sha256(shard_path.read_bytes()).hexdigest() != str(
        payload.get("shard_sha256", "")
    ):
        return False
    if expected_schema_path is not None:
        if not expected_schema_path.exists():
            return False
        try:
            library = FrozenCausalSchemaLibrary.from_dict(
                json.loads(expected_schema_path.read_text(encoding="utf-8"))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if library.content_checksum != str(
            payload.get("schema_checksum", "")
        ):
            return False
    return True


def _publication_allocation(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    target_transitions: int,
) -> Dict[str, int]:
    """Prefer all train capacity, then balance validation row by row."""
    available = {
        game: int(report["accepted_transitions"])
        for game, report in reports.items()
    }
    if sum(available.values()) < target_transitions:
        raise RuntimeError(
            "SAGE.11 exact-dedup source capacity is below target: "
            f"{sum(available.values())}/{target_transitions}"
        )
    allocation = {game: 0 for game in reports}
    remaining = int(target_transitions)
    for group in (SOURCE_TRAIN, SOURCE_VALIDATION):
        group_games = [
            game
            for game in group
            if game in available and available[game] > 0
        ]
        while remaining > 0 and group_games:
            progressed = False
            for game in group_games:
                if remaining <= 0:
                    break
                if allocation[game] >= available[game]:
                    continue
                allocation[game] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break
    if remaining:
        raise RuntimeError(
            "SAGE.11 source allocation cannot satisfy target after split "
            f"balancing; {remaining} rows remain"
        )
    return allocation


def _publish_selected_shard(
    work_path: Path,
    published_path: Path,
    *,
    transitions: int,
) -> DatasetShard:
    builder = Sage11DatasetBuilder(
        purpose=ArtifactPurpose.VALIDATE_SOURCE,
        target_transitions=max(1, int(transitions)),
        per_game_cap=max(DEFAULT_PER_GAME_CAP, int(transitions)),
    )
    if transitions:
        with work_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= transitions:
                    break
                if not line.strip():
                    continue
                if not builder.add(
                    NeuroTransition.from_dict(json.loads(line))
                ):
                    raise ValueError(
                        f"rejected publication row {work_path}:{index + 1}"
                    )
    return builder.write_jsonl_shard(published_path)


def _overflow_allocation(
    transitions: int,
    *,
    games: Sequence[str],
    selected: Sequence[str],
) -> Dict[str, int]:
    normalized = tuple(short_game_id(game) for game in games)
    if len(normalized) != len(set(normalized)):
        raise ValueError("SAGE.11 overflow games must be unique")
    eligible = tuple(game for game in normalized if game in selected)
    if transitions and not eligible:
        raise ValueError("SAGE.11 overflow pool has no eligible games")
    SAGE11_SPLITS.assert_authorized(
        eligible,
        purpose=ArtifactPurpose.TRAIN,
    )
    allocation: Counter[str] = Counter()
    for index in range(max(0, int(transitions))):
        allocation[eligible[index % len(eligible)]] += 1
    return dict(allocation)


def _load_source_schema(
    path: Path,
    *,
    expected_checksum: str,
) -> FrozenCausalSchemaLibrary:
    library = FrozenCausalSchemaLibrary.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if library.content_checksum != expected_checksum:
        raise ValueError(f"SAGE.10g source-schema checksum mismatch: {path}")
    return library


def verify_source_dataset(manifest_path: str | Path) -> DatasetManifest:
    """Reload and verify the published manifest and all referenced shards."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.from_dict(payload)
    expected_checksum = str(payload.get("manifest_checksum", ""))
    if manifest.checksum != expected_checksum:
        raise ValueError("SAGE.11 manifest checksum mismatch")
    verify_manifest(manifest)
    return manifest


def verify_frozen_curriculum(path: str | Path) -> FrozenSchemaCurriculum:
    """Verify the merged curriculum checksum and source-only provenance."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    library = FrozenCausalSchemaLibrary.from_dict(
        dict(payload.get("library", {}) or {})
    )
    source_checksums = dict(payload.get("source_checksums", {}) or {})
    if set(source_checksums) != set(SOURCE_TRAIN):
        raise ValueError("SAGE.10g curriculum source set is incomplete")
    if set(library.source_tags).difference(SOURCE_TRAIN):
        raise ValueError("SAGE.10g curriculum contains non-source provenance")
    curriculum = FrozenSchemaCurriculum(
        library=library,
        source_checksums=source_checksums,
        split_registry_checksum=str(
            payload.get("split_registry_checksum", "")
        ),
        format_version=str(payload.get("format_version", "")),
    )
    observed = curriculum.to_dict()
    if observed["checksum"] != payload.get("checksum"):
        raise ValueError("SAGE.10g curriculum checksum mismatch")
    return curriculum


def write_capacity_report(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    target_transitions: int = DEFAULT_TARGET_TRANSITIONS,
) -> Dict[str, Any]:
    """Publish a fail-closed upper bound from completed game checkpoints."""
    destination = Path(output_dir)
    checkpoint_dir = destination / "work_shards"
    reports = {
        str(payload["game_id"]): payload
        for path in sorted(checkpoint_dir.glob("*.checkpoint.json"))
        for payload in (
            json.loads(path.read_text(encoding="utf-8")),
        )
    }
    expected_games = set((*SOURCE_TRAIN, *SOURCE_VALIDATION))
    if set(reports) != expected_games:
        missing = sorted(expected_games.difference(reports))
        raise ValueError(
            "capacity report requires a checkpoint for every source game: "
            + ", ".join(missing)
        )
    saturated = {
        game: int(report["accepted_transitions"])
        for game, report in reports.items()
        if report.get("status") == "SATURATED"
    }
    optimistic_counts = {
        game: (
            saturated[game]
            if game in saturated
            else min(
                DEFAULT_PER_GAME_CAP,
                int(report.get("quota", DEFAULT_PER_GAME_CAP)),
            )
        )
        for game, report in reports.items()
    }
    optimistic_upper_bound = sum(optimistic_counts.values())
    payload = {
        "format_version": "sage11-source-capacity-v1",
        "status": (
            "BLOCKED_CAPACITY"
            if optimistic_upper_bound < int(target_transitions)
            else "CAPACITY_NOT_DISPROVED"
        ),
        "target_transitions": int(target_transitions),
        "per_game_cap": DEFAULT_PER_GAME_CAP,
        "duplicate_saturation_patience": (
            DEFAULT_DUPLICATE_SATURATION_PATIENCE
        ),
        "seeds": [0, 1, 2, 3, 4],
        "saturated_game_counts": dict(sorted(saturated.items())),
        "optimistic_game_counts": dict(sorted(optimistic_counts.items())),
        "optimistic_upper_bound": optimistic_upper_bound,
        "minimum_shortfall": max(
            0,
            int(target_transitions) - optimistic_upper_bound,
        ),
        "checkpointed_accepted_transitions": sum(
            int(report["accepted_transitions"])
            for report in reports.values()
        ),
        "checkpoint_statuses": {
            game: {
                "status": str(report["status"]),
                "accepted_transitions": int(
                    report["accepted_transitions"]
                ),
                "raw_transitions": int(report["raw_transitions"]),
                "seeds_attempted": list(
                    report.get("seeds_attempted", ())
                ),
                "shard_sha256": str(report["shard_sha256"]),
            }
            for game, report in sorted(reports.items())
        },
        "manifest_published": False,
        "holdout_or_historical_games_touched": False,
        "legacy_weights_loaded": False,
    }
    payload["report_checksum"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(destination / "capacity_report.json", payload)
    return payload


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the real 100k-transition SAGE.11 source corpus and "
            "freeze the SAGE.10g-i multi-game curriculum."
        )
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--curriculum-out",
        default=str(DEFAULT_CURRICULUM_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--games", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--train-quota",
        type=int,
        default=SOURCE_TRAIN_QUOTA,
    )
    parser.add_argument(
        "--validation-quota",
        type=int,
        default=SOURCE_VALIDATION_QUOTA,
    )
    parser.add_argument("--actions-per-reset", type=int, default=400)
    parser.add_argument("--max-raw-multiplier", type=int, default=50)
    parser.add_argument("--checkpoint-every-resets", type=int, default=10)
    parser.add_argument(
        "--duplicate-saturation-patience",
        type=int,
        default=DEFAULT_DUPLICATE_SATURATION_PATIENCE,
    )
    parser.add_argument(
        "--overflow-transitions",
        type=int,
        default=DEFAULT_OVERFLOW_TRANSITIONS,
    )
    parser.add_argument(
        "--overflow-games",
        default=",".join(DEFAULT_OVERFLOW_GAMES),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    games = (
        None
        if args.games is None
        else tuple(
            item.strip()
            for item in str(args.games).split(",")
            if item.strip()
        )
    )
    result = run_source_dataset_collection(
        output_dir=args.out_dir,
        curriculum_path=args.curriculum_out,
        environments_dir=args.environments_dir,
        game_ids=games,
        source_train_quota=args.train_quota,
        source_validation_quota=args.validation_quota,
        seeds=tuple(
            int(item.strip())
            for item in str(args.seeds).split(",")
            if item.strip()
        ),
        action_budget_per_reset=args.actions_per_reset,
        max_raw_multiplier=args.max_raw_multiplier,
        checkpoint_every_resets=args.checkpoint_every_resets,
        duplicate_saturation_patience=(
            args.duplicate_saturation_patience
        ),
        overflow_transitions=args.overflow_transitions,
        overflow_games=tuple(
            item.strip()
            for item in str(args.overflow_games).split(",")
            if item.strip()
        ),
        require_full_curriculum=games is None,
        workers=args.workers,
    )
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTION_FORMAT_VERSION",
    "DEFAULT_CAPACITY_REPORT_PATH",
    "DEFAULT_OVERFLOW_GAMES",
    "DEFAULT_OVERFLOW_TRANSITIONS",
    "DEFAULT_CURRICULUM_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "SOURCE_TRAIN_QUOTA",
    "SOURCE_VALIDATION_QUOTA",
    "run_source_dataset_collection",
    "verify_frozen_curriculum",
    "verify_source_dataset",
    "write_capacity_report",
]
