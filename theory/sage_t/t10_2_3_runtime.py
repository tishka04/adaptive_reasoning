"""Exact donor-cache continuation runtime for SAGE.T10.2.3.

The cache is built parent-side before the T10.2.2 reset watchdog is armed.  It
contains the exact ``GaugeProgramPosterior`` state produced by the frozen
T10.2 algorithm over the exact ordered donor events.  Partial builds alternate
between two authenticated pickle slots, so interruption can lose at most one
checkpoint interval and can never make a partial state authoritative.
"""

from __future__ import annotations

import argparse
import copyreg
import hashlib
import json
import math
import os
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_2_runtime as _parent_runtime
from . import t10_2_3_protocol as _protocol
from . import t10_2_runtime as _science

FORMAT_VERSION = "sage-t10.2.3-runtime-v1"
CACHE_FORMAT_VERSION = "sage-t10.2.3-exact-donor-cache-v1"
CONTINUATION_REPORT_FORMAT_VERSION = "sage-t10.2.3-continuation-report-v1"
CONTINUATION_REPORT_FILENAME = "t10_2_3_continuation_report.json"

ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
WorkerProtocolError = _kernel_runtime.WorkerProtocolError
canonical_sha256 = _kernel_protocol.canonical_sha256
canonical_json = _kernel_protocol.canonical_json
signed_payload = _kernel_protocol.signed_payload


def _restore_mapping_proxy(value: Mapping[str, Any]) -> MappingProxyType:
    return MappingProxyType(dict(value))


def _reduce_mapping_proxy(value: MappingProxyType) -> tuple[Any, tuple[Any, ...]]:
    return _restore_mapping_proxy, (dict(value),)


copyreg.pickle(type(MappingProxyType({})), _reduce_mapping_proxy)


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _event_binding(
    donor_events: Sequence[Mapping[str, Any]], training_games: Sequence[str]
) -> dict[str, Any]:
    events = [dict(event) for event in donor_events]
    return {
        "training_games": list(training_games),
        "event_count": len(events),
        "ordered_event_ids_sha256": canonical_sha256(
            [str(event.get("event_id", "")) for event in events]
        ),
        "ordered_event_checksums_sha256": canonical_sha256(
            [str(event.get("event_checksum", "")) for event in events]
        ),
        "ordered_events_sha256": canonical_sha256(events),
    }


def _donor_events_from_hydrated_factory(
    factory: Any, training_games: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(event)
        for game in training_games
        for event in factory._discovery_events[game]
    )


def _donor_events_from_collection(
    discovery_events: Sequence[Mapping[str, Any]], training_games: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[dict[str, Any]]] = {
        game: [] for game in _kernel_protocol.SOURCE_GAMES
    }
    for raw in discovery_events:
        event = dict(raw)
        game = str(event.get("game_id", ""))
        if game not in grouped or event.get("split") != "discovery":
            raise JournalIntegrityError("donor cache received non-discovery evidence")
        grouped[game].append(event)
    return tuple(dict(event) for game in training_games for event in grouped[game])


class ExactDonorPosteriorCache:
    """Authenticated two-slot checkpoint store for one exact donor fold."""

    def __init__(
        self,
        *,
        root: str | Path,
        continuation_manifest_checksum: str,
        parent_kernel_manifest_checksum: str,
        checkpoint_interval: int = 8,
        maximum_bytes: int = 536_870_912,
    ) -> None:
        self.root = Path(root).resolve()
        self.continuation_manifest_checksum = str(continuation_manifest_checksum)
        self.parent_kernel_manifest_checksum = str(parent_kernel_manifest_checksum)
        self.checkpoint_interval = int(checkpoint_interval)
        self.maximum_bytes = int(maximum_bytes)
        if self.checkpoint_interval <= 0 or self.maximum_bytes <= 0:
            raise ValueError("cache bounds must be positive")

    def binding(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
    ) -> dict[str, Any]:
        event_binding = _event_binding(donor_events, training_games)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "continuation_manifest_checksum": self.continuation_manifest_checksum,
            "parent_kernel_manifest_checksum": self.parent_kernel_manifest_checksum,
            "runtime_source_sha256": _kernel_protocol.canonical_file_sha256(
                Path(__file__)
            ),
            "candidate_limit": 256,
            "posterior_factory": "GaugeProgramPosterior",
            "donors": event_binding,
        }
        return {**payload, "cache_key": canonical_sha256(payload)}

    def _directory(self, binding: Mapping[str, Any]) -> Path:
        return self.root / str(binding["cache_key"])

    def _metadata_path(self, binding: Mapping[str, Any]) -> Path:
        return self._directory(binding) / "metadata.json"

    def _read_metadata(self, binding: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._metadata_path(binding)
        if not path.is_file():
            return None
        metadata = _kernel_protocol._read_signed_json(
            path, checksum_key="cache_checksum"
        )
        if metadata.get("format_version") != CACHE_FORMAT_VERSION:
            raise ManifestDriftError("donor cache format drifted")
        if metadata.get("binding") != dict(binding):
            raise ManifestDriftError("donor cache binding drifted")
        return metadata

    def _load_state(
        self, binding: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        metadata = self._read_metadata(binding)
        if metadata is None:
            return None
        directory = self._directory(binding)
        state_name = str(metadata.get("state_file", ""))
        if state_name not in {"state-a.pkl", "state-b.pkl"}:
            raise ManifestDriftError("donor cache state slot drifted")
        state_path = directory / state_name
        if not state_path.is_file() or state_path.is_symlink():
            raise ManifestDriftError("donor cache state is missing")
        raw = state_path.read_bytes()
        if len(raw) != int(metadata.get("state_bytes", -1)):
            raise ManifestDriftError("donor cache state size drifted")
        if _raw_sha256(raw) != metadata.get("state_sha256"):
            raise ManifestDriftError("donor cache state checksum drifted")
        try:
            state = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 - fail closed on local cache bytes.
            raise ManifestDriftError("donor cache state is unreadable") from exc
        if not isinstance(state, dict) or state.get("binding") != dict(binding):
            raise ManifestDriftError("donor cache payload binding drifted")
        if bool(state.get("finalized")) != bool(metadata.get("finalized")):
            raise ManifestDriftError("donor cache finalization flag drifted")
        if int(state.get("next_event_index", -1)) != int(
            metadata.get("next_event_index", -2)
        ):
            raise ManifestDriftError("donor cache progress drifted")
        return state, metadata

    def _write_state(
        self, *, binding: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        directory = self._directory(binding)
        directory.mkdir(parents=True, exist_ok=True)
        previous = self._read_metadata(binding)
        previous_slot = None if previous is None else str(previous["state_file"])
        slot = "state-b.pkl" if previous_slot == "state-a.pkl" else "state-a.pkl"
        raw = pickle.dumps(dict(state), protocol=5)
        if len(raw) > self.maximum_bytes:
            raise JournalIntegrityError("donor cache exceeded its registered byte cap")
        target = directory / slot
        temporary = directory / f".{slot}.{os.getpid()}.tmp"
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _kernel_runtime._fsync_directory(directory)
        metadata = signed_payload(
            {
                "format_version": CACHE_FORMAT_VERSION,
                "binding": dict(binding),
                "state_file": slot,
                "state_sha256": _raw_sha256(raw),
                "state_bytes": len(raw),
                "next_event_index": int(state["next_event_index"]),
                "observation_count": int(state["observations"]),
                "error_count": len(state["errors"]),
                "candidate_count": len(state["candidates"]),
                "candidate_hashes_sha256": canonical_sha256(
                    [candidate.canonical_hash for candidate in state["candidates"]]
                ),
                "finalized": bool(state["finalized"]),
            },
            checksum_key="cache_checksum",
        )
        _kernel_runtime._atomic_write_json(self._metadata_path(binding), metadata)
        if previous_slot and previous_slot != slot:
            stale = directory / previous_slot
            if stale.is_file() and not stale.is_symlink():
                stale.unlink()
                _kernel_runtime._fsync_directory(directory)
        return metadata

    def ensure(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        events = tuple(dict(event) for event in donor_events)
        binding = self.binding(donor_events=events, training_games=training_games)
        self.root.mkdir(parents=True, exist_ok=True)
        lease = _kernel_runtime._CollectionLease.acquire(
            self.root / f".{binding['cache_key']}.lock"
        )
        try:
            loaded = self._load_state(binding)
            if loaded is None:
                candidates = tuple(
                    _science._synthesize_gauge_candidates(events, maximum=256)
                )
                if not candidates:
                    state: dict[str, Any] = {
                        "binding": dict(binding),
                        "posterior": None,
                        "candidates": (),
                        "next_event_index": len(events),
                        "observations": 0,
                        "errors": ["no_executed_sequence_candidates"],
                        "branch": None,
                        "finalized": True,
                    }
                    metadata = self._write_state(binding=binding, state=state)
                    return metadata
                posterior = _science.GaugeProgramPosterior()
                posterior.seed(candidates)
                state = {
                    "binding": dict(binding),
                    "posterior": posterior,
                    "candidates": candidates,
                    "next_event_index": 0,
                    "observations": 0,
                    "errors": [],
                    "branch": None,
                    "finalized": False,
                }
                metadata = self._write_state(binding=binding, state=state)
            else:
                state, metadata = loaded
                if state["finalized"]:
                    return metadata

            total = len(events)
            while int(state["next_event_index"]) < total:
                index = int(state["next_event_index"])
                event = events[index]
                try:
                    bundle = _science._bundle_from_compact_event(event)
                    if bundle is not None:
                        selection = event.get("selection", {})
                        current_branch = (
                            str(event.get("game_id", "")),
                            int(event.get("seed", -1)),
                            int(
                                selection.get("reset_index", -1)
                                if isinstance(selection, Mapping)
                                else -1
                            ),
                        )
                        if (
                            state["branch"] is not None
                            and current_branch != state["branch"]
                        ):
                            start_branch = getattr(
                                state["posterior"], "start_branch", None
                            )
                            if callable(start_branch):
                                start_branch()
                        state["branch"] = current_branch
                        state["posterior"].observe(bundle)
                        state["observations"] = int(state["observations"]) + 1
                except Exception as exc:  # noqa: BLE001 - exact parent semantics.
                    state["errors"].append(type(exc).__name__)
                state["next_event_index"] = index + 1
                if progress is not None:
                    progress(
                        {
                            "phase": "build_donor_cache",
                            "cache_key": binding["cache_key"],
                            "events_processed": index + 1,
                            "events_total": total,
                            "checkpointed": (index + 1) % self.checkpoint_interval == 0,
                        }
                    )
                if (index + 1) % self.checkpoint_interval == 0:
                    metadata = self._write_state(binding=binding, state=state)

            if state["observations"]:
                start_branch = getattr(state["posterior"], "start_branch", None)
                if callable(start_branch):
                    start_branch()
            state["finalized"] = True
            metadata = self._write_state(binding=binding, state=state)
            return metadata
        finally:
            lease.release()

    def load_final(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
    ) -> tuple[Any | None, tuple[Any, ...], int, tuple[str, ...], dict[str, Any]]:
        binding = self.binding(
            donor_events=donor_events, training_games=training_games
        )
        loaded = self._load_state(binding)
        if loaded is None or not loaded[0].get("finalized"):
            raise ManifestDriftError("exact donor cache is absent or incomplete")
        state, metadata = loaded
        if int(state["next_event_index"]) != len(donor_events):
            raise ManifestDriftError("exact donor cache does not cover all donors")
        return (
            state["posterior"],
            tuple(state["candidates"]),
            int(state["observations"]),
            tuple(str(item) for item in state["errors"]),
            metadata,
        )


class CachedDonorSourceFactory(_parent_runtime.ActionBudgetSourceFactory):
    """T10.2.2 worker with an exact, manifest-bound donor cache."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        cache_root: str | Path,
        continuation_manifest_checksum: str,
        watchdog: Any = None,
        runtime_loader: Callable[[], Any] | None = None,
        bundle_builder: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            manifest=manifest,
            watchdog=watchdog,
            runtime_loader=runtime_loader,
            bundle_builder=bundle_builder,
        )
        self.cache_root = str(Path(cache_root).resolve())
        self.continuation_manifest_checksum = str(continuation_manifest_checksum)

    def _cache(self) -> ExactDonorPosteriorCache:
        policy = _protocol.continuation_policy()
        return ExactDonorPosteriorCache(
            root=self.cache_root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            parent_kernel_manifest_checksum=str(self.manifest_checksum),
            checkpoint_interval=int(policy["cache_checkpoint_interval_events"]),
            maximum_bytes=int(policy["cache_maximum_bytes"]),
        )

    def clone_for_worker(self) -> "CachedDonorSourceFactory":
        return CachedDonorSourceFactory(
            manifest=self.manifest,
            cache_root=self.cache_root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            runtime_loader=self._runtime_loader,
            bundle_builder=self._bundle_builder,
        )

    def _donor_posterior_scores(
        self, training_games: Sequence[str]
    ) -> tuple[Any | None, tuple[Any, ...], tuple[Mapping[str, Any], ...], dict[Any, float], dict[str, Any]]:
        donor_events = _donor_events_from_hydrated_factory(self, training_games)
        posterior, candidates, observations, errors, cache_metadata = (
            self._cache().load_final(
                donor_events=donor_events, training_games=training_games
            )
        )
        scores = {} if posterior is None else _science._posterior_action_scores(posterior)
        metadata = {
            "posterior_used": posterior is not None,
            "posterior_candidates": len(candidates),
            "posterior_classes": len(
                {candidate.gauge_equivalence_key for candidate in candidates}
            ),
            "posterior_observations": observations,
            "posterior_errors": len(errors),
            "option_conditioned": bool(scores),
            "exact_donor_cache_used": True,
            "exact_donor_cache_checksum": cache_metadata["cache_checksum"],
        }
        return posterior, candidates, donor_events, scores, metadata

    def run_reset(self, **kwargs: Any) -> Any:
        work = kwargs["work"]
        if work.lane.split != "leave_one_game_out_confirmation":
            return super().run_reset(**kwargs)
        training_games = tuple(
            game for game in _kernel_protocol.SOURCE_GAMES if game != work.lane.game_id
        )
        donor_events = _donor_events_from_collection(
            kwargs["discovery_events"], training_games
        )
        started = float(kwargs.get("clock", time.perf_counter)())

        def emit(payload: Mapping[str, Any]) -> None:
            print(canonical_json(payload), flush=True)

        self._cache().ensure(
            donor_events=donor_events,
            training_games=training_games,
            progress=emit,
        )
        elapsed = max(0.0, float(kwargs.get("clock", time.perf_counter)()) - started)
        adjusted = dict(kwargs)
        for name in (
            "lane_remaining_seconds",
            "cooperative_collection_remaining_seconds",
            "absolute_collection_remaining_seconds",
        ):
            adjusted[name] = float(adjusted[name]) - elapsed
        print(
            canonical_json(
                {
                    "phase": "donor_cache_ready",
                    "held_out_game": work.lane.game_id,
                    "donor_event_count": len(donor_events),
                    "elapsed_seconds": elapsed,
                    "reset_watchdog_armed": False,
                }
            ),
            flush=True,
        )
        return super().run_reset(**adjusted)


@contextmanager
def continuation_factory_binding(manifest: Mapping[str, Any]):
    """Bind the new adapter while retaining the legacy audit vocabulary."""

    original = _kernel_protocol._factory_binding

    def bind(factory: Any, kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
        if type(factory) is not CachedDonorSourceFactory:
            return original(factory, kernel_manifest)
        relative = "theory/sage_t/t10_2_3_runtime.py"
        observed = _kernel_protocol.canonical_file_sha256(Path(__file__))
        continuation_bound = bool(
            observed == manifest.get("portable_code_sha256", {}).get(relative)
            and factory.continuation_manifest_checksum
            == manifest.get("manifest_checksum")
            and factory.manifest_checksum == kernel_manifest.get("manifest_checksum")
        )
        return {
            "module": "theory.sage_t.t10_2_1_runtime",
            "class": "T10_2_1SourceFactory",
            "source_sha256": kernel_manifest.get("portable_code_sha256", {}).get(
                "theory/sage_t/t10_2_1_runtime.py", ""
            ),
            "manifest_checksum": str(factory.manifest_checksum),
            "code_bound": continuation_bound,
            "continuation_factory": {
                "module": type(factory).__module__,
                "class": type(factory).__name__,
                "source_sha256": observed,
                "continuation_manifest_checksum": (
                    factory.continuation_manifest_checksum
                ),
                "code_bound": continuation_bound,
            },
        }

    _kernel_protocol._factory_binding = bind
    try:
        yield
    finally:
        _kernel_protocol._factory_binding = original


def _load_execution_context(
    *, manifest_path: str | Path, repo_root: str | Path | None
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    root = Path(repo_root or _kernel_protocol._repo_root()).resolve()
    manifest = _protocol.load_manifest(
        manifest_path, repo_root=root, verify_repository=True,
        verify_live_migration=True,
    )
    parent = _parent_protocol.load_manifest(
        root / _parent_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=True,
    )
    kernel, kernel_path, artifact_root = _parent_protocol.load_kernel_manifest(
        manifest=parent, mode="full", repo_root=root
    )
    return root, manifest, parent, kernel, kernel_path, artifact_root


def _cache_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not cache_root.is_dir():
        return entries
    for path in sorted(cache_root.glob("*/metadata.json")):
        metadata = _kernel_protocol._read_signed_json(
            path, checksum_key="cache_checksum"
        )
        if metadata.get("finalized") is True:
            entries.append(
                {
                    "cache_key": metadata["binding"]["cache_key"],
                    "cache_checksum": metadata["cache_checksum"],
                    "donors": metadata["binding"]["donors"],
                    "state_bytes": metadata["state_bytes"],
                }
            )
    return entries


def prepare_next_cache(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build only the next registered fold cache; perform no physical action."""

    root, manifest, _, kernel, kernel_path, artifact_root = _load_execution_context(
        manifest_path=manifest_path, repo_root=repo_root
    )
    receipt = manifest["migration_receipt"]
    next_lane = _kernel_runtime.SourceLaneKey.from_dict(receipt["next_lane"])
    if next_lane.split != "leave_one_game_out_confirmation":
        raise ManifestDriftError("next registered lane does not require a donor cache")
    relative_kernel = kernel_path.relative_to(root)
    with (
        _parent_protocol.kernel_protocol_bindings(
            artifact_root=artifact_root,
            manifest_relative_path=relative_kernel,
            mode="full",
        ),
        _parent_runtime.execution_bindings(mode="full", artifact_root=artifact_root),
    ):
        journal = _parent_runtime.IncrementalDurableCollectionJournal(
            root / artifact_root / _kernel_runtime.JOURNAL_DIRECTORY_NAME,
            manifest_checksum=kernel["manifest_checksum"],
        )
        training_games = tuple(
            game for game in _kernel_protocol.SOURCE_GAMES if game != next_lane.game_id
        )
        donor_events = _donor_events_from_collection(
            journal.completed_discovery_events(), training_games
        )
    cache = ExactDonorPosteriorCache(
        root=root / _protocol.DEFAULT_CACHE_ROOT,
        continuation_manifest_checksum=manifest["manifest_checksum"],
        parent_kernel_manifest_checksum=kernel["manifest_checksum"],
    )
    metadata = cache.ensure(
        donor_events=donor_events,
        training_games=training_games,
        progress=lambda payload: print(canonical_json(payload), flush=True),
    )
    return {
        "status": "READY_T10_2_3_CONTINUATION",
        "next_lane": next_lane.to_dict(),
        "cache_checksum": metadata["cache_checksum"],
        "donor_event_count": len(donor_events),
        "physical_actions_issued": 0,
    }


def collect_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    recorder: _parent_runtime.CollectionTimingRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
    **kwargs: Any,
) -> dict[str, Any]:
    """Continue the exact T10.2.2 journal under the corrected orchestration."""

    root, manifest, parent, kernel, kernel_path, artifact_root = (
        _load_execution_context(manifest_path=manifest_path, repo_root=repo_root)
    )
    destination = (root / artifact_root).resolve()
    if output_dir is not None:
        candidate = Path(output_dir)
        candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if candidate != destination:
            raise ManifestDriftError("continuation output escaped the parent journal")
    recorder = recorder or _parent_runtime.CollectionTimingRecorder()
    factory = CachedDonorSourceFactory(
        manifest=kernel,
        cache_root=root / _protocol.DEFAULT_CACHE_ROOT,
        continuation_manifest_checksum=manifest["manifest_checksum"],
        watchdog=_parent_runtime.ActionBudgetWatchdog(recorder=recorder, clock=clock),
    )
    relative_kernel = kernel_path.relative_to(root)
    with (
        _parent_protocol.kernel_protocol_bindings(
            artifact_root=artifact_root,
            manifest_relative_path=relative_kernel,
            mode="full",
        ),
        _parent_runtime.execution_bindings(mode="full", artifact_root=artifact_root),
        continuation_factory_binding(manifest),
    ):
        verified = _kernel_protocol.load_manifest(kernel_path, repo_root=root)
        if verified != kernel:
            raise ManifestDriftError("verified execution kernel escaped T10.2.3")
        _kernel_runtime.collect_phase(
            manifest_path=kernel_path,
            output_dir=artifact_root,
            repo_root=root,
            env_factory=factory,
            clock=clock,
            **kwargs,
        )
    parent_report = _parent_runtime.write_t10_2_2_collection_report(
        recorder=recorder,
        output_dir=destination,
        repo_root=root,
        manifest=parent,
        mode="full",
    )
    report = signed_payload(
        {
            "format_version": CONTINUATION_REPORT_FORMAT_VERSION,
            "status": "COMPLETE_T10_2_3_CONTINUATION",
            "manifest_checksum": manifest["manifest_checksum"],
            "migration_receipt_checksum": manifest["migration_receipt"][
                "receipt_checksum"
            ],
            "parent_t10_2_2_report_checksum": parent_report["report_checksum"],
            "cache_entries": _cache_entries(root / _protocol.DEFAULT_CACHE_ROOT),
            "replayed_physical_actions": 0,
            "authority_opened": False,
        },
        checksum_key="report_checksum",
    )
    report_path = root / _protocol.DEFAULT_CACHE_ROOT / CONTINUATION_REPORT_FILENAME
    if report_path.is_file():
        existing = _kernel_protocol._read_signed_json(
            report_path, checksum_key="report_checksum"
        )
        if existing != report:
            raise ManifestDriftError("existing continuation report drifted")
    else:
        _kernel_protocol.write_compact_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("status", "prepare", "collect"))
    parser.add_argument("--manifest", default=str(_protocol.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "status":
            manifest = _protocol.load_manifest(
                args.manifest, repo_root=args.repo_root
            )
            payload = {
                "status": "READY_T10_2_3_CONTINUATION",
                "manifest_checksum": manifest["manifest_checksum"],
                "migration": _protocol.verify_migration_receipt_live(
                    manifest["migration_receipt"], repo_root=args.repo_root
                ),
                "cache_entries": _cache_entries(
                    Path(args.repo_root or _kernel_protocol._repo_root()).resolve()
                    / _protocol.DEFAULT_CACHE_ROOT
                ),
            }
        elif args.phase == "prepare":
            payload = prepare_next_cache(
                manifest_path=args.manifest, repo_root=args.repo_root
            )
        else:
            payload = collect_phase(
                manifest_path=args.manifest, repo_root=args.repo_root
            )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

