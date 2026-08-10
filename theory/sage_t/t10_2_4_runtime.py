"""Dual exact-cache continuation runtime for SAGE.T10.2.4."""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_2_runtime as _parent_runtime
from . import t10_2_3_protocol as _predecessor_protocol
from . import t10_2_3_runtime as _predecessor_runtime
from . import t10_2_4_protocol as _protocol
from . import t10_2_runtime as _science
from .factorized_posterior_v10_2 import (
    FactorizedCandidateBank,
    FactorizedGaugeProgramPosterior,
)
from .gauge_inference_v10_2 import GaugeProgramPosterior

FORMAT_VERSION = "sage-t10.2.4-runtime-v1"
CACHE_FORMAT_VERSION = "sage-t10.2.4-exact-fold-cache-v1"
CONTINUATION_REPORT_FORMAT_VERSION = "sage-t10.2.4-continuation-report-v1"
CONTINUATION_REPORT_FILENAME = "t10_2_4_continuation_report.json"

ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
canonical_sha256 = _kernel_protocol.canonical_sha256
canonical_json = _kernel_protocol.canonical_json
signed_payload = _kernel_protocol.signed_payload


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _event_binding(
    donor_events: Sequence[Mapping[str, Any]], training_games: Sequence[str]
) -> dict[str, Any]:
    return _predecessor_runtime._event_binding(donor_events, training_games)


def _candidate_binding(candidates: Sequence[Any], *, kind: str) -> dict[str, Any]:
    hashes = [str(candidate.canonical_hash) for candidate in candidates]
    payload: dict[str, Any] = {
        "kind": kind,
        "type": f"{type(candidates).__module__}.{type(candidates).__name__}",
        "count": len(candidates),
        "ordered_candidate_hashes_sha256": canonical_sha256(hashes),
    }
    if isinstance(candidates, FactorizedCandidateBank):
        metrics_payload = candidates.metrics.as_dict()
        payload["factor_rows_sha256"] = canonical_sha256(
            [list(row) for row in candidates.factor_rows]
        )
        payload["metrics"] = dict(metrics_payload)
        payload["marginals"] = {
            marginal.name: [list(entry) for entry in marginal.entries]
            for marginal in candidates.marginals
        }
    return payload


def _ordered_training_games(events: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    games: list[str] = []
    for event in events:
        game = str(event.get("game_id", ""))
        if game and game not in games:
            games.append(game)
    return tuple(games)


class ExactFoldPosteriorCache:
    """Authenticated resumable cache for gauge or factorized donor fits."""

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
            raise ValueError("dual-cache bounds must be positive")

    def binding(
        self,
        *,
        kind: str,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        candidates: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in {"gauge", "factorized"}:
            raise ValueError(f"unregistered exact cache kind: {kind}")
        if kind == "factorized" and not isinstance(candidates, FactorizedCandidateBank):
            raise ValueError("factorized cache requires an audited candidate bank")
        candidate_binding = (
            {
                "kind": "gauge",
                "source": "deterministic_synthesis_from_exact_events",
                "maximum_candidates": 256,
            }
            if candidates is None
            else _candidate_binding(candidates, kind=kind)
        )
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "kind": kind,
            "continuation_manifest_checksum": self.continuation_manifest_checksum,
            "parent_kernel_manifest_checksum": self.parent_kernel_manifest_checksum,
            "runtime_source_sha256": _kernel_protocol.canonical_file_sha256(
                Path(__file__)
            ),
            "posterior_factory": (
                "GaugeProgramPosterior"
                if kind == "gauge"
                else "FactorizedGaugeProgramPosterior"
            ),
            "donors": _event_binding(donor_events, training_games),
            "candidates": candidate_binding,
        }
        return {**payload, "cache_key": canonical_sha256(payload)}

    def _directory(self, binding: Mapping[str, Any]) -> Path:
        return self.root / str(binding["kind"]) / str(binding["cache_key"])

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
            raise ManifestDriftError("T10.2.4 donor cache format drifted")
        if metadata.get("binding") != dict(binding):
            raise ManifestDriftError("T10.2.4 donor cache binding drifted")
        return metadata

    def _load_state(
        self, binding: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        metadata = self._read_metadata(binding)
        if metadata is None:
            return None
        state_name = str(metadata.get("state_file", ""))
        if state_name not in {"state-a.pkl", "state-b.pkl"}:
            raise ManifestDriftError("T10.2.4 donor cache slot drifted")
        state_path = self._directory(binding) / state_name
        if not state_path.is_file() or state_path.is_symlink():
            raise ManifestDriftError("T10.2.4 donor cache state is missing")
        raw = state_path.read_bytes()
        if len(raw) != int(metadata.get("state_bytes", -1)):
            raise ManifestDriftError("T10.2.4 donor cache size drifted")
        if _raw_sha256(raw) != metadata.get("state_sha256"):
            raise ManifestDriftError("T10.2.4 donor cache checksum drifted")
        try:
            state = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 - local cache is fail-closed.
            raise ManifestDriftError("T10.2.4 donor cache is unreadable") from exc
        if not isinstance(state, dict) or state.get("binding") != dict(binding):
            raise ManifestDriftError("T10.2.4 donor cache payload drifted")
        if bool(state.get("finalized")) != bool(metadata.get("finalized")):
            raise ManifestDriftError("T10.2.4 cache finalization drifted")
        if int(state.get("next_event_index", -1)) != int(
            metadata.get("next_event_index", -2)
        ):
            raise ManifestDriftError("T10.2.4 cache progress drifted")
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
            raise JournalIntegrityError("T10.2.4 cache exceeded its byte cap")
        target = directory / slot
        temporary = directory / f".{slot}.{os.getpid()}.tmp"
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _kernel_runtime._fsync_directory(directory)
        candidate_bank = state["candidates"]
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
                "candidate_binding": _candidate_binding(
                    candidate_bank, kind=str(binding["kind"])
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
        kind: str,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        candidates: Sequence[Any] | None = None,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        events = tuple(dict(event) for event in donor_events)
        bank: Sequence[Any]
        if kind == "gauge":
            bank = tuple(
                _science._synthesize_gauge_candidates(events, maximum=256)
            )
            binding_candidates = None
        elif kind == "factorized" and isinstance(candidates, FactorizedCandidateBank):
            bank = candidates
            binding_candidates = candidates
        else:
            raise ValueError("exact factorized cache requires its audited bank")
        binding = self.binding(
            kind=kind,
            donor_events=events,
            training_games=training_games,
            candidates=binding_candidates,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        lease = _kernel_runtime._CollectionLease.acquire(
            self.root / f".{kind}-{binding['cache_key']}.lock"
        )
        try:
            loaded = self._load_state(binding)
            if loaded is not None:
                state, metadata = loaded
                if state["finalized"]:
                    if _candidate_binding(
                        state["candidates"], kind=kind
                    ) != _candidate_binding(bank, kind=kind):
                        raise ManifestDriftError(
                            "T10.2.4 cached candidate bank drifted"
                        )
                    return metadata
            else:
                if not bank:
                    raise JournalIntegrityError(
                        "T10.2.4 exact cache has no executable candidates"
                    )
                posterior = (
                    GaugeProgramPosterior()
                    if kind == "gauge"
                    else FactorizedGaugeProgramPosterior()
                )
                posterior.seed(bank)
                state = {
                    "binding": dict(binding),
                    "posterior": posterior,
                    "candidates": bank,
                    "next_event_index": 0,
                    "observations": 0,
                    "errors": [],
                    "branch": None,
                    "finalized": False,
                }
                metadata = self._write_state(binding=binding, state=state)

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
                            "phase": "build_exact_fold_cache",
                            "kind": kind,
                            "cache_key": binding["cache_key"],
                            "events_processed": index + 1,
                            "events_total": total,
                            "checkpointed": (index + 1) % self.checkpoint_interval
                            == 0,
                        }
                    )
                if (index + 1) % self.checkpoint_interval == 0:
                    metadata = self._write_state(binding=binding, state=state)

            if state["observations"]:
                start_branch = getattr(state["posterior"], "start_branch", None)
                if callable(start_branch):
                    start_branch()
            state["finalized"] = True
            return self._write_state(binding=binding, state=state)
        finally:
            lease.release()

    def load_final(
        self,
        *,
        kind: str,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        candidates: Sequence[Any] | None = None,
    ) -> tuple[Any, Sequence[Any], int, tuple[str, ...], dict[str, Any]]:
        binding = self.binding(
            kind=kind,
            donor_events=donor_events,
            training_games=training_games,
            candidates=candidates if kind == "factorized" else None,
        )
        loaded = self._load_state(binding)
        if loaded is None or loaded[0].get("finalized") is not True:
            raise ManifestDriftError(f"exact {kind} donor cache is incomplete")
        state, metadata = loaded
        if int(state["next_event_index"]) != len(donor_events):
            raise ManifestDriftError(f"exact {kind} donor cache is incomplete")
        if candidates is not None and _candidate_binding(
            state["candidates"], kind=kind
        ) != _candidate_binding(candidates, kind=kind):
            raise ManifestDriftError(f"exact {kind} candidate bank drifted")
        return (
            state["posterior"],
            state["candidates"],
            int(state["observations"]),
            tuple(str(item) for item in state["errors"]),
            metadata,
        )


class DualCacheCoordinator:
    def __init__(
        self,
        *,
        root: str | Path,
        predecessor_root: str | Path,
        continuation_manifest_checksum: str,
        predecessor_manifest_checksum: str,
        parent_kernel_manifest_checksum: str,
    ) -> None:
        self.root = Path(root).resolve()
        self.predecessor_root = Path(predecessor_root).resolve()
        self.continuation_manifest_checksum = str(continuation_manifest_checksum)
        self.predecessor_manifest_checksum = str(predecessor_manifest_checksum)
        self.parent_kernel_manifest_checksum = str(parent_kernel_manifest_checksum)
        policy = _protocol.continuation_policy()
        self.current = ExactFoldPosteriorCache(
            root=self.root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            parent_kernel_manifest_checksum=self.parent_kernel_manifest_checksum,
            checkpoint_interval=int(policy["cache_checkpoint_interval_events"]),
            maximum_bytes=int(policy["cache_maximum_bytes"]),
        )

    def _predecessor(self) -> _predecessor_runtime.ExactDonorPosteriorCache:
        return _predecessor_runtime.ExactDonorPosteriorCache(
            root=self.predecessor_root,
            continuation_manifest_checksum=self.predecessor_manifest_checksum,
            parent_kernel_manifest_checksum=self.parent_kernel_manifest_checksum,
        )

    @contextmanager
    def _predecessor_pickle_binding(self):
        """Resolve caches written while T10.2.3 was the ``-m`` entrypoint.

        The frozen predecessor registered its MappingProxy reducer from the
        executing module.  Caches therefore name ``__main__`` when they were
        produced by the documented CLI.  Bind only that exact reducer name to
        the byte-identical predecessor function while unpickling, then restore
        the caller's main module.
        """

        main_module = sys.modules.get("__main__")
        if main_module is None:
            raise ManifestDriftError("Python main module is unavailable")
        name = "_restore_mapping_proxy"
        sentinel = object()
        previous = getattr(main_module, name, sentinel)
        setattr(main_module, name, _predecessor_runtime._restore_mapping_proxy)
        try:
            yield
        finally:
            if previous is sentinel:
                delattr(main_module, name)
            else:
                setattr(main_module, name, previous)

    def _load_predecessor_final(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
    ) -> tuple[Any, Sequence[Any], int, tuple[str, ...], dict[str, Any]]:
        with self._predecessor_pickle_binding():
            return self._predecessor().load_final(
                donor_events=donor_events, training_games=training_games
            )

    def _predecessor_exists(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
    ) -> bool:
        cache = self._predecessor()
        binding = cache.binding(
            donor_events=donor_events, training_games=training_games
        )
        return cache._metadata_path(binding).is_file()

    def ensure_gauge(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self._predecessor_exists(
            donor_events=donor_events, training_games=training_games
        ):
            _, _, _, _, metadata = self._load_predecessor_final(
                donor_events=donor_events, training_games=training_games
            )
            return {**metadata, "cache_source": "t10_2_3_read_only"}
        metadata = self.current.ensure(
            kind="gauge",
            donor_events=donor_events,
            training_games=training_games,
            progress=progress,
        )
        return {**metadata, "cache_source": "t10_2_4"}

    def load_gauge(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
    ) -> tuple[Any, Sequence[Any], int, tuple[str, ...], dict[str, Any]]:
        if self._predecessor_exists(
            donor_events=donor_events, training_games=training_games
        ):
            posterior, candidates, observations, errors, metadata = (
                self._load_predecessor_final(
                    donor_events=donor_events, training_games=training_games
                )
            )
            return (
                posterior,
                candidates,
                observations,
                errors,
                {**metadata, "cache_source": "t10_2_3_read_only"},
            )
        return self.current.load_final(
            kind="gauge",
            donor_events=donor_events,
            training_games=training_games,
        )

    def ensure_fold(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        gauge_metadata = self.ensure_gauge(
            donor_events=donor_events,
            training_games=training_games,
            progress=progress,
        )
        _, gauge_candidates, _, _, _ = self.load_gauge(
            donor_events=donor_events, training_games=training_games
        )
        try:
            factorized_bank = _science._capacity_matched_independent_bank(
                gauge_candidates
            )
        except _kernel_protocol.DataGateError as exc:
            return {
                "gauge": gauge_metadata,
                "factorized": None,
                "factorized_refusal": str(exc),
            }
        factorized_metadata = self.current.ensure(
            kind="factorized",
            donor_events=donor_events,
            training_games=training_games,
            candidates=factorized_bank,
            progress=progress,
        )
        return {
            "gauge": gauge_metadata,
            "factorized": factorized_metadata,
            "factorized_refusal": "",
        }

    def load_factorized(
        self,
        *,
        donor_events: Sequence[Mapping[str, Any]],
        training_games: Sequence[str],
        candidates: FactorizedCandidateBank,
    ) -> tuple[Any, Sequence[Any], int, tuple[str, ...], dict[str, Any]]:
        return self.current.load_final(
            kind="factorized",
            donor_events=donor_events,
            training_games=training_games,
            candidates=candidates,
        )


class DualCachedDonorSourceFactory(_parent_runtime.ActionBudgetSourceFactory):
    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        cache_root: str | Path,
        predecessor_cache_root: str | Path,
        continuation_manifest_checksum: str,
        predecessor_manifest_checksum: str,
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
        self.predecessor_cache_root = str(Path(predecessor_cache_root).resolve())
        self.continuation_manifest_checksum = str(continuation_manifest_checksum)
        self.predecessor_manifest_checksum = str(predecessor_manifest_checksum)

    def _coordinator(self) -> DualCacheCoordinator:
        return DualCacheCoordinator(
            root=self.cache_root,
            predecessor_root=self.predecessor_cache_root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            predecessor_manifest_checksum=self.predecessor_manifest_checksum,
            parent_kernel_manifest_checksum=self.manifest_checksum,
        )

    def clone_for_worker(self) -> "DualCachedDonorSourceFactory":
        return DualCachedDonorSourceFactory(
            manifest=self.manifest,
            cache_root=self.cache_root,
            predecessor_cache_root=self.predecessor_cache_root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            predecessor_manifest_checksum=self.predecessor_manifest_checksum,
            runtime_loader=self._runtime_loader,
            bundle_builder=self._bundle_builder,
        )

    def _donor_posterior_scores(
        self, training_games: Sequence[str]
    ) -> tuple[Any, tuple[Any, ...], tuple[Mapping[str, Any], ...], dict[Any, float], dict[str, Any]]:
        donor_events = _predecessor_runtime._donor_events_from_hydrated_factory(
            self, training_games
        )
        posterior, candidates, observations, errors, cache_metadata = (
            self._coordinator().load_gauge(
                donor_events=donor_events, training_games=training_games
            )
        )
        scores = _science._posterior_action_scores(posterior)
        candidate_tuple = tuple(candidates)
        metadata = {
            "posterior_used": True,
            "posterior_candidates": len(candidate_tuple),
            "posterior_classes": len(
                {candidate.gauge_equivalence_key for candidate in candidate_tuple}
            ),
            "posterior_observations": observations,
            "posterior_errors": len(errors),
            "option_conditioned": bool(scores),
            "exact_donor_cache_used": True,
            "exact_donor_cache_checksum": cache_metadata["cache_checksum"],
            "exact_donor_cache_source": cache_metadata.get(
                "cache_source", "t10_2_4"
            ),
        }
        return posterior, candidate_tuple, donor_events, scores, metadata

    def ensure_fold(
        self,
        *,
        discovery_events: Sequence[Mapping[str, Any]],
        held_out_game: str,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        training_games = tuple(
            game for game in _kernel_protocol.SOURCE_GAMES if game != held_out_game
        )
        donor_events = _predecessor_runtime._donor_events_from_collection(
            discovery_events, training_games
        )
        caches = self._coordinator().ensure_fold(
            donor_events=donor_events,
            training_games=training_games,
            progress=progress,
        )
        return {
            "held_out_game": held_out_game,
            "training_games": list(training_games),
            "donor_event_count": len(donor_events),
            "caches": caches,
        }

    def run_reset(self, **kwargs: Any) -> Any:
        work = kwargs["work"]
        if work.lane.split != "leave_one_game_out_confirmation":
            return _parent_runtime.ActionBudgetSourceFactory.run_reset(self, **kwargs)
        clock = kwargs.get("clock", time.perf_counter)
        started = float(clock())
        fold = self.ensure_fold(
            discovery_events=kwargs["discovery_events"],
            held_out_game=work.lane.game_id,
            progress=lambda payload: print(canonical_json(payload), flush=True),
        )
        elapsed = max(0.0, float(clock()) - started)
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
                    "phase": "dual_donor_cache_ready",
                    "held_out_game": work.lane.game_id,
                    "donor_event_count": fold["donor_event_count"],
                    "gauge_cache_source": fold["caches"]["gauge"].get(
                        "cache_source", "t10_2_4"
                    ),
                    "factorized_cache_ready": fold["caches"]["factorized"]
                    is not None,
                    "elapsed_seconds": elapsed,
                    "reset_watchdog_armed": False,
                }
            ),
            flush=True,
        )
        return _parent_runtime.ActionBudgetSourceFactory.run_reset(self, **adjusted)


@contextmanager
def exact_donor_fit_binding(factory: DualCachedDonorSourceFactory):
    """Substitute only exact authenticated donor-only reconstructions."""

    original = _science._fit_compact_posterior

    def exact_fit(
        events: Sequence[Mapping[str, Any]],
        *,
        candidates: Sequence[Any] = (),
        posterior_factory: Callable[..., Any] = GaugeProgramPosterior,
        maximum_candidates: int = 256,
    ) -> tuple[Any, Sequence[Any], int, list[str]]:
        event_tuple = tuple(dict(event) for event in events)
        training_games = _ordered_training_games(event_tuple)
        if (
            maximum_candidates != 256
            or not candidates
            or len(training_games) != 2
            or set(training_games) - set(_kernel_protocol.SOURCE_GAMES)
        ):
            return original(
                events,
                candidates=candidates,
                posterior_factory=posterior_factory,
                maximum_candidates=maximum_candidates,
            )
        hydrated = _predecessor_runtime._donor_events_from_hydrated_factory(
            factory, training_games
        )
        if _event_binding(event_tuple, training_games) != _event_binding(
            hydrated, training_games
        ):
            return original(
                events,
                candidates=candidates,
                posterior_factory=posterior_factory,
                maximum_candidates=maximum_candidates,
            )
        coordinator = factory._coordinator()
        if posterior_factory is GaugeProgramPosterior and not isinstance(
            candidates, FactorizedCandidateBank
        ):
            posterior, cached_bank, observations, errors, _ = coordinator.load_gauge(
                donor_events=event_tuple, training_games=training_games
            )
            if _candidate_binding(
                tuple(cached_bank), kind="gauge"
            ) == _candidate_binding(tuple(candidates), kind="gauge"):
                return posterior, tuple(cached_bank), observations, list(errors)
        if posterior_factory is FactorizedGaugeProgramPosterior and isinstance(
            candidates, FactorizedCandidateBank
        ):
            posterior, cached_bank, observations, errors, _ = (
                coordinator.load_factorized(
                    donor_events=event_tuple,
                    training_games=training_games,
                    candidates=candidates,
                )
            )
            return posterior, cached_bank, observations, list(errors)
        return original(
            events,
            candidates=candidates,
            posterior_factory=posterior_factory,
            maximum_candidates=maximum_candidates,
        )

    _science._fit_compact_posterior = exact_fit
    try:
        yield
    finally:
        _science._fit_compact_posterior = original


def _dual_cache_reset_worker_entry(*args: Any) -> None:
    factory = args[0]
    if not isinstance(factory, DualCachedDonorSourceFactory):
        raise _kernel_runtime.WorkerProtocolError(
            "T10.2.4 worker received an unregistered source factory"
        )
    with (
        _parent_runtime.gauge_preview_copy_binding(),
        exact_donor_fit_binding(factory),
    ):
        _kernel_runtime._reset_worker_entry(*args)


@contextmanager
def dual_cache_worker_binding():
    original = _parent_runtime._action_budget_reset_worker_entry
    _parent_runtime._action_budget_reset_worker_entry = _dual_cache_reset_worker_entry
    try:
        yield
    finally:
        _parent_runtime._action_budget_reset_worker_entry = original


@contextmanager
def continuation_factory_binding(manifest: Mapping[str, Any]):
    original = _kernel_protocol._factory_binding

    def bind(factory: Any, kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
        if type(factory) is not DualCachedDonorSourceFactory:
            return original(factory, kernel_manifest)
        relative = "theory/sage_t/t10_2_4_runtime.py"
        observed = _kernel_protocol.canonical_file_sha256(Path(__file__))
        continuation_bound = bool(
            observed == manifest.get("portable_code_sha256", {}).get(relative)
            and factory.continuation_manifest_checksum
            == manifest.get("manifest_checksum")
            and factory.predecessor_manifest_checksum
            == manifest.get("predecessor_t10_2_3_manifest_checksum")
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
                "predecessor_manifest_checksum": (
                    factory.predecessor_manifest_checksum
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
        manifest_path,
        repo_root=root,
        verify_repository=True,
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


def _factory(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    kernel: Mapping[str, Any],
    watchdog: Any = None,
) -> DualCachedDonorSourceFactory:
    return DualCachedDonorSourceFactory(
        manifest=kernel,
        cache_root=root / _protocol.DEFAULT_CACHE_ROOT,
        predecessor_cache_root=root / _predecessor_protocol.DEFAULT_CACHE_ROOT,
        continuation_manifest_checksum=str(manifest["manifest_checksum"]),
        predecessor_manifest_checksum=str(
            manifest["predecessor_t10_2_3_manifest_checksum"]
        ),
        watchdog=watchdog,
    )


def _cache_entries(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not cache_root.is_dir():
        return entries
    for path in sorted(cache_root.glob("*/*/metadata.json")):
        metadata = _kernel_protocol._read_signed_json(
            path, checksum_key="cache_checksum"
        )
        if metadata.get("finalized") is True:
            entries.append(
                {
                    "kind": metadata["binding"]["kind"],
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
    root, manifest, _, kernel, kernel_path, artifact_root = _load_execution_context(
        manifest_path=manifest_path, repo_root=repo_root
    )
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
        complete_ids = {report.lane.lane_id for report in journal.lane_reports()}
        next_lane = next(
            lane
            for lane in _parent_runtime._execution_lanes("full")
            if lane.lane_id not in complete_ids
        )
        if next_lane.split != "leave_one_game_out_confirmation":
            raise ManifestDriftError("next lane does not require a donor cache")
        discovery_events = journal.completed_discovery_events()
    fold = _factory(root=root, manifest=manifest, kernel=kernel).ensure_fold(
        discovery_events=discovery_events,
        held_out_game=next_lane.game_id,
        progress=lambda payload: print(canonical_json(payload), flush=True),
    )
    return {
        "status": "READY_T10_2_4_CONTINUATION",
        "next_lane": next_lane.to_dict(),
        "donor_event_count": fold["donor_event_count"],
        "gauge_cache_source": fold["caches"]["gauge"].get(
            "cache_source", "t10_2_4"
        ),
        "factorized_cache_checksum": (
            None
            if fold["caches"]["factorized"] is None
            else fold["caches"]["factorized"]["cache_checksum"]
        ),
        "factorized_refusal": fold["caches"]["factorized_refusal"],
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
    root, manifest, parent, kernel, kernel_path, artifact_root = (
        _load_execution_context(manifest_path=manifest_path, repo_root=repo_root)
    )
    destination = (root / artifact_root).resolve()
    if output_dir is not None:
        candidate = Path(output_dir)
        candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if candidate != destination:
            raise ManifestDriftError("T10.2.4 output escaped the parent journal")
    recorder = recorder or _parent_runtime.CollectionTimingRecorder()
    factory = _factory(
        root=root,
        manifest=manifest,
        kernel=kernel,
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
        dual_cache_worker_binding(),
        continuation_factory_binding(manifest),
    ):
        verified = _kernel_protocol.load_manifest(kernel_path, repo_root=root)
        if verified != kernel:
            raise ManifestDriftError("verified execution kernel escaped T10.2.4")
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
            "status": "COMPLETE_T10_2_4_CONTINUATION",
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
            raise ManifestDriftError("existing T10.2.4 continuation report drifted")
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
            manifest = _protocol.load_manifest(args.manifest, repo_root=args.repo_root)
            payload = {
                "status": "READY_T10_2_4_CONTINUATION",
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
