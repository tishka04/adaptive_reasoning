from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from theory.sage_t import t10_2_3_runtime as runtime


class _Candidate:
    def __init__(self, name: str) -> None:
        self.canonical_hash = name
        self.gauge_equivalence_key = (name,)


class _Posterior:
    def __init__(self) -> None:
        self.candidates = ()
        self.events: list[str] = []
        self.branches = 0

    def seed(self, candidates: object) -> None:
        self.candidates = tuple(candidates)  # type: ignore[arg-type]

    def observe(self, bundle: object) -> None:
        self.events.append(str(bundle))

    def start_branch(self) -> None:
        self.branches += 1


def _events() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "event_id": f"event-{index}",
            "event_checksum": f"checksum-{index}",
            "game_id": "bp9" if index < 2 else "su18",
            "seed": 101,
            "split": "discovery",
            "selection": {"reset_index": 0},
        }
        for index in range(4)
    )


def test_event_binding_is_order_sensitive() -> None:
    events = _events()
    forward = runtime._event_binding(events, ("bp9", "su18"))
    reverse = runtime._event_binding(tuple(reversed(events)), ("bp9", "su18"))

    assert forward["event_count"] == 4
    assert forward["ordered_events_sha256"] != reverse["ordered_events_sha256"]


def test_exact_cache_builds_and_reloads_without_refit(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime._science,
        "_synthesize_gauge_candidates",
        lambda events, maximum: (_Candidate("one"), _Candidate("two")),
    )
    monkeypatch.setattr(runtime._science, "GaugeProgramPosterior", _Posterior)
    monkeypatch.setattr(
        runtime._science, "_bundle_from_compact_event", lambda event: event["event_id"]
    )
    cache = runtime.ExactDonorPosteriorCache(
        root=tmp_path,
        continuation_manifest_checksum="c" * 64,
        parent_kernel_manifest_checksum="k" * 64,
        checkpoint_interval=2,
    )
    progress: list[dict[str, object]] = []

    metadata = cache.ensure(
        donor_events=_events(),
        training_games=("bp9", "su18"),
        progress=lambda payload: progress.append(dict(payload)),
    )
    monkeypatch.setattr(
        runtime._science,
        "_synthesize_gauge_candidates",
        lambda *args, **kwargs: pytest.fail("completed cache was refit"),
    )
    posterior, candidates, observations, errors, loaded = cache.load_final(
        donor_events=_events(), training_games=("bp9", "su18")
    )
    second = cache.ensure(
        donor_events=_events(), training_games=("bp9", "su18")
    )

    assert metadata["finalized"] is True
    assert second == metadata
    assert loaded == metadata
    assert observations == 4
    assert errors == ()
    assert len(candidates) == 2
    assert posterior.events == ["event-0", "event-1", "event-2", "event-3"]
    assert posterior.branches == 2
    assert len(progress) == 4


def test_exact_cache_rejects_state_tampering(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime._science,
        "_synthesize_gauge_candidates",
        lambda events, maximum: (_Candidate("one"),),
    )
    monkeypatch.setattr(runtime._science, "GaugeProgramPosterior", _Posterior)
    monkeypatch.setattr(runtime._science, "_bundle_from_compact_event", lambda event: event)
    cache = runtime.ExactDonorPosteriorCache(
        root=tmp_path,
        continuation_manifest_checksum="c" * 64,
        parent_kernel_manifest_checksum="k" * 64,
    )
    metadata = cache.ensure(
        donor_events=_events(), training_games=("bp9", "su18")
    )
    binding = cache.binding(
        donor_events=_events(), training_games=("bp9", "su18")
    )
    state_path = cache._directory(binding) / metadata["state_file"]
    state_path.write_bytes(state_path.read_bytes() + b"tamper")

    with pytest.raises(runtime.ManifestDriftError, match="size"):
        cache.load_final(
            donor_events=_events(), training_games=("bp9", "su18")
        )


def test_mapping_proxy_is_pickle_safe() -> None:
    payload = runtime.MappingProxyType({"a": 1})
    restored = runtime.pickle.loads(runtime.pickle.dumps(payload, protocol=5))

    assert dict(restored) == {"a": 1}


def test_factory_clone_retains_continuation_binding(tmp_path: object) -> None:
    factory = runtime.CachedDonorSourceFactory(
        manifest={"manifest_checksum": "k" * 64},
        cache_root=tmp_path,
        continuation_manifest_checksum="c" * 64,
    )

    clone = factory.clone_for_worker()

    assert type(clone) is runtime.CachedDonorSourceFactory
    assert clone.cache_root == factory.cache_root
    assert clone.continuation_manifest_checksum == "c" * 64


def test_confirmation_cache_time_is_removed_before_parent_watchdog(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = runtime.CachedDonorSourceFactory(
        manifest={"manifest_checksum": "k" * 64},
        cache_root=tmp_path,
        continuation_manifest_checksum="c" * 64,
    )
    monkeypatch.setattr(
        runtime.ExactDonorPosteriorCache,
        "ensure",
        lambda self, **kwargs: {"cache_checksum": "x" * 64},
    )
    captured: dict[str, object] = {}

    def parent_run_reset(self: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "parent-outcome"

    monkeypatch.setattr(
        runtime._parent_runtime.ActionBudgetSourceFactory,
        "run_reset",
        parent_run_reset,
    )
    ticks = iter((10.0, 17.5))
    held_out = runtime._kernel_protocol.SOURCE_GAMES[1]
    donors = (
        runtime._kernel_protocol.SOURCE_GAMES[0],
        runtime._kernel_protocol.SOURCE_GAMES[2],
    )
    work = SimpleNamespace(
        lane=SimpleNamespace(
            split="leave_one_game_out_confirmation", game_id=held_out
        )
    )
    discovery = [
        {"game_id": donors[0], "split": "discovery"},
        {"game_id": donors[1], "split": "discovery"},
    ]

    result = factory.run_reset(
        work=work,
        discovery_events=discovery,
        lane_remaining_seconds=100.0,
        cooperative_collection_remaining_seconds=200.0,
        absolute_collection_remaining_seconds=300.0,
        clock=lambda: next(ticks),
    )

    assert result == "parent-outcome"
    assert captured["lane_remaining_seconds"] == 92.5
    assert captured["cooperative_collection_remaining_seconds"] == 192.5
    assert captured["absolute_collection_remaining_seconds"] == 292.5


def test_continuation_factory_binding_records_actual_adapter(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_hash = runtime._kernel_protocol.canonical_file_sha256(runtime.Path(runtime.__file__))
    top = {
        "manifest_checksum": "c" * 64,
        "portable_code_sha256": {
            "theory/sage_t/t10_2_3_runtime.py": source_hash,
        },
    }
    kernel = {
        "manifest_checksum": "k" * 64,
        "portable_code_sha256": {
            "theory/sage_t/t10_2_1_runtime.py": "legacy-source",
        },
    }
    factory = runtime.CachedDonorSourceFactory(
        manifest=kernel,
        cache_root=tmp_path,
        continuation_manifest_checksum="c" * 64,
    )

    with runtime.continuation_factory_binding(top):
        binding = runtime._kernel_protocol._factory_binding(factory, kernel)

    assert binding["code_bound"] is True
    assert binding["class"] == "T10_2_1SourceFactory"
    assert binding["continuation_factory"]["class"] == "CachedDonorSourceFactory"
    assert binding["continuation_factory"]["code_bound"] is True
