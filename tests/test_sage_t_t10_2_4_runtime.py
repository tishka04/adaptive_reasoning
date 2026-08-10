from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from theory.sage_t import t10_2_4_runtime as runtime


class _Candidate:
    def __init__(self, name: str) -> None:
        self.canonical_hash = name
        self.gauge_equivalence_key = (name,)


class _Posterior:
    def __init__(self) -> None:
        self.candidates: Sequence[object] = ()
        self.events: list[str] = []
        self.branches = 0

    def seed(self, candidates: Sequence[object]) -> None:
        self.candidates = candidates

    def observe(self, bundle: object) -> None:
        self.events.append(str(bundle))

    def start_branch(self) -> None:
        self.branches += 1


class _Metrics:
    def as_dict(self) -> dict[str, object]:
        return {"capacity_matched": True, "target_particles": 2}


class _Marginal:
    def __init__(self, name: str) -> None:
        self.name = name
        self.entries = ((name, 0.0),)


class _FactorBank(Sequence[object]):
    def __init__(self, candidates: Sequence[object]) -> None:
        self.hypotheses = tuple(candidates)
        self.factor_rows = tuple((candidate.canonical_hash,) for candidate in candidates)
        self.metrics = _Metrics()
        self.marginals = (_Marginal("dynamics"),)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __getitem__(self, index: int) -> object:
        return self.hypotheses[index]


def _events() -> tuple[dict[str, object], ...]:
    games = runtime._kernel_protocol.SOURCE_GAMES
    return tuple(
        {
            "event_id": f"event-{index}",
            "event_checksum": f"checksum-{index}",
            "game_id": games[0] if index < 2 else games[2],
            "seed": 101,
            "split": "discovery",
            "selection": {"reset_index": 0},
        }
        for index in range(4)
    )


def test_gauge_cache_builds_exact_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime._science,
        "_synthesize_gauge_candidates",
        lambda events, maximum: (_Candidate("one"), _Candidate("two")),
    )
    monkeypatch.setattr(runtime, "GaugeProgramPosterior", _Posterior)
    monkeypatch.setattr(
        runtime._science, "_bundle_from_compact_event", lambda event: event["event_id"]
    )
    cache = runtime.ExactFoldPosteriorCache(
        root=tmp_path,
        continuation_manifest_checksum="c" * 64,
        parent_kernel_manifest_checksum="k" * 64,
        checkpoint_interval=2,
    )

    metadata = cache.ensure(
        kind="gauge",
        donor_events=_events(),
        training_games=(
            runtime._kernel_protocol.SOURCE_GAMES[0],
            runtime._kernel_protocol.SOURCE_GAMES[2],
        ),
    )
    posterior, candidates, observations, errors, loaded = cache.load_final(
        kind="gauge",
        donor_events=_events(),
        training_games=(
            runtime._kernel_protocol.SOURCE_GAMES[0],
            runtime._kernel_protocol.SOURCE_GAMES[2],
        ),
    )

    assert metadata == loaded
    assert metadata["finalized"] is True
    assert observations == 4
    assert errors == ()
    assert len(candidates) == 2
    assert posterior.events == ["event-0", "event-1", "event-2", "event-3"]
    assert posterior.branches == 2


def test_factorized_cache_preserves_audited_bank_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "FactorizedCandidateBank", _FactorBank)
    monkeypatch.setattr(runtime, "FactorizedGaugeProgramPosterior", _Posterior)
    monkeypatch.setattr(runtime._science, "_bundle_from_compact_event", lambda event: event)
    bank = _FactorBank((_Candidate("one"), _Candidate("two")))
    cache = runtime.ExactFoldPosteriorCache(
        root=tmp_path,
        continuation_manifest_checksum="c" * 64,
        parent_kernel_manifest_checksum="k" * 64,
    )
    games = (
        runtime._kernel_protocol.SOURCE_GAMES[0],
        runtime._kernel_protocol.SOURCE_GAMES[2],
    )

    cache.ensure(
        kind="factorized",
        donor_events=_events(),
        training_games=games,
        candidates=bank,
    )
    posterior, restored_bank, observations, errors, _ = cache.load_final(
        kind="factorized",
        donor_events=_events(),
        training_games=games,
        candidates=bank,
    )

    assert isinstance(restored_bank, _FactorBank)
    assert posterior.candidates is restored_bank
    assert observations == 4
    assert errors == ()


def test_exact_fit_uses_cached_gauge_only_on_full_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _events()
    games = (
        runtime._kernel_protocol.SOURCE_GAMES[0],
        runtime._kernel_protocol.SOURCE_GAMES[2],
    )
    candidates = (_Candidate("one"), _Candidate("two"))
    factory = object.__new__(runtime.DualCachedDonorSourceFactory)
    factory._discovery_events = {
        game: [dict(event) for event in events if event["game_id"] == game]
        for game in runtime._kernel_protocol.SOURCE_GAMES
    }
    coordinator = SimpleNamespace(
        load_gauge=lambda **kwargs: (
            "cached-posterior",
            candidates,
            4,
            (),
            {"cache_checksum": "x" * 64},
        )
    )
    factory._coordinator = lambda: coordinator
    delegated: list[int] = []

    def original(events: object, **kwargs: object) -> tuple[object, tuple[()], int, list[str]]:
        delegated.append(int(kwargs["maximum_candidates"]))
        return "original", (), 0, []

    monkeypatch.setattr(runtime._science, "_fit_compact_posterior", original)
    with runtime.exact_donor_fit_binding(factory):
        cached = runtime._science._fit_compact_posterior(
            events, candidates=candidates, maximum_candidates=256
        )
        fallback = runtime._science._fit_compact_posterior(
            events, candidates=candidates, maximum_candidates=128
        )

    assert cached[:3] == ("cached-posterior", candidates, 4)
    assert fallback[0] == "original"
    assert delegated == [128]


def test_worker_binding_is_scoped() -> None:
    original = runtime._parent_runtime._action_budget_reset_worker_entry

    with runtime.dual_cache_worker_binding():
        assert (
            runtime._parent_runtime._action_budget_reset_worker_entry
            is runtime._dual_cache_reset_worker_entry
        )

    assert runtime._parent_runtime._action_budget_reset_worker_entry is original


def test_predecessor_main_pickle_binding_is_scoped(tmp_path: Path) -> None:
    coordinator = runtime.DualCacheCoordinator(
        root=tmp_path / "new",
        predecessor_root=tmp_path / "old",
        continuation_manifest_checksum="c" * 64,
        predecessor_manifest_checksum="p" * 64,
        parent_kernel_manifest_checksum="k" * 64,
    )
    main_module = sys.modules["__main__"]
    name = "_restore_mapping_proxy"
    existed = hasattr(main_module, name)
    previous = getattr(main_module, name, None)

    with coordinator._predecessor_pickle_binding():
        assert getattr(main_module, name) is runtime._predecessor_runtime._restore_mapping_proxy

    assert hasattr(main_module, name) is existed
    if existed:
        assert getattr(main_module, name) is previous


def test_confirmation_build_time_is_removed_before_reset_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = runtime.DualCachedDonorSourceFactory(
        manifest={"manifest_checksum": "k" * 64},
        cache_root=tmp_path / "new",
        predecessor_cache_root=tmp_path / "old",
        continuation_manifest_checksum="c" * 64,
        predecessor_manifest_checksum="p" * 64,
    )
    monkeypatch.setattr(
        factory,
        "ensure_fold",
        lambda **kwargs: {
            "donor_event_count": 546,
            "caches": {
                "gauge": {"cache_source": "t10_2_3_read_only"},
                "factorized": {"cache_checksum": "f" * 64},
            },
        },
    )
    captured: dict[str, object] = {}

    def parent(self: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "outcome"

    monkeypatch.setattr(
        runtime._parent_runtime.ActionBudgetSourceFactory, "run_reset", parent
    )
    ticks = iter((10.0, 27.5))
    result = factory.run_reset(
        work=SimpleNamespace(
            lane=SimpleNamespace(
                split="leave_one_game_out_confirmation",
                game_id=runtime._kernel_protocol.SOURCE_GAMES[1],
            )
        ),
        discovery_events=(),
        lane_remaining_seconds=100.0,
        cooperative_collection_remaining_seconds=200.0,
        absolute_collection_remaining_seconds=300.0,
        clock=lambda: next(ticks),
    )

    assert result == "outcome"
    assert captured["lane_remaining_seconds"] == 82.5
    assert captured["cooperative_collection_remaining_seconds"] == 182.5
    assert captured["absolute_collection_remaining_seconds"] == 282.5
