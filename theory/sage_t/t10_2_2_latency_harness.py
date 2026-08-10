"""Cold-vs-warm latency harness for SAGE.T10.2.2 (read-only diagnostic).

This harness answers the decisive question raised for T10.2.2: does the per-reset
bookkeeping cost grow with the *global* collection history?  If a fixed unit of
per-reset work (that ought to be O(1)) becomes more expensive as the journal
accumulates events, the whole collection is O(N^2) and later lanes (su15,
confirmations) are structurally disadvantaged -- which no deadline increase can
fix.

It is entirely read-only with respect to real artifacts:

* the synthetic probe times ``canonical_json`` + ``canonical_sha256`` over
  checkpoint-shaped payloads of growing size -- the confirmed per-reset cost the
  frozen loop pays (twice) via ``reconstruct_checkpoint``;
* the optional real-journal probe opens the existing journal in place and calls
  read-only methods only (needed on Windows because deep paths exceed 260 chars).

The gate mirrors the requested criterion::

    p95_warm / p95_cold <= 1.10

for an identical unit of per-reset work.  A ratio well above 1.10 that tracks the
history size is direct evidence of an O(N) per-step operation.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _t10_2_1
from . import t10_2_1_runtime as _t10_2_1_runtime
from . import t10_2_2_runtime as _t10_2_2_runtime

FORMAT_VERSION = "sage-t10.2.2-latency-harness-v1"
LATENCY_REPORT_FORMAT_VERSION = "sage-t10.2.2-latency-report-v1"

# The requested cold/warm gate: an identical unit of per-reset work must not get
# more than 10% slower purely because the global history grew.
WARM_OVER_COLD_GATE = 1.10
# A per-reset unit of work is ~24 sealed events (one discovery reset); "warm"
# ~400 events is roughly a full game's discovery already collected.
COLD_EVENT_COUNT = 24
WARM_EVENT_COUNT = 408
DEFAULT_CURVE_EVENT_COUNTS = (24, 48, 96, 192, 384, 768)
# A single idempotent O(1) update is sub-millisecond and too sensitive to the
# Windows scheduler for a 10% ratio gate.  Measure an identical fixed batch in
# both histories so each sample is several milliseconds while retaining the
# same per-reset operation and asymptotic comparison.
INCREMENTAL_BOOKKEEPING_BATCH_SIZE = 20_000

canonical_json = _t10_2_1.canonical_json
canonical_sha256 = _t10_2_1.canonical_sha256


# ---------------------------------------------------------------------------
# Timing primitives (pure / testable).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatencyStats:
    count: int
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    max_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_seconds": self.mean_seconds,
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "max_seconds": self.max_seconds,
        }

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencyStats":
        ordered = sorted(float(value) for value in samples)
        if not ordered:
            raise ValueError("latency stats require at least one sample")
        return cls(
            count=len(ordered),
            mean_seconds=statistics.fmean(ordered),
            p50_seconds=_percentile(ordered, 0.50),
            p95_seconds=_percentile(ordered, 0.95),
            max_seconds=ordered[-1],
        )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        raise ValueError("percentile of empty sample")
    if len(ordered) == 1:
        return float(ordered[0])
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def measure(
    operation: Callable[[], Any],
    *,
    repeat: int = 25,
    warmup: int = 3,
    disable_gc: bool = True,
) -> LatencyStats:
    """Time ``operation`` ``repeat`` times, discarding ``warmup`` runs."""

    if repeat <= 0:
        raise ValueError("repeat must be positive")
    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    try:
        for _ in range(max(0, warmup)):
            operation()
        samples: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            operation()
            samples.append(time.perf_counter() - start)
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()
    return LatencyStats.from_samples(samples)


# ---------------------------------------------------------------------------
# Growth analysis and the cold/warm gate.
# ---------------------------------------------------------------------------
def cold_warm_gate(
    cold: LatencyStats,
    warm: LatencyStats,
    *,
    threshold: float = WARM_OVER_COLD_GATE,
) -> dict[str, Any]:
    """Compare an identical unit of work cold vs warm; pass if within threshold."""

    ratio = (
        warm.p95_seconds / cold.p95_seconds if cold.p95_seconds > 0 else float("inf")
    )
    return {
        "cold_p95_seconds": cold.p95_seconds,
        "warm_p95_seconds": warm.p95_seconds,
        "warm_over_cold_p95": ratio,
        "threshold": float(threshold),
        "passed": ratio <= float(threshold),
    }


def linear_growth_report(points: Sequence[tuple[int, LatencyStats]]) -> dict[str, Any]:
    """Classify latency-vs-history-size as constant / sublinear / linear-or-worse."""

    ordered = sorted(((int(n), stats) for n, stats in points), key=lambda item: item[0])
    if len(ordered) < 2:
        raise ValueError("growth analysis needs at least two sizes")
    (min_n, min_stats) = ordered[0]
    (max_n, max_stats) = ordered[-1]
    size_ratio = max_n / min_n
    latency_ratio = (
        max_stats.p95_seconds / min_stats.p95_seconds
        if min_stats.p95_seconds > 0
        else float("inf")
    )
    # Per-unit cost: if the operation were O(1) per event, latency/N would fall as
    # N grows.  If latency/N is roughly flat and latency_ratio ~ size_ratio, the
    # operation is O(N) in the global history.
    if latency_ratio <= WARM_OVER_COLD_GATE:
        verdict = "constant"
    elif latency_ratio >= 0.5 * size_ratio:
        verdict = "linear_or_worse"
    else:
        verdict = "sublinear"
    return {
        "points": [{"event_count": n, **stats.to_dict()} for n, stats in ordered],
        "min_event_count": min_n,
        "max_event_count": max_n,
        "size_ratio": size_ratio,
        "p95_latency_ratio": latency_ratio,
        "verdict": verdict,
        "implies_quadratic_collection": verdict == "linear_or_worse",
    }


# ---------------------------------------------------------------------------
# Synthetic per-reset bookkeeping cost (checkpoint serialize + checksum).
# ---------------------------------------------------------------------------
def _synthetic_reset(index: int, *, events: int = 24) -> dict[str, Any]:
    return {
        "format_version": _t10_2_1_runtime.RESET_REPORT_FORMAT_VERSION,
        "work": {"reset_index": index % 4, "controller": "balanced_discovery"},
        "status": "COMPLETE",
        "issued_intents": events,
        "sealed_events": events,
        "unresolved_intents": 0,
        "posterior_updates": events,
        "elapsed_seconds": 53.0,
        "stop_reason": "terminal",
        "event_ids_sha256": f"{index:064x}",
        "continuation": {
            "grounding_counts": {f"move:[{k}]": 1 for k in range(events)},
            "learned_schema_counts": {},
            "independent_schema_counts": {},
        },
        "report_checksum": f"{index:064x}",
    }


def _synthetic_checkpoint_payload(
    event_count: int, *, events_per_reset: int = 24
) -> dict[str, Any]:
    reset_count = max(1, event_count // events_per_reset)
    resets = [_synthetic_reset(i, events=events_per_reset) for i in range(reset_count)]
    return {
        "format_version": _t10_2_1_runtime.CHECKPOINT_FORMAT_VERSION,
        "manifest_checksum": "0" * 64,
        "lane_registry_sha256": "0" * 64,
        "lane_reports": [{"resets": resets}],
        "cumulative_active_seconds": 1.0,
        "revision": reset_count,
    }


def _serialize_and_checksum(payload: dict[str, Any]) -> str:
    # Exactly the frozen per-reset cost: canonical JSON + canonical sha256, which
    # ``reconstruct_checkpoint`` pays over the whole growing structure each reset.
    return canonical_sha256(payload)


def checkpoint_serialization_curve(
    event_counts: Sequence[int] = DEFAULT_CURVE_EVENT_COUNTS,
    *,
    repeat: int = 25,
) -> list[tuple[int, LatencyStats]]:
    curve: list[tuple[int, LatencyStats]] = []
    for count in event_counts:
        payload = _synthetic_checkpoint_payload(count)
        stats = measure(lambda: _serialize_and_checksum(payload), repeat=repeat)
        curve.append((int(count), stats))
    return curve


def incremental_bookkeeping_stats(
    history_size: int,
    *,
    repeat: int = 25,
    batch_size: int = INCREMENTAL_BOOKKEEPING_BATCH_SIZE,
) -> LatencyStats:
    """Measure a fixed idempotent update against a pre-sized running state."""

    if history_size <= 0 or batch_size <= 0:
        raise ValueError("history_size and batch_size must be positive")
    state = _t10_2_2_runtime.IncrementalCollectionState()
    for index in range(history_size):
        state.record_reset(
            work_id=f"history-{index}",
            split="discovery",
            status="COMPLETE",
            issued=1,
            sealed=1,
            unresolved=0,
            posterior_updates=0,
        )
    probe = {
        "work_id": f"history-{history_size - 1}",
        "split": "discovery",
        "status": "COMPLETE",
        "issued": 1,
        "sealed": 1,
        "unresolved": 0,
        "posterior_updates": 0,
    }

    def operation() -> None:
        for _ in range(batch_size):
            state.record_reset(**probe)
            state.accounting().to_dict()

    return measure(operation, repeat=repeat, warmup=3)


# ---------------------------------------------------------------------------
# Optional real-journal probe (read-only, in place; never mutates artifacts).
# ---------------------------------------------------------------------------
def journal_scan_probe(
    *,
    journal_root: str | Path,
    manifest_checksum: str,
    repeat: int = 5,
) -> dict[str, Any]:
    """Time the O(N) journal scans over the real journal, read-only.

    The three probed methods (``lane_reports``, ``accounting``,
    ``_completed_discovery_events``) are pure reads, and constructing the journal
    is idempotent on an existing directory (it only ensures ``journal.json``
    exists), so no real artifact is mutated.  Deep journal paths exceed Windows'
    260-char limit, so the frozen ``_extended_length_path`` prefix is relied on
    rather than copying the tree.
    """

    source = Path(journal_root)
    if not source.is_dir():
        raise FileNotFoundError(f"journal directory not found: {source}")
    journal = _t10_2_1_runtime.DurableCollectionJournal(
        source, manifest_checksum=manifest_checksum
    )
    probes = {
        "lane_reports": lambda: journal.lane_reports(),
        "accounting": lambda: journal.accounting(),
        "completed_discovery_events": lambda: (
            _t10_2_1_runtime._completed_discovery_events(journal)
        ),
    }
    results: dict[str, Any] = {}
    for name, operation in probes.items():
        try:
            results[name] = measure(operation, repeat=repeat, warmup=1).to_dict()
        except Exception as exc:  # noqa: BLE001 - diagnostic best-effort
            results[name] = {"error": f"{type(exc).__name__}:{exc}"}
    return results


# ---------------------------------------------------------------------------
# Top-level report.
# ---------------------------------------------------------------------------
def build_latency_report(
    *,
    event_counts: Sequence[int] = DEFAULT_CURVE_EVENT_COUNTS,
    repeat: int = 25,
    journal_root: str | Path | None = None,
    manifest_checksum: str | None = None,
) -> dict[str, Any]:
    curve = checkpoint_serialization_curve(event_counts, repeat=repeat)
    growth = linear_growth_report(curve)
    by_count = {n: stats for n, stats in curve}
    cold = by_count.get(min(by_count))
    warm = by_count.get(max(by_count))
    legacy_gate = cold_warm_gate(cold, warm)
    incremental_cold = incremental_bookkeeping_stats(COLD_EVENT_COUNT, repeat=repeat)
    incremental_warm = incremental_bookkeeping_stats(WARM_EVENT_COUNT, repeat=repeat)
    incremental_gate = {
        **cold_warm_gate(incremental_cold, incremental_warm),
        "batch_size": INCREMENTAL_BOOKKEEPING_BATCH_SIZE,
    }
    journal_probe: dict[str, Any] | None = None
    if journal_root is not None and manifest_checksum is not None:
        try:
            journal_probe = journal_scan_probe(
                journal_root=journal_root,
                manifest_checksum=manifest_checksum,
                repeat=max(3, repeat // 5),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort diagnostic
            journal_probe = {"error": f"{type(exc).__name__}:{exc}"}
    return _t10_2_1.signed_payload(
        {
            "format_version": LATENCY_REPORT_FORMAT_VERSION,
            "legacy_diagnostic_unit": "checkpoint_serialize_and_checksum_per_reset",
            "legacy_checkpoint_serialization_growth": growth,
            "legacy_cold_warm_gate": legacy_gate,
            "registered_unit_of_work": "incremental_idempotent_reset_bookkeeping",
            "incremental_bookkeeping_gate": incremental_gate,
            "journal_scan_probe": journal_probe,
        },
        checksum_key="report_checksum",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument(
        "--journal-root",
        default=None,
        help="optional real journal directory to probe read-only in place",
    )
    parser.add_argument(
        "--manifest-checksum",
        default=None,
        help="manifest checksum required to open the real journal probe",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_latency_report(
        repeat=args.repeat,
        journal_root=args.journal_root,
        manifest_checksum=args.manifest_checksum,
    )
    print(canonical_json(report))
    gate = report["incremental_bookkeeping_gate"]
    # The legacy diagnostic is expected to fail; T10.2.2 is gated on the fixed
    # incremental unit remaining flat as history grows.
    return 0 if gate["passed"] else 1


__all__ = [
    "COLD_EVENT_COUNT",
    "DEFAULT_CURVE_EVENT_COUNTS",
    "FORMAT_VERSION",
    "INCREMENTAL_BOOKKEEPING_BATCH_SIZE",
    "LATENCY_REPORT_FORMAT_VERSION",
    "LatencyStats",
    "WARM_EVENT_COUNT",
    "WARM_OVER_COLD_GATE",
    "build_latency_report",
    "build_parser",
    "checkpoint_serialization_curve",
    "cold_warm_gate",
    "incremental_bookkeeping_stats",
    "journal_scan_probe",
    "linear_growth_report",
    "main",
    "measure",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
