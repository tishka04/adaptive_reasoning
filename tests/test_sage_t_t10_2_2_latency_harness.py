from __future__ import annotations

import pytest

from theory.sage_t import t10_2_2_latency_harness as harness


# ---------------------------------------------------------------------------
# Timing primitives.
# ---------------------------------------------------------------------------
def test_latency_stats_percentiles() -> None:
    stats = harness.LatencyStats.from_samples([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats.count == 5
    assert stats.p50_seconds == pytest.approx(3.0)
    assert stats.max_seconds == 100.0
    assert stats.p95_seconds > stats.p50_seconds


def test_measure_runs_operation() -> None:
    calls = {"n": 0}

    def op() -> None:
        calls["n"] += 1

    stats = harness.measure(op, repeat=10, warmup=2)
    assert stats.count == 10
    assert calls["n"] == 12  # warmup + repeat
    assert stats.mean_seconds >= 0.0


# ---------------------------------------------------------------------------
# Cold/warm gate.
# ---------------------------------------------------------------------------
def test_cold_warm_gate_passes_when_flat() -> None:
    cold = harness.LatencyStats.from_samples([1.0, 1.0, 1.0])
    warm = harness.LatencyStats.from_samples([1.05, 1.05, 1.05])
    gate = harness.cold_warm_gate(cold, warm)
    assert gate["passed"] is True
    assert gate["warm_over_cold_p95"] == pytest.approx(1.05)


def test_cold_warm_gate_fails_when_warm_much_slower() -> None:
    cold = harness.LatencyStats.from_samples([1.0, 1.0, 1.0])
    warm = harness.LatencyStats.from_samples([2.0, 2.0, 2.0])
    gate = harness.cold_warm_gate(cold, warm)
    assert gate["passed"] is False
    assert gate["warm_over_cold_p95"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Growth classification (synthetic, deterministic latencies).
# ---------------------------------------------------------------------------
def _stats(value: float) -> harness.LatencyStats:
    return harness.LatencyStats.from_samples([value, value, value])


def test_linear_growth_detected() -> None:
    # Latency proportional to N -> linear_or_worse -> quadratic collection.
    points = [(10, _stats(1.0)), (20, _stats(2.0)), (40, _stats(4.0))]
    report = harness.linear_growth_report(points)
    assert report["verdict"] == "linear_or_worse"
    assert report["implies_quadratic_collection"] is True
    assert report["size_ratio"] == pytest.approx(4.0)
    assert report["p95_latency_ratio"] == pytest.approx(4.0)


def test_constant_growth_detected() -> None:
    points = [(10, _stats(1.0)), (40, _stats(1.02))]
    report = harness.linear_growth_report(points)
    assert report["verdict"] == "constant"
    assert report["implies_quadratic_collection"] is False


def test_sublinear_growth_detected() -> None:
    # Latency grows, but far slower than N (e.g., ~log).
    points = [(10, _stats(1.0)), (160, _stats(1.5))]
    report = harness.linear_growth_report(points)
    assert report["verdict"] == "sublinear"
    assert report["implies_quadratic_collection"] is False


# ---------------------------------------------------------------------------
# Synthetic checkpoint-serialization curve is real work and grows with N.
# ---------------------------------------------------------------------------
def test_checkpoint_serialization_curve_shape() -> None:
    curve = harness.checkpoint_serialization_curve((24, 96, 384), repeat=5)
    assert [n for n, _ in curve] == [24, 96, 384]
    for _, stats in curve:
        assert isinstance(stats, harness.LatencyStats)
        assert stats.p95_seconds >= 0.0


def test_build_latency_report_is_signed_and_complete() -> None:
    report = harness.build_latency_report(event_counts=(24, 96, 384), repeat=5)
    assert report["format_version"] == harness.LATENCY_REPORT_FORMAT_VERSION
    assert "legacy_checkpoint_serialization_growth" in report
    assert "legacy_cold_warm_gate" in report
    assert "incremental_bookkeeping_gate" in report
    assert (
        report["incremental_bookkeeping_gate"]["batch_size"]
        == harness.INCREMENTAL_BOOKKEEPING_BATCH_SIZE
    )
    assert report["journal_scan_probe"] is None
    unsigned = {k: v for k, v in report.items() if k != "report_checksum"}
    assert report["report_checksum"] == harness.canonical_sha256(unsigned)


def test_incremental_bookkeeping_gate_measures_fixed_work() -> None:
    cold = harness.incremental_bookkeeping_stats(24, repeat=5, batch_size=100)
    warm = harness.incremental_bookkeeping_stats(408, repeat=5, batch_size=100)
    assert cold.count == 5
    assert warm.count == 5
    assert cold.p95_seconds >= 0.0
    assert warm.p95_seconds >= 0.0
