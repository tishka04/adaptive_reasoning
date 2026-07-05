"""Pre-training diagnostics for the abstraction transition dataset.

This script answers the boring-but-dangerous questions before fitting models:
class balance, source balance, split leakage, event coverage, and whether the
current rows contain temporal/history features.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import game_splits
from abstraction_dataset_io import ACTION_EFFECT_TARGETS, ACTIONS, load_dataset
from abstraction_labels import MACRO_LABELS, MACRO_SCORE_NAMES
from extract_state_abstractions import FEATURE_SCHEMA

DEFAULT_DATASET = "training/abstraction_dataset.jsonl"
DEFAULT_OUT = "diagnostics/learned_diagnostics/dataset_analysis.json"
HISTORY_KEYS = ("last_action", "last_two_actions", "action_repeat_count")


def _pct(count: int, total: int) -> float:
    return round(100.0 * float(count) / float(total), 2) if total else 0.0


def _ordered_counts(counter: Counter[str], order: Sequence[str] | None = None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if order:
        for key in order:
            if counter.get(key, 0):
                out[key] = int(counter[key])
    for key, value in counter.most_common():
        if key not in out:
            out[str(key)] = int(value)
    return out


def _with_pct(counter: Counter[str], total: int, order: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    return [
        {"name": key, "count": count, "pct": _pct(count, total)}
        for key, count in _ordered_counts(counter, order).items()
    ]


def _event_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "level_up": sum(1 for row in rows if bool(row.get("level_up"))),
        "game_over": sum(1 for row in rows if bool(row.get("game_over"))),
        "future_level_up": sum(1 for row in rows if bool(row.get("future_level_up"))),
        "future_level_up_within_5": sum(
            1 for row in rows if _future_level_within(row, 5)
        ),
        "future_level_up_within_10": sum(
            1 for row in rows if _future_level_within(row, 10)
        ),
        "future_game_over_soon": sum(1 for row in rows if bool(row.get("future_game_over_soon"))),
        "unchanged_grid": sum(1 for row in rows if int(row.get("changed_cells", 0)) == 0),
        "near_no_op": sum(1 for row in rows if float(row.get("no_op", 0.0)) >= 0.999),
    }


def _future_level_within(row: Mapping[str, Any], horizon: int) -> bool:
    key = f"future_level_up_within_{horizon}"
    if key in row:
        return bool(row.get(key))
    steps_to = int(row.get("steps_to_level_up", -1))
    return 0 <= steps_to <= horizon


def _episode_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = {
        (
            str(row.get("game_id", "")),
            str(row.get("episode_source", "")),
            int(row.get("episode_index", -1)),
        )
        for row in rows
    }
    return len(keys)


def _nested_counts(
    rows: Iterable[Mapping[str, Any]],
    outer_key: str,
    inner_key: str,
    *,
    inner_order: Sequence[str] | None = None,
) -> Dict[str, Dict[str, int]]:
    nested: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        nested[str(row.get(outer_key, "UNKNOWN"))][str(row.get(inner_key, "UNKNOWN"))] += 1
    return {
        outer: _ordered_counts(counter, inner_order)
        for outer, counter in sorted(nested.items())
    }


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        split = game_splits.split_for_game(str(row.get("game_id", ""))) or "unknown"
        counts[split] += 1
    return counts


def _target_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for target in ACTION_EFFECT_TARGETS:
        values = [
            float(row.get("delta_features", {}).get(target, 0.0))
            for row in rows
        ]
        if not values:
            stats[target] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "nonzero_count": 0,
                "nonzero_pct": 0.0,
            }
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        nonzero = sum(1 for value in values if abs(value) > 1e-9)
        stats[target] = {
            "mean": round(mean, 4),
            "std": round(math.sqrt(var), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "nonzero_count": int(nonzero),
            "nonzero_pct": _pct(nonzero, len(values)),
        }
    return stats


def _macro_score_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for target in MACRO_SCORE_NAMES:
        values = [
            float(row.get("macro_scores", {}).get(target, 0.0))
            if isinstance(row.get("macro_scores"), dict)
            else 0.0
            for row in rows
        ]
        if not values:
            stats[target] = {"mean": 0.0, "std": 0.0, "positive_pct": 0.0}
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        positive = sum(1 for value in values if value > 1e-9)
        stats[target] = {
            "mean": round(mean, 4),
            "std": round(math.sqrt(var), 4),
            "positive_pct": _pct(positive, len(values)),
        }
    return stats


def _action_change_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("action", "UNKNOWN"))].append(row)
    out: Dict[str, Dict[str, float]] = {}
    for action in ACTIONS:
        items = grouped.get(action, [])
        if not items:
            continue
        unchanged = sum(1 for row in items if int(row.get("changed_cells", 0)) == 0)
        mean_changed = sum(float(row.get("changed_cells", 0)) for row in items) / len(items)
        out[action] = {
            "count": int(len(items)),
            "unchanged_count": int(unchanged),
            "unchanged_pct": _pct(unchanged, len(items)),
            "mean_changed_cells": round(mean_changed, 4),
        }
    return out


def _macro_entropy(counter: Counter[str], total: int) -> Dict[str, float]:
    if total <= 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0}
    probs = [count / total for count in counter.values() if count > 0]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    max_entropy = math.log(max(1, len(MACRO_LABELS)), 2)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return {"entropy": round(entropy, 4), "normalized_entropy": round(normalized, 4)}


def _history_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows_with_container = sum(1 for row in rows if isinstance(row.get("history_features"), dict))
    top_level_counts = {
        key: sum(1 for row in rows if key in row)
        for key in HISTORY_KEYS
    }
    has_any = rows_with_container > 0 or any(count > 0 for count in top_level_counts.values())
    return {
        "present": has_any,
        "rows_with_history_features": rows_with_container,
        "top_level_key_counts": top_level_counts,
        "recommended_minimal_keys": list(HISTORY_KEYS),
    }


def _warnings(summary: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    total = int(summary["total_rows"])
    macro_distribution = summary["macro_label_distribution"]
    if macro_distribution:
        largest = max(macro_distribution, key=lambda item: item["count"])
        if largest["pct"] >= 75.0:
            warnings.append(
                f"Macro label skew: {largest['name']} is {largest['pct']}% of rows."
            )
    missing_macros = [
        label
        for label in MACRO_LABELS
        if label not in {item["name"] for item in macro_distribution}
    ]
    if missing_macros:
        warnings.append(f"Missing macro labels: {', '.join(missing_macros)}.")
    sparse_macros = [
        item["name"]
        for item in macro_distribution
        if item["count"] < max(50, int(0.005 * total))
    ]
    if sparse_macros:
        warnings.append(f"Sparse macro labels: {', '.join(sparse_macros)}.")
    events = summary["event_counts"]
    if events["level_up"] == 0:
        warnings.append("No level_up transitions found.")
    if events["game_over"] == 0:
        warnings.append("No game_over transitions found.")
    if not summary["history_features"]["present"]:
        warnings.append("No explicit history features found in rows.")
    if not any(v["positive_pct"] > 0.0 for v in summary["macro_score_stats"].values()):
        warnings.append("No macro_scores found; Model C-score will fall back to legacy labels.")
    ar25_rows = summary["rows_by_split"].get("ar25", 0)
    if ar25_rows and _pct(ar25_rows, total) > 10.0:
        warnings.append(f"ar25 is {_pct(ar25_rows, total)}% of rows; keep it out of final training eval.")
    return warnings


def analyze(dataset_path: str) -> Dict[str, Any]:
    rows = load_dataset(dataset_path).rows
    total = len(rows)
    source_counts = Counter(str(row.get("episode_source", "UNKNOWN")) for row in rows)
    macro_counts = Counter(str(row.get("macro_label", "UNKNOWN")) for row in rows)
    action_counts = Counter(str(row.get("action", "UNKNOWN")) for row in rows)
    game_counts = Counter(str(row.get("game_id", "UNKNOWN")) for row in rows)
    split_counts = _split_counts(rows)

    summary: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "total_rows": total,
        "episode_count": _episode_count(rows),
        "game_count": len(game_counts),
        "feature_dim": len(FEATURE_SCHEMA),
        "rows_by_source": _ordered_counts(source_counts),
        "rows_by_split": _ordered_counts(split_counts, ["seen", "unseen", "ar25", "unknown"]),
        "rows_by_game": _ordered_counts(game_counts),
        "action_distribution": _with_pct(action_counts, total, ACTIONS),
        "macro_label_distribution": _with_pct(macro_counts, total, MACRO_LABELS),
        "macro_entropy": _macro_entropy(macro_counts, total),
        "macro_by_source": _nested_counts(rows, "episode_source", "macro_label", inner_order=MACRO_LABELS),
        "macro_by_action": _nested_counts(rows, "action", "macro_label", inner_order=MACRO_LABELS),
        "event_counts": _event_counts(rows),
        "action_effect_target_stats": _target_stats(rows),
        "macro_score_stats": _macro_score_stats(rows),
        "action_change_summary": _action_change_summary(rows),
        "history_features": _history_report(rows),
    }
    summary["warnings"] = _warnings(summary)
    return summary


def _print_distribution(title: str, items: Sequence[Mapping[str, Any]]) -> None:
    print(title)
    for item in items:
        print(f"  {item['name']:28s} {item['count']:7d}  {item['pct']:6.2f}%")


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"Dataset: {summary['dataset']}")
    print(
        f"Rows: {summary['total_rows']} | episodes: {summary['episode_count']} | "
        f"games: {summary['game_count']} | features: {summary['feature_dim']}"
    )
    print()
    _print_distribution("Macro label distribution", summary["macro_label_distribution"])
    entropy = summary["macro_entropy"]
    print(
        f"  entropy={entropy['entropy']} "
        f"normalized={entropy['normalized_entropy']}"
    )
    print()
    _print_distribution("Action distribution", summary["action_distribution"])
    print()
    print("Rows by source")
    for source, count in summary["rows_by_source"].items():
        print(f"  {source:28s} {count:7d}  {_pct(count, summary['total_rows']):6.2f}%")
    print()
    print("Rows by split")
    for split, count in summary["rows_by_split"].items():
        print(f"  {split:28s} {count:7d}  {_pct(count, summary['total_rows']):6.2f}%")
    print()
    print("Event counts")
    for name, count in summary["event_counts"].items():
        print(f"  {name:28s} {count:7d}  {_pct(count, summary['total_rows']):6.2f}%")
    print()
    print("Action no-op summary")
    for action, stats in summary["action_change_summary"].items():
        print(
            f"  {action:28s} unchanged={stats['unchanged_pct']:6.2f}% "
            f"mean_changed={stats['mean_changed_cells']:.2f}"
        )
    print()
    print("Model A target stats")
    for target, stats in summary["action_effect_target_stats"].items():
        print(
            f"  {target:42s} mean={stats['mean']:+8.4f} std={stats['std']:8.4f} "
            f"min={stats['min']:+8.4f} max={stats['max']:+8.4f} "
            f"nonzero={stats['nonzero_pct']:6.2f}%"
        )
    if any(v["positive_pct"] > 0.0 for v in summary["macro_score_stats"].values()):
        print()
        print("Macro score stats")
        for target, stats in summary["macro_score_stats"].items():
            print(
                f"  {target:28s} mean={stats['mean']:8.4f} "
                f"std={stats['std']:8.4f} positive={stats['positive_pct']:6.2f}%"
            )
    print()
    history = summary["history_features"]
    print(f"History features present: {history['present']}")
    if not history["present"]:
        print(f"  recommended: {', '.join(history['recommended_minimal_keys'])}")
    if summary["warnings"]:
        print()
        print("Warnings")
        for warning in summary["warnings"]:
            print(f"  - {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze abstraction dataset balance and coverage.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = analyze(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
