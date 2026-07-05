"""Deduplicate, merge and prioritize M2 hypotheses."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .metric_registry import is_metric_measurable
from .schema import (
    FrontierConditionedHypothesis,
    SourceGenerationAudit,
)


def dedupe_key(hypothesis: FrontierConditionedHypothesis) -> tuple[str, str, str, str, str]:
    return (
        hypothesis.game_id,
        hypothesis.frontier_context_id,
        hypothesis.candidate_action,
        hypothesis.hypothesis_family,
        hypothesis.predicted_metric,
    )


def merge_hypotheses(
    hypotheses: Sequence[FrontierConditionedHypothesis],
    *,
    frontiers_by_request_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[FrontierConditionedHypothesis, ...]:
    by_key: dict[tuple[str, str, str, str, str], FrontierConditionedHypothesis] = {}
    for hypothesis in hypotheses:
        key = dedupe_key(hypothesis)
        if key not in by_key:
            by_key[key] = hypothesis
            continue
        by_key[key] = merge_two_hypotheses(by_key[key], hypothesis)

    merged = []
    for hypothesis in by_key.values():
        frontier = (frontiers_by_request_id or {}).get(hypothesis.source_request_id, {})
        score = priority_score(hypothesis, frontier_request=frontier)
        audit = replace(
            hypothesis.source_generation,
            priority_score=score,
            priority_score_counted_as_support=False,
        )
        merged.append(replace(hypothesis, source_generation=audit, support=0))
    return tuple(sorted(merged, key=lambda item: (-item.source_generation.priority_score, dedupe_key(item))))


def merge_two_hypotheses(
    left: FrontierConditionedHypothesis,
    right: FrontierConditionedHypothesis,
) -> FrontierConditionedHypothesis:
    audit = SourceGenerationAudit(
        sources=tuple(_dedupe((*left.source_generation.sources, *right.source_generation.sources))),
        raw_proposal_ids=tuple(
            _dedupe(
                (
                    *left.source_generation.raw_proposal_ids,
                    *right.source_generation.raw_proposal_ids,
                )
            )
        ),
        rationales=tuple(
            _dedupe(
                (
                    *left.source_generation.rationales,
                    *right.source_generation.rationales,
                )
            )
        ),
        normalization_warnings=tuple(
            _dedupe(
                (
                    *left.source_generation.normalization_warnings,
                    *right.source_generation.normalization_warnings,
                )
            )
        ),
        priority_score=max(
            float(left.source_generation.priority_score),
            float(right.source_generation.priority_score),
        ),
        priority_score_counted_as_support=False,
    )
    return replace(left, source_generation=audit, support=0)


def priority_score(
    hypothesis: FrontierConditionedHypothesis,
    *,
    frontier_request: Mapping[str, Any] | None = None,
) -> float:
    frontier = frontier_request or {}
    score = 0.0
    if hypothesis.frontier_reason == "confirmed_skill_blocked_by_failed_precondition":
        score += 2.0
    blocked_skill = str(frontier.get("blocked_skill", ""))
    failed_precondition = str(frontier.get("failed_precondition", ""))
    if blocked_skill and hypothesis.candidate_action != blocked_skill:
        score += 1.5
    if is_metric_measurable(hypothesis.predicted_metric):
        score += 1.0
    else:
        score -= 3.0
    if hypothesis.falsification.metric:
        score += 1.0
    snapshot = hypothesis.context_snapshot
    if snapshot.available_actions and snapshot.live_state_signature is not None:
        score += 1.0
    if "world_model" in set(hypothesis.source_generation.sources):
        score += 0.5
    if len(set(hypothesis.source_generation.sources)) > 1:
        score += 0.5
    if (
        blocked_skill
        and failed_precondition
        and hypothesis.candidate_action == blocked_skill
    ):
        score -= 2.0
    return round(score, 4)


def assign_stable_hypothesis_ids(
    hypotheses: Sequence[FrontierConditionedHypothesis],
) -> tuple[FrontierConditionedHypothesis, ...]:
    counters: dict[str, int] = {}
    result = []
    for hypothesis in hypotheses:
        context = hypothesis.frontier_context_id or "unknown"
        counters[context] = counters.get(context, 0) + 1
        result.append(
            replace(
                hypothesis,
                hypothesis_id=f"m2::{context}::h{counters[context]:03d}",
                support=0,
            )
        )
    return tuple(result)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
