"""M2 frontier-conditioned hypothesis expansion runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.a40.frontier_handoff_requests import (
    DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
)

from .frontier_intake import (
    DEFAULT_M2_FRONTIER_INTAKE_OUTPUT_PATH,
    run_frontier_intake,
    run_frontier_intake_from_payload,
    write_frontier_intake,
)
from .heuristic_generator import generate_heuristic_proposals
from .hypothesis_merger import assign_stable_hypothesis_ids, merge_hypotheses
from .normalizer import normalize_raw_proposals
from .schema import (
    M2_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    FrontierConditionedHypothesis,
    RawHypothesisProposal,
    RejectedProposal,
)
from .testability_compiler import (
    DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    build_m3_requests_payload,
    write_m3_requests_payload,
)
from .validators import validate_hypothesis


DEFAULT_M2_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "frontier_conditioned_hypotheses.json"
)
DEFAULT_M2_GENERATION_AUDIT_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "generation_audit.json"
)


def run_frontier_conditioned_hypotheses(
    *,
    frontier_path: str | Path = DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
    generator_mode: str = "heuristic_only",
    llm_enabled: bool = False,
    world_model_enabled: bool = False,
) -> Dict[str, Any]:
    intake_payload = run_frontier_intake(frontier_path=frontier_path)
    return build_frontier_conditioned_outputs(
        intake_payload=intake_payload,
        input_frontier_path=str(frontier_path),
        generator_mode=generator_mode,
        llm_enabled=llm_enabled,
        world_model_enabled=world_model_enabled,
    )


def run_frontier_conditioned_hypotheses_from_payload(
    frontier_payload: Mapping[str, Any],
    *,
    input_frontier_path: str,
    generator_mode: str = "heuristic_only",
    llm_enabled: bool = False,
    world_model_enabled: bool = False,
) -> Dict[str, Any]:
    intake_payload = run_frontier_intake_from_payload(
        frontier_payload,
        input_frontier_path=input_frontier_path,
    )
    return build_frontier_conditioned_outputs(
        intake_payload=intake_payload,
        input_frontier_path=input_frontier_path,
        generator_mode=generator_mode,
        llm_enabled=llm_enabled,
        world_model_enabled=world_model_enabled,
    )


def build_frontier_conditioned_outputs(
    *,
    intake_payload: Mapping[str, Any],
    input_frontier_path: str,
    generator_mode: str,
    llm_enabled: bool,
    world_model_enabled: bool,
) -> Dict[str, Any]:
    frontiers = tuple(
        dict(row)
        for row in intake_payload.get("frontier_requests", []) or []
        if isinstance(row, Mapping)
    )
    frontiers_by_request_id = {
        str(frontier.get("request_id", "")): frontier for frontier in frontiers
    }
    raw_proposals = list(generate_heuristic_proposals(frontiers))
    raw_proposals.extend(
        _optional_mock_llm_proposals(
            frontiers,
            enabled=llm_enabled or "mock_llm" in generator_mode,
        )
    )
    raw_proposals.extend(
        _optional_mock_world_model_proposals(
            frontiers,
            enabled=world_model_enabled or "mock_world_model" in generator_mode,
        )
    )
    normalized, rejected = normalize_raw_proposals(
        raw_proposals,
        frontiers_by_request_id=frontiers_by_request_id,
    )
    merged = merge_hypotheses(
        normalized,
        frontiers_by_request_id=frontiers_by_request_id,
    )
    hypotheses = assign_stable_hypothesis_ids(merged)
    hypothesis_payload = build_hypothesis_payload(
        frontiers=frontiers,
        hypotheses=hypotheses,
        raw_proposals=raw_proposals,
        rejected=rejected,
        input_frontier_path=input_frontier_path,
        generator_mode=generator_mode,
        llm_enabled=llm_enabled,
        world_model_enabled=world_model_enabled,
    )
    m3_payload = build_m3_requests_payload(
        hypotheses,
        source_hypothesis_path=str(DEFAULT_M2_HYPOTHESES_OUTPUT_PATH),
    )
    audit_payload = build_generation_audit_payload(
        frontiers=frontiers,
        raw_proposals=raw_proposals,
        normalized=normalized,
        hypotheses=hypotheses,
        rejected=rejected,
        generator_mode=generator_mode,
        llm_enabled=llm_enabled,
        world_model_enabled=world_model_enabled,
    )
    return {
        "intake_payload": dict(intake_payload),
        "hypothesis_payload": hypothesis_payload,
        "m3_payload": m3_payload,
        "generation_audit": audit_payload,
    }


def build_hypothesis_payload(
    *,
    frontiers: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[FrontierConditionedHypothesis],
    raw_proposals: Sequence[RawHypothesisProposal],
    rejected: Sequence[RejectedProposal],
    input_frontier_path: str,
    generator_mode: str,
    llm_enabled: bool,
    world_model_enabled: bool,
) -> Dict[str, Any]:
    by_request_id: dict[str, list[FrontierConditionedHypothesis]] = {}
    for hypothesis in hypotheses:
        by_request_id.setdefault(hypothesis.source_request_id, []).append(hypothesis)

    batches = []
    for frontier in frontiers:
        request_id = str(frontier.get("request_id", ""))
        candidates = by_request_id.get(request_id, [])
        batches.append(
            {
                "frontier_context_id": str(frontier.get("frontier_context_id", "")),
                "frontier_reason": str(frontier.get("reason", "")),
                "source_request_id": request_id,
                "candidate_hypotheses": [
                    hypothesis.to_dict() for hypothesis in candidates
                ],
            }
        )

    testable = [
        hypothesis for hypothesis in hypotheses if hypothesis.testability.testable
    ]
    blocked = [
        hypothesis for hypothesis in hypotheses if not hypothesis.testability.testable
    ]
    invalid = [
        hypothesis.hypothesis_id
        for hypothesis in hypotheses
        if not validate_hypothesis(hypothesis).valid
    ]
    return {
        "config": {
            "input_frontier_path": input_frontier_path,
            "generator_mode": generator_mode,
            "llm_enabled": bool(llm_enabled),
            "world_model_enabled": bool(world_model_enabled),
            "schema_version": M2_SCHEMA_VERSION,
        },
        "hypothesis_batches": batches,
        "summary": {
            "frontier_requests_consumed": len(frontiers),
            "hypothesis_batches": len(frontiers),
            "raw_proposals_generated": len(raw_proposals),
            "hypotheses_generated": len(hypotheses),
            "testable_hypotheses": len(testable),
            "blocked_not_testable_hypotheses": len(blocked),
            "experiment_requests_ready_for_m3": len(testable),
            "final_invalid_hypotheses": len(invalid),
            "rejected_raw_proposals": len(rejected),
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
    }


def build_generation_audit_payload(
    *,
    frontiers: Sequence[Mapping[str, Any]],
    raw_proposals: Sequence[RawHypothesisProposal],
    normalized: Sequence[FrontierConditionedHypothesis],
    hypotheses: Sequence[FrontierConditionedHypothesis],
    rejected: Sequence[RejectedProposal],
    generator_mode: str,
    llm_enabled: bool,
    world_model_enabled: bool,
) -> Dict[str, Any]:
    return {
        "config": {
            "generator_mode": generator_mode,
            "llm_enabled": bool(llm_enabled),
            "world_model_enabled": bool(world_model_enabled),
            "schema_version": "m2.audit.v1",
        },
        "summary": {
            "frontier_requests_consumed": len(frontiers),
            "raw_proposals_seen": len(raw_proposals),
            "normalized_hypotheses": len(normalized),
            "deduplicated_hypotheses": len(hypotheses),
            "rejected_raw_proposals": len(rejected),
            "merged_sources_recorded": any(
                len(set(h.source_generation.sources)) > 1 for h in hypotheses
            ),
            "priority_score_counted_as_support": any(
                h.source_generation.priority_score_counted_as_support
                for h in hypotheses
            ),
            "support_unchanged_at_zero": all(h.support == 0 for h in hypotheses),
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "rejected_proposals": [item.to_dict() for item in rejected],
    }


def write_frontier_conditioned_outputs(
    outputs: Mapping[str, Any],
    *,
    intake_out: str | Path = DEFAULT_M2_FRONTIER_INTAKE_OUTPUT_PATH,
    hypotheses_out: str | Path = DEFAULT_M2_HYPOTHESES_OUTPUT_PATH,
    m3_out: str | Path = DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
    generation_audit_out: str | Path = DEFAULT_M2_GENERATION_AUDIT_OUTPUT_PATH,
) -> None:
    write_frontier_intake(outputs["intake_payload"], intake_out)
    write_hypothesis_payload(outputs["hypothesis_payload"], hypotheses_out)
    write_m3_requests_payload(outputs["m3_payload"], m3_out)
    write_generation_audit_payload(outputs["generation_audit"], generation_audit_out)


def write_hypothesis_payload(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_HYPOTHESES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_generation_audit_payload(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_GENERATION_AUDIT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _optional_mock_llm_proposals(
    frontiers: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> tuple[RawHypothesisProposal, ...]:
    if not enabled:
        return ()
    from .mock_llm_generator import MockLLMGenerator

    generator = MockLLMGenerator()
    proposals: list[RawHypothesisProposal] = []
    for frontier in frontiers:
        proposals.extend(generator.generate(frontier_request=frontier))
    return tuple(proposals)


def _optional_mock_world_model_proposals(
    frontiers: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> tuple[RawHypothesisProposal, ...]:
    if not enabled:
        return ()
    from .mock_world_model import generate_world_model_raw_proposals

    proposals: list[RawHypothesisProposal] = []
    for frontier in frontiers:
        proposals.extend(generate_world_model_raw_proposals(frontier))
    return tuple(proposals)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M2 frontier-conditioned hypotheses.",
    )
    parser.add_argument(
        "--frontiers",
        type=Path,
        default=DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_M2_HYPOTHESES_OUTPUT_PATH)
    parser.add_argument("--m3-out", type=Path, default=DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH)
    parser.add_argument(
        "--generation-audit-out",
        type=Path,
        default=DEFAULT_M2_GENERATION_AUDIT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--intake-out",
        type=Path,
        default=DEFAULT_M2_FRONTIER_INTAKE_OUTPUT_PATH,
    )
    parser.add_argument("--generator-mode", default="heuristic_only")
    parser.add_argument("--llm-enabled", action="store_true")
    parser.add_argument("--world-model-enabled", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_frontier_conditioned_hypotheses(
        frontier_path=args.frontiers,
        generator_mode=args.generator_mode,
        llm_enabled=args.llm_enabled,
        world_model_enabled=args.world_model_enabled,
    )
    write_frontier_conditioned_outputs(
        outputs,
        intake_out=args.intake_out,
        hypotheses_out=args.out,
        m3_out=args.m3_out,
        generation_audit_out=args.generation_audit_out,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "m3_output_path": str(args.m3_out),
                "summary": outputs["hypothesis_payload"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
