"""SAGE.0 known-game closed-loop integration scaffold.

The scaffold exercises the integration contract without claiming benchmark
performance. It distinguishes offline trace observations from live environment
actions, and it never treats policy outcomes as scientific confirmations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


DEFAULT_M2_FUSED_REQUESTS_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_m3_candidate_requests.json"
)
DEFAULT_M3_FUSED_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "fused_llm_wm_experiment_results.json"
)
DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH = (
    Path("diagnostics") / "m3" / "offline_frame_counterfactual_feasibility.json"
)
DEFAULT_P1_POLICY_PROBE_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_probe.json"
)
DEFAULT_P1_UTILITY_HANDOFF_PATH = (
    Path("diagnostics") / "p1" / "bp35_candidate_policy_utility_handoff.json"
)
DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH = (
    Path("diagnostics") / "sage" / "sage0_known_game_closed_loop_scaffold.json"
)
SAGE0_SCHEMA_VERSION = "sage.known_game_closed_loop_scaffold.v1"
SAGE_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_0"
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_BUDGET = 2


@dataclass(frozen=True)
class SageLiveFrame:
    """Minimal live observation exposed to SAGE.0."""

    game_id: str
    step: int
    grid: tuple[tuple[int, ...], ...]
    available_actions: tuple[str, ...]
    game_state: str = "NOT_FINISHED"
    levels_completed: int = 0
    env_state_restore_available: bool = False

    @property
    def grid_hash(self) -> str:
        payload = {
            "grid": [list(row) for row in self.grid],
            "game_state": self.game_state,
            "levels_completed": self.levels_completed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "step": int(self.step),
            "grid_hash": self.grid_hash,
            "available_actions": list(self.available_actions),
            "game_state": self.game_state,
            "levels_completed": int(self.levels_completed),
            "env_state_restore_available": self.env_state_restore_available,
            "observation_context": "live_env_context",
        }


@dataclass(frozen=True)
class SageActionDecision:
    """One candidate action selected by the scaffold."""

    step: int
    action: str
    action_args: Dict[str, Any] = field(default_factory=dict)
    decision_source: str = "candidate_policy_memory"
    selection_context: str = "live_env_context"
    candidate_policy_used: bool = True
    offline_counterfactual_allowed: bool = False
    active_counterfactual_collection_allowed: bool = True
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "action": self.action,
            "action_args": dict(self.action_args),
            "decision_source": self.decision_source,
            "selection_context": self.selection_context,
            "candidate_policy_used": self.candidate_policy_used,
            "offline_counterfactual_allowed": self.offline_counterfactual_allowed,
            "active_counterfactual_collection_allowed": (
                self.active_counterfactual_collection_allowed
            ),
            "rationale": self.rationale,
        }


class SyntheticKnownGameEnv:
    """Tiny deterministic live-env stand-in for integration tests.

    It has an internal current state, so actions are live transitions. It does
    not expose a restore API; that absence is intentional for SAGE.0.
    """

    def __init__(self, *, game_id: str = DEFAULT_GAME_ID) -> None:
        self.game_id = game_id
        self._step = 0
        self._grid = (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
        )

    def reset(self) -> SageLiveFrame:
        self._step = 0
        return self._frame()

    def step(self, action: str, action_args: Mapping[str, Any] | None = None) -> SageLiveFrame:
        action_name = str(action)
        args = dict(action_args or {})
        grid = [list(row) for row in self._grid]
        if action_name == "ACTION4":
            grid[1][0] = 4
        elif action_name == "ACTION6":
            x_value = int(args.get("x", 0) or 0)
            y_value = int(args.get("y", 0) or 0)
            grid[2][2] = 6
            grid[3][3] = (x_value + y_value) % 10
        elif action_name == "ACTION3":
            grid[0][0] = 3
        elif action_name == "ACTION1":
            grid[0][1] = 1
        self._grid = tuple(tuple(int(cell) for cell in row) for row in grid)
        self._step += 1
        return self._frame()

    def _frame(self) -> SageLiveFrame:
        return SageLiveFrame(
            game_id=self.game_id,
            step=self._step,
            grid=self._grid,
            available_actions=("ACTION1", "ACTION3", "ACTION4", "ACTION6"),
            game_state="NOT_FINISHED",
            levels_completed=0,
            env_state_restore_available=False,
        )


def run_sage0_known_game_scaffold(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    output_path: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    budget: int = DEFAULT_BUDGET,
) -> Dict[str, Any]:
    inputs = load_sage_inputs(
        m2_fused_requests_path=m2_fused_requests_path,
        m3_fused_results_path=m3_fused_results_path,
        m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
        p1_policy_probe_path=p1_policy_probe_path,
        p1_utility_handoff_path=p1_utility_handoff_path,
    )
    env = SyntheticKnownGameEnv(game_id=game_id)
    payload = execute_sage0_scaffold_loop(
        inputs=inputs,
        env=env,
        budget=budget,
        config_paths={
            "m2_fused_requests_path": str(m2_fused_requests_path),
            "m3_fused_results_path": str(m3_fused_results_path),
            "m3_counterfactual_feasibility_path": str(m3_counterfactual_feasibility_path),
            "p1_policy_probe_path": str(p1_policy_probe_path),
            "p1_utility_handoff_path": str(p1_utility_handoff_path),
        },
    )
    if output_path is not None:
        write_sage0_known_game_scaffold(payload, output_path)
    return payload


def load_sage_inputs(
    *,
    m2_fused_requests_path: str | Path,
    m3_fused_results_path: str | Path,
    m3_counterfactual_feasibility_path: str | Path,
    p1_policy_probe_path: str | Path,
    p1_utility_handoff_path: str | Path,
) -> Dict[str, Any]:
    return {
        "m2_fused_requests": _load_json(m2_fused_requests_path),
        "m3_fused_results": _load_json(m3_fused_results_path),
        "m3_counterfactual_feasibility": _load_json(
            m3_counterfactual_feasibility_path
        ),
        "p1_policy_probe": _load_json(p1_policy_probe_path),
        "p1_utility_handoff": _load_json(p1_utility_handoff_path),
    }


def execute_sage0_scaffold_loop(
    *,
    inputs: Mapping[str, Any],
    env: SyntheticKnownGameEnv,
    budget: int,
    config_paths: Mapping[str, str],
) -> Dict[str, Any]:
    frame = env.reset()
    observations: list[Dict[str, Any]] = [frame.to_dict()]
    decisions: list[Dict[str, Any]] = []
    env_steps: list[Dict[str, Any]] = []
    phase_log = [
        phase_entry(
            "live_observation",
            "observed current live frame",
            frame.to_dict(),
        ),
        phase_entry(
            "hypothesis_context_loaded",
            "loaded candidate-only M2/M3/P1 context",
            hypothesis_context_summary(inputs),
        ),
        phase_entry(
            "m3_tests_possible",
            "rankable M3 requests remain diagnostic or candidate-only",
            m3_tests_summary(inputs),
        ),
    ]

    policy_memory = dict(
        inputs.get("p1_policy_probe", {}).get("candidate_policy_memory", {}) or {}
    )
    for step_index in range(max(0, int(budget))):
        decision = select_sage_action(
            step_index=step_index,
            frame=frame,
            policy_memory=policy_memory,
        )
        decisions.append(decision.to_dict())
        phase_log.append(
            phase_entry(
                "action_candidate",
                "selected candidate action from live context",
                decision.to_dict(),
            )
        )
        after = env.step(decision.action, decision.action_args)
        step_record = build_env_step_record(
            before=frame,
            after=after,
            decision=decision,
        )
        env_steps.append(step_record)
        observations.append(after.to_dict())
        phase_log.append(
            phase_entry(
                "env_step",
                "executed action in live scaffold env",
                step_record,
            )
        )
        frame = after

    logger_contract = sage_logger_contract(inputs)
    phase_log.append(
        phase_entry(
            "scientific_log",
            "wrote candidate-only scientific log",
            logger_contract,
        )
    )
    summary = summarize_sage0_scaffold(
        inputs=inputs,
        decisions=decisions,
        env_steps=env_steps,
        logger_contract=logger_contract,
    )
    return {
        "config": {
            **dict(config_paths),
            "schema_version": SAGE0_SCHEMA_VERSION,
            "game_id": env.game_id,
            "budget": int(budget),
            "scaffold_only": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "logger_contract": logger_contract,
        "phase_log": phase_log,
        "live_observations": observations,
        "action_decisions": decisions,
        "env_steps": env_steps,
        "input_summaries": {
            "hypothesis_context": hypothesis_context_summary(inputs),
            "m3_tests": m3_tests_summary(inputs),
            "policy_context": policy_context_summary(inputs),
        },
        "summary": summary,
        "status": "UNRESOLVED",
        "truth_status": SAGE_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "policy_result_counted_as_confirmation": False,
    }


def select_sage_action(
    *,
    step_index: int,
    frame: SageLiveFrame,
    policy_memory: Mapping[str, Any],
) -> SageActionDecision:
    available = set(frame.available_actions)
    target_action = str(policy_memory.get("target_action", "ACTION6") or "ACTION6")
    repositioning_action = str(
        policy_memory.get("repositioning_action", "ACTION4") or "ACTION4"
    )
    if step_index == 0 and repositioning_action in available:
        return SageActionDecision(
            step=step_index,
            action=repositioning_action,
            action_args={},
            rationale="candidate_policy_live_repositioning_scaffold",
        )
    success_args = [
        dict(row)
        for row in policy_memory.get("known_success_args", []) or []
        if isinstance(row, Mapping)
    ]
    action_args = success_args[0] if success_args else {}
    if target_action in available:
        return SageActionDecision(
            step=step_index,
            action=target_action,
            action_args=action_args,
            rationale="candidate_policy_live_target_action_scaffold",
        )
    fallback = sorted(available - {"RESET"})
    action = fallback[0] if fallback else "NOOP"
    return SageActionDecision(
        step=step_index,
        action=action,
        action_args={},
        candidate_policy_used=False,
        rationale="candidate_policy_action_unavailable_live_fallback",
    )


def build_env_step_record(
    *,
    before: SageLiveFrame,
    after: SageLiveFrame,
    decision: SageActionDecision,
) -> Dict[str, Any]:
    return {
        "step": int(decision.step),
        "game_id": before.game_id,
        "selection_context": decision.selection_context,
        "action": decision.action,
        "action_args": dict(decision.action_args),
        "state_signature_before": before.grid_hash,
        "state_signature_after": after.grid_hash,
        "state_changed": before.grid_hash != after.grid_hash,
        "env_actions": 1,
        "live_env_context": True,
        "offline_trace_context": False,
        "offline_counterfactual_allowed": decision.offline_counterfactual_allowed,
        "active_counterfactual_collection_allowed": (
            decision.active_counterfactual_collection_allowed
        ),
        "env_state_restore_available": after.env_state_restore_available,
        "game_state_before": before.game_state,
        "game_state_after": after.game_state,
        "levels_before": before.levels_completed,
        "levels_after": after.levels_completed,
        "truth_status": SAGE_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def sage_logger_contract(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    m3_7f_summary = dict(
        inputs.get("m3_counterfactual_feasibility", {}).get("summary", {}) or {}
    )
    return {
        "offline_counterfactual_allowed": False,
        "active_counterfactual_collection_allowed": True,
        "env_state_restore_available": False,
        "blocked_capability_frontiers_logged": bool(
            m3_7f_summary.get("frontier_recommendations", 0)
        ),
        "offline_trace_context_role": "observation_diagnostic_grounding_only",
        "live_env_context_role": "only_authorized_context_for_alternative_actions",
        "policy_result_counted_as_confirmation": False,
        "m2_m3_support_counted_as_scientific_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": 0,
        "truth_status": SAGE_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def hypothesis_context_summary(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    m2_summary = dict(inputs.get("m2_fused_requests", {}).get("summary", {}) or {})
    return {
        "m2_requests_seen": int(m2_summary.get("experiment_requests", 0) or 0),
        "m2_ready_for_m3": int(m2_summary.get("ready_for_m3", 0) or 0),
        "m2_blocked_not_testable": int(m2_summary.get("blocked_not_testable", 0) or 0),
        "llm_output_counted_as_evidence": bool(
            m2_summary.get("llm_output_counted_as_evidence", False)
        ),
        "world_model_counted_as_evidence": bool(
            m2_summary.get("world_model_counted_as_evidence", False)
        ),
        "support": 0,
    }


def m3_tests_summary(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    m3_7e_summary = dict(inputs.get("m3_fused_results", {}).get("summary", {}) or {})
    m3_7f_summary = dict(
        inputs.get("m3_counterfactual_feasibility", {}).get("summary", {}) or {}
    )
    return {
        "m3_fused_requests_executed": int(
            m3_7e_summary.get("fused_requests_executed", 0) or 0
        ),
        "m3_fused_requests_skipped_blocked": int(
            m3_7e_summary.get("fused_requests_skipped_blocked", 0) or 0
        ),
        "fusion_hypothesis_routing_validated": bool(
            m3_7e_summary.get("fusion_hypothesis_routing_validated", False)
        ),
        "new_independent_terminal_risk_evidence": bool(
            m3_7e_summary.get("new_independent_terminal_risk_evidence", False)
        ),
        "counterfactual_requests_seen": int(
            m3_7f_summary.get("counterfactual_requests_seen", 0) or 0
        ),
        "offline_counterfactual_feasible": bool(
            m3_7f_summary.get("feasible_counterfactual_requests", 0)
        ),
        "blocked_capability_frontiers_logged": bool(
            m3_7f_summary.get("frontier_recommendations", 0)
        ),
        "support": 0,
    }


def policy_context_summary(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    policy_probe = dict(inputs.get("p1_policy_probe", {}) or {})
    utility = dict(inputs.get("p1_utility_handoff", {}) or {})
    memory = dict(policy_probe.get("candidate_policy_memory", {}) or {})
    return {
        "policy_memory_enabled": bool(memory.get("enabled", False)),
        "candidate_policy_status": str(
            policy_probe.get("summary", {}).get(
                "candidate_policy_status",
                utility.get("candidate_policy_status", "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"),
            )
        ),
        "agentic_utility_status": str(
            utility.get("summary", {}).get("agentic_utility_status", "")
        ),
        "a33_ready": bool(memory.get("a33_ready", False) or utility.get("a33_ready", False)),
        "policy_result_counted_as_confirmation": bool(
            policy_probe.get("candidate_policy_counted_as_confirmation", False)
            or utility.get("policy_result_counted_as_mechanistic_confirmation", False)
        ),
        "support": 0,
    }


def summarize_sage0_scaffold(
    *,
    inputs: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    env_steps: Sequence[Mapping[str, Any]],
    logger_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "closed_loop_scaffold_completed": bool(env_steps),
        "benchmark_run": False,
        "live_observations": len(env_steps) + 1 if env_steps else 1,
        "candidate_actions_selected": len(decisions),
        "env_steps_executed": len(env_steps),
        "m2_ready_for_m3": hypothesis_context_summary(inputs)["m2_ready_for_m3"],
        "m3_fused_requests_executed": m3_tests_summary(inputs)[
            "m3_fused_requests_executed"
        ],
        "offline_counterfactual_allowed": bool(
            logger_contract.get("offline_counterfactual_allowed", False)
        ),
        "active_counterfactual_collection_allowed": bool(
            logger_contract.get("active_counterfactual_collection_allowed", False)
        ),
        "env_state_restore_available": bool(
            logger_contract.get("env_state_restore_available", False)
        ),
        "blocked_capability_frontiers_logged": bool(
            logger_contract.get("blocked_capability_frontiers_logged", False)
        ),
        "offline_trace_context_is_observation_only": True,
        "live_env_context_authorizes_alternative_actions": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def phase_entry(phase: str, description: str, details: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "phase": phase,
        "description": description,
        "details": dict(details),
        "support": 0,
        "truth_status": SAGE_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def write_sage0_known_game_scaffold(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.0 known-game closed-loop scaffold.",
    )
    parser.add_argument("--m2-fused-requests", default=str(DEFAULT_M2_FUSED_REQUESTS_PATH))
    parser.add_argument("--m3-fused-results", default=str(DEFAULT_M3_FUSED_RESULTS_PATH))
    parser.add_argument(
        "--m3-counterfactual-feasibility",
        default=str(DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH),
    )
    parser.add_argument("--p1-policy-probe", default=str(DEFAULT_P1_POLICY_PROBE_PATH))
    parser.add_argument(
        "--p1-utility-handoff",
        default=str(DEFAULT_P1_UTILITY_HANDOFF_PATH),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH),
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)
    run_sage0_known_game_scaffold(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        output_path=args.out,
        game_id=args.game_id,
        budget=args.budget,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
