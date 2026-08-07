"""SAGE.T9.3e efficient bounded confirmation and T9.4 authorization gate."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import bounded_live_gate_common as gate_common
from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6j_r3 as repair_r3
from . import compact_bounded_v9_3c as compact
from . import fast_bounded_v9_3d as fast
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v7 as live_i
from . import reachability_audit_v9 as t9_0
from . import trajectory_planning_v9_2 as t9_2
from .contracts import AbstractState, ObservedTransition
from .controller import SageTConfig
from .decision import CandidateSequence
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .replay_gate import fast_panel_from_binding_pair
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import ProgramAssembler
from .terminal_calibration_v9 import T9_1_POLICIES

FORMAT_VERSION = "sage-t9.3e-efficient-bounded-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_3e_efficient_bounded_manifest.json"
)
DEFAULT_PARENT_REPORT = fast.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "bounded_v9_3e"
EFFICIENT_CAPS: Mapping[str, int] = {
    "maximum_programs": 16,
    "maximum_sequences": 6,
    "maximum_particles_per_decision": 2,
    "ordinary_horizon": 3,
    "maximum_structural_macros": 4,
    "maximum_executor_cache_entries": 128,
}


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "bounded_live_gate_common.py": v86c._file_sha256(
            directory / "bounded_live_gate_common.py"
        ),
        "efficient_bounded_v9_3e.py": v86c._file_sha256(
            directory / "efficient_bounded_v9_3e.py"
        ),
    }


def _load_parent_report(
    path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3d report checksum mismatch")
    if report.get("status") != "T9_3D_FAILED_CLOSED":
        raise ValueError("T9.3e requires the failed-closed T9.3d gate")
    checks = dict(report.get("checks", {}))
    if {name for name, passed in checks.items() if not passed} != {
        "observation_p95"
    }:
        raise ValueError("T9.3d must have failed only observation latency")
    metrics = dict(report.get("metrics", {}))
    if (
        int(metrics.get("levels_completed", 0)) < 1
        or int(metrics.get("game_over_delta", 1)) > 0
        or int(metrics.get("bounded_actions", 0)) < 120
    ):
        raise ValueError("T9.3d lacks complete safe-progress evidence")
    if report.get("t9_4_authorized") is not False:
        raise ValueError("T9.3d did not remain fail closed")
    return report


def _winning_prefix_audit(
    caps: Mapping[str, int],
    *,
    terminal_policy: str,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    provisional = {
        "controller_caps": dict(caps),
        "selected_terminal_policy": terminal_policy,
        "authority": dict(authority),
    }
    pairs = load_pairs(t9_0.DEFAULT_SHARD_DIR, t9_0.SOURCE_GAMES)
    paths = t9_0.winner_paths(pairs)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    rows = []
    for root_key, winning_paths in sorted(paths.items()):
        winning_path = winning_paths[0]
        root_pairs = grouped[root_key]
        pairs_by_path = {pair.path: pair for pair in root_pairs}
        root = pairs_by_path[""]
        root_panel = fast_panel_from_binding_pair(root)
        controller = build_controller(provisional)
        history: list[ObservedTransition] = []
        names = tuple(sorted({arm.action.action_name for arm in root_panel.arms}))

        def assemble(
            *,
            seed: bool,
            state: AbstractState,
            controller: EfficientBoundedController = controller,
            names: tuple[str, ...] = names,
            history: list[ObservedTransition] = history,
        ) -> None:
            proposal = controller.proposer.propose(
                available_actions=names,
                transitions=tuple(history),
            )
            programs = controller.assembler.assemble(
                proposal.fragments,
                available_actions=names,
            )
            if seed:
                controller.posterior.seed(programs, initial_state=state)
            else:
                controller.posterior.add_programs(programs, initial_state=state)

        assemble(seed=True, state=root_panel.state)
        for event in root.context:
            evidence = t9_0._context_transition(event, root_panel.state)
            controller.posterior.observe(evidence)
            controller.terminal_calibrator.observe(evidence)
            history.append(evidence)
            assemble(seed=False, state=root_panel.state)

        prefix = ""
        for _ in winning_path:
            pair = pairs_by_path[prefix]
            panel = fast_panel_from_binding_pair(pair)
            legal = tuple(arm.action for arm in panel.arms)
            remaining = t9_0._remaining_actions(
                pairs_by_path,
                current_path=prefix,
                winning_path=winning_path,
            )
            expected = CandidateSequence(remaining, source="oracle")
            macros = t9_2.structural_macros(
                panel.state,
                legal,
                maximum=int(caps["maximum_structural_macros"]),
            )
            started = time.perf_counter()
            decision = controller.decision_engine.decide(
                controller.posterior,
                panel.state,
                legal,
                memory_macros=macros,
            )
            latency = (time.perf_counter() - started) * 1000.0
            match = next(
                (
                    item
                    for item in decision.assessments
                    if item.candidate.key == expected.key
                ),
                None,
            )
            chosen = decision.chosen
            rows.append(
                {
                    "generated": match is not None,
                    "rank": next(
                        (
                            index
                            for index, item in enumerate(
                                decision.assessments,
                                start=1,
                            )
                            if item.candidate.key == expected.key
                        ),
                        None,
                    ),
                    "correct_first": bool(
                        chosen is not None
                        and chosen.first_action.key == remaining[0].key
                    ),
                    "latency_ms": latency,
                }
            )
            symbol = winning_path[len(prefix)]
            observed = panel.arms[0 if symbol == "L" else 1]
            controller.posterior.observe(observed)
            controller.terminal_calibrator.observe(observed)
            history.append(observed)
            controller.executor.clear_cache()
            assemble(seed=False, state=observed.state_after)
            prefix += symbol

    result = {
        "prefixes": len(rows),
        "exact_sequence_generated": sum(row["generated"] for row in rows),
        "exact_sequence_top8": sum(
            row["rank"] is not None and int(row["rank"]) <= 8 for row in rows
        ),
        "correct_first_action": sum(row["correct_first"] for row in rows),
        "decision_p95_ms": fast.r1._quantile(
            [float(row["latency_ms"]) for row in rows], 0.95
        ),
    }
    result["passed"] = bool(
        result["prefixes"] == 9
        and result["exact_sequence_generated"] == 9
        and result["exact_sequence_top8"] == 9
        and result["correct_first_action"] == 9
    )
    return result


def freeze_manifest(
    *, output_path: str | Path = DEFAULT_MANIFEST_PATH
) -> dict[str, Any]:
    parent = fast.load_manifest()
    parent_report = _load_parent_report()
    audit = _winning_prefix_audit(
        EFFICIENT_CAPS,
        terminal_policy=str(parent["selected_terminal_policy"]),
        authority=parent["authority"],
    )
    if not audit["passed"]:
        raise ValueError("efficient beam failed the frozen winning-prefix audit")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_3E_EFFICIENT_BOUNDED_SOURCE_TRAIN",
        "frozen_at": "2026-08-06",
        "parent_t9_3d_manifest_checksum": parent["manifest_checksum"],
        "parent_t9_3d_report_checksum": parent_report["report_checksum"],
        "code_sha256": _code_hashes(),
        "controller_caps": dict(EFFICIENT_CAPS),
        "winning_prefix_audit": {
            key: value for key, value in audit.items() if key != "rows"
        },
        "selected_terminal_policy": parent["selected_terminal_policy"],
        "source_train_games": list(parent["source_train_games"]),
        "seeds": list(parent["seeds"]),
        "resets": int(parent["resets"]),
        "action_budget_per_reset": int(parent["action_budget_per_reset"]),
        "runtime": dict(parent["runtime"]),
        "authority": dict(parent["authority"]),
        "registered_changes": [
            "program cap 32 to 16 after winning-prefix audit",
            "sequence beam 8 to 6 after 9/9 winning-prefix audit",
            "decision particles 4 to 2 after 9/9 winning-prefix audit",
            "structural macro cap 8 to 4 after 9/9 winning-prefix audit",
            "executor cache 256 to 128 entries",
        ],
        "gate": {
            **dict(parent["gate"]),
            "maximum_peak_executor_cache_entries": 128,
        },
        "firewall": dict(parent["firewall"]),
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    v86c._write_json(
        Path(output_path).with_name("sage_t9_3e_winning_prefix_audit.json"),
        audit,
    )
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3e manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.3e manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_3E_EFFICIENT_BOUNDED_SOURCE_TRAIN":
        raise ValueError("T9.3e manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.3e code drifted")
    if payload.get("controller_caps") != dict(EFFICIENT_CAPS):
        raise ValueError("T9.3e caps drifted")
    if not bool(payload.get("winning_prefix_audit", {}).get("passed")):
        raise ValueError("T9.3e winning-prefix audit did not pass")
    if payload.get("parent_t9_3d_manifest_checksum") != fast.load_manifest().get(
        "manifest_checksum"
    ):
        raise ValueError("T9.3d manifest drifted")
    if payload.get("parent_t9_3d_report_checksum") != _load_parent_report().get(
        "report_checksum"
    ):
        raise ValueError("T9.3d report drifted")
    firewall = payload.get("firewall", {})
    if any(
        bool(firewall.get(key))
        for key in (
            "source_validation_opened",
            "ar25_opened",
            "holdout_opened",
            "active_authority",
        )
    ):
        raise ValueError("T9.3e firewall is open")
    return payload


class EfficientBoundedController(compact.CompactBoundedController):
    """Winning-prefix-equivalent controller under the efficient budget."""


def build_controller(manifest: Mapping[str, Any]) -> EfficientBoundedController:
    caps = manifest["controller_caps"]
    executor = StructuralRoleProgramExecutor(
        maximum_cache_entries=int(caps["maximum_executor_cache_entries"])
    )
    t7 = v86.load_t7_manifest(verify_code=True)
    posterior_config = t7["posterior"]
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=T8_6G_POLICIES[live_i.SELECTED_POLICY].with_repair_v2(),
        maximum_particles=min(
            int(posterior_config["maximum_particles"]),
            int(caps["maximum_programs"]),
        ),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(
            posterior_config["unknown_coverage_penalty"]
        ),
        repair_ess_threshold=float(
            posterior_config["repair_ess_threshold"]
        ),
        repair_log_likelihood_threshold=float(
            posterior_config["repair_log_likelihood_threshold"]
        ),
        maximum_repair_contexts=repair_r3.MAXIMUM_REPAIR_CONTEXTS,
    )
    authority = manifest["authority"]
    return EfficientBoundedController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="bounded",
            counterfactual_gate_passed=True,
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(
                caps["maximum_particles_per_decision"]
            ),
            ordinary_horizon=int(caps["ordinary_horizon"]),
            bounded_maximum_interventions_per_reset=int(
                authority["maximum_interventions_per_reset"]
            ),
            bounded_maximum_terminal_risk=float(
                authority["maximum_marginal_terminal_risk"]
            ),
        ),
        terminal_policy=T9_1_POLICIES[str(manifest["selected_terminal_policy"])],
        maximum_structural_macros=int(caps["maximum_structural_macros"]),
        repeat_bonus_per_extra_action=0.35,
        strong_surprise_threshold=float(authority["strong_surprise_lockout_threshold"]),
    )


def run_pilot(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    return gate_common.run_bounded_gate(
        manifest=manifest,
        runtime=live_base.runtime_capabilities(),
        controller_builder=build_controller,
        format_version=FORMAT_VERSION,
        passed_status="T9_3E_PASSED",
        failed_status="T9_3E_FAILED_CLOSED",
        environments_dir=environments_dir,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(output_path=args.manifest)
    else:
        result = run_pilot(
            manifest_path=args.manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "T9_3E_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EFFICIENT_CAPS",
    "FORMAT_VERSION",
    "EfficientBoundedController",
    "build_controller",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_pilot",
]
