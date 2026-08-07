"""T8.6i structural-goal live confirmation under shadow authority."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6h as v86h
from . import calibration_gate_v8_6i as v86i
from . import live_shadow_pilot_v5 as t8_5
from . import live_shadow_pilot_v6 as t8_6_live
from .contracts import ObjectSchema
from .controller import SageTConfig
from .goal_generation_v2 import goal_progress_bridge_fragment
from .goal_generation_v3 import (
    observed_goal_trigger_roles,
    structural_goal_guard_fragments,
)
from .posterior_v8 import T8_6G_POLICIES
from .structural_roles import (
    STRUCTURAL_TARGET_ROLES,
    StructuralRoleProgramExecutor,
)
from .synthesis import (
    DeterministicFragmentProposer,
    FragmentProposal,
    ProgramAssembler,
)

FORMAT_VERSION = "sage-t8.6i-live-confirmation-v1"
DEFAULT_CONFIRMATION_MANIFEST = Path(__file__).with_name(
    "sage_t8_6i_confirmation_manifest.json"
)
DEFAULT_SELECTION_REPORT = v86i.DEFAULT_OUTPUT_DIR / "selection_report.json"
DEFAULT_T8_5_MANIFEST = Path(__file__).with_name(
    "sage_t8_5_long_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6i_live"
SELECTED_POLICY = v86i.SELECTED_POLICY


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: v86c._file_sha256(directory / name)
        for name in ("live_shadow_pilot_v6.py", "live_shadow_pilot_v7.py")
    }


def _load_selection_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6i selection report checksum mismatch")
    if report.get("status") != "READY_FOR_T8_6I_LIVE_CONFIRMATION":
        raise ValueError("T8.6i offline gate did not authorize live confirmation")
    if report.get("selected_challenger") != SELECTED_POLICY:
        raise ValueError("T8.6i selected policy drifted")
    if not bool(
        (report.get("offline_repair_v2_confirmation") or {}).get("passed")
    ):
        raise ValueError("T8.6i offline Repair V2 confirmation failed")
    if report.get("source_validation_authorized") is not False:
        raise ValueError("source-validation opened before live confirmation")
    return report


def freeze_confirmation_manifest(
    *,
    output_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    selection_manifest_path: str | Path = v86i.DEFAULT_MANIFEST_PATH,
    selection_report_path: str | Path = DEFAULT_SELECTION_REPORT,
    t8_5_manifest_path: str | Path = DEFAULT_T8_5_MANIFEST,
) -> dict[str, Any]:
    selection = v86i.load_manifest(selection_manifest_path)
    report = _load_selection_report(selection_report_path)
    live_base = t8_5.load_frozen_manifest(t8_5_manifest_path)
    repair_policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6I_LIVE_CONFIRMATION",
        "frozen_at": "2026-08-06",
        "selection_manifest_checksum": selection["manifest_checksum"],
        "selection_report_checksum": report["report_checksum"],
        "selection_scientific_checksum": report["scientific_checksum"],
        "t8_5_long_manifest_checksum": live_base["manifest_checksum"],
        "code_sha256": _code_hashes(),
        "selected_challenger": SELECTED_POLICY,
        "repair_policy": v86c._json_safe(asdict(repair_policy)),
        "source_train_games": list(live_base["source_train_games"]),
        "actions": 50,
        "actions_per_game": 25,
        "seeds": [0],
        "authority": "shadow",
        "gate": dict(selection["confirmation"]),
        "frozen_invariants": {
            "posterior": "t8_6g_minimum_kl_repair_v2",
            "goal_generator": "t8_6i_structural_goal_guard",
            "executor": "t8_6i_structural_role_executor",
            "controller_action": "baseline_materialized_action_only",
            "policies_executed": [repair_policy.name],
            "source_train_only": True,
        },
        "source_validation_authorized": False,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_confirmation_manifest(
    path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    *,
    selection_manifest_path: str | Path = v86i.DEFAULT_MANIFEST_PATH,
    selection_report_path: str | Path = DEFAULT_SELECTION_REPORT,
    t8_5_manifest_path: str | Path = DEFAULT_T8_5_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6i live manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T8.6i live manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6I_LIVE_CONFIRMATION":
        raise ValueError("T8.6i live manifest is not frozen")
    if payload.get("authority") != "shadow":
        raise ValueError("T8.6i live confirmation must remain shadow")
    selection = v86i.load_manifest(selection_manifest_path)
    report = _load_selection_report(selection_report_path)
    live_base = t8_5.load_frozen_manifest(t8_5_manifest_path)
    if payload.get("selection_manifest_checksum") != selection.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6i live selection manifest drifted")
    if payload.get("selection_report_checksum") != report.get(
        "report_checksum"
    ):
        raise ValueError("T8.6i live selection report drifted")
    if payload.get("t8_5_long_manifest_checksum") != live_base.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6i live action protocol drifted")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T8.6i live code drifted after freeze")
    expected_policy = v86c._json_safe(
        asdict(T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2())
    )
    if payload.get("repair_policy") != expected_policy:
        raise ValueError("T8.6i Repair V2 policy drifted")
    if any(
        bool(payload.get(key))
        for key in (
            "source_validation_authorized",
            "bounded_authority_authorized",
            "active_authority_authorized",
        )
    ):
        raise ValueError("T8.6i live firewall is open")
    return payload, report


class StructuralGoalFragmentProposer:
    """Controller adapter exactly matching the frozen T8.6i generator."""

    def __init__(self) -> None:
        self.base = DeterministicFragmentProposer()

    def propose(self, **kwargs: Any) -> FragmentProposal:
        proposal = self.base.propose(**kwargs)
        trigger_roles = observed_goal_trigger_roles(
            tuple(kwargs.get("transitions", ()))
        )
        if not trigger_roles:
            return proposal
        fragments = []
        for fragment in proposal.fragments:
            if fragment.kind == "schema" and isinstance(
                fragment.payload, ObjectSchema
            ):
                fragments.append(
                    replace(
                        fragment,
                        payload=ObjectSchema(
                            tuple(
                                sorted(
                                    set(fragment.payload.roles)
                                    | set(STRUCTURAL_TARGET_ROLES)
                                )
                            )
                        ),
                    )
                )
            else:
                fragments.append(fragment)
        guards = structural_goal_guard_fragments(fragments, trigger_roles)
        return FragmentProposal(
            fragments=(
                *fragments,
                *guards,
                goal_progress_bridge_fragment(),
            ),
            plan_sequences=proposal.plan_sequences,
        )


def _controller(
    *,
    caps: Mapping[str, Any],
) -> t8_5.MaterializedActionController:
    executor = StructuralRoleProgramExecutor()
    t7 = v86.load_t7_manifest(verify_code=True)
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    posterior = v86h._new_minimum_kl_posterior(
        policy,
        executor=executor,
        manifest=t7,
    )
    return t8_5.MaterializedActionController(
        executor=executor,
        posterior=posterior,
        proposer=StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(
            maximum_programs=int(caps["maximum_programs"])
        ),
        config=SageTConfig(
            mode="shadow",
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(
                caps["maximum_particles_per_decision"]
            ),
            ordinary_horizon=int(caps["ordinary_horizon"]),
        ),
    )


class StructuralLiveController:
    """Single selected posterior with the broadcast-controller interface."""

    def __init__(self, *, caps: Mapping[str, Any]) -> None:
        selected = _controller(caps=caps)
        self.selected_name = (
            T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2().name
        )
        self.controllers = {self.selected_name: selected}
        self.selected = selected
        self.posterior = selected.posterior

    @property
    def effective_mode(self):  # type: ignore[no-untyped-def]
        return self.selected.effective_mode

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        return self.selected.decide(**kwargs)

    def observe_transition(self, record: Any) -> None:
        self.selected.observe_transition(record)

    def start_branch(self, *, regime_index: int | None = None) -> None:
        self.selected.start_branch(regime_index=regime_index)

    def note_level_change(self) -> None:
        self.selected.note_level_change()

    def summary(self) -> Mapping[str, Any]:
        payload = dict(self.selected.summary())
        payload["t8_6i_selected_policy"] = self.selected_name
        return payload


def _factory_builder(
    *,
    registry: dict[str, StructuralLiveController],
) -> Any:
    def builder(*, mode: str, manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
        caps = manifest["controller"]

        def factory(game_id: str) -> UnifiedCognitiveController:
            if mode == "off":
                return UnifiedCognitiveController(
                    game_id,
                    config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
                )
            sage_t = StructuralLiveController(caps=caps)
            registry[str(game_id)] = sage_t
            return UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="shadow"),
                sage_t_controller=sage_t,  # type: ignore[arg-type]
            )

        return factory

    return builder


def run_live_confirmation(
    *,
    confirmation_manifest_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    selection_manifest_path: str | Path = v86i.DEFAULT_MANIFEST_PATH,
    selection_report_path: str | Path = DEFAULT_SELECTION_REPORT,
    t8_5_manifest_path: str | Path = DEFAULT_T8_5_MANIFEST,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    confirmation, selection = load_confirmation_manifest(
        confirmation_manifest_path,
        selection_manifest_path=selection_manifest_path,
        selection_report_path=selection_report_path,
        t8_5_manifest_path=t8_5_manifest_path,
    )
    registry: dict[str, StructuralLiveController] = {}
    previous_factory = t8_5._controller_factory
    t8_5._controller_factory = _factory_builder(registry=registry)
    started = time.perf_counter()
    try:
        base_report = t8_5.run_live_shadow_pilot(
            manifest_path=t8_5_manifest_path,
            environments_dir=environments_dir,
            output_dir=output_dir,
        )
    finally:
        t8_5._controller_factory = previous_factory
    destination = Path(output_dir)
    base_rows = [
        json.loads(line)
        for line in (destination / "rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    policy_rows = t8_6_live._policy_live_rows(registry, base_rows)
    selected_condition = str(confirmation["repair_policy"]["name"])
    live = t8_6_live._live_checks(
        policy_rows,
        selected_condition=selected_condition,
        base_report=base_report,
        confirmation=confirmation,
    )
    offline = dict(selection["offline_repair_v2_confirmation"])
    passed = bool(offline.get("passed")) and bool(live.get("passed"))
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "READY_TO_PREPARE_T8_7_SOURCE_VALIDATION"
            if passed
            else "T8_6I_LIVE_CONFIRMATION_FAILED_CLOSED"
        ),
        "manifest_checksum": confirmation["manifest_checksum"],
        "selection_report_checksum": selection["report_checksum"],
        "selected_challenger": confirmation["selected_challenger"],
        "repair_policy": confirmation["repair_policy"],
        "offline_confirmation": offline,
        "live_confirmation": live,
        "base_live_report_checksum": base_report.get("report_checksum"),
        "elapsed_seconds": time.perf_counter() - started,
        "conclusion": (
            "CALIBRATION_RECOVERED"
            if passed
            else "INCONCLUSIVE_FAIL_CLOSED"
        ),
        "source_validation_authorized": passed,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "policy_rows.jsonl", policy_rows)
    v86c._write_json(
        destination / "t8_6i_confirmation_report.json", report
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-manifest", default=str(DEFAULT_CONFIRMATION_MANIFEST)
    )
    parser.add_argument(
        "--selection-manifest", default=str(v86i.DEFAULT_MANIFEST_PATH)
    )
    parser.add_argument(
        "--selection-report", default=str(DEFAULT_SELECTION_REPORT)
    )
    parser.add_argument("--t8-5-manifest", default=str(DEFAULT_T8_5_MANIFEST))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        report = freeze_confirmation_manifest(
            output_path=args.confirmation_manifest,
            selection_manifest_path=args.selection_manifest,
            selection_report_path=args.selection_report,
            t8_5_manifest_path=args.t8_5_manifest,
        )
    else:
        report = run_live_confirmation(
            confirmation_manifest_path=args.confirmation_manifest,
            selection_manifest_path=args.selection_manifest,
            selection_report_path=args.selection_report,
            t8_5_manifest_path=args.t8_5_manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(report), indent=2, sort_keys=True))
    return 0 if args.freeze or report.get("source_validation_authorized") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIRMATION_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "StructuralGoalFragmentProposer",
    "StructuralLiveController",
    "freeze_confirmation_manifest",
    "load_confirmation_manifest",
    "main",
    "run_live_confirmation",
]
