"""Preregistered paired experiments for the SAGE.T causal-program runtime.

The module deliberately separates four immutable inputs: causal programs,
exact-prefix bundle plans, the experiment manifest, and the resulting signed
receipts.  Real environment access is confined to ``run_replay`` and
``run_experiment``; all other phases are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from theory.live_transition_loop import build_observation, build_transition_record
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    DefaultEnvironmentReplayAdapter,
    EnvironmentReplayAdapter,
    _make_real_env,
    select_live_action,
    state_signature_from_frame,
)
from theory.sage11.splits import SAGE11_SPLITS, short_game_id
from theory.unified_cognition_ab_benchmark import (
    SharedLegacyProposalPolicy,
    _run_attempt,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from ..compiler import compile_causal_observation, compile_transition_record
from ..controller import SageTConfig
from .adapters import causal_state_from_abstract, transition_evidence_from_observed
from .comparison import compare_particle
from .contracts import (
    ActionProgram,
    CausalProgram,
    GroundedAction,
    InterventionBranch,
    InterventionBundle,
    TransitionEvidence,
    causal_program_from_dict,
)
from .controller import CausalSageTController
from .decision import CausalDecisionEngine
from .executor import CausalExecutor
from .posterior import CausalPosterior, PosteriorUpdate
from .protocol import (
    CausalEvaluationFirewall,
    CausalProtocol,
    CausalProtocolStage,
    ft09_efficiency_gain,
    ft09_non_regression,
)
from .replay import InterventionBundleRunner, PrefixReplayEnvironment
from .runtime import CausalRuntime

PROGRAM_REGISTRY_FORMAT = "sage-t11-causal-program-registry-v1"
BUNDLE_PLAN_FORMAT = "sage-t11-causal-intervention-plan-v1"
EXPERIMENT_FORMAT = "sage-t11-causal-paired-experiment-v1"
REPLAY_REPORT_FORMAT = "sage-t11-causal-replay-report-v1"
RUN_REPORT_FORMAT = "sage-t11-causal-paired-report-v1"
RECEIPT_FORMAT = "sage-t11-causal-gate-receipt-v1"

DEFAULT_ARMS = (
    "baseline",
    "posterior_full",
    "no_posterior_update",
    "no_information_gain",
    "no_a40_memory",
    "no_mdl_prior",
)
SUPPORTED_ARMS = frozenset(
    (*DEFAULT_ARMS, "no_intergame_mechanisms", "symbolic_only")
)
CORE_CODE_PATHS = (
    "theory/sage_t/compiler.py",
    "theory/sage_t/causal/bp35_iteration_v2.py",
    "theory/sage_t/causal/adapters.py",
    "theory/sage_t/causal/comparison.py",
    "theory/sage_t/causal/compiler.py",
    "theory/sage_t/causal/contracts.py",
    "theory/sage_t/causal/controller.py",
    "theory/sage_t/causal/decision.py",
    "theory/sage_t/causal/executor.py",
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/experiment_cli.py",
    "theory/sage_t/causal/memory.py",
    "theory/sage_t/causal/mechanisms.py",
    "theory/sage_t/causal/posterior.py",
    "theory/sage_t/causal/protocol.py",
    "theory/sage_t/causal/replay.py",
    "theory/sage_t/causal/runtime.py",
    "theory/sage/live_prefix_counterfactual_collector.py",
    "theory/unified_cognitive_controller.py",
    "theory/unified_cognition_ab_benchmark.py",
)

EnvFactory = Callable[[str], Any]


class ArtifactBudgetExceeded(RuntimeError):
    """Fail-closed signal raised before a run can exceed its signed budget."""


class RunStorageBudget:
    def __init__(self, root: str | Path, maximum_bytes: int) -> None:
        self.root = Path(root)
        self.maximum_bytes = int(maximum_bytes)
        if self.maximum_bytes <= 0:
            raise ValueError("artifact budget must be positive")

    @property
    def used_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def reserve(self, additional_bytes: int) -> None:
        additional = max(0, int(additional_bytes))
        used = self.used_bytes
        if used + additional > self.maximum_bytes:
            raise ArtifactBudgetExceeded(
                "run artifact budget would be exceeded: "
                f"used={used} additional={additional} maximum={self.maximum_bytes}"
            )

    def snapshot(self) -> dict[str, Any]:
        used = self.used_bytes
        return {
            "maximum_artifact_bytes": self.maximum_bytes,
            "used_artifact_bytes": used,
            "remaining_artifact_bytes": max(0, self.maximum_bytes - used),
            "within_budget": used <= self.maximum_bytes,
        }


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(payload)).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _checksum(result)
    return result


def _verify_signed(payload: Mapping[str, Any], field: str) -> None:
    unsigned = dict(payload)
    observed = str(unsigned.pop(field, ""))
    if not observed or observed != _checksum(unsigned):
        raise ValueError(f"{field} mismatch")


def _write_json_once(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    storage_budget: RunStorageBudget | None = None,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n"
    if storage_budget is not None:
        storage_budget.reserve(len(encoded.encode("utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        encoded,
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl_once(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    storage_budget: RunStorageBudget | None = None,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    encoded_rows = tuple(_canonical(dict(row)) + "\n" for row in rows)
    if storage_budget is not None:
        storage_budget.reserve(sum(len(row.encode("utf-8")) for row in encoded_rows))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.writelines(encoded_rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bound_path(path: str | Path, *, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_bound_path(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to bind experiment to git state") from exc
    return {"commit": commit, "dirty": bool(status), "dirty_entries": len(status)}


def _core_code_hashes(root: Path) -> dict[str, str]:
    missing = [path for path in CORE_CODE_PATHS if not (root / path).is_file()]
    if missing:
        raise ValueError(f"causal experiment code inventory is incomplete: {missing}")
    return {path: _file_sha256(root / path) for path in CORE_CODE_PATHS}


def seal_program_registry(
    source_path: str | Path,
    output_path: str | Path,
    *,
    protocol: CausalProtocol | None = None,
) -> dict[str, Any]:
    """Validate and immutably sign complete rival programs per game."""
    selected_protocol = protocol or CausalProtocol()
    source = _read_json(source_path)
    games = dict(source.get("games", {}) or {})
    if not games:
        raise ValueError("program registry contains no games")
    executor = CausalExecutor()
    sealed_games: dict[str, Any] = {}
    for raw_game, raw_entry in sorted(games.items()):
        game = short_game_id(str(raw_game))
        SAGE11_SPLITS.split_for(game)
        entry = dict(raw_entry or {})
        action_catalog = tuple(
            dict.fromkeys(str(item).strip() for item in entry.get("action_catalog", ()))
        )
        if not action_catalog or any(not action for action in action_catalog):
            raise ValueError(f"{game} needs a non-empty action catalog")
        programs = tuple(
            causal_program_from_dict(dict(item))
            for item in entry.get("programs", ())
        )
        if len(programs) < 2:
            raise ValueError(f"{game} needs at least two complete rival programs")
        hashes = {program.canonical_hash for program in programs}
        if len(hashes) < 2:
            raise ValueError(f"{game} rival programs are structurally identical")
        catalog = set(action_catalog)
        for program in programs:
            declared = {item.action_name for item in program.action_model}
            if declared != catalog:
                raise ValueError(
                    f"{game}:{program.program_id} must declare the complete action catalog"
                )
            executor.compile(program, action_catalog=action_catalog)
        sealed_games[game] = {
            "action_catalog": list(action_catalog),
            "programs": [
                program.to_dict()
                for program in sorted(programs, key=lambda item: item.canonical_hash)
            ],
            "program_hashes": sorted(hashes),
        }
    payload = {
        "format_version": PROGRAM_REGISTRY_FORMAT,
        "protocol_checksum": selected_protocol.checksum,
        "games": sealed_games,
    }
    result = _signed(payload, "registry_checksum")
    _write_json_once(output_path, result)
    return result


def load_program_registry(
    path: str | Path,
    *,
    protocol: CausalProtocol | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != PROGRAM_REGISTRY_FORMAT:
        raise ValueError("unsupported causal program registry")
    if payload.get("protocol_checksum") != (protocol or CausalProtocol()).checksum:
        raise ValueError("program registry protocol checksum mismatch")
    return payload


def _action_from_payload(payload: Mapping[str, Any]) -> GroundedAction:
    name = str(payload.get("action_name", payload.get("name", ""))).strip()
    data = dict(payload.get("action_data", payload.get("action_args", {})) or {})
    return GroundedAction(name, data)


def _action_payload(action: GroundedAction) -> dict[str, Any]:
    return {"action_name": action.action_name, "action_data": dict(action.action_data)}


def seal_bundle_plan(
    source_path: str | Path,
    program_registry_path: str | Path,
    output_path: str | Path,
    *,
    protocol: CausalProtocol | None = None,
) -> dict[str, Any]:
    """Seal exact-prefix plans without observing any branch outcome."""
    selected_protocol = protocol or CausalProtocol()
    registry = load_program_registry(program_registry_path, protocol=selected_protocol)
    source = _read_json(source_path)
    bundles = []
    seen_ids: set[str] = set()
    for index, raw_bundle in enumerate(source.get("bundles", ())):
        item = dict(raw_bundle)
        bundle_id = str(item.get("bundle_id", f"bundle-{index}")).strip()
        if not bundle_id or bundle_id in seen_ids:
            raise ValueError("bundle ids must be non-empty and unique")
        seen_ids.add(bundle_id)
        game = short_game_id(str(item["game_id"]))
        if game not in registry["games"]:
            raise ValueError(f"bundle {bundle_id} has no rival program registry")
        prefix_hash = str(item.get("prefix_hash", "")).strip()
        if not prefix_hash:
            raise ValueError(f"bundle {bundle_id} needs a preregistered prefix hash")
        prefix = tuple(_action_from_payload(dict(raw)) for raw in item.get("prefix", ()))
        if not prefix:
            raise ValueError(f"bundle {bundle_id} needs a non-empty replay prefix")
        branches = tuple(
            _action_from_payload(dict(raw)) for raw in item.get("branches", ())
        )
        if len(branches) < 2 or len({branch.key for branch in branches}) < 2:
            raise ValueError(f"bundle {bundle_id} needs two distinct branches")
        catalog = set(registry["games"][game]["action_catalog"])
        unavailable = {
            action.action_name for action in (*prefix, *branches)
            if action.action_name not in catalog
        }
        if unavailable:
            raise ValueError(
                f"bundle {bundle_id} uses actions outside the registry: {sorted(unavailable)}"
            )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "game_id": game,
                "prefix_hash": prefix_hash,
                "prefix": [_action_payload(action) for action in prefix],
                "branches": [_action_payload(action) for action in branches],
            }
        )
    if not bundles:
        raise ValueError("intervention plan contains no bundles")
    payload = {
        "format_version": BUNDLE_PLAN_FORMAT,
        "protocol_checksum": selected_protocol.checksum,
        "program_registry_checksum": registry["registry_checksum"],
        "bundles": bundles,
    }
    result = _signed(payload, "plan_checksum")
    _write_json_once(output_path, result)
    return result


def load_bundle_plan(
    path: str | Path,
    *,
    registry: Mapping[str, Any],
    protocol: CausalProtocol | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "plan_checksum")
    if payload.get("format_version") != BUNDLE_PLAN_FORMAT:
        raise ValueError("unsupported causal intervention plan")
    if payload.get("protocol_checksum") != (protocol or CausalProtocol()).checksum:
        raise ValueError("intervention plan protocol checksum mismatch")
    if payload.get("program_registry_checksum") != registry.get("registry_checksum"):
        raise ValueError("intervention plan is bound to another program registry")
    return payload


def load_receipt(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "receipt_checksum")
    if payload.get("format_version") != RECEIPT_FORMAT:
        raise ValueError("unsupported causal experiment receipt")
    if payload.get("protocol_checksum") != CausalProtocol().checksum:
        raise ValueError("receipt protocol checksum mismatch")
    return payload


def freeze_experiment(
    *,
    program_registry_path: str | Path,
    bundle_plan_path: str | Path,
    output_path: str | Path,
    stage: CausalProtocolStage | str,
    game_ids: Sequence[str],
    seeds: Sequence[int],
    resets: int,
    action_budget_per_reset: int,
    authority: str = "shadow",
    arms: Sequence[str] = DEFAULT_ARMS,
    parent_receipt_path: str | Path | None = None,
    allow_dirty: bool = False,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the complete paired design before reading environment outcomes."""
    repo_root = Path(root).resolve() if root is not None else _repo_root()
    protocol = CausalProtocol()
    normalized_stage = (
        stage if isinstance(stage, CausalProtocolStage)
        else CausalProtocolStage(str(stage))
    )
    if normalized_stage is CausalProtocolStage.HOLDOUT_CONFIRMATION:
        raise ValueError("the experimental CLI cannot open the one-shot holdout")
    games = tuple(dict.fromkeys(short_game_id(game) for game in game_ids))
    if not games:
        raise ValueError("experiment needs at least one game")
    CausalEvaluationFirewall(protocol=protocol).assert_authorized(
        games,
        stage=normalized_stage,
    )
    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized_seeds:
        raise ValueError("experiment needs at least one seed")
    if not 1 <= int(resets) <= 32:
        raise ValueError("resets must be in [1, 32]")
    if not 1 <= int(action_budget_per_reset) <= 256:
        raise ValueError("action budget must be in [1, 256]")
    normalized_authority = str(authority).lower()
    if normalized_authority not in {"shadow", "bounded"}:
        raise ValueError("causal experiments support only shadow or bounded authority")
    normalized_arms = tuple(dict.fromkeys(str(arm) for arm in arms))
    if "baseline" not in normalized_arms or "posterior_full" not in normalized_arms:
        raise ValueError("paired experiment requires baseline and posterior_full")
    unknown_arms = set(normalized_arms) - SUPPORTED_ARMS
    if unknown_arms:
        raise ValueError(f"unsupported causal ablations: {sorted(unknown_arms)}")
    registry = load_program_registry(program_registry_path, protocol=protocol)
    missing_programs = sorted(set(games) - set(registry["games"]))
    if missing_programs:
        raise ValueError(f"experiment games lack rival programs: {missing_programs}")
    plan = load_bundle_plan(bundle_plan_path, registry=registry, protocol=protocol)
    planned_games = {str(item["game_id"]) for item in plan["bundles"]}
    missing_bundles = sorted(set(games) - planned_games)
    if missing_bundles:
        raise ValueError(f"experiment games lack exact-prefix bundles: {missing_bundles}")
    parent_receipt = None
    if normalized_stage is CausalProtocolStage.SOURCE_VALIDATION:
        if parent_receipt_path is None:
            raise ValueError("source validation needs a passing source-train receipt")
        parent_receipt = load_receipt(parent_receipt_path)
        if (
            parent_receipt.get("stage") != CausalProtocolStage.SOURCE_TRAIN.value
            or parent_receipt.get("passed") is not True
        ):
            raise ValueError("source-validation parent receipt did not pass source train")
        parent_metrics = dict(parent_receipt.get("metrics", {}) or {})
        if (
            int(parent_metrics.get("games_with_progress", 0)) < 1
            or int(parent_metrics.get("safety_regressions", 1)) != 0
            or parent_metrics.get("posterior_ablation_advantage") is not True
            or not parent_receipt.get("experiment_manifest_checksum")
            or not parent_receipt.get("report_checksum")
        ):
            raise ValueError("source-train receipt does not satisfy the frozen parent gate")
    git_state = _git_state(repo_root)
    if git_state["dirty"] and not allow_dirty:
        raise ValueError("refusing to freeze a scientific experiment from a dirty tree")
    scientific_claims_authorized = not git_state["dirty"]
    registry_path = Path(program_registry_path).resolve()
    plan_path = Path(bundle_plan_path).resolve()
    payload = {
        "format_version": EXPERIMENT_FORMAT,
        "status": (
            "FROZEN_BEFORE_CAUSAL_EXPERIMENT"
            if scientific_claims_authorized
            else "SMOKE_ONLY_DIRTY_TREE"
        ),
        "protocol_checksum": protocol.checksum,
        "git": git_state,
        "scientific_claims_authorized": scientific_claims_authorized,
        "code_sha256": _core_code_hashes(repo_root),
        "program_registry": {
            "path": _bound_path(registry_path, root=repo_root),
            "sha256": _file_sha256(registry_path),
            "registry_checksum": registry["registry_checksum"],
        },
        "bundle_plan": {
            "path": _bound_path(plan_path, root=repo_root),
            "sha256": _file_sha256(plan_path),
            "plan_checksum": plan["plan_checksum"],
        },
        "stage": normalized_stage.value,
        "games": list(games),
        "seeds": list(normalized_seeds),
        "resets": int(resets),
        "action_budget_per_reset": int(action_budget_per_reset),
        "arms": list(normalized_arms),
        "authority": {
            "requested": normalized_authority,
            "bounded_requires_passing_replay_receipt": True,
            "production_authority": False,
            "maximum_interventions_per_reset": protocol.maximum_interventions_per_reset,
            "maximum_terminal_probe_risk": protocol.maximum_terminal_probe_risk,
        },
        "pairing": {
            "same_game_seed_reset_budget": True,
            "fresh_environment_per_arm": True,
            "fresh_controller_per_reset": True,
            "a40_reload_between_resets": True,
            "memory_reset_between_validation_games": True,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": (
                protocol.maximum_artifact_bytes_per_run
            ),
            "hard_fail_before_write": True,
            "a40_declared_variables_only": True,
        },
        "parent_receipt": (
            None
            if parent_receipt is None
            else {
                "path": _bound_path(parent_receipt_path, root=repo_root),
                "receipt_checksum": parent_receipt["receipt_checksum"],
            }
        ),
        "runtime": {"arc-agi": "0.9.1", "arcengine": "0.9.3"},
        "firewall": {
            "holdout_opened": False,
            "production_authority": False,
            "stage": normalized_stage.value,
        },
    }
    result = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, result)
    freeze_receipt = _signed(
        {
            "format_version": RECEIPT_FORMAT,
            "kind": "freeze",
            "stage": normalized_stage.value,
            "passed": scientific_claims_authorized,
            "protocol_checksum": protocol.checksum,
            "experiment_manifest_checksum": result["manifest_checksum"],
            "reason": (
                "frozen_clean_tree" if scientific_claims_authorized
                else "dirty_tree_smoke_only"
            ),
        },
        "receipt_checksum",
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return result


def load_experiment_manifest(
    path: str | Path,
    *,
    verify_code: bool = True,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else _repo_root()
    payload = _read_json(path)
    _verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != EXPERIMENT_FORMAT:
        raise ValueError("unsupported causal experiment manifest")
    if payload.get("protocol_checksum") != CausalProtocol().checksum:
        raise ValueError("experiment protocol checksum mismatch")
    storage = dict(payload.get("storage", {}) or {})
    if int(storage.get("maximum_artifact_bytes_per_run", 0)) != (
        CausalProtocol().maximum_artifact_bytes_per_run
    ):
        raise ValueError("experiment artifact budget mismatch")
    if storage.get("hard_fail_before_write") is not True:
        raise ValueError("experiment artifact budget is not fail-closed")
    if verify_code and payload.get("code_sha256") != _core_code_hashes(repo_root):
        raise ValueError("causal experiment code drifted after freeze")
    for field, checksum_field in (
        ("program_registry", "registry_checksum"),
        ("bundle_plan", "plan_checksum"),
    ):
        binding = dict(payload[field])
        bound = _resolve_bound_path(str(binding["path"]), root=repo_root)
        if _file_sha256(bound) != binding["sha256"]:
            raise ValueError(f"{field} bytes drifted after freeze")
        loaded = _read_json(bound)
        if loaded.get(checksum_field) != binding[checksum_field]:
            raise ValueError(f"{field} semantic checksum drifted after freeze")
    parent_binding = payload.get("parent_receipt")
    if parent_binding:
        parent = dict(parent_binding)
        receipt = load_receipt(
            _resolve_bound_path(str(parent["path"]), root=repo_root)
        )
        if receipt.get("receipt_checksum") != parent.get("receipt_checksum"):
            raise ValueError("source-validation parent receipt drifted after freeze")
        if (
            receipt.get("stage") != CausalProtocolStage.SOURCE_TRAIN.value
            or receipt.get("passed") is not True
        ):
            raise ValueError("source-validation parent receipt is no longer admissible")
    CausalEvaluationFirewall().assert_authorized(
        tuple(payload["games"]),
        stage=str(payload["stage"]),
    )
    return payload


def _programs_for_game(
    registry: Mapping[str, Any], game_id: str
) -> tuple[CausalProgram, ...]:
    entry = dict(registry["games"][short_game_id(game_id)])
    return tuple(causal_program_from_dict(dict(item)) for item in entry["programs"])


class CausalPrefixReplayEnvironment(PrefixReplayEnvironment):
    """Real/injected ARC environment adapter for exact-prefix bundles."""

    def __init__(
        self,
        *,
        game_id: str,
        environments_dir: str | Path,
        env_factory: EnvFactory | None = None,
        replay_adapter: EnvironmentReplayAdapter | None = None,
    ) -> None:
        self.game_id = str(game_id)
        self.environments_dir = Path(environments_dir)
        self.env_factory = env_factory
        self.replay_adapter = replay_adapter or DefaultEnvironmentReplayAdapter()
        self._frame_environments: dict[int, Any] = {}

    def reset_and_replay(self, prefix: object) -> Any:
        if not isinstance(prefix, ActionProgram):
            raise TypeError("prefix replay expects an ActionProgram")
        env = (
            self.env_factory(self.game_id)
            if self.env_factory is not None
            else _make_real_env(self.game_id, self.environments_dir)
        )
        frame = self.replay_adapter.reset(env)
        for action in prefix.actions:
            selected = select_live_action(
                env,
                action.action_name,
                action_args=action.action_data,
            )
            if selected is None:
                raise ValueError(f"prefix action unavailable: {action.action_name}")
            frame = self.replay_adapter.step(env, selected)
        self._frame_environments[id(frame)] = env
        return frame

    def state_hash(self, frame: Any) -> str:
        return state_signature_from_frame(frame)

    def legal_action_names(self, frame: Any) -> Sequence[str]:
        env = self._frame_environments[id(frame)]
        from theory.non_ar25_active_micro_run import _valid_actions

        return tuple(action.name for action in _valid_actions(env))

    def execute(self, frame: Any, action: object) -> Any:
        if not isinstance(action, GroundedAction):
            raise TypeError("branch execution expects a GroundedAction")
        env = self._frame_environments.pop(id(frame))
        selected = select_live_action(
            env,
            action.action_name,
            action_args=action.action_data,
        )
        if selected is None:
            raise ValueError(f"branch action unavailable: {action.action_name}")
        return self.replay_adapter.step(env, selected)


def _causal_state_from_frame(frame: Any, *, prefix_hash: str):
    snapshot = snapshot_frame(frame)
    observation = build_observation(
        snapshot.grid,
        available_actions=snapshot.available_actions,
        game_state=snapshot.game_state,
        levels_completed=snapshot.levels_completed,
    )
    return causal_state_from_abstract(
        compile_causal_observation(observation),
        observation_hash=prefix_hash,
    )


def _evidence_builder(
    *,
    game_id: str,
    prefix_hash: str,
) -> Callable[[Any, object, Any, int], TransitionEvidence]:
    def build(before: Any, action: object, after: Any, index: int) -> TransitionEvidence:
        if not isinstance(action, GroundedAction):
            raise TypeError("evidence builder expects a GroundedAction")
        before_snapshot = snapshot_frame(before)
        after_snapshot = snapshot_frame(
            after,
            fallback_available_actions=before_snapshot.available_actions,
        )
        record = build_transition_record(
            action=action.action_name,
            action_args=dict(action.action_data),
            grid_before=before_snapshot.grid,
            grid_after=after_snapshot.grid,
            available_actions=before_snapshot.available_actions,
            game_state_before=before_snapshot.game_state,
            game_state_after=after_snapshot.game_state,
            levels_completed_before=before_snapshot.levels_completed,
            levels_completed_after=after_snapshot.levels_completed,
            timestamp=index,
        )
        return transition_evidence_from_observed(
            compile_transition_record(record, compact_causal_state=True),
            game_id=game_id,
            prefix_hash=prefix_hash,
        )

    return build


def _runtime_versions() -> dict[str, Any]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {"ready": False, "reason": "importlib_metadata_missing", "versions": {}}
    versions = {}
    for package in ("arc-agi", "arcengine"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    try:
        from arc_agi import Arcade, EnvironmentWrapper, OperationMode

        del Arcade, EnvironmentWrapper, OperationMode
        ready = True
        reason = ""
    except (ImportError, AttributeError) as exc:
        ready = False
        reason = f"arc_sdk_unavailable:{exc}"
    return {"ready": ready, "reason": reason, "versions": versions}


def run_replay(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    replay_adapter: EnvironmentReplayAdapter | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else _repo_root()
    manifest = load_experiment_manifest(manifest_path, root=repo_root)
    registry_path = _resolve_bound_path(
        str(manifest["program_registry"]["path"]), root=repo_root
    )
    plan_path = _resolve_bound_path(str(manifest["bundle_plan"]["path"]), root=repo_root)
    registry = load_program_registry(registry_path)
    plan = load_bundle_plan(plan_path, registry=registry)
    runtime_status = (
        {"ready": True, "reason": "injected_environment", "versions": {}}
        if env_factory is not None
        else _runtime_versions()
    )
    expected_versions = dict(manifest["runtime"])
    versions_match = env_factory is not None or all(
        runtime_status["versions"].get(name) == value
        for name, value in expected_versions.items()
    )
    runtime_status["expected_versions"] = expected_versions
    runtime_status["versions_match"] = versions_match
    runtime_status["ready"] = bool(runtime_status["ready"] and versions_match)
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable run directory: {destination}"
        )
    storage_budget = RunStorageBudget(
        destination,
        int(manifest["storage"]["maximum_artifact_bytes_per_run"]),
    )
    results = []
    if runtime_status["ready"]:
        for item in plan["bundles"]:
            game_id = str(item["game_id"])
            programs = _programs_for_game(registry, game_id)
            executor = CausalExecutor()
            # Keep the preregistered rival set fixed while measuring whether
            # the intervention discriminates it. Repair is evaluated in the
            # paired online arms; allowing it here would confound information
            # gain with hypothesis-set expansion.
            runtime = CausalRuntime(
                executor=executor,
                posterior=CausalPosterior(
                    executor=executor,
                    maximum_repair_parents=0,
                ),
            )
            runtime.seed(programs)
            environment = CausalPrefixReplayEnvironment(
                game_id=game_id,
                environments_dir=environments_dir,
                env_factory=env_factory,
                replay_adapter=replay_adapter,
            )
            prefix = ActionProgram(
                tuple(_action_from_payload(raw) for raw in item["prefix"]),
                source="exact_route",
            )
            preview = environment.reset_and_replay(prefix)
            observed_hash = environment.state_hash(preview)
            state = _causal_state_from_frame(preview, prefix_hash=observed_hash)
            branches = []
            for raw_action in item["branches"]:
                action = _action_from_payload(raw_action)
                predictions = {
                    particle.program.canonical_hash: runtime.executor.predict_step(
                        particle.program, state, action
                    ).structured_signature
                    for particle in runtime.posterior.particles
                }
                branches.append(InterventionBranch(action, predictions))
            bundle = InterventionBundle(
                prefix=prefix,
                prefix_hash=str(item["prefix_hash"]),
                branches=tuple(branches),
            )
            result = InterventionBundleRunner(runtime=runtime).run(
                bundle,
                environment=environment,
                evidence_builder=_evidence_builder(
                    game_id=game_id,
                    prefix_hash=str(item["prefix_hash"]),
                ),
            )
            results.append(
                {
                    "bundle_id": item["bundle_id"],
                    "game_id": game_id,
                    "preregistered_prefix_hash": item["prefix_hash"],
                    "preview_prefix_hash": observed_hash,
                    "status": result.status,
                    "reason": result.reason,
                    "predictions_registered_before_execution": (
                        result.predictions_registered_before_execution
                    ),
                    "entropy_reduction": result.entropy_reduction,
                    "branches": [
                        {
                            "action_name": branch.action_name,
                            "prefix_hash": branch.prefix_hash,
                            "evidence_id": branch.evidence_id,
                            "entropy_before": branch.entropy_before,
                            "entropy_after": branch.entropy_after,
                        }
                        for branch in result.branches
                    ],
                }
            )
    all_complete = bool(results) and all(
        row["status"] == "BUNDLE_COMPLETE"
        and row["predictions_registered_before_execution"]
        and row["preview_prefix_hash"] == row["preregistered_prefix_hash"]
        and len(row["branches"]) >= 2
        for row in results
    )
    total_entropy_reduction = sum(float(row["entropy_reduction"]) for row in results)
    passed = bool(
        runtime_status["ready"]
        and all_complete
        and total_entropy_reduction > 1e-9
        and manifest["scientific_claims_authorized"]
    )
    report_core = {
        "format_version": REPLAY_REPORT_FORMAT,
        "status": "PASS_CAUSAL_REPLAY_GATE" if passed else "FAIL_CAUSAL_REPLAY_GATE",
        "protocol_checksum": manifest["protocol_checksum"],
        "experiment_manifest_checksum": manifest["manifest_checksum"],
        "runtime": runtime_status,
        "metrics": {
            "bundles": len(results),
            "complete_bundles": sum(row["status"] == "BUNDLE_COMPLETE" for row in results),
            "total_entropy_reduction": total_entropy_reduction,
        },
        "bundles": results,
        "storage": storage_budget.snapshot(),
        "passed": passed,
    }
    report = _signed(report_core, "report_checksum")
    receipt = _signed(
        {
            "format_version": RECEIPT_FORMAT,
            "kind": "replay",
            "stage": manifest["stage"],
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "experiment_manifest_checksum": manifest["manifest_checksum"],
            "report_checksum": report["report_checksum"],
            "metrics": report["metrics"],
            "maximum_artifact_bytes": storage_budget.maximum_bytes,
            "reason": report["status"],
        },
        "receipt_checksum",
    )
    _write_jsonl_once(
        destination / "intervention_bundles.jsonl",
        results,
        storage_budget=storage_budget,
    )
    _write_json_once(
        destination / "replay_report.json", report, storage_budget=storage_budget
    )
    _write_json_once(
        destination / "replay_receipt.json", receipt, storage_budget=storage_budget
    )
    return report


class StaticPosteriorRuntime(CausalRuntime):
    """Ablation: compare evidence but never perform A39/A40 updates."""

    def observe(self, evidence: TransitionEvidence) -> PosteriorUpdate:
        entropy = self.posterior.entropy
        comparisons = tuple(
            compare_particle(
                program=particle.program,
                evidence=evidence,
                executor=self.executor,
            )
            for particle in self.posterior.particles
        )
        return PosteriorUpdate(
            evidence_id=evidence.evidence_id,
            entropy_before=entropy,
            entropy_after=entropy,
            effective_sample_size=self.posterior.effective_sample_size,
            comparisons=comparisons,
        )


class ExperimentalCausalController(CausalSageTController):
    """Bounded in-memory instrumentation for paired scientific runs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.initial_particle_count = len(self.posterior.particles)
        self.decision_latencies_ms: list[float] = []
        self.observation_latencies_ms: list[float] = []
        self.compact_records: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            return super().decide(**kwargs)
        finally:
            self.decision_latencies_ms.append(
                (time.perf_counter() - started) * 1000.0
            )

    def observe_transition(self, record: Any) -> None:
        started = time.perf_counter()
        try:
            super().observe_transition(record)
        finally:
            self.observation_latencies_ms.append(
                (time.perf_counter() - started) * 1000.0
            )

    def _record(self, payload: Mapping[str, Any]) -> None:
        compact = dict(payload)
        posterior = compact.get("posterior")
        if isinstance(posterior, Mapping):
            trimmed = dict(posterior)
            trimmed["particles"] = list(trimmed.get("particles", ()))[:8]
            compact["posterior"] = trimmed
        self.compact_records.append(compact)
        super()._record(payload)

    def summary(self) -> Mapping[str, Any]:
        result = dict(super().summary())
        result["instrumentation"] = {
            "initial_particle_count": self.initial_particle_count,
            "decision_latencies_ms": tuple(self.decision_latencies_ms),
            "observation_latencies_ms": tuple(self.observation_latencies_ms),
            "records": tuple(self.compact_records),
        }
        return result


def _filtered_programs(
    programs: Sequence[CausalProgram], arm: str
) -> tuple[CausalProgram, ...]:
    selected = tuple(programs)
    if arm == "no_intergame_mechanisms":
        selected = tuple(
            program
            for program in selected
            if not any(
                str(item).startswith(("memory:", "intergame:"))
                for item in program.provenance
            )
        )
    elif arm == "symbolic_only":
        selected = tuple(
            program
            for program in selected
            if program.observation_model.neural_module_id is None
            and all(mechanism.neural_module_id is None for mechanism in program.mechanisms)
        )
    if len({program.canonical_hash for program in selected}) < 2:
        raise ValueError(f"ablation {arm} leaves fewer than two rival programs")
    return selected


def _build_controller(
    *,
    game_id: str,
    arm: str,
    programs: Sequence[CausalProgram],
    authority: str,
    memory_path: Path | None,
    reserve_memory_bytes: Callable[[int], None] | None,
    replay_gate_passed: bool,
) -> tuple[UnifiedCognitiveController, ExperimentalCausalController | None]:
    if arm == "baseline":
        return (
            UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
            ),
            None,
        )
    selected_programs = _filtered_programs(programs, arm)
    executor = CausalExecutor()
    posterior = CausalPosterior(
        executor=executor,
        mdl_beta=0.0 if arm == "no_mdl_prior" else 1.0,
    )
    decision_engine = CausalDecisionEngine(
        executor=executor,
        information_gain_scale=0.0 if arm == "no_information_gain" else 1.0,
    )
    runtime_type = StaticPosteriorRuntime if arm == "no_posterior_update" else CausalRuntime
    durable_memory = (
        memory_path
        if arm not in {"no_a40_memory", "no_posterior_update"}
        else None
    )
    runtime = runtime_type(
        executor=executor,
        posterior=posterior,
        decision_engine=decision_engine,
        memory_path=durable_memory,
        reserve_memory_bytes=reserve_memory_bytes,
    )
    runtime.seed(selected_programs)
    if durable_memory is not None and durable_memory.exists():
        runtime.reload_memory()
    config = SageTConfig(
        mode=authority,
        counterfactual_gate_passed=replay_gate_passed,
        active_gate_passed=False,
        bounded_maximum_interventions_per_reset=(
            CausalProtocol().maximum_interventions_per_reset
        ),
        bounded_maximum_terminal_risk=CausalProtocol().maximum_terminal_probe_risk,
    )
    causal = ExperimentalCausalController(config=config, runtime=runtime)
    if causal.initial_particle_count < 2:
        raise ValueError("rival programs were not initialized before first choice")
    unified = UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(
            sage_t_authority_mode=authority,
            sage_t_counterfactual_gate_passed=replay_gate_passed,
            sage_t_active_gate_passed=False,
        ),
        sage_t_controller=causal,
    )
    return unified, causal


def _run_arm_restarting(
    *,
    game_id: str,
    seed: int,
    arm: str,
    programs: Sequence[CausalProgram],
    authority: str,
    replay_gate_passed: bool,
    resets: int,
    action_budget: int,
    environments_dir: str | Path,
    memory_path: Path | None,
    reserve_memory_bytes: Callable[[int], None] | None,
    env_factory: EnvFactory | None,
) -> dict[str, Any]:
    attempts = []
    controller_errors: list[str] = []
    decision_sources: Counter[str] = Counter()
    summaries = []
    for reset_index in range(resets):
        controller, causal = _build_controller(
            game_id=game_id,
            arm=arm,
            programs=programs,
            authority=authority,
            memory_path=memory_path,
            reserve_memory_bytes=reserve_memory_bytes,
            replay_gate_passed=replay_gate_passed,
        )
        controller.on_reset()
        policy = SharedLegacyProposalPolicy(
            game_id=game_id,
            seed=seed,
            reset_index=reset_index,
        )
        attempt = _run_attempt(
            arm="unified",
            game_id=game_id,
            reset_index=reset_index,
            action_budget=action_budget,
            env_dir=Path(environments_dir),
            env_factory=env_factory,
            policy=policy,
            controller=controller,
            decision_sources=decision_sources,
            controller_errors=controller_errors,
        )
        attempts.append(attempt)
        summaries.append(None if causal is None else dict(causal.summary()))
    return {
        "arm": arm,
        "game_id": game_id,
        "seed": seed,
        "attempts": attempts,
        "decision_sources": dict(decision_sources),
        "controller_errors": controller_errors,
        "causal_summaries": summaries,
        "memory_path": None if memory_path is None else str(memory_path),
    }


def _arm_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    attempts = tuple(result.get("attempts", ()) or ())
    steps = tuple(
        step for attempt in attempts for step in attempt.get("trace", ()) or ()
    )
    summaries = tuple(
        summary for summary in result.get("causal_summaries", ()) or ()
        if isinstance(summary, Mapping)
    )
    instrumentations = [dict(summary.get("instrumentation", {}) or {}) for summary in summaries]
    records = [
        dict(record)
        for instrumentation in instrumentations
        for record in instrumentation.get("records", ()) or ()
    ]
    interventions = sum(int(summary.get("interventions", 0) or 0) for summary in summaries)
    causal_decisions = sum(int(summary.get("decisions", 0) or 0) for summary in summaries)
    game_over_actions = sum(
        str(step.get("game_state_after", "")).upper()
        in {"GAME_OVER", "FAILED", "FAILURE", "LOSE", "LOSS"}
        for step in steps
    )
    levels = sum(
        max(0, int(step.get("levels_after", 0)) - int(step.get("levels_before", 0)))
        for step in steps
    )
    errors = tuple(str(item) for item in result.get("controller_errors", ()) or ())
    memory_path = result.get("memory_path")
    memory_records = 0
    if memory_path and Path(str(memory_path)).exists():
        memory_records = sum(
            1
            for line in Path(str(memory_path)).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return {
        "actions": len(steps),
        "levels_completed": levels,
        "progress_events": sum(
            int(step.get("levels_after", 0)) > int(step.get("levels_before", 0))
            for step in steps
        ),
        "wins": sum(bool(attempt.get("win")) for attempt in attempts),
        "max_level": max(
            (int(attempt.get("max_level_reached", 0)) for attempt in attempts),
            default=0,
        ),
        "game_over_actions": game_over_actions,
        "controller_errors": len(errors),
        "illegal_actions": sum("unavailable_decision" in error for error in errors),
        "environment_errors": sum(
            str(attempt.get("failure_cause", "")).startswith("environment_")
            for attempt in attempts
        ),
        "interventions": interventions,
        "causal_decisions": causal_decisions,
        "causal_pipeline_fallbacks": max(0, len(steps) - causal_decisions),
        "protected_route_preemptions": sum(
            bool(record.get("applied")) and bool(record.get("protected_route"))
            for record in records
            if record.get("kind") == "causal_decision"
        ),
        "initial_particle_counts": [
            int(item.get("initial_particle_count", 0)) for item in instrumentations
        ],
        "decision_latency_ms": [
            float(value)
            for item in instrumentations
            for value in item.get("decision_latencies_ms", ()) or ()
        ],
        "observation_latency_ms": [
            float(value)
            for item in instrumentations
            for value in item.get("observation_latencies_ms", ()) or ()
        ],
        "memory_records": memory_records,
        "reset_visual_digests": [
            str(attempt.get("reset_visual_digest", "")) for attempt in attempts
        ],
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = min(len(ordered) - 1, int(probability * (len(ordered) - 1)))
    return ordered[index]


def _load_replay_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = load_receipt(path)
    if receipt.get("kind") != "replay":
        raise ValueError("expected an exact-prefix replay receipt")
    if receipt.get("experiment_manifest_checksum") != manifest.get("manifest_checksum"):
        raise ValueError("replay receipt belongs to another experiment")
    return receipt


def _posterior_advantage(conditions: Sequence[Mapping[str, Any]]) -> bool:
    full = sum(
        int(condition["arms"]["posterior_full"]["metrics"]["levels_completed"])
        for condition in conditions
    )
    ablated = sum(
        int(condition["arms"]["no_posterior_update"]["metrics"]["levels_completed"])
        for condition in conditions
        if "no_posterior_update" in condition["arms"]
    )
    if not any("no_posterior_update" in condition["arms"] for condition in conditions):
        return False
    if full != ablated:
        return full > ablated
    if full <= 0:
        return False
    full_actions = sum(
        int(condition["arms"]["posterior_full"]["metrics"]["actions"])
        for condition in conditions
    )
    ablated_actions = sum(
        int(condition["arms"]["no_posterior_update"]["metrics"]["actions"])
        for condition in conditions
    )
    return full_actions < ablated_actions


def run_experiment(
    *,
    manifest_path: str | Path,
    replay_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute baseline/full/ablation arms under strict paired resets."""
    repo_root = Path(root).resolve() if root is not None else _repo_root()
    manifest = load_experiment_manifest(manifest_path, root=repo_root)
    replay_receipt = _load_replay_receipt(replay_receipt_path, manifest=manifest)
    replay_passed = replay_receipt.get("passed") is True
    authority = str(manifest["authority"]["requested"])
    if authority == "bounded" and not replay_passed:
        raise ValueError("bounded authority remains closed without replay gate")
    registry_path = _resolve_bound_path(
        str(manifest["program_registry"]["path"]), root=repo_root
    )
    registry = load_program_registry(registry_path)
    runtime_status = (
        {"ready": True, "reason": "injected_environment", "versions": {}}
        if env_factory is not None
        else _runtime_versions()
    )
    expected_versions = dict(manifest["runtime"])
    versions_match = env_factory is not None or all(
        runtime_status["versions"].get(name) == value
        for name, value in expected_versions.items()
    )
    runtime_status["expected_versions"] = expected_versions
    runtime_status["versions_match"] = versions_match
    runtime_status["ready"] = bool(runtime_status["ready"] and versions_match)
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable run directory: {destination}"
        )
    storage_budget = RunStorageBudget(
        destination,
        int(manifest["storage"]["maximum_artifact_bytes_per_run"]),
    )
    if not runtime_status["ready"]:
        conditions: list[dict[str, Any]] = []
    else:
        conditions = []
        for game_id in manifest["games"]:
            programs = _programs_for_game(registry, game_id)
            for seed in manifest["seeds"]:
                arm_results: dict[str, Any] = {}
                for arm in manifest["arms"]:
                    memory_path = (
                        destination
                        / "memory"
                        / str(game_id)
                        / str(seed)
                        / str(arm)
                        / "posterior.jsonl"
                    )
                    raw = _run_arm_restarting(
                        game_id=str(game_id),
                        seed=int(seed),
                        arm=str(arm),
                        programs=programs,
                        authority=authority,
                        replay_gate_passed=replay_passed,
                        resets=int(manifest["resets"]),
                        action_budget=int(manifest["action_budget_per_reset"]),
                        environments_dir=environments_dir,
                        memory_path=(None if arm == "baseline" else memory_path),
                        reserve_memory_bytes=storage_budget.reserve,
                        env_factory=env_factory,
                    )
                    storage_budget.reserve(0)
                    arm_results[str(arm)] = {
                        "metrics": _arm_metrics(raw),
                        "attempts": raw["attempts"],
                        "decision_sources": raw["decision_sources"],
                        "controller_errors": raw["controller_errors"],
                        "memory_path": raw["memory_path"],
                    }
                baseline_digests = arm_results["baseline"]["metrics"][
                    "reset_visual_digests"
                ]
                conditions.append(
                    {
                        "game_id": str(game_id),
                        "seed": int(seed),
                        "strict_prestate_pairing": all(
                            arm["metrics"]["reset_visual_digests"] == baseline_digests
                            for arm in arm_results.values()
                        ),
                        "arms": arm_results,
                    }
                )
    nonbaseline = [
        arm
        for condition in conditions
        for name, arm in condition["arms"].items()
        if name != "baseline"
    ]
    safety_regressions = sum(
        int(
            condition["arms"]["posterior_full"]["metrics"]["game_over_actions"]
            > condition["arms"]["baseline"]["metrics"]["game_over_actions"]
            or condition["arms"]["posterior_full"]["metrics"]["illegal_actions"] > 0
            or condition["arms"]["posterior_full"]["metrics"]["controller_errors"] > 0
            or condition["arms"]["posterior_full"]["metrics"]["environment_errors"] > 0
        )
        for condition in conditions
    )
    games_with_progress = len(
        {
            condition["game_id"]
            for condition in conditions
            if condition["arms"]["posterior_full"]["metrics"]["levels_completed"] > 0
        }
    )
    posterior_advantage = _posterior_advantage(conditions)
    checks = {
        "runtime_ready": bool(runtime_status["ready"]),
        "replay_receipt_passed": replay_passed,
        "strict_prestate_pairing": bool(conditions)
        and all(condition["strict_prestate_pairing"] for condition in conditions),
        "rivals_initialized_before_first_choice": bool(nonbaseline)
        and all(
            metrics["initial_particle_counts"]
            and min(metrics["initial_particle_counts"]) >= 2
            for metrics in (arm["metrics"] for arm in nonbaseline)
        ),
        "zero_controller_errors": all(
            arm["metrics"]["controller_errors"] == 0 for arm in nonbaseline
        ),
        "zero_causal_pipeline_fallbacks": all(
            arm["metrics"]["causal_pipeline_fallbacks"] == 0
            for arm in nonbaseline
        ),
        "zero_environment_errors": all(
            arm["metrics"]["environment_errors"] == 0 for arm in nonbaseline
        ),
        "zero_illegal_actions": all(
            arm["metrics"]["illegal_actions"] == 0 for arm in nonbaseline
        ),
        "zero_protected_route_preemptions": all(
            arm["metrics"]["protected_route_preemptions"] == 0
            for arm in nonbaseline
        ),
        "validation_memory_isolated": len(
            {
                str(arm.get("memory_path"))
                for condition in conditions
                for arm in condition["arms"].values()
                if arm.get("memory_path")
            }
        )
        == sum(
            bool(arm.get("memory_path"))
            for condition in conditions
            for arm in condition["arms"].values()
        ),
        "scientific_claims_authorized": bool(
            manifest["scientific_claims_authorized"]
        ),
        "storage_budget_enforced": storage_budget.snapshot()["within_budget"],
    }
    full_metrics = [
        condition["arms"]["posterior_full"]["metrics"] for condition in conditions
    ]
    aggregate_metrics = {
        "games_with_progress": games_with_progress,
        "safety_regressions": safety_regressions,
        "posterior_ablation_advantage": posterior_advantage,
        "full_levels_completed": sum(item["levels_completed"] for item in full_metrics),
        "full_actions": sum(item["actions"] for item in full_metrics),
        "full_wins": sum(item["wins"] for item in full_metrics),
        "decision_p95_ms": _percentile(
            [value for item in full_metrics for value in item["decision_latency_ms"]],
            0.95,
        ),
        "observation_p95_ms": _percentile(
            [value for item in full_metrics for value in item["observation_latency_ms"]],
            0.95,
        ),
        "mean_bundle_entropy_reduction": (
            float(replay_receipt.get("metrics", {}).get("total_entropy_reduction", 0.0))
            / max(1, int(replay_receipt.get("metrics", {}).get("bundles", 0)))
        ),
    }
    stage = CausalProtocolStage(str(manifest["stage"]))
    integrity_passed = all(checks.values()) and safety_regressions == 0
    if stage is CausalProtocolStage.SOURCE_VALIDATION:
        scientific_gate = games_with_progress >= 2 and posterior_advantage
    elif stage is CausalProtocolStage.SOURCE_TRAIN:
        scientific_gate = games_with_progress >= 1 and posterior_advantage
    elif stage is CausalProtocolStage.HISTORICAL and set(manifest["games"]) == {"ft09"}:
        ft09_metrics = {
            "actions": aggregate_metrics["full_actions"],
            "levels": aggregate_metrics["full_levels_completed"],
            "max_level": max((item["max_level"] for item in full_metrics), default=0),
            "wins": aggregate_metrics["full_wins"],
            "protected_route_preemptions": sum(
                item["protected_route_preemptions"] for item in full_metrics
            ),
        }
        aggregate_metrics["ft09_non_regression"] = ft09_non_regression(ft09_metrics)
        aggregate_metrics["ft09_efficiency_gain"] = ft09_efficiency_gain(ft09_metrics)
        scientific_gate = bool(aggregate_metrics["ft09_non_regression"])
    else:
        scientific_gate = games_with_progress >= 1
    passed = bool(integrity_passed and scientific_gate)
    report_core = {
        "format_version": RUN_REPORT_FORMAT,
        "status": "PASS_CAUSAL_PAIRED_GATE" if passed else "FAIL_CAUSAL_PAIRED_GATE",
        "protocol_checksum": manifest["protocol_checksum"],
        "experiment_manifest_checksum": manifest["manifest_checksum"],
        "replay_receipt_checksum": replay_receipt["receipt_checksum"],
        "stage": stage.value,
        "authority": authority,
        "runtime": runtime_status,
        "checks": checks,
        "metrics": aggregate_metrics,
        "conditions": conditions,
        "storage": storage_budget.snapshot(),
        "passed": passed,
        "holdout_opened": False,
        "production_authority": False,
    }
    report = _signed(report_core, "report_checksum")
    receipt = _signed(
        {
            "format_version": RECEIPT_FORMAT,
            "kind": "paired_run",
            "stage": stage.value,
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "experiment_manifest_checksum": manifest["manifest_checksum"],
            "replay_receipt_checksum": replay_receipt["receipt_checksum"],
            "report_checksum": report["report_checksum"],
            "metrics": aggregate_metrics,
            "maximum_artifact_bytes": storage_budget.maximum_bytes,
            "reason": report["status"],
        },
        "receipt_checksum",
    )
    condition_rows = [
        {
            "game_id": condition["game_id"],
            "seed": condition["seed"],
            "strict_prestate_pairing": condition["strict_prestate_pairing"],
            "arms": {
                name: arm["metrics"] for name, arm in condition["arms"].items()
            },
        }
        for condition in conditions
    ]
    _write_jsonl_once(
        destination / "conditions.jsonl",
        condition_rows,
        storage_budget=storage_budget,
    )
    _write_json_once(
        destination / "paired_report.json", report, storage_budget=storage_budget
    )
    _write_json_once(
        destination / "gate_receipt.json", receipt, storage_budget=storage_budget
    )
    return report


def experiment_status(
    *,
    manifest_path: str | Path,
    replay_receipt_path: str | Path | None = None,
    gate_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_experiment_manifest(manifest_path, root=root)
    result: dict[str, Any] = {
        "format_version": EXPERIMENT_FORMAT,
        "manifest_valid": True,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "stage": manifest["stage"],
        "scientific_claims_authorized": manifest["scientific_claims_authorized"],
        "replay_receipt": None,
        "gate_receipt": None,
        "holdout_opened": False,
    }
    for key, path in (
        ("replay_receipt", replay_receipt_path),
        ("gate_receipt", gate_receipt_path),
    ):
        if path is None:
            continue
        receipt = load_receipt(path)
        if receipt.get("experiment_manifest_checksum") != manifest["manifest_checksum"]:
            raise ValueError(f"{key} belongs to another experiment")
        result[key] = {
            "passed": receipt["passed"],
            "receipt_checksum": receipt["receipt_checksum"],
            "reason": receipt["reason"],
        }
    return result


__all__ = [
    "BUNDLE_PLAN_FORMAT",
    "CORE_CODE_PATHS",
    "DEFAULT_ARMS",
    "EXPERIMENT_FORMAT",
    "ExperimentalCausalController",
    "PROGRAM_REGISTRY_FORMAT",
    "RECEIPT_FORMAT",
    "SUPPORTED_ARMS",
    "StaticPosteriorRuntime",
    "experiment_status",
    "freeze_experiment",
    "load_bundle_plan",
    "load_experiment_manifest",
    "load_program_registry",
    "load_receipt",
    "run_experiment",
    "run_replay",
    "seal_bundle_plan",
    "seal_program_registry",
]
