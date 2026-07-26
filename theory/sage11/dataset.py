"""Versioned SAGE.11 transition dataset and fixed collection policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .atoms import frame_diff_atoms, observation_atoms
from .splits import ArtifactPurpose, SAGE11_SPLITS, short_game_id


DATASET_FORMAT_VERSION = "sage11-transition-v1"
MIXTURE_POLICY_VERSION = "sage11-mixture-v1"
DEFAULT_TARGET_TRANSITIONS = 100_000
DEFAULT_PER_GAME_CAP = 8_000
MINIMUM_STRONG_TERMINAL_EVENTS = 100


@dataclass(frozen=True)
class MixturePolicy:
    """Pre-registered transition collection mixture."""

    active_controller: float = 0.70
    uniform_legal: float = 0.20
    frontier_stall_probe: float = 0.10
    version: str = MIXTURE_POLICY_VERSION

    def __post_init__(self) -> None:
        total = (
            self.active_controller
            + self.uniform_legal
            + self.frontier_stall_probe
        )
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("SAGE.11 mixture weights must sum to 1")
        if min(
            self.active_controller,
            self.uniform_legal,
            self.frontier_stall_probe,
        ) <= 0.0:
            raise ValueError("SAGE.11 mixture weights must be positive")

    def arm_for(
        self,
        *,
        game_id: str,
        seed: int,
        reset_index: int,
        step_index: int,
    ) -> str:
        """Assign an arm deterministically, independent of transition outcome."""
        key = (
            f"{self.version}|{short_game_id(game_id)}|{int(seed)}|"
            f"{int(reset_index)}|{int(step_index)}"
        ).encode("utf-8")
        unit = int.from_bytes(
            hashlib.sha256(key).digest()[:8],
            "big",
        ) / float(2**64)
        if unit < self.active_controller:
            return "active_controller"
        if unit < self.active_controller + self.uniform_legal:
            return "uniform_legal"
        return "frontier_stall_probe"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressLabels:
    """Strong and explicitly weak supervision for one transition.

    Only an observed level completion or WIN is a strong progress/terminal
    label.  Frontier credit, subgoal-graph advances, route confirmations, and
    sub-effect relays are weak progress labels and are never counted toward the
    terminal-head activation threshold.
    """

    terminal_event: bool = False
    level_completed: bool = False
    won: bool = False
    frontier_credit: bool = False
    subgoal_graph_advance: bool = False
    route_confirmation: bool = False
    subeffect_relay: bool = False

    @property
    def strong_progress(self) -> bool:
        return bool(self.terminal_event or self.level_completed or self.won)

    @property
    def weak_progress(self) -> bool:
        return bool(
            self.frontier_credit
            or self.subgoal_graph_advance
            or self.route_confirmation
            or self.subeffect_relay
        )

    @property
    def progress_target(self) -> float:
        return float(self.strong_progress or self.weak_progress)

    @property
    def strength(self) -> str:
        if self.strong_progress:
            return "strong"
        if self.weak_progress:
            return "weak"
        return "negative"


@dataclass(frozen=True)
class NeuroTransition:
    """Hashable model-ready transition with no implicit source weights."""

    game_id: str
    seed: int
    reset_index: int
    step_index: int
    policy_arm: str
    action_name: str
    action_data: Mapping[str, Any]
    atoms_before: Tuple[str, ...]
    atoms_after: Tuple[str, ...]
    effect_atoms: Tuple[str, ...]
    changed: bool
    noop: bool
    unsafe: bool
    labels: ProgressLabels = field(default_factory=ProgressLabels)
    format_version: str = DATASET_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.policy_arm not in {
            "active_controller",
            "uniform_legal",
            "frontier_stall_probe",
        }:
            raise ValueError(f"unknown SAGE.11 policy arm {self.policy_arm}")
        if self.format_version != DATASET_FORMAT_VERSION:
            raise ValueError("unsupported SAGE.11 transition version")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action_data"] = _json_safe_mapping(self.action_data)
        payload["atoms_before"] = list(self.atoms_before)
        payload["atoms_after"] = list(self.atoms_after)
        payload["effect_atoms"] = list(self.effect_atoms)
        return payload

    @property
    def transition_signature(self) -> str:
        """Deduplicate the behavioral transition, not run metadata."""
        payload = {
            "game_id": short_game_id(self.game_id),
            "action_name": self.action_name,
            "action_data": _json_safe_mapping(self.action_data),
            "atoms_before": list(self.atoms_before),
            "atoms_after": list(self.atoms_after),
            "effect_atoms": list(self.effect_atoms),
        }
        return _checksum_json(payload)


@dataclass(frozen=True)
class DatasetShard:
    path: str
    sha256: str
    transitions: int
    games: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetManifest:
    """Checksummed manifest kept alongside Git-LFS transition shards."""

    shards: Tuple[DatasetShard, ...]
    split_registry_checksum: str
    mixture_policy: Mapping[str, Any]
    target_transitions: int
    per_game_cap: int
    total_transitions: int
    strong_terminal_events: int
    weak_progress_events: int
    game_counts: Mapping[str, int]
    policy_counts: Mapping[str, int]
    action6_argument_coverage: Mapping[str, int]
    dataset_format_version: str = DATASET_FORMAT_VERSION
    legacy_weights_loaded: bool = False

    @property
    def terminal_head_enabled(self) -> bool:
        return self.strong_terminal_events >= MINIMUM_STRONG_TERMINAL_EVENTS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_format_version": self.dataset_format_version,
            "split_registry_checksum": self.split_registry_checksum,
            "mixture_policy": dict(self.mixture_policy),
            "target_transitions": int(self.target_transitions),
            "per_game_cap": int(self.per_game_cap),
            "total_transitions": int(self.total_transitions),
            "strong_terminal_events": int(self.strong_terminal_events),
            "weak_progress_events": int(self.weak_progress_events),
            "terminal_head_enabled": self.terminal_head_enabled,
            "game_counts": dict(sorted(self.game_counts.items())),
            "policy_counts": dict(sorted(self.policy_counts.items())),
            "action6_argument_coverage": dict(
                sorted(self.action6_argument_coverage.items())
            ),
            "transition_signature_deduplication": True,
            "legacy_weights_loaded": bool(self.legacy_weights_loaded),
            "shards": [
                {
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "transitions": shard.transitions,
                    "games": list(shard.games),
                }
                for shard in self.shards
            ],
        }

    @property
    def checksum(self) -> str:
        return _checksum_json(self.to_dict())


class Sage11DatasetBuilder:
    """Bounded collector enforcing source splits, caps, and deduplication."""

    def __init__(
        self,
        *,
        purpose: ArtifactPurpose = ArtifactPurpose.TRAIN,
        mixture_policy: MixturePolicy | None = None,
        target_transitions: int = DEFAULT_TARGET_TRANSITIONS,
        per_game_cap: int = DEFAULT_PER_GAME_CAP,
    ) -> None:
        self.purpose = purpose
        self.mixture_policy = mixture_policy or MixturePolicy()
        self.target_transitions = max(1, int(target_transitions))
        self.per_game_cap = max(1, int(per_game_cap))
        self._records: list[NeuroTransition] = []
        self._signatures: set[str] = set()
        self._game_counts: Counter[str] = Counter()
        self._policy_counts: Counter[str] = Counter()
        self._action6_coverage: Counter[str] = Counter()
        self._rejected_duplicates = 0
        self._rejected_caps = 0

    @property
    def records(self) -> Tuple[NeuroTransition, ...]:
        return tuple(self._records)

    def add(self, transition: NeuroTransition) -> bool:
        game = short_game_id(transition.game_id)
        SAGE11_SPLITS.assert_authorized([game], purpose=self.purpose)
        if self._game_counts[game] >= self.per_game_cap:
            self._rejected_caps += 1
            return False
        signature = transition.transition_signature
        if signature in self._signatures:
            self._rejected_duplicates += 1
            return False
        if len(self._records) >= self.target_transitions:
            return False
        self._signatures.add(signature)
        self._records.append(transition)
        self._game_counts[game] += 1
        self._policy_counts[transition.policy_arm] += 1
        if transition.action_name == "ACTION6":
            data = _json_safe_mapping(transition.action_data)
            keys = ",".join(sorted(data)) or "no_arguments"
            self._action6_coverage[f"keys:{keys}"] += 1
            if "x" in data and "y" in data:
                self._action6_coverage[
                    f"xy:{data['x']}:{data['y']}"
                ] += 1
        return True

    def write_jsonl_shard(self, path: str | Path) -> DatasetShard:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                record.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in self._records
        ]
        content = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        target.write_bytes(content)
        return DatasetShard(
            path=target.as_posix(),
            sha256=hashlib.sha256(content).hexdigest(),
            transitions=len(self._records),
            games=tuple(sorted(self._game_counts)),
        )

    def manifest(
        self,
        shards: Sequence[DatasetShard] = (),
    ) -> DatasetManifest:
        return DatasetManifest(
            shards=tuple(shards),
            split_registry_checksum=SAGE11_SPLITS.checksum,
            mixture_policy=self.mixture_policy.to_dict(),
            target_transitions=self.target_transitions,
            per_game_cap=self.per_game_cap,
            total_transitions=len(self._records),
            strong_terminal_events=sum(
                item.labels.strong_progress for item in self._records
            ),
            weak_progress_events=sum(
                item.labels.weak_progress for item in self._records
            ),
            game_counts=dict(self._game_counts),
            policy_counts=dict(self._policy_counts),
            action6_argument_coverage=dict(self._action6_coverage),
        )

    def summary(self) -> Dict[str, Any]:
        manifest = self.manifest()
        return {
            **manifest.to_dict(),
            "rejected_duplicates": self._rejected_duplicates,
            "rejected_per_game_cap": self._rejected_caps,
            "manifest_checksum": manifest.checksum,
        }


class Sage11ControllerCollector:
    """Adapt live controller updates to the frozen transition schema."""

    def __init__(
        self,
        builder: Sage11DatasetBuilder,
        *,
        game_id: str,
        seed: int,
        policy_arm: str = "active_controller",
    ) -> None:
        self.builder = builder
        self.game_id = short_game_id(game_id)
        self.seed = int(seed)
        self.policy_arm = str(policy_arm)
        self.reset_index = 0
        self.step_index = 0

    def on_reset(self) -> None:
        self.reset_index += 1
        self.step_index = 0

    def record(
        self,
        update: Any,
        *,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        terminal_event: bool,
        level_completed: bool,
        won: bool,
        unsafe: bool,
        frontier_credit: bool = False,
        subgoal_graph_advance: bool = False,
        route_confirmation: bool = False,
        subeffect_relay: bool = False,
    ) -> bool:
        transition = NeuroTransition(
            game_id=self.game_id,
            seed=self.seed,
            reset_index=self.reset_index,
            step_index=self.step_index,
            policy_arm=self.policy_arm,
            action_name=str(action_name),
            action_data=dict(action_data or {}),
            atoms_before=tuple(
                atom.key
                for atom in observation_atoms(update.record.obs_before)
            ),
            atoms_after=tuple(
                atom.key
                for atom in observation_atoms(update.record.obs_after)
            ),
            effect_atoms=tuple(
                atom.key for atom in frame_diff_atoms(update.record.diff)
            ),
            changed=bool(update.record.diff.num_changed),
            noop=not bool(update.record.diff.num_changed),
            unsafe=bool(unsafe),
            labels=ProgressLabels(
                terminal_event=bool(terminal_event),
                level_completed=bool(level_completed),
                won=bool(won),
                frontier_credit=bool(frontier_credit),
                subgoal_graph_advance=bool(subgoal_graph_advance),
                route_confirmation=bool(route_confirmation),
                subeffect_relay=bool(subeffect_relay),
            ),
        )
        self.step_index += 1
        return self.builder.add(transition)


def verify_manifest(
    manifest: DatasetManifest,
    *,
    root: str | Path = ".",
) -> None:
    """Fail closed on split drift, legacy weights, or shard corruption."""
    if manifest.split_registry_checksum != SAGE11_SPLITS.checksum:
        raise ValueError("SAGE.11 dataset was built with different splits")
    if manifest.legacy_weights_loaded:
        raise ValueError("SAGE.11 artifacts may not load M2/v4 weights")
    base = Path(root)
    for shard in manifest.shards:
        path = base / shard.path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != shard.sha256:
            raise ValueError(f"SAGE.11 shard checksum mismatch: {path}")


def _checksum_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe_mapping(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(dict(payload), sort_keys=True, default=str))


__all__ = [
    "DATASET_FORMAT_VERSION",
    "DEFAULT_PER_GAME_CAP",
    "DEFAULT_TARGET_TRANSITIONS",
    "DatasetManifest",
    "DatasetShard",
    "MINIMUM_STRONG_TERMINAL_EVENTS",
    "MixturePolicy",
    "NeuroTransition",
    "ProgressLabels",
    "Sage11DatasetBuilder",
    "Sage11ControllerCollector",
    "verify_manifest",
]
