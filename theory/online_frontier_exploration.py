"""Online frontier-oriented exploration for stalled ARC episodes.

The explorer has no access to game identifiers, level indices, rewards, or
solutions.  It observes only the live grid, legal actions, branch-progress
diagnostics, and the transition produced by its own interventions.

SAGE.9v turns a sterile branch into a bounded scientific phase:

* detect repeated states or a lack of terminal progress;
* enumerate concrete, parameterized actuators rather than action names only;
* describe clicked objects by palette-invariant structural roles;
* prioritize state/action and object/action pairs that have not been tested;
* continue a productive intervention for a short, bounded burst;
* credit only observed novel effects, states, or terminal outcomes.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


ActionData = Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class FrontierExperimentSelection:
    """One concrete intervention selected at a detected frontier."""

    action_name: str
    action_data: Dict[str, Any]
    frontier_id: str
    state_signature: str
    actuator_signature: str
    target_role_signature: str
    information_score: float
    sequence_id: str
    sequence_step: int
    sequence_limit: int
    state_action_untested: bool
    actuator_untested: bool
    object_role_untested: bool
    reason: str


@dataclass
class _ActuatorEvidence:
    trials: int = 0
    noops: int = 0
    unsafe_outcomes: int = 0
    terminal_outcomes: int = 0
    effect_signatures: Counter[str] = field(default_factory=Counter)


class OnlineFrontierExplorer:
    """Select bounded information-seeking actions when progress stalls."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_stagnant_steps: int = 6,
        max_experiments_per_state: int = 8,
        max_sequence_actions: int = 3,
        max_trials_per_actuator: int = 2,
        minimum_failed_branches: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.minimum_stagnant_steps = max(
            1,
            int(minimum_stagnant_steps),
        )
        self.max_experiments_per_state = max(
            1,
            int(max_experiments_per_state),
        )
        self.max_sequence_actions = max(1, int(max_sequence_actions))
        self.max_trials_per_actuator = max(
            1,
            int(max_trials_per_actuator),
        )
        self.minimum_failed_branches = max(
            0,
            int(minimum_failed_branches),
        )

        self._state_visits: Counter[str] = Counter()
        self._state_action_trials: Counter[Tuple[str, str]] = Counter()
        self._state_experiments: Counter[str] = Counter()
        self._actuator_evidence: Dict[str, _ActuatorEvidence] = {}
        self._tested_target_roles: set[str] = set()
        self._seen_effects: set[str] = set()
        self._seen_states: set[str] = set()
        self._frontier_states: set[str] = set()
        self._target_role_actions: Dict[str, set[str]] = defaultdict(set)

        self._pending: FrontierExperimentSelection | None = None
        self._active_sequence_id = ""
        self._active_sequence_step = 0
        self._active_sequence_remaining = 0
        self._sequence_serial = 0
        self._branches_started = 0
        self._failed_branches = 0
        self._branch_terminal_progress = False
        self._terminal_progress_observed = False

        self._states_assessed = 0
        self._stagnation_detections = 0
        self._frontier_entries = 0
        self._experiments = 0
        self._sequence_actions = 0
        self._multi_step_sequences = 0
        self._untested_state_actions = 0
        self._untested_actuator_actions = 0
        self._untested_object_actions = 0
        self._productive_experiments = 0
        self._noop_experiments = 0
        self._unsafe_experiments = 0
        self._novel_effects = 0
        self._novel_states = 0
        self._terminal_credits = 0
        self._information_gain = 0.0

    def select(
        self,
        *,
        current_grid: Any,
        available_actions: Sequence[str],
        available_action_candidates: Sequence[Any] | None,
        branch_diagnostics: Mapping[str, Any],
    ) -> FrontierExperimentSelection | None:
        """Return the most informative safe-looking concrete intervention."""
        if not self.enabled:
            return None
        if (
            self._terminal_progress_observed
            or self._failed_branches < self.minimum_failed_branches
        ):
            return None
        grid = np.asarray(current_grid, dtype=np.int32)
        if grid.ndim != 2 or grid.size == 0:
            return None

        self._states_assessed += 1
        state_signature = _state_signature(grid)
        self._state_visits[state_signature] += 1
        self._seen_states.add(state_signature)
        in_active_sequence = bool(self._active_sequence_remaining > 0)
        stagnant = self._is_stagnant(
            state_signature,
            branch_diagnostics,
        )
        if not stagnant and not in_active_sequence:
            return None
        if stagnant:
            self._stagnation_detections += 1
        if (
            self._state_experiments[state_signature]
            >= self.max_experiments_per_state
        ):
            if in_active_sequence:
                self._clear_sequence()
            return None

        candidates = _concrete_candidates(
            grid,
            available_actions,
            available_action_candidates,
        )
        if not candidates:
            return None

        ranked = []
        for action_name, action_data, actuator, target_role in candidates:
            state_trials = self._state_action_trials[
                (state_signature, actuator)
            ]
            evidence = self._actuator_evidence.get(actuator)
            actuator_trials = 0 if evidence is None else evidence.trials
            role_untested = bool(
                target_role
                and target_role not in self._tested_target_roles
            )
            noop_rate = (
                0.0
                if evidence is None or evidence.trials <= 0
                else evidence.noops / evidence.trials
            )
            unsafe_rate = (
                0.0
                if evidence is None or evidence.trials <= 0
                else evidence.unsafe_outcomes / evidence.trials
            )
            effect_diversity = (
                0 if evidence is None else len(evidence.effect_signatures)
            )
            if actuator_trials >= self.max_trials_per_actuator:
                continue
            score = (
                8.0 * float(state_trials == 0)
                + 5.0 * float(actuator_trials == 0)
                + 3.0 * float(role_untested)
                + 1.5 * float(effect_diversity == 0)
                + 0.5 * float(in_active_sequence)
                - 4.0 * noop_rate
                - 10.0 * unsafe_rate
                - 0.25 * float(state_trials)
            )
            ranked.append((
                score,
                -state_trials,
                -actuator_trials,
                action_name,
                repr(action_data),
                action_data,
                actuator,
                target_role,
                state_trials == 0,
                actuator_trials == 0,
                role_untested,
            ))
        if not ranked:
            self._clear_sequence()
            return None
        ranked.sort(reverse=True)
        (
            score,
            _,
            _,
            action_name,
            _,
            action_data,
            actuator,
            target_role,
            state_untested,
            actuator_untested,
            role_untested,
        ) = ranked[0]

        if not self._active_sequence_id:
            self._sequence_serial += 1
            self._active_sequence_id = (
                f"frontier-sequence-{self._sequence_serial:04d}"
            )
            self._active_sequence_step = 0
        self._active_sequence_step += 1
        frontier_id = f"frontier::{state_signature}"
        selection = FrontierExperimentSelection(
            action_name=action_name,
            action_data=dict(action_data),
            frontier_id=frontier_id,
            state_signature=state_signature,
            actuator_signature=actuator,
            target_role_signature=target_role,
            information_score=float(score),
            sequence_id=self._active_sequence_id,
            sequence_step=self._active_sequence_step,
            sequence_limit=self.max_sequence_actions,
            state_action_untested=bool(state_untested),
            actuator_untested=bool(actuator_untested),
            object_role_untested=bool(role_untested),
            reason=(
                "stagnant frontier: maximize causal information over "
                "untested actuator/object interventions"
            ),
        )
        self._pending = selection
        self._state_action_trials[(state_signature, actuator)] += 1
        self._state_experiments[state_signature] += 1
        self._experiments += 1
        self._sequence_actions += 1
        if state_untested:
            self._untested_state_actions += 1
        if actuator_untested:
            self._untested_actuator_actions += 1
        if role_untested:
            self._untested_object_actions += 1
        if state_signature not in self._frontier_states:
            self._frontier_states.add(state_signature)
            self._frontier_entries += 1
        return selection

    def observe_transition(
        self,
        *,
        grid_before: Any,
        grid_after: Any,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        no_effect: bool,
        game_over: bool,
        terminal_success: bool,
    ) -> Dict[str, Any]:
        """Credit only information physically observed after our action."""
        pending = self._pending
        self._pending = None
        if (
            pending is None
            or pending.action_name != str(action_name)
            or dict(pending.action_data) != dict(action_data or {})
        ):
            return {"observed": False}

        before = np.asarray(grid_before, dtype=np.int32)
        after = np.asarray(grid_after, dtype=np.int32)
        effect_signature = _effect_signature(before, after)
        after_signature = _state_signature(after)
        evidence = self._actuator_evidence.setdefault(
            pending.actuator_signature,
            _ActuatorEvidence(),
        )
        evidence.trials += 1
        evidence.noops += int(bool(no_effect))
        evidence.unsafe_outcomes += int(bool(game_over))
        evidence.terminal_outcomes += int(bool(terminal_success))
        evidence.effect_signatures[effect_signature] += 1
        if pending.target_role_signature:
            self._tested_target_roles.add(
                pending.target_role_signature
            )
            self._target_role_actions[
                pending.target_role_signature
            ].add(pending.actuator_signature)

        novel_effect = effect_signature not in self._seen_effects
        novel_state = after_signature not in self._seen_states
        self._seen_effects.add(effect_signature)
        self._seen_states.add(after_signature)
        gain = (
            float(novel_effect)
            + float(novel_state)
            + 2.0 * float(terminal_success)
        )
        self._information_gain += gain
        self._novel_effects += int(novel_effect)
        self._novel_states += int(novel_state)
        self._terminal_credits += int(bool(terminal_success))
        self._noop_experiments += int(bool(no_effect))
        self._unsafe_experiments += int(bool(game_over))
        productive = bool(
            terminal_success
            or (not no_effect and not game_over and (novel_effect or novel_state))
        )
        self._productive_experiments += int(productive)

        if (
            productive
            and not terminal_success
            and not game_over
            and pending.sequence_step < self.max_sequence_actions
        ):
            if pending.sequence_step == 1:
                self._multi_step_sequences += 1
            self._active_sequence_remaining = (
                self.max_sequence_actions - pending.sequence_step
            )
        elif self._active_sequence_remaining > 0 and not game_over:
            self._active_sequence_remaining -= 1
            if self._active_sequence_remaining <= 0:
                self._clear_sequence()
        else:
            self._clear_sequence()

        return {
            "observed": True,
            "productive": productive,
            "novel_effect": novel_effect,
            "novel_state": novel_state,
            "terminal_credit": bool(terminal_success),
            "information_gain": gain,
            "effect_signature": effect_signature,
        }

    def start_branch(self) -> None:
        """End any censored burst while retaining cross-reset evidence."""
        if self._branches_started > 0:
            if self._branch_terminal_progress:
                self._failed_branches = 0
            else:
                self._failed_branches += 1
        self._branches_started += 1
        self._branch_terminal_progress = False
        self._pending = None
        self._clear_sequence()

    def note_transition(self, *, terminal_success: bool) -> None:
        """Suspend upstream exploration once any existing skill progresses."""
        if terminal_success:
            self._branch_terminal_progress = True
            self._terminal_progress_observed = True
            self._pending = None
            self._clear_sequence()

    @property
    def active_sequence(self) -> bool:
        return bool(self._active_sequence_remaining > 0)

    def cancel_pending(self) -> None:
        """Cancel a selection vetoed before execution without fabricating data."""
        pending = self._pending
        if pending is None:
            return
        key = (pending.state_signature, pending.actuator_signature)
        self._state_action_trials[key] = max(
            0,
            self._state_action_trials[key] - 1,
        )
        self._state_experiments[pending.state_signature] = max(
            0,
            self._state_experiments[pending.state_signature] - 1,
        )
        self._experiments = max(0, self._experiments - 1)
        self._sequence_actions = max(0, self._sequence_actions - 1)
        if pending.state_action_untested:
            self._untested_state_actions = max(
                0,
                self._untested_state_actions - 1,
            )
        if pending.actuator_untested:
            self._untested_actuator_actions = max(
                0,
                self._untested_actuator_actions - 1,
            )
        if pending.object_role_untested:
            self._untested_object_actions = max(
                0,
                self._untested_object_actions - 1,
            )
        self._pending = None
        self._clear_sequence()

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "minimum_stagnant_steps": self.minimum_stagnant_steps,
            "max_experiments_per_state": self.max_experiments_per_state,
            "max_sequence_actions": self.max_sequence_actions,
            "max_trials_per_actuator": self.max_trials_per_actuator,
            "minimum_failed_branches": self.minimum_failed_branches,
            "branches_started": self._branches_started,
            "failed_branches": self._failed_branches,
            "terminal_progress_observed": (
                self._terminal_progress_observed
            ),
            "states_assessed": self._states_assessed,
            "stagnation_detections": self._stagnation_detections,
            "frontier_entries": self._frontier_entries,
            "frontier_states": len(self._frontier_states),
            "experiments": self._experiments,
            "sequence_actions": self._sequence_actions,
            "multi_step_sequences": self._multi_step_sequences,
            "untested_state_actions": self._untested_state_actions,
            "untested_actuator_actions": self._untested_actuator_actions,
            "untested_object_actions": self._untested_object_actions,
            "productive_experiments": self._productive_experiments,
            "noop_experiments": self._noop_experiments,
            "unsafe_experiments": self._unsafe_experiments,
            "novel_effects": self._novel_effects,
            "novel_states": self._novel_states,
            "terminal_credits": self._terminal_credits,
            "information_gain": round(self._information_gain, 4),
            "actuator_models": len(self._actuator_evidence),
            "tested_object_roles": len(self._tested_target_roles),
            "active_sequence_id": self._active_sequence_id,
            "actuators": {
                actuator: {
                    "trials": evidence.trials,
                    "noops": evidence.noops,
                    "unsafe_outcomes": evidence.unsafe_outcomes,
                    "terminal_outcomes": evidence.terminal_outcomes,
                    "effect_signatures": dict(evidence.effect_signatures),
                }
                for actuator, evidence in self._actuator_evidence.items()
            },
        }

    def _is_stagnant(
        self,
        state_signature: str,
        diagnostics: Mapping[str, Any],
    ) -> bool:
        branch_actions = int(diagnostics.get("branch_actions", 0) or 0)
        if branch_actions < self.minimum_stagnant_steps:
            return False
        terminal_stall = int(
            diagnostics.get("actions_since_terminal_improvement", 0)
            or 0
        )
        max_hash_repeat = int(
            diagnostics.get("max_hash_repeat", 0) or 0
        )
        unique_states = int(
            diagnostics.get("unique_states_in_window", 0) or 0
        )
        window_actions = int(
            diagnostics.get("window_actions", branch_actions) or 0
        )
        repeated_state = bool(
            max_hash_repeat >= max(
                3,
                self.minimum_stagnant_steps // 2,
            )
        )
        low_novelty_cycle = bool(
            window_actions >= self.minimum_stagnant_steps * 2
            and unique_states
            <= max(2, int(window_actions * 0.35))
        )
        recurrent_current_state = bool(
            self._state_visits[state_signature] >= 3
            and max_hash_repeat >= 2
        )
        return bool(
            terminal_stall >= self.minimum_stagnant_steps
            and (
                repeated_state
                or low_novelty_cycle
                or recurrent_current_state
            )
        )

    def _clear_sequence(self) -> None:
        self._active_sequence_id = ""
        self._active_sequence_step = 0
        self._active_sequence_remaining = 0


def _concrete_candidates(
    grid: np.ndarray,
    available_actions: Sequence[str],
    raw_candidates: Sequence[Any] | None,
) -> Tuple[Tuple[str, Dict[str, Any], str, str], ...]:
    allowed = {
        str(action)
        for action in available_actions
        if str(action) and str(action) != "RESET"
    }
    concrete: list[Tuple[str, Dict[str, Any]]] = []
    for raw in tuple(raw_candidates or ()):
        name = str(getattr(raw, "name", ""))
        if name not in allowed:
            continue
        data = dict(getattr(raw, "action_args", {}) or {})
        concrete.append((name, data))
    represented = {name for name, _ in concrete}
    for name in sorted(allowed - represented):
        concrete.append((name, {}))

    result = []
    seen = set()
    for name, data in concrete:
        normalized = _normalized_action_data(data)
        identity = (name, normalized)
        if identity in seen:
            continue
        seen.add(identity)
        target_role = _target_role_signature(grid, data)
        actuator = _actuator_signature(name, data, target_role)
        result.append((name, dict(data), actuator, target_role))
    return tuple(result)


def _normalized_action_data(data: Mapping[str, Any]) -> ActionData:
    normalized = []
    for key, value in sorted(data.items(), key=lambda item: str(item[0])):
        if isinstance(value, (int, float, str, bool)) or value is None:
            stable = value
        else:
            stable = repr(value)
        normalized.append((str(key), stable))
    return tuple(normalized)


def _actuator_signature(
    action_name: str,
    action_data: Mapping[str, Any],
    target_role: str,
) -> str:
    argument_schema = tuple(sorted(str(key) for key in action_data))
    payload = (str(action_name), argument_schema, target_role)
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _target_role_signature(
    grid: np.ndarray,
    action_data: Mapping[str, Any],
) -> str:
    if "x" not in action_data or "y" not in action_data:
        return ""
    try:
        x = int(action_data["x"])
        y = int(action_data["y"])
    except (TypeError, ValueError):
        return ""
    height, width = grid.shape
    if not (0 <= x < width and 0 <= y < height):
        return "outside_grid"
    background = _background_value(grid)
    value = int(grid[y, x])
    occupancy = "background" if value == background else "object"
    component = _component(grid, x, y, value)
    xs = [coordinate[0] for coordinate in component]
    ys = [coordinate[1] for coordinate in component]
    component_width = max(xs) - min(xs) + 1
    component_height = max(ys) - min(ys) + 1
    area = len(component)
    area_bucket = (
        "single"
        if area == 1
        else "small"
        if area <= 4
        else "medium"
        if area <= 15
        else "large"
    )
    position = (
        min(2, (3 * x) // max(1, width)),
        min(2, (3 * y) // max(1, height)),
    )
    payload = (
        occupancy,
        area_bucket,
        min(component_width, 5),
        min(component_height, 5),
        position,
    )
    return repr(payload)


def _component(
    grid: np.ndarray,
    start_x: int,
    start_y: int,
    value: int,
) -> Tuple[Tuple[int, int], ...]:
    height, width = grid.shape
    pending = [(start_x, start_y)]
    seen = {(start_x, start_y)}
    while pending:
        x, y = pending.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in seen or int(grid[ny, nx]) != value:
                continue
            seen.add((nx, ny))
            pending.append((nx, ny))
    return tuple(sorted(seen))


def _background_value(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _state_signature(grid: np.ndarray) -> str:
    payload = (
        tuple(int(value) for value in grid.shape),
        grid.astype(np.int32, copy=False).tobytes(),
    )
    return hashlib.sha1(repr(payload).encode("latin1")).hexdigest()[:16]


def _effect_signature(before: np.ndarray, after: np.ndarray) -> str:
    if before.shape != after.shape:
        payload = ("shape_change", before.shape, after.shape)
    else:
        changed = np.argwhere(before != after)
        if not len(changed):
            payload = ("noop",)
        else:
            y_min, x_min = changed.min(axis=0)
            y_max, x_max = changed.max(axis=0)
            changed_count = len(changed)
            count_bucket = (
                "one"
                if changed_count == 1
                else "few"
                if changed_count <= 4
                else "many"
            )
            before_values = len(set(int(v) for v in before[before != after]))
            after_values = len(set(int(v) for v in after[before != after]))
            payload = (
                "change",
                count_bucket,
                int(x_max - x_min + 1),
                int(y_max - y_min + 1),
                before_values,
                after_values,
            )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "FrontierExperimentSelection",
    "OnlineFrontierExplorer",
]
