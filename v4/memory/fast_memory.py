"""Fast, branch-local memory for V4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..schemas import ActionIntent, DissentReport, ObservationV4, PrimitiveAction, TransitionRecord


@dataclass
class FastMemory:
    """Short-timescale memory for the currently inhabited branch."""

    prev_frame: Optional[np.ndarray] = None
    prev_obs: Optional[ObservationV4] = None
    current_obs: Optional[ObservationV4] = None
    last_action: Optional[PrimitiveAction] = None
    last_operator_id: Optional[str] = None
    last_intent: Optional[ActionIntent] = None
    last_transition: Optional[TransitionRecord] = None
    last_dissent: Optional[DissentReport] = None
    current_project_id: Optional[str] = None
    current_ontology_id: Optional[str] = None
    current_phase: str = "sensory_ignorance"
    just_completed_level: bool = False
    recent_transitions: deque[TransitionRecord] = field(
        default_factory=lambda: deque(maxlen=80)
    )
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    recent_hashes: deque[int] = field(default_factory=lambda: deque(maxlen=80))
    queued_primitives: deque[PrimitiveAction] = field(
        default_factory=lambda: deque(maxlen=64)
    )
    queued_operators: deque[str] = field(default_factory=lambda: deque(maxlen=64))

    def on_observation(self, obs: ObservationV4) -> None:
        self.current_obs = obs
        self.recent_hashes.append(obs.grid_hash)

    def on_transition(self, transition: TransitionRecord) -> None:
        self.last_transition = transition
        self.recent_transitions.append(transition)
        self.just_completed_level = transition.level_completed

    def remember_action(self, action: PrimitiveAction, intent: ActionIntent, operator_id: str | None) -> None:
        self.last_action = action
        self.last_intent = intent
        self.last_operator_id = operator_id
        self.current_project_id = intent.project_id
        self.current_ontology_id = intent.ontology_id
        self.recent_actions.append(action.name)

    def load_intent(self, intent: ActionIntent) -> None:
        self.clear_plan()
        for primitive in intent.primitive_plan:
            self.queued_primitives.append(primitive)
        for operator_id in intent.operator_plan:
            self.queued_operators.append(operator_id)
        self.current_project_id = intent.project_id
        self.current_ontology_id = intent.ontology_id
        self.last_intent = intent

    def clear_plan(self) -> None:
        self.queued_primitives.clear()
        self.queued_operators.clear()

    def has_plan(self) -> bool:
        return bool(self.queued_primitives or self.queued_operators)
