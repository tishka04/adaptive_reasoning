"""Infer lightweight action-dynamics hypotheses from observed effects."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .game_memory import ActionProfile, GameMemory
from .hypothesis_models import ActionHypothesis, BeliefState
from .state_describer import GameObservation


class DynamicsInducer:
    """Build local action-effect beliefs from GameMemory statistics."""

    def induce(
        self,
        *,
        memory: Optional[GameMemory],
        observation: GameObservation,
        available_actions: Sequence[str],
    ) -> BeliefState:
        hypotheses: List[ActionHypothesis] = []
        for action in available_actions:
            action_name = str(action).upper()
            profile = None if memory is None else memory.action_profiles.get(action_name)
            hypotheses.append(
                self._hypothesis_for_action(
                    action_name=action_name,
                    profile=profile,
                    memory=memory,
                    observation=observation,
                )
            )
        return BeliefState(hypotheses=hypotheses)

    def _hypothesis_for_action(
        self,
        *,
        action_name: str,
        profile: Optional[ActionProfile],
        memory: Optional[GameMemory],
        observation: GameObservation,
    ) -> ActionHypothesis:
        if profile is None or profile.times_tried <= 0:
            return self._untried_hypothesis(
                action_name=action_name,
                observation=observation,
            )

        tried = max(1, int(profile.times_tried))
        change_rate = float(profile.change_rate)
        move_rate = float(profile.move_rate)
        game_over_rate = float(profile.times_caused_game_over / tried)
        win_rate = float(profile.times_caused_win / tried)
        support_conf = min(1.0, tried / 6.0)
        displacement = profile.dominant_displacement
        kind = "state_change"
        predicted_signal = "grid changes"

        if game_over_rate >= 0.30 and tried >= 2:
            kind = "hazard"
            predicted_signal = "game over risk"
        elif move_rate >= 0.20 or displacement is not None:
            kind = "movement"
            predicted_signal = "controlled object moves"
        elif action_name == "ACTION6" and self._click_is_plausible(memory, profile):
            kind = "click_activation"
            predicted_signal = "click changes a target"
        elif action_name in {"ACTION5", "ACTION7"} and change_rate >= 0.25:
            kind = "control_switch"
            predicted_signal = "active controllable object may change"
        elif change_rate < 0.08:
            kind = "noop"
            predicted_signal = "little or no visible effect"

        confidence = self._confidence_for_kind(
            kind=kind,
            support_conf=support_conf,
            change_rate=change_rate,
            move_rate=move_rate,
            game_over_rate=game_over_rate,
            win_rate=win_rate,
        )
        information_gain = max(0.05, 1.0 - support_conf)
        if kind in {"unknown", "control_switch", "click_activation"}:
            information_gain = max(information_gain, 0.35)
        risk = min(1.0, 0.05 + game_over_rate)
        if kind == "hazard":
            risk = max(risk, 0.70)
        elif kind == "noop":
            risk = max(risk, 0.15)

        return ActionHypothesis(
            action_name=action_name,
            kind=kind,
            confidence=confidence,
            support=tried,
            information_gain=information_gain,
            risk=risk,
            displacement=displacement,
            predicted_signal=predicted_signal,
            evidence={
                "change_rate": round(change_rate, 4),
                "move_rate": round(move_rate, 4),
                "game_over_rate": round(game_over_rate, 4),
                "win_rate": round(win_rate, 4),
                "level": int(getattr(observation, "level", 0) or 0),
            },
        )

    @staticmethod
    def _untried_hypothesis(
        *,
        action_name: str,
        observation: GameObservation,
    ) -> ActionHypothesis:
        """Assign a generic affordance prior before the first observation.

        ARC-AGI action names are stable enough to distinguish movement-ish
        actions from interact/click affordances, but confidence stays low
        until observed effects confirm or refute the guess.
        """
        semantics = ""
        try:
            semantics = str((observation.action_semantics or {}).get(action_name, "")).lower()
        except Exception:
            semantics = ""

        kind = "unknown"
        confidence = 0.20
        predicted_signal = "observe first effect"
        evidence = {"untried": True}

        if action_name == "ACTION6" or "click" in semantics:
            kind = "click_activation"
            confidence = 0.18
            predicted_signal = "click may change a target"
            evidence["affordance_prior"] = "click_probe"
        elif (
            action_name == "ACTION5"
            or "switch" in semantics
            or "interact" in semantics
            or "toggle" in semantics
        ):
            kind = "control_switch"
            confidence = 0.18
            predicted_signal = "control or interaction may change"
            evidence["affordance_prior"] = "interaction_probe"

        return ActionHypothesis(
            action_name=action_name,
            kind=kind,
            confidence=confidence,
            support=0,
            information_gain=1.0,
            risk=0.05,
            predicted_signal=predicted_signal,
            evidence=evidence,
        )

    @staticmethod
    def _click_is_plausible(memory: Optional[GameMemory], profile: ActionProfile) -> bool:
        if memory is not None and memory.get_effective_click_values():
            return True
        return profile.change_rate >= 0.15

    @staticmethod
    def _confidence_for_kind(
        *,
        kind: str,
        support_conf: float,
        change_rate: float,
        move_rate: float,
        game_over_rate: float,
        win_rate: float,
    ) -> float:
        if kind == "movement":
            raw = 0.25 + 0.45 * move_rate + 0.25 * support_conf
        elif kind == "control_switch":
            raw = 0.20 + 0.35 * change_rate + 0.25 * support_conf
        elif kind == "click_activation":
            raw = 0.20 + 0.35 * change_rate + 0.25 * support_conf
        elif kind == "hazard":
            raw = 0.25 + 0.55 * game_over_rate + 0.15 * support_conf
        elif kind == "noop":
            raw = 0.20 + 0.35 * support_conf + 0.20 * (1.0 - change_rate)
        else:
            raw = 0.20 + 0.35 * change_rate + 0.15 * support_conf + 0.30 * win_rate
        return max(0.05, min(0.95, raw))
