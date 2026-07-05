"""Intent-to-primitive executor for V4."""

from __future__ import annotations

import random

from ..schemas import ActionIntent, PrimitiveAction


class ActionExecutor:
    """Resolve control intents into primitive ARC actions."""

    def act(self, intent, obs, memory) -> PrimitiveAction:
        if intent is not None:
            memory.fast.load_intent(intent)

        action = self._next_from_queue(obs, memory)
        if action is None:
            non_reset = [name for name in obs.available_actions if name != "RESET"]
            action = PrimitiveAction(random.choice(non_reset) if non_reset else "ACTION1")
            operator_id = None
            active_intent = intent or ActionIntent(source="fallback")
        else:
            operator_id = getattr(self, "_last_operator_id", None)
            active_intent = memory.fast.last_intent or intent or ActionIntent(source="fallback")

        memory.fast.remember_action(action, active_intent, operator_id)
        return action

    def _next_from_queue(self, obs, memory) -> PrimitiveAction | None:
        self._last_operator_id = None
        if memory.fast.queued_primitives:
            return memory.fast.queued_primitives.popleft()

        if memory.fast.queued_operators:
            operator_id = memory.fast.queued_operators.popleft()
            operator = memory.game.inducer.operators.get(operator_id)
            self._last_operator_id = operator_id
            if operator is None:
                return None
            if memory.game.total_actions > 20 and operator.survival_score < 0.05:
                return None
            if getattr(memory.game, "learning", None) is not None and memory.game.total_actions > 12:
                utility = memory.game.learning.operator_utility.estimate(operator, memory)
                if utility < 0.10:
                    return None
                if utility < 0.22 and operator.kind not in {"move", "click"} and operator.support < 4:
                    return None
            if operator.kind == "click" and obs.objects and any(action == "ACTION6" for action in obs.available_actions):
                target_value = operator.parameters.get("target_value")
                target = next((obj for obj in obs.objects if obj.value == target_value), None)
                if target is None:
                    target = min(obs.objects, key=lambda obj: obj.area)
                return PrimitiveAction(
                    operator.primitive_action or "ACTION6",
                    x=int(round(target.center[1])),
                    y=int(round(target.center[0])),
                )
            return PrimitiveAction(operator.primitive_action or "ACTION1")

        return None
