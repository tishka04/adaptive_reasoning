"""Reactive execution for V4."""

from __future__ import annotations

import random

from ..schemas import PrimitiveAction


class ReactiveController:
    """Cheap grounded control when full committee reasoning is unnecessary."""

    def act(self, obs, memory, proposal=None):
        if proposal is not None and proposal.primitive_plan:
            return proposal.primitive_plan[0]

        player = obs.best_player
        escape_ops = memory.game.inducer.get_by_kind("escape")
        if escape_ops:
            best = max(escape_ops, key=lambda item: item.confidence - item.risk_estimate)
            return PrimitiveAction(best.primitive_action)

        transform_ops = memory.game.inducer.get_by_kind("global_transform")
        if transform_ops and obs.surprise.total > 0.35:
            best = max(transform_ops, key=lambda item: item.confidence)
            return PrimitiveAction(best.primitive_action)

        click_ops = memory.game.inducer.get_by_kind("click")
        if click_ops and obs.objects and any(action == "ACTION6" for action in obs.available_actions):
            target = min(obs.objects, key=lambda obj: obj.area)
            return PrimitiveAction(
                "ACTION6",
                x=int(round(target.center[1])),
                y=int(round(target.center[0])),
            )

        move_ops = memory.game.inducer.get_by_kind("move")
        if player is not None and move_ops:
            best = max(move_ops, key=lambda item: item.confidence - 0.5 * item.risk_estimate)
            return PrimitiveAction(best.primitive_action)

        non_reset = [action for action in obs.available_actions if action != "RESET"]
        return PrimitiveAction(random.choice(non_reset) if non_reset else "ACTION1")
