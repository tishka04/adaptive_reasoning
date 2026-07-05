"""Action executor — translates operator calls into primitive game actions.

The executor is the *only* module that touches the raw ARC action space.
Everything upstream speaks operators.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..schemas import (
    GameObservation,
    MacroAction,
    Operator,
    OperatorCall,
    OperatorKind,
    PlannedAction,
    PrimitiveAction,
)
from ..mechanics.operator_inducer import OperatorInducer

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Convert operator calls into primitive ARC actions."""

    def __init__(self, inducer: OperatorInducer) -> None:
        self.inducer = inducer
        self._macro_library: Dict[str, MacroAction] = {}
        self._execution_count: int = 0

    def set_macros(self, macros: Dict[str, MacroAction]) -> None:
        self._macro_library = macros

    def next_primitive(
        self,
        operator_call: OperatorCall,
        obs: GameObservation,
    ) -> PrimitiveAction:
        """Resolve one operator call into a primitive action.

        Falls back to the operator's mapped primitive_action.
        """
        self._execution_count += 1
        op = self.inducer.operators.get(operator_call.operator_id)

        # ── Macro execution ──
        if operator_call.operator_id.startswith("macro_"):
            macro_id = operator_call.args.get("macro_id", "")
            macro = self._macro_library.get(macro_id)
            if macro and macro.steps:
                step_idx = operator_call.args.get("step_idx", 0)
                if step_idx < len(macro.steps):
                    step = macro.steps[step_idx]
                    if step.primitive:
                        return step.primitive
            return PrimitiveAction(name="ACTION1")  # fallback

        # ── Raw action passthrough ──
        if operator_call.operator_id.startswith("raw_click_"):
            action_name = operator_call.args.get("action", "ACTION6")
            x = operator_call.args.get("x")
            y = operator_call.args.get("y")
            return PrimitiveAction(name=action_name, x=x, y=y)

        if operator_call.operator_id.startswith("probe_"):
            action_name = operator_call.args.get("action", "ACTION1")
            return PrimitiveAction(name=action_name)

        if operator_call.operator_id.startswith("experiment_"):
            action_name = operator_call.args.get("action", "ACTION1")
            return PrimitiveAction(name=action_name)

        if operator_call.operator_id.startswith("unstick_"):
            action_name = operator_call.args.get("action", "ACTION1")
            return PrimitiveAction(name=action_name)

        if operator_call.operator_id.startswith("seq_step_"):
            action_name = operator_call.args.get("action", "ACTION1")
            return PrimitiveAction(name=action_name)

        if operator_call.operator_id.startswith("seq_"):
            action_name = operator_call.args.get("action", "ACTION1")
            return PrimitiveAction(name=action_name)

        # ── Operator-mapped primitives ──
        if op is not None:
            if op.kind == OperatorKind.MOVE:
                return PrimitiveAction(name=op.primitive_action or "ACTION1")

            if op.kind == OperatorKind.CLICK:
                x = operator_call.args.get("x", op.primitive_x)
                y = operator_call.args.get("y", op.primitive_y)
                return PrimitiveAction(
                    name=op.primitive_action or "ACTION6",
                    x=x, y=y,
                )

            if op.kind == OperatorKind.GLOBAL_TRANSFORM:
                return PrimitiveAction(name=op.primitive_action or "ACTION1")

            if op.kind == OperatorKind.AVOID:
                # Use the safest movement operator
                move_ops = self.inducer.get_movement_ops()
                safe = [m for m in move_ops if m.risk_estimate < 0.2]
                if safe:
                    best = max(safe, key=lambda m: m.confidence)
                    return PrimitiveAction(name=best.primitive_action or "ACTION1")

            # Generic fallback for known operators
            if op.primitive_action:
                return PrimitiveAction(name=op.primitive_action)

        # ── Final fallback ──
        logger.warning(
            f"Executor: no mapping for {operator_call.operator_id}, "
            f"falling back to ACTION1"
        )
        return PrimitiveAction(name="ACTION1")
