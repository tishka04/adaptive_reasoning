"""Reusable nine-prefix reachability audit for concrete SAGE.T controllers."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import bounded_active_v9_3 as r1
from . import reachability_audit_v9 as t9_0
from . import trajectory_planning_v9_2 as t9_2
from .contracts import AbstractState, ObservedTransition
from .decision import CandidateSequence
from .replay_gate import fast_panel_from_binding_pair


def audit_winning_prefixes(
    caps: Mapping[str, int],
    *,
    controller_builder: Callable[[Mapping[str, Any]], Any],
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
        pairs_by_path = {pair.path: pair for pair in grouped[root_key]}
        root = pairs_by_path[""]
        root_panel = fast_panel_from_binding_pair(root)
        controller = controller_builder(provisional)
        history: list[ObservedTransition] = []
        names = tuple(sorted({arm.action.action_name for arm in root_panel.arms}))

        def assemble(
            *,
            seed: bool,
            state: AbstractState,
            controller: Any = controller,
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
            controller.executor.clear_cache()
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
        "decision_p95_ms": r1._quantile(
            [float(row["latency_ms"]) for row in rows],
            0.95,
        ),
    }
    result["passed"] = bool(
        result["prefixes"] == 9
        and result["exact_sequence_generated"] == 9
        and result["exact_sequence_top8"] == 9
        and result["correct_first_action"] == 9
    )
    return result


__all__ = ["audit_winning_prefixes"]
