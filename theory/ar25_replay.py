"""Closed-loop ar25 belief-revision prototype, driven by recorded outcomes.

This proves the loop *hypothesis -> discriminating experiment -> revision*
without any live environment: the per-action ``examples`` recorded in the
action ontology act as a replayable environment. The designer picks which
action to probe; we return a real recorded outcome for it; the theory updates.

Success is judged by ``theory.epistemic_metrics`` against the ar25 oracle,
NOT by any game score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ar25_oracle import _DEFAULT_ONTOLOGY, _DEFAULT_TASK_PROGRAM, build_ar25_oracle
from .epistemic_metrics import EpistemicScore, score_beliefs
from .experiment_designer import DiscriminatingExperimentDesigner
from .mechanic_hypothesis import GameTheory, ObservedEffect
from .revision import revise_theory
from .role_hypotheses import load_task_program_semantic_hypotheses


def load_action_outcomes(
    path: Optional[Path] = None,
) -> Dict[str, List[ObservedEffect]]:
    """Extract a replayable ObservedEffect stream per action from the ontology."""
    path = Path(path or _DEFAULT_ONTOLOGY)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        ontology = json.load(handle).get("action_ontology", {})

    outcomes: Dict[str, List[ObservedEffect]] = {}
    for action, info in ontology.items():
        if not str(action).upper().startswith("ACTION"):
            continue
        effects: List[ObservedEffect] = []
        for ex in info.get("examples", []):
            before_level = int(ex.get("before_level", 0) or 0)
            after_level = int(ex.get("after_level", before_level) or before_level)
            effects.append(ObservedEffect(
                num_changed=int(ex.get("changed_cells", 0) or 0),
                player_moved=False,  # ar25 has no avatar movement
                game_over=str(ex.get("after_state", "")) == "GAME_OVER",
                level_complete=after_level > before_level,
            ))
        if effects:
            outcomes[str(action).upper()] = effects
    return outcomes


def run_ar25_belief_loop(
    budget: int = 200,
    revise_every: int = 20,
) -> Tuple[GameTheory, EpistemicScore, Dict[str, int]]:
    """Run the closed-loop prototype; return (theory, epistemic score, stats)."""
    outcomes = load_action_outcomes()
    actions = sorted(outcomes.keys())
    theory = GameTheory("ar25-e3c63847")
    theory.seed_actions(actions)
    action_roles, goal_families = load_task_program_semantic_hypotheses(
        _DEFAULT_TASK_PROGRAM
    )
    theory.add_semantic_hypotheses(action_roles, goal_families)
    designer = DiscriminatingExperimentDesigner()

    cursors = {a: 0 for a in actions}
    experiments = 0
    revisions = 0

    for step in range(budget):
        choice = designer.design(theory, actions)
        if choice is None:
            break  # every hypothesis resolved
        action = choice.action
        samples = outcomes.get(action)
        if not samples:
            continue
        effect = samples[cursors[action] % len(samples)]
        cursors[action] += 1
        theory.observe(action, effect, was_experiment=True)
        experiments += 1
        if revise_every and (step + 1) % revise_every == 0:
            revisions += revise_theory(theory)

    score = score_beliefs(
        theory.to_ledger(),
        build_ar25_oracle(),
        experiment_actions=experiments,
    )
    stats = {"experiments": experiments, "revisions": revisions}
    return theory, score, stats


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    theory, score, stats = run_ar25_belief_loop()
    print(f"stats: {stats}")
    print(f"theory: {theory.summary()}")
    print("epistemic score:")
    for key, value in score.to_dict().items():
        print(f"  {key}: {value}")
