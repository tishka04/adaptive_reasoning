# SAGE12 semantic trajectory planner

SAGE12 implements the higher-semantic planning path proposed after the
SAGE11 relational pilot failed. It does not revive the failed SAGE11
atom-to-effect model. Instead, it represents candidate mechanics explicitly,
rolls them out in an abstract semantic state, ranks complete trajectories, and
executes at most the first action before replanning from the next observation.

## Implemented path

```text
GameObservation
  -> grounded scene graph + observed semantic memory
  -> small local open-weight LLM proposes typed hypotheses (support=0)
  -> validator/compiler grounds roles and rejects illegal actions
  -> abstract world model rolls out bounded semantic trajectories
  -> heuristic energy, or a separately gated learned EBM, ranks trajectories
  -> hierarchical controller returns at most the first action
  -> symbolic legality, danger, and protected-competence guards retain veto
  -> observed transition updates evidence and triggers replanning
```

The implementation is split as follows:

- `scene_graph.py` creates entities without an explicit game-identity feature
  plus contact, alignment, proximity, and directional relations from each
  structured observation.
- `hypotheses.py` defines the bounded JSON/DSL contract. Free-form rationale is
  non-executable, predicates come from an allowlist, and proposals are rejected
  if they claim non-zero support.
- `llm.py` provides a strict proposal adapter, a lazy local Transformers
  backend, and a deterministic template baseline. The default local path is
  `models/qwen2_5_0.5b_instruct`; model weights are loaded once per backend.
  `device="auto"` permits CUDA placement when the local runtime supports it.
- `compiler.py` binds structural roles to current entities, checks
  preconditions, and rejects proposals whose exact action plus arguments are
  absent from the legal candidate set.
- `world_model.py` implements a cheap Beta-smoothed semantic transition model
  and bounded beam rollout. It learns only from executed before/action/after
  records; it never steps or restores the environment.
- `energy.py` supplies an auditable lower-is-better heuristic energy over goal
  distance, risk, uncertainty, likelihood, action cost, and contradiction. A
  tiny optional PyTorch pairwise EBM is implemented but cannot become active
  merely because it was trained.
- `controller.py` implements `off`, `shadow`, `bounded`, and `active` modes,
  independent proposal/world-model/energy gates, one-probe-per-context bounded
  authority, hard danger and protected-route blocks, and receding-horizon
  execution.
- `dataset.py` defines the append-only
  `sage12-semantic-trajectory-v1` record. Proposals, rankings, execution, and
  observed outcomes are stored separately so a generated claim cannot be
  mistaken for evidence.

## Live integration

`UnifiedCognitiveController` owns one SAGE12 controller. Its default mode is
`off`, so it does not generate hypotheses or alter a decision. An alternative
generator can be injected:

```python
from theory.sage12 import (
    LocalHypothesisGenerator,
    Sage12Config,
    Sage12Mode,
    SemanticPlanningController,
    TransformersJSONModel,
)

semantic = SemanticPlanningController(
    game_id=game_id,
    generator=LocalHypothesisGenerator(TransformersJSONModel()),
    config=Sage12Config(mode=Sage12Mode.SHADOW),
)
controller = UnifiedCognitiveController(
    game_id,
    available_actions=actions,
    semantic_controller=semantic,
)
```

Shadow mode ranks trajectories but returns the current symbolic action
unchanged. Bounded and active modes automatically downgrade to shadow unless
all three semantic pilot gates are explicitly marked passed. Active also
requires its separate active gate.

## Authority invariants

1. LLM output is an abductive proposal, never proof; `support` must be zero.
2. Only an observed transition updates support or refutation counts.
3. The compiler can emit only an exact legal action/argument candidate.
4. A symbolic danger-memory veto removes a trajectory before ranking.
5. Existing protected competence blocks semantic intervention.
6. One action is executed; the remaining trajectory is discarded and
   replanned after observation.
7. Learned energy is optional, defaults off, and requires both training data
   and independent promotion gates.
8. The SAGE11 holdout and historical firewalls remain closed until the SAGE12
   protocol explicitly authorizes their evaluation phase.

## Current status

The software stack and controller integration are implemented and tested.
No SAGE12 semantic corpus has been collected, no local LLM benchmark has been
run, no EBM checkpoint has been trained, and no authority gate is claimed to
have passed. Production authority therefore remains `off`.

Run the focused software validation from the repository root:

```powershell
python -m pytest -q tests\test_sage12_semantic_planning.py
```

The future evidence gates and data firewall are specified in
`reports/SAGE12_VALIDATION_PROTOCOL.md` and
`training/SAGE12_DATA_POLICY.md`. The implementation outcome is recorded in
`reports/SAGE12_IMPLEMENTATION_RESULT.md`; limitations are summarized in
`models/SAGE12_MODEL_CARD.md`.

## Frozen grounded-proposal pilot

Stage A is preregistered in
`reports/SAGE12_PROPOSAL_PILOT_PROTOCOL.md` and the checksummed
`training/sage12/proposal_pilot_v1/frozen_manifest.json`. It collects exactly
2,104 source-only executed traces under balanced randomized legal-action
coverage, times identical Qwen2.5 0.5B decoding on CPU and CUDA, and evaluates
112 outcome-blind representative scenes plus relation-shuffled controls.

The collector and evaluator are:

```powershell
python -m theory.sage12.proposal_pilot_runner benchmark
python -m theory.sage12.proposal_pilot_collection --workers 4
python -m theory.sage12.proposal_pilot_runner evaluate
```

The protocol must be committed before the first command. Any failed JSON,
grounding, recall, relation-sensitivity, per-game, or identity-leakage gate
stops the experiment before semantic world-model fitting.
