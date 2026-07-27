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
Stage A collected 2,104 source-only executed traces and evaluated 224 local
Qwen2.5 0.5B outputs. The pilot failed all seven frozen gates: no output passed
the strict typed parser, recall@8 was zero against an action-only baseline of
0.895, relation-shuffle degradation was zero, every validation-game gain was
negative, and full compact scene signatures identified source-training games
at 99.94% accuracy. No semantic world model or EBM was fit. Production
authority remains `off`.

Run the focused software validation from the repository root:

```powershell
python -m pytest -q tests\test_sage12_semantic_planning.py
```

The future evidence gates and data firewall are specified in
`reports/SAGE12_VALIDATION_PROTOCOL.md` and
`training/SAGE12_DATA_POLICY.md`. The implementation outcome is recorded in
`reports/SAGE12_IMPLEMENTATION_RESULT.md`; limitations are summarized in
`models/SAGE12_MODEL_CARD.md`. The Stage A result is published in
`reports/SAGE12_PROPOSAL_PILOT_RESULT.md`.

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

The first hardware preflight found that the unbounded scene serializer could
produce a 1.68-million-token prompt. Before any completed generation or
outcome, the protocol was amended to a deterministic 24-entity/96-relation
proposal view with an 8,192-token hard cap. All task, data, decoding, baseline,
and pass-gate choices remain unchanged.

The completed benchmark selected `cuda:0`: median inference fell from 26.478
seconds on CPU to 6.953 seconds on the laptop RTX 4050, a 3.808x speedup with
unchanged decoding. The clean evaluation then produced `FAIL_CLOSED`, checksum
`fbb86c17fee57ff46199dd94594936694bf2b0e63b05ece2c9e323813422d35a`.
Run the non-gating explanatory diagnostics with:

```powershell
python -m theory.sage12.proposal_pilot_runner diagnose
```

## Constrained effect pilot V2

V2 replaces free-form generation with a frozen Qwen encoder, independent
class-balanced linear heads, and deterministic typed-JSON rendering. Its
state input is one binary `actor_interaction` bit plus the selected action;
entity inventories, identities, counts, shapes, directions, available-action
sets, and scene signatures are excluded.

The first six-bit source-training design still leaked +32.08 game-identity
points beyond selected action and was rejected without reading V2 validation.
The retained one-bit view leaks +8.99 points beyond selected action and is
frozen in `reports/SAGE12_CONSTRAINED_PILOT_V2_PROTOCOL.md`. It reuses the
exact V1 source corpus.

V2 finished `FAIL_CLOSED`, checksum
`7440cbf5a15edd4ca2c7c70fbebdcb2ced1bdf88817bdf1f7c0f417a6db81e3a`.
JSON, support-zero, grounding, and both leakage controls passed. Predictive
transfer did not: Qwen macro-F1 was 0.484 versus 0.549 for action-only, the
relation shuffle improved macro-F1 to 0.582, and `re86` lost 0.237. The full
result is in `reports/SAGE12_CONSTRAINED_PILOT_V2_RESULT.md`. The semantic
world-model gate remains closed.

```powershell
python -m theory.sage12.constrained_pilot preflight
python -m theory.sage12.constrained_pilot evaluate
python -m theory.sage12.constrained_pilot diagnose
```

## Action-target effect pilot V3

V3 uses 4,000 fresh executed source-only transitions and grounds each action
to its exact movement destination, clicked object, clicked empty cell, or
targetless anchor. It replaces the joint-tuple target with four independently
scored observed effects and masks ambiguous before/after matches. Absolute
coordinates, values, IDs, game signatures, raw grids, and outcomes remain
outside model inputs.

The source-training leakage ladder selected the `coarse` projection at +0.0987
game-identity accuracy beyond action-only. Leave-one-game-out training selected
shallow gradient boosting. On the frozen validation games, that model reached
0.232 macro-F1 versus 0.237 for action-only and 0.371 for the deterministic
template. The -0.140 primary gain, 0.0005 target-shuffle degradation, negative
transfer on two games, and 0.397 macro ECE produced `FAIL_CLOSED`.

The fixed typed output contract nevertheless passed JSON, support-zero, and
grounding at 1.00. A diagnostic showed that the selected projection contained
only 26 unique training signatures and that the target shuffle changed only
1.25% of validation rows. V3 therefore rejects this coarse global one-step
effect representation, not the complete high-semantic planning architecture.
No world model or EBM was fit.

```powershell
python -m theory.sage12.action_target_collection source_train
python -m theory.sage12.action_target_pilot preflight
python -m theory.sage12.action_target_collection source_validation
python -m theory.sage12.action_target_pilot evaluate
python -m theory.sage12.action_target_pilot diagnose
```

The protocol is in
`reports/SAGE12_ACTION_TARGET_PILOT_V3_PROTOCOL.md`; the complete gate ledger,
checksums, and interpretation are in
`reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`.
