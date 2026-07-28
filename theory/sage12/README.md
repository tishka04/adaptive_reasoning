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

## Temporal mechanic-induction pilot V4

V4 tests a different unit of learning: eight observed semantic transitions
induce typed game-local rules, which predict the next transition. A
reset-local tracker emits only roles and action-anchor conditions; raw frames,
coordinates, values, track IDs, game identity, and the query outcome remain
outside the model view. Proposed rules always enter with `support=0`, while a
separate Beta evidence record stores observed support and refutation.

The structured rule inducer is primary. Qwen2.5 0.5B generates the same finite
rule vocabulary on a frozen diagnostic subset. The old V3 validation outcomes
are non-gating; V4 uses 768 fresh prospective transitions after its
source-training preflight is frozen.

```powershell
python -m theory.sage12.mechanic_induction preflight
python -m theory.sage12.mechanic_collection
python -m theory.sage12.mechanic_induction evaluate
```

The frozen design is documented in
`reports/SAGE12_MECHANIC_INDUCTION_V4_PROTOCOL.md`. V4 remains offline and
cannot fit a world model or EBM unless every structured gate passes.

V4 finished `FAIL_CLOSED`, but unlike V1-V3 it found a strong temporal signal.
The structured inducer gained +0.4676 macro Brier skill and +0.1526 macro-F1
over the local action-only baseline, with positive transfer in all three
games. Source actor tracking and prospective calibration failed their frozen
thresholds, so no world model was fit. Qwen generated nothing because the
frozen 512-token cap rejected every 879-token prompt. See
`reports/SAGE12_MECHANIC_INDUCTION_V4_RESULT.md`.

## Clean temporal replication V4.1

V4.1 preserves the V4 rule inducer and adds causal role states, source-only
leave-one-game-out Platt calibration, frozen per-label thresholds, a compact
Qwen rule compiler, separate structured/Qwen authority, and a per-effect V5
capability ledger.

```powershell
python -m theory.sage12.mechanic_replication preflight
python -m theory.sage12.mechanic_replication_collection
python -m theory.sage12.mechanic_replication evaluate
```

The source preflight must be published before the fresh 768-transition
collection. The frozen protocol is
`reports/SAGE12_MECHANIC_REPLICATION_V4_1_PROTOCOL.md`.

V4.1 stopped at `FAIL_SOURCE_TRAIN_PREFLIGHT`. Causal role resolution,
source-only calibration, and the compact Qwen token budget passed, but
`actor_displaced` source capacity and the static game-identity probe failed.
No prospective trace or Qwen output was generated. See
`reports/SAGE12_MECHANIC_REPLICATION_V4_1_RESULT.md`.

## Invariant target-mechanic replication V4.2

V4.2 is a separately versioned target-only test. It maps anchors to
`occupied`, `free`, or `none`, models only target creation/removal/movement,
and keeps actor displacement audit-only. Source calibration, structured
mechanics, deterministic controls, Qwen diagnostics, and all prospective
authority remain fail-closed.

```powershell
python -m theory.sage12.target_mechanic_replication preflight
python -m theory.sage12.target_mechanic_replication_collection
python -m theory.sage12.target_mechanic_replication evaluate
```

The frozen design is
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_PROTOCOL.md`.

The source preflight passed all 11 gates without opening validation outcomes.
See `reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_PREFLIGHT.md`.

The subsequent frozen collection completed at 768 transitions and was
published before evaluation. See
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_COLLECTION.md`.

Evaluation finished `FAIL_RUNTIME_CLOSED`: the frozen public serializer did
not handle the structured engine's generic `any` anchor. No metric verdict or
downstream authority exists. See
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_RESULT.md`.

## Runtime-safe target replication V4.2.1

V4.2.1 is a clean recovery run. It adds `any` only to the public structured
rule vocabulary; state/query anchors and the Qwen contract remain
`occupied`, `free`, or `none`. A source rehearsal must round-trip every rule
and serialize all 1,911 source predictions before the ordinary preflight can
authorize a new collection.

```powershell
python -m theory.sage12.target_mechanic_recovery rehearsal
python -m theory.sage12.target_mechanic_recovery preflight
python -m theory.sage12.target_mechanic_recovery_collection
python -m theory.sage12.target_mechanic_recovery evaluate
```

Any authorized collection uses 768 fresh rows under seeds 661, 709, 757, and
809; V4.2 shards cannot be reused. The evaluator commits its structured
predictions and verdict before Qwen and writes an automatic runtime-failure
artifact on an uncaught error. The frozen design is
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_PROTOCOL.md`.

The source rehearsal passed all seven checks and serialized every one of the
1,911 source predictions, including generic `any` evidence. See
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_REHEARSAL.md`. Only the source
preflight is authorized at this checkpoint.

The unchanged source preflight subsequently passed all 14 conjunctive gates
without opening validation outcomes. See
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_PREFLIGHT.md`. The frozen
768-transition collection, and nothing downstream, is now authorized.

The fresh collection subsequently completed at 768 rows with balanced legal
actions, 24 total resets, and 79 retained chronological repeats. See
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_COLLECTION.md`. Its raw shards
must be published before the single frozen evaluation is run.

The evaluation completed `FAIL_CLOSED`. Structured mechanics passed 18/19
gates and strongly beat action-only, but binding-shuffle loss was +0.017061
against the +0.020000 minimum. Qwen separately failed all six gates after
emitting only Markdown-fenced responses. See
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_RESULT.md`. No V5, world
model, EBM, or controller authority exists.

## Causal binding and conditional world model V4.3

V4.3 replaces the weak binding permutation with executed counterfactual
pairs. From one deterministic replay-verified pre-state, it runs two legal
action bindings, retains both outcomes, and recursively constructs a binary
tree to depth three. Its independent
`sage12-bound-trajectory-v4.3` audit record keeps raw frames, arguments,
coordinates, IDs, hashes, seeds, and paths outside the model view.

The source-only projection ladder freezes `minimal`, `relational`, or `typed`
binding semantics before validation collection. The primary
`BoundMechanicRule` combines action, binding, and an eight-event history while
keeping rule `support=0` and observed evidence separate. It is compared with
action-plus-history, action-only, binding-only, and deterministic-template
baselines. Binding swaps exchange the two actually executed target
descriptions while preserving action and label.

Only a complete binding-gate pass permits the structured
`BoundSemanticWorldModel`: an identity-free occupancy state, factorized target
effects, applicability constraints, horizon three, and beam width eight. Qwen,
GNNs, EBM training, and controller execution are excluded.

```powershell
python -m theory.sage12.bound_mechanic_pilot collect-source
python -m theory.sage12.bound_mechanic_pilot preflight
python -m theory.sage12.bound_mechanic_pilot collect-validation
python -m theory.sage12.bound_mechanic_pilot evaluate-binding
python -m theory.sage12.bound_mechanic_pilot evaluate-world-model
```

The frozen design and publication sequence are in
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_PROTOCOL.md`. At this checkpoint,
only the implementation, tests, protocol, and manifest exist; no V4.3 outcome
has been observed and all downstream authority remains closed.

The source collector subsequently completed all 352 roots, producing 2,396
pairs and 4,792 executed arms with zero replay failures. Creation and removal
meet their frozen class-capacity minima, but target movement has only 8 source
positives versus the required 75. The raw corpus is published before the
official source preflight and no validation game is open. See
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_COLLECTION.md`.

The source preflight subsequently returned `FAIL_CLOSED`. Target movement had
8 positives versus the required 75. All three projections also had negative
LOGO Brier skill and macro-F1 gain, while game-identity gain ranged from
+0.2089 to +0.5624 against the +0.05 maximum. No projection was frozen, no
validation game was opened, and both binding and world-model stages wrote
explicit skipped artifacts. See
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_RESULT.md`.

## Paired causal contrast V4.4

V4.4 reuses only the published V4.3 source pairs and changes the learning
question from absolute arm prediction to direct causal comparison: given two
interventions executed from an identical pre-state, which arm produces the
effect?

Each model view is `left_features - right_features`. Logistic models have no
intercept and train on both `(x, y)` and `(-x, 1-y)`, so a complete arm swap
exactly inverts the prediction. The primary model combines action, a frozen
binding projection, and arm-conditioned evidence from the shared eight-event
context. Baselines use history without binding, action only, binding only, or
the deterministic occupied/free template.

The source-only audit admits creation and removal, with 172 and 189 discordant
pairs. Movement has zero discordant pairs and remains diagnostic-only. A
source LOGO preflight must pass utility, binding-swap, identity, calibration,
per-game, bootstrap, and exact-antisymmetry gates before any fresh validation
tree can be collected.

```powershell
python -m theory.sage12.pairwise_causal_pilot preflight
python -m theory.sage12.pairwise_causal_pilot collect-validation
python -m theory.sage12.pairwise_causal_pilot evaluate
```

The frozen protocol is
`reports/SAGE12_PAIRWISE_CAUSAL_PILOT_V4_4_PROTOCOL.md`. At this checkpoint no
V4.4 predictive result exists, no validation game is open, and no world model
or downstream authority is granted.

V4.4 subsequently returned `FAIL_CLOSED` at source preflight. The relational
projection was least negative but reached only +0.0143 directional-accuracy
gain, -0.0337 Brier skill, 0.1246 ECE, and +0.0906 identity gain. Swapping
bindings improved rather than degraded accuracy for every projection. Exact
complete-arm inversion passed, confirming that the negative result comes from
the binding signal rather than the antisymmetric implementation. No
validation shard was created. See
`reports/SAGE12_PAIRWISE_CAUSAL_PILOT_V4_4_RESULT.md`.

## Rooted intervention events V4.5

V4.5 replaces the rejected binding buckets with a tri-view compiler over the
common pre-state and both executed post-states. It matches objects with
translation-normalized shape, overlap, relative size, distance, and tolerant
value evidence; handles splits and merges; cancels common dynamics; and emits
direct, local, or collateral intervention-exclusive events.

The event vocabulary is discovered from V4.3 source data with a deterministic
fine-to-coarse capacity rule, then frozen. `RootedTargetGraph` represents only
the action, its occupied/virtual/actor root, two-hop relative relations, roles,
and identity-free temporal buckets. It excludes coordinates, IDs, raw values,
shape signatures, global signatures, and game identity.

The feasibility audit is design-only. It must pass compiler quality, source
LOGO utility, root-swap and relation-shuffle sensitivity, identity,
calibration, per-game, bootstrap, and exact-antisymmetry gates before fresh
source collection is possible.

```powershell
python -m theory.sage12.object_causal_pilot feasibility
python -m theory.sage12.object_causal_pilot collect-source
python -m theory.sage12.object_causal_pilot preflight
python -m theory.sage12.object_causal_pilot collect-validation
python -m theory.sage12.object_causal_pilot evaluate
```

The frozen design is in
`reports/SAGE12_OBJECT_CAUSAL_PILOT_V4_5_PROTOCOL.md`; its manifest checksum is
`cfae89ac0de9f263af52dbb042e352869324f301a633012d44ad7b85ec028741`.
The protocol checkpoint preceded the source-only audit.

V4.5 subsequently returned `FAIL_CLOSED`. Correspondence confidence was high
at 0.9840, but root grounding was only 0.8130 and exclusive-event localization
0.8370. All 11 promoted events were local or collateral; none was a direct
target event. The structured model had -0.0757 Brier skill, +0.0039 accuracy
gain, +0.2287 identity gain, and 0.1087 ECE. Root swapping and relation
shuffling improved accuracy, while exact complete-arm inversion passed.
Fresh source and validation collection remained closed. See
`reports/SAGE12_OBJECT_CAUSAL_PILOT_V4_5_RESULT.md`.

## Candidate-complete semantic slots (V4.7)

V4.7 replaces free-form proposal coverage with one `SemanticActionSlot` per
legal action. `SlotAnnotation` carries seven frozen-model probabilities with
`support=0`; `HypothesisCompiler.compile_slots` cannot discard an all-zero
slot. Exact execution arguments remain separate from the identity-free
semantic signature.

The offline pilot in `integration_pilot_v4_7.py` scores 14 constrained Qwen
bits for both slots at all 2,380 complete-tree nodes, fits nested
leave-one-game-out effect and utility models, ranks all eight depth-three
leaves with an eight-feature EBM, and returns only the selected trajectory's
first action. The V4.3 future-node topology is an explicitly non-deployable
oracle; root-slot reuse is the deployable control.

```powershell
python -m theory.sage12.integration_pilot_v4_7 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.integration_pilot_v4_7 generate-qwen --device cuda:0
python -m theory.sage12.integration_pilot_v4_7 evaluate
```

The frozen contract is documented in
`reports/SAGE12_CANDIDATE_COMPLETE_SLOT_PILOT_V4_7_PROTOCOL.md`. No V4.7
outcome can grant live authority or open held-out data.

V4.7 returned `CURRENT_STACK_NEGATIVE_QWEN_SEMANTICS_BOTTLENECK`. Candidate
coverage and syntax reached 1.00, but the full stack lost 0.4373 utility to the
fold-selected baseline and Qwen did not improve structured-only features.
True effect annotations instead gained 0.3978, and true world outputs let the
learned EBM match the oracle exactly. See
`reports/SAGE12_CANDIDATE_COMPLETE_SLOT_PILOT_V4_7_RESULT.md`.

## Paired semantic adaptation (V4.8)

V4.8 targets the localized semantic bottleneck without changing the V4.7
world model, EBM, controller, candidate trees, or baseline selection. It
combines deterministic same-prestate pairs from the published SAGE11
source-train corpus with all replay-verified V4.3 pairs. Raw coordinates and
identity fields are excluded.

The local Qwen2.5 0.5B model remains frozen and supplies paired prompt
embeddings. A rank-16 external low-rank residual adapter predicts
left/right/both/neither for the seven slot effects plus auxiliary progress.
Representation selection and every published slot annotation are
leave-one-game-out. The experiment proceeds through the full architecture even
if direct semantic diagnostics are weak; its thresholds are exploratory and
cannot promote live authority.

```powershell
python -m theory.sage12.semantic_adapter_v4_8 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_adapter_v4_8 embed --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_adapter_v4_8 adapt --device cuda:0
python -m theory.sage12.semantic_adapter_v4_8 evaluate
```

The frozen design is
`reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_PROTOCOL.md`.

The semantic checkpoint selected the invariant-context view but returned
`DIRECT_SEMANTIC_ADAPTATION_NEGATIVE`: seven-effect LOGO Brier was 0.093676
versus 0.067048 for action-only, semantic outputs still predicted game
identity at 91.43%, and explicit completion recall was zero. The checkpoint is
published before world-model fitting in
`reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_SEMANTIC_RESULT.md`. Per the
frozen exploratory protocol, this negative intermediate result does not stop
the full architecture evaluation.

The full V4.8 evaluation subsequently returned
`EXPLORATORY_ARCHITECTURE_NOT_SUPPORTED`. Adapted semantics improved the
unchanged structured stack by +0.2068 utility with an entirely positive
bootstrap interval and improved V4.7 zero-shot utility by +0.1938. They still
lost 0.2435 to the strong action-sequence baseline, were non-negative on only
5/11 games, selected 0/3 completion trajectories, and exposed game identity
at 85.61%. True annotations again gained +0.3978 over the baseline. The result
rejects this frozen-Qwen external-adapter route, not the conditional global
architecture. See
`reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_RESULT.md`.

## Object-relative semantic teacher/student (V4.9)

V4.9 replaces language-model classification with a separated teacher/student
contract. A deterministic post-transition teacher compiles seven physical and
ten functional effects with applicability masks. Its student sees only the
pre-action legal action plus an unordered, identity-free graph rooted at the
action target. Absolute coordinates, colours/raw values, object IDs, game IDs,
frames, and future fields are excluded.

The student is a small DeepSets encoder trained with masked imitation,
same-prestate productive ranking, and an adversarial game-identity objective.
Every reported prediction is leave-one-source-game-out. Action-only and
root-only baselines, relation shuffling, neighbor-order invariance, output
identity, per-game transfer, and completion recall@8 are frozen controls.

```powershell
python -m theory.sage12.semantic_teacher_v4_9 freeze
python -m theory.sage12.semantic_teacher_v4_9 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.object_relative_student_v4_9 evaluate --device cuda:0
```

The design and decision thresholds are frozen in
`reports/SAGE12_OBJECT_RELATIVE_TEACHER_STUDENT_V4_9_PROTOCOL.md`. At this
checkpoint no V4.9 result has been observed and no live authority exists.

The teacher corpus subsequently passed QA with 7,541 unique source-only
transitions, 2,396 same-prestate pairs, 100% valid semantic roots, and all
identity/firewall checks passing. It contains exactly five non-synthetic
completion events. The corpus is published before student training; see
`reports/SAGE12_OBJECT_RELATIVE_TEACHER_V4_9_QA.md`.

The LOGO student subsequently returned
`OBJECT_RELATIVE_TEACHER_IMITATION_NOT_YET_SUPPORTED`. It slightly beat the
root-only model and improved productive-pair ranking over action-only, but lost
0.0031 macro-Brier to action-only, improved when relations were shuffled,
exposed game identity at 90.64%, and retrieved 0/5 completions at rank eight.
The valid teacher remains useful, but these probabilities must not train the
semantic world model or EBM. See
`reports/SAGE12_OBJECT_RELATIVE_TEACHER_STUDENT_V4_9_RESULT.md`.

## Action-aligned invariant semantics (V4.10)

V4.10 is the narrow follow-up to the V4.9 diagnosis. It adds a fresh
source-only offline collector targeting underrepresented functional effects,
replaces compass relations with intervention-relative axes and contact
topology, balances every training step across games, aligns latent/output
distributions, and applies a game-balanced prevalence calibration rule.

```powershell
python -m theory.sage12.action_aligned_semantics_v4_10 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.functional_intervention_collection_v4_10
python -m theory.sage12.action_aligned_semantics_v4_10 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.action_aligned_semantics_v4_10 evaluate --device cuda:0
```

The source-only quotas, leakage boundary and decisions are frozen in
`reports/SAGE12_ACTION_ALIGNED_INVARIANT_SEMANTICS_V4_10_PROTOCOL.md`. At this
checkpoint no V4.10 collection or model result has been observed.

The collector subsequently retained 1,587 fresh unique rows. Ten games met
their original quotas; `su15` saturated at 83 rows after 40 resets and 1,197
exact-repeat rejections. A pre-model capacity amendment lowers only its minimum
to 80, leaving every model and evaluation decision unchanged. See
`reports/SAGE12_ACTION_ALIGNED_SEMANTICS_V4_10_COLLECTION.md`.

The augmented teacher corpus subsequently passed QA with 9,128 records and
100% action-aligned, compass-free model graphs. It contains six genuine
completion events and preserves all 2,396 same-prestate pairs. See
`reports/SAGE12_ACTION_ALIGNED_TEACHER_V4_10_QA.md`.

The strict LOGO evaluation subsequently returned
`ACTION_ALIGNED_INVARIANT_SEMANTICS_NOT_YET_SUPPORTED`. V4.10 improved
macro-Brier by 0.01029 over V4.9 and narrowly beat action-only by 0.00042.
Productive-pair accuracy rose to 0.7681 versus 0.3713 for action-only.
However, root-only remained better for both macro-Brier (0.07299 versus
0.07819) and pair ranking (0.7738 versus 0.7681), relation shuffling improved
macro-Brier, only 4/11 games beat action-only, semantic outputs exposed game
identity at 85.51%, and completion recall@8 was 0/6. The probabilities remain
audit-only and do not authorize semantic world-model or EBM fitting. See
`reports/SAGE12_ACTION_ALIGNED_INVARIANT_SEMANTICS_V4_10_RESULT.md`.

## Counterfactual semantic panels (V4.11)

V4.11 makes the strongest V4.10 signal—same-state productive ordering—the
primary learning problem. It executes two to four legal actions from one
replay-verified source state, compiles immediate semantic deltas and
deterministic horizon-three progress, and trains an exactly antisymmetric
Siamese comparator.

The absolute export is anchored to the stronger root-only model. Only a
panel-centered relational residual may modify that anchor, with its weight
selected inside the outer training games. Comparative success and absolute
distillation have separate frozen decisions; neither world-model nor EBM
fitting is permitted in V4.11.

```powershell
python -m theory.sage12.counterfactual_semantic_panels_v4_11 freeze
python -m theory.sage12.counterfactual_panel_collection_v4_11
python -m theory.sage12.counterfactual_semantic_panels_v4_11 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.counterfactual_semantic_panels_v4_11 evaluate --device cuda:0
```

The source-only collection, capacity rules, LOGO controls, confidence
intervals, and decisions were frozen before collection in
`reports/SAGE12_COUNTERFACTUAL_SEMANTIC_PANELS_V4_11_PROTOCOL.md`.

The collector subsequently retained the complete quota: 1,056 panels, 3,914
unique immediate arms, and 15,124 deterministic continuation transitions.
Every game reached 96 panels with zero replay failure and no exact
pre-state/action repeat against V4.3–V4.10. Source validation, holdout,
historical, and live data remained closed. See
`reports/SAGE12_COUNTERFACTUAL_SEMANTIC_PANELS_V4_11_COLLECTION.md`.

The deterministic teacher then returned
`COMPARATIVE_CAUSAL_TEACHER_CAPACITY_FAILED` before model fitting. It compiled
5,529 comparisons and eight immediate effects met their eligibility rule, but
only 7/11 games supplied the frozen minimum of 20 progress-discordant panels;
eight completion-positive arms appeared in only two games. The Siamese and
distillation implementations therefore remain tested but unfitted, and no GPU,
world model, EBM, or controller authority followed. See
`reports/SAGE12_COUNTERFACTUAL_TEACHER_V4_11_QA.md` and
`reports/SAGE12_COUNTERFACTUAL_SEMANTIC_PANELS_V4_11_RESULT.md`.

## Descriptive semantics and conditional integration (V4.12)

V4.12 reuses all 1,056 V4.11 panels without collecting new data and removes
the failed scalar progress target. It trains the same identity-free,
object-relative comparator on the eight immediate effects that already met
V4.11's capacity rule, preserving partial credit per effect. A panel-centred
relation residual is distilled onto a root-only probability anchor under
strict leave-one-game-out evaluation.

The protocol freezes a two-stage decision. The semantic model must beat both
root-only and action-only controls, react correctly to relation shuffling,
improve absolute calibration across games and avoid incremental game-signature
leakage. Only a complete semantic pass opens the unchanged source-only V4.7
chain: world model, depth-three trajectories, EBM and controller. That second
stage compares action-only, action-sequence, structured, root-semantic,
fixed-heuristic, relation-shuffle and oracle ladders, including utility and
completion capture.

```powershell
python -m theory.sage12.descriptive_semantic_integration_v4_12 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.descriptive_semantic_integration_v4_12 run --device cuda:0
```

The complete frozen design, thresholds and interpretation boundary are in
`reports/SAGE12_DESCRIPTIVE_SEMANTIC_INTEGRATION_V4_12_PROTOCOL.md`. At this
checkpoint no V4.12 model result has been observed and no authority is
promoted.

The completed GPU evaluation returned
`DESCRIPTIVE_SEMANTICS_NOT_SUPPORTED`. The comparator beat action-only
effect-pair loss with a positive confidence interval and improved ECE and
identity leakage relative to root-only. However, its pair-loss gain over
root-only was indistinguishable from zero, its active macro-Brier was worse
than both root-only and action-only, only 4/11 games were nonnegative, and
relation shuffling improved both pair loss and Brier. The frozen conditional
gate therefore emitted `SKIPPED_SEMANTIC_GATE_FAILED` without fitting a world
model or EBM. See
`reports/SAGE12_DESCRIPTIVE_SEMANTIC_INTEGRATION_V4_12_RESULT.md`.
