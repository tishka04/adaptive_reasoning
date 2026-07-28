# SAGE12 implementation result

Date: 2026-07-27.

Outcome: the guarded high-semantic planning scaffold is implemented and
integrated; empirical promotion remains closed.

## Delivered

- grounded game-identity-free scene graphs and semantic memory;
- a typed hypothesis/effect DSL with strict JSON parsing and mandatory
  `support=0`;
- a cached local Transformers backend for a small open-weight model plus a
  deterministic template baseline;
- role grounding, precondition checks, exact legal-action compilation, and
  explicit rejection reasons;
- a Beta-smoothed semantic transition model and bounded beam trajectories;
- a six-component heuristic energy and optional tiny pairwise PyTorch EBM;
- hierarchical subgoals and one-action receding-horizon control;
- `off`, `shadow`, `bounded`, and `active` modes with independent gates;
- symbolic danger veto, protected-competence supremacy, bounded context probes,
  and outcome-only evidence updates;
- versioned semantic-trajectory audit records and append-only JSONL support;
- integration with transition observation, branch reset, decision arbitration,
  and controller summary in `UnifiedCognitiveController`.

## Software evidence

Focused validation:

```text
python -m pytest -q tests\test_sage12_semantic_planning.py
14 passed
```

The tests cover scene grounding, support-zero enforcement, illegal/unbound
proposal rejection, local-model JSON parsing, shadow identity, authority gate
downgrade, active receding-horizon selection, danger/protected blocks,
observed-only model updates, integrated-controller activation, deterministic
template behavior, and pairwise EBM optimization mechanics.

Full repository regression:

```text
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q
1681 passed, 1 environment warning in 190.19s
```

The warning is Joblib falling back from an unavailable physical-core query to
the logical core count; it does not affect a test result. Focused Ruff
validation of `theory/sage12`, its tests, and the unified-controller
integration also passes.

Post-pilot regression:

```text
python -m pytest -q
1692 passed, 1 environment warning in 212.93s
```

The focused semantic-planning and proposal-pilot suites pass 25 tests
together, and targeted Ruff validation remains clean.

Post-V2 regression:

```text
python -m pytest -q
1698 passed, 1 environment warning in 192.04s
```

The three focused SAGE12 suites pass 31 tests together. The warning remains
the same harmless Joblib physical-core query fallback.

## GPU decision

The subsequent Stage A proposal pilot compared identical Qwen2.5 0.5B
decoding on CPU and the laptop RTX 4050. Median inference fell from 26.478
seconds to 6.953 seconds, a 3.808x speedup, so the GPU was used for the 224
clean proposal generations. This was inference only. No semantic world model
or EBM was trained because the proposal gate failed.

## Authority result

The proposal gate has now been evaluated and failed closed. Strict typed JSON,
grounding, recall gain, relation sensitivity, per-game transfer, and
game-signature leakage all failed their frozen requirements. The world-model,
energy, shadow, bounded, active, and holdout stages were therefore not run.
The integrated default remains `off`; bounded and active downgrade to shadow
without all prerequisite gates. See
`reports/SAGE12_PROPOSAL_PILOT_RESULT.md` for the complete result and artifact
checksums.

The separately preregistered constrained V2 repair passed JSON, support-zero,
grounding, and both reduced-leakage gates. It still failed predictive gates:
Qwen primary macro-F1 0.484 versus action-only 0.549, shuffle degradation
-0.098, and `re86` gain -0.237. V2 result checksum:
`7440cbf5a15edd4ca2c7c70fbebdcb2ced1bdf88817bdf1f7c0f417a6db81e3a`.
No world-model fitting followed.

## Action-target V3 implementation and result

V3 adds a separate `sage12-action-target-trace-v3` schema, deterministic
action-anchor resolution, conservative one-to-one before/after object
matching, four masked component labels, balanced/adaptive source-only
collection, source-training leakage selection, independent structured heads,
frozen Qwen embedding ablation, deterministic and action-only baselines,
shuffle/permutation controls, calibration, bootstrap intervals, and complete
artifact checksums.

Collection produced exactly 3,040 source-training and 960 source-validation
rows with no exact duplicates. The source-only preflight selected the coarse
projection and shallow gradient boosting. The once-only validation evaluation
failed closed: structured macro-F1 0.232 versus 0.371 for the stronger
template, primary gain -0.140, target-shuffle degradation 0.0005, and macro
ECE 0.397. JSON, support-zero, and grounding were all 1.00. Result checksum:
`10b1d84b6ff675c3fd05f73ad853d0618658b79045824ad4c2f9e79e6466fdb4`.

The explanatory diagnostic found only 26 unique training signatures in the
selected model view and a target shuffle that changed 12 of 960 validation
rows. This leaves the software and audit corpus available for research but
keeps all later authority closed. See
`reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`.

Post-V3 focused validation:

```text
python -m pytest -q tests/test_sage12_action_target_pilot.py
tests/test_sage12_semantic_planning.py tests/test_sage12_proposal_pilot.py
tests/test_sage12_constrained_pilot.py
48 passed
```

Post-V3 full repository regression:

```text
python -m pytest -q
1715 passed, 1 environment warning in 242.33s
```

Targeted Ruff validation of the V3 schema, collector, pilot, and tests also
passes. The warning remains the same harmless Joblib physical-core query
fallback.

## Temporal mechanic-induction V4

V4 implements reset-local role tracking, bounded semantic transition events,
eight-transition windows, typed zero-support mechanic rules, separate Beta
evidence, source-only priors, outcome-blind prospective queries, balanced
chronological collection, global/local/template baselines, context, binding,
outcome, and label controls, block bootstrap, calibration, Qwen diagnostics,
and full checksummed artifacts.

Post-V4 validation:

```text
targeted SAGE12 tests: 57 passed in 12.50s
targeted Ruff checks: All checks passed!
full repository suite: 1724 passed, 1 warning in 271.84s (0:04:31)
```

The sole warning remains the harmless Joblib physical-core query fallback.

It derived 1,911 source windows and collected 768 fresh transitions yielding
576 prospective windows. Structured prediction strongly beat the local
action-only baseline, but source actor-role quality and calibration failed the
frozen conjunctive protocol. Result checksum:
`5987eb9531f568dc814dad46eb9e78d13a3813a9c30db3d6cb1fa8a319e16927`.
No world model, EBM, or controller evaluation followed.

## Clean temporal replication V4.1

V4.1 is implemented as a separate version so V4 remains reproducible. It adds
an online `translational` / `non_translational` / `ambiguous` role contract,
source leave-one-game-out Platt calibration, source-only decision thresholds,
raw-versus-calibrated replication metrics, a compact Qwen JSON compiler,
separate Qwen authority, and a per-effect V5 eligibility ledger.

The implementation, tests, protocol, and manifest are frozen before the
source preflight. No world model, EBM, or controller is fit at this checkpoint.

The published source preflight then derived all 1,911 source windows and
finished `FAIL_SOURCE_TRAIN_PREFLIGHT`. Role resolution reached 0.9984;
calibration improved macro Brier from 0.0483 to 0.0430 and macro ECE to
0.0360; all Qwen prompts fit at 322–345 tokens. The run stopped because
`actor_displaced` had 35 positives against a minimum 75 and the static
identity gain was 0.1293 against a maximum 0.10. The prospective collector,
Qwen generator, semantic world model, EBM, and controller were not run.
Result checksum:
`cffa41e2ae980f64dfc76cbe40076809b301da4e8f98dffbc02122eb2bfa147c`.
See `reports/SAGE12_MECHANIC_REPLICATION_V4_1_RESULT.md`.

Post-V4.1 validation:

```text
targeted SAGE12 tests: 71 passed in 9.93s
targeted Ruff checks: All checks passed!
full repository suite: 1738 passed, 1 warning in 320.20s (0:05:20)
```

The full suite uses the repository's bundled Python 3.12 environment because
its ARC dependencies contain Python 3.12 native extensions. Running it from
the host Python 3.11 process incorrectly mixes those extensions and fails
during collection. The sole successful-run warning remains Joblib's harmless
physical-core query fallback.

## Invariant target-mechanic replication V4.2

V4.2 is implemented as a separate public contract over the unchanged V4.1
engine. Its compatibility adapter makes the actor effect inapplicable before
rule induction, while public windows, prompts, calibration, metrics, and
authority contain only target creation/removal/movement. Anchors are reduced
to `occupied`, `free`, and `none`.

The implementation includes source-only calibration and utility gates,
identity probes, outcome and anchor-binding shuffles, per-effect authority,
a separate three-effect Qwen compiler, fresh-seed prospective collection,
checksum firewalls, and fail-closed CLIs. The protocol and manifest are
frozen before the V4.2 source preflight; no prospective outcome is open at
this checkpoint.

Pre-preflight validation:

```text
targeted SAGE12 tests: 85 passed in 12.87s
targeted Ruff checks: All checks passed!
full repository suite: 1752 passed, 1 warning in 196.39s (0:03:16)
```

The warning remains Joblib's harmless physical-core query fallback.

The subsequent V4.2 source preflight passed all 11 frozen gates on 1,911
windows. Identity gain was +0.0387, calibrated Brier skill +0.1821,
macro-F1 gain +0.0749, context skill +0.4328, macro-ECE 0.0365, and the
complete Qwen prompt range 295–317 tokens. Preflight checksum:
`68747717f45289775cd543aaa027eb24164200b255b42b57368e4c6fba0816ff`.
Only the frozen prospective collection is authorized by this checkpoint.

The collector then produced exactly 768 fresh rows, 256 per validation game,
with balanced legal-action coverage, eight resets per game, no outcome
adaptation, and 91 chronological repeats retained. Collection report checksum:
`6bdec774c744061e3e5014ced8d3d0191d1cdc13243130817ea9ec84fd50dce7`.
The raw shards were frozen before any prospective metric or Qwen generation.

The frozen evaluator subsequently wrote 576 validation windows and both
128-row Qwen streams, then raised `KeyError('any')` while serializing a
generic structured rule. It wrote neither predictions nor a pilot result.
Because outcomes were already opened, V4.2 was not patched or rerun and
finished `FAIL_RUNTIME_CLOSED`. Failure checksum:
`17934d7b576ac11c36abcac6235e7bc259247f225f49edf5e05126971390be6a`.
No structured, Qwen, V5, world-model, EBM, or controller authority followed.

## Runtime-safe target replication V4.2.1 freeze

V4.2.1 adds a separate recovery implementation without changing V4.2. Its
public rule serializer and inverse loader now represent the internal generic
`any` anchor, while observed states and Qwen remain restricted to the same
three concrete anchors. A mandatory source rehearsal enumerates and
round-trips all rules, executes the full prediction writer on 1,911 source
windows, and gates the existing source preflight.

The prospective evaluator now persists all structured predictions and a
checksummed intermediate structured verdict before invoking Qwen. Its
top-level fail-closed wrapper always records the failed stage, exception, and
available artifact hashes. The fresh collector requires both source gates,
uses seeds 661, 709, 757, and 809, and rejects reuse of V4.2 artifacts through
its versioned manifest and destination.

At this checkpoint only the implementation, tests, protocol, and manifest are
frozen. No V4.2.1 prospective outcome has been opened, and no V5, world-model,
EBM, or controller authority has been granted.

Pre-rehearsal validation:

```text
V4.2/V4.2.1 targeted tests: 26 passed in 29.04s
targeted Ruff checks: All checks passed!
full repository suite: 1764 passed, 1 warning in 284.35s (0:04:44)
```

The warning is the unchanged harmless Joblib physical-core query fallback.

The subsequent frozen source rehearsal passed all seven checks. It wrote
1,911/1,911 source prediction rows, round-tripped 168/168 rules, and
serialized 2,120 selected generic-`any` evidence entries. Rehearsal checksum:
`cd2164ecdfab094d99364cfdec213767987e974e9fd5b4dc01f98db423873b92`.
This authorizes only the source preflight.

The source preflight then passed all 14 conjunctive gates. Structured
source-only macro Brier was 0.047919 versus 0.058585 for local action,
macro-F1 gained +0.074908, context skill was +0.432771, macro-ECE was
0.036452, identity gain was +0.038723, and all Qwen prompts fit at 295-317
tokens. Preflight checksum:
`4ce44b0a0eacaa041106813649d6782be44c21790385c31fda03dbe605abecdb`.
This authorizes only the fresh 768-transition collection.

The collector then produced exactly 768 fresh rows, 256 per validation game,
under the four new seeds. All games used eight resets, action counts were
balanced within one, selection was outcome-independent, and 79 chronological
repeats were retained. Collection report checksum:
`8fc7989a30ec4a42e2c1d9d8f49592dc37371d17d1c9f0f406770c62e3fb8785`.
The raw shards were published before prospective evaluation.

The frozen prospective evaluator then completed `FAIL_CLOSED`, checksum
`27861c650c1cd51f5ee96c03e3ae297497a4d04e39f49391b1631840b43757ff`.
Its transactional path wrote 576 predictions and the structured intermediate
before Qwen. The structured branch passed 18/19 gates, with +0.703788
calibrated Brier skill, +0.307529 macro-F1 gain, +0.462532 outcome-shuffle
loss, 0.071950 macro-ECE, and positive transfer in every game. The sole
failure was binding-shuffle loss +0.017061 versus the +0.020000 minimum.

The initial invocation was externally stopped at six minutes during Qwen,
after the structured artifact was safe. One identical retry with a longer
orchestration timeout reproduced its checksum exactly and completed the
Qwen branch. Qwen failed all six separate gates: every response was
Markdown-fenced, strict validity and recall@8 were zero, shuffle loss was
zero, and Brier skill was -0.295586. No downstream fitting or controller
authority followed.

Final validation re-derived the structured and result checksums, confirmed
the expected 576/576/128/128 artifact row counts, passed all 26 targeted
V4.2/V4.2.1 tests in 31.33 seconds, and passed focused Ruff checks.

## V4.3 implementation freeze

The V4.3 implementation adds:

- the independent `sage12-bound-trajectory-v4.3` pair/tree schema;
- deterministic prefix replay with exact pre-state verification;
- outcome-blind same-action/different-argument branch selection;
- three identity-free binding projections and a source-only LOGO selector;
- calibrated Beta `BoundMechanicRule` prediction with support/evidence
  separation;
- action/history, action-only, binding-only, and deterministic baselines;
- executed binding swap, outcome shuffle, discordant-pair bootstrap,
  per-game transfer, calibration, and game-signature diagnostics;
- the gated `BoundSemanticWorldModel` with explicit occupancy updates,
  applicability constraints, horizon three, and beam width eight;
- fail-closed commands for source collection, preflight, validation
  collection, binding evaluation, and conditional world-model evaluation.

The frozen manifest checksum is
`2376ddd8c9c1c10083dc42ae92b9633ffc55272cf675770908b9467642370cea`.
Focused V4.3 and V4.2/V4.2.1 regression validation passed 37 tests in 26.21
seconds. At this checkpoint no V4.3 corpus or outcome has been opened. The
world model, EBM, and controller remain untrained and unauthorized.

The first source-collection invocation wrote no shard and stopped before its
first branch because `bp35` exposes duplicate byte-identical legal candidates.
The pre-outcome replay resolver now deterministically accepts the first
identical candidate; exact replay-state verification remains authoritative.
A regression test covers this representation detail. No schema, feature,
seed, quota, outcome rule, metric, or gate changed.

The successful source run then completed 352/352 roots, 2,396 pairs, and
4,792 arms with zero replay failures. Raw capacity includes 188 creation, 619
removal, and 8 movement positives. The collection is documented in
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_COLLECTION.md` and published before
source preflight.

The source preflight then failed before validation collection. The fail-closed
binding command now writes an explicit `SKIPPED_SOURCE_PREFLIGHT` artifact
instead of raising without a closure record; the world-model command consumes
that record and writes its own skipped artifact. This persistence-only change
does not fit a model or alter the frozen scientific verdict.

The final V4.3 status is `FAIL_CLOSED` at source preflight. Minimal,
relational, and typed projections had Brier skills of -0.1620, -0.1548, and
-0.0712, with identity gains of +0.2089, +0.2550, and +0.5624. Target movement
had only 8 positives. No validation game was opened, and no binding or world
model was fit. Full ledger:
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_RESULT.md`.

Final validation passed 39 focused V4.3/V4.2/V4.2.1 tests in 31.35 seconds,
passed focused Ruff checks, reloaded every closure artifact, and confirmed
that no source-validation shard directory exists.

## V4.4 implementation freeze

V4.4 adds an identity-free `PairEffectExample`, left-minus-right action and
binding projections, arm-conditioned temporal evidence, intercept-free
antisymmetric logistic models, slope-only calibration, four baselines,
source LOGO scoring, binding and complete-arm swaps, pair-difference identity
probes, per-game transfer, paired bootstrap gates, checksummed JSON model
serialization, and fail-closed validation collection/evaluation commands.

The source-only design audit admits 172 creation and 189 removal discordant
pairs across two substantial games per effect. Movement has no discordant pair
and is diagnostic-only. The frozen manifest checksum is
`598cdbca8ef50b05d3c9743cbbf4245c0e4c0495b81fd5fc3fd06e67bc623f5d`.
Pre-preflight validation passed 48 focused V4.4/V4.3/V4.2 tests in 29.08
seconds and focused Ruff checks. No V4.4 metric or validation outcome exists
at this checkpoint.

The frozen V4.4 source preflight then completed `FAIL_CLOSED`, checksum
`d58cc5825ab0932629496a1206b478ce310be88136e21e4b8799cc902dc18935`.
All projections had negative Brier skill. Relational was least negative at
-0.0337 skill and +0.0143 accuracy gain, but identity gain was +0.0906 and
ECE 0.1246. Binding swaps improved accuracy for every projection. Exact
complete-arm inversion passed at numerical error below `1.7e-16`.

The validation collector wrote only a `SKIPPED_SOURCE_PREFLIGHT` closure and
created no shard. The final result similarly records no validation opening or
downstream authority. Full ledger:
`reports/SAGE12_PAIRWISE_CAUSAL_PILOT_V4_4_RESULT.md`.

Final validation passed 48 focused V4.4/V4.3/V4.2 tests in 24.44 seconds,
passed focused Ruff checks, and verified the absence of both a source model
and validation shards.

## V4.5 implementation freeze

V4.5 adds deterministic tri-view object correspondence with explicit
ambiguity, split/merge hypotheses, intervention-exclusive event compilation,
fine-to-coarse source vocabulary discovery, identity-free rooted two-hop
graphs, temporal track buckets, action/history/root/template baselines,
antisymmetric source LOGO scoring, root-swap and relation-shuffle controls,
identity and calibration diagnostics, per-game transfer, paired bootstrap,
and mechanically closed downstream commands.

The protocol and code were frozen before feasibility. The manifest checksum
is `cfae89ac0de9f263af52dbb042e352869324f301a633012d44ad7b85ec028741`.
Focused V4.5 tests cover translation, recolouring, splits, ambiguity, common
dynamics, vocabulary backoff, the model firewall, temporal buckets,
antisymmetry, and fail-closed collection. No V4.5 outcome or validation data
existed at that checkpoint.

The official audit then completed `FAIL_CLOSED`, checksum
`1e19f7df0cdb315dc473e9a430c29e0cd29feda562e69d7a873d4d289b4099e6`.
It compiled all 2,396 V4.3 source pairs and 425,857 component assignments.
Correspondence confidence passed at 0.9840, but target grounding was 0.8130
and exclusive-event localization 0.8370. Eleven events passed capacity, all
local or collateral.

The structured model reached 0.7306 macro accuracy and 0.1448 Brier against
0.7267 and 0.1346 for the stronger root-without-history baseline: -0.0757
Brier skill and +0.0039 accuracy gain. Root-swap and relation-shuffle drops
were -0.0039 and -0.0222, identity gain was +0.2287, ECE was 0.1087, and the
bootstrap lower bound was -0.0120. Exact antisymmetry passed at `1.54e-16`.

All conditional collection/evaluation commands wrote explicit closure
artifacts. No fresh source or validation shard, model bundle, Qwen model, GNN,
world model, EBM, or controller was created. Full ledger:
`reports/SAGE12_OBJECT_CAUSAL_PILOT_V4_5_RESULT.md`.

The source audit ran on CPU in about two minutes. Its component matcher and
small scikit-learn linear models have no effective CUDA execution path, so the
laptop GPU was not used. Final validation passed 47 focused tests, focused
Ruff checks, the persisted artifact contract, and explicit absence checks for
fresh shards and a feasibility model.

## V4.6 global integration freeze

V4.6 adds the first end-to-end architecture harness. It loads all 340 complete
V4.3 replay trees, freezes a hierarchical endpoint utility, executes an
oracle ladder, trains the existing semantic world model and pairwise EBM with
leave-one-game-out firewalls, and routes their trajectories through the
existing receding-horizon controller. Deterministic left, action-only,
template, heuristic-energy, no-hierarchy, strict-Qwen, and relation-shuffle
ablations are included.

`SemanticWorldModel` now has an optional `action_key_mode="name"` for
cross-layout transfer; the original grounded-key behavior remains the
default. Qwen keeps the V1 weights and decoding. A separately reported
deterministic adapter can normalize emitted fenced legacy action/effect JSON,
but cannot invent a legal action or semantic predicate.

The source-only manifest was frozen before Qwen generation with checksum
`04c89af7426586169b603a373163da9eb03e60ede655ff95ce61125bb10e16c8`.
It selects 44 outcome-blind roots and two prompt variants, for 88 CUDA
generations. V4.6 is exploratory, cannot promote authority, and does not
change any V4.3–V4.5 verdict. Full protocol:
`reports/SAGE12_GLOBAL_INTEGRATION_PILOT_V4_6_PROTOCOL.md`.

## V4.6 result

The full integration pilot completed
`GLOBAL_CHAIN_NEGATIVE_PROPOSAL_OR_GROUNDING_BOTTLENECK`, checksum
`dc40f20ba3818eff6fa73fb5f86ef46828d399330ba7fc856dd3fe40f9dfe351`.
The oracle pipeline retained 1.00 best-action accuracy on all 340 roots and
gained +0.6657 utility over the stronger simple baseline, with paired interval
`[+0.4308, +0.9400]`. The myopic oracle scored 0.3120 below the depth-three
oracle, confirming hierarchical headroom.

Qwen ran all 88 original/shuffled prompts on the RTX 4050 in 847.7 model
seconds at 11.257 seconds median. Strict validity was zero; conservative
normalization retained 18/88 responses and 11/44 original roots. Qwen with
learned-world coverage but oracle energy was -0.2839 below the stronger
same-root baseline. The full chain was -0.3859 with interval
`[-0.9072, +0.0381]`. Learned depth three equalled depth one, relation
shuffling improved utility by 0.0944, and game-identity accuracy was 1.00.

The first CUDA launcher failed before loading Qwen because its Windows Store
virtual-environment shim was invalid; another existing Python 3.11 CUDA
environment loaded the identical local packages. Three evaluation attempts
then expired before result serialization due repeated graph construction,
full relation-state copying, repeated changed-cell counts, and thread-pool
overhead. The final implementation caches prebuilt graphs and deterministic
utilities, projects rollout state to consulted predicates, and uses one CPU
thread for the 16-unit EBM. These are exact runtime optimizations; frozen
inputs, scores, training pairs, and decision rules did not change. Final
focused validation passed 19 tests. Full result:
`reports/SAGE12_GLOBAL_INTEGRATION_PILOT_V4_6_RESULT.md`.
