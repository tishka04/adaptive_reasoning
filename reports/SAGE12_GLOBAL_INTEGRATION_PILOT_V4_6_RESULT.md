# SAGE12 global integration pilot V4.6 result

Date: 2026-07-28.

Outcome:
**GLOBAL_CHAIN_NEGATIVE_PROPOSAL_OR_GROUNDING_BOTTLENECK**.

This is a negative result for the current learned SAGE12 chain, but it is not
a global architecture refutation under the frozen rule. The fully oracle
pipeline preserved the best action on every informative root and exposed
clear multi-step headroom. The first collapse occurred when the oracle
hypothesis source was replaced by the current Qwen proposal/grounding
interface.

No live environment, source-validation game, historical game,
`NEURO_HOLDOUT_V1`, or `ar25` was opened. No authority was promoted.

## What ran

The pilot used all 340 complete, replay-verified V4.3 source trees:

- 11 games;
- two executed choices from each tree node;
- eight real depth-three leaves per complete root;
- leave-one-game-out world-model and EBM training;
- 44 outcome-blind Qwen roots, four per game;
- original and relation-shuffled Qwen prompts, for 88 generations.

The frozen manifest checksum is
`04c89af7426586169b603a373163da9eb03e60ede655ff95ce61125bb10e16c8`.
The result checksum is
`dc40f20ba3818eff6fa73fb5f86ef46828d399330ba7fc856dd3fe40f9dfe351`.

## Plain result

There is useful information in the action trees, and a three-step controller
could exploit it:

- stronger simple baseline utility: 7.380;
- direct three-step oracle utility: 8.046;
- oracle gain: +0.666, paired 95% interval `[+0.431, +0.940]`;
- myopic oracle utility: 7.734;
- hierarchical oracle gain over myopic action choice: +0.312;
- typed oracle pipeline: 100% coverage and 100% best-action accuracy.

So the basic `hypothesis → compiler → trajectory choice` skeleton can
represent the useful decision, and looking beyond the immediate action has
measurable value.

The learned chain did not recover that value:

- action-only utility on all roots: 7.337;
- deterministic template + heuristic world planning: 7.379;
- deterministic template + learned EBM: 7.342;
- Qwen full-chain utility on its 44 roots: 6.845;
- stronger same-root baseline: 7.231;
- full-chain gain: -0.386, paired 95% interval `[-0.907, +0.038]`.

The point estimate is negative and only 6/11 per-game gains were
non-negative, many because an invalid proposal simply fell back to the
deterministic left action. The exploratory-support rule therefore did not
pass.

## Oracle ladder

| Stage | Roots | Coverage | Mean utility | Best-action accuracy |
|---|---:|---:|---:|---:|
| Direct oracle | 340 | 1.000 | 8.046 | 1.000 |
| Typed oracle pipeline | 340 | 1.000 | 8.046 | 1.000 |
| Qwen strict + oracle world/energy | 44 | 0.000 | 7.030 | 0.419 |
| Qwen repaired + oracle world/energy | 44 | 0.250 | 6.947 | 0.323 |
| Qwen repaired + learned world + oracle energy | 44 | 0.250 | 6.947 | 0.323 |
| Qwen repaired + learned world + heuristic | 44 | 0.250 | 6.940 | 0.290 |
| Full learned chain | 44 | 0.250 | 6.845 | 0.290 |

Accuracy is measured on informative roots while utility includes fallback
behavior. The strict Qwen row has zero advisory coverage; its nonzero accuracy
comes only from the frozen left-action fallback.

The ladder localizes the losses:

1. **Proposal/grounding is the dominant collapse.** Qwen supplies a usable
   grounded option on only 11/44 original scenes. Even with true future
   outcomes and an oracle ranker, this restricted set scores -0.284 below the
   stronger same-root baseline.
2. **The learned semantic world model does not cause an additional coverage
   loss.** Its oracle-energy row is identical to the Qwen oracle row.
3. **Heuristic energy loses another 0.007 utility.**
4. **The learned EBM loses another 0.095 utility.**
5. **Depth three equals depth one in the learned arms.** The current semantic
   state/effect vocabulary does not carry information that the controller can
   compose over depth, despite the +0.312 oracle hierarchy headroom.

This is stronger and more useful than another component gate: it shows the
current proposal interface is the first blocker, while also showing that the
current EBM and trajectory representation fail to add downstream value.

## Qwen and GPU

The RTX 4050 completed all 88 frozen generations:

- median inference: 11.257 seconds;
- total model inference: 847.705 seconds;
- strict typed validity: 0/88;
- conservative normalized validity: 18/88 (20.45%);
- original scenes: 11/44 valid, 14 hypotheses;
- relation-shuffled scenes: 7/44 valid, 11 hypotheses.

The first CUDA launcher was a broken Windows Store virtual-environment shim
and failed before model loading. The run used another existing Python 3.11
CUDA environment with the same local Transformers installation. Weights,
prompt, schema, token caps, and decoding did not change.

## Controls

Relation shuffling changed the final selected action on only 11.36% of roots.
More importantly, shuffled relations improved utility:

- original minus shuffled utility: -0.094;
- paired 95% interval: `[-0.185, -0.023]`.

This is evidence against productive use of the supplied relation structure.

The fixed scene view still identifies the source game perfectly:

- five-fold game-identity accuracy: 1.000;
- majority accuracy: 0.094;
- gain: +0.906.

The logistic diagnostic reached perfect held-fold predictions despite
iteration-limit warnings. That reinforces, rather than weakens, the leakage
finding, but the exact coefficient fit is not used by any decision method.

Template and hierarchy controls also remain negative:

- template heuristic utility 7.379, essentially deterministic-left 7.380;
- learned EBM utility 7.342, below heuristic;
- learned depth three and learned depth one are exactly equal.

## Runtime corrections

Three evaluation attempts reached their wall-time limit before writing a
result. They exposed performance bugs in the harness, not model outcomes:

1. the controller rebuilt the identical scene graph for every ablation;
2. rollout copied thousands of irrelevant relation predicates despite having
   no compiled preconditions;
3. branch utilities repeatedly recounted the same changed cells;
4. PyTorch used a large CPU thread pool for a 16-unit network.

The final harness reuses the prebuilt graph, seeds rollout with only compiled
preconditions plus terminal predicates, caches utility by trace/root digest,
and trains the tiny EBM on one CPU thread. Focused tests passed after each
change. No input, hypothesis, option, trajectory probability, energy feature,
training pair, utility, threshold, or output stream changed. The successful
evaluation completed in about 31 seconds after caches were added.

## Scientific interpretation

V4.6 does **not** support the current learned implementation. It also does
not justify saying “higher semantics cannot work.”

What it establishes is narrower:

- the offline trees contain exploitable multi-step headroom;
- the compiler/controller skeleton can transmit an oracle decision exactly;
- Qwen2.5 0.5B with the current scene contract fails to expose enough legal,
  discriminating alternatives;
- the current learned world/energy stack does not recover the missing value;
- the current depth-three abstraction behaves like depth one;
- relation use and cross-game invariance are still absent.

The architecture hypothesis remains open as an existence claim, but the
current instantiation is negative. The next credible architecture test should
not merely lower a scalar gate. It should give the proposer a constrained
action-slot interface that always represents every legal candidate, let Qwen
rank or annotate those slots instead of generating action identity, and train
the world/EBM on object-relative action bindings and multi-step return. The
same V4.6 oracle ladder can then be rerun unchanged.

## Published artifacts

- `frozen_manifest.json`: frozen protocol and data checksums;
- `qwen_outputs.jsonl`: all 88 raw, strict, and normalized outputs;
- `qwen_summary.json`: device, validity, and timing;
- `folds.jsonl`: 11 world-model/EBM training summaries;
- `decisions.jsonl`: every method/root decision and realized utility;
- `result.json`: complete metrics, comparisons, diagnostics, and checksums.

Primary artifact checksums:

- Qwen outputs:
  `86439fda99a18e540e0b579f3fda2c5ed6b7d6b04aa9934c1f009102c8d2e169`;
- decisions:
  `05cfb171ac40d1fc45856a1b371fd640c0379cdfe1eb987acc98b19678dc671b`;
- folds:
  `9e9d6ee48f5fcd4b33dbb125939a9c8f3ac9a76ba657485a0e8e7fbfa01f522e`.
