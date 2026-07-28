# SAGE12 global integration pilot V4.6 protocol

Date frozen: 2026-07-28.

V4.6 is the first deliberately end-to-end SAGE12 architecture probe. The
previous V4.3–V4.5 gates remain valid tests of their individual
representations, but they did not execute the proposed global chain. This
pilot therefore does not reuse their promotion thresholds and does not stop
at intermediate predictive gates.

The question is:

> When hypotheses, a semantic world model, trajectory energy, and a
> hierarchical receding-horizon controller are composed, where does useful
> action selection first disappear?

This is an exploratory, source-only, offline test. It cannot promote live
authority and it does not weaken or relabel any earlier confirmatory result.

## Frozen data and firewall

The input is the already published V4.3 replay-verified counterfactual corpus:

- 340 complete roots from 11 SAGE11 source-training games;
- seven paired nodes per root and eight executed depth-three leaves;
- exactly two real actions executed from each node's identical pre-state;
- no source-validation, historical, `NEURO_HOLDOUT_V1`, `ar25`, or live
  environment access.

All deterministic arms use all 340 roots under leave-one-game-out training.
Qwen uses a hash-selected, outcome-blind sample of four roots per game:
44 roots, evaluated once with original relations and once with a deterministic
relation-binding shuffle. The 88 prompts are frozen before generation.

Manifest:
`training/sage12/integration_pilot_v4_6/frozen_manifest.json`.
Checksum:
`04c89af7426586169b603a373163da9eb03e60ede655ff95ce61125bb10e16c8`.

## Frozen endpoint utility

Each actual transition receives:

- +20 for level completion or level-count progress;
- -20 for game over;
- +2 for a productive non-noop;
- +1 for actor displacement;
- +1 for each grounded target creation, removal, or movement effect;
- up to +1 logarithmic credit for changed cells.

Three-step utility is discounted by 0.9. For a first action, the endpoint
value is the best actually executed depth-three leaf below that action. This
implements the hierarchy `complete level → avoid failure → obtain productive
change`; it is fixed before evaluation. The primary continuous measurements
are selected utility, regret to the best observed branch, normalized utility,
oracle-action accuracy, coverage, and unsafe first-action rate.

The utility is not presented as a universal ARC score. It is a consistent
offline surrogate for determining whether the composed controller can exploit
the counterfactual headroom present in these trees.

## Oracle ladder

The ladder removes one oracle at a time:

1. **Direct oracle:** select the first action with the best true depth-three
   branch.
2. **Oracle pipeline:** express both real candidate actions as typed
   hypotheses, compile them, then use oracle world and energy ranking.
3. **Real Qwen + oracle world/energy:** Qwen proposals and compiler grounding
   restrict the legal candidate set; true branch value ranks what remains.
4. **Real Qwen + learned world + oracle energy:** the cross-game semantic
   world model must produce a trajectory for the candidate; true branch value
   still ranks the surviving first actions.
5. **Full learned chain:** Qwen → compiler → learned semantic world model →
   learned pairwise EBM → hierarchical controller.

The semantic world model is the repository's `SemanticWorldModel` with
Beta-smoothed effects. V4.6 adds a `name` action-key mode so training can
transfer across different grounded click coordinates; the existing
`grounded` default is unchanged. It learns only from other games in each
fold.

The pairwise EBM is the existing 16-hidden-unit
`PairwiseTrajectoryEBM`. It is trained on other-game template trajectories
whose actual branch ordering supplies the pairwise preference. The EBM stays
on CPU because it is tiny. Qwen uses the RTX 4050 CUDA environment because
the already published identical-decoding benchmark measured a 3.808× median
speedup.

## Qwen contract

The local frozen Qwen2.5 0.5B Instruct weights, temperature zero, sampling
disabled, 256 new-token cap, eight-hypothesis cap, and compact 24-entity /
96-relation prompt are unchanged.

Two outputs are reported separately:

- **strict:** the existing typed JSON parser;
- **deterministically normalized:** a post-decode adapter may remove one
  Markdown fence and map the already emitted legacy `{action_id, effect}`
  shape into the typed contract.

The adapter cannot invent an action or effect. It emits a hypothesis only
when the output already contains a legal action and an allowed semantic
predicate, and it always preserves `support=0`. This tests whether schema
friction, rather than semantic content, is the first bottleneck without
misreporting repaired output as strict validity.

## Baselines and ablations

The same executed roots compare:

- deterministic left-branch fallback;
- leave-one-game-out action-identity mean;
- deterministic template hypotheses;
- direct and compiler-mediated oracles;
- heuristic energy instead of the learned EBM;
- depth one instead of hierarchical depth three;
- strict versus normalized Qwen;
- original versus relation-shuffled Qwen scenes.

A five-fold game-identity probe on action availability, entity-role/shape
counts, and relation-kind counts is reported as a leakage diagnostic.

## Interpretation rule

There is no `+0.10 macro-F1` promotion gate in V4.6.

- The architecture is **refuted in this scope** only if the oracle pipeline
  has no positive utility headroom over the stronger simple baseline, or if
  typed compilation cannot preserve at least 0.95 oracle-action accuracy.
- The full chain receives **exploratory support** if its point-estimate
  utility gain over the stronger same-root action/template baseline is
  positive and its gain is non-negative on at least 6 of 11 games.
- Otherwise the earliest oracle-ladder collapse is reported as a component or
  interface bottleneck. That outcome is negative evidence for the current
  implementation, not a blanket refutation of every higher-semantic
  architecture.

Paired 1,000-sample bootstrap intervals are descriptive. They are not used to
move a threshold after seeing the result. Regardless of outcome,
`authority_promoted=false`; no live action is executed.

## Reproduction

```powershell
python -m theory.sage12.integration_pilot freeze
agi\Scripts\python.exe -m theory.sage12.integration_pilot generate-qwen --device cuda:0
agi\Scripts\python.exe -m theory.sage12.integration_pilot evaluate
```

Focused tests:

```powershell
python -m pytest -q tests/test_sage12_integration_pilot.py tests/test_sage12_semantic_planning.py
```
