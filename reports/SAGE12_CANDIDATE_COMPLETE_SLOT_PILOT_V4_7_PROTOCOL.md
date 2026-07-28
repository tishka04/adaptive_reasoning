# SAGE12 candidate-complete semantic-slot pilot V4.7 protocol

## Question

V4.6 showed that the three-step architecture has real oracle headroom, but its
free-form Qwen proposal and grounding boundary retained too few legal actions
to test the downstream learned chain cleanly. V4.7 asks whether the same
architecture becomes useful when proposal coverage is structural: every legal
action is represented by a persistent semantic slot and Qwen can annotate, but
never remove, a slot.

This is an exploratory source-only experiment. It cannot promote authority,
open validation or holdout games, collect a new transition, or execute an
action in a live environment.

## Frozen source view

The only transitions are the replay-verified complete binary trees in the
immutable V4.3 source-training shards. The runner admits 340 complete roots,
2,380 pre-action nodes, and 4,760 executed legal slots across the same 11
source games used by V4.6. The observed difference from the planning estimate
of 2,396 nodes is recorded rather than filled: only nodes belonging to complete
roots are eligible.

Each slot contains two distinct views:

- an exact execution key, retained only for matching the selected first action
  to the archived branch;
- an identity-free semantic signature containing action/family, requested
  direction, occupied/path state, actor relation and direction, target
  size/aspect bucket, and target affordance.

Coordinates, object IDs, raw colors/values, frame hashes, game identity, and
future outcomes are forbidden from model inputs. The last eight already
observed context events are allowed. Their effects are history, not query
labels.

## Candidate-complete compiler

`SemanticActionSlot` and `SlotAnnotation` are public SAGE12 interfaces.
Annotations contain continuous scores for exactly:

1. `changed`;
2. `moved`;
3. `target_created`;
4. `target_removed`;
5. `target_moved`;
6. `level_complete`;
7. `game_over`.

`support` is always zero. `HypothesisCompiler.compile_slots` emits exactly one
compiled option per legal slot even when all seven scores are zero. Proposal
coverage is therefore 1.00 by construction and is not treated as a learned
metric.

## Frozen Qwen annotation

The local Qwen2.5 0.5B model remains frozen, with no download and no
fine-tuning. One prompt contains the same eight-event context and both legal
slots. The semantically equivalent compact serialization is capped at 512
tokens; the preflight maximum is 470 after the chat template.

The decoder does not generate free JSON. It performs 14 autoregressive
positions. At every position all vocabulary logits except the atomic `0` and
`1` tokens are masked; their two-way softmax is stored as the continuous
effect score and the larger token is fed to the next position. Decoding has no
sampling, temperature, repair, or post-hoc schema inference.

Qwen processes:

- all 2,380 original nodes;
- all 340 roots a second time with a deterministic relation-field shuffle.

The RTX 4050 is preferred, with batch 32. Device, token IDs, timing, token
lengths, prompt checksums, scores, hard bits, and output checksums are
published.

## World models and leakage firewalls

Evaluation has 11 outer leave-one-game-out folds. Each fold fits:

- a structured-slot-only world model;
- the same model with Qwen scores;
- an oracle-annotation diagnostic with true current-slot effect bits.

Each applicable effect uses a regularized logistic head. A label with fewer
than two positive or two negative training examples uses a Beta-smoothed
constant instead. Non-applicable target effects are forced to zero. Immediate
V4.6 utility uses a ridge head.

Calibration is fitted only on inner leave-one-game-out predictions from the
outer training games. Those same nested out-of-game predictions generate the
EBM training features. No label, effect, utility, or fitted parameter from the
outer game enters its world model, calibration, baseline choice, or EBM.

## Trajectories, energy, and control

For every root, the primary diagnostic constructs all eight depth-three leaves
using the archived future nodes' pre-action slot descriptions. Future frames,
effects, and utilities remain hidden. This topology is explicitly an oracle
and is not deployable. A control reuses the two root slots at later depths,
matching the deployable limitation of V4.6.

Each leaf has eight learned features:

1. discounted predicted return;
2. probability of at least one completion;
3. probability of at least one game-over;
4. expected productive-step count;
5. mean Bernoulli effect entropy;
6. accumulated value uncertainty;
7. cost in steps;
8. target-effect contradiction.

`PairwiseTrajectoryEBM` uses eight inputs, 32 hidden units, 150 epochs, and
learning rate 0.003. It trains on every unequal-utility pair of leaves within
each outer-training root. The controller ranks trajectories but returns only
the first slot, preserving receding-horizon semantics.

## Baselines, ablations, and oracle ladder

Every outer fold selects its primary baseline using training games only from:

- deterministic left;
- action-name-only;
- action-sequence-only.

V4.6 template and Qwen results are included as historical reported baselines,
not mixed into the new root population. V4.7 also evaluates:

- structured slots versus structured plus Qwen;
- depth three versus depth one;
- oracle topology versus root-slot reuse;
- original versus relation-shuffled Qwen root scores;
- true annotations;
- true world outputs;
- true leaf energy.

Metrics include first-branch utility and regret, exact selected-leaf utility
and regret, informative-root action accuracy, leaf accuracy, safety,
coverage, effect Brier/ECE/recall, utility RMSE, per-game transfer, relation
sensitivity, and game-signature leakage. Utility and discount are exactly
V4.6. Paired intervals use 1,000 frozen bootstrap resamples.

## Verdict

V4.7 returns `EXPLORATORY_SUPPORT` only when the full Qwen, learned-world,
learned-EBM, oracle-topology stack has positive mean first-branch utility gain
over the fold-selected primary baseline and is non-negative on at least 6 of
11 games.

Otherwise the first positive oracle recovery is attributed to
`QWEN_SEMANTICS`, `WORLD_MODEL`, `ENERGY`, or `ROLLOUT_TOPOLOGY`. If no single
replacement suffices, the result is
`CURRENT_STACK_NEGATIVE_MULTIPLE_BOTTLENECKS`.

Confidence intervals, identity leakage, calibration, and relation sensitivity
remain descriptive. No threshold is changed after results are observed. A
positive result would support further offline work only; it would not validate
deployability because the primary tree topology is oracle.

## Reproduction

```powershell
python -m theory.sage12.integration_pilot_v4_7 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.integration_pilot_v4_7 generate-qwen --device cuda:0
python -m theory.sage12.integration_pilot_v4_7 evaluate
```

The manifest and source checksums are frozen in
`training/sage12/integration_pilot_v4_7/frozen_manifest.json` before Qwen is
run. Results are published positive or negative before any semantic world
model is reused elsewhere.
