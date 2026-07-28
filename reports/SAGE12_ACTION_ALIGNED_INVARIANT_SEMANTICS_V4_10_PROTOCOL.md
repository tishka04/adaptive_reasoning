# SAGE12 V4.10 — Action-aligned invariant semantics protocol

Status at freeze: **source-only exploratory protocol; no collection or model
result observed**.

## Motivation

V4.9 validated the deterministic semantic teacher but rejected its first
object-relative student. Compass relations transferred with the wrong sign,
the root-only network ranked productive pairs better than the full graph,
semantic outputs exposed game identity at 90.64%, and completion recall@8 was
zero.

V4.10 tests the narrow follow-up implied by that diagnosis. It does not change
the global SAGE12 architecture and cannot grant live authority.

## Data policy

Only the eleven immutable SAGE11 `SOURCE_TRAIN` games may be opened. The
collector uses their local offline environments. Source validation, historical
games, neural holdout and online/live environments remain closed.

Fresh fixed quotas are:

- 160 unique transitions for each source game except `lp85`;
- 64 unique transitions for `lp85`;
- 1,664 total requested transitions.

An exact pre-state/action/argument repeat already present in V4.9 or V4.10 is
not retained. Collection uses 30% outcome-blind coverage exploration and 70%
past-yield targeting. Its scoring can use only previously observed source
outcomes; the current candidate is selected from pre-action fields.

The targeted functional effects are local change, path opening/closing,
approach/contact, reachability increase/decrease, target creation/removal/
movement, productive change and risk. Quotas are row quotas, not promised
positive-label quotas: the collector may not manufacture a mechanic that a
game does not contain.

Every game must reach at least 90% of its row quota before teacher compilation.
The raw source shards and collection report are published before model fitting.

## Action-aligned representation

The V4.9 graph is deterministically transformed:

- movement actions use their requested movement axis;
- targeted actions use the actor-to-target axis when available;
- neighboring objects become `ahead`, `behind`, `lateral_left`,
  `lateral_right`, `overlap`, or `radial`;
- compass directions and row/column alignment fields are deleted;
- contact, adjacency, relative size, coarse shape, actor role and boundary
  contact remain;
- root features add contact/adjacency degree, actor-neighbor presence and
  ahead-contact topology.

Coordinates are used only by the already audited grounding compiler. They do
not enter the graph. Game IDs, object IDs, colours/raw values, frames and future
state remain forbidden.

## Invariant student

The DeepSets encoder retains stable hashed embeddings and permutation-invariant
neighbor pooling. V4.10 changes training rather than increasing model size:

- every optimizer step receives the same number of rows from every training
  game;
- semantic loss is averaged per game, then across games;
- the identity-confusion weight increases from 0.08 to 0.20;
- latent game-mean alignment and semantic-output game-mean alignment are added;
- same-prestate productive ranking remains an auxiliary objective;
- a fixed calibration rule shifts each effect logit to the game-balanced
  training prevalence. It is fitted only on the ten outer-training games.

All published probabilities are strict outer leave-one-game-out. For each held
game, neither its labels nor its rows enter training or calibration.

## Baselines and controls

The action-aligned student is compared with:

- action-only source rates;
- the same invariant network with no neighbor set;
- the frozen V4.9 object-relative result.

Controls rotate only intervention-relative axes, reverse neighbor order, probe
game identity from semantic outputs, measure per-game transfer, rank
same-prestate productive contrasts, and retrieve completion events at rank
eight.

## Frozen exploratory decision

All checks are required:

- collection row ratio ≥ 0.90 in every game;
- macro-Brier strictly better than action-only;
- macro-Brier strictly better than root-only;
- macro-Brier strictly better than V4.9;
- productive-pair accuracy strictly better than root-only;
- rotating action-relative relations strictly worsens macro-Brier;
- neighbor-order probability delta ≤ `1e-6`;
- semantic-output identity accuracy ≤ 0.60;
- identity accuracy improves by at least 0.15 from V4.9;
- completion recall@8 ≥ 0.20;
- at least 6/11 games have Brier no worse than action-only.

These are exploratory support criteria, not architecture-wide falsification
thresholds. A negative result rejects this representation/training/data
combination and still blocks the semantic world model and EBM.

## Publication sequence

1. Publish implementation, tests, protocol and frozen manifest.
2. Run and publish the source-only raw collection.
3. Compile and publish the augmented teacher corpus and QA.
4. Train on GPU, evaluate LOGO, export V4.7-compatible base effects, and
   publish the result whether positive or negative.

## Capacity amendment before model fitting

The source collector subsequently filled every frozen quota except `su15`.
That game exposes only `ACTION6` and produced 83 new unique rows after all 40
resets (1,280 executed steps), while rejecting 1,197 exact repeats already in
V4.9/V4.10. No model result had been opened.

The narrowly scoped amendment authorizes a minimum of 80 rows for `su15`;
every other per-game quota remains unchanged. The total minimum becomes 1,584,
and the observed 1,587 rows pass it. Representation, training, evaluation
thresholds and data-access policy are unchanged. See
`reports/SAGE12_ACTION_ALIGNED_SEMANTICS_V4_10_COLLECTION.md`.
