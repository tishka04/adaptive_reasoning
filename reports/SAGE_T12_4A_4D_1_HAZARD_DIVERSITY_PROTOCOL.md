# SAGE.T12.4a.4d.1 — Abstract hazard and diverse search protocol

## Scientific question

T12.4a.4d was an integrity-clean negative result: no arm found progress and
both terminal rates exceeded 10%. The contract-guided arm also selected
`ACTION6` on all 615 explored transitions, while the frozen exact-cell shield
recognized none of the 101 distinct terminal state/action pairs encountered at
level 1.

T12.4a.4d.1 asks two narrower questions without adding a network:

1. can terminal risk be transported through a target-local, identity-free
   action context rather than an exact archive cell id;
2. can explicit action-family balancing prevent the search policy from
   collapsing onto one action schema?

The experiment remains restricted to `bp35` source-train. It grants no
validation, holdout, option-control or production authority.

## Frozen offline abstraction

The compile phase consumes only the sealed T12.4a.4d archives from search seeds
9101, 9102 and 9103. Duplicate interventions are removed by search seed,
lineage, exact source hash and grounded action. A hazard signature contains:

- the action schema;
- whether the action is coordinate-grounded;
- entities within a seven-cell square around the intervention;
- their relative row/column offsets, typed attributes and informative roles.

It excludes absolute coordinates, entity ids, cell ids, exact hashes and game
identity. A signature is unsafe with at least two observations and a terminal
rate of at least 0.75.

Before the scientific freeze, a read-only source diagnostic found 1,193 unique
interventions and 208 terminal examples. Leave-one-search-seed-out evaluation
gave micro recall 0.5144, precision 0.9640 and false-positive rate 0.0041. Two
of three folds exceeded 0.50 recall; seed 9103 remained weak at 0.1143 recall.
These values motivated the frozen gates but are not a scientific compile
receipt. The prospective run is required precisely because the fold behavior
is heterogeneous.

The compile gate requires micro recall at least 0.50, precision at least 0.90,
false-positive rate at most 0.02, and at least two folds with recall at least
0.50. Failure stops all physical execution.

## Prospective three-arm design

Fresh seeds 9201, 9202 and 9203 are crossed with the two exact route lineages
8701 and 8705. Every condition runs:

1. `local_archive_control`: the unchanged symbolic archive ordering and frozen
   exact terminal shield;
2. `diversity_control`: the contract scorer under explicit action-family
   balancing, with the same frozen shield;
3. `abstract_hazard_diversity`: the identical diverse policy plus the compiled
   local hazard vetoes.

The second contrast identifies action diversity. The third identifies the
incremental effect of hazard abstraction. All arms share the initial action
catalogue, exact anchor, lineage schedule, burst schedule and 2,048-call
budget. The complete run is capped at 38,000 SDK calls, 10,000 cells per
archive and 3 GiB. Raw frames are never persisted.

## Active gate

The gate passes only if:

- all anchors and archive restorations are exact;
- all three action catalogues are identical within every paired condition;
- the old contracted option remains blocked;
- the abstract shield actually vetoes at least one proposal;
- neither diverse arm allocates more than 70% of actions to one schema;
- the abstract-hazard arm has terminal rate at most 10% and no greater than
  the diversity-only control;
- per-arm and global SDK limits hold;
- a suffix of at most 64 actions reaches a higher level;
- that suffix is confirmed twice from each exact lineage, without terminal
  failure, and all confirmations reach one common exact hash.

A pass authorizes only a separate T12.4a.4e option-extraction freeze. A guidance
claim additionally requires progress in the abstract-hazard arm with no higher
terminal rate than the diversity control.

## Negative-result policy

A compile miss forbids the prospective run. An active miss retains its signed
archives, bundles and report but leaves T12.4a.4e closed. Seeds, thresholds,
budgets and action catalogues must not be substituted under the same manifest.
No miss opens source validation, holdout data, neural control or production.

