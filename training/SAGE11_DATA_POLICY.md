# SAGE.11 data policy and manifest contract

Status: policy and collection path implemented; the 100,000-transition corpus
has not yet been collected or committed. This document is the pre-registration
for that collection and must not be edited in response to holdout outcomes.

## Frozen game roles

| Role | Games | Allowed use |
| --- | --- | --- |
| Source train | `bp35 cd82 dc22 g50t ka59 lf52 lp85 sp80 su15 tr87 tu93` | Schema curriculum, pilot training, world-model training |
| Source validation | `re86 ls20 sc25` | Pilot/model gates and all tuning |
| `NEURO_HOLDOUT_V1` | `s5i5 vc33 m0r0 sk48 r11l` | One final active-vs-off confirmation only |
| Historical benchmark | `wa30 tn36 ft09 cn04 sb26` | ft09 non-regression plus one final report; never tune |
| Regression only | `ar25` | Regression reporting only |

Registry checksum:
`22de34d38dbd1ce4169fd09694f4449e9ad531f5536dccfb274d2a41de2ce37d`.

## Collection policy

Target at least 100,000 accepted transitions. Assignment is deterministic from
game, seed, reset, and step:

- 70% current controller active arm;
- 20% uniform legal action;
- 10% frontier-stall-triggered probe.

Each game is capped at 8,000 accepted transitions. Duplicate
`before/action/arguments/after/effect` signatures are discarded. Every ACTION6
argument-key pattern and `(x,y)` pair is counted. Collection writes one
checksummed JSONL shard at a time so interrupted runs remain recoverable.

No M2 or v4 weights may be loaded. `verify_manifest()` rejects split checksum
drift, legacy-weight declarations, and shard checksum mismatches.

## Labels

Strong labels are observed level completion, WIN, and terminal events. Weak
progress labels are:

- `frontier_credit`: delayed frontier eligibility receives observed credit;
- `subgoal_graph_advance`: a causal-subgoal or locally confirmed transferred
  effect advances;
- `route_confirmation`: an executed protected route reaches level/WIN;
- `subeffect_relay`: a causally linked sub-effect relay is observed.

Weak labels have weight 0.25 for the progress loss. They never supervise the
terminal head. The terminal head stays disabled until the manifest contains at
least 100 strong events.

## Required manifest fields

The manifest records its format, split checksum, mixture weights, target and
cap, accepted counts by game/arm, strong and weak event counts, ACTION6
coverage, every shard path/checksum/count/game set, whether the terminal head
may be enabled, and `legacy_weights_loaded=false`.

Data shards belong under `training/` as `.jsonl` files and are already covered
by the repository's Git LFS rule. A corpus is publishable only when every shard
passes `verify_manifest()`.
