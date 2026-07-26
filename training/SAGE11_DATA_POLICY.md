# SAGE.11 data policy and manifest contract

Status: policy amended by explicit user approval after a measured
exact-dedup-capacity failure. The amendment changes only the aggregate cap; no
holdout outcome was observed.

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

Target exactly 100,000 published accepted transitions. Assignment is
deterministic from game, seed, reset, and accepted-row index in shuffled
10-transition blocks:

- 70% current controller active arm;
- 20% uniform legal action;
- 10% frontier-stall-triggered probe.

The base cap remains 8,000 accepted transitions per game; the cap is not a
quota. After the five-seed capacity result proved a minimum 1,292-row
shortfall, the user approved one global overflow pool of exactly 1,292 rows.
It is distributed deterministically across proven high-capacity
source-training games:

- `cd82`: +259 (cap 8,259)
- `dc22`: +259 (cap 8,259)
- `g50t`: +258 (cap 8,258)
- `ka59`: +258 (cap 8,258)
- `tr87`: +258 (cap 8,258)

No validation, holdout, historical, or regression game receives overflow.
Duplicate exact `before/action/arguments/after/effect` signatures are
discarded. A finite-state game is declared saturated only after 4,000
consecutive duplicate signatures on every registered seed. Seeds `0` through
`4` rotate deterministically in 200-reset windows and keep independent
duplicate-streak counters. The bounded window is long enough for seed-local
controller context while preventing one sparse seed from consuming the full
multi-seed raw budget. Controller learning state is fresh at each seed-window
boundary; the per-window frozen causal libraries are content-addressed and
merged for that source game at completion. The accepted-row arm schedule uses
one canonical seed-independent sequence, so seed rotation cannot disturb its
exact 70/20/10 blocks.
Collection first exhausts source-training capacity, then balances the
remaining requirement across the three source-validation games. A final
partial publication block may make aggregate counts differ from the ideal
ratio by only its bounded tail. Every ACTION6 argument-key pattern and `(x,y)`
pair is counted.

Collection checkpoints in-progress, completed, and saturated games into
private work shards. An interrupted in-progress shard resumes only when its
row checksum, quota, format, and accepted count verify; a fresh controller
continues from the next deterministic seed while the exact accepted-row
schedule continues from the restored row count. Source games also checkpoint
a content-addressed frozen schema snapshot. Because per-game limits are caps,
not quotas, collection closes a verified in-progress game as
`TARGET_CAPACITY_REACHED` when all game shards and source-schema snapshots
verify and their aggregate unique capacity is at least 100,000. It then writes
separate checksummed publication shards. The private work shards are ignored
by Git; only manifest-referenced publication shards are released.

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

The manifest records its format, split checksum, mixture weights, target,
8,000-row base cap, exact per-game amended caps, aggregate overflow, accepted
counts by game/split/arm, strong and weak event counts, ACTION6 coverage,
every shard path/checksum/count/game set, whether the terminal head may be
enabled, and `legacy_weights_loaded=false`.

Data shards belong under `training/` as `.jsonl` files and are covered
recursively by the repository's Git LFS rule. A corpus is publishable only when
the target, per-split/per-game/per-arm totals, row counts, and every shard
checksum pass `verify_manifest()`.

The production command is:

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.source_dataset_runner --workers 8
```

The five-seed run found only 27 exact transitions for `lp85` and 2,681 for
`sp80`. Even granting every other game its base cap, the upper bound was
98,708, 1,292 below target. The user approved the minimum aggregate overflow
above; the original signed evidence is preserved in
`reports/SAGE11_SOURCE_CAPACITY_RESULT.md`.

## Published amended corpus

The amended run completed on 2026-07-26 with exactly 100,000 verified rows:
76,908 source-train and 23,092 source-validation. Its manifest checksum is
`d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`.
The fixed mixture yielded 69,999 active-controller, 20,002 uniform-legal, and
9,999 frontier-probe rows; the bounded two-row deviation is due only to final
shard tails. The corpus contains 44 strong and 3,821 weak progress events, so
the terminal head remains disabled. The 11-game frozen curriculum checksum is
`d11948c5cfcb70ce888b435d63d217b95ce2a0006e4423ae7ac70374d81c630c`.
