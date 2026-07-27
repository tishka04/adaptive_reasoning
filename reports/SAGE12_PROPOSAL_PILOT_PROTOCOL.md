# SAGE12 grounded-proposal pilot protocol

Status: frozen before device timing, collection, model generation, or outcome
evaluation.

Frozen manifest:
`training/sage12/proposal_pilot_v1/frozen_manifest.json`

Manifest checksum:
`0260eb15fd9a0cecb21644160888bde9b6e5be03b4428f1afd989401686c148b`

## Scope and firewall

The pilot reuses the immutable SAGE11 split registry checksum
`22de34d38dbd1ce4169fd09694f4449e9ad531f5536dccfb274d2a41de2ce37d`:
11 source-training games and three source-validation games. Holdout,
historical, and `ar25` environments are forbidden.

The collector must publish exactly 2,104 executed rows:

- 160 rows for each of ten non-`lp85` source-training games;
- 24 rows for finite-capacity `lp85`;
- 160 rows for each of the three source-validation games;
- 1,624 source-training and 480 source-validation rows in total.

Rows are chosen by a deterministic policy that balances legal action
families, then selects an action instance with a frozen seeded random choice.
No state/action pair may be archived more than four times. Raw coordinates are
kept only as executed-action provenance; scene graphs exclude coordinates,
raw grids, raw colors, game ID, and state hashes from model-facing fields.

## Local-model device decision

Model: the existing local Qwen2.5 0.5B Instruct weights, SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

The CPU and RTX 4050 paths receive identical prompts, schema, weights,
temperature 0, sampling disabled, and maximum 256 new tokens. Four initial
source-only scenes (`bp35`, `cd82`, `re86`, `ls20`) are timed. GPU is selected
only when median generation is at least 1.10× faster. No task-quality outcome
is used for device selection. Exact CPU/GPU response equality is reported but
is not a selection gate.

## Representative evaluation

Eight rows per game are selected by the lowest SHA-256 values under the frozen
salt `sage12-proposal-pilot-sample-v1`; selection does not inspect outcomes.
The resulting 112 scenes receive:

1. the original relation graph;
2. a deterministic entity-permutation relation shuffle.

This produces 224 local-model generations. Every proposal is strictly parsed,
required to retain `support=0`, and compiled against legal action families.
An argument-free action-family hypothesis may expand to exact currently legal
action arguments only inside the deterministic compiler.

Productive-mechanism recall excludes the generic `changed` and weak
`progress` labels. It scores grounded moved, relational, existence, terminal,
and level-completion mechanisms for the actually executed action. Rows with no
such observed mechanism are ineligible rather than counted as negatives.

## Baselines and leakage control

- Template baseline: the fixed SAGE12 deterministic template generator.
- Action-only baseline: the single majority semantic mechanism per action,
  fitted on all source-training collection rows only.
- Relation control: repeat Qwen generation after deterministic entity binding
  permutation while preserving entity inventory and action identity.
- Game-identity probe: three-fold logistic regression on source-training
  entity-role/shape counts, relation counts, and available-action flags.

## Frozen gates

All gates must pass:

- strict JSON validity at least 0.99;
- parsed-proposal `support=0` rate at least 0.95;
- grounded hypothesis rate at least 0.90;
- source-validation recall@8 gain at least +0.10 over the stronger template or
  action-only baseline;
- relation shuffle degradation at least 0.10;
- non-negative gain on each of `re86`, `ls20`, and `sc25`, with at least one
  eligible row per game;
- game-identity accuracy no more than majority accuracy +0.10.

Any failed gate produces `FAIL_CLOSED`, authorizes no world-model fitting, and
must be documented and published without threshold changes.

## Reproduction order

```powershell
python -m theory.sage12.proposal_pilot_runner benchmark
python -m theory.sage12.proposal_pilot_collection --workers 4
python -m theory.sage12.proposal_pilot_runner evaluate
```

The first command must run only after this protocol and manifest are committed.
The final command must run only after the checksummed collection is complete.
