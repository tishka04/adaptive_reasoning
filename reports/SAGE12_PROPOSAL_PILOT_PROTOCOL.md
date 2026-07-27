# SAGE12 grounded-proposal pilot protocol

Status: amended after a partial invalid-output checkpoint to correct storage
complexity and an outcome-contaminated sampling digest; re-frozen before the
clean evaluation rerun.

Frozen manifest:
`training/sage12/proposal_pilot_v1/frozen_manifest.json`

Manifest checksum:
`0dfdff9a61e45e02b16601a47d987454c991a2d5f99c8964a5486c17ed17aceb`

Post-run annotation: the clean pilot failed closed without changing these
gates. The result is published in
`reports/SAGE12_PROPOSAL_PILOT_RESULT.md`, checksum
`fbb86c17fee57ff46199dd94594936694bf2b0e63b05ece2c9e323813422d35a`.

## Feasibility amendment before outcomes

The first device preflight exposed an unbounded serialization defect: the
initial scene prompt tokenized to 1,681,642 tokens, above Qwen's 131,072-token
model limit. The process timed out after 902.1 seconds with zero completed
generations, no parsed proposal, no quality metric, no collection, and no
device selection.

Before rerunning, the prompt view was deterministically capped at 24 entities
and 96 relations. Entities are selected by fixed semantic-role priority then
identifier; relations are selected round-robin across relation kinds. An
8,192-token hard limit now fails closed before inference. The model, weights,
four benchmark games, decoding, 2,104-row collection, representative sample,
baselines, shuffles, gates, and firewalls are unchanged. This is a feasibility
amendment, not post-outcome tuning.

The first checkpointed evaluation segment then exposed two implementation
defects. Full quadratic relation graphs were still archived even though only
the bounded view reached Qwen, producing about 2.0 GB of redundant storage
and severe preprocessing latency. Separately, the representative-row digest
included post-action fields despite the protocol's outcome-independent
selection requirement. Seven generations completed before interruption; all
seven were strict-JSON failures. That already makes the 0.99 JSON gate
mathematically impossible, and neither the JSON contract, model, decoding nor
threshold is changed.

The existing 2,104 executed transitions are now projected
outcome-independently to explicit 24-entity/96-relation original and shuffled
views. Sampling and shuffle salts use only game/split, policy seed,
reset/step, pre-action scene signature, legal actions, and the preselected
action/arguments. The seven preliminary outputs are excluded and the full
evaluation restarts from zero. This amendment corrects storage and removes
outcome leakage; it does not attempt to rescue the already-failed JSON gate.

The completed device benchmark remains valid under the final manifest because
the subsequent amendments do not alter its four scene prompts, model weights,
decoding, devices, or speed-selection rule. The final manifest explicitly
binds its predecessor checksum
`03fe976b8a96b15c51dbf93ac527bb363ab5e4145ea398de80b1609bab9c4287`
and benchmark result checksum
`e6874d94708611e870415a6decd32314f15e117b2fdd24dd1c20d9d33a66ecdb`.

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

The CPU and RTX 4050 paths receive identical compact prompts, schema, weights,
temperature 0, sampling disabled, and maximum 256 new tokens. Four initial
source-only scenes (`bp35`, `cd82`, `re86`, `ls20`) are timed. GPU is selected
only when median generation is at least 1.10× faster. No task-quality outcome
is used for device selection. Exact CPU/GPU response equality is reported but
is not a selection gate. Every prompt is capped at 24 entities, 96
relation-kind-stratified relations, and 8,192 input tokens.

## Representative evaluation

Eight rows per game are selected by the lowest SHA-256 values under the frozen
salt `sage12-proposal-pilot-sample-v1` applied to a pre-action-only trace
digest; selection cannot inspect outcomes.
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
