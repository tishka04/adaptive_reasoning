# SAGE12 grounded-proposal pilot result

Date: 2026-07-27.

Outcome: **FAIL_CLOSED**. The local Qwen2.5 0.5B proposal path failed all
seven preregistered Stage A gates. No semantic world model was fit, no EBM was
trained, no holdout or historical environment was opened, and SAGE12 remains
off.

The negative result applies to the frozen compact scene-graph prompt, typed
hypothesis contract, and deterministic decoding evaluated here. It does not
show that all higher-semantic approaches are impossible. It does show that
this proposal representation and unconstrained text-generation path are not a
valid foundation for the next training stage.

## Frozen scope

The pilot reused the existing SAGE11 game split without opening a new target:

- 11 source-training games;
- three source-validation games: `ls20`, `re86`, and `sc25`;
- no `NEURO_HOLDOUT_V1`, historical, or `ar25` access;
- manifest checksum
  `0dfdff9a61e45e02b16601a47d987454c991a2d5f99c8964a5486c17ed17aceb`;
- SAGE11 split checksum
  `22de34d38dbd1ce4169fd09694f4449e9ad531f5536dccfb274d2a41de2ce37d`.

The final manifest freezes Qwen2.5 0.5B Instruct, temperature 0, sampling
disabled, 256 maximum new tokens, eight maximum hypotheses, 24 entities, 96
relation-kind-stratified relations, and an 8,192-token hard input cap. Model
weights checksum:
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

## Amendments and audit trail

Two feasibility corrections were published before the clean evaluation run:

1. The first hardware preflight exposed a 1,681,642-token unbounded scene
   prompt and timed out after 902.1 seconds. It produced no completed
   generation and no quality outcome. The prompt view was then capped
   deterministically at 24 entities, 96 relations, and 8,192 input tokens.
2. The first evaluation checkpoint exposed redundant quadratic graph storage
   and a representative-row digest that included outcome fields. Seven
   generations had completed; all seven were already strict-JSON failures.
   They were quarantined, the stored views and sampling digest were corrected
   to be compact and pre-action-only, and all 224 outputs were regenerated
   from zero. The model, weights, decoding, schema, baselines, thresholds, and
   game split were unchanged.

The seven preliminary outputs remain in
`training/sage12/proposal_pilot_v1/preliminary_model_outputs_invalid.jsonl`
for audit only and are excluded from every reported metric.

## CPU/GPU benchmark

The same four compact prompts, weights, schema, and decoding were evaluated on
CPU and the laptop RTX 4050.

| Device | Median inference | Mean inference | Strict JSON | Selection |
|---|---:|---:|---:|---|
| CPU | 26.478 s | 25.706 s | 0/4 | not selected |
| CUDA RTX 4050 | 6.953 s | 6.880 s | 0/4 | selected |

CUDA was 3.808x faster by median wall time, above the frozen 1.10x threshold,
so `cuda:0` was used for the clean evaluation. Exact CPU/GPU response equality
was 25%; this was reported but was not a device-selection or quality gate.
Benchmark result checksum:
`e6874d94708611e870415a6decd32314f15e117b2fdd24dd1c20d9d33a66ecdb`.

## Executed semantic traces

The collector archived exactly 2,104 real source-only transitions:

- 1,624 source-training rows and 480 source-validation rows;
- 160 rows per game except finite-capacity `lp85`, which contributed 24;
- 2,026 productive rows;
- 2,109 raw steps, with five repeats rejected by the frozen cap;
- balanced legal-action-family coverage followed by seeded randomized
  selection inside the least-covered family;
- single-family coverage for `lp85` and `su15`, where no action-family
  diversity was available;
- no outcome fields used by representative sampling or relation shuffling.

The compact shards occupy 138,695,637 bytes. Their combined checksum is
`ce5cfe1217f9add9ab250f60315ed66d154ae8ed903e51bb572b69a4b3`.
The canonical collection-manifest checksum bound by the result is
`69182fef9d397768aace54a301dc5046bd801589524ae14b7d6e8ad728ad0e05`.

## Stage A gates

Eight outcome-blind rows per game produced 112 representative scenes. Each
scene was generated once with its original compact relations and once after
the frozen relation-binding shuffle, for exactly 224 clean model outputs.

| Gate | Frozen requirement | Result | Pass |
|---|---:|---:|:---:|
| Strict JSON validity | >= 0.99 | 0.000 | no |
| Parsed proposals retain `support=0` | >= 0.95 | 0.000 | no |
| Grounded hypothesis rate | >= 0.90 | 0.000 | no |
| Recall@8 gain over stronger baseline | >= +0.10 | -0.895 | no |
| Relation-shuffle degradation | >= 0.10 | 0.000 | no |
| Every validation game non-negative | required | 0/3 games | no |
| Game-identity gain over majority | <= +0.10 | +0.901 | no |

No strict parser output yielded a typed proposal. Consequently the
support-zero, compiler-grounding, and LLM recall metrics are zero by the
preregistered fail-closed convention; the compiler was never given a valid
typed candidate.

## Baselines and per-game transfer

There were 19 productive-mechanism-eligible rows among the 24 representative
source-validation rows.

| Method | Productive-mechanism recall |
|---|---:|
| Qwen2.5 0.5B, recall@8 | 0.000 |
| Deterministic templates | 0.000 |
| Source-train action-only majority | 0.895 |

The action-only baseline was the stronger baseline. Transfer was negative on
every validation game:

| Game | Eligible rows | Qwen recall@8 | Action-only | Gain |
|---|---:|---:|---:|---:|
| `ls20` | 6 | 0.000 | 0.833 | -0.833 |
| `re86` | 8 | 0.000 | 0.875 | -0.875 |
| `sc25` | 5 | 0.000 | 1.000 | -1.000 |

Original and relation-shuffled Qwen recall were both zero, so the required
relation sensitivity was absent.

## Output-format and leakage diagnostics

Post-hoc diagnostics do not alter any gate:

- all 224 responses used a Markdown `json` fence, so strict JSON validity was
  zero;
- removing only one surrounding fence made 96/224 responses syntactically
  valid JSON (42.9%);
- all 96 decoded to a top-level list, and 0/224 conformed to the typed
  hypothesis schema after fence removal;
- exact original/shuffled response equality was 12.5%;
- clean GPU inference median was 7.696 seconds per output.

The format failure is therefore not just cosmetic fencing. Outputs also used
the wrong top-level structure, omitted required typed fields, or confused
actions and effects.

The source-training game-identity probe showed severe representation leakage:

| Fixed feature view | Accuracy | Majority | Gain |
|---|---:|---:|---:|
| Available actions only | 0.590 | 0.099 | +0.491 |
| Entity structure only | 0.935 | 0.099 | +0.837 |
| Relations only | 0.843 | 0.099 | +0.744 |
| Full compact scene signature | 0.999 | 0.099 | +0.901 |

No explicit game ID, raw coordinate, raw color, or grid hash was supplied to
the model. The leakage instead comes from game-specific action availability,
entity inventories, and relation structures. Removing an explicit identifier
was insufficient to create a transferable representation.

Post-hoc diagnostic checksum:
`e5549bcf6f1f5c19371a771cbf95d76b9f4d8d38bdff535362f7e35f1198742d`.

## Decision

Stage A is rejected. `authorized_next_stage` is `none` and
`world_model_fit_started` is `false`. The semantic world model must not be fit
on this representation, the EBM must not be trained, and no SAGE12 authority
mode may be promoted from this result.

A future pilot requires a new version and new preregistration. The most direct
repairs to test cheaply are constrained schema decoding or a native
classification head, plus a scene abstraction explicitly optimized against
the demonstrated action/entity/relation game signatures. Thresholds from this
failed run must not be changed retroactively.

Primary result checksum:
`fbb86c17fee57ff46199dd94594936694bf2b0e63b05ece2c9e323813422d35a`.
Clean model-output checksum:
`79886838b5145c53375ee50ae71db311f3786c63172595f4036e6aa1c625eed0`.

## Software and artifact validation

- focused SAGE12 validation: 25 tests passed;
- full repository regression: 1,692 tests passed with one non-failing Joblib
  physical-core-detection warning;
- targeted Ruff checks passed;
- all 2,104 shard rows, per-shard hashes, compact graph bounds, the combined
  shard checksum, 112 original outputs, 112 shuffled outputs, and seven
  excluded preliminary outputs were reverified before publication.

## Reproduction

From the repository root:

```powershell
python -m theory.sage12.proposal_pilot_runner benchmark
python -m theory.sage12.proposal_pilot_collection --workers 4
python -m theory.sage12.proposal_pilot_runner evaluate
python -m theory.sage12.proposal_pilot_runner diagnose
```

The first three commands are the frozen experiment order. `diagnose` is
post-hoc and writes only the explanatory diagnostic artifact.
