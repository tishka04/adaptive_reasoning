# SAGE12 V4.17 — sequence-conditioned transformation policy

Status at freeze: **protocol specified; no V4.16 or V4.17 result observed**.

## Question

V4.15 learned a useful causal sequence prior but not relation-grounded
milestones. V4.16 represents observed changes as transferable
morpho-topological transformation latents but has no behavioral scaffold.
V4.17 tests their composition:

> Can a human sequence prior propose plausible actions while a causal
> transformation model and the temporal EBM select proposals that advance
> transferable, object-relative changes?

Every registered condition runs. Neither a failed V4.16 component gate nor a
failed offline V4.17 gate may skip the bounded active comparison.

## Frozen boundary

The data split is unchanged:

- human training: `ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`;
- untraced offline transfer: `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
  `tr87`, `tu93`;
- bounded active validation: `re86`, `ls20`, `sc25`;
- final confirmation: `NEURO_HOLDOUT_V1`, kept closed.

V4.17 fingerprints:

- the V4.15 manifest, teacher QA, semantic result, checkpoint metadata,
  checkpoint, offline decisions, active runs and integrated result;
- the V4.14 temporal checkpoint, EBM checkpoint and transfer predictions;
- the V4.11 panel manifest and source shards used by V4.16;
- all six human-trace files;
- the complete V4.16 implementation and this V4.17 implementation.

V4.16 is fitted only after this protocol and manifest are frozen. V4.17 does
not fit a mixing coefficient on transfer or active games.

## Composition

For every legal candidate:

1. V4.15 encodes the object-relative candidate plus the causal executed
   history and produces its learned-milestone policy score.
2. V4.16 compiles the same current state/action into a palette-free
   morpho-topological graph, predicts a transformation latent and uncertainty,
   retrieves up to eight eligible multi-game prototypes, and estimates
   productive minus risky transformation value.
3. V4.14 predicts the registered depth-three semantic rollout and the learned
   EBM assigns its trajectory energy.

Scores are standardized only within the current candidate set:

```text
hybrid = z(V4.15 learned policy)
       + 0.5 × z(V4.16 transformation value)
       - 0.5 × z(V4.14 temporal energy)
```

An unknown V4.16 query receives only its frozen negative-uncertainty score.
Ties use the existing content-addressed action signature. No future frame,
teacher transition latent, panel outcome, game identity or absolute
coordinate enters the deployed score.

The V4.15 recurrent belief and V4.14 temporal belief advance only after the
executed transition. V4.16 prototype evidence is not updated during the
registered validation, preventing controller-order leakage across seeds.

## Validation ladder

### V4.16 component audit

Run its frozen compiler, CUDA training, HDBSCAN selection and transfer
evaluation. Report all V4.16 gates, but continue regardless of their result.

### Offline transfer

Evaluate all 768 V4.15/V4.11 panels and 2,831 arms with:

- V4.15 learned milestone policy;
- V4.15 policy plus V4.14 temporal EBM;
- V4.16 transformation-only;
- V4.15 plus V4.16;
- the full V4.17 hybrid;
- V4.17 with V4.16 relations removed;
- V4.17 with transformation scores permuted within panel;
- oracle observed transformation latent plus V4.15 and temporal EBM;
- true-world effects plus the learned EBM;
- exact oracle.

Future outcomes remain scoring-only. The observed transformation latent is an
oracle diagnostic and never enters a deployable lane.

`HYBRID_OFFLINE_SUPPORTED` requires all of:

- the full hybrid's paired 95% utility-gain lower bound is positive versus
  V4.15 policy plus EBM;
- its lower bound is also positive versus transformation-only;
- nonnegative per-game gain over V4.15 policy plus EBM on at least 5/8 games;
- relation removal has a positive paired 95% degradation lower bound;
- at least one completion and at least 50% of exact-oracle completion panels;
- every registered condition executed and future outcomes remained
  scoring-only.

### Active validation

Reuse by checksum the nine V4.15 action-sequence runs, nine V4.15 behavior
runs and nine V4.15 milestone-plus-EBM runs. Execute nine fresh V4.17 hybrid
runs on the same games, seeds, 1,000-action budget and 14-reset cap.

Report actions, levels, WINs, GAME_OVERs, illegal proposals, latency,
prototype coverage and paired deltas. `HYBRID_ACTIVE_PROGRESS` requires at
least one new level and no increase in illegal proposals. Active results are
descriptive and cannot rescue a failed offline causal gate.

## Decision

- `SEQUENCE_TRANSFORMATION_SUPPORTED` requires both the complete offline gate
  and non-zero active progress.
- `LIVE_PROGRESS_WITH_CAUSAL_GATES_FAILED` records progress without granting
  semantic authority.
- `TRANSFORMATION_COMPONENT_BOTTLENECK` applies when V4.16 itself is
  unsupported and V4.17 makes no live progress.
- `SEQUENCE_TRANSFORMATION_BOTTLENECK` applies otherwise.

No V4.17 outcome opens the final holdout or promotes controller authority.

## Commands

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 prepare --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 evaluate --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 active --device cuda:0
```

Artifacts are written under
`training/sage12/sequence_transformation_policy_v4_17`. V4.16 component
artifacts remain under `training/sage12/morpho_topological_v4_16`.
