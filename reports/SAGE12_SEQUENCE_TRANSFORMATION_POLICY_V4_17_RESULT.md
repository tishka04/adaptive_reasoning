# SAGE12 V4.17 — sequence-conditioned transformation policy result

## Verdict

**`TRANSFORMATION_COMPONENT_BOTTLENECK`**

V4.17 executed the complete registered composition:

```text
V4.15 causal sequence prior
  + V4.16 predicted transformation value
  + V4.14 temporal world model and EBM
  → receding-horizon controller
```

The result is negative overall, but it contains one concrete advance. The
hybrid significantly improves the V4.15 policy-plus-EBM offline, and removing
topological relations significantly removes that gain. This is the first
global SAGE12 result where learned relations help rather than behave like a
vacuous feature set.

The advance is not yet sufficient. The weak two-prototype V4.16 component is
better than the hybrid by itself, completion capture remains 1/8, and the
active controller still wins no level.

No final-confirmation game was opened and no controller authority was
promoted.

## Frozen composition

No coefficient was fitted on transfer or active games. Every candidate used:

```text
z(V4.15 learned-milestone policy)
  + 0.5 × z(V4.16 causal transformation value)
  - 0.5 × z(V4.14 temporal trajectory energy)
```

All standardization was within the current candidate set. V4.15 and V4.14
maintained separate recurrent beliefs. V4.16 prototypes were frozen throughout
validation, so earlier games or seeds could not update later decisions.

The V4.17 manifest was frozen before the V4.16 model was fitted. Its checksum
is `e5cf3ccfd802231e5ce7c9569fc942d244cb0466ef15d37fb7031541f88ef63a`.

## V4.16 prerequisite

V4.16 returned `SAGE_MT_NOT_YET_SUPPORTED`:

- 5,344 human transitions and 2,831 transfer arms;
- two eligible prototypes;
- 20.40% eligible training coverage;
- bootstrap ARI 0.618;
- zero cross-game recall@8;
- no recall degradation when relations were removed;
- 1/8 completion capture.

V4.17 continued unconditionally, as registered.

## Offline transfer

All 768 panels and 2,831 arms were evaluated. Future outcomes were used only
to score selected arms.

| Condition | Mean utility | Mean regret | Oracle-action accuracy | Completion |
|---|---:|---:|---:|---:|
| V4.15 learned policy | 0.39318 | 0.51112 | 0.27344 | 2/8 |
| V4.15 policy + temporal EBM | 0.37462 | 0.52967 | 0.27344 | 1/8 |
| V4.16 transformation-only | **0.44876** | **0.45553** | 0.23307 | 1/8 |
| sequence + transformation | 0.36568 | 0.53862 | 0.24349 | 2/8 |
| **V4.17 full hybrid** | **0.42770** | **0.47660** | 0.27214 | 1/8 |
| hybrid without relations | 0.37357 | 0.53073 | 0.29948 | 1/8 |
| hybrid with permuted MT scores | 0.42467 | 0.47962 | 0.31641 | 1/8 |
| observed-transform oracle hybrid | 0.36753 | 0.53677 | 0.27214 | 1/8 |
| true world + learned EBM | 0.50758 | 0.39672 | 0.71875 | **8/8** |
| exact oracle | 0.90430 | 0.00000 | 1.00000 | **8/8** |

The full hybrid improves V4.15 plus EBM by **+0.05307**, with a paired 95%
interval of **[+0.02515, +0.08560]**.

Removing V4.16 relations degrades the hybrid by **0.05413**, interval
**[+0.02903, +0.08193]**. This passes the registered relational control.

However:

- the hybrid is `−0.02107` below transformation-only, interval
  `[-0.07098, +0.02799]`;
- only 4/8 games are nonnegative against V4.15 plus EBM, below the required
  5/8;
- it captures 1/8 completion opportunities, below the required 4/8;
- permuting transformation scores changes utility by only +0.00302 with an
  interval spanning zero;
- supplying the observed teacher transformation latent makes the hybrid
  0.06017 worse, because the two coarse prototypes do not turn a better latent
  into a better candidate value.

The apparent tension between useful relations and harmless score permutation
has a simple explanation: relations change prediction calibration and the set
of unknown queries, but the prototype vocabulary is too small for individual
prototype identities to encode specific subgoals.

Across offline arms, 80.0% retrieve at least one of the two prototypes, with
mean uncertainty 0.585. This high assignment rate must not be confused with
semantic coverage: the two prototypes cover only 20.4% of training
transitions as stable eligible clusters.

## Active validation

V4.17 reused all 27 content-addressed V4.15 runs and executed nine fresh
hybrid runs on `re86`, `ls20` and `sc25`, seeds 0–2.

| Active result | V4.15 milestone + EBM | V4.17 hybrid |
|---|---:|---:|
| actions | 8,184 | 8,184 |
| levels | 0 | 0 |
| WINs | 0 | 0 |
| GAME_OVERs | 93 | 93 |
| illegal proposals | 0 | 0 |
| mean decision latency | 0.07737 s | **0.24788 s** |

V4.17 is 3.20× slower at decision time. Prototype assignment was 100% on
`re86` and `sc25`, and 80.1% on `ls20`, but there was no progress.

The aggregate result is identical to V4.15 plus EBM, although the trajectories
are not always identical:

- `ls20`: all 3,000 compared actions and states match V4.15;
- `re86`: 1,560/3,000 actions match and the states diverge rapidly;
- `sc25`: 1,092/2,184 actions match and the states diverge rapidly.

Thus V4.16 does alter decisions on two games, but the alternatives lead to the
same terminal failure profile rather than useful subgoal progress.

The fresh panel took 1,964.10 seconds. All 36 registered run rows are present,
with zero execution error or illegal proposal.

## Gate audit

| V4.17 offline gate | Result |
|---|---|
| hybrid beats V4.15+EBM with positive CI | pass |
| hybrid beats transformation-only with positive CI | fail |
| nonnegative on at least 5/8 games | fail: 4/8 |
| removing relations causes significant degradation | pass |
| at least 4/8 completion opportunities | fail: 1/8 |
| all conditions executed | pass |
| future outcomes scoring-only | pass |

The active progress gate also fails: zero levels and zero WINs.

## Interpretation

Combining V4.15 and V4.16 is better than simply extending the V4.15 effect
classifier. The new representation finally produces a measurable relational
contribution to global offline utility.

The remaining bottleneck is more specific:

1. The transformation vocabulary is too fragmented before clustering and too
   coarse afterward: 3,891 training signatures collapse to two eligible
   prototypes.
2. Prototype value is a broad productive-minus-risk average. It does not say
   which transformation is needed by the current hierarchical subgoal.
3. V4.15 proposes actions one step at a time. The transformation scorer ranks
   immediate changes rather than options that establish preconditions and
   realize a multi-step transformation.
4. The temporal EBM still receives the weak V4.14 rollout, so correct
   transformation intent is not propagated through predicted future states.

The next useful experiment should learn a compact compositional
transformation grammar—primitive edits plus preconditions—and train
multi-step options to realize a requested transformation. V4.15 can remain
the proposal prior, while the subgoal supplies the requested transformation
instead of using unconditional prototype productivity.

## Reproducibility

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 prepare --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 evaluate --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.sequence_transformation_policy_v4_17 active --device cuda:0
```

Checksums:

- V4.17 manifest:
  `e5cf3ccfd802231e5ce7c9569fc942d244cb0466ef15d37fb7031541f88ef63a`;
- V4.16 result:
  `f11823db5f4bf3f0afeea397d2f82303d16172db6fdeebb9735ec4634556fda4`;
- V4.17 preparation:
  `53088166c39351c6603211b83eb5a7d8583be01c42ff739c6172a805a2eb45ee`;
- V4.17 active:
  `6325ef41cc23f5a164080fdfa128dccfc73d15de7ce0415e0bb45f8e1bfe0ef9`;
- V4.17 integrated result:
  `935f8c1c6d0b57f6aef1ffdb6ad735a0f414665d08d1f5067fc9c1b07a6c6faa`.
