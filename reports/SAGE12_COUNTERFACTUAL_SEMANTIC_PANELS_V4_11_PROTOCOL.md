# SAGE12 V4.11 — Counterfactual semantic-panel protocol

Status: **FROZEN BEFORE COLLECTION**.

Manifest checksum:
`ebeaac9905bcea932f791122902da53ed1dabb72e821cb870e04a4ccaa6d9018`.

## Question

V4.10 learned useful same-state action ordering but failed to show that its
neighbor relations improved cross-game absolute semantics. V4.11 tests a
narrower causal route:

> Can action-aligned relations predict which of several actions executed from
> the exact same state has the better semantic consequence, and can that
> relative signal be distilled onto the stronger root-only absolute predictor
> without adding game signature?

The comparative and absolute-distillation decisions are separate. A
comparative success cannot bypass a failed absolute export by feeding an EBM
directly or by changing the world model to ordinal state.

## Source boundary

Only the eleven frozen SAGE11 source-train games are authorized:

`bp35`, `cd82`, `dc22`, `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
`tr87`, and `tu93`.

The V4.10 manifest, teacher corpus, same-prestate pairs, and published result
are fingerprinted in the V4.11 manifest. Source validation, holdout, external
historical data, and live environments remain closed.

## Counterfactual panels

The collector targets 96 unique pre-states per game and requires at least 80.
It may use at most 60 resets and 128 base actions per reset, under fixed seeds
`6011`, `6029`, `6053`, `6079`, and `6101`.

Each panel contains two to four distinct legal actions:

- the reset-to-prefix path is replayed before every panel;
- the restored pre-state checksum must exactly match;
- every arm is executed from a deep copy of that verified state;
- arm selection greedily maximizes the diversity of student-visible,
  object-relative signatures with a stable hash tie-break;
- selection sees no post-action label or return;
- pre-state/action repeats already present in V4.3–V4.10 are rejected.

The immediate transition supplies the existing 17 semantic effects. Each
first action also receives two deterministic continuation rolls to total
horizon three. Its progress return is

`score(t0) + 0.8 × score(t1) + 0.8² × score(t2)`,

averaged over the two continuations. Pair differences below `0.25` are ties.

The teacher must retain at least 20 progress-discordant panels in at least
eight games. An immediate effect enters the absolute macro-score only with at
least 100 discordant comparisons spanning at least four games. Completion is
eligible for a decision only with at least 20 positive arms in four games.

## Student and distillation

A shared 32-wide token embedding and 96-wide DeepSets encoder produces 17
effect logits plus one progress score. Both full and root-only models use 30
epochs, equal sampling from every training game, and the same absolute,
effect-pair, progress-pair, and tie-consistency objectives.

For each candidate panel:

1. root-only logits provide the absolute anchor;
2. full-minus-root relation residuals are centered to zero across the panel;
3. an effect-specific `alpha` in `{0, .25, .5, .75, 1}` is selected using
   held-game calibration inside the outer training games;
4. final probabilities are
   `sigmoid(root_logit + alpha × centered_residual + shared_shift)`.

Pair preference is exactly
`sigmoid(progress_score(left) - progress_score(right))`, so swapping arms must
return the complement. A one-action panel necessarily falls back to root-only.
The existing `SlotAnnotation` interface is unchanged; the V4.11 adapter exports
only its seven frozen effects with `support=0`.

## Strict LOGO evaluation

Every primary prediction is leave-one-source-game-out. Training may use the
published V4.10 rows and pairs from the ten allowed games, plus their fresh
panels. The primary test set is only the fresh panels from the held-out game.

Controls are:

- action-only progress/effect priors;
- root-only Siamese student;
- relation-shuffled full student;
- neighbor-order reversal;
- exact arm swap.

Confidence intervals use 10,000 bootstrap draws, with panels resampled inside
each game and games receiving equal weight.

### Comparative decision

All of the following must pass:

- teacher capacity and firewall;
- pair log-loss gain over root-only with 95% lower bound above zero;
- pair log-loss gain over action-only with 95% lower bound above zero;
- top-1 regret reduction over root-only with 95% lower bound above zero;
- relation shuffle degradation with 95% lower bound above zero;
- non-negative pair transfer in at least 6/11 games;
- arm-swap complement error at most `1e-6`.

### Absolute-distillation decision

All of the following must also pass:

- eligible-effect macro-Brier gain over root-only with 95% lower bound above
  zero;
- macro-ECE no worse than root-only;
- relation shuffle worsens macro-Brier;
- non-negative absolute transfer in at least 6/11 games;
- the 95% upper bound on identity accuracy added over root-only is at most
  `0.02`;
- neighbor permutation changes no probability by more than `1e-6`;
- completion recall@8 is at least `0.20` only when terminal capacity is
  eligible.

Possible verdicts are:

- `COMPARATIVE_CAUSAL_TEACHER_CAPACITY_FAILED`;
- `COMPARATIVE_CAUSAL_SEMANTICS_NOT_SUPPORTED`;
- `COMPARATIVE_SUPPORTED_ABSOLUTE_DISTILLATION_FAILED`;
- `READY_FOR_SOURCE_WORLD_MODEL_PILOT`.

Even the positive verdict authorizes only a later source-only world-model
pilot. V4.11 cannot fit a world model or EBM, open protected data, or grant
controller/live authority.

## Commands and artifacts

```powershell
python -m theory.sage12.counterfactual_semantic_panels_v4_11 freeze
python -m theory.sage12.counterfactual_panel_collection_v4_11
python -m theory.sage12.counterfactual_semantic_panels_v4_11 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.counterfactual_semantic_panels_v4_11 evaluate --device cuda:0
```

The output directory is
`training/sage12/counterfactual_semantics_v4_11/`. Collection, teacher QA,
predictions, slot annotations, checksums, and the final result must be
published whether the iteration passes or fails.
