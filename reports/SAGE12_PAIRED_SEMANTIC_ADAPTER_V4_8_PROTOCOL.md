# SAGE12 paired semantic adapter V4.8 protocol

## Question

V4.7 showed that the complete semantic architecture beats its action-sequence
baseline when its slot effects are correct, but frozen Qwen probabilities add
no useful information. V4.8 asks the next narrow question:

> Can a cheap source-only adaptation of Qwen's semantic representation recover
> decision-relevant, cross-game slot effects strongly enough to improve the
> unchanged V4.7 world-model, EBM, and controller chain?

This is deliberately an exploratory end-to-end test. It does not grant live
authority, open the three source-validation games, or use the neural holdout.

## Frozen data

Only the eleven registered SAGE11 `source_train` games are readable:

`bp35, cd82, dc22, g50t, ka59, lf52, lp85, sp80, su15, tr87, tu93`.

The corpus has two components:

1. Same-prestate SAGE11 pairs. Rows are grouped by `(game, state digest)`,
   exact duplicate actions are collapsed, and two distinct legal actions are
   compared. Sampling is deterministic and rare contrasts are retained before
   common pairs.
2. All complete replay-verified V4.3 nodes. These supply the seven V4.7 slot
   labels, including target creation, removal, and movement.

The source capacity audit is descriptive and precedes outcome fitting. It
found 12,044 repeated states. After assigning each pair to its rarest primary
contrast, there are 2,924 progression, 332 non-completion game-over, and only
19 explicit level-completion pairs. All 19 completion pairs are retained.
Together with V4.3, the frozen corpus contains 1,172 progression, 351
game-over, and 22 level-completion contrasts after deterministic sampling.
The smaller-than-planned completion count is reported as a limitation; it is
not filled with cross-state or validation examples.

The frozen corpus, source checksums, caps, and counts are written to
`training/sage12/semantic_adapter_v4_8/frozen_manifest.json` and
`pair_corpus.jsonl` before any embedding or fit.

## Identity-free model view

Each comparison prompt contains two semantic interventions and, for the
context candidate, unordered count buckets. It excludes:

- game, root, node, reset, seed, and state identifiers;
- raw frames, values, object IDs, and shape signatures;
- raw `x/y`, row, and column coordinates;
- labels, utilities, and future outcomes.

Two click arguments are represented only by their translation-invariant
direction and distance relative to each other. Two frozen prompt candidates
are evaluated:

- `minimal`: the two interventions only;
- `invariant_context`: interventions plus unordered state/history summaries.

The candidate with the lowest SAGE11-only leave-one-game-out macro Brier is
selected. Output-identity accuracy and then `minimal` break exact ties. V4.3
outcomes do not select the representation.

## Semantic adapter

The local Qwen2.5 0.5B model is a frozen pair encoder. Both left/right orders
are embedded with unchanged weights. A rank-16 external residual adapter and
eight four-class heads are trained over:

`neither, left, right, both`.

The eight heads are the seven V4.7 effects plus an auxiliary progress head.
SAGE11 rows mask the three unavailable target-effect labels. Training includes
both action orders, and inference averages the original prediction with the
exactly relabelled swapped prediction.

This is not transformer LoRA: it is an external low-rank adaptation of the
frozen Qwen representation. That distinction is part of the result. It avoids
installing unpinned packages and fits on the laptop GPU.

Every published V4.3 semantic prediction is leave-one-game-out. For game
`g`, neither its SAGE11 pairs nor its V4.3 outcomes train the adapter that
annotates `g`.

## End-to-end evaluation

The selected semantic annotations are passed through the unchanged V4.7:

- candidate-complete slot compiler;
- regularized seven-head world model and calibration;
- depth-three eight-feature pairwise EBM;
- receding-horizon first-action controller;
- fold-local baseline selection.

World-model fitting remains nested out-of-game. The future V4.3 tree topology
is still a non-deployable evaluation oracle. No live action is executed.

Comparators are deterministic-left, action-only, action-sequence-only,
structured-only, V4.7 zero-shot Qwen diagnostics, true annotations, and a
relation-shuffled adapted semantic view.

## Exploratory decision rule

V4.8 is positive only if all frozen checks hold:

1. direct V4.3 macro Brier is better than an action-only LOGO baseline;
2. adapted end-to-end mean utility is strictly above structured-only;
3. adapted mean utility is strictly above the fold-selected primary baseline;
4. adapted utility is non-negative versus that baseline on at least 6/11 games;
5. the original representation beats its relation shuffle;
6. at least two available completion opportunities are selected;
7. game classification from the seven semantic outputs is at most 0.60.

The bootstrap interval is reported but is not a conjunctive exploratory
threshold. This is the intentional, documented relaxation: a positive mean
plus cross-game, causal-shuffle, rare-event, and leakage checks is enough to
justify a stronger follow-up, but never live authority.

Failure rejects this frozen-Qwen/external-adapter semantic acquisition route.
It does not by itself refute the global architecture because V4.7's oracle
ladder already established that correct semantics can make the downstream
composition work.

## Reproduction

```powershell
python -m theory.sage12.semantic_adapter_v4_8 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_adapter_v4_8 embed --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_adapter_v4_8 adapt --device cuda:0
python -m theory.sage12.semantic_adapter_v4_8 evaluate
```

The manifest and corpus checkpoint must be published before `embed`.
