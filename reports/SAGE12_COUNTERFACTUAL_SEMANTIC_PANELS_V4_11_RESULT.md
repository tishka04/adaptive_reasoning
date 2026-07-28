# SAGE12 V4.11 — Counterfactual semantic-panel result

Verdict: **COMPARATIVE_CAUSAL_TEACHER_CAPACITY_FAILED**.

Result checksum:
`6399b8d95c5b389be621460d4a5187b8b5292e400d5c90e62eb9557b31932474`.

## Outcome

V4.11 successfully built the requested causal data path:

- 1,056 identical-prestate panels across all eleven source games;
- 3,914 distinct executed first actions;
- 15,124 continuation transitions;
- deterministic 17-effect and horizon-three teacher compilation;
- a tested Siamese comparator and root-anchored absolute distillation
  implementation.

The model stage did not run because the teacher failed its frozen pre-model
capacity gate by one game. The requirement was 20 progress-discordant panels
in at least eight games; the corpus reached that floor in seven. `cd82`,
`dc22`, `lp85`, and `tr87` supplied only 4, 3, 3, and 7 such panels.

## Why stopping is necessary

The corpus has 517 progress-discordant panels overall, but most are
concentrated in `bp35`, `g50t`, `ka59`, `lf52`, `sp80`, `su15`, and `tu93`.
For the other games, the current teacher assigns nearly the same
horizon-three return to the legal alternatives.

Proceeding would make several LOGO folds structurally uninformative:

- a model could appear accurate by predicting ties;
- relation-shuffle and root-only comparisons would have little causal
  headroom;
- average metrics would mostly reflect the seven high-variance games;
- no result could establish transfer across the full source split.

The threshold was published before collection and was not changed after
observing this distribution.

## What this result does and does not say

V4.11 does **not** show that the Siamese comparator fails: it was not fitted.
It also does not refute the proposed
hypotheses → world model → EBM → hierarchical-controller architecture.

It rejects this specific semantic-teacher target as ready for that test:

`one-step hand-compiled productive score + two deterministic continuations +
horizon three`.

The immediate teacher remains valuable. Eight semantic effects have adequate
pair discordance across at least four games, and every graph/firewall check
passed. The missing ingredient is a progress target that varies across more
mechanics, not simply more rows from the same continuation policy.

## Consequence

- No GPU training was launched because the gate precedes model fitting.
- No absolute `SlotAnnotation` export was generated.
- No world model or EBM was fitted.
- No protected or live data was opened.
- No downstream authority was granted.

A scientifically useful follow-up should keep these panels and replace only
the progress teacher: use several policy-diverse continuation families or a
longer, budget-matched value target, then require the same eight-game capacity
before fitting the already implemented comparator.

## Published artifacts and validation

- `teacher_panels.jsonl`: 1,056 compiled audit panels, SHA-256
  `0642073627de0f61a52c3adc55f984210e9304af56bf1bd1606e69ea2b74adf8`;
- `teacher_qa.json`: capacity, effect eligibility, terminal capacity, firewall,
  and collection linkage;
- `student_result.json`: explicit pre-model capacity verdict and downstream
  non-authority flags.

Ruff passed. Artifact cardinalities and checksums matched, and all 168 SAGE12
tests passed under the bundled Python 3.12 environment.
