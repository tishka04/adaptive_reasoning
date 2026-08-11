# SAGE T10.3.12f - causal identification and control procedure

Status before `freeze`: implemented, not frozen, and no T10.3.12f physical
action has been authorized.

## Scientific question

T10.3.12f does not transfer the effect that completed a level in `lp85` or
`su15`.  It tests a transfer-safe procedure which identifies target-local
interventions, predicts and verifies their effects, controls a verified model,
and revises that model after a mismatch.  A level increment is the only
control-success signal.

Two claims are adjudicated separately:

1. a generic closed-loop causal procedure can produce terminal progress on
   several target games;
2. a procedural prior compiled from the existing signed `lp85` and `su15`
   interventions improves that same procedure.

The nine target games were already observed in T10.3.12c-e.  Consequently,
this experiment is a falsification and mechanism diagnostic, not prospective
evidence.  A positive verdict can only nominate one frozen candidate for a
separate T10.3.13 confirmation.

## Source evidence and label quality

The only fitting inputs are the frozen v4.3 source shards for `lp85` and
`su15`, plus the checksum-bound signed T10.3.8 journal and reports.  The
shortest successful SAGE-T chain in each source journal is used only to link
an abstract local control family to terminal support; no grounded path leaves
the offline compiler.  No new source action is permitted.

Effects are reconstructed from the before/after frames.  Stored historical
effect labels are not trusted.  Relation changes, motion, and transformation
are compiled exclusively through confident one-to-one persistent-object
correspondences.  Free space and created or removed objects do not contribute
relation deltas.  Ambiguous correspondences and contradictory added/removed
relation labels are rejected.  Because source shards do not contain a
post-action legal-action set, action-space stability is treated as unobserved,
not inferred.

The source QA stops before fitting when either source has fewer than two
admissible effect modes, any label covers at least 95 percent of admissible
transitions, provenance diverges, or a transfer-safe projection cannot be
constructed.

The compiled prior contains only weights over causal control families and
experiment-scoring terms.  It must contain no game identifier, action name,
concrete argument, coordinate, raw color, entity identifier, frame hash,
grounded path, or source effect signature.  Each source contributes exactly
one half of the prior.

## Frozen procedure and arms

Every arm uses the same target-local learner and the same phase machine:

`IDENTIFY -> VERIFY -> CONTROL -> REVISE | ABSTAIN`.

The frozen causal model library is `stable_repeat`,
`relational_successor`, `state_conditioned_switch`, and `null_or_unsafe`.
The controller scores no more than 16 legal candidates per decision, keeps no
more than 8 active hypotheses, requires posterior 0.80 with margin 0.20 before
verification, demotes predictions below probability 0.10, revises after four
stagnant or repeated-state transitions, permits two revisions, and limits an
option to 16 actions.  Grounding is reacquired from the current legal action
set after every observation.

The four arms are:

- `source_closed_loop`: source procedural prior and normal revision;
- `uniform_closed_loop`: uniform prior and normal revision;
- `permuted_source_closed_loop`: deterministic permutation of the source
  weights with identical norm and entropy;
- `source_open_loop`: source prior, but the first verified hypothesis is
  locked and mismatch evidence cannot select a replacement.

No arm may use a legacy fallback or share a posterior with another work item.

## Historical matrix and budgets

The diagnostic panel is `bp35`, `cd82`, `dc22`, `g50t`, `ka59`, `lf52`,
`sp80`, `tr87`, and `tu93`.  Each game is evaluated under all four arms and
four deterministic work scopes.  Work-scope labels affect tie-breaking only;
they are not environment seeds.  Arm order follows a complete Latin rotation.

The matrix contains 144 resets.  Each reset has at most 48 physical actions
and 180 seconds.  The global bounds are 6,912 actions, 28,800 seconds, and 128
MiB of compact artifacts.  Raw frames and grounded arguments are never
persisted by T10.3.12f.

Each action must have a durable intent before the environment call and an
immediately sealed event afterwards.  Physical replay is forbidden.  An
interrupted or unattributable environment call remains unresolved and makes
the run invalid rather than being retried.

Sequence games, source validation, `ar25`, protected holdouts, production
authority, program promotion, automatic retuning, and T10.3.12c-e events as
training data remain closed.

## Endpoints and verdicts

For action budget `B=48`, reset utility is `(B + 1 - t) / B` when the first
level is sealed at action `t`, and zero otherwise.  The four work scopes are
averaged within game before any statistical contrast.  The nine games, not
the 144 resets, are the experimental units.

Source-informed candidacy requires success on at least two games and positive
paired utility contrasts over the uniform, permuted, and open-loop arms under
exact sign-flip tests with Holm family-wise alpha 0.05.  Generic candidacy
requires the uniform closed loop to succeed on at least two games and beat
the open-loop control without a supported source-prior advantage.

Prediction log loss, verification latency, mismatch, revision, context
coverage, noops, and frame changes are secondary diagnostics.  They cannot
replace terminal progress.

`CAUSAL_IDENTIFICATION_WITHOUT_CONTROL` is emitted only when a predeclared
candidate enters control after two verified contexts in all four scopes on at
least two games, and has both lower prequential log loss and fewer
interventions before verification than every relevant control on at least
five of nine games.  It remains a negative diagnostic, never a PASS.

Negative verdicts include `CAUSAL_LABEL_QA_MISS`,
`PROCEDURE_NOT_SOURCE_IDENTIFIABLE`, `CAUSAL_PROCEDURE_NO_TARGET_PROGRESS`,
`CAUSAL_IDENTIFICATION_WITHOUT_CONTROL`, `SINGLE_GAME_EFFECT_ONLY`, and
`SOURCE_PRIOR_NOT_SPECIFIC`.

Positive verdicts are prefixed
`PASS_T10_3_12F_HISTORICAL_` and authorize only a separately approved and
frozen T10.3.13 confirmation.  They do not prove generalization, promote a
program, or open any authority boundary.

Exit code 0 means a phase or complete collection succeeded, 2 means invalid
integrity/provenance, and 3 means a complete scientific gate miss.  The active
phase reports collection integrity with code 0 even when no level was won;
`adjudicate` owns the scientific verdict.
