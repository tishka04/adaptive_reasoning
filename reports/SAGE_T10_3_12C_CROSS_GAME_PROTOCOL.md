# SAGE T10.3.12c — Cross-game factor falsification

Status before `freeze`: implemented, not frozen, no T10.3.12c physical action.

## Question and claim boundary

T10.3.12b identified four transfer-safe candidates in the two parent mechanisms:
operator, role binding, transition, and termination. T10.3.12c asks whether the
exact frozen factors produce level progress on genuinely different games, beyond
what a bounded source-free grammar can obtain.

A PASS is cross-game evidence for the factors named in the adjudication report.
It is not sequence composition, holdout validation, production authority, or
program promotion. A PASS authorizes only a separately frozen independent
cross-game reproduction.

## Outcome-independent target set

The target set is every remaining game in the frozen v4.3 `source_train` split:

`bp35`, `cd82`, `dc22`, `g50t`, `ka59`, `lf52`, `sp80`, `tr87`, `tu93`.

`lp85` and `su15` remain parent sources, not target scores. `re86`, `ls20`,
`sc25`, `ar25`, source-validation games, and holdouts remain closed. All nine
targets are included; no game is selected after observing a T10.3.12c outcome.

The target audit reads only action names and argument arities from frozen
source-train shards. It does not read frames, effects, rewards, terminal labels,
or success outcomes, and it does not compile target-specific policy data.

## Frozen arms

Each target receives one fresh reset per arm, in a game-rotated order:

1. `factorized_source`: exact T10.3.12b factor bundle, support zero.
2. `generic_source_free`: bounded legal schema/binding enumeration with no source descriptor.
3. `operator_ablation`: only `parameterized_apply` is replaced by `unparameterized_apply`.
4. `role_binding_ablation`: only the relational role is replaced by a local lexical binding.
5. `transition_ablation`: only the source transition is broken.
6. `termination_ablation`: only the stop rule is replaced by a fixed two-step stop.

There are 54 resets. Each has at most 16 actions and 180 seconds. The global
maximum is 864 physical actions and 10,800 seconds. There are no pseudo-seed
replicates: the work labels do not seed the environment. Initial frame hashes
are published.

## Grounding and abstention

The source operator acts only when a parameterized legal action can be grounded.
For a path context, the current frame is recompiled and the successor toward a
salient end is re-grounded. For a repeat context, a unique D4-equivariant
relative boundary role is re-grounded. Ambiguity, a repeated state, an
incompatible action schema, or exhaustion causes abstention.

Abstention is a complete zero-action scientific result. It is never replaced by
a legacy decision. All executed actions must have source
`sage_t_cross_game_factor`. Grounded arguments and raw frames are not persisted.

## Gates and verdicts

Collection integrity requires all 54 signed receipts, intent-before-action,
immediate event sealing, no replay, no unresolved intent, no illegal action, and
zero legacy fallback.

The factorized source must:

- be applicable on at least three target games;
- achieve progress on at least three target games;
- succeed on at least two thirds of applicable games;
- beat the generic arm in success count, or tie on the identical success set
  using at most 75% of its actions to success;
- show at least one factor with two paired full-over-ablation wins and zero
  reverse wins;
- incur no game over, controller error, or illegal action.

Every factor is adjudicated separately and reported by name. Principal negative
verdicts are:

- `SOURCE_OPERATOR_COVERAGE_MISS`
- `CROSS_GAME_TRANSFER_MISS`
- `GENERIC_SEARCH_EXPLAINS_TRANSFER`
- `FACTOR_CAUSALITY_MISS`
- `CROSS_GAME_SAFETY_OR_INTEGRITY_MISS`

The positive verdict is
`PASS_T10_3_12C_CROSS_GAME_FACTORS_IDENTIFIED`.

Code 0 means the requested phase completed. Code 2 means provenance or
accounting invalidity. Code 3 means a complete negative scientific gate. No
post-freeze repair or automatic retuning is allowed; changed code requires a new
protocol suffix.

