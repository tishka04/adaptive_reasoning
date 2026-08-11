# SAGE.T10.3.12 — Relational mechanism source test

## Status and scope

This iteration is a core-only, fail-closed test of whether the successful
`lp85` and `su15` trajectories contain source-specific relational information.
It is not another sequence discovery run. `re86`, `ls20`, `sc25`, `ar25`, all
holdouts and production authority remain closed.

The protocol must be frozen before offline scoring. Any code, test, protocol or
runbook change after `freeze` requires a new iteration suffix; it cannot repair
the frozen run.

## Hypothesis

A factorised program comprising an operator schema, a causal target role, a
transition relation and a terminal stop condition is equivariant under D4,
candidate-order and palette transformations, rejects ambiguous counterfactuals,
and outperforms or is materially more efficient than source-free generic
rediscovery at matched budgets.

The two tested mechanisms are:

- repeat a uniquely productive target role until level progress, with a safety
  horizon of 8;
- follow a path successor toward the uniquely salient enclosure, with a safety
  horizon of 16.

No transferable payload may contain a game name, seed, coordinates, colors,
raw grids, entity identities, action arguments or argument checksums. Grounded
arguments are reset-local and ephemeral.

## Parent quarantine and source allowlist

T10.3.11 is recorded as `SUPERSEDED_INCOMPLETE_NEGATIVE`: 1,758 intents,
1,756 sealed events, two invalid inflight intents, 24 branch receipts, three
`PermissionError` runtime failures and zero level increments. Its journal is
hashed but never repaired, replayed, fitted or loaded as a registry.

Only the following sources are allowed:

- the abstract canonical witness descriptors from T10.3.8;
- the T10.0b report as an audited source projection, never its grounded paths;
- the frozen SAGE12 v4.3 `lp85` and `su15` source shards for offline projection.

The compiler explicitly rejects `grounded_evidence`.

## Experimental arms

1. `factorized_relational_source`: abstract source program, active support zero.
2. `generic_grammar_source_free`: the same operator grammar enabled a priori,
   with no source descriptor.
3. `schema_swap_wrong_source`: path semantics in repeat contexts and repeat
   semantics in path contexts, with no fallback to the generic arm.
4. `relation_ablation`: hash-offset repeat binding and path orientation without
   enclosure salience.

Every active arm/reset receives a fresh standard program registry. No evidence,
posterior or support is shared across resets, arms or games.

## Offline matrix and gates

The runtime materialises 96 compact recipes, never raw grids:

- 64 positives: two mechanisms × eight D4 transforms × two candidate orders ×
  two palette permutations;
- 32 controls: repeat binding conflict and ambiguous effects; broken path
  bridge and ambiguous path orientation, each under eight D4 transforms.

All gates are required before active collection:

- factorised arm correct on 64/64 positives and 32/32 controls;
- six option-prefix alignment cases invariant at prefix lengths 0, 4 and 17;
- one abstract program hash per mechanism;
- source arm beats generic by at least eight fixtures, or has no lower accuracy
  and at most half its median inspections in each mechanism;
- source arm beats wrong-source and relation-ablation by at least eight fixtures;
- at most 12,288 recorded candidate inspections, 600 seconds and 10 MiB of
  compact artifacts.

An offline miss returns code 3 and forbids `active-core`.

## Active matrix and gates

The active matrix is two core games × labels 3521–3524 × four arms: 32 fresh
resets, Latin-square ordered, at most 16 actions per reset and 512 actions total.
The labels diversify reset-local tie breaking but do not seed the environment;
the report publishes distinct initial-frame hashes.

`active-core` returns code 0 when the bounded collection is complete and its
accounting is sealable, independent of scientific outcome. `adjudicate` alone
emits code 3 for a complete scientific miss.

PASS requires all of the following:

- clean intent/event accounting, no replay, unresolved intent, illegal action,
  controller error or legacy/fallback action;
- factorised arm 8/8, zero `GAME_OVER`, and SAGE.T attribution for every action;
- repeat mechanism 4/4, median at most 6 and maximum 8 actions;
- path mechanism 4/4, successor and orientation attestations at every reset,
  median at most 10 and maximum 16 actions;
- factorised arm not below any baseline in either game;
- at least two more successes than generic, or, only for an 8/8 tie, at most
  75% of generic actions-to-level;
- at least two more successes than wrong-source and relation-ablation.

An equal-success/equal-efficiency generic result is
`GENERIC_REDISCOVERY_ONLY`. One mechanism passing and the other failing is
`PARTIAL_MECHANISM_SUPPORT`. Neither result promotes a program.

The sole positive terminal verdict is
`PASS_T10_3_12_RELATIONAL_MECHANISM_SOURCE`. It authorises only preregistration
of a later T10.3.13 sequence-composition experiment; it does not open sequence
games or production authority automatically.

