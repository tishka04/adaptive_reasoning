# SAGE.T10.2.2 action-budget acquisition protocol

Status: implementation protocol to be frozen before the first real smoke run.

## Objective

Test whether the T10.2 source-acquisition failures were caused by wall-clock
budgets and repeated whole-history journal scans rather than by the scientific
kernel.  The scientific T10.2/T10.2.1 controller, environment firewall, event
sealing, and posterior updates remain unchanged.

## Isolation and authority

- The partial T10.2.1 journal is diagnostic evidence and is read-only.
- T10.2.2 writes only under
  `training/sage_t/t10_2_2_action_budget_collection`.
- The real smoke writes only under
  `training/sage_t/t10_2_2_action_budget_smoke`.
- Source games are unchanged. Validation, AR25, holdout, and production
  authority remain closed.

## Registered acquisition policy

- Maximum 64 physical actions per reset.
- Wall time is a liveness watchdog only: 600 s/reset plus 30 s hard grace,
  2,700 s/lane, stop-new-actions at 42,600 s, and a 43,200 s absolute bound.
- The full schedule first completes one discovery lane for each source game.
  Confirmation and remaining discovery lanes are then interleaved.
- A confirmation lane is never scheduled before discovery donors from both
  other source games exist.

## Complexity and durability

- Reconstruct journal accounting, completed reports, and discovery evidence
  once at process start/resume.
- Maintain accounting and discovery evidence incrementally thereafter.
- Do not call a whole-history accounting, lane-report, or discovery-event scan
  inside the per-reset hot loop.
- Persist an authenticated compact cursor after reset-level progress.
- Persist the full derived checkpoint at lane boundaries and finalization.
- The append-only per-intent/per-event journal remains the source of truth.
- Resume never replays a physical action and rejects unknown topology,
  checksum drift, duplicate event ids, or an unsealed intent.

## Pre-collection gates

1. Syntax/import/CLI checks pass.
2. Ruff passes on all T10.2.2 files.
3. Focused T10.2.2 and T10.2.1 regression tests pass.
4. The cold/warm fixed-work gate is at most 1.10 for incremental bookkeeping;
   each timing sample batches 20,000 identical idempotent updates to stay above
   sub-millisecond Windows scheduler noise.
5. The T10.2.2 and compatibility manifests are frozen and bind exact code,
   documentation, schedule, artifact roots, and parent lineage.

## Real smoke

The donor-safe smoke uses two resets in each of three lanes: discovery for the
two donor games, followed by confirmation for the held-out game.  The odd
confirmation seed executes `capacity_matched_independent` and then `learned`, so
both posterior preview paths are exercised before the full matrix.  The smoke
is diagnostic only and cannot open validation or authorize a scientific
verdict.  It must demonstrate bounded actions, exact accounting, no duplicate
event ids, durable resume state, donor-safe confirmation, learned-controller
coverage, and no hot-loop history scans.

## Negative-result handling

- A liveness timeout is reported as an acquisition/resource miss.
- Missing donor evidence, accounting failure, checksum drift, replay, or unknown
  topology is a data/provenance failure.
- A failed smoke blocks the full collection. It does not support a claim about
  schema learning or transfer.
- Any code or protocol change after freezing requires a new manifest checksum
  before another real run.

## Publication boundary

The complete collection, downstream compile/cross-fit/canary, result report,
commit, push, and pull request are deliberately outside this implementation
turn. The user launches the complete collection manually after the smoke gate.
