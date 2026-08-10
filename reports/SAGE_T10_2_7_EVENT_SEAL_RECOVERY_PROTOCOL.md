# SAGE.T10.2.7 — Event-seal recovery protocol

## Scope

T10.2.7 supersedes only the failed replacement-lane execution introduced by
T10.2.6. It does not edit, resume, complete, or replay the T10.2.6 journal. The
parent T10.2.2 collection, the T10.2.4 donor caches, the T10.2.5 zero-action
failure, and the partial T10.2.6 lane remain immutable inputs.

The T10.2.6 failure occurred after one action intent was persisted and before
the corresponding physical event could be sealed. The missing
`environment_sha256` field was caused by passing the recovery protocol manifest
where the frozen scientific execution manifest was required. Because the
environment call crossed the durable intent boundary, T10.2.7 conservatively
classifies one physical action as potentially executed and unattestable. The
whole T10.2.6 lane is quarantined and is never eligible for model fitting.

## Frozen migration

The migration receipt binds:

- the exact T10.2.6 protocol manifest and migration receipt;
- the exact partial journal metadata, sole action intent, and two-file journal
  topology;
- the absence of sealed events, unresolved receipts, reset/lane reports,
  checkpoint, recovery report, accepted ledger, audit, and collection report;
- the parent terminal checkpoint and the T10.2.5 zero-action failure;
- three fresh deterministic odd recovery seeds, disjoint from the scientific,
  T10.2.5, and T10.2.6 seed registries.

The stale T10.2.6 coordination lock is not treated as scientific evidence. No
T10.2.6 file is changed during migration or collection.

## Hybrid execution-manifest contract

The recovery worker receives a deterministic hybrid manifest. Every scientific
field is copied byte-for-byte from the frozen T10.2.2 kernel manifest. Only two
keys are overlaid:

1. `manifest_checksum`, set to the frozen T10.2.7 protocol identity;
2. `migration_receipt`, set to the frozen T10.2.7 migration receipt.

The contract binds the full kernel payload, the inherited payload projection,
the frozen `environment_sha256`, and the exact overlay-key allowlist. Any other
scientific-field change is a manifest-drift error.

Before collection authority is granted, the runtime exercises the production
parent journal handler in a temporary durable journal. It persists a synthetic
intent and seals a synthetic physical event. The resulting event must carry the
T10.2.7 manifest checksum and the frozen T10.2.2 environment checksum. This
preflight performs no real environment action and contributes no model data.

## Durable failure boundary

Any parent-side exception returned by the reset runner is converted into:

- an `environment_call_unattestable` receipt for every unsealed intent;
- a terminal `UNATTESTABLE` reset report;
- a terminal non-complete lane report;
- fail-closed recovery accounting and a non-passing collection exit code.

The runner cannot leave a new open `intent = 1, sealed = 0, unresolved = 0`
boundary as T10.2.6 did.

## Accounting and acceptance

The predecessor T10.2.6 intent remains open inside its immutable journal. The
final aggregate report therefore uses the explicit quarantine equation:

`attempted intents = sealed events + unresolved receipts + quarantined
predecessor intents`.

An accepted source collection requires exactly 18 logical lanes and 72 complete
resets after replacing the original orphan lane with one complete T10.2.7
physical lane. Events from the original orphan lane, all T10.2.5 attempts, the
partial T10.2.6 lane, and every failed T10.2.7 attempt are excluded.

Source validation, AR25, holdout access, and production authority remain closed.

## Negative-result rule

If all three fresh recovery lanes fail, the immutable recovery report is
`FAIL_T10_2_7_RECOVERY`, the command exits nonzero, and no accepted event ledger
or collection-complete report is authorized. A new append-only migration would
be required; this protocol must not be edited or resumed across an unsafe
physical-reset boundary.
