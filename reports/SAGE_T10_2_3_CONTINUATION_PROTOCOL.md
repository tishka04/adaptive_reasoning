# SAGE.T10.2.3 — corrected continuation protocol

Status: **frozen before continuation**.

## Registered problem

The partial SAGE.T10.2.2 journal is internally consistent and stopped after
nine complete lanes.  Its next lane is the `lp85` confirmation lane.  Before
opening the environment or emitting an intent, every confirmation reset fits a
`GaugeProgramPosterior` over the same ordered donor evidence.  The donor bank
grows from 16 candidates on the completed `bp9` fold to the registered cap of
256 candidates on the `lp85` fold.  Rebuilding that state inside each
600-second reset liveness window caused the worker process tree to be killed
before its first parent-visible message.  A retry therefore performed the same
computation again and could not advance the journal.

## Frozen correction

T10.2.3 precomputes the exact donor posterior in the parent process before the
T10.2.2 reset watchdog is armed.  Candidate synthesis, event order, branch
boundaries, posterior updates, exception accounting, candidate cap, lane
schedule, action count and scientific controller remain byte-for-byte those of
the frozen parent kernel.

The cache:

- is keyed by the ordered donor event IDs, event checksums and complete compact
  event payloads, the two ordered donor games, the T10.2.2 kernel checksum, the
  T10.2.3 manifest checksum and runtime source bytes;
- checkpoints after every eight donor events using two authenticated pickle
  slots and atomic metadata replacement;
- resumes only an exact authenticated prefix and refuses drift or corruption;
- is limited to 512 MiB per state image;
- is outside the T10.2.2 collection-root allowlist and has no scientific or
  validation authority;
- is loaded by confirmation workers only after their completed discovery
  evidence reconstructs the identical cache key.

Cache build time remains charged to the registered lane and collection clocks.
Only the reset liveness clock moves: it begins after the exact cached posterior
is ready.  A completed cache is immutable.

## Migration contract

The signed migration receipt binds the parent T10.2.2 top manifest and kernel,
the initial checkpoint revision and checksum, the compact cursor, every
completed lane and reset report, their event-ID digests, the global action
equation and the ordered completed discovery evidence.  It registers the next
lane and explicitly denies physical replay.

Before `status`, `prepare`, or `collect`, live reconstruction must prove that
the frozen lane prefix is unchanged, action accounting remains closed, no
intent is unresolved or unknown, and discovery evidence has not regressed.
Later lanes may append records; the frozen prefix may never change.

## Unchanged boundaries

- The existing T10.2.2 journal remains the sole physical evidence ledger.
- No completed reset or action is replayed.
- No raw frame, coordinate, colour or persistent entity identifier enters the
  side cache.
- Source holdout evidence never enters its own donor fold.
- Validation games and AR25 remain closed.
- T10.2.3 does not promote a scientific verdict; final T10.2.2 gates and
  reports remain mandatory.

