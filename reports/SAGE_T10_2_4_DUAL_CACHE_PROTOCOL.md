# SAGE.T10.2.4 — dual-cache continuation protocol

Status: **frozen before continuation**.

The first preflight manifest (`aadf02d5…`) is superseded.  Its `prepare`
command exposed an adapter typo (`FactorizedBankMetrics.to_dict` instead of
the frozen `as_dict` API) before the T10.2.4 cache root was created and before
any physical action.  The replacement manifest records that checksum in its
lineage and binds the corrected adapter plus regression test.

## Registered failure

T10.2.3 correctly moved the ordinary gauge-posterior fit outside the reset
watchdog and advanced the durable collection from nine to thirteen complete
lanes.  At lane 14 (`lp85`, confirmation seed 113), the 546-event gauge cache
completed successfully.  The first reset then rebuilt the capacity-matched
five-factor control posterior inside the 630-second watchdog.  It consumed CPU
continuously but was terminated at exactly 630 seconds before emitting an
intent.  The lane remained empty and the journal action equation stayed closed.

The frozen worker also reconstructs the ordinary gauge posterior on later
`learned` resets even though each spawned worker has just loaded the same fresh
donor-only state.  With 418 events those redundant fits took roughly 467
seconds; the factorized fits took 530–540 seconds.  Both scale beyond the reset
watchdog at 546 events.

## Frozen correction

T10.2.4 retains the T10.2.2 scientific kernel, journal, action budgets, lane
schedule, cross-fit folds and authority boundary.  Before a confirmation reset
watchdog is armed it ensures two exact donor-only states:

1. `GaugeProgramPosterior`, adopting an exact signed T10.2.3 cache read-only
   when available, otherwise building a new resumable cache;
2. `FactorizedGaugeProgramPosterior`, seeded with the unchanged audited
   `FactorizedCandidateBank` and fitted over the same ordered donor events.

Inside the child, the frozen fit function is intercepted only when all of the
following match an authenticated cache: ordered compact events and checksums,
ordered training games, candidate hashes, factor rows and marginals when
factorized, posterior class, and the 256-candidate limit.  A mismatch delegates
to the original frozen fit.  Online held-out observations, preview copies and
all non-donor fits remain untouched.

Both new cache kinds use alternating authenticated pickle slots, atomic
metadata, an eight-event checkpoint interval and a 512 MiB state-image limit.
Cache construction is visible, resumable, performed before the reset watchdog,
and remains charged to lane and collection clocks.

## Migration and boundaries

The T10.2.4 receipt binds the thirteen complete lanes, 52 reset reports, 998
sealed actions, checkpoint revision 188 and checksum, compact cursor, discovery
evidence, and the exact metadata and binary hashes of the two adopted T10.2.3
caches.  No completed action is replayed.

Validation games and AR25 remain closed.  Caches have no scientific authority
and cannot promote a verdict.  Final source reports, action accounting,
cross-fit audit, evidence funnel and registered gates remain mandatory.
