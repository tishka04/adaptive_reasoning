# SAGE12 V4.11 — Counterfactual-panel collection

Status: **COMPLETE**.

The frozen source-only collector retained all 1,056 requested panels: exactly
96 in each of the eleven SAGE11 source-train games. The panels contain 3,914
distinct immediate intervention arms and 15,124 deterministic continuation
transitions.

Collection checksum:
`6d85ce2f20b635f8a15432aa47505a6cd9a65d64c796f533fa78dbd0bf99d5dc`.

## Capacity

| Game | Panels | Arms | Resets | Base steps | Continuation steps | Candidate shortfalls |
|---|---:|---:|---:|---:|---:|---:|
| `bp35` | 96 | 384 | 2 | 97 | 1,352 | 1 |
| `cd82` | 96 | 346 | 2 | 135 | 1,376 | 39 |
| `dc22` | 96 | 353 | 2 | 99 | 1,403 | 3 |
| `g50t` | 96 | 349 | 2 | 165 | 1,396 | 69 |
| `ka59` | 96 | 356 | 1 | 97 | 1,424 | 1 |
| `lf52` | 96 | 382 | 2 | 97 | 1,504 | 1 |
| `lp85` | 96 | 342 | 13 | 236 | 1,328 | 140 |
| `sp80` | 96 | 356 | 4 | 113 | 1,323 | 17 |
| `su15` | 96 | 384 | 3 | 96 | 1,464 | 0 |
| `tr87` | 96 | 377 | 1 | 97 | 1,508 | 1 |
| `tu93` | 96 | 285 | 7 | 334 | 1,046 | 238 |
| **Total** | **1,056** | **3,914** | **39** | **1,566** | **15,124** | **510** |

A candidate shortfall means that the current state did not expose two novel
actions after the frozen exact-repeat filter. It does not remove a retained
panel. `lp85` and `tu93` therefore needed more base traversal and resets, but
both reached their full quotas without an amendment.

## Frozen checks

All collection checks passed:

- every game reached 96 panels, above the frozen minimum of 80;
- all rows belong to the source-train split;
- all panel IDs are unique;
- all 3,914 immediate pre-state/action keys are unique;
- none repeats an exact V4.3–V4.10 intervention;
- every arm shares its panel's replay-verified pre-state checksum;
- there were zero replay failures and zero admitted duplicate arms.

Source validation, holdout, external historical data, and live environments
remained closed. The action-selection policy used only pre-action
object-relative signatures and stable hash tie-breaks; no outcome label or
horizon return influenced collection.

The raw shards are teacher audit material. They expose before/after frames and
continuation traces, so they must never be used as student inputs. The next
step is the deterministic V4.11 teacher compilation and its pre-model capacity
audit.
