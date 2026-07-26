# SAGE.11 source-corpus capacity result

Date: 2026-07-26
Status: `COMPLETE_AFTER_APPROVED_AMENDMENT`

## Original base-cap outcome

The real-environment collector cannot publish the requested 100,000 exact
deduplicated source transitions while preserving the pre-registered
8,000-transition per-game cap.

Two games exhausted all policy seeds (`0` through `4`) and then produced 4,000
consecutive duplicate transition signatures:

| Game | Exact accepted transitions | Raw transitions | Result |
| --- | ---: | ---: | --- |
| `lp85` | 27 | 21,217 | saturated |
| `sp80` | 2,681 | 28,675 | saturated |

Giving every other one of the 14 source/source-validation games the full
8,000-row cap yields the optimistic upper bound:

```text
12 × 8,000 + 27 + 2,681 = 98,708
```

The minimum shortfall is therefore 1,292 rows. This is a mathematical upper
bound, not a throughput estimate; continuing the unfinished games cannot
change it.

The machine-readable evidence is
[`training/sage11/source_dataset_v2/capacity_report.json`](../training/sage11/source_dataset_v2/capacity_report.json).
Its checksum is
`3300ccf48451bbc1f56473c33627e4bded3ed17db8932d69d60e2357204fb748`.

## Protocol actually executed

- Real offline ARC environments only.
- Source-train and source-validation games only; no holdout or historical
  game was touched.
- Seeds `0,1,2,3,4`.
- Fixed accepted-row mixture in deterministic 10-row blocks: 70% active
  controller, 20% uniform legal, 10% frontier probe.
- Exact
  `before-state/action/arguments/after-state/effect` deduplication.
- Exact state digests include grid shape, game state, level count, and raw grid
  bytes; reset and episode identifiers are deliberately excluded.
- Per-game cap 8,000.
- Saturation only after 4,000 consecutive duplicates in each seed.
- No M2/v4 weights loaded.

At the stop decision, 76,165 accepted rows had been checkpointed. Those private
work shards remain excluded from Git because they are incomplete controller
runs. No 100,000-row manifest or frozen 11-game curriculum is claimed or
published.

## Implementation delivered

The repository now contains:

- a real, resumable, multi-process source collector;
- exact 70/20/10 scheduling and multi-seed exploration;
- exact-state deduplication and explicit source split fields;
- completed/saturated per-game checkpoints;
- deterministic exact-row publication allocation with bounded final-block
  tails;
- manifest row-count, split-count, policy-count, target, and checksum
  verification;
- source-schema freezing and source-only curriculum verification;
- recursive Git LFS coverage for published JSONL shards;
- behavior-preserving performance repairs in multiform relation matching,
  spatial facts, frontier component normalization, and structural relation
  typing;
- a fail-closed capacity report with an optimistic upper-bound proof.

## Collection-performance evidence

Profiling the sparse source tail found 94,695 repeated connected-component
searches in one 320-transition `su15` sample; `_component()` accounted for
410.983 of 456.765 profiled seconds. The repair computes the role signature
map once per exact grid and keeps a bounded 1,024-grid LRU cache. On the same
deterministic 90-accepted/320-raw sample, wall time fell from 322.288 seconds
to 10.439 seconds (30.9x) with identical accepted output.

The relation learner now also caches immutable object-match indices, spatial
fact multisets, and normalized shape keys. Structural-frontier relation
typing is computed once per object rather than once per pair; a representative
`tu93` 1,569-accepted/3,000-raw sample fell from about 110 seconds to 77.049
seconds with the same row result. These are local collection-throughput
diagnostics, not model-quality evidence, and none changes action selection,
transition signatures, labels, caps, or split authority.

## Decision that was required

At least one pre-registered constraint must change before collection can
continue:

1. Raise the 8,000 cap by an aggregate minimum of 1,292 rows on games that
   still have unique capacity.
2. Add at least one new training-eligible source game and update the immutable
   split registry/checksum.
3. Redefine deduplication to retain repeated episodes or controller contexts.
   This is not recommended because it weakens the behavioral uniqueness
   guarantee.

## Amendment decision

On 2026-07-26 the user approved option 1: a single aggregate overflow pool of
1,292 additional rows on games with remaining unique capacity. The
implementation distributes it deterministically as +259 to `cd82` and `dc22`,
and +258 to each of `g50t`, `ka59`, and `tr87`. The original capacity report
and checksum remain unchanged as the evidence that justified the amendment.

## Amended collection result

The amended runner published and independently verified exactly 100,000 unique
rows across 14 source shards:

| Split | Published rows |
| --- | ---: |
| Source train | 76,908 |
| Source validation | 23,092 |
| Total | 100,000 |

The five amended games use the complete approved pool: `cd82=8,259`,
`dc22=8,259`, `g50t=8,258`, `ka59=8,258`, and `tr87=8,258`. No validation,
holdout, historical, or regression game received overflow. The two sparse
source games closed at verified target capacity with `sp80=5,948` and
`tu93=5,641`; deterministic validation allocation supplied the remaining
23,092 rows (`re86=7,698`, `ls20=7,697`, `sc25=7,697`).

The fixed accepted-row mixture published 69,999 active-controller, 20,002
uniform-legal, and 9,999 frontier-probe rows. The two-row aggregate deviation
from ideal 70/20/10 is the bounded sum of final shard tails. The corpus
contains 44 strong terminal/level events and 3,821 explicitly weak progress
events, so the terminal head remains disabled under the pre-registered
100-strong-event gate. ACTION6 coverage contains 24,095 rows across 633
distinct `(x,y)` keys.

Evidence checksums:

- manifest:
  `d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`;
- frozen curriculum:
  `d11948c5cfcb70ce888b435d63d217b95ce2a0006e4423ae7ac70374d81c630c`;
- merged 31-schema library:
  `79d5622ba3b8dc4bb9621b931e08ca3119b9035fc49c3461a62599560055b8ac`.

The data and curriculum gates for roadmap steps 1-3 are complete. Neural
training, authority promotion, and holdout evaluation were not started.

## Validation

- 1,648 repository tests passed.
- 48 focused SAGE.10g-i, source-runner, dataset, frontier, and multiform tests
  passed.
- Ruff passed on every changed Python file.
- `git diff --check` passed.
- The only test warning was the existing joblib fallback from unavailable
  physical-core discovery to the Windows logical-core count.
