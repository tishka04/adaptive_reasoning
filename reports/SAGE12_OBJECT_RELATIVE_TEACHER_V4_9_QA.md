# SAGE12 V4.9 — Semantic teacher corpus and QA

The source-only semantic teacher is **READY**. This is the published
pre-training checkpoint; no student result had been observed when it was
created.

## Corpus

- 3,040 V3 action-target traces read.
- 4,792 V4.3 executed arms read.
- 291 duplicate V4.3 trace digests removed.
- 7,541 unique teacher records retained.
- 2,396 same-prestate counterfactual pairs retained.
- 11/11 games belong to immutable `SOURCE_TRAIN`.
- Source validation, holdout, historical and live environments remained closed.

The student-facing corpus is intentionally frame-free. Each row contains audit
provenance, the identity-free pre-action graph, teacher labels, applicability
masks and compact post-transition evidence. Raw frames remain in their
previously published source shards.

## QA

All frozen teacher checks passed:

- root grounding: **1.000** (minimum 0.950);
- strict JSON round-trip: pass;
- same-prestate pair references: pass;
- forbidden student fields absent: pass;
- source-only firewall: pass.

Actions with neither an object target nor a detectable actor use the legal
intervention itself as `action_root`. This is a valid deployable semantic root,
not an inferred object or an absolute position. The change corrected the root
taxonomy; it did not lower the threshold or alter effect labels.

## Label capacity

| Effect | Applicable | Positive |
|---|---:|---:|
| changed | 7,541 | 7,101 |
| moved | 6,129 | 1,775 |
| target_created | 6,129 | 332 |
| target_removed | 4,407 | 961 |
| target_moved | 4,407 | 155 |
| level_complete | 7,541 | 5 |
| game_over | 7,541 | 151 |
| local_change | 6,129 | 1,988 |
| path_opened | 6,129 | 403 |
| path_closed | 6,129 | 119 |
| actor_approached_root | 6,030 | 54 |
| contact_gained | 6,030 | 36 |
| contact_lost | 6,030 | 1,484 |
| reachable_area_increased | 6,129 | 150 |
| reachable_area_decreased | 6,129 | 177 |
| productive | 7,541 | 1,333 |
| risk | 7,541 | 394 |

The five completion positives occur only in `lp85` (4) and `sp80` (1).
Completion supervision was neither synthesized nor duplicated. Several
functional effects remain concentrated in a small subset of games, so LOGO
transfer—not in-sample fit—is the decisive next measurement.

## Content addresses

- Frozen manifest:
  `311f7a1ac4fee1cb265e12cfb8cd92ab5654eb135ce7bbdbcc44117279f35710`
- Teacher corpus SHA-256:
  `30cd76d33d05ed309e58b44b6569b5079a1ac967e18ae9e7ea9eaa78a981c956`
- Same-prestate pairs SHA-256:
  `53fa00b17581f22c28fa58f79d97308f1150da935154c0f163c4025cc318c751`
- Teacher QA checksum:
  `d3424198a670479fc31874e282dbee8b5e39aadd554bbe09b58a964dfb47335e`

The next authorized action is training the frozen object-relative student with
outer leave-one-game-out evaluation.
