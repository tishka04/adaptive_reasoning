# SAGE12 V4.10 — Functional intervention collection

Status: **READY UNDER CAPACITY AMENDMENT**. No V4.10 model result had been
observed at publication.

## Outcome

- Requested rows: 1,664.
- Collected unique rows: 1,587.
- Ten games reached their original quota.
- `su15` reached 83/160 after all 40 resets and 1,280 executed steps.
- `su15` rejected 1,197 exact repeats against V4.9/V4.10.
- Source validation, historical, holdout and online/live environments remained
  closed.

The capacity amendment lowers only the `su15` minimum to 80. All other quotas,
the representation, training configuration and evaluation thresholds remain
frozen. The amended total minimum is 1,584; the collection contains 1,587.

## Fresh semantic yield

| Effect | New positives |
|---|---:|
| changed | 1,464 |
| moved | 169 |
| target_created | 42 |
| target_removed | 62 |
| target_moved | 28 |
| level_complete | 1 |
| game_over | 28 |
| local_change | 156 |
| path_opened | 1 |
| path_closed | 11 |
| actor_approached_root | 8 |
| contact_gained | 0 |
| contact_lost | 114 |
| reachable_area_increased | 29 |
| reachable_area_decreased | 32 |
| productive | 107 |
| risk | 67 |

The collection adds useful density for productive change, risk, target effects
and reachability, plus one genuine completion. It does not solve every capacity
problem: contact gain remains absent and path opening remains almost entirely
game-specific. Those limitations remain visible to LOGO evaluation.

## Audit

- Original manifest:
  `0c1fa71390784cfa49710283047a542a12b5f5cce2df4e1f446690079b3f36db`
- Collection checksum:
  `51fbdfbb7f379730c4797f0188d06db648acaeb7a61b5cebdb192db2a2e1b534`
- Capacity amendment:
  `cef75fd72e8fd2f7b89673b330870301d59e718ef9b64dfaa2ddf9104106f0e9`

The next authorized step is compiling the augmented action-aligned teacher
corpus, then publishing its QA before GPU training.
