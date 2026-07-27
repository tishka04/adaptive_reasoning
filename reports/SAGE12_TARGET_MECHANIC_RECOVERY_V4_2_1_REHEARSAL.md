# SAGE12 V4.2.1 source rehearsal result

Date executed: 2026-07-27

Status: `PASS_SOURCE_REHEARSAL`

Rehearsal checksum:
`cd2164ecdfab094d99364cfdec213767987e974e9fd5b4dc01f98db423873b92`

Prediction SHA-256:
`cc5263941941a9e974e268602097cb4fa89a6643bd3569e2722eb0733b8d423b`

## Result

The mandatory source-only runtime rehearsal passed every frozen check before
source validation or prospective outcomes were opened.

- 1,911 immutable source-training windows produced 1,911 serialized
  prediction rows;
- 14 unique queries generated 168 rules across the three target effects;
- all 168 rules survived public serialization and internal restoration, for a
  round-trip rate of 1.00;
- 42 exact and 42 family generic-`any` rules were explicitly exercised;
- selected structured evidence serialized a generic `any` rule 2,120 times;
- actor outcomes remained excluded and the model-view firewall passed.

The 2,433,083-byte prediction stream is stored through Git LFS. Its checksum
is part of the rehearsal artifact, whose manifest reference matches frozen
manifest
`81f14c655dc6b824970b2ecd8638ca62360abedcc7f4dcf3abed2b86cdd3a3c8`.

## Authority

This result authorizes only the unchanged V4.2.1 source-training preflight.
Source validation, V4.2.1 prospective collection, Qwen evaluation, V5,
world-model fitting, EBM fitting, and controller use remain unauthorized.
