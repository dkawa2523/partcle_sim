# M3-C0 boundary semantics v1

Evaluation status: **COMPLETE**. Scientific decision: **PASS** for the two
isolated COMSOL terminal-boundary probes in this package. The evaluated table
contains **44 PASS, 0 FAIL, and 6 `CHARACTERIZED_NOT_GATED`** rows.

## Scope and result

The probe uses one 100 nm particle per scenario, force-free motion, no dynamic
charge, classical RK4, and fixed steps of `10`, `5`, and `2.5 us`. Every run
has 61 saved frames over `0--150 us`. The analytic normal-impact time is
`7.3e-5 s`; the last active frame is `7.25e-5 s` and the first terminal frame
is `7.5e-5 s` in all six runs.

| scenario | step | event time (s) | event-time error (s) | hit-position error (m) | terminal observation |
|---|---:|---:|---:|---:|---|
| boundary 37 Freeze | `10 us` | `7.300000000001076e-5` | `1.0760706561918632e-17` | `2.7755575615628914e-17` | 31 held frames; position spread `0.0 m` |
| boundary 37 Freeze | `5 us` | `7.300000000001432e-5` | `1.4325021203964727e-17` | `2.7755575615628914e-17` | 31 held frames; position spread `0.0 m` |
| boundary 37 Freeze | `2.5 us` | `7.30000000000415e-5` | `4.149783815188268e-17` | `2.7755575615628914e-17` | 31 held frames; position spread `0.0 m` |
| boundary 35 Disappear | `10 us` | `7.299999999999997e-5` | `2.710505431213761e-20` | `2.168404344971009e-19` | position and velocity unavailable in all 31 terminal frames |
| boundary 35 Disappear | `5 us` | `7.3e-5` | `0.0` | `0.0` | position and velocity unavailable in all 31 terminal frames |
| boundary 35 Disappear | `2.5 us` | `7.299999999999997e-5` | `2.710505431213761e-20` | `2.168404344971009e-19` | position and velocity unavailable in all 31 terminal frames |

The event-time spread across the three steps is
`3.073713158996405e-17 s` for Freeze and `2.710505431213761e-20 s` for
Disappear, against the `5e-9 s` limit. Freeze hit positions are direct COMSOL
frozen-state observations. Disappear exposes no direct post-event position, so
its labelled hit position is an analytic reconstruction from the COMSOL
initial state; it must not be described as a direct event-position
observation.

Post-event velocity has no acceptance threshold. COMSOL retained the exact
source velocity in all 31 Freeze terminal frames and returned no finite
velocity pair in the 31 Disappear terminal frames. Those six rows are
characterization only and account for every non-PASS row.

## Integrity

The source MPH was loaded with `ModelUtil.loadCopy`, was not saved, and its
recomputed SHA-256 remained
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`,
matching both the before- and after-run attestations. The authoritative output
contains 35 files and 272,319 bytes. All 33 entries in its artifact ledger
were rehashed and re-sized with zero mismatches; the ledger itself and final
run status are the two intentionally unindexed files. No raw output is copied
into this evidence directory.

## Claim boundary

This result certifies only COMSOL boundary 37 Freeze and boundary 35 Disappear
terminal semantics for the two isolated, force-free, normal-impact
microcases. It does not test production-solver boundary parity, full-physics
trajectory agreement, grazing or corner events, probabilistic sticking or
reflection, native-field versus canonical-P1 equivalence, other boundaries or
cases, or universal COMSOL-equivalent accuracy. COMSOL is not treated as
golden truth. Charge was not exported, post-event velocity was not gated, and
the raw output has no hit-boundary-ID column; therefore this does not certify a
canonical lifecycle mapping or close P18-H. The hash audit establishes
internal integrity at audit time, not an external immutability attestation.
The three-step event-time check is self-consistency evidence, not a claim of
fourth-order RK4 convergence.

Exact run metrics and artifact hashes are in
[`comparison_manifest.json`](comparison_manifest.json). All 50 evaluated rows
are in [`gates.csv`](gates.csv).
