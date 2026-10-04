# M3-C0 boundary semantics v2

Evaluation status: **COMPLETE**. Scientific decision: **PASS** only for the
active free-flight path and two isolated COMSOL terminal-boundary probes in
this package. The evaluated table contains **56 PASS, 0 FAIL, and 6
`CHARACTERIZED_NOT_GATED`** rows.

## Scope and result

The probe uses one 100 nm particle per scenario, force-free motion, no dynamic
charge, classical RK4, and fixed steps of `10`, `5`, and `2.5 us`. Every run
has 61 saved frames over `0--150 us`: 30 active frames followed by 31 terminal
frames. The analytic normal-impact time is `7.3e-5 s`; the last active frame is
`7.25e-5 s` and the first terminal frame is `7.5e-5 s` in all six runs.

Version 2 adds an analytic check of every active frame against
`x(t) = x0 + v0*t` and `v(t) = v0`. Across all six runs, the maximum active
position error is `4.726604209672303e-16 m` and the maximum active velocity
error is `1.7763568394002505e-15 m/s`, below their respective `1e-12` limits.

| scenario | step | event time (s) | event-time error (s) | hit-position error (m) | terminal observation |
|---|---:|---:|---:|---:|---|
| boundary 37 Freeze | `10 us` | `7.300000000001076e-5` | `1.0760706561918632e-17` | `2.7755575615628914e-17` | status 2; fixed position; source velocity retained |
| boundary 37 Freeze | `5 us` | `7.300000000001432e-5` | `1.4325021203964727e-17` | `2.7755575615628914e-17` | status 2; fixed position; source velocity retained |
| boundary 37 Freeze | `2.5 us` | `7.30000000000415e-5` | `4.149783815188268e-17` | `2.7755575615628914e-17` | status 2; fixed position; source velocity retained |
| boundary 35 Disappear | `10 us` | `7.299999999999997e-5` | `2.710505431213761e-20` | `2.168404344971009e-19` | status 4; position and velocity unavailable |
| boundary 35 Disappear | `5 us` | `7.3e-5` | `0.0` | `0.0` | status 4; position and velocity unavailable |
| boundary 35 Disappear | `2.5 us` | `7.299999999999997e-5` | `2.710505431213761e-20` | `2.168404344971009e-19` | status 4; position and velocity unavailable |

The event-time spread across the three steps is
`3.073713158996405e-17 s` for Freeze and `2.710505431213761e-20 s` for
Disappear, against the `5e-9 s` limit. Freeze hit positions are direct COMSOL
frozen-state observations. Disappear exposes no direct post-event position, so
its labelled hit position is an analytic reconstruction from the COMSOL
initial state; it is not a direct event-position observation.

Post-event velocity has no acceptance threshold. COMSOL retained the exact
source velocity in all 31 Freeze terminal frames and returned no finite
position or velocity pair in the 31 Disappear terminal frames. Those six
velocity rows are characterization only and account for every non-PASS row.

## Independent integrity audit

The audit reparsed each raw wide table without importing the normalizer. All
`6 * 61 * 9` raw values matched the normalized tables, including NaN placement,
with zero mismatches. It independently recomputed every active-path error,
event time, status transition, hit-position error, and terminal-state rule;
all six summaries and all 62 gate outcomes agreed.

The COMSOL process log contains one run-start and one run-pass receipt, six
load-start, load-pass, solve-pass, and configuration receipts, and no error
receipt. Every configuration receipt matches its corresponding step summary.
The output contains 35 files and 301,502 bytes. All 33 entries in its artifact
ledger were independently rehashed and re-sized with zero mismatches; the
ledger and final run status are the two intentionally unindexed files. No raw
output is copied into this evidence directory.

The source MPH was loaded with `ModelUtil.loadCopy`, was not saved, and its
recomputed SHA-256 remained
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`,
matching the configuration and before/after attestations. The four current
configuration/normalizer/Java/runner sources also match their staged run
copies byte-for-byte.

## Claim boundary

This result certifies only the COMSOL active force-free path and boundary 37
Freeze / boundary 35 Disappear terminal semantics for these two isolated,
normal-impact microcases. It does **not** test production-solver parity, close
P18-H, establish full-physics trajectory agreement, or certify grazing,
corners, probabilistic walls, reflection, charge evolution, native-field versus
canonical-P1 equivalence, other boundaries/cases/sizes, or universal COMSOL
accuracy. COMSOL is not treated as golden truth.

Charge was not exported, post-event velocity was not gated, and the raw output
has no hit-boundary-ID column. The three-step event-time check is
self-consistency evidence, not fourth-order RK4 convergence. The hash audit is
an audit-time internal-integrity check, not an external immutability
attestation.

Exact metrics, receipts, hashes, sizes, and claim exclusions are in
[`comparison_manifest.json`](comparison_manifest.json). All 62 evaluated rows
are in [`gates.csv`](gates.csv).
