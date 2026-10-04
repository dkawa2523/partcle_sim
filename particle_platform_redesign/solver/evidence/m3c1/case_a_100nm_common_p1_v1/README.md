# M3-C1 Case-A 100 nm common-P1 v1

Evaluation status: **COMPLETE**. Locked same-field decision: **PASS on all
9/9 predeclared gates**.

## Scope

This is the follow-up diagnostic triggered by the separate native-FE versus
canonical-P1 failure. Both integrations consume the same canonical
exact-connectivity P1 field tables: 1,987 nodes, 3,779 TRI3 cells, 17 nodal
fields, and 22 field components. The locked case is Case A, 100 nm,
axisymmetric R-Z without swirl, 287 particles, 46 output frames over
`0--450 us`, dynamic charge, seven deterministic force contributions, and
Brownian motion disabled.

The candidate and COMSOL reference each ran internal steps of `0.625`,
`0.3125`, and `0.15625 us`. Every trajectory has 287 x 46 = 13,202 rows and
eight columns. Independent parsing found every state finite and active, with
the complete particle/time key grid. Candidate failures, candidate boundary
events, and reference event observations are all zero.

## Independent integrity and convergence checks

The reference ledger contains 61 files and 90,668,703 indexed bytes; all
SHA-256 digests and sizes were recomputed with zero mismatches. The 26 staged
common-P1 artifacts total 3,913,705 bytes and match both their table receipt
and the candidate's `common_p1_tables_v2` copies byte for byte. The current
source MPH also retains the run-attested SHA-256. No large raw artifact is
copied into this evidence directory.

The candidate input independently resolves to 1,987 nodes, 3,779 TRI3 cells,
160 boundary lines, 287 source particles, and 17 nodal fields / 22 components.
For each reference run, all 6,314 release primitive values (287 x 22) agree
with the canonical probes within the fixed 4,096-ULP criterion; the largest
component-scale difference is 19 ULP. The candidate/reference initial state
is not bitwise identical because of decimal export roundoff, but all 1,435
values (287 x 5) pass the same locked criterion; the maximum is 277 ULP.

Candidate self-comparison remains `ROUNDOFF_LIMITED`: its fine-pair relative
L2 values are `2.677331196127922e-14` for position,
`1.448610117908637e-14` for velocity, and
`5.456212097587488e-15` for charge. This is precision stability, not an
independent RK4-order result. The common-P1 COMSOL reference is separately
order-evaluated and passes, with fine-pair relative L2 values
`6.099791412768047e-8`, `8.321361051196223e-8`, and
`1.3796076667903878e-8`, and observed RMS orders `2.0298752666513638`,
`2.0816962034597633`, and `2.044082972214284`.

## Locked same-field comparison

The fine trajectories were compared only after the absolute and relative
limits were registered. Recalculation from the two CSV files gives:

| quantity | RMS (observed / limit) | maximum (observed / limit) | relative L2 (observed / limit) | decision |
|---|---:|---:|---:|---:|
| position | `2.4360183916994177e-11 / 6e-8 m` | `2.3107811844634912e-10 / 7e-7 m` | `1.2165503469347329e-8 / 1e-4` | `PASS` |
| velocity | `1.8055229266990532e-7 / 4e-4 m/s` | `1.1037594178013694e-6 / 3.5e-3 m/s` | `1.6810197404693313e-8 / 5e-4` | `PASS` |
| charge | `7.930847904707173e-7 / 4e-4 e` | `7.105382977101726e-6 / 4.5e-3 e` | `3.133867418408752e-9 / 2e-5` | `PASS` |

## Claim boundary

- Same-field solver agreement is supported only for this locked Case-A
  100 nm, common-canonical-P1, Brownian-off, `0--450 us` window.
- The earlier native-FE versus canonical-P1 result remains a separate `FAIL`;
  this diagnostic neither replaces nor relabels it.
- No boundary event occurred. Boundary-event accuracy is therefore
  `NOT_TESTED_PRE_EVENT_WINDOW`.
- Brownian accuracy is `NOT_TESTED_DISABLED`.
- Physical model validity and applicability are `NOT_CLAIMED` by this
  numerical comparison.
- Universal COMSOL-equivalent accuracy is `NOT_CLAIMED`, and COMSOL is not
  treated as golden truth.

Exact hashes, shapes, statuses, and independently recomputed metrics are in
[`comparison_manifest.json`](comparison_manifest.json). Individual gate rows
are in [`gates.csv`](gates.csv).
