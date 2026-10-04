# M3-C1 Case-A 100 nm material-event v1

Evaluation status: **COMPLETE**. The locked material-event decision is
**PASS on 20/20 gates**. The current v14 pre-event same-field comparison is
separately **PASS on 9/9 gates**, and the v14 three-step candidate
self-convergence check is **PASS on 6/6 recorded acceptance gates**.

## Scope

This compact record covers only Case A, 100 nm, axisymmetric R-Z without
swirl, 287 particles, Brownian motion disabled, the common canonical
exact-connectivity P1 field, and the first wafer `stick` event. The material
run has 48 saved frames over `0--458.75 us`; the pre-event prefix has 46
frames over `0--450 us`. Dynamic charge and the same seven deterministic
contributions used by the common-P1 checkpoint are active.

The candidate first event is particle 57, ordinal 1, canonical boundary 6
(`wafer`, external ID 134), primary facet 38, law `stick`, and outcome
`stuck`. Candidate and reference each have exactly one nonactive particle and
two saved terminal rows.

## First material-event result

| measure | candidate v14 | COMSOL v7 reference | absolute difference | locked limit |
|---|---:|---:|---:|---:|
| event time | `0.00045782250019099044 s` | `0.0004578225004067447 s` | `2.157542807607049e-13 s` | `1e-9 s` |
| hit position | `(0.14790084010874993, 0.022) m` | `(0.1479008401087231, 0.022) m` | `2.683964162031316e-14 m` | `2e-9 m` |
| terminal charge | `8.974770710019452 e` | `8.974770714747834 e` | `4.7283812421028415e-9 e` | `4.5e-3 e` |

Both event times lie between the last active saved frame (`450 us`) and the
first terminal saved frame (`458 us`). Both terminal position holds and both
terminal charge holds have zero observed span. The candidate event post
velocity and both saved stuck velocities are exactly zero. These observations
are the narrow first-stick checks represented by the 20 material-event gates;
they are not a general boundary-law accuracy result.

## Current v14 pre-event confirmation

The final current same-field authority is the separately registered v14 fine
trajectory from `_out_m3c1/caseA_100nm_exported_p1_v5`, not the earlier
roundoff-limited candidate. Direct recalculation gives:

| quantity | RMS (observed / limit) | maximum (observed / limit) | relative L2 (observed / limit) | decision |
|---|---:|---:|---:|---:|
| position | `4.06807283316903e-13 / 6e-8 m` | `1.2035778717837921e-12 / 7e-7 m` | `2.0316001855367802e-10 / 1e-4` | `PASS` |
| velocity | `2.10847522277453e-9 / 4e-4 m/s` | `3.844306466969233e-9 / 3.5e-3 m/s` | `1.9630813983926987e-10 / 5e-4` | `PASS` |
| charge | `1.1818881019499895e-7 / 4e-4 e` | `2.3758877887303242e-7 / 4.5e-3 e` | `4.670220207738041e-10 / 2e-5` | `PASS` |

The v14 candidate is also independently order-evaluated across `0.625`,
`0.3125`, and `0.15625 us` internal steps:

| quantity | observed RMS order | fine-pair relative L2 | decision |
|---|---:|---:|---:|
| position | `2.029875353701904` | `6.099791486063973e-8` | `PASS` |
| velocity | `2.0816971911764033` | `8.321356016032579e-8` | `PASS` |
| charge | `2.044084026475049` | `1.3796067752988052e-8` | `PASS` |

These approximately second-order observations are empirical for this
piecewise-P1, mesh-crossing case. They do **not** certify formal fourth-order
RK4 behavior.

The older v13 candidate self-comparison remains valid historical evidence of
precision stability; it is not invalidated. Its three runs were classified
`ROUNDOFF_LIMITED` because artificial event-certificate subdivision drove
them to the same 8,450,307 accepted-particle-piece count, so independent
temporal convergence could not be evaluated. The v14 pre-event reruns have
zero refinements and accepted-piece counts proportional to the macro-step
counts (`206,927`, `413,567`, and `826,847`), permitting the order check above.

## v13 to v14 event-search work

| counter | v13 | v14 | change |
|---|---:|---:|---:|
| candidate queries (Q) | `16,427,517` | `842,927` | `-94.8688106668829%` |
| refinements (R) | `7,792,306` | `11` | `-99.9998588351125%` |
| accepted particle pieces (A) | `8,635,211` | `842,916` | `-90.2386172150281%` |
| maximum refinement depth | `16` | `11` | `-5` |

For both runs, `Q = A + R`. At the 450-us checkpoint, v13 had already
accumulated 7,623,460 of its final 7,792,306 refinements
(`97.8331703092769%`); v14 had accumulated zero and used only 11 refinements
after that checkpoint to localize the first event.

The observed elapsed-time change, approximately `14 min 13 s` (`~853 s`) to
`~36.5 s`, is an operator-observed shell wall-time, not a solver-reported or
artifact-embedded timer. It is approximate, machine-local, and non-gating;
the ratio from the rounded observations is about `23.37x`.

## Integrity and claim boundary

The COMSOL v7 artifact ledger contains 40 files and 19,174,223 indexed bytes.
Every indexed SHA-256 and size, ledger coverage, the three raw table shapes,
candidate shapes, embedded candidate hashes, evaluation statuses, and gate
tables were independently rechecked with no mismatch. No large raw artifact
is copied here.

- The accepted claim is limited to the locked Case-A 100 nm common-P1,
  Brownian-off first wafer-stick case and the stated windows.
- The earlier native-FE versus canonical-P1 cross-representation `FAIL`
  remains a separate authoritative result. Native-field agreement is not
  claimed here.
- Other particles, sizes, cases, field representations, boundary groups,
  boundary laws, later events, and a 30-ms trajectory are not certified.
- Brownian accuracy is `NOT_TESTED_DISABLED`.
- Physical validity and applicability are `NOT_CLAIMED`.
- Universal COMSOL-equivalent accuracy is `NOT_CLAIMED`; COMSOL is not golden
  truth.
- Terminal-velocity parity is not generalized beyond the saved first-stick
  observations recorded by the material gates.

Exact source hashes, shapes, statuses, recomputed metrics, and performance
provenance are in [`comparison_manifest.json`](comparison_manifest.json).
The four integrity checks, six v14 self-convergence checks, nine current
same-field gates, and twenty material-event gates are separated in
[`gates.csv`](gates.csv).
