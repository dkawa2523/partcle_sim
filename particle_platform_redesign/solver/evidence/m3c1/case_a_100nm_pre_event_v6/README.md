# M3-C1 Case-A 100 nm pre-event v6

Evaluation status: **COMPLETE**. Locked cross-representation decision:
**FAIL; common-field full-physics diagnostic required**.

## Scope and execution

This Brownian-off, axisymmetric R-Z case couples dynamic aggregate charge to
electric force, relative-flow ion drag, linear Epstein drag, Waldmann
thermophoresis, free-molecular lift sensitivity, DEP, and gravity/buoyancy.
The pre-event window is `0--450 us`; boundary behavior is not exercised.

The solver candidate is the canonical exact-connectivity P1 projection of
exported nodal values, not COMSOL's native finite-element interpolant. It
completed fixed RK4 steps of `0.625`, `0.3125`, and `0.15625 us`. Every run
contains 287 particles x 46 frames = 13,202 trajectory rows, with zero boundary
events, zero failures, and all 287 particles active at the end.
Candidate and reference initial position, velocity, and charge share the exact
initial-state hash
`cc23ff17294a768fbe81fe45b8310e850159f3787373815663e0a4937fa157ed`.

P19's bounded local applicability certificate removed the historical
time-zero blocker without relaxing the physical limits. Each run accepted
`8,450,307` particle pieces. The coarse, middle, and fine runs used maximum
refinement depths `8`, `7`, and `6`, respectively. The earlier blocked
candidate remains unchanged in
[`../exported_p1_candidate_blocker_v1`](../exported_p1_candidate_blocker_v1/README.md).

## Self precision and reference convergence

Candidate self-comparison is `PASS`, but position, velocity, and charge are all
classified `ROUNDOFF_LIMITED`. The fine-pair relative L2 values are
`2.677331196127922e-14`, `1.448610117908637e-14`, and
`5.456212097587488e-15`, respectively. These establish stable precision for
this projected-field calculation; they do not establish RK4 order below the
float64 floor.

The independently locked COMSOL native-field reference is `PASS` for its
narrow operational self-convergence gates. Its fine-pair relative L2 values
are:

- position: `3.102727085428029e-5`;
- velocity: `3.924832510840385e-5`;
- charge: `1.3511393490811462e-6`.

The corresponding observed RMS orders are `0.9041136219836077`,
`0.9443119501831206`, and `1.1238380758116033`. This is a pre-event step
selection result, not universal COMSOL accuracy or solver agreement.

## Locked cross-representation comparison

Before comparing fine trajectories, the envelope for each statistic was
registered as the COMSOL fine-step change plus the candidate's float64-scale
precision contribution. All six gates fail:

| quantity | statistic | observed | registered envelope | decision |
|---|---:|---:|---:|---:|
| position (m) | RMS | `3.437485727086595e-4` | `5.3387228259403245e-8` | `FAIL` |
| position (m) | maximum | `2.660729292906687e-3` | `6.744158047424948e-7` | `FAIL` |
| velocity (m/s) | RMS | `1.919910730173562` | `3.6430541525847614e-4` | `FAIL` |
| velocity (m/s) | maximum | `11.668961060303342` | `3.133282330386992e-3` | `FAIL` |
| charge (e) | RMS | `17.58471490428266` | `3.3741975576776693e-4` | `FAIL` |
| charge (e) | maximum | `157.15163693423065` | `3.8981579387940995e-3` | `FAIL` |

The candidate and reference use different field representations by design.
In addition, this v4 candidate input retains the preparation report's
`NOT_TESTED_PRIMITIVE_AUTHORITY` classification for its reconstructed heat
flux, and applies canonical radial regularity projection at 34 axis nodes.
The cross-representation failure therefore does not isolate an integrator,
force model, field interpolation, axis policy, or producer-primitive error.

The source comparison report records the preregistered diagnostic as
`CONDITIONAL_NOT_RUN`. All six trigger gates have now failed, so this checkpoint
promotes it to `TRIGGERED_NOT_RUN_REQUIRED`. The next required external V&V step
is one common-field full-physics diagnostic
that holds release, physics, time schedule, and field values fixed while
removing native-FE-versus-canonical-P1 sampling as a variable. No solver
constant, physical limit, or core branch should be tuned to this failed mixed-
representation comparison.

## Claim boundary

- Candidate execution and candidate self precision: `PASS` in this case/window.
- COMSOL reference operational self-convergence: `PASS` in this case/window.
- Locked native-FE versus canonical-P1 trajectory comparison: `FAIL` on 6/6 gates.
- Locked cross-representation case/window accuracy: `NOT_SUPPORTED`.
- Same-field full-physics solver agreement: `NOT_TESTED`.
- Boundary accuracy: `NOT_TESTED_PRE_EVENT_WINDOW`.
- Physical applicability: `NOT_TESTED` / `NOT_CERTIFIED`.
- Universal COMSOL-equivalent accuracy: `NOT_CLAIMED`.
- COMSOL as golden truth: not claimed.

P19 timing records are included only as non-gating provenance for the local
certificate and candidate hot-path work. They do not alter any scientific
decision above. Exact hashes and compact numerical details are in
[`comparison_manifest.json`](comparison_manifest.json); individual gate rows
are in [`gates.csv`](gates.csv). Large trajectories are not copied here.
