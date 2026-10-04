# M3-C1 external V&V checkpoint

M3-C1 separates saved-state formula closure from integrated trajectory
agreement. COMSOL remains an external comparison producer; none of these
artifacts defines the solver's physics or numerical truth.

| evidence | decision | scope |
|---|---|---|
| [Frozen RHS v1](pre_event_frozen_rhs_v1/README.md) | 7/8 producer forms replayed; one PPR primitive unresolved | 13,202 saved active states only |
| [Thermophoresis PPR v1](thermophoresis_ppr_v1/README.md) | `PASS` for the eighth saved-row producer form | recovered PPR heat flux and Waldmann replay only |
| [Exported-P1 candidate blocker v1](exported_p1_candidate_blocker_v1/README.md) | `BLOCKED` before the first accepted trajectory piece | canonical exact-connectivity P1 projection, not the native COMSOL interpolant |
| [Case-A 100 nm pre-event v6](case_a_100nm_pre_event_v6/README.md) | candidate execution `COMPLETE`; locked cross-representation comparison `FAIL` | three deterministic step sizes, 287 particles, 46 frames |
| [Case-A 100 nm common-P1 v1](case_a_100nm_common_p1_v1/README.md) | locked same-field comparison `PASS` on 9/9 gates | both solvers consume the common canonical exact-connectivity P1 field; Brownian off; no boundary event |
| [Case-A 100 nm material-event v1](case_a_100nm_material_event_v1/README.md) | first-material-event `PASS` on 20/20 gates; current v14 prefix 9/9; v14 temporal self-check 6/6 | locked common-P1, Brownian-off case through the first wafer stick only |
| [100 nm, 30 ms saved-reference characterization v1](existing_30ms_reference_v1/comparison.json) | candidate-v3 self-convergence `PASS`; saved COMSOL comparison `CHARACTERIZED`, not a core gate | Case A/P candidate is Brownian off; saved COMSOL is native-field, Brownian on, and a single fixed-RK4 10 us run |

Taken together, the first two records close all eight saved-row producer
forms: dynamic charge plus seven deterministic force contributions. That is
not an integrated solver result. Production pure-model comparison remains
partial, stage and continuous-path wiring remain untested, and no trajectory
agreement is claimed.

The first exported-P1 candidate is retained as historical pre-P19 blocker
evidence. All 287 initial point samples were supported and locally applicable,
but the former global continuous-path enclosure rejected every particle at
time zero. P19's bounded local certificate subsequently allowed the v6
candidate to complete without events or failures; this does not erase the
blocker record.

The completed v6 candidate is roundoff-limited across its three internal step
sizes. The locked COMSOL reference also passes its own operational
self-convergence gates. Nevertheless, all six preregistered RMS/maximum gates
fail when the canonical exported-nodal P1 candidate is compared with COMSOL's
native finite-element representation. That cross-representation `FAIL` remains
authoritative for its own comparison and is not replaced by a later result.

The triggered common-field diagnostic is now complete. With both solvers
consuming the same canonical exact-connectivity P1 field, all nine fixed
absolute and relative trajectory gates pass for Case A, 100 nm, Brownian off,
and `0--450 us`. This establishes only the narrow same-field agreement. It does
not make the prior cross-representation result pass, exercise a boundary
event, establish physical validity, or claim universal COMSOL-equivalent
accuracy.

The material-event checkpoint extends that same locked common-P1 case only
through its first wafer stick. Its 20/20 material-event gates pass, including
event-time, hit-position, and terminal-charge differences of
`2.157542807607049e-13 s`, `2.683964162031316e-14 m`, and
`4.7283812421028415e-09 e`. The current v14 pre-event prefix independently
retains 9/9 same-field trajectory gates and adds a three-step temporal
self-check with 6/6 gates. The observed orders near two are empirical for this
piecewise-P1, mesh-crossing case; they do not certify formal fourth-order RK4
convergence.

The older roundoff-limited self-check remains valid historical evidence of
precision stability. Its artificial event subdivision equalized accepted
piece counts, so it could not independently evaluate temporal order; the v14
three-step run supplies that separate check. The recorded shell wall-time
change and work-counter reductions are performance observations, not accuracy
gates. This checkpoint does not certify the native COMSOL field, physical
validity, Brownian operation, any behavior after the first stick, or the
broader 30 ms cases.

The later 100 nm, 30 ms candidate-v3 extension uses independently selected
`h,h/2,h/4` series and passes candidate self-convergence for both Case A and
Case P; Case A also has matching event identity and converged event quantities,
while Case P has no events in the three candidate runs. This candidate result
is the primary numerical decision. The saved Case A/P COMSOL histories use
manual explicit fixed RK4 at 10 us, COMSOL native finite-element fields, and an
enabled Brownian feature, whereas candidate v3 is Brownian off. Their aligned
time histories are therefore external cross-representation characterization,
not deterministic pathwise parity or a core acceptance gate. The candidate is
not required to mirror the COMSOL step. A future stochastic comparison must use
preregistered independent-seed ensembles unless model and RNG identity are
proved.
