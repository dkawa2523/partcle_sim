# M3-C1 pre-event frozen-RHS comparison v1

This external V&V record evaluates the production pure-physics functions on
the 13,202 saved active local states from the accepted M3-C0b v6 finest run
(287 particles, 46 output times, 0--450 microseconds). It does not rerun
COMSOL, integrate either trajectory, or change the production solver.

The three decisions are intentionally independent:

| contribution | saved producer formula | production model | saved-row applicability |
|---|---|---|---|
| dynamic charge | `PASS` | documented constant convention difference | `PASS` |
| electric | `PASS` | `PASS` | no additional local gate |
| relative-flow ion drag | `PASS` | documented constant convention difference | `PASS` |
| linear Epstein drag | `PASS` | `PASS` | `PASS` |
| Waldmann thermophoresis | `NOT_TESTED_PRIMITIVE_AUTHORITY` | `NOT_TESTED_PRIMITIVE_AUTHORITY` | `PASS` |
| free-molecular lift sensitivity | `PASS` | `PASS` | `PASS` |
| dielectrophoresis | `PASS` | documented constant convention difference | point-dipole certificate not tested |
| gravity/buoyancy | `PASS` | `PASS` | no additional local gate |

All seven testable saved producer formulas replay at a maximum normalized
residual of `1.2659055420084512e-15` or less. The production comparisons that
use the same constants replay at `9.672453607659756e-16` or less. Dynamic
charge, ion drag, and DEP differ by at most `3.929288147943532e-9` under the
already documented COMSOL-versus-production physical-constant convention.
The COMSOL value inferred from the saved electric force is
`8.854187818826714e-12 F/m`; its relative difference from the production
constant is `6.806624921756877e-10`.

Thermophoresis is not marked failed. The COMSOL feature has `UsePPR=1`, while
M3-C0b v6 saved the unrecovered `-k*d(T,r/z)` primitive. Replaying that
non-authoritative primitive gives a diagnostic maximum normalized residual of
`0.1518848029947857`. A new finest-step-only no-clobber export of the recovered
gradient or heat flux is required to close formula parity.

For the saved states, the largest relative ion speed is
`17992.100259448238 m/s`, the largest effective ion speed is
`17995.554013580226 m/s`, the largest gas relative-speed ratio is
`0.06455651014478188`, and the minimum gas mean-free-path/radius ratio is
`39553.74651385394`. These are saved-frame results only; continuous accepted
paths and integrator stages remain `NOT_TESTED`.

The non-gating total-force diagnostic has relative L2 residual
`0.0011919715089522942`, dominated by the unresolved PPR thermophoretic
primitive. It must not be used as a trajectory-accuracy gate.

`report.json` fixes all five raw table hashes, the configuration hash, source
model hash and load/no-save provenance, and the evaluator hash
`7c7b0c7f8f697a8e5bfc62c8e62fdd6811341d9da47680021c711882ab812f46`.
The exact closure rerun and native-mesh export plan is documented in
`tools/vv/comsol/M3C1_THERMOPHORESIS_EXPORT_PLAN.md`.
