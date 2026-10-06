# Case-A size and ion-drag common-P1 companion

## Result

`PASS` for all three pre-event cells:

- 10 nm, relative-flow screened collection/orbital ion drag
- 30 nm, relative-flow screened collection/orbital ion drag
- 100 nm, electric-field-directed image/orbital ion drag

Each side used 287 particles, 46 output times from 0 to 450 us, and fixed RK4
steps of 0.625, 0.3125, and 0.15625 us. Candidate and COMSOL consumed the same
17-field/22-component exact-connectivity P1 primitive field and the same
size-specific release table. All nine histories contain 13,202 active rows and
no event or failure.

The fixed gates are inherited without tuning from
`tools/vv/comsol/cases/m3c1_caseA_100nm_common_p1_v1.json` (SHA-256
`db85d2187edd2b8807717a9f7272b6e198b3dc1a1a080df46e67980458fc763d`).
Both solvers pass the minimum 0.75 self-convergence order and fine-pair
relative-L2 gates. Every cell also passes all nine cross-solver position,
velocity, and charge RMS/maximum/relative-L2 gates.

## Applicability correction

The first 10/30 nm candidate run used `maximum_speed_ratio: 0.1` for both
effective-gas Epstein drag and effective-gas Waldmann thermophoresis. This is a
declared model envelope, not a time-step or stability limit. The normalized
COMSOL fine histories reach saved-point ratios of 0.393342 (10 nm), 0.132057
(30 nm), and 0.073586 (100 nm). The external campaign now sets the documented
subthermal sensitivity ceiling to 1.0 for both models. No solver-core gate was
removed or weakened.

COMSOL uses the same linear Epstein coefficient, so changing this comparison
to the finite-speed Epstein revision would change the physical model. That is
a separate sensitivity study, not a correction to this same-form comparison.

## Scope

This evidence supports time-integration agreement only for the locked
common-P1, Brownian-off, event-free 0--450 us slice. It does not certify native
COMSOL FE interpolation, Brownian motion, material-boundary events, universal
COMSOL equivalence, or the physical validity of the effective-gas and
ion-drag approximations.

