# M3-V result

M3-V completed two distinct external decisions. The original 12-package
assessment remains an applicability/relevance result and does not certify its
unmatched saved trajectories. A later deterministic matched slice established
time-integration parity for one hash-fixed Case-A 100 nm pre-event problem.
`NOT_APPLICABLE` and `NOT_TESTED` remain explicit scope results, not hidden
passes.

## Deterministic matched trajectory certification

The detailed assessment is
[matched_caseA_100nm_deterministic_v1.md](matched_caseA_100nm_deterministic_v1.md).
It uses 287 particles, 41 output times over 0..0.4 ms, fixed charge -1, and the
common electric, linear-Epstein, and gravity/buoyancy force set. Brownian,
dynamic charge, ion drag, thermophoresis, lift, and DEP are disabled.

Both solvers consume the same canonical P1 node values and exact triangle
connectivity. The comparison budget was registered from independent 10/5/2.5
us self-convergence reports before the fine histories were compared. The
locked comparison passed with position RMS `8.643e-16 m` and velocity RMS
`1.999e-14 m/s`. Field and force parity are at float64 roundoff scale.

This supports equal time-discretization accuracy only for that matched,
hash-fixed pre-event window with zero observed material events. It is not
universal COMSOL equivalence.
Boundary accuracy, native finite-element field extraction/remeshing, dynamic
charge, additional forces, and stochastic Brownian validation remain separate
external gates. The production solver has no COMSOL dependency or comparison
mode.

## Gates

| Gate | Status | Reason |
|---|---|---|
| M3V-01-direct-model-inventory | PASS | two MPH files loaded read-only; no study run and no save |
| M3V-02-reference-package-structure | PASS | 12 reference-only packages have the declared rectangular history and provenance |
| M3V-03-p15-charge-trajectory-applicability | NOT_APPLICABLE | sampled P15 applicability range=0..0; scalar-ion-mass mismatch cases=6/12; species mismatch cases=6/12; continuous-path certification is also required |
| M3V-03b-reference-charge-rate-formula-parity | PASS | frozen-state relative-drift regularized two-current law reconstructed from saved primitives; parity is provenance evidence, not physical acceptance |
| M3V-03c-reference-charge-local-stiffness-characterization | PASS | analytic frozen-state dR/dZ sampled at saved rows; this is not a continuous-path or integrator stability certificate |
| M3V-04-p15-epstein-trajectory-applicability | NOT_APPLICABLE | sampled linear-Epstein coverage range=0.542834..1; continuous-path certification is also required |
| M3V-04b-reference-epstein-formula-parity | PASS | COMSOL Epstein force reconstructed with delta=1+sigma_R*pi/8; formula parity does not extend the linear model applicability domain |
| M3V-05-force-and-charge-relevance | PASS | all derived values are finite and independently replayable formulas match; saved-grid force quadrature remains relevance-only |
| M3V-06-reference-package-variant-sensitivity | PASS | full time histories compared; Case A has an ion-drag-only configuration diff but only one stochastic realization; Case P also has a lift expression diff |
| M3V-07-full-production-trajectory-comparison | NOT_APPLICABLE | reference trajectories contain unmatched charge and force closures |
| M3V-08-boundary-validation | NOT_TESTED | reference releases are internal and contain no enabled reflection case |
| M3V-09-stochastic-validation | NOT_TESTED | one Brownian seed cannot validate a stochastic distribution |
| M3V-10-reduced-electrostatic-builder-parity | NOT_TESTED | MPH closure inventoried; independent canonical-field builder is a later slice |

## Historical readiness record

`matched_case_readiness.json` is the immutable pre-run readiness decision for
the original saved Case-A 100 nm history. Its
`BLOCKED_DO_NOT_RUN_COMSOL` status remains correct for that unmatched history:
it has Brownian enabled, unmatched deterministic forces and charge evolution,
no RK-stage export, and no positive Freeze event. It must not be relabelled as
a matched trajectory reference.

The later deterministic certification did not unblock or reuse that history.
It created a separate COMSOL companion with Brownian and unmatched physics
disabled, exact canonical P1 connectivity, fixed charge, and a pre-event time
window. Its accepted scope and hashes are recorded in
`matched_caseA_100nm_deterministic_v1.md`. The original applicability gate rows
above and the historical readiness JSON therefore remain unchanged; neither
extends the matched PASS to native fields, boundary events, or stochastic
physics.

## Historical implementation priority by independent workstream

This table records the priority decision made by the original 12-package
applicability closeout; it is not the current backlog. The corresponding
production slices through P15-D/E/F, P16, F01/F02, and B02 were completed
later. The Brownian multi-seed external campaign and P17 remain separate
future gates.

| Workstream | Priority | Candidate | Decision |
|---|---:|---|---|
| canonical_field_production | 1 | independent_reduced_electrostatic_builder | mandatory_first_party_preprocessor_with_canonical_output |
| trajectory_physics | 1 | relative_drift_regularized_two_current_charge_v1 | required_before_reference_trajectory_replay |
| trajectory_physics | 2 | finite_speed_epstein_drag | required_for_uncovered_reference_states |
| trajectory_physics | 3 | versioned_relative_flow_ion_drag | implement_as_explicit_model_not_case_profile |
| trajectory_physics | 4 | waldmann_thermophoresis_p16 | retain_as_separate_deterministic_model |
| trajectory_physics | 5 | brownian_multi_seed_campaign | validate_distribution_before_production_acceptance |
| state_dimension | 1 | cartesian_3d_p17 | retain_independent_product_roadmap_item |
