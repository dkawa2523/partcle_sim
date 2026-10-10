# Particle trajectory platform clean-room workspace

This workspace has been reorganized for a zero-based implementation of a
semiconductor-chamber particle trajectory solver. The former implementation is
not the foundation of the new product.

## Start here

1. Read [`particle_platform_redesign/README.md`](particle_platform_redesign/README.md).
2. Use [`particle_platform_redesign/solver/docs/support_and_errors.md`](particle_platform_redesign/solver/docs/support_and_errors.md)
   for the 0.2.0 support matrix, qualified uses, limitations, and error guidance.
3. Read [`particle_platform_redesign/AGENTS.md`](particle_platform_redesign/AGENTS.md)
   before editing the design or future solver.
4. Use [`particle_platform_redesign/implementation_plan.md`](particle_platform_redesign/implementation_plan.md)
   for implementation order and exit criteria.
5. Use [`particle_platform_redesign/quality_tooling_plan.md`](particle_platform_redesign/quality_tooling_plan.md)
   for uv, Ruff, import-linter, Pyrefly, and Radon policy.

## Directory roles

- `particle_platform_redesign/`: new specifications, reviewed architecture,
  implementation plan, the independent solver project, audit evidence, and
  external V&V design.
- `model_dataset/`: preserved reference models and exports used only as input
  evidence and external validation material.
- `old_code/`: read-only archive of the previous codebase, tests, tools,
  configuration, environments, and generated caches.
- `.git/`: retained at the repository root so history and recovery remain
  available.

The ACL-protected legacy scratch directories have now been moved into
`old_code/`; no legacy implementation directory remains active at the root.

The published `0.1.0` release remains the static 2-D baseline with canonical
HDF5 data schema 1. This checkout targets `0.2.0`: canonical
case/data/result schema 3, checkpoint schema 2, fixed-topology linear time fields,
finite-radius material contact, static Cartesian-XY translation-periodic topology,
Maxwell thermal walls, realized internal/surface schedules, and Brownian
continuation after active-wall events. Older data is not silently accepted or
migrated by the schema-3 reader.

The 0.2.0 qualification is limited to registered engineering uses. See the
[use-case acceptance](particle_platform_redesign/reviews/v0_2_release_use_case_acceptance_2026-10-10.json)
and [qualification record](particle_platform_redesign/reviews/v0_2_release_qualification_2026-10-10.json).
Release wheels and checksums are provided through the
[`v0.2.0` release](https://github.com/dkawa2523/partcle_sim/releases/tag/v0.2.0).
Experimental manufacturing prediction and user-specific runtime/RSS SLAs require
their own inputs and acceptance conditions.

The independent uv project at `particle_platform_redesign/solver/` now includes
the P00--P16 foundation, B01--B05 Brownian slices, accepted optional-physics
revisions, and the fixed-topology time-field slice. The current engine includes
producer-neutral YAML/HDF5 input, static or linearly time-interpolated XY/RZ
regular/P1/Q1 fields, continuous charge and deterministic force coupling,
RK4/exponential-midpoint motion, exact and curved material events, deterministic
wall RNG, Maxwell thermal wall handling, realized internal/surface schedules,
Brownian active-wall continuation,
durable segmented output, checkpoint/resume, and the P14 synthetic performance
baseline.

P14-P has closed the ineffective multithreaded runtime and converged production
execution on one deterministic compiled serial engine. See
[`particle_platform_redesign/solver/docs/parallel_execution_plan.md`](particle_platform_redesign/solver/docs/parallel_execution_plan.md).
The current semantics use engine v46, compiled tile v21, proposal v10, event v22,
runtime layout v6, memory plan v16, geometry v7, physics catalog
`inertial_langevin_2d_catalog_v23`, physics runtime v22, source schedule
`realized_internal_surface_contact_schedule_v5`, and wall laws
`contact_wall_laws_v7`, with topology `translation_periodic_xy_v1` and result
algorithm `durable_segmented_result_v6`. Perfect specular reflection is parameterless, non-unit
restitution is a separate law, and standard RZ gravity cannot have a radial
component. Curved and exact paths retain every facet simultaneous at the first
localized hit and apply one shared priority/combined-normal rule. Brownian
`interval_tree_depth` remains the uniform numerical-path depth; optional
`adaptive_max_depth` only conditionally refines boundary/axis candidates and
does not claim exact continuous-OU first passage. The historical P14-U formal
10k/100k/1M release gate retained the single compiled serial production engine.
Its machine-local v0.1 report is preserved at
[`particle_platform_redesign/solver/evidence/v0.1/p14u_release_v1.json`](particle_platform_redesign/solver/evidence/v0.1/p14u_release_v1.json).
T03 analysis/visualization is complete. P14-R and the `0.1.0.dev0` development
baseline's distribution-readiness closure are complete: the receipt-fixed release
workflow passed on remote Windows and Linux, including 620 tests, a seven-row
performance smoke (one cold and six warm), wheel build, runtime-only clean install,
and the three-public-API smoke. This CI result is not COMSOL V&V or portable
performance evidence. The external M3-C0b v6 run sequentially confirmed
only the Case-A 100 nm pre-event operational step selection; it is not solver
agreement or a universal accuracy claim. B03 is complete for the explicitly projected two-degree-of-freedom RZ
closure: fixed/continuous charge, native/effective-gas linear Epstein drag,
additive forces, frozen-midpoint macro roots, exact conditional OU trees, linear
dense charge, and fresh-root continuation after an axis fold. It is not an
isotropic 3-D Brownian model or a general strong/weak-order-two result. No COMSOL
study was rerun for this core closeout. Charge-stable coupling and the work-scaled
durable cadence are complete. The later candidate-v3 Case A/P runs pass their own
`h,h/2,h/4` self-convergence, and the meaning-matched common-P1 100 nm independent-
seed comparison is closed as `CLOSED_ACCEPTED_WITH_LIMITATIONS`. The saved native-
field COMSOL runs remain descriptive characterization rather than a core gate.
Additional packages and product-scale performance are separate work packages;
the preserved M3-C1 event-v14 artifacts remain historical evidence. The accepted
2-D conclusion includes the critical-boundary microcase, the Case-A common-P1
10/30 nm relative-flow and 100 nm image-ion-drag companions, and the Case-P
common-P1 100 nm aggregate-three-current trajectory (`2D_CRITICAL_VV_COMPLETE`).
Native COMSOL FE fields, 3-D, arbitrary geometry, and general COMSOL equivalence
are not certified.
